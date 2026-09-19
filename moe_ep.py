"""moe.py with vanilla expert parallelism (no MoonEP): experts are sharded
across GPUs (epn = E/R per rank), tokens travel with plain NCCL all_to_all —
sort the S*K routed pairs by expert (which also groups them by owner rank),
exchange counts, exchange rows, grouped GEMM over the local shard, send the
outputs back, weighted scatter into token order.

Run:
    torchrun --nproc_per_node=8 moe_ep.py
"""
import os
import time
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
# autograd-aware all_to_all: plain dist.all_to_all_single severs the graph, so
# the expert weights would never receive a gradient. The differentiable
# version's backward is the same collective with the splits swapped.
import torch.distributed.nn.functional as dist_nn


class EPMoE(nn.Module):
    """Top-K MoE with grouped-GEMM experts, sharded via all_to_all EP."""

    def __init__(self, E, K, H, Hi, group=None):
        super().__init__()
        self.E, self.K, self.H, self.Hi = E, K, H, Hi
        self.group = group
        self.rank = dist.get_rank(group)
        self.R = dist.get_world_size(group)
        assert E % self.R == 0, "E must be divisible by the EP group size"
        self.epn = E // self.R

        # router is replicated: construct under a fixed seed on every rank
        torch.manual_seed(0)
        self.router = nn.Linear(H, E, bias=False)

        # this rank's expert shard: experts [rank*epn, (rank+1)*epn)
        g = torch.Generator().manual_seed(100 + self.rank)
        self.w_gate = nn.Parameter(torch.empty(self.epn, H, Hi).normal_(0.0, 0.02, generator=g))
        self.w_up = nn.Parameter(torch.empty(self.epn, H, Hi).normal_(0.0, 0.02, generator=g))
        self.w_down = nn.Parameter(torch.empty(self.epn, Hi, H).normal_(0.0, 0.02, generator=g))

    def forward(self, x):
        # --- route ---
        # fp32 logits/softmax, matching moe_moon_ep.route: bf16 logits produce
        # ties that make topk pick different experts, so the two EP paths would
        # not be comparable
        logits = self.router(x).float()
        weights, idx = torch.topk(logits, k=self.K, dim=-1)
        weights = F.softmax(weights, dim=-1)

        # sort the S*K (token, expert) pairs by expert; since rank r owns the
        # contiguous expert range [r*epn, (r+1)*epn), this also groups the
        # rows by destination rank
        flat_e = idx.flatten()
        order = torch.argsort(flat_e, stable=True)
        toks = order // self.K
        w_sorted = weights.flatten()[order]
        e_sorted = flat_e[order]

        # --- exchange counts, then rows (dispatch) ---
        counts = torch.bincount(flat_e, minlength=self.E)
        send_counts = counts.view(self.R, self.epn).sum(1)
        recv_counts = torch.empty_like(send_counts)
        dist.all_to_all_single(recv_counts, send_counts, group=self.group)
        in_splits = send_counts.tolist()    # host sync: NCCL needs CPU splits
        out_splits = recv_counts.tolist()
        n_recv = sum(out_splits)

        x_send = x[toks].contiguous()
        # differentiable: backward is the reverse all_to_all (splits swapped)
        x_recv = dist_nn.all_to_all_single(
            x.new_empty(n_recv, x.size(1)), x_send, out_splits, in_splits,
            group=self.group,
        )
        e_recv = e_sorted.new_empty(n_recv)
        dist.all_to_all_single(e_recv, e_sorted.contiguous(), out_splits, in_splits,
                               group=self.group)   # int ids: no grad needed

        # --- local expert compute (grouped GEMM over the shard) ---
        # each source's chunk arrives expert-sorted, but the concatenation of
        # R chunks is not: re-sort by local expert id
        local_e = e_recv - self.rank * self.epn
        rorder = torch.argsort(local_e, stable=True)
        xr = x_recv[rorder]
        offs = torch.bincount(local_e, minlength=self.epn).cumsum(0).to(torch.int32)
        gate = torch._grouped_mm(xr, self.w_gate, offs=offs)
        up = torch._grouped_mm(xr, self.w_up, offs=offs)
        yr = torch._grouped_mm(F.silu(gate) * up, self.w_down, offs=offs)
        y_recv = torch.empty_like(yr)
        y_recv[rorder] = yr                 # undo the local sort

        # --- send outputs back (combine) and weighted-scatter to tokens ---
        y_send = dist_nn.all_to_all_single(
            torch.empty_like(x_send), y_recv, in_splits, out_splits, group=self.group,
        )

        # scale slot -> bf16, K-sum in fp32 (same rounding as the MoonEP path)
        z = (y_send.float() * w_sorted[:, None]).to(x.dtype)
        out = torch.zeros(x.size(0), self.H, dtype=torch.float32, device=x.device)
        out = out.index_add(0, toks, z.float())
        return out.to(x.dtype)

    def full_weights(self):
        """all_gather the expert shards into full [E, ...] tensors."""
        full = []
        for w in (self.w_gate, self.w_up, self.w_down):
            parts = [torch.empty_like(w) for _ in range(self.R)]
            dist.all_gather(parts, w.contiguous(), group=self.group)
            full.append(torch.cat(parts))
        return full

    def forward_reference(self, x):
        """No-communication reference: per-expert loop over gathered weights."""
        wg, wu, wd = self.full_weights()
        logits = self.router(x).float()
        weights, idx = torch.topk(logits, k=self.K, dim=-1)
        weights = F.softmax(weights, dim=-1)
        flat_w = weights.flatten()
        flat_e = idx.flatten()
        flat_t = torch.arange(x.size(0), device=x.device).repeat_interleave(self.K)

        out = torch.zeros(x.size(0), self.H, dtype=torch.float32, device=x.device)
        for e in range(self.E):
            sel = flat_e == e
            if not sel.any():
                continue
            t = flat_t[sel]
            y = (F.silu(x[t] @ wg[e]) * (x[t] @ wu[e])) @ wd[e]
            z = (y.float() * flat_w[sel, None]).to(x.dtype)
            out = out.index_add(0, t, z.float())
        return out.to(x.dtype)


def main():
    rank = int(os.getenv('LOCAL_RANK'))
    dist.init_process_group(backend='nccl', device_id=rank)
    torch.cuda.set_device(rank)
    dev = f'cuda:{rank}'

    # S halved vs the balanced-only config: the imbalanced run concentrates
    # all R*S*K rows on rank 0, whose FFN intermediates would OOM at S=8192
    S, K, E, H, Hi = 1024*4, 16, 128, 4096, 4096*2
    moe = EPMoE(E, K, H, Hi).to(dev, torch.bfloat16)

    # IMBALANCED=1: zero the router weight => every logit ties => topk breaks
    # ties by index and picks experts 0..K-1 for every token. K == epn here,
    # so that is exactly rank 0's shard: rank 0 receives all R*S*K rows,
    # ranks 1..7 receive none. Forward pass untouched — init only.
    with torch.no_grad():
        moe.router.weight[:64].zero_()

    g = torch.Generator(device=dev).manual_seed(42 + rank)
    x = torch.randn(S, H, dtype=torch.bfloat16, device=dev, generator=g).requires_grad_(True)
    gout = torch.randn(S, H, dtype=torch.float32, device=dev, generator=g)

    n_iters = 100
    for _ in range(10):
        moe.zero_grad(set_to_none=True)
        x.grad = None
        y = moe(x)
        (y.float() * gout).sum().backward()

    torch.cuda.synchronize()        # drain warmup before reading the clock
    dist.barrier()                  # align all ranks at the start line
    time_start = time.perf_counter()
    for _ in range(n_iters):
        moe.zero_grad(set_to_none=True)
        x.grad = None
        y = moe(x)
        (y.float() * gout).sum().backward()
        
    torch.cuda.synchronize()        # wait for the queued tail of iter 1000
    time_end = time.perf_counter()

    y_ref = moe.forward_reference(x)

    ms = (time_end - time_start) / n_iters * 1e3
    t = torch.tensor([ms], device=dev)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)

    diff = (y.float() - y_ref.float()).abs()
    print(f"[rank {rank}] {ms:.3f} ms/iter (slowest rank: {t.item():.3f} ms)  "
          f"max abs diff {diff.max().item():.3e}")

    dist.barrier()
    dist.destroy_process_group()

# S, K, E, H, Hi = 1024*8, 16, 128, 4096, 4096*2
# total disbalance
# 219.971 ms/iter
# torchrun --nproc_per_node=8 moe_ep.py
if __name__ == '__main__':
    main()
