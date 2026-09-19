"""Expert-parallel top-K MoE over NVLink GPUs using MoonEP dispatch/combine.

Mirrors moe.py's grouped-GEMM expert compute, but the experts are sharded
across the EP group: rank r owns experts [r*epn, (r+1)*epn). MoonEP moves the
tokens (dispatch/combine over NVLink) and prefetches the weights of the
load-balancing "copied" experts into the B local prefetch slots.

Weight storage follows bench_vs_deepep's composite [E+B] layout: every rank
allocates its own [epn, H, *] expert chunk plus a [B, H, *] prefetch chunk,
exchanges VMM fds, and maps all R expert chunks + its local prefetch chunk
into one contiguous VA. Rows [0, E) are the global expert pool (remote rows
walk NVLink), rows [E, E+B) are local prefetch slots — exactly the tensor
Buffer.prefetch_weight expects. Requires B == epn and the chunk byte size to
be a multiple of the VMM granularity.

Autograd: dispatch and combine are transposes of each other, so each gets a
custom autograd.Function whose backward is the other call with the saved plan
(api.py's documented recipe). The FFN between them is ordinary autograd over
the [E+B] weight Parameters. The prefetch rows [E, E+B) accumulate real
gradients for *copied* (remote-owned) experts; reduce_expert_grads() ships
them home over NVLink with the grad_reduce kernel and adds them into the
owners' rows.

Run:
    torchrun --nproc_per_node=8 moe_moon_ep.py
"""
import os
import time

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from moonep import Buffer
from moonep._C import nvl_dist_alloc, nvl_dist_map, nvl_release_mem_handle, get_vmm_granularity
from moonep.buffer import _exchange_ipc_fds, create_nvl_dist_tensor
from moonep.inter_rank_sync import launch_inter_rank_sync


class _MoonEPDispatch(torch.autograd.Function):
    """dispatch fwd / combine bwd.

    Forward scatters [S, H] tokens (and their [S, K] routing weights) into
    the expert-grouped [NvS, H] layout across ranks. Backward combines the
    slot gradients back: each token's K slot grads are summed into grad_x,
    and each slot's weight grad is gathered back to its (s, k) position.
    """

    @staticmethod
    def forward(ctx, x, weights, topk, tpe, buffer, holder):
        h_nvs, w_nvs, cu_seqlens, plan = buffer.dispatch(
            x.contiguous(), weights.contiguous(), topk, tpe, zero_copy=False,
        )
        holder['plan'] = plan
        ctx.buffer, ctx.plan = buffer, plan
        ctx.mark_non_differentiable(cu_seqlens)
        return h_nvs, w_nvs, cu_seqlens

    @staticmethod
    def backward(ctx, grad_h, grad_w, _grad_cu):
        grad_x, grad_w_sk, _ = ctx.buffer.combine(
            plan=ctx.plan,
            hidden_nvsh=grad_h.contiguous(),
            route_weights_nvs=grad_w.contiguous() if grad_w is not None else None,
        )
        return grad_x, grad_w_sk, None, None, None, None


class _MoonEPCombine(torch.autograd.Function):
    """combine fwd / dispatch bwd.

    Forward gathers every token's K expert outputs from the [NvS, H] shard
    back to its source rank and sums -> [S, H]. Backward re-dispatches the
    output grad with the saved plan: grad_out[s] is scattered to each of
    token s's slots (duplicate slots included, via the epilogue expansion).
    """

    @staticmethod
    def forward(ctx, z_nvs, buffer, plan):
        out, _, _ = buffer.combine(plan=plan, hidden_nvsh=z_nvs.contiguous())
        ctx.buffer, ctx.plan = buffer, plan
        return out

    @staticmethod
    def backward(ctx, grad_out):
        grad_z, _, _, _ = ctx.buffer.dispatch(
            grad_out.contiguous(), plan=ctx.plan,
        )
        return grad_z, None, None


class MoonEPMoE(nn.Module):
    """Top-K MoE with grouped-GEMM experts, expert-parallel via MoonEP."""

    def __init__(self, E, K, H, Hi, S, group=None, num_sms=32):
        super().__init__()
        self.E, self.K, self.H, self.Hi = E, K, H, Hi
        self.group = group
        self.rank = dist.get_rank(group)
        self.R = dist.get_world_size(group)
        self.epn = E // self.R

        # router is replicated: construct under a fixed seed on every rank
        torch.manual_seed(0)
        self.router = nn.Linear(H, E, bias=False, dtype=torch.bfloat16)

        self.buffer = Buffer(S, H, K, E, self.R, num_sms=num_sms, group=group)
        self.B = int(self.buffer._require_ctx()['B'])
        assert self.B == self.epn, "composite [E+B] weight layout expects B == epn"

        # composite [E+B, ...] weight pools as Parameters; this rank physically
        # owns rows [rank*epn, (rank+1)*epn) and the prefetch rows [E, E+B)
        self._keepalives = []
        self.w_gate = nn.Parameter(self._composite_weight((self.epn, H, Hi), group))
        self.w_up = nn.Parameter(self._composite_weight((self.epn, H, Hi), group))
        self.w_down = nn.Parameter(self._composite_weight((self.epn, Hi, H), group))

        # fp32 reduce buffers for the copied experts' weight grads: each rank
        # owns its [B, ...] chunk, all R chunks mapped as one [R, B, ...] view
        self.rb_gate = create_nvl_dist_tensor(
            [self.B, H, Hi], torch.float32, self.rank, self.R, group=group,
        ).view(self.R, self.B, H, Hi)
        self.rb_up = create_nvl_dist_tensor(
            [self.B, H, Hi], torch.float32, self.rank, self.R, group=group,
        ).view(self.R, self.B, H, Hi)
        self.rb_down = create_nvl_dist_tensor(
            [self.B, Hi, H], torch.float32, self.rank, self.R, group=group,
        ).view(self.R, self.B, Hi, H)

        # init only the locally-owned expert rows; zero own reduce chunks
        with torch.no_grad():
            g = torch.Generator(device=self.w_gate.device).manual_seed(100 + self.rank)
            lo, hi = self.rank * self.epn, (self.rank + 1) * self.epn
            for w in (self.w_gate, self.w_up, self.w_down):
                w[lo:hi].normal_(0.0, 0.02, generator=g)
            for rb in (self.rb_gate, self.rb_up, self.rb_down):
                rb[self.rank].zero_()
        self._last_plan = None
        torch.cuda.synchronize()
        dist.barrier(group=group)

    def _composite_weight(self, chunk_shape, group):
        chunk_bytes = chunk_shape[0] * chunk_shape[1] * chunk_shape[2] * 2
        gran = get_vmm_granularity()
        assert chunk_bytes % gran == 0, \
            f"expert chunk {chunk_bytes} B must be a multiple of VMM granularity {gran}"
        ka_w, w_fd, w_owned = nvl_dist_alloc(shape=list(chunk_shape), dtype=torch.bfloat16)
        ka_b, b_fd, b_owned = nvl_dist_alloc(shape=list(chunk_shape), dtype=torch.bfloat16)
        for ka, owned in ((ka_w, w_owned), (ka_b, b_owned)):
            self._keepalives.append(ka)
            nvl_release_mem_handle(owned)
        fds = _exchange_ipc_fds(w_fd, list(range(self.R)), self.rank, self.R, group)
        os.close(w_fd)
        all_fds = [fds[r] for r in range(self.R)] + [b_fd]
        try:
            full = nvl_dist_map(
                chunk_shape=list(chunk_shape), dtype=torch.bfloat16,
                fds=all_fds, local_rank=self.rank, world_size=self.R + 1,
            )
        finally:
            for r in range(self.R):
                os.close(fds[r])
            os.close(b_fd)
        return full  # [E + B, chunk_shape[1], chunk_shape[2]]

    def route(self, x):
        logits = self.router(x).float()
        weights, idx = torch.topk(logits, k=self.K, dim=-1)
        weights = F.softmax(weights, dim=-1)
        topk = idx.to(torch.int32)
        tpe = torch.bincount(topk.flatten(), minlength=self.E).to(torch.int32)
        return weights.contiguous(), topk.contiguous(), tpe

    def forward(self, x):
        """[s, H] -> [s, H] for any 1 <= s <= S. S is only the Buffer's
        capacity: the comm kernels loop over s, no padding tokens are made.
        """
        weights, topk, tpe = self.route(x)

        # dispatch: scatter tokens to their experts' home/copy ranks over
        # NVLink (autograd-aware: backward is combine with the saved plan)
        holder = {}
        h_nvs, w_nvs, cu_seqlens = _MoonEPDispatch.apply(
            x, weights, topk, tpe, self.buffer, holder,
        )
        plan = holder['plan']
        self._last_plan = plan
        # pull the weights of this rank's copied experts into rows [E, E+B).
        # Raw kernel write into the Parameter storage — deliberately outside
        # autograd; it must run before the grouped GEMMs of this step and not
        # between a forward and its backward.
        self.buffer.prefetch_weight(
            plan=plan,
            full_gate_weight=self.w_gate.data,
            full_up_weight=self.w_up.data,
            full_down_weight=self.w_down.data,
        )

        # grouped GEMM straight over the dispatched layout: the shard holds
        # E+B VM groups and cu_seqlens is already _grouped_mm's offs format.
        # Active groups < E are this rank's own experts (local weight rows);
        # groups >= E are the copied experts (prefetch rows). Padding rows
        # are zero-filled by dispatch, so their FFN output is exactly zero.
        # No host sync: _grouped_mm only touches rows inside the cu_seqlens
        # groups, so feed the whole [NvS, H] shard. Rows past the last group
        # end are referenced by no dst slot; they come out as exact zeros in
        # forward and their grads never reach a weight grad.
        gate = torch._grouped_mm(h_nvs, self.w_gate, offs=cu_seqlens)
        up = torch._grouped_mm(h_nvs, self.w_up, offs=cu_seqlens)
        y = torch._grouped_mm(F.silu(gate) * up, self.w_down, offs=cu_seqlens)

        # scale each slot by its routing weight. Padding and tail slots carry
        # stale weight values (never written for this step) — nan_to_num keeps
        # 0-row * garbage from minting NaNs that a GEMM backward would spread.
        w_used = torch.nan_to_num(w_nvs)
        z = (y.float() * w_used[:, None]).to(torch.bfloat16)

        # combine: gather every token's K expert outputs back to its source
        # rank and sum -> [S, H] (backward: re-dispatch with the saved plan)
        return _MoonEPCombine.apply(z, self.buffer, plan)


    def reduce_expert_grads(self):
        """dispatch bwd, weight side: ship the prefetch-slot (copied expert)
        grads to their home ranks and accumulate into the owners' rows.

        Returns fp32 [epn, ...] grads (gate, up, down) for the locally-owned
        expert rows [rank*epn, (rank+1)*epn), fully reduced. param.grad is
        updated in place: owned rows reduced, prefetch rows zeroed (consumed)."""
        assert self._last_plan is not None, "reduce_expert_grads needs a prior forward"
        E = self.E
        lo, hi = self.rank * self.epn, (self.rank + 1) * self.epn
        ctx = self.buffer._require_ctx()
        pairs = (
            (self.w_gate, self.rb_gate),
            (self.w_up, self.rb_up),
            (self.w_down, self.rb_down),
        )
        # torch-based NVLink reduce instead of launch_grad_reduce: the kernel
        # addresses rows by global expert id, so it requires contiguous fp32
        # [E, H, H'] grads (~17 GiB per projection at benchmark shapes). The
        # reduce chunks are NVL-mapped and readable from every rank, so the
        # owner can accumulate its few slots directly; fp32 staging shrinks
        # to one [epn, H, H'] slice.
        with torch.no_grad():
            for p, rb in pairs:
                assert p.grad is not None, "call backward() before reduce_expert_grads()"
                rb[self.rank].copy_(p.grad[E:])   # publish my copies' grads (-> fp32)
            # all ranks must finish publishing before anyone reads (device-side)
            launch_inter_rank_sync(ctx)

            etc = self._last_plan.experts_to_copy.cpu()   # [R, B], small
            outs = []
            for p, rb in pairs:
                g_local = p.grad[lo:hi].float()           # [epn, H, H'] fp32
                for r in range(self.R):
                    for b in range(self.B):
                        e = int(etc[r, b])
                        if lo <= e < hi:
                            g_local[e - lo] += rb[r, b]   # remote read over NVLink
                p.grad[lo:hi].copy_(g_local.to(p.grad.dtype))
                p.grad[E:].zero_()                        # consumed by the reduce
                outs.append(g_local)
            # peers must finish reading my chunk before I clear it for reuse
            launch_inter_rank_sync(ctx)
            for _, rb in pairs:
                rb[self.rank].zero_()
        return outs

    def forward_reference(self, x):
        """Single-rank reference: per-expert loop over the global weight pool
        (remote rows read over NVLink). Same math, no token movement.
        Autograd-capable when called with grad-enabled tensors."""
        weights, topk, _ = self.route(x)
        flat_w = weights.flatten()
        flat_e = topk.flatten().long()
        flat_t = torch.arange(x.size(0), device=x.device).repeat_interleave(self.K)

        out = torch.zeros(x.size(0), self.H, dtype=torch.float32, device=x.device)
        for e in range(self.E):
            sel = flat_e == e
            if not sel.any():
                continue
            toks = flat_t[sel]
            gate = x[toks] @ self.w_gate[e]
            up = x[toks] @ self.w_up[e]
            y = (F.silu(gate) * up) @ self.w_down[e]
            # match the EP path's rounding: scaled slot is bf16 before the sum
            y = (y.float() * flat_w[sel, None]).to(torch.bfloat16)
            out = out.index_add(0, toks, y.float())
        return out.to(torch.bfloat16)

    def destroy(self):
        self.buffer.destroy()


def reference_grads(moe, x, gout):
    """Dense-clone reference: same routing + FFN math on cloned weights,
    plain autograd, no communication. Returns (out, dx, drouter, dwg, dwu, dwd)."""
    E, K = moe.E, moe.K
    xr = x.detach().clone().requires_grad_()
    rw = moe.router.weight.detach().clone().requires_grad_()
    wg = moe.w_gate.detach()[:E].clone().requires_grad_()
    wu = moe.w_up.detach()[:E].clone().requires_grad_()
    wd = moe.w_down.detach()[:E].clone().requires_grad_()

    logits = (xr @ rw.T).float()
    w, idx = torch.topk(logits, k=K, dim=-1)
    w = F.softmax(w, dim=-1)
    flat_w = w.flatten()
    flat_e = idx.flatten()
    flat_t = torch.arange(x.size(0), device=x.device).repeat_interleave(K)

    out = torch.zeros(x.size(0), moe.H, dtype=torch.float32, device=x.device)
    for e in range(E):
        sel = flat_e == e
        if not sel.any():
            continue
        t = flat_t[sel]
        y = (F.silu(xr[t] @ wg[e]) * (xr[t] @ wu[e])) @ wd[e]
        z = (y.float() * flat_w[sel, None]).to(torch.bfloat16)
        out = out.index_add(0, t, z.float())
    out = out.to(torch.bfloat16)
    (out.float() * gout).sum().backward()
    return out, xr.grad, rw.grad, wg.grad, wu.grad, wd.grad


def main():
    local_rank = int(os.getenv('LOCAL_RANK'))
    dist.init_process_group(backend='nccl', device_id=local_rank)
    rank = dist.get_rank()
    torch.cuda.set_device(local_rank)
    dev = f'cuda:{local_rank}'

    S, K, E, H, Hi = 1024*4, 16, 128, 4096, 4096*2
    moe = MoonEPMoE(E, K, H, Hi, S).to(dev)
    
    with torch.no_grad():
        moe.router.weight[:64].zero_()
        
    g = torch.Generator(device=dev).manual_seed(42 + rank)
    x = torch.randn(S, H, dtype=torch.bfloat16, device=dev, generator=g).requires_grad_(True)
    gout = torch.randn(S, H, dtype=torch.float32, device=dev, generator=g)

    # ---- EP forward + backward ----
    y = moe(x)
    (y.float() * gout).sum().backward()
    g_gate, g_up, g_down = moe.reduce_expert_grads()

    # ---- dense reference on the same weights (needs ~4x the expert weight
    # bytes for clones + their grads: skip when it cannot fit) ----
    lo, hi = rank * moe.epn, (rank + 1) * moe.epn
    ref_bytes = 4 * 3 * E * H * Hi * 2
    free_bytes, _ = torch.cuda.mem_get_info()
    if ref_bytes < free_bytes * 0.8:
        y_ref, dx_ref, drouter_ref, dwg_ref, dwu_ref, dwd_ref = reference_grads(moe, x, gout)

        def report(name, mine, ref):
            mine = mine.float()
            ref = ref.float()
            diff = (mine - ref).abs().max().item()
            scale = ref.abs().max().clamp_min(1e-12).item()
            print(f"[rank {rank}] {name:12s} max abs diff {diff:.3e}  (ref max {scale:.3e})")

        report("forward", y, y_ref)
        report("dx", x.grad, dx_ref)
        report("drouter", moe.router.weight.grad, drouter_ref)

        # expert grads: the true total sums every rank's token contributions;
        # compare my owned slice of the reduced EP grads against the
        # allreduced reference
        for name, mine, ref in (
            ("dw_gate", g_gate, dwg_ref),
            ("dw_up", g_up, dwu_ref),
            ("dw_down", g_down, dwd_ref),
        ):
            total = ref.float().contiguous()
            dist.all_reduce(total)
            report(name, mine, total[lo:hi])
        del y_ref, dx_ref, drouter_ref, dwg_ref, dwu_ref, dwd_ref
    elif rank == 0:
        print(f"[rank {rank}] skipping dense reference "
              f"(needs ~{ref_bytes / (1 << 30):.0f} GiB, {free_bytes / (1 << 30):.0f} GiB free)")

    # ---- timed train steps (fwd + bwd + grad reduce) ----
    for _ in range(10):
        moe.zero_grad(set_to_none=True)
        x.grad = None
        y2 = moe(x)
        (y2.float() * gout).sum().backward()
        g_gate2, _, _ = moe.reduce_expert_grads()

    torch.cuda.synchronize()
    dist.barrier()
    time_start = time.perf_counter()
    n_iters = 100
    for _ in range(n_iters):
        moe.zero_grad(set_to_none=True)
        x.grad = None
        y2 = moe(x)
        (y2.float() * gout).sum().backward()
        g_gate2, _, _ = moe.reduce_expert_grads()
    torch.cuda.synchronize()
    time_end = time.perf_counter()

    ms = (time_end - time_start) / n_iters * 1e3
    t = torch.tensor([ms], device=dev)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    print(f"[rank {rank}] {ms:.3f} ms/step fwd+bwd+reduce (slowest rank: {t.item():.3f} ms)")

    # buffer/reduce-slot reuse across steps must reproduce step 1 exactly
    step_diff = (g_gate2 - g_gate).abs().max().item()
    print(f"[rank {rank}] step replay dw_gate diff {step_diff:.3e}")

    torch.cuda.synchronize()
    dist.barrier()
    moe.destroy()
    dist.destroy_process_group()

# torchrun --nproc_per_node=8 moe_moon_ep.py
if __name__ == '__main__':
    main()
