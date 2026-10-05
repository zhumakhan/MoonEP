'''
Run:
    torchrun --nproc-per-node=8 my_moon_ep.py
'''

import math
import os
import time
from typing import Any

import torch, torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from moonep import Buffer
from moonep._C import nvl_dist_alloc, nvl_dist_map, nvl_release_mem_handle, get_vmm_granularity
from moonep.buffer import (
    _all_gather_shareables, _exchange_ipc_fds, _use_fabric_for_group
)
from moonep.inter_rank_sync import launch_inter_rank_sync

# torch 2.14 renamed ``all_gather_into_tensor`` to ``all_gather_single``.
_all_gather_single = getattr(dist, "all_gather_single", None) or dist.all_gather_into_tensor


class _MoonEpDispatch(torch.autograd.Function):
    """dispatch fwd / combine bwd.

    Forward scatters [s, H] tokens (and their [s, K] routing weights),
    1 <= s <= S, into the expert-grouped shard across ranks. The returned
    h_nvs / w_nvs are [plan.nvs_s, H] / [plan.nvs_s] views of that shard
    (nvs_s = ceil(total_tokens * K / R) + segment padding bound, rounded up to
    token_padding; <= NvS and the same value on every rank), so they scale
    with the step, not the Buffer capacity. Backward combines the slot
    gradients back: each token's K slot grads are summed into grad_x, and each
    slot's weight grad is gathered back to its (token, k) position. combine
    accepts the [nvs_s, ...] grads directly.
    """

    @staticmethod
    def forward(
        ctx: Any,
        x: torch.Tensor,  # [s, H]
        weights: torch.Tensor,  # [s, K], routing weights
        topk: torch.Tensor,  # [s, K], top-k expert ids of every token
        tpe: torch.Tensor,  # [E], local histogram of tokens over experts
        buffer: Buffer,  # MoonEP's buffer object
        holder: Any,  # dict for bookkeeping (receives the plan)
        total_num_tokens: int | None,  # sum of s over the EP group; None = same s everywhere
    ):
        x_nvs, w_nvs, cu_seqlens, plan = buffer.dispatch(
            x.contiguous(), weights.contiguous(), topk, tpe, zero_copy=False,
            total_num_tokens=total_num_tokens,
        )
        holder['plan'] = plan
        ctx.buffer, ctx.plan = buffer, plan
        ctx.mark_non_differentiable(cu_seqlens)
        return x_nvs, w_nvs, cu_seqlens

    @staticmethod
    def backward(
        ctx: Any,
        grad_h: torch.Tensor,
        grad_w: torch.Tensor,
        _grad_cu: torch.Tensor
    ):
        grad_x, grad_w_sk, _ = ctx.buffer.combine(
            plan=ctx.plan,
            hidden_nvsh=grad_h.contiguous(),
            route_weights_nvs=grad_w.contiguous() if grad_w is not None else None,
        )
        return grad_x, grad_w_sk, None, None, None, None, None


class _MoonEPCombine(torch.autograd.Function):
    """combine fwd / dispatch bwd.

    Forward gathers every token's K expert outputs from the [plan.nvs_s, H]
    view of the shard back to its source rank and sums -> [s, H]. Backward
    re-dispatches the output grad with the saved plan: grad_out[t] is
    scattered to each of token t's slots (duplicate slots included, via the
    epilogue expansion) and comes back as the same [plan.nvs_s, H] view.
    """

    @staticmethod
    def forward(ctx, z_nvs, buffer, plan):
        # Buffer.combine also accepts the full [NvS, H] shard, but backward
        # returns a [plan.nvs_s, H] grad, so autograd needs the prefix view here.
        assert z_nvs.shape[0] == plan.nvs_s, (
            f"expected the [plan.nvs_s={plan.nvs_s}, H] view dispatch returned, "
            f"got {tuple(z_nvs.shape)}"
        )
        out, _, _ = buffer.combine(plan=plan, hidden_nvsh=z_nvs.contiguous())
        ctx.buffer, ctx.plan = buffer, plan
        return out

    @staticmethod
    def backward(ctx, grad_out):
        grad_z, _, _, _ = ctx.buffer.dispatch(
            grad_out.contiguous(), plan=ctx.plan,
        )
        return grad_z, None, None


class _ExpertGroupedMM(torch.autograd.Function):
    """``torch._grouped_mm`` over the [2*epn] compute view whose weight grad
    goes straight to fp32 storage.

    Rows [0, epn) of the view are this rank's own experts and rows
    [epn, 2*epn) its prefetch slots, and the planner's ``cu_seqlens`` covers
    exactly those 2*epn groups. So dW is [2*epn, ...] and there are no dead
    groups to skip; the old [E+B] pool needed a compacted ``offs`` for that.
    A group with no tokens this step (an own expert nobody routed to, an
    unused slot) costs no FLOPs and its dW is exactly zero. dW is added into
    the fp32 targets (this rank's expert grads, then its reduce slots), so the
    bf16 view never gets a ``.grad``.
    """

    @staticmethod
    def forward(ctx, x, w, offs, fp32_targets):
        # `w` rides on ctx rather than `save_for_backward`: prefetch_weight
        # writes its slot rows in place outside autograd and it carries no grad
        # of its own, so version tracking would only produce false positives.
        ctx.save_for_backward(x)
        ctx.w, ctx.offs, ctx.fp32_targets = w, offs, fp32_targets
        return torch._grouped_mm(x, w, offs=offs)

    @staticmethod
    def backward(ctx, grad_out):
        x, = ctx.saved_tensors
        grad_out = grad_out.contiguous()
        grad_x = torch._grouped_mm(grad_out, ctx.w.transpose(-2, -1), offs=ctx.offs)
        grad_w = torch._grouped_mm(x.t(), grad_out, offs=ctx.offs)
        with torch.no_grad():
            own, slots = ctx.fp32_targets
            own.add_(grad_w[: own.shape[0]])
            slots.add_(grad_w[own.shape[0]:])
        return grad_x, None, None, None


class MoonEPMoe(nn.Module):
    """Top-K MoE with grouped GEMM experts, expert-parallel via MoonEP.

    Mixed precision: fp32 masters (``parameters()``) for the optimizer, bf16
    compute copies of the experts for the forward/backward, fp32 gating. The
    expert masters' ``.grad`` alias the module's persistent fp32 grad storage,
    so use ``moe.zero_grad()`` (never ``opt.zero_grad(set_to_none=True)``) and
    call ``sync_compute_weights()`` after every ``opt.step()``.

    Expert memory per projection (epn = E/R experts per rank):
      - ``w_*`` [2*epn, in, out] bf16: one VA that maps this rank's own-expert
        chunk and then its prefetch chunk, the layout ``cu_seqlens`` indexes;
      - ``pf_*`` [R, epn, in, out] bf16: every rank's prefetch chunk;
        ``prefetch_weight`` pushes each rank's experts into the slots of the
        ranks that need them, so ``pf_*[rank]`` is the same memory as
        ``w_*[epn:]``;
      - ``g_*`` [epn, in, out] fp32: this rank's expert grads (the masters'
        ``.grad``);
      - ``rb_*`` [R, epn, in, out] fp32: every rank's reduce chunk. The dW of
        the experts copied to this rank lands in ``rb_*[rank]``, and
        ``reduce_grad`` sums every rank's slots into the owners' ``g_*``.
    The pools belong to this one layer. A model with several layers can share
    one prefetch pool and one reduce pool across layers, but must then
    re-prefetch before each layer's backward.
    """

    def __init__(self, E, K, H, Hi, S, group=None, num_sms=32):
        super().__init__()
        self.E, self.K, self.H, self.Hi = E, K, H, Hi
        self.group = group
        self.rank = dist.get_rank(group)
        self.R = dist.get_world_size(group)
        self.epn = E // self.R
        dev = torch.device('cuda', torch.cuda.current_device())

        self.buffer = Buffer(S, H, K, E, self.R, num_sms=num_sms, group=group)
        self.lo, self.hi = self.rank * self.epn, (self.rank + 1) * self.epn

        # ----- router: fp32, replicated (same seed on every rank, own generator)
        # and used directly by the forward. Gating stays fp32 end to end: bf16
        # logits flip top-k choices on near-ties, which destabilizes MoE
        # training. Its grad is summed over the EP group in reduce_expert_grads().
        router_w = torch.empty(E, H, dtype=torch.float32)
        nn.init.kaiming_uniform_(  # nn.Linear's default init
            router_w, a=math.sqrt(5), generator=torch.Generator().manual_seed(0)
        )
        self.router_w = nn.Parameter(router_w.to(dev))

        # ----- experts: bf16 compute views + prefetch pools, fp32 grads +
        # reduce pools (see the class docstring). Buffers, not Parameters: the
        # optimizer must never see them. Every rank allocates and maps in the
        # same order (the handle exchange is collective).
        self._keepalives = []
        self._use_fabric = _use_fabric_for_group(group)
        shapes = (('gate', (self.epn, H, Hi)),
                  ('up', (self.epn, H, Hi)),
                  ('down', (self.epn, Hi, H)))
        for name, shape in shapes:
            view, pool = self._weight_buffers(shape)
            self.register_buffer(f'w_{name}', view, persistent=False)
            setattr(self, f'pf_{name}', pool)
        for name, shape in shapes:
            setattr(self, f'g_{name}', torch.zeros(shape, dtype=torch.float32, device=dev))
            rb = self._reduce_pool(shape)
            # cuMemCreate memory is not zero-filled, and the backward adds into
            # this rank's slots; reduce_grad clears the slots it consumes.
            rb[self.rank].zero_()
            setattr(self, f'rb_{name}', rb)

        # ---- experts, fp32 masters of this rank's own rows: plain local
        # tensors. Their .grad permanently alias g_*, so after
        # reduce_expert_grads() they hold the reduced grad.
        g = torch.Generator(device=dev).manual_seed(100 + self.rank)

        def master(*shape):
            return nn.Parameter(
                torch.empty(*shape, dtype=torch.float32, device=dev)
                .normal_(0.0, 0.02, generator=g)
            )

        self.wm_gate = master(self.epn, H, Hi)
        self.wm_up = master(self.epn, H, Hi)
        self.wm_down = master(self.epn, Hi, H)
        self.wm_gate.grad = self.g_gate
        self.wm_up.grad = self.g_up
        self.wm_down.grad = self.g_down
        self.sync_compute_weights()  # bf16 copies of the own expert rows

        self._last_plan = None
        torch.cuda.synchronize()
        dist.barrier(group=group)

    # ---- VMM mapping helpers --------------------------------------------
    # A "share item" is what nvl_dist_map imports for one chunk: an int fd on
    # the same node, or a uint8[64] CPU fabric handle. Every rank calls these
    # in the same order (the fd exchange is collective). Received fds are dups
    # owned by this process and are closed once the mappings exist.

    def _alloc_chunk(self, chunk_shape, dtype):
        """Allocate one VMM chunk on this GPU; return its exported shareable."""
        nbytes = dtype.itemsize
        for d in chunk_shape:
            nbytes *= d
        gran = get_vmm_granularity()
        assert nbytes % gran == 0, (
            f"chunk {tuple(chunk_shape)} {dtype} = {nbytes} B must be a "
            f"multiple of VMM granularity {gran}"
        )
        ka, share, owned = nvl_dist_alloc(
            shape=list(chunk_shape), dtype=dtype, use_fabric=self._use_fabric
        )
        self._keepalives.append(ka)
        nvl_release_mem_handle(owned)
        return share

    def _local_item(self, share):
        return share.cpu().view(-1) if self._use_fabric else int(share.item())

    def _gathered_items(self, share):
        """Every rank's item for the chunk `share` describes on that rank."""
        if self._use_fabric:
            handles = _all_gather_shareables(share, self.group)  # uint8 [R, 64] CPU
            return [handles[r] for r in range(self.R)]
        fds = _exchange_ipc_fds(int(share.item()), list(range(self.R)),
                                self.rank, self.R, self.group)
        return [fds[r] for r in range(self.R)]

    def _map_items(self, chunk_shape, dtype, items):
        """Map the chunks in `items` back to back into one contiguous VA."""
        if self._use_fabric:
            shareables = torch.stack(items)
        else:
            shareables = torch.tensor(items, dtype=torch.int64)
        return nvl_dist_map(
            chunk_shape=list(chunk_shape), dtype=dtype, shareables=shareables,
            local_rank=self.rank, world_size=len(items), use_fabric=self._use_fabric,
        )

    def _close_items(self, items):
        if not self._use_fabric:
            for fd in items:
                os.close(fd)

    def _weight_buffers(self, chunk_shape):
        """bf16 (compute view [2*epn, ...], prefetch pool [R, epn, ...])."""
        w_share = self._alloc_chunk(chunk_shape, torch.bfloat16)  # own experts
        p_share = self._alloc_chunk(chunk_shape, torch.bfloat16)  # prefetch slots
        p_items = self._gathered_items(p_share)
        own = [self._local_item(w_share), self._local_item(p_share)]
        try:
            view = self._map_items(chunk_shape, torch.bfloat16, own)
            pool = self._map_items(chunk_shape, torch.bfloat16, p_items)
        finally:
            self._close_items(p_items + own)
        return view, pool.view(self.R, *chunk_shape)

    def _reduce_pool(self, chunk_shape):
        """fp32 [R, epn, ...]: every rank's reduce chunk."""
        r_share = self._alloc_chunk(chunk_shape, torch.float32)
        r_items = self._gathered_items(r_share)
        try:
            pool = self._map_items(chunk_shape, torch.float32, r_items)
        finally:
            self._close_items(r_items + [self._local_item(r_share)])
        return pool.view(self.R, *chunk_shape)

    def zero_grad(self, set_to_none: bool = True):
        """Zero the expert masters' persistent fp32 grads in place (their
        .grad alias them, so `set_to_none` is ignored for them) and drop the
        router's ordinary autograd grad. The bf16 compute views never get a
        .grad; reduce_grad clears the reduce slots it consumes.
        """
        with torch.no_grad():
            for g in (self.g_gate, self.g_up, self.g_down):
                g.zero_()
        self.router_w.grad = None

    @torch.no_grad()
    def sync_compute_weights(self):
        """fp32 expert masters -> rows [0, epn) of the bf16 compute views (the
        router has no bf16 copy). Call after every optimizer step. Only this
        rank reads those rows: its next prefetch_weight pushes them to the
        ranks that need copies, so stream order is the only ordering needed.
        """
        for w16, wm in ((self.w_gate, self.wm_gate),
                        (self.w_up, self.wm_up),
                        (self.w_down, self.wm_down)):
            w16[: self.epn].copy_(wm)

    def route(self, x):
        logits = F.linear(x.float(), self.router_w)  # fp32 gating: GEMM, top-k and softmax
        weights, idx = torch.topk(logits, k=self.K, dim=-1)
        weights = F.softmax(weights, dim=-1)
        topk = idx.to(torch.int32)
        tpe = torch.bincount(topk.flatten(), minlength=self.E).to(torch.int32)
        return weights.contiguous(), topk.contiguous(), tpe

    def forward(self, x, total_num_tokens=None):
        """[s, H] -> [s, H] for any 1 <= s <= S. S is only the Buffer's
        capacity: the comm kernels loop over s, no padding tokens are made.
        Ranks may pass different s when every rank passes ``total_num_tokens``
        (the sum of s over the EP group).
        """
        weights, topk, tpe = self.route(x)

        # dispatch: scatter tokens to their experts' home/copy ranks over
        # NVLink (autograd-aware: backward is combine with the saved plan)
        holder = {}
        h_nvs, w_nvs, cu_seqlens = _MoonEpDispatch.apply(
            x, weights, topk, tpe, self.buffer, holder, total_num_tokens
        )
        plan = holder['plan']
        self._last_plan = plan
        # push this rank's experts into the prefetch slots of the ranks that
        # copy them. Collective: every rank calls it, and when it returns this
        # rank's own slots (w_*[epn:]) are filled. Raw kernel writes outside
        # autograd; the slots must keep this step's experts until the backward.
        epn = self.epn
        self.buffer.prefetch_weight(
            plan=plan,
            local_gate_weight=self.w_gate[:epn],
            local_up_weight=self.w_up[:epn],
            local_down_weight=self.w_down[:epn],
            gate_prefetch_buffer=self.pf_gate,
            up_prefetch_buffer=self.pf_up,
            down_prefetch_buffer=self.pf_down,
        )

        # grouped GEMM straight over the dispatched layout: cu_seqlens [2*epn]
        # is already _grouped_mm's offs format, groups [0, epn) are this rank's
        # experts and groups [epn, 2*epn) its copies. Padding rows are
        # zero-filled by dispatch, so their FFN output is exactly zero. No host
        # sync: _grouped_mm only touches rows inside the cu_seqlens groups, and
        # cu_seqlens[-1] <= plan.nvs_s keeps offs inside the [plan.nvs_s, H]
        # view. Rows past cu_seqlens[-1] are left uninitialized; no dst slot
        # references them, so nothing reads them.
        gmm = _ExpertGroupedMM.apply
        r = self.rank
        gate = gmm(h_nvs, self.w_gate, cu_seqlens, (self.g_gate, self.rb_gate[r]))
        up = gmm(h_nvs, self.w_up, cu_seqlens, (self.g_up, self.rb_up[r]))
        y = gmm(F.silu(gate) * up, self.w_down, cu_seqlens, (self.g_down, self.rb_down[r]))

        # scale each slot by its routing weight. Padding and tail slots carry
        # stale weight values (never written for this step); nan_to_num keeps
        # 0-row * garbage from minting NaNs that a GEMM backward would spread
        w_used = torch.nan_to_num(w_nvs)
        z = (y.float() * w_used[:, None]).to(torch.bfloat16)

        # combine: gather every token's K expert outputs back to its source
        # rank and sum -> [s, H] (backward: re-dispatch with the saved plan)
        return _MoonEPCombine.apply(z, self.buffer, plan)

    def reduce_expert_grads(self):
        """dispatch bwd, weight side: sum the copied experts' grads (this
        rank's reduce slots rb_*[rank]) into their owners' g_* with
        Buffer.reduce_grad, and sum the replicated router's grad over the EP
        group.

        Call once per micro-batch, after backward and before any DP grad
        reduction / optimizer step: the next forward's plan reassigns the
        slots. The router all-reduce assumes one micro-batch per step; with
        gradient accumulation, sum the router grad once per step instead.
        Returns the fp32 [epn, ...] grads (gate, up, down).
        """
        assert self._last_plan is not None, "reduce_expert_grads needs a prior forward"
        assert self.router_w.grad is not None, "call backward() before reduce_expert_grads()"
        # The router is replicated across the EP group but every rank routed a
        # different token batch: sum the local grads (the loss is a sum over
        # tokens, matching the summed expert grads) so every replica takes the
        # same optimizer step and stays bit-identical.
        dist.all_reduce(self.router_w.grad, op=dist.ReduceOp.SUM, group=self.group)
        # reduce_grad has no entry barrier: every rank's backward must have
        # finished writing its reduce slots before any rank reads them.
        # Device-side sync, no host round trip.
        launch_inter_rank_sync(self.buffer._require_ctx())
        self.buffer.reduce_grad(
            plan=self._last_plan,
            local_gate_grad=self.g_gate,
            local_up_grad=self.g_up,
            local_down_grad=self.g_down,
            gate_reduce_buffer=self.rb_gate,
            up_reduce_buffer=self.rb_up,
            down_reduce_buffer=self.rb_down,
        )
        return self.g_gate, self.g_up, self.g_down

    def destroy(self):
        self.buffer.destroy()


def check_against_dense_reference(moe, s, total_num_tokens=None, seed=0, tol=3e-2):
    """One fwd + bwd + reduce_expert_grads() at s tokens on this rank,
    compared with a dense per-expert reference that runs the same math on
    every rank's experts (all-gathered, so the check needs no MoonEP memory).

    Returns the worst relative error (max abs error / max abs reference) over
    the EP group of (output, dx, router grad, expert grads) and whether all are
    within ``tol``. Both sides round to bf16 at the same points, but the
    reference accumulates each token's K slot grads in bf16, so dx differs by
    up to ~1-2% at bf16.
    """
    dev = torch.device('cuda', torch.cuda.current_device())
    gen = torch.Generator(device=dev).manual_seed(1000 * seed + moe.rank)
    x = torch.randn(s, moe.H, dtype=torch.bfloat16, device=dev, generator=gen).requires_grad_()
    gout = torch.randn(s, moe.H, dtype=torch.float32, device=dev, generator=gen)

    moe.zero_grad()
    out = moe(x, total_num_tokens=total_num_tokens)
    (out.float() * gout).sum().backward()
    g_gate, g_up, g_down = moe.reduce_expert_grads()

    # dense reference: same routing (fp32 router), same rounding points
    wr = {}
    for name, w16 in (('gate', moe.w_gate), ('up', moe.w_up), ('down', moe.w_down)):
        full = torch.empty(moe.E, *w16.shape[1:], dtype=w16.dtype, device=dev)
        _all_gather_single(full, w16[: moe.epn].contiguous(), group=moe.group)
        wr[name] = full.requires_grad_()
    xr = x.detach().clone().requires_grad_()
    rw = moe.router_w.detach().clone().requires_grad_()
    logits = F.linear(xr.float(), rw)
    w, idx = torch.topk(logits, k=moe.K, dim=-1)
    w = F.softmax(w, dim=-1)
    flat_w, flat_e = w.flatten(), idx.flatten()
    flat_t = torch.arange(s, device=dev).repeat_interleave(moe.K)
    ref = torch.zeros(s, moe.H, dtype=torch.float32, device=dev)
    for e in range(moe.E):
        sel = flat_e == e
        if not bool(sel.any()):
            continue
        t = flat_t[sel]
        y = (F.silu(xr[t] @ wr['gate'][e]) * (xr[t] @ wr['up'][e])) @ wr['down'][e]
        z = (y.float() * flat_w[sel, None]).to(torch.bfloat16)
        ref = ref.index_add(0, t, z.float())
    ref = ref.to(torch.bfloat16)
    (ref.float() * gout).sum().backward()
    # sum every rank's contributions in fp32, like reduce_grad does
    ref_grads = [rw.grad] + [wr[n].grad.float() for n in ('gate', 'up', 'down')]
    for t in ref_grads:
        dist.all_reduce(t, group=moe.group)
    ref_router, ref_gate, ref_up, ref_down = ref_grads

    def rel(a, b):
        return ((a.float() - b.float()).abs().max() / b.float().abs().max().clamp(min=1e-6)).item()

    lo, hi = moe.lo, moe.hi
    errs = torch.tensor([
        rel(out, ref),
        rel(x.grad, xr.grad),
        rel(moe.router_w.grad, ref_router),
        max(rel(g_gate, ref_gate[lo:hi]), rel(g_up, ref_up[lo:hi]),
            rel(g_down, ref_down[lo:hi])),
    ], device=dev)
    dist.all_reduce(errs, op=dist.ReduceOp.MAX, group=moe.group)
    errs = errs.tolist()
    return errs, all(e <= tol for e in errs)


def main():
    local_rank = int(os.getenv('LOCAL_RANK'))
    dist.init_process_group(backend='nccl', device_id=torch.device('cuda', local_rank))
    rank = dist.get_rank()
    R = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    dev = f'cuda:{local_rank}'

    # ---- correctness: a small layer against a dense reference, at the full
    # capacity, partial s and per-rank s. Half the router rows are zeroed so
    # the routing is skewed and the prefetch / grad-reduce paths run.
    S_chk = 512
    chk = MoonEPMoe(E=8 * R, K=8, H=1024, Hi=1024, S=S_chk)
    with torch.no_grad():
        chk.router_w[: chk.E // 2].zero_()
    per_rank = [S_chk, 37, 1, 300, 64, 5, 128, 256]
    s_r = per_rank[rank % len(per_rank)]
    total = sum(per_rank[r % len(per_rank)] for r in range(R))
    for label, s, tot in (('s=S', S_chk, None), ('s=131', 131, None), ('s=1', 1, None),
                          (f'per-rank s (total={total})', s_r, total)):
        errs, ok = check_against_dense_reference(chk, s, tot, seed=s)
        if rank == 0:
            print(f"[check {label}] {'PASS' if ok else 'FAIL'} max rel err: out {errs[0]:.1e}, "
                  f"dx {errs[1]:.1e}, router grad {errs[2]:.1e}, expert grads {errs[3]:.1e}",
                  flush=True)
    chk.destroy()
    del chk

    # ---- training-step timing
    S, K, E, H, Hi = 1024 * 4, 16, 128, 4096, 4096 * 2
    moe = MoonEPMoe(E, K, H, Hi, S)

    # skew the routing by zeroing part of the (fp32, replicated) router
    with torch.no_grad():
        moe.router_w[: E // 2].zero_()

    # AdamW over the fp32 masters only: parameters() excludes the bf16 compute
    # copies. The masters' .grad alias MoonEPMoe's persistent fp32 grad
    # storage, so grads are cleared with moe.zero_grad(), never opt.zero_grad().
    opt = torch.optim.AdamW(moe.parameters(), lr=1e-4, weight_decay=0.0)

    g = torch.Generator(device=dev).manual_seed(42 + rank)
    x = torch.randn(S, H, dtype=torch.bfloat16, device=dev, generator=g).requires_grad_(True)
    gout = torch.randn(S, H, dtype=torch.float32, device=dev, generator=g)

    def train_step():
        moe.zero_grad()
        x.grad = None
        out = moe(x)
        (out.float() * gout).sum().backward()
        moe.reduce_expert_grads()
        opt.step()
        moe.sync_compute_weights()

    for _ in range(10):
        train_step()

    torch.cuda.synchronize()
    dist.barrier()
    time_start = time.perf_counter()
    n_iters = 100
    for _ in range(n_iters):
        train_step()
    torch.cuda.synchronize()
    time_end = time.perf_counter()

    ms = (time_end - time_start) / n_iters * 1e3
    t = torch.tensor([ms], device=dev)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    print(f"[rank {rank}] {ms:.3f} ms/step fwd+bwd+reduce+AdamW (slowest rank: {t.item():.3f} ms)")

    # after the last refresh the bf16 compute copies must be exactly the cast
    # masters, and the replicated router must be bit-identical on every rank
    with torch.no_grad():
        consistent = all(
            torch.equal(w16[: moe.epn], wm.to(torch.bfloat16))
            for w16, wm in ((moe.w_gate, moe.wm_gate), (moe.w_up, moe.wm_up),
                            (moe.w_down, moe.wm_down))
        )
        router_ref = moe.router_w.detach().clone()
        dist.broadcast(router_ref, src=0)
        router_in_sync = torch.equal(moe.router_w, router_ref)

    print(f"[rank {rank}] bf16 compute copies == cast fp32 masters: {consistent}; "
          f"router identical across ranks: {router_in_sync}")
    torch.cuda.synchronize()
    dist.barrier()
    moe.destroy()
    dist.destroy_process_group()


# torchrun --nproc-per-node=8 my_moon_ep.py
if __name__ == '__main__':
    main()
