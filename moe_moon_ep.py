"""Expert-parallel top-K MoE over NVLink GPUs using MoonEP dispatch/combine.

Mirrors moe.py's grouped-GEMM expert compute, but the experts are sharded
across the EP group: rank r owns experts [r*epn, (r+1)*epn). MoonEP moves the
tokens (dispatch/combine over NVLink) and prefetches the weights of the
load-balancing "copied" experts into the B local prefetch slots.

Weight storage follows bench_vs_deepep's composite [E+B] layout: every rank
allocates its own [epn, H, *] expert chunk plus a [B, H, *] prefetch chunk,
exchanges VMM handles, and maps all R expert chunks + its local prefetch chunk
into one contiguous VA. Rows [0, E) are the global expert pool (remote rows
walk NVLink), rows [E, E+B) are local prefetch slots — exactly the tensor
Buffer.prefetch_weight expects. Requires B == epn and the chunk byte size to
be a multiple of the VMM granularity.

Gradients mirror that layout in fp32, which is what Buffer.reduce_grad takes:
per projection one contiguous [E+B, H, *] fp32 "main grad" whose rows [0, E)
are every rank's own-expert grad chunk (this rank writes only its own rows)
and whose rows [E, E+B) are this rank's reduce chunk. The same reduce chunks
are also mapped as the [R, B, H, *] all-rank view reduce_grad reads. A
post-accumulate-grad hook folds autograd's bf16 .grad into the main grad
(own rows + copied-expert rows) and frees it, so copied experts' grads land
in the reduce buffer directly, with no publish copy.

Autograd: dispatch and combine are transposes of each other, so each gets a
custom autograd.Function whose backward is the other call with the saved plan
(api.py's documented recipe). The FFN between them is ordinary autograd over
the [E+B] weight Parameters. reduce_expert_grads() then runs
Buffer.reduce_grad: every rank remote-reads the reduce slots that hold its
own experts, accumulates them into its main-grad rows, and clears the slots
it consumed for the next microbatch.

Precision: fp32 master weights, bf16 expert compute, fp32 gating.
``parameters()`` are the fp32 masters (this rank's [epn, H, *] expert rows
and the replicated router); the optimizer updates only those. The expert
forward/backward run on bf16 compute copies (non-persistent buffers): the
[E+B] composites above. Their bf16 grads are folded into fp32 storage that
the expert masters' .grad permanently alias, so after reduce_expert_grads()
the optimizer sees fully reduced fp32 grads. sync_compute_weights() casts the
masters back into the bf16 copies after every optimizer step; peers pick up
the new expert rows over NVLink at their next prefetch, which every dispatch's
inter-rank sync orders after the refresh. The router has no bf16 copy: gating
(logits, top-k, softmax) runs in fp32 on the master itself, the usual MoE
practice since bf16 logits flip top-k choices on near-ties. Every rank routes
a different token batch, so reduce_expert_grads() also sums the router grad
over the EP group; the replicas then take identical optimizer steps.

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
from moonep.buffer import (
    _all_gather_shareables, _exchange_ipc_fds, _use_fabric_for_group,
)
from moonep.inter_rank_sync import launch_inter_rank_sync


class _MoonEPDispatch(torch.autograd.Function):
    """dispatch fwd / combine bwd.

    Forward scatters [s, H] tokens (and their [s, K] routing weights),
    1 <= s <= S, into the expert-grouped shard across ranks. The returned
    h_nvs / w_nvs are [plan.nvs_s, H] / [plan.nvs_s] prefix views of that
    shard (nvs_s = ceil(total_tokens*K/R) + segment padding bound <= NvS, the same
    value on every rank),
    so they scale with the step, not the Buffer capacity. Backward combines
    the slot gradients back: each token's K slot grads are summed into
    grad_x, and each slot's weight grad is gathered back to its (token, k)
    position. combine accepts the [nvs_s, ...] grads directly.
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


class MoonEPMoE(nn.Module):
    """Top-K MoE with grouped-GEMM experts, expert-parallel via MoonEP.

    Mixed precision: fp32 masters (``parameters()``) for the optimizer, bf16
    compute copies of the experts for the forward/backward, fp32 gating. The
    expert masters' ``.grad`` alias the module's persistent fp32 grad storage,
    so use ``moe.zero_grad()`` (never ``opt.zero_grad(set_to_none=True)``) and
    call ``sync_compute_weights()`` after every ``opt.step()``.
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
        self.B = int(self.buffer._require_ctx()['B'])
        assert self.B == self.epn, "composite [E+B] weight layout expects B == epn"

        self.lo, self.hi = self.rank * self.epn, (self.rank + 1) * self.epn

        # ---- router: fp32, replicated (fixed seed on every rank) and used
        # directly by the forward. Gating stays fp32 end to end: bf16 logits
        # flip top-k choices on near-ties, which destabilizes MoE training.
        # Its grad is summed over the EP group in reduce_expert_grads().
        torch.manual_seed(0)
        router = nn.Linear(H, E, bias=False)
        self.router_w = nn.Parameter(router.weight.detach().to(dev, torch.float32))

        # ---- experts, bf16 compute copies: composite [E+B, ...] VMM pools.
        # This rank physically owns rows [lo, hi) and the prefetch rows
        # [E, E+B). Buffers, not Parameters: the optimizer must never see them.
        self._keepalives = []
        for name, shape in (('w_gate', (self.epn, H, Hi)),
                            ('w_up', (self.epn, H, Hi)),
                            ('w_down', (self.epn, Hi, H))):
            w16 = self._composite_weight(shape, group).requires_grad_(True)
            self.register_buffer(name, w16, persistent=False)

        # fp32 main grads in the same [E+B, ...] layout (what Buffer.reduce_grad
        # takes) plus the [R, B, ...] all-rank view of the reduce chunks. Rows
        # [E, E+B) of each main grad ARE this rank's reduce chunk, so the FFN
        # backward's copied-expert grads land in the reduce buffer directly.
        self.mg_gate, self.rb_gate = self._grad_buffers((self.epn, H, Hi), group)
        self.mg_up, self.rb_up = self._grad_buffers((self.epn, H, Hi), group)
        self.mg_down, self.rb_down = self._grad_buffers((self.epn, Hi, H), group)
        with torch.no_grad():
            for mg in (self.mg_gate, self.mg_up, self.mg_down):
                mg[self.lo:self.hi].zero_()
                mg[self.E:].zero_()   # == rb_*[rank]

        # ---- experts, fp32 masters of this rank's own rows: plain local
        # tensors. Their .grad permanently alias the owned rows of the main
        # grads, so after reduce_expert_grads() they hold the reduced grad.
        g = torch.Generator(device=dev).manual_seed(100 + self.rank)

        def master(*shape):
            return nn.Parameter(
                torch.empty(*shape, dtype=torch.float32, device=dev)
                .normal_(0.0, 0.02, generator=g))

        self.wm_gate = master(self.epn, H, Hi)
        self.wm_up = master(self.epn, H, Hi)
        self.wm_down = master(self.epn, Hi, H)
        self.wm_gate.grad = self.mg_gate[self.lo:self.hi]
        self.wm_up.grad = self.mg_up[self.lo:self.hi]
        self.wm_down.grad = self.mg_down[self.lo:self.hi]
        self.sync_compute_weights()   # bf16 copies of the own expert rows

        # fold autograd's bf16 grads into the fp32 storage as soon as each
        # compute copy's grad is accumulated, then drop the bf16 grad
        lo, hi, E_ = self.lo, self.hi, self.E
        self._grad_hooks = [
            self._register_grad_fold(w16, [(slice(lo, hi), mg[lo:hi]),
                                           (slice(E_, None), mg[E_:])])
            for w16, mg in ((self.w_gate, self.mg_gate),
                            (self.w_up, self.mg_up),
                            (self.w_down, self.mg_down))
        ]
        self._last_plan = None
        torch.cuda.synchronize()
        dist.barrier(group=group)

    # ---- VMM mapping helpers ------------------------------------------------
    # A "share item" is what nvl_dist_map imports for one chunk: an int fd on
    # the same node, or a uint8[64] CPU fabric handle. Every rank calls these
    # in the same order (the fd exchange is collective). Received fds are dups
    # owned by this process and are closed once the mappings exist.

    def _alloc_chunk(self, chunk_shape, dtype, use_fabric):
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
            shape=list(chunk_shape), dtype=dtype, use_fabric=use_fabric)
        self._keepalives.append(ka)
        nvl_release_mem_handle(owned)
        return share

    @staticmethod
    def _local_item(share, use_fabric):
        return share.cpu().view(-1) if use_fabric else int(share.item())

    def _gathered_items(self, share, group, use_fabric):
        """Every rank's item for the chunk `share` describes on that rank."""
        if use_fabric:
            handles = _all_gather_shareables(share, group)   # uint8 [R, 64] CPU
            return [handles[r] for r in range(self.R)]
        fds = _exchange_ipc_fds(int(share.item()), list(range(self.R)),
                                self.rank, self.R, group)
        return [fds[r] for r in range(self.R)]

    def _map_items(self, chunk_shape, dtype, items, use_fabric):
        """Map the chunks in `items` back to back into one contiguous VA."""
        if use_fabric:
            shareables = torch.stack(items)
        else:
            shareables = torch.tensor(items, dtype=torch.int64)
        return nvl_dist_map(
            chunk_shape=list(chunk_shape), dtype=dtype, shareables=shareables,
            local_rank=self.rank, world_size=len(items), use_fabric=use_fabric,
        )

    @staticmethod
    def _close_items(items, use_fabric):
        if not use_fabric:
            for fd in items:
                os.close(fd)

    def _composite_weight(self, chunk_shape, group):
        """bf16 [E+B, ...]: all R ranks' expert chunks, then my prefetch chunk."""
        use_fabric = _use_fabric_for_group(group)
        w_share = self._alloc_chunk(chunk_shape, torch.bfloat16, use_fabric)
        b_share = self._alloc_chunk(chunk_shape, torch.bfloat16, use_fabric)
        w_items = self._gathered_items(w_share, group, use_fabric)
        b_item = self._local_item(b_share, use_fabric)
        try:
            return self._map_items(
                chunk_shape, torch.bfloat16, w_items + [b_item], use_fabric)
        finally:
            self._close_items(
                w_items + [b_item, self._local_item(w_share, use_fabric)], use_fabric)

    def _grad_buffers(self, chunk_shape, group):
        """fp32 (main_grad [E+B, ...], reduce view [R, B, ...]) for one projection.

        main_grad rows [0, E) are the R ranks' own-expert grad chunks: this
        rank writes only its own rows, the others are mapped so reduce_grad can
        address rows by global expert id. Rows [E, E+B) are this rank's reduce
        chunk. The reduce view maps every rank's reduce chunk, so its [rank]
        entry is the same physical memory as main_grad[E:]. B == epn keeps all
        chunks the same shape, which nvl_dist_map requires.
        """
        use_fabric = _use_fabric_for_group(group)
        g_share = self._alloc_chunk(chunk_shape, torch.float32, use_fabric)
        r_share = self._alloc_chunk(chunk_shape, torch.float32, use_fabric)
        g_items = self._gathered_items(g_share, group, use_fabric)
        r_items = self._gathered_items(r_share, group, use_fabric)
        r_local = self._local_item(r_share, use_fabric)
        try:
            main_grad = self._map_items(
                chunk_shape, torch.float32, g_items + [r_local], use_fabric)
            reduce = self._map_items(chunk_shape, torch.float32, r_items, use_fabric)
        finally:
            self._close_items(
                g_items + r_items + [r_local, self._local_item(g_share, use_fabric)],
                use_fabric)
        return main_grad, reduce.view(self.R, self.B, *chunk_shape[1:])

    def _register_grad_fold(self, w16, targets):
        """Post-accumulate-grad hook on a bf16 compute copy: add the listed
        row slices of its .grad into fp32 storage, then free the bf16 grad.

        For the expert composites only the owned rows and the prefetch rows
        are folded: rows of other ranks' experts are live remote mappings
        (and are zero here anyway, their cu_seqlens segments are empty).
        """
        def fold(t):
            g = t.grad
            with torch.no_grad():
                for slc, dst in targets:
                    dst.add_(g[slc])
            t.grad = None

        return w16.register_post_accumulate_grad_hook(fold)

    def zero_grad(self, set_to_none: bool = True):
        """Zero the expert masters' persistent fp32 grad storage in place
        (their .grad alias it, so `set_to_none` is ignored for them) and drop
        the router's ordinary autograd grad. The bf16 compute copies' .grad is
        already None after the fold hooks; reduce_grad clears the consumed
        reduce rows itself."""
        with torch.no_grad():
            for mg in (self.mg_gate, self.mg_up, self.mg_down):
                mg[self.lo:self.hi].zero_()
        self.router_w.grad = None

    @torch.no_grad()
    def sync_compute_weights(self):
        """fp32 expert masters -> bf16 compute copies (the router has no bf16
        copy). Call after every optimizer step. Only this rank's own expert
        rows of the composites are written; peers read them over NVLink at
        their next prefetch_weight, which the inter-rank sync at the start of
        every dispatch orders after this."""
        for w16, wm in ((self.w_gate, self.wm_gate),
                        (self.w_up, self.wm_up),
                        (self.w_down, self.wm_down)):
            w16[self.lo:self.hi].copy_(wm)

    def route(self, x):
        logits = F.linear(x.float(), self.router_w)     # fp32 gating: GEMM, top-k, softmax
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
        # groups, so feed the returned [plan.nvs_s, H] view as is. The
        # planner keeps every slot of this rank inside that prefix, i.e.
        # cu_seqlens[-1] <= plan.nvs_s, so offs stay in bounds; cu_seqlens
        # itself stays [E+B] for any s. Rows in [cu_seqlens[-1], nvs_s) are
        # referenced by no dst slot; they come out as exact zeros in forward
        # and their grads never reach a weight grad.
        gate = torch._grouped_mm(h_nvs, self.w_gate, offs=cu_seqlens)
        up = torch._grouped_mm(h_nvs, self.w_up, offs=cu_seqlens)
        y = torch._grouped_mm(F.silu(gate) * up, self.w_down, offs=cu_seqlens)

        # scale each slot by its routing weight. Padding and tail slots carry
        # stale weight values (never written for this step) — nan_to_num keeps
        # 0-row * garbage from minting NaNs that a GEMM backward would spread.
        w_used = torch.nan_to_num(w_nvs)
        z = (y.float() * w_used[:, None]).to(torch.bfloat16)

        # combine: gather every token's K expert outputs back to its source
        # rank and sum -> [s, H] (backward: re-dispatch with the saved plan)
        return _MoonEPCombine.apply(z, self.buffer, plan)


    def reduce_expert_grads(self):
        """dispatch bwd, weight side: ship the copied experts' grads (my reduce
        chunk == main_grad[E:]) to their home ranks and accumulate them into
        the owners' main-grad rows with Buffer.reduce_grad, and sum the
        replicated router's grad over the EP group.

        Call once per microbatch, after backward() and before any DP grad
        reduction / optimizer step: the next forward's plan reassigns the
        slots. Returns fp32 [epn, ...] views of the fully reduced owned rows
        (gate, up, down); the kernel zeros the consumed reduce slots itself."""
        assert self._last_plan is not None, "reduce_expert_grads needs a prior forward"
        ctx = self.buffer._require_ctx()
        # The router is replicated across the EP group but every rank routed a
        # different token batch: sum the local grads (the loss is a sum over
        # tokens, matching the summed expert grads) so every replica takes the
        # same optimizer step and stays bit-identical.
        assert self.router_w.grad is not None, "call backward() before reduce_expert_grads()"
        dist.all_reduce(self.router_w.grad, op=dist.ReduceOp.SUM, group=self.group)
        # reduce_grad has no entry barrier: every rank's backward (the grad
        # hooks) must have finished writing its reduce chunk before any rank
        # remote-reads it. Device-side sync, no host round trip.
        launch_inter_rank_sync(ctx)
        self.buffer.reduce_grad(
            plan=self._last_plan,
            full_gate_grad=self.mg_gate,
            full_up_grad=self.mg_up,
            full_down_grad=self.mg_down,
            gate_reduce_buffer=self.rb_gate,
            up_reduce_buffer=self.rb_up,
            down_reduce_buffer=self.rb_down,
        )
        lo, hi = self.lo, self.hi
        return self.mg_gate[lo:hi], self.mg_up[lo:hi], self.mg_down[lo:hi]

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
    rw = moe.router_w.detach().clone().requires_grad_()
    wg = moe.w_gate.detach()[:E].clone().requires_grad_()
    wu = moe.w_up.detach()[:E].clone().requires_grad_()
    wd = moe.w_down.detach()[:E].clone().requires_grad_()

    logits = xr.float() @ rw.T          # fp32 gating, like MoonEPMoE.route
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
    moe = MoonEPMoE(E, K, H, Hi, S)

    # skew the routing by zeroing part of the (fp32, replicated) router
    with torch.no_grad():
        moe.router_w[:64].zero_()

    # AdamW over the fp32 masters only: parameters() excludes the bf16 compute
    # copies. The masters' .grad alias MoonEPMoE's persistent fp32 grad
    # storage, so grads are cleared with moe.zero_grad(), never opt.zero_grad().
    opt = torch.optim.AdamW(moe.parameters(), lr=1e-4, weight_decay=0.0)

    g = torch.Generator(device=dev).manual_seed(42 + rank)
    x = torch.randn(S, H, dtype=torch.bfloat16, device=dev, generator=g).requires_grad_(True)
    gout = torch.randn(S, H, dtype=torch.float32, device=dev, generator=g)

    # ---- EP forward + backward + grad reduce (no update yet) ----
    y = moe(x)
    (y.float() * gout).sum().backward()
    g_gate, g_up, g_down = moe.reduce_expert_grads()
    g_gate_first = g_gate.clone()   # views into the main grads: snapshot for the replay check

    # ---- dense reference on the same bf16 compute weights (needs ~4x the
    # expert weight bytes for clones + their grads: skip when it cannot fit) ----
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
        # the router grad is summed over the EP group in reduce_expert_grads;
        # the reference saw only this rank's tokens, so sum it the same way
        drouter_total = drouter_ref.float().contiguous()
        dist.all_reduce(drouter_total)
        report("drouter", moe.router_w.grad, drouter_total)

        # expert grads: the true total sums every rank's token contributions;
        # compare my owned slice of the reduced EP grads (== the masters'
        # .grad) against the allreduced reference
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

    # ---- buffer/reduce-slot reuse: identical steps without a weight update
    # must reproduce step 1 exactly ----
    for _ in range(10):
        moe.zero_grad()
        x.grad = None
        y2 = moe(x)
        (y2.float() * gout).sum().backward()
        g_gate2, _, _ = moe.reduce_expert_grads()
    step_diff = (g_gate2 - g_gate_first).abs().max().item()
    print(f"[rank {rank}] step replay dw_gate diff {step_diff:.3e}")

    # ---- timed train steps: fwd + bwd + grad reduce + AdamW on the fp32
    # masters + bf16 refresh of the compute copies ----
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
            torch.equal(w16[lo:hi], wm.to(torch.bfloat16))
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

# torchrun --nproc_per_node=8 moe_moon_ep.py
if __name__ == '__main__':
    main()
