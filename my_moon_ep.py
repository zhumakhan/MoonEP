'''
Run:
    torchrun --nproc-per-node=8 my_moon_ep.py

NUM_LAYERS (default 2) sets the depth of the timed model.
'''

import math
import os
import time
from typing import Any

import torch, torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from transformer_engine.pytorch.optimizers import FusedAdam
from moonep import Buffer, ExpertPools

# torch 2.14 renamed ``all_gather_into_tensor`` to ``all_gather_single``.
_all_gather_single = getattr(dist, "all_gather_single", None) or dist.all_gather_into_tensor

_PROJECTIONS = ('gate', 'up', 'down')


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


class MoonEPShared:
    """One MoonEP Buffer and one ExpertPools for every MoE layer of a model.

    Every layer dispatches and combines through the same Buffer: each dispatch
    returns a fresh plan, so a layer's saved plan stays valid while later
    layers run. Every layer also borrows the same pools while it computes, so
    MoonEP's memory is paid once, not per layer. The pools hold one layer's
    experts at a time; ``stage_and_push`` remembers whose and re-stages only
    when that changed. Construction and ``destroy`` are collective over
    ``group``.
    """

    def __init__(self, E, K, H, Hi, S, group=None, num_sms=32):
        self.E, self.K, self.H, self.Hi, self.S = E, K, H, Hi, S
        self.group = group
        self.rank = dist.get_rank(group)
        self.R = dist.get_world_size(group)
        self.epn = E // self.R
        self.lo, self.hi = self.rank * self.epn, (self.rank + 1) * self.epn

        self.buffer = Buffer(S, H, K, E, self.R, num_sms=num_sms, group=group)
        # ExpertPools calls the per-expert dims (out, in); its kernels do not
        # care, and the layers store [in, out] (y = x @ w).
        self.pools = ExpertPools(
            self.buffer, group, shapes={'gate': (H, Hi), 'up': (H, Hi), 'down': (Hi, H)}
        )
        self._pushed = None  # the plan whose copies the pools hold

    def stage_and_push(self, plan, masters):
        """Make the pools hold the experts of the layer call that made
        ``plan``: cast its fp32 masters (gate, up, down) into the staging rows
        and push the plan's copies into the prefetch slots. A no-op if no
        other plan was pushed since (e.g. the last layer's backward).

        Collective when it pushes. Every rank runs the layers in the same
        order, so every rank makes the same choice. The push has no entry
        barrier: the dispatch before every call here orders it after all
        ranks' reads of the previous layer's slots.
        """
        if self._pushed is plan:
            return
        for name, w in zip(_PROJECTIONS, masters):
            self.pools.staging(name).copy_(w)  # fp32 master -> bf16 rows [0, epn)
        self.pools.push(plan)
        self._pushed = plan

    def destroy(self):
        self._pushed = None
        self.pools.destroy()
        self.buffer.destroy()


class _ExpertFFN(torch.autograd.Function):
    """SwiGLU experts over the [2*epn] compute view, plus the weight side of
    dispatch bwd.

    Forward stages the layer's experts and pushes the plan's copies, then runs
    the grouped GEMMs over ``pools.weights(n)``: rows [0, epn) are this rank's
    experts and rows [epn, 2*epn) its prefetch slots, exactly the 2*epn groups
    of ``cu_seqlens``. A group with no tokens this step costs no FLOPs and its
    dW is zero.

    Backward first makes the shared pools hold this layer's experts again
    (later layers pushed theirs since), then writes dW of all 2*epn rows into
    ``pools.grads(n)``, reduces the slot rows into their owners' local rows
    (``pools.reduce``, collective) and returns this rank's rows as the
    masters' grads. Those are ordinary autograd grads: nothing aliases
    ``.grad``, and gradient accumulation just works. ``grads(n)`` is shared by
    all layers: every backward overwrites all of its rows and clones the local
    ones out before the next layer's backward. The combine and dispatch
    between two layers' backwards order that overwrite after every rank's
    reduce reads of the slots.
    """

    @staticmethod
    def forward(ctx, h, offs, w_gate, w_up, w_down, ep, plan):
        ep.stage_and_push(plan, (w_gate, w_up, w_down))
        pools = ep.pools
        gate = torch._grouped_mm(h, pools.weights('gate'), offs=offs)
        up = torch._grouped_mm(h, pools.weights('up'), offs=offs)
        y = torch._grouped_mm(F.silu(gate) * up, pools.weights('down'), offs=offs)
        # The pool views stay off save_for_backward: push writes them outside
        # autograd, and the staging copies of later layers bump the version
        # counter that all views of one mapping share.
        ctx.save_for_backward(h, offs, gate, up, w_gate, w_up, w_down)
        ctx.ep, ctx.plan = ep, plan
        return y

    @staticmethod
    def backward(ctx, grad_y):
        h, offs, gate, up, *masters = ctx.saved_tensors
        ep = ctx.ep
        ep.stage_and_push(ctx.plan, masters)
        pools = ep.pools
        grad_y = grad_y.contiguous()

        def gmm_t(a, name):  # a @ W[name]^T, per group
            return torch._grouped_mm(a, pools.weights(name).transpose(-2, -1), offs=offs)

        def dw(a, b, name):  # dW[name] = a^T @ b per group, into the fp32 pool
            pools.grads(name).copy_(torch._grouped_mm(a.t(), b, offs=offs))

        silu_gate = F.silu(gate)
        dw(silu_gate * up, grad_y, 'down')
        grad_act = gmm_t(grad_y, 'down')
        grad_up = grad_act * silu_gate
        grad_gate = torch.ops.aten.silu_backward(grad_act * up, gate)
        dw(h, grad_gate, 'gate')
        dw(h, grad_up, 'up')
        grad_h = gmm_t(grad_gate, 'gate') + gmm_t(grad_up, 'up')

        # The slot rows are other ranks' experts: add every rank's slot grads
        # of this rank's experts into local_grads(n) = grads(n)[:epn].
        pools.reduce(ctx.plan)
        grad_ws = [pools.local_grads(name).clone() for name in _PROJECTIONS]
        return grad_h, None, *grad_ws, None, None


class _AllReduceGrad(torch.autograd.Function):
    """Identity forward; backward sums the grad over the EP group.

    The router is replicated across the EP group but every rank routes its
    own tokens, so each rank's grad is a partial sum (the loss is a sum over
    tokens, like the expert grads). Summing it in the backward gives every
    replica the same grad, so they take the same optimizer step and stay
    bit-identical.
    """

    @staticmethod
    def forward(ctx, w, group):
        ctx.group = group
        return w.view_as(w)

    @staticmethod
    def backward(ctx, grad):
        grad = grad.contiguous()
        dist.all_reduce(grad, op=dist.ReduceOp.SUM, group=ctx.group)
        return grad, None


class MoonEPMoe(nn.Module):
    """Top-K MoE layer with grouped GEMM experts, expert-parallel via MoonEP.

    The parameters are fp32 masters: the replicated router and this rank's
    epn = E/R experts per projection. Their grads are ordinary autograd grads,
    so any optimizer and ``zero_grad()`` work as usual. Each forward casts the
    experts to bf16 for the GEMMs. Gating stays fp32.

    MoonEP memory comes from ``ep``, shared by every layer: one Buffer and
    one ExpertPools, per projection
      - ``weights(n)`` [2*epn, in, out] bf16: the computing layer's experts,
        then this rank's prefetch slots (the rows ``cu_seqlens`` indexes);
      - ``grads(n)`` [2*epn, in, out] fp32: dW of the same rows.
    Layers run one at a time (two layers' expert computations cannot overlap),
    and every rank runs them in the same order.
    """

    def __init__(self, ep: MoonEPShared, seed=0):
        super().__init__()
        self.ep = ep
        E, H, Hi, epn = ep.E, ep.H, ep.Hi, ep.epn
        dev = torch.device('cuda', torch.cuda.current_device())

        # ----- router: fp32, replicated (same seed on every rank, own
        # generator). Gating stays fp32 end to end: bf16 logits flip top-k
        # choices on near-ties, which destabilizes MoE training.
        router_w = torch.empty(E, H, dtype=torch.float32)
        nn.init.kaiming_uniform_(  # nn.Linear's default init
            router_w, a=math.sqrt(5), generator=torch.Generator().manual_seed(seed)
        )
        self.router_w = nn.Parameter(router_w.to(dev))

        # ----- experts: fp32 masters of this rank's own rows
        g = torch.Generator(device=dev).manual_seed(100 + seed * ep.R + ep.rank)

        def master(*shape):
            return nn.Parameter(
                torch.empty(*shape, dtype=torch.float32, device=dev)
                .normal_(0.0, 0.02, generator=g)
            )

        self.w_gate = master(epn, H, Hi)
        self.w_up = master(epn, H, Hi)
        self.w_down = master(epn, Hi, H)

    def route(self, x):
        E, K = self.ep.E, self.ep.K
        router_w = _AllReduceGrad.apply(self.router_w, self.ep.group)
        logits = F.linear(x.float(), router_w)  # fp32 gating: GEMM, top-k and softmax
        weights, idx = torch.topk(logits, k=K, dim=-1)
        weights = F.softmax(weights, dim=-1)
        topk = idx.to(torch.int32)
        tpe = torch.bincount(topk.flatten(), minlength=E).to(torch.int32)
        return weights.contiguous(), topk.contiguous(), tpe

    def forward(self, x, total_num_tokens=None):
        """[s, H] -> [s, H] for any 1 <= s <= S. S is only the Buffer's
        capacity: the comm kernels loop over s, no padding tokens are made.
        Ranks may pass different s when every rank passes ``total_num_tokens``
        (the sum of s over the EP group).
        """
        ep = self.ep
        weights, topk, tpe = self.route(x)

        # dispatch: scatter tokens to their experts' home/copy ranks over
        # NVLink (autograd-aware: backward is combine with the saved plan)
        holder = {}
        h_nvs, w_nvs, cu_seqlens = _MoonEpDispatch.apply(
            x, weights, topk, tpe, ep.buffer, holder, total_num_tokens
        )
        plan = holder['plan']

        # expert FFN straight over the dispatched layout: cu_seqlens [2*epn] is
        # already _grouped_mm's offs format. Padding rows are zero-filled by
        # dispatch, so their FFN output is exactly zero. No host sync:
        # _grouped_mm only touches rows inside the cu_seqlens groups, and
        # cu_seqlens[-1] <= plan.nvs_s keeps offs inside the [plan.nvs_s, H]
        # view. Rows past cu_seqlens[-1] are left uninitialized; no dst slot
        # references them, so nothing reads them.
        y = _ExpertFFN.apply(
            h_nvs, cu_seqlens, self.w_gate, self.w_up, self.w_down, ep, plan
        )

        # scale each slot by its routing weight. Padding and tail slots carry
        # stale weight values (never written for this step); nan_to_num keeps
        # 0-row * garbage from minting NaNs that a GEMM backward would spread
        w_used = torch.nan_to_num(w_nvs)
        z = (y.float() * w_used[:, None]).to(torch.bfloat16)

        # combine: gather every token's K expert outputs back to its source
        # rank and sum -> [s, H] (backward: re-dispatch with the saved plan)
        return _MoonEPCombine.apply(z, ep.buffer, plan)


def _dense_reference_errors(layer, x_in, out):
    """Relative errors (max abs error / max abs reference) of one layer's
    (output, dx, router grad, expert grads) against a dense per-expert
    reference fed the same input ``x_in`` and output grad ``out.grad``. The
    reference runs the same math on every rank's experts (all-gathered, so it
    needs no MoonEP memory). Collective over the EP group.
    """
    ep = layer.ep
    s, dev = x_in.shape[0], x_in.device

    # dense reference: same routing (fp32 router), same rounding points
    wr = {}
    for name in _PROJECTIONS:
        w = getattr(layer, f'w_{name}').detach().to(torch.bfloat16)
        full = torch.empty(ep.E, *w.shape[1:], dtype=w.dtype, device=dev)
        _all_gather_single(full, w, group=ep.group)
        wr[name] = full.requires_grad_()
    xr = x_in.detach().clone().requires_grad_()
    rw = layer.router_w.detach().clone().requires_grad_()
    logits = F.linear(xr.float(), rw)
    w, idx = torch.topk(logits, k=ep.K, dim=-1)
    w = F.softmax(w, dim=-1)
    flat_w, flat_e = w.flatten(), idx.flatten()
    flat_t = torch.arange(s, device=dev).repeat_interleave(ep.K)
    ref = torch.zeros(s, ep.H, dtype=torch.float32, device=dev)
    for e in range(ep.E):
        sel = flat_e == e
        if not bool(sel.any()):
            continue
        t = flat_t[sel]
        y = (F.silu(xr[t] @ wr['gate'][e]) * (xr[t] @ wr['up'][e])) @ wr['down'][e]
        z = (y.float() * flat_w[sel, None]).to(torch.bfloat16)
        ref = ref.index_add(0, t, z.float())
    ref = ref.to(torch.bfloat16)
    ref.backward(out.grad)
    # sum every rank's contributions in fp32, like reduce_grad does
    ref_grads = [rw.grad] + [wr[n].grad.float() for n in _PROJECTIONS]
    for t in ref_grads:
        dist.all_reduce(t, group=ep.group)
    ref_router, ref_gate, ref_up, ref_down = ref_grads

    def rel(a, b):
        return ((a.float() - b.float()).abs().max() / b.float().abs().max().clamp(min=1e-6)).item()

    lo, hi = ep.lo, ep.hi
    return torch.tensor([
        rel(out, ref),
        rel(x_in.grad, xr.grad),
        rel(layer.router_w.grad, ref_router),
        max(rel(layer.w_gate.grad, ref_gate[lo:hi]), rel(layer.w_up.grad, ref_up[lo:hi]),
            rel(layer.w_down.grad, ref_down[lo:hi])),
    ], device=dev)


def check_against_dense_reference(layers, s, total_num_tokens=None, seed=0, tol=3e-2):
    """One fwd + bwd at s tokens on this rank through ``layers`` (a stack
    sharing one MoonEPShared), then every layer against a dense reference.

    Each reference gets its layer's actual input and output grad, so bf16
    differences do not compound across layers (they could flip a later
    layer's top-k on near-ties). Every layer but the last re-stages and
    re-pushes in its backward, so a wrong expert or slot shows up in its dx
    and expert grads.

    Returns the worst relative error over the layers and the EP group of
    (output, dx, router grad, expert grads), and whether all are within
    ``tol``. Both sides round to bf16 at the same points, but the reference
    accumulates each token's K slot grads in bf16, so dx differs by up to
    ~1-2% at bf16.
    """
    ep = layers[0].ep
    dev = torch.device('cuda', torch.cuda.current_device())
    gen = torch.Generator(device=dev).manual_seed(1000 * seed + ep.rank)
    x = torch.randn(s, ep.H, dtype=torch.bfloat16, device=dev, generator=gen).requires_grad_()
    gout = torch.randn(s, ep.H, dtype=torch.float32, device=dev, generator=gen)

    layers.zero_grad()
    ins, outs = [], []
    h = x
    for layer in layers:
        ins.append(h)
        h = layer(h, total_num_tokens=total_num_tokens)
        h.retain_grad()  # the next layer's dx and this layer's output grad
        outs.append(h)
    (h.float() * gout).sum().backward()

    errs = torch.zeros(4, device=dev)
    for layer, x_in, out in zip(layers, ins, outs):
        errs = torch.maximum(errs, _dense_reference_errors(layer, x_in, out))
    dist.all_reduce(errs, op=dist.ReduceOp.MAX, group=ep.group)
    errs = errs.tolist()
    return errs, all(e <= tol for e in errs)


def main():
    local_rank = int(os.getenv('LOCAL_RANK'))
    dist.init_process_group(backend='nccl', device_id=torch.device('cuda', local_rank))
    rank = dist.get_rank()
    R = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    dev = f'cuda:{local_rank}'

    # ---- correctness: a small 3-layer stack on one shared Buffer and pools
    # against a dense reference, at the full capacity, partial s and per-rank
    # s. Half of every router's rows are zeroed so the routing is skewed and
    # the prefetch / grad-reduce paths run.
    S_chk = 512
    ep_chk = MoonEPShared(E=8 * R, K=8, H=1024, Hi=1024, S=S_chk)
    chk = nn.ModuleList(MoonEPMoe(ep_chk, seed=l) for l in range(3))
    with torch.no_grad():
        for layer in chk:
            layer.router_w[: ep_chk.E // 2].zero_()
    per_rank = [S_chk, 37, 1, 300, 64, 5, 128, 256]
    s_r = per_rank[rank % len(per_rank)]
    total = sum(per_rank[r % len(per_rank)] for r in range(R))
    for label, s, tot in (('s=S', S_chk, None), ('s=131', 131, None), ('s=1', 1, None),
                          (f'per-rank s (total={total})', s_r, total)):
        errs, ok = check_against_dense_reference(chk, s, tot, seed=s)
        if rank == 0:
            print(f"[check {len(chk)} layers, {label}] {'PASS' if ok else 'FAIL'} max rel err: "
                  f"out {errs[0]:.1e}, dx {errs[1]:.1e}, router grad {errs[2]:.1e}, "
                  f"expert grads {errs[3]:.1e}", flush=True)
    del chk
    ep_chk.destroy()

    # ---- training-step timing: NUM_LAYERS layers on one shared Buffer and pools
    S, K, E, H, Hi = 1024 * 4, 16, 128, 4096, 4096 * 2
    num_layers = int(os.getenv('NUM_LAYERS', '2'))
    ep = MoonEPShared(E, K, H, Hi, S)
    model = nn.ModuleList(MoonEPMoe(ep, seed=l) for l in range(num_layers))

    # skew the routing by zeroing part of the (fp32, replicated) routers
    with torch.no_grad():
        for layer in model:
            layer.router_w[: E // 2].zero_()

    # opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.0)
    opt = FusedAdam(model.parameters(), lr=1e-4, betas=(0.9, 0.999), weight_decay=0.9, eps=1e-8)
    

    g = torch.Generator(device=dev).manual_seed(42 + rank)
    x = torch.randn(S, H, dtype=torch.bfloat16, device=dev, generator=g).requires_grad_(True)
    gout = torch.randn(S, H, dtype=torch.float32, device=dev, generator=g)
    
    ep_cpu_group = dist.new_group(dist.get_process_group_ranks(dist.group.WORLD), backend='gloo')
    def train_step():
        opt.zero_grad()
        x.grad = None
        h = x
        seq_len = torch.tensor([h.size(0)])
        dist.all_reduce(seq_len, op=dist.ReduceOp.SUM, group=ep_cpu_group)
        total_num_tokens = int(seq_len)
        
        for layer in model:
            h = layer(h, total_num_tokens=total_num_tokens)
        (h.float() * gout).sum().backward()
        opt.step()

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
    print(f"[rank {rank}] {ms:.3f} ms/step fwd+bwd+AdamW, {num_layers} layers "
          f"(slowest rank: {t.item():.3f} ms)")

    # every replicated router must be bit-identical on every rank
    with torch.no_grad():
        router_in_sync = True
        for layer in model:
            router_ref = layer.router_w.detach().clone()
            dist.broadcast(router_ref, src=0)
            router_in_sync &= torch.equal(layer.router_w, router_ref)

    print(f"[rank {rank}] routers identical across ranks: {router_in_sync}")
    torch.cuda.synchronize()
    dist.barrier()
    ep.destroy()
    dist.destroy_process_group()


# torchrun --nproc-per-node=8 my_moon_ep.py
if __name__ == '__main__':
    main()
