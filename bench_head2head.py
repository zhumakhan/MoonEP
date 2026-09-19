"""Head-to-head: vanilla NCCL all_to_all EP (moe_ep.EPMoE) vs MoonEP
dispatch/combine EP (moe_moon_ep.MoonEPMoE), same shapes, same weights, same
input, same routing math.

Both are checked against each other and against a dense no-communication
reference before any timing is reported, then timed for forward-only and
full training step (fwd + bwd + expert-grad reduce for MoonEP).

Env:
    MOE_S / MOE_K / MOE_E / MOE_H / MOE_HI   shape overrides
    IMBALANCED=1   zero the router weight -> every token picks experts 0..K-1,
                   i.e. all tokens land on rank 0's shard (worst-case skew)
    ITERS          timed iterations (default 30)
    SKIP_CHECK=1   skip the correctness pass (for large shapes)

Run:
    torchrun --nproc_per_node=8 bench_head2head.py
"""
import os
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F

from moe_ep import EPMoE
from moe_moon_ep import MoonEPMoE


def env_int(name, default):
    return int(os.getenv(name, default))


def sync_weights(moon, ep):
    """Copy MoonEP's authoritative weights into the vanilla-EP model.

    MoonEP holds a composite [E+B, ...] pool and physically owns rows
    [rank*epn, (rank+1)*epn); EPMoE holds exactly that slice as [epn, ...].
    """
    lo, hi = moon.rank * moon.epn, (moon.rank + 1) * moon.epn
    with torch.no_grad():
        ep.router.weight.copy_(moon.router.weight)
        ep.w_gate.copy_(moon.w_gate[lo:hi])
        ep.w_up.copy_(moon.w_up[lo:hi])
        ep.w_down.copy_(moon.w_down[lo:hi])


def dense_reference(moon, x, gout):
    """Per-expert loop over the global weight pool, plain autograd, no comms.
    Returns (out, dx, drouter). Expert-weight grads are checked separately."""
    E, K = moon.E, moon.K
    xr = x.detach().clone().requires_grad_()
    rw = moon.router.weight.detach().clone().requires_grad_()
    wg, wu, wd = (moon.w_gate.detach(), moon.w_up.detach(), moon.w_down.detach())

    logits = (xr @ rw.T).float()
    w, idx = torch.topk(logits, k=K, dim=-1)
    w = F.softmax(w, dim=-1)
    flat_w, flat_e = w.flatten(), idx.flatten()
    flat_t = torch.arange(x.size(0), device=x.device).repeat_interleave(K)

    out = torch.zeros(x.size(0), moon.H, dtype=torch.float32, device=x.device)
    for e in range(E):
        sel = flat_e == e
        if not sel.any():
            continue
        t = flat_t[sel]
        y = (F.silu(xr[t] @ wg[e]) * (xr[t] @ wu[e])) @ wd[e]
        z = (y.float() * flat_w[sel, None]).to(x.dtype)
        out = out.index_add(0, t, z.float())
    out = out.to(x.dtype)
    (out.float() * gout).sum().backward()
    return out, xr.grad, rw.grad


def rel(a, b):
    a, b = a.float(), b.float()
    return ((a - b).abs().max() / b.abs().max().clamp_min(1e-12)).item()


def timeit(step, iters, warmup=5):
    for _ in range(warmup):
        step()
    torch.cuda.synchronize()
    dist.barrier()
    t0 = time.perf_counter()
    for _ in range(iters):
        step()
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    ms = (t1 - t0) / iters * 1e3
    t = torch.tensor([ms], device='cuda')
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return t.item()          # slowest rank sets the pace for a collective


def main():
    local_rank = env_int('LOCAL_RANK', 0)
    dist.init_process_group(backend='nccl', device_id=local_rank)
    torch.cuda.set_device(local_rank)
    rank, R = dist.get_rank(), dist.get_world_size()
    dev = f'cuda:{local_rank}'

    S = env_int('MOE_S', 4096)
    K = env_int('MOE_K', 8)
    E = env_int('MOE_E', 64)
    H = env_int('MOE_H', 2048)
    Hi = env_int('MOE_HI', 4096)
    iters = env_int('ITERS', 30)
    imbalanced = os.getenv('IMBALANCED', '0') == '1'
    do_check = os.getenv('SKIP_CHECK', '0') != '1'

    if rank == 0:
        print(f"config: R={R} S={S} K={K} E={E} H={H} Hi={Hi} "
              f"epn={E // R} routing={'IMBALANCED' if imbalanced else 'balanced'} "
              f"iters={iters}", flush=True)

    moon = MoonEPMoE(E, K, H, Hi, S).to(dev)
    ep = EPMoE(E, K, H, Hi).to(dev, torch.bfloat16)
    if imbalanced:
        # every logit ties -> topk picks experts 0..K-1 for every token
        with torch.no_grad():
            moon.router.weight.zero_()
    sync_weights(moon, ep)

    g = torch.Generator(device=dev).manual_seed(42 + rank)
    x = torch.randn(S, H, dtype=torch.bfloat16, device=dev, generator=g)
    gout = torch.randn(S, H, dtype=torch.float32, device=dev, generator=g)

    # ---------------- correctness ----------------
    if do_check:
        lo, hi = rank * moon.epn, (rank + 1) * moon.epn

        x_m = x.clone().requires_grad_()
        y_m = moon(x_m)
        (y_m.float() * gout).sum().backward()
        gm_gate, gm_up, gm_down = moon.reduce_expert_grads()

        x_e = x.clone().requires_grad_()
        y_e = ep(x_e)
        (y_e.float() * gout).sum().backward()

        y_ref, dx_ref, drouter_ref = dense_reference(moon, x, gout)

        rows = [
            ("forward",  rel(y_m, y_ref),               rel(y_e, y_ref)),
            ("dx",       rel(x_m.grad, dx_ref),         rel(x_e.grad, dx_ref)),
            ("drouter",  rel(moon.router.weight.grad, drouter_ref),
                         rel(ep.router.weight.grad, drouter_ref)),
            # both EP paths hold only this rank's owned experts after reduce
            ("dw_gate",  0.0, rel(ep.w_gate.grad, gm_gate)),
            ("dw_up",    0.0, rel(ep.w_up.grad, gm_up)),
            ("dw_down",  0.0, rel(ep.w_down.grad, gm_down)),
        ]
        moon_vs_ep = rel(y_m, y_e)
        if rank == 0:
            print("\ncorrectness (max rel err vs dense reference; dw_* is ep vs moonep)")
            print(f"  {'tensor':10s} {'moonep':>12s} {'vanilla ep':>12s}")
            for name, a, b in rows:
                a_s = "  (ref)" if name.startswith("dw") else f"{a:12.3e}"
                print(f"  {name:10s} {a_s:>12s} {b:12.3e}")
            print(f"  moonep vs vanilla-ep forward: {moon_vs_ep:.3e}")

        del y_m, y_e, y_ref, dx_ref, drouter_ref, gm_gate, gm_up, gm_down
        del x_m, x_e
        moon.zero_grad(set_to_none=True)
        ep.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()

    # ---------------- timing ----------------
    xt = x.clone().requires_grad_()

    def moon_fwd():
        with torch.no_grad():
            moon(x)

    def ep_fwd():
        with torch.no_grad():
            ep(x)

    def moon_step():
        moon.zero_grad(set_to_none=True)
        xt.grad = None
        y = moon(xt)
        (y.float() * gout).sum().backward()
        moon.reduce_expert_grads()

    def ep_step():
        ep.zero_grad(set_to_none=True)
        xt.grad = None
        y = ep(xt)
        (y.float() * gout).sum().backward()

    results = {}
    results['moonep fwd'] = timeit(moon_fwd, iters)
    results['vanilla-ep fwd'] = timeit(ep_fwd, iters)
    results['moonep fwd+bwd'] = timeit(moon_step, iters)
    results['vanilla-ep fwd+bwd'] = timeit(ep_step, iters)

    if rank == 0:
        print("\ntiming (ms/iter, slowest rank)")
        for label in ('fwd', 'fwd+bwd'):
            m = results[f'moonep {label}']
            e = results[f'vanilla-ep {label}']
            print(f"  {label:8s}  moonep {m:9.3f}   vanilla-ep {e:9.3f}   "
                  f"speedup {e / m:5.2f}x")
        peak = torch.cuda.max_memory_allocated() / (1 << 30)
        print(f"\npeak allocated (rank 0, both models resident): {peak:.1f} GiB")

    torch.cuda.synchronize()
    dist.barrier()
    moon.destroy()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
