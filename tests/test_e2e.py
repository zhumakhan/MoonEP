"""
End-to-end smoke test for the public Buffer dispatch/combine API.

Verifies sync/async dispatch, separate prefetch, combine behavior, and the
public plan-reuse dispatch path.

Run with:
    torchrun --nproc_per_node=8 -m pytest tests/test_e2e.py
"""

import torch
import torch.distributed as dist

from moonep.buffer import create_nvl_single_owner_tensor, pad_dim0_for_alignment
from moonep import Buffer, MoonEPCommPlan
from tests.kernel_test_utils import (
    clone_dedup_plan_fields,
    local_device_index,
    dedup_plan_fields_equal,
    dedup_plan_semantic_errors,
)


def setup():
    """Process-group setup for direct ``torchrun tests/test_e2e.py`` runs. Under
    pytest the tests take the session-scoped ``dist_env`` fixture (conftest)
    instead: re-initializing the default group inside a torchrun worker reuses
    the agent store and hands the peers a stale NCCL bootstrap address."""
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(local_device_index())
    return rank, dist.get_world_size()


def make_inputs(rank, S, H, K, E, seed=0):
    dev = "cuda"
    g = torch.Generator(device=dev).manual_seed(seed + rank)
    hidden = torch.randn(S, H, dtype=torch.bfloat16, device=dev, generator=g)
    weights = torch.rand(S, K, dtype=torch.float32, device=dev, generator=g)
    topk = torch.randint(0, E, (S, K), dtype=torch.int32, device=dev, generator=g)
    tpe = torch.bincount(topk.flatten(), minlength=E).to(torch.int32)
    return hidden, weights, topk, tpe


def make_remote_expert(rank, R, E, H, Hp, owner_offset=1):
    dev = "cuda"
    padded_E = pad_dim0_for_alignment([E, H, Hp], torch.bfloat16)
    owners = []
    for owner in range(R):
        mapped = create_nvl_single_owner_tensor(
            [padded_E, H, Hp],
            torch.bfloat16,
            owner_rank=owner,
            local_rank=rank,
        )
        if rank == owner:
            g = torch.Generator(device=dev).manual_seed(5000 + owner)
            mapped[:E].copy_(
                torch.randn(E, H, Hp, dtype=torch.bfloat16, device=dev, generator=g)
            )
            if padded_E > E:
                mapped[E:].zero_()
        torch.cuda.synchronize()
        dist.barrier(device_ids=[local_device_index()])
        owners.append(mapped[:E])

    remote_owner = (rank + owner_offset) % R
    if remote_owner == rank:
        remote_owner = (rank + 1) % R
    return owners[remote_owner]


def make_full_weight(rank, remote_expert, B, source_offset):
    E, H, Hp = remote_expert.shape
    dev = "cuda"
    full_weight = torch.empty(E + B, H, Hp, dtype=torch.bfloat16, device=dev)
    full_weight[:E].copy_(remote_expert)
    if source_offset:
        full_weight[:E].add_(source_offset)
    if B:
        full_weight[E:].zero_()
    return full_weight


def make_prefetch_args(rank, remote_expert, B):
    return {
        "full_gate_weight": make_full_weight(rank, remote_expert, B, 0.0),
        "full_up_weight": make_full_weight(rank, remote_expert, B, 1.0),
        "full_down_weight": make_full_weight(rank, remote_expert, B, 2.0),
    }


def assert_prefetched(prefetch_args, experts_to_copy):
    assert experts_to_copy.dim() == 1
    B = experts_to_copy.numel()
    for b in range(experts_to_copy.numel()):
        expert = int(experts_to_copy[b].item())
        if expert < 0:
            continue
        for name, full_weight in prefetch_args.items():
            E = full_weight.shape[0] - B
            assert torch.equal(full_weight[E + b], full_weight[expert]), (
                f"{name} prefetch buffer {b} does not match expert {expert}"
            )


def fill_prefetch_slots(prefetch_args, B, value):
    for full_weight in prefetch_args.values():
        full_weight[-B:].fill_(value)


def assert_prefetch_slots_equal(prefetch_args, B, value):
    for name, full_weight in prefetch_args.items():
        expected = torch.full_like(full_weight[-B:], value)
        assert torch.equal(full_weight[-B:], expected), (
            f"{name} prefetch slots changed without prefetch_weight"
        )


def assert_dedup_plan_semantic_equal(actual, expected):
    errors = dedup_plan_semantic_errors("dedup plan", actual, expected)
    assert not errors, "; ".join(errors[:5])


def grad_base(E, H, Hp, offset, dev):
    expert = torch.arange(E, dtype=torch.float32, device=dev).view(E, 1, 1)
    row = torch.arange(H, dtype=torch.float32, device=dev).view(1, H, 1)
    col = torch.arange(Hp, dtype=torch.float32, device=dev).view(1, 1, Hp)
    return offset + expert * 17.0 + row * 0.125 + col * 0.0078125


def reduce_base(R, B, H, Hp, offset, dev):
    src = torch.arange(R, dtype=torch.float32, device=dev).view(R, 1, 1, 1)
    slot = torch.arange(B, dtype=torch.float32, device=dev).view(1, B, 1, 1)
    row = torch.arange(H, dtype=torch.float32, device=dev).view(1, 1, H, 1)
    col = torch.arange(Hp, dtype=torch.float32, device=dev).view(1, 1, 1, Hp)
    return offset + (src + 1.0) * 101.0 + slot * 11.0 + row * 0.03125 + col * 0.00390625


def make_grad_reduce_args(rank, R, E, B, H, Hp, offsets):
    dev = "cuda"
    full_E = E + B
    return {
        "full_gate_grad": grad_base(full_E, H, Hp, offsets[0], dev).contiguous(),
        "full_up_grad": grad_base(full_E, H, Hp, offsets[1], dev).contiguous(),
        "full_down_grad": grad_base(full_E, H, Hp, offsets[2], dev).contiguous(),
        "gate_reduce_buffer": reduce_base(R, B, H, Hp, offsets[3], dev).contiguous(),
        "up_reduce_buffer": reduce_base(R, B, H, Hp, offsets[4], dev).contiguous(),
        "down_reduce_buffer": reduce_base(R, B, H, Hp, offsets[5], dev).contiguous(),
    }


def expected_local_grad(rank, R, E, H, Hp, full_offset, reduce_offset, experts_to_copy):
    dev = "cuda"
    expected = grad_base(E, H, Hp, full_offset, dev)
    reduce_vals = reduce_base(R, experts_to_copy.shape[1], H, Hp, reduce_offset, dev)
    for src_rank in range(R):
        for b in range(experts_to_copy.shape[1]):
            expert = int(experts_to_copy[src_rank, b].item())
            if expert >= 0:
                expected[expert].add_(reduce_vals[src_rank, b])
    epn = E // R
    return expected[rank * epn:(rank + 1) * epn].contiguous()


def assert_grad_reduced(rank, R, E, H, Hp, experts_to_copy, args, offsets):
    epn = E // R
    local_start = rank * epn
    local_end = local_start + epn
    pairs = [
        ("gate", args["full_gate_grad"], args["gate_reduce_buffer"], offsets[0], offsets[3]),
        ("up", args["full_up_grad"], args["up_reduce_buffer"], offsets[1], offsets[4]),
        ("down", args["full_down_grad"], args["down_reduce_buffer"], offsets[2], offsets[5]),
    ]
    for name, full_grad, reduce_buffer, full_offset, reduce_offset in pairs:
        expected = expected_local_grad(
            rank, R, E, H, Hp, full_offset, reduce_offset, experts_to_copy
        )
        assert torch.allclose(
            full_grad[local_start:local_end], expected, rtol=0.0, atol=1e-5
        ), f"{name} local grad reduce mismatch"
        assert torch.equal(
            full_grad[:local_start],
            grad_base(E, H, Hp, full_offset, full_grad.device)[:local_start],
        ), f"{name} non-local prefix grad changed"
        assert torch.equal(
            full_grad[local_end:],
            grad_base(E + experts_to_copy.shape[1], H, Hp, full_offset, full_grad.device)[local_end:],
        ), f"{name} non-local suffix grad changed"

        original_reduce = reduce_base(
            R, experts_to_copy.shape[1], H, Hp, reduce_offset, reduce_buffer.device
        )
        for src_rank in range(R):
            for b in range(experts_to_copy.shape[1]):
                expert = int(experts_to_copy[src_rank, b].item())
                if expert >= 0 and src_rank == rank:
                    assert torch.equal(
                        reduce_buffer[src_rank, b],
                        torch.zeros_like(reduce_buffer[src_rank, b]),
                    ), f"{name} consumed reduce slot ({src_rank}, {b}) was not cleared"
                else:
                    assert torch.equal(
                        reduce_buffer[src_rank, b],
                        original_reduce[src_rank, b],
                    ), f"{name} non-local reduce slot ({src_rank}, {b}) changed"


def assert_raises_assertion(expected_substr, fn):
    try:
        fn()
    except AssertionError as exc:
        assert expected_substr in str(exc), (
            f"expected assertion containing {expected_substr!r}, got {exc!r}"
        )
    else:
        raise AssertionError(f"expected AssertionError containing {expected_substr!r}")


def test_e2e(dist_env):
    rank, R = dist_env
    S, H, K, E = 256, 1024, 4, R * 4
    B = 2
    Hp = 128
    num_sms = 32
    buffer = Buffer(S, H, K, E, R, B=B, num_sms=num_sms)
    remote_expert = make_remote_expert(rank, R, E, H, Hp)
    sync_prefetch_args = make_prefetch_args(rank, remote_expert, B)
    async_prefetch_args = make_prefetch_args(rank, remote_expert, B)
    reuse_prefetch_args = make_prefetch_args(rank, remote_expert, B)
    no_prefetch_args = make_prefetch_args(rank, remote_expert, B)

    hidden, weights, topk, tpe = make_inputs(rank, S, H, K, E)

    # --- Sync reference ---
    (h_sync, w_sync, cu_sync, plan_sync) = buffer.dispatch(
        hidden, weights, topk, tpe,
    )
    h_sync_cpu = h_sync.clone()
    w_sync_cpu = w_sync.clone() if w_sync is not None else None
    cu_sync_cpu = cu_sync.clone()
    # Snapshot the plan's tensors so a later dispatch can't mutate them.
    assert isinstance(plan_sync, MoonEPCommPlan)
    plan_snapshot = plan_sync.clone()
    buffer.prefetch_weight(plan=plan_snapshot, **sync_prefetch_args)
    torch.cuda.synchronize()
    assert_prefetched(sync_prefetch_args, plan_snapshot.experts_to_copy[rank])

    # --- Async variant (fresh inputs to avoid NVL aliasing with sync run) ---
    hidden2, weights2, topk2, tpe2 = make_inputs(rank, S, H, K, E)
    assert torch.equal(hidden, hidden2)
    assert torch.equal(weights, weights2)

    (h_a, w_a, cu_a, plan_a, _dispatch_event) = buffer.dispatch(
        hidden2, weights2, topk2, tpe2, async_finish=True,
    )
    prefetch_event = buffer.prefetch_weight(
        plan=plan_a, async_finish=True, **async_prefetch_args,
    )
    # Caller must explicitly wait before reading.
    prefetch_event.wait(torch.cuda.current_stream())
    h_a_snap = h_a.clone()
    w_a_snap = w_a.clone() if w_a is not None else None
    torch.cuda.synchronize()
    assert_prefetched(async_prefetch_args, plan_a.experts_to_copy[rank])

    # Compare
    assert torch.equal(h_sync_cpu, h_a_snap), "dispatch hidden mismatch"
    assert torch.equal(w_sync_cpu, w_a_snap), "dispatch weights mismatch"
    assert torch.equal(cu_sync_cpu, cu_a), "cu_seqlens mismatch"
    for name in ("dst", "experts_to_copy", "zero_fill_ranges", "remote_stats"):
        assert torch.equal(getattr(plan_snapshot, name), getattr(plan_a, name)), \
            f"plan.{name} mismatch"
    assert_dedup_plan_semantic_equal(plan_a, plan_snapshot)

    # --- Public plan-reuse path: planning is skipped, caller passes the
    # saved plan back. ---
    reuse_dedup_before = clone_dedup_plan_fields(plan_snapshot)
    (h_reuse, w_reuse, cu_reuse, plan_reuse) = buffer.dispatch(
        hidden, plan=plan_snapshot,
    )
    buffer.prefetch_weight(plan=plan_snapshot, **reuse_prefetch_args)
    torch.cuda.synchronize()
    assert torch.equal(h_reuse.clone(), h_sync_cpu), "plan-reuse hidden buffer mismatch"
    assert w_reuse is None, "plan-reuse hidden-only dispatch should not return weights buffer"
    assert cu_reuse is None, "plan-reuse path should skip planning outputs"
    assert plan_reuse is plan_snapshot, "plan-reuse should echo the input plan back"
    assert dedup_plan_fields_equal(plan_snapshot, reuse_dedup_before), \
        "plan-reuse dispatch should not rebuild or mutate dedup structures"
    assert_prefetched(reuse_prefetch_args, plan_snapshot.experts_to_copy[rank])

    # --- Public plan-reuse path without prefetch: same hidden scatter, but
    # prefetch slots must remain untouched. This matches backward redispatch.
    sentinel = 7.0
    fill_prefetch_slots(no_prefetch_args, B, sentinel)
    no_prefetch_dedup_before = clone_dedup_plan_fields(plan_snapshot)
    (h_no_prefetch, w_no_prefetch, cu_no_prefetch, plan_no_prefetch) = buffer.dispatch(
        hidden, plan=plan_snapshot,
    )
    torch.cuda.synchronize()
    assert torch.equal(h_no_prefetch.clone(), h_sync_cpu), "no-prefetch hidden buffer mismatch"
    assert w_no_prefetch is None, "no-prefetch hidden-only dispatch should not return weights buffer"
    assert cu_no_prefetch is None, "no-prefetch plan-reuse path should skip planning outputs"
    assert dedup_plan_fields_equal(plan_snapshot, no_prefetch_dedup_before), \
        "no-prefetch plan-reuse dispatch should not rebuild or mutate dedup structures"
    assert plan_no_prefetch is plan_snapshot, "no-prefetch plan-reuse should echo the input plan back"
    assert_prefetch_slots_equal(no_prefetch_args, B, sentinel)

    (h_no_weights, _, _, _) = buffer.dispatch(
        hidden, plan=plan_snapshot,
    )
    torch.cuda.synchronize()
    assert torch.equal(h_no_weights.clone(), h_sync_cpu), "dispatch without weights hidden mismatch"

    (h_no_prefetch_async, _, _, _, ev_no_prefetch) = buffer.dispatch(
        hidden, plan=plan_snapshot, async_finish=True,
    )
    ev_no_prefetch.wait(torch.cuda.current_stream())
    torch.cuda.synchronize()
    assert torch.equal(
        h_no_prefetch_async.clone(), h_sync_cpu
    ), "async no-prefetch hidden mismatch"

    # --- Combine: compare sync vs async for the symmetric path ---
    # Use sync_dispatch's NVL population as the "expert output" proxy.
    # Re-run sync dispatch first to restore NVL buffer to a known state.
    h_for_combine, w_for_combine, _, _ = buffer.dispatch(
        hidden, weights, topk, tpe,
    )

    out_sync, _, _ = buffer.combine(plan=plan_snapshot, hidden_nvsh=h_for_combine)
    out_sync_snap = out_sync.clone()
    torch.cuda.synchronize()

    h_for_combine, w_for_combine, _, _ = buffer.dispatch(
        hidden, weights, topk, tpe,
    )
    out_with_weights, gathered_weights, _ = buffer.combine(
        plan=plan_snapshot,
        hidden_nvsh=h_for_combine,
        route_weights_nvs=w_for_combine,
    )
    torch.cuda.synchronize()
    assert torch.equal(out_sync_snap, out_with_weights), "combine weight-gather hidden mismatch"
    assert torch.equal(gathered_weights, weights), "combine route_weights_sk gather mismatch"

    # --- zero_copy round-trip: dispatch returns NVL views, the "FFN" is an
    # identity on the shard, combine consumes the views in place. Must match
    # the zero_copy=False result bit-exactly.
    h_zc, w_zc, _, plan_zc = buffer.dispatch(
        hidden, weights, topk, tpe, zero_copy=True, router_weights_zero_copy=True,
    )
    assert h_zc.data_ptr() == buffer._require_ctx()['hidden_buf_local'].data_ptr(), \
        "dispatch(zero_copy=True) must return the NVL shard view"
    assert torch.equal(h_zc, h_for_combine), \
        "zero_copy dispatch hidden view mismatch vs copied tensor"
    assert torch.equal(w_zc, w_for_combine), \
        "zero_copy dispatch weights view mismatch vs copied tensor"
    out_zc, gathered_weights_zc, _ = buffer.combine(
        plan=plan_zc,
        hidden_nvsh=h_zc,
        route_weights_nvs=w_zc,
        zero_copy=True,
        router_weights_zero_copy=True,
    )
    torch.cuda.synchronize()
    assert torch.equal(out_sync_snap, out_zc), "zero_copy combine hidden mismatch"
    assert torch.equal(gathered_weights_zc, weights), \
        "zero_copy combine route_weights_sk gather mismatch"
    assert_raises_assertion(
        "alias",
        lambda: buffer.combine(
            plan=plan_zc,
            hidden_nvsh=h_for_combine,
            zero_copy=True,
        ),
    )

    # --- Combine + sync grad_reduce ---
    h_for_combine, _, _, _ = buffer.dispatch(
        hidden, weights, topk, tpe,
    )
    grad_offsets = (1000.0, 2000.0, 3000.0, 4000.0, 5000.0, 6000.0)
    grad_args = make_grad_reduce_args(rank, R, E, B, H, Hp, grad_offsets)
    out_grad_sync, _, _ = buffer.combine(
        plan=plan_snapshot,
        hidden_nvsh=h_for_combine,
    )
    buffer.reduce_grad(plan=plan_snapshot, **grad_args)
    torch.cuda.synchronize()
    assert torch.equal(out_sync_snap, out_grad_sync), "sync grad_reduce combine output mismatch"
    assert_grad_reduced(rank, R, E, H, Hp, plan_snapshot.experts_to_copy, grad_args, grad_offsets)

    # Re-run dispatch to re-populate NVL for async combine
    h_for_combine, _, _, _ = buffer.dispatch(
        hidden, weights, topk, tpe,
    )
    async_grad_offsets = (11000.0, 12000.0, 13000.0, 14000.0, 15000.0, 16000.0)
    async_grad_args = make_grad_reduce_args(rank, R, E, B, H, Hp, async_grad_offsets)
    out_async, _, _combine_ev = buffer.combine(
        plan=plan_snapshot,
        hidden_nvsh=h_for_combine,
        async_finish=True,
    )
    reduce_ev = buffer.reduce_grad(
        plan=plan_snapshot,
        async_finish=True,
        **async_grad_args,
    )
    reduce_ev.wait(torch.cuda.current_stream())
    torch.cuda.synchronize()

    assert torch.equal(out_sync_snap, out_async), "combine output mismatch"
    assert_grad_reduced(
        rank, R, E, H, Hp, plan_snapshot.experts_to_copy, async_grad_args, async_grad_offsets
    )

    if rank == 0:
        print("[test_e2e] PASS: public API sync/async, separate prefetch, and plan reuse match.")

    buffer.destroy()


def _e2e_partial_tokens(buffer, rank, R, S, H, K, E, s):
    """dispatch -> combine round trip with s <= S tokens per rank. The Buffer's
    S is only a capacity: the kernels loop over s and no padding tokens exist."""
    ctx = buffer._require_ctx()
    hidden, weights, topk, tpe = make_inputs(rank, s, H, K, E, seed=100 + s)

    # --- fresh planning at s tokens ---
    h_sync, w_sync, cu_sync, plan = buffer.dispatch(hidden, weights, topk, tpe)
    torch.cuda.synchronize()
    assert plan.num_tokens == s and plan.N == s * K, f"s={s}: plan sized {plan.N}"
    assert tuple(plan.dst.shape) == (s * K,), f"s={s}: dst shape {tuple(plan.dst.shape)}"
    # every rank receives exactly s*K real tokens plus per-group padding
    total = int(cu_sync[-1].item())
    pad_extra = int(ctx["token_padding_extra"])
    assert s * K <= total <= s * K + pad_extra, \
        f"s={s}: padded total {total} outside [{s * K}, {s * K + pad_extra}]"
    h_snap, w_snap = h_sync[:total].clone(), w_sync[:total].clone()

    # --- async path gives the same plan and shard ---
    h_a, w_a, cu_a, plan_a, ev = buffer.dispatch(
        hidden, weights, topk, tpe, async_finish=True,
    )
    ev.wait(torch.cuda.current_stream())
    torch.cuda.synchronize()
    assert torch.equal(cu_sync, cu_a), f"s={s}: cu_seqlens mismatch"
    assert torch.equal(plan.dst, plan_a.dst), f"s={s}: dst mismatch"
    assert torch.equal(h_snap, h_a[:total]), f"s={s}: dispatch hidden mismatch"
    assert torch.equal(w_snap, w_a[:total]), f"s={s}: dispatch weights mismatch"

    # --- identity "FFN": combine sums each token's K dispatched copies, so
    # the round trip returns K * hidden (up to one bf16 rounding when
    # duplicate slots are pre-reduced in the combine prologue) ---
    h_c, w_c, _, _ = buffer.dispatch(hidden, weights, topk, tpe)
    out, gathered, _ = buffer.combine(
        plan=plan, hidden_nvsh=h_c, route_weights_nvs=w_c,
    )
    torch.cuda.synchronize()
    assert tuple(out.shape) == (s, H), f"s={s}: combine output {tuple(out.shape)}"
    assert tuple(gathered.shape) == (s, K), f"s={s}: gathered weights {tuple(gathered.shape)}"
    expected = hidden.float() * K
    assert torch.allclose(out.float(), expected, rtol=2e-2, atol=1e-2), \
        f"s={s}: combine(dispatch(x)) != K*x, max diff {(out.float() - expected).abs().max().item()}"
    assert torch.equal(gathered, weights), f"s={s}: route weight gather mismatch"

    # --- plan reuse (the backward paths) carries s along ---
    h_r, w_r, cu_r, plan_r = buffer.dispatch(hidden, plan=plan)
    torch.cuda.synchronize()
    assert cu_r is None and w_r is None and plan_r is plan
    assert torch.equal(h_r[:total], h_snap), f"s={s}: plan-reuse hidden mismatch"
    if s > 1:
        assert_raises_assertion(
            "tokens", lambda: buffer.dispatch(hidden[: s - 1], plan=plan),
        )


def test_e2e_partial_tokens(dist_env):
    rank, R = dist_env
    S, H, K, E = 256, 1024, 4, R * 4
    buffer = Buffer(S, H, K, E, R, B=2, num_sms=32)
    # s < num_sms leaves trailing blocks of the per-token loops empty; s = 1 is
    # the smallest step. Then full capacity on the same Buffer.
    for s in (S // 2 + 3, 5, 1, S):
        _e2e_partial_tokens(buffer, rank, R, S, H, K, E, s)
    # a step larger than the capacity is rejected up front
    hidden, weights, topk, tpe = make_inputs(rank, S + 1, H, K, E, seed=7)
    assert_raises_assertion(
        "capacity", lambda: buffer.dispatch(hidden, weights, topk, tpe),
    )
    if rank == 0:
        print("[test_e2e_partial_tokens] PASS: s <= S dispatch/combine round trips match.")
    buffer.destroy()


if __name__ == "__main__":
    env = setup()
    test_e2e(env)
    test_e2e_partial_tokens(env)
    dist.destroy_process_group()
