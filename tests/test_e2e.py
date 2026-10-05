"""
End-to-end smoke test for the public Buffer dispatch/combine API.

Verifies sync/async dispatch, separate prefetch, combine behavior, and the
public plan-reuse dispatch path, at the Buffer capacity S, at partial step
sizes s < S (same s on every rank) and with a different s on every rank.

Run with:
    torchrun --nproc_per_node=8 -m pytest tests/test_e2e.py
"""

import torch
import torch.distributed as dist

from moonep.buffer import create_nvl_dist_tensor, pad_dim0_for_alignment
from moonep import Buffer, MoonEPCommPlan
from tests.kernel_test_utils import (
    assert_ulp_all_ranks,
    clone_dedup_plan_fields,
    expected_nvs_s,
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


def weight_base(experts, H, Hp, offset, dev):
    expert = experts.to(dtype=torch.float32, device=dev).view(-1, 1, 1)
    row = torch.arange(H, dtype=torch.float32, device=dev).view(1, H, 1)
    col = torch.arange(Hp, dtype=torch.float32, device=dev).view(1, 1, Hp)
    return (offset + expert * 17.0 + row * 0.125 + col * 0.0078125).to(
        torch.bfloat16
    )


def make_dist_pool(rank, R, epn, H, Hp, dtype):
    padded_epn = pad_dim0_for_alignment([epn, H, Hp], dtype)
    assert padded_epn == epn, (
        f"public pool chunk [epn={epn}, H={H}, H'={Hp}] must be VMM aligned"
    )
    return create_nvl_dist_tensor([epn, H, Hp], dtype, rank, R).view(
        R, epn, H, Hp
    )


def make_prefetch_args(rank, R, epn, H, Hp):
    dev = "cuda"
    local_ids = torch.arange(rank * epn, (rank + 1) * epn, device=dev)
    args = {}
    for name, offset in (("gate", 0.0), ("up", 1.0), ("down", 2.0)):
        args[f"local_{name}_weight"] = weight_base(
            local_ids, H, Hp, offset, dev
        ).contiguous()
        pool = make_dist_pool(rank, R, epn, H, Hp, torch.bfloat16)
        pool[rank].zero_()
        args[f"{name}_prefetch_buffer"] = pool
    return args


def assert_prefetched(rank, H, Hp, prefetch_args, experts_to_copy):
    assert experts_to_copy.dim() == 2
    for b in range(experts_to_copy.shape[1]):
        expert = int(experts_to_copy[rank, b].item())
        if expert < 0:
            continue
        expert_id = torch.tensor([expert], device="cuda")
        for name, offset in (("gate", 0.0), ("up", 1.0), ("down", 2.0)):
            pool = prefetch_args[f"{name}_prefetch_buffer"]
            expected = weight_base(expert_id, H, Hp, offset, pool.device)[0]
            assert torch.equal(pool[rank, b], expected), (
                f"{name} prefetch buffer {b} does not match expert {expert}"
            )


def fill_prefetch_slots(rank, prefetch_args, value):
    for name in ("gate", "up", "down"):
        prefetch_args[f"{name}_prefetch_buffer"][rank].fill_(value)


def assert_prefetch_slots_equal(rank, prefetch_args, value):
    for name in ("gate", "up", "down"):
        slots = prefetch_args[f"{name}_prefetch_buffer"][rank]
        assert torch.equal(slots, torch.full_like(slots, value)), (
            f"{name} prefetch slots changed without prefetch_weight"
        )


def assert_dedup_plan_semantic_equal(actual, expected):
    errors = dedup_plan_semantic_errors("dedup plan", actual, expected)
    assert not errors, "; ".join(errors[:5])


def grad_base(rank, epn, H, Hp, offset, dev):
    expert = (
        rank * epn + torch.arange(epn, dtype=torch.float32, device=dev)
    ).view(epn, 1, 1)
    row = torch.arange(H, dtype=torch.float32, device=dev).view(1, H, 1)
    col = torch.arange(Hp, dtype=torch.float32, device=dev).view(1, 1, Hp)
    return offset + expert * 17.0 + row * 0.125 + col * 0.0078125


def reduce_base(R, epn, H, Hp, offset, dev):
    src = torch.arange(R, dtype=torch.float32, device=dev).view(R, 1, 1, 1)
    slot = torch.arange(epn, dtype=torch.float32, device=dev).view(1, epn, 1, 1)
    row = torch.arange(H, dtype=torch.float32, device=dev).view(1, 1, H, 1)
    col = torch.arange(Hp, dtype=torch.float32, device=dev).view(1, 1, 1, Hp)
    return offset + (src + 1.0) * 101.0 + slot * 11.0 + row * 0.03125 + col * 0.00390625


def make_grad_reduce_args(rank, R, epn, H, Hp, offsets):
    dev = "cuda"
    args = {}
    for i, name in enumerate(("gate", "up", "down")):
        args[f"local_{name}_grad"] = grad_base(
            rank, epn, H, Hp, offsets[i], dev
        ).contiguous()
        pool = make_dist_pool(rank, R, epn, H, Hp, torch.float32)
        pool[rank].copy_(reduce_base(R, epn, H, Hp, offsets[i + 3], dev)[rank])
        args[f"{name}_reduce_buffer"] = pool
    torch.cuda.synchronize()
    dist.barrier(device_ids=[local_device_index()])
    return args


def expected_local_grad(rank, R, epn, H, Hp, grad_offset, reduce_offset, experts_to_copy):
    dev = "cuda"
    expected = grad_base(rank, epn, H, Hp, grad_offset, dev)
    reduce_vals = reduce_base(R, experts_to_copy.shape[1], H, Hp, reduce_offset, dev)
    local_start = rank * epn
    local_end = local_start + epn
    for src_rank in range(R):
        for b in range(experts_to_copy.shape[1]):
            expert = int(experts_to_copy[src_rank, b].item())
            if local_start <= expert < local_end:
                expected[expert - local_start].add_(reduce_vals[src_rank, b])
    return expected


def assert_grad_reduced(rank, R, epn, H, Hp, experts_to_copy, args, offsets):
    pairs = [
        ("gate", args["local_gate_grad"], args["gate_reduce_buffer"], offsets[0], offsets[3]),
        ("up", args["local_up_grad"], args["up_reduce_buffer"], offsets[1], offsets[4]),
        ("down", args["local_down_grad"], args["down_reduce_buffer"], offsets[2], offsets[5]),
    ]
    for name, local_grad, reduce_buffer, grad_offset, reduce_offset in pairs:
        expected = expected_local_grad(
            rank, R, epn, H, Hp, grad_offset, reduce_offset, experts_to_copy
        )
        assert torch.allclose(
            local_grad, expected, rtol=0.0, atol=1e-5
        ), f"{name} local grad reduce mismatch"

        original_reduce = reduce_base(
            R, experts_to_copy.shape[1], H, Hp, reduce_offset, reduce_buffer.device
        )
        for b in range(experts_to_copy.shape[1]):
            expert = int(experts_to_copy[rank, b].item())
            if expert >= 0:
                assert torch.equal(
                    reduce_buffer[rank, b],
                    torch.zeros_like(reduce_buffer[rank, b]),
                ), f"{name} consumed reduce slot ({rank}, {b}) was not cleared"
            else:
                assert torch.equal(
                    reduce_buffer[rank, b],
                    original_reduce[rank, b],
                ), f"{name} unused reduce slot ({rank}, {b}) changed"


def assert_raises_assertion(expected_substr, fn):
    """``fn()`` must raise AssertionError; ``expected_substr=None`` accepts
    any message."""
    try:
        fn()
    except AssertionError as exc:
        assert expected_substr is None or expected_substr in str(exc), (
            f"expected assertion containing {expected_substr!r}, got {exc!r}"
        )
    else:
        raise AssertionError(f"expected AssertionError containing {expected_substr!r}")


def test_e2e(dist_env):
    rank, R = dist_env
    # epn=8 keeps both bf16 weight and fp32 grad chunks VMM-granularity aligned.
    S, H, K, E = 256, 1024, 4, R * 8
    epn = E // R
    Hp = 128
    num_sms = 32
    buffer = Buffer(S, H, K, E, R, num_sms=num_sms)
    sync_prefetch_args = make_prefetch_args(rank, R, epn, H, Hp)
    async_prefetch_args = make_prefetch_args(rank, R, epn, H, Hp)
    reuse_prefetch_args = make_prefetch_args(rank, R, epn, H, Hp)
    no_prefetch_args = make_prefetch_args(rank, R, epn, H, Hp)

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
    assert_prefetched(rank, H, Hp, sync_prefetch_args, plan_snapshot.experts_to_copy)

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
    assert_prefetched(rank, H, Hp, async_prefetch_args, plan_a.experts_to_copy)

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
    assert_prefetched(rank, H, Hp, reuse_prefetch_args, plan_snapshot.experts_to_copy)

    # --- Public plan-reuse path without prefetch: same hidden scatter, but
    # prefetch slots must remain untouched. This matches backward redispatch.
    sentinel = 7.0
    fill_prefetch_slots(rank, no_prefetch_args, sentinel)
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
    assert_prefetch_slots_equal(rank, no_prefetch_args, sentinel)

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
    grad_args = make_grad_reduce_args(rank, R, epn, H, Hp, grad_offsets)
    out_grad_sync, _, _ = buffer.combine(
        plan=plan_snapshot,
        hidden_nvsh=h_for_combine,
    )
    buffer.reduce_grad(plan=plan_snapshot, **grad_args)
    torch.cuda.synchronize()
    assert torch.equal(out_sync_snap, out_grad_sync), "sync grad_reduce combine output mismatch"
    assert_grad_reduced(rank, R, epn, H, Hp, plan_snapshot.experts_to_copy, grad_args, grad_offsets)

    # Re-run dispatch to re-populate NVL for async combine
    h_for_combine, _, _, _ = buffer.dispatch(
        hidden, weights, topk, tpe,
    )
    async_grad_offsets = (11000.0, 12000.0, 13000.0, 14000.0, 15000.0, 16000.0)
    async_grad_args = make_grad_reduce_args(rank, R, epn, H, Hp, async_grad_offsets)
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
        rank, R, epn, H, Hp, plan_snapshot.experts_to_copy, async_grad_args, async_grad_offsets
    )

    if rank == 0:
        print("[test_e2e] PASS: public API sync/async, separate prefetch, and plan reuse match.")

    buffer.destroy()


# Per-rank step sizes for the differing-s test, indexed by rank (mod 8) and
# clamped into [1, S]; deterministic on every rank so the step total needs no
# collective. Mirrors tests/test_planning.py.
# 38 (not 37) makes total*K indivisible by R=8 for the e2e shape (K=4), so
# the +1 remainder slots of the cap rule are exercised.
PER_RANK_TOKENS_PATTERN = (1, 5, 38, None, "half", 3, 64, 2)


def per_rank_tokens(S, rank):
    entry = PER_RANK_TOKENS_PATTERN[rank % len(PER_RANK_TOKENS_PATTERN)]
    if entry is None:
        s = S
    elif entry == "half":
        s = S // 2
    else:
        s = int(entry)
    return max(1, min(int(s), S))


def assert_zero_fill_rows_zero(plan, hidden_nvsh, route_weights_nvs, tag):
    """Every segment-padding row listed in plan.zero_fill_ranges is zero in
    the returned shard (hidden and, when given, the weight slot)."""
    ranges = plan.zero_fill_ranges.cpu()
    rows = 0
    for g in range(ranges.shape[0]):
        start, n_pad = int(ranges[g, 0].item()), int(ranges[g, 1].item())
        if n_pad <= 0:
            continue
        rows += n_pad
        assert start + n_pad <= int(hidden_nvsh.shape[0]), \
            f"{tag}: zero-fill range [{start}, {start + n_pad}) past the returned rows"
        assert not bool(hidden_nvsh[start:start + n_pad].any()), \
            f"{tag}: hidden padding rows [{start}, {start + n_pad}) are not zero"
        if route_weights_nvs is not None:
            assert not bool(route_weights_nvs[start:start + n_pad].view(torch.int32).any()), \
                f"{tag}: weight padding rows [{start}, {start + n_pad}) are not zero"
    return rows


def assert_identity_round_trip(out, hidden, K, rank, R, tag):
    """combine(dispatch(x)) with an identity FFN returns K*x: the K copies are
    summed in fp32 and rounded once, except that duplicate slots are
    pre-reduced in bf16 once in the combine prologue (one more rounding)."""
    expected = (hidden.float() * K).to(torch.bfloat16)
    assert tuple(out.shape) == tuple(expected.shape), f"{tag}: shape {tuple(out.shape)}"
    assert_ulp_all_ranks(f"{tag} combine(dispatch(x)) == K*x", out, expected, rank, R, max_ulps=2)


def _check_dispatch_views(buffer, plan, h, w, cu, s, K, H, tag, total_num_tokens=None):
    """Shape and bound checks on a fresh dispatch's outputs at s tokens."""
    ctx = buffer._require_ctx()
    R = int(ctx["R"])
    assert isinstance(plan, MoonEPCommPlan)
    assert plan.num_tokens == s and plan.N == s * K, f"{tag}: plan sized {plan.N}"
    assert tuple(plan.dst.shape) == (s * K,), f"{tag}: dst shape {tuple(plan.dst.shape)}"
    want_nvs_s = expected_nvs_s(ctx, R * s if total_num_tokens is None else total_num_tokens)
    assert plan.nvs_s == want_nvs_s, f"{tag}: plan.nvs_s={plan.nvs_s}, expected {want_nvs_s}"
    assert plan.nvs_s <= int(ctx["NvS"])
    assert tuple(h.shape) == (plan.nvs_s, H), f"{tag}: hidden_nvsh {tuple(h.shape)}"
    assert tuple(w.shape) == (plan.nvs_s,), f"{tag}: route_weights_nvs {tuple(w.shape)}"
    assert w.dtype == torch.float32 and h.dtype == torch.bfloat16
    assert tuple(cu.shape) == (2 * (int(ctx["E"]) // R),)
    total = int(cu[-1].item())
    assert total <= plan.nvs_s, f"{tag}: cu_seqlens[-1]={total} > plan.nvs_s={plan.nvs_s}"
    return total


def _e2e_partial_tokens(buffer, rank, R, S, H, K, E, Hp, s):
    """dispatch -> combine round trip with s <= S tokens per rank. The Buffer's
    S is only a capacity: the kernels loop over s, no padding tokens exist and
    the returned shard views are ``plan.nvs_s`` rows (s*K + token_padding_extra
    rounded up to token_padding). Prefetch and grad reduce run with the plan of
    this step. Returns what the caller needs to replay this step's plan later."""
    ctx = buffer._require_ctx()
    epn = E // R
    tag = f"s={s}"
    hidden, weights, topk, tpe = make_inputs(rank, s, H, K, E, seed=100 + s)

    # --- fresh planning at s tokens ---
    h_sync, w_sync, cu_sync, plan = buffer.dispatch(hidden, weights, topk, tpe)
    torch.cuda.synchronize()
    total = _check_dispatch_views(buffer, plan, h_sync, w_sync, cu_sync, s, K, H, tag)
    pad_extra = int(ctx["token_padding_extra"])
    # every rank receives exactly s*K real tokens plus per-group padding
    assert s * K <= total <= s * K + pad_extra, \
        f"{tag}: padded total {total} outside [{s * K}, {s * K + pad_extra}]"
    pad_rows = assert_zero_fill_rows_zero(plan, h_sync, w_sync, tag)
    assert total - pad_rows == s * K, f"{tag}: {total - pad_rows} real slots, expected {s * K}"
    h_snap, w_snap = h_sync.clone(), w_sync.clone()
    plan_snapshot = plan.clone()

    # --- async path gives the same plan and shard ---
    h_a, w_a, cu_a, plan_a, ev = buffer.dispatch(
        hidden, weights, topk, tpe, async_finish=True,
    )
    ev.wait(torch.cuda.current_stream())
    torch.cuda.synchronize()
    assert torch.equal(cu_sync, cu_a), f"{tag}: cu_seqlens mismatch"
    assert torch.equal(plan.dst, plan_a.dst), f"{tag}: dst mismatch"
    assert plan_a.nvs_s == plan.nvs_s
    assert torch.equal(h_snap[:total], h_a[:total]), f"{tag}: dispatch hidden mismatch"
    assert torch.equal(w_snap[:total], w_a[:total]), f"{tag}: dispatch weights mismatch"
    assert_dedup_plan_semantic_equal(plan_a, plan_snapshot)

    # --- prefetch driven by the partial-s plan (owners push into the pools) ---
    prefetch_args = make_prefetch_args(rank, R, epn, H, Hp)
    # make_prefetch_args zero-fills this rank's pool slice, but peers push into
    # that slice and prefetch has no entry barrier: the zero-fill must finish
    # on every rank before any rank pushes, or it erases a peer's push.
    torch.cuda.synchronize()
    dist.barrier(device_ids=[local_device_index()])
    buffer.prefetch_weight(plan=plan, **prefetch_args)
    torch.cuda.synchronize()
    assert_prefetched(rank, H, Hp, prefetch_args, plan.experts_to_copy)

    # --- identity "FFN": combine sums each token's K dispatched copies ---
    h_c, w_c, _, _ = buffer.dispatch(hidden, weights, topk, tpe)
    out, gathered, _ = buffer.combine(
        plan=plan, hidden_nvsh=h_c, route_weights_nvs=w_c,
    )
    torch.cuda.synchronize()
    assert tuple(out.shape) == (s, H), f"{tag}: combine output {tuple(out.shape)}"
    assert tuple(gathered.shape) == (s, K), f"{tag}: gathered weights {tuple(gathered.shape)}"
    assert_identity_round_trip(out, hidden, K, rank, R, tag)
    assert torch.equal(gathered, weights), f"{tag}: route weight gather mismatch"
    out_snap = out.clone()

    # --- async combine equals the sync result ---
    h_c, w_c, _, _ = buffer.dispatch(hidden, weights, topk, tpe)
    out_async, gathered_async, combine_ev = buffer.combine(
        plan=plan, hidden_nvsh=h_c, route_weights_nvs=w_c, async_finish=True,
    )
    combine_ev.wait(torch.cuda.current_stream())
    torch.cuda.synchronize()
    assert torch.equal(out_async, out_snap), f"{tag}: async combine hidden mismatch"
    assert torch.equal(gathered_async, weights), f"{tag}: async combine weights mismatch"

    # --- zero_copy round trip: dispatch returns prefix views of the NVL
    # shard, combine consumes them in place; bit-exact vs the copies ---
    h_zc, w_zc, cu_zc, plan_zc = buffer.dispatch(
        hidden, weights, topk, tpe, zero_copy=True, router_weights_zero_copy=True,
    )
    torch.cuda.synchronize()
    assert h_zc.data_ptr() == ctx["hidden_buf_local"].data_ptr(), \
        f"{tag}: dispatch(zero_copy=True) must return the NVL shard view"
    assert w_zc.data_ptr() == ctx["weights_buf_local"].data_ptr(), \
        f"{tag}: dispatch(router_weights_zero_copy=True) must return the NVL weights view"
    assert tuple(h_zc.shape) == (plan_zc.nvs_s, H) and tuple(w_zc.shape) == (plan_zc.nvs_s,)
    assert torch.equal(cu_zc, cu_sync) and torch.equal(plan_zc.dst, plan.dst)
    assert torch.equal(h_zc[:total], h_snap[:total]), f"{tag}: zero_copy hidden view mismatch"
    assert torch.equal(w_zc[:total], w_snap[:total]), f"{tag}: zero_copy weights view mismatch"
    out_zc, gathered_zc, _ = buffer.combine(
        plan=plan_zc, hidden_nvsh=h_zc, route_weights_nvs=w_zc,
        zero_copy=True, router_weights_zero_copy=True,
    )
    torch.cuda.synchronize()
    assert torch.equal(out_zc, out_snap), f"{tag}: zero_copy combine hidden mismatch"
    assert torch.equal(gathered_zc, weights), f"{tag}: zero_copy combine weights mismatch"
    assert_raises_assertion(
        "alias",
        lambda: buffer.combine(plan=plan_zc, hidden_nvsh=h_snap, zero_copy=True),
    )

    # --- grad reduce driven by the partial-s plan (dispatch bwd, weight side) ---
    grad_offsets = (1000.0 + s, 2000.0 + s, 3000.0 + s, 4000.0 + s, 5000.0 + s, 6000.0 + s)
    grad_args = make_grad_reduce_args(rank, R, epn, H, Hp, grad_offsets)
    buffer.reduce_grad(plan=plan, **grad_args)
    torch.cuda.synchronize()
    assert_grad_reduced(rank, R, epn, H, Hp, plan.experts_to_copy, grad_args, grad_offsets)

    # --- plan reuse (the backward paths) carries s along ---
    dedup_before = clone_dedup_plan_fields(plan)
    h_r, w_r, cu_r, plan_r = buffer.dispatch(hidden, plan=plan)
    torch.cuda.synchronize()
    assert cu_r is None and w_r is None and plan_r is plan
    assert tuple(h_r.shape) == (plan.nvs_s, H), f"{tag}: plan-reuse hidden {tuple(h_r.shape)}"
    assert torch.equal(h_r[:total], h_snap[:total]), f"{tag}: plan-reuse hidden mismatch"
    assert dedup_plan_fields_equal(plan, dedup_before), \
        f"{tag}: plan-reuse dispatch must not rebuild dedup structures"
    # weights path on plan reuse
    h_rw, w_rw, cu_rw, plan_rw = buffer.dispatch(hidden, weights, plan=plan)
    torch.cuda.synchronize()
    assert cu_rw is None and plan_rw is plan
    assert tuple(w_rw.shape) == (plan.nvs_s,), f"{tag}: plan-reuse weights {tuple(w_rw.shape)}"
    assert torch.equal(h_rw[:total], h_snap[:total]), f"{tag}: plan-reuse (weights) hidden mismatch"
    assert torch.equal(w_rw[:total], w_snap[:total]), f"{tag}: plan-reuse weights mismatch"
    out_rw, gathered_rw, _ = buffer.combine(plan=plan, hidden_nvsh=h_rw, route_weights_nvs=w_rw)
    torch.cuda.synchronize()
    assert torch.equal(out_rw, out_snap) and torch.equal(gathered_rw, weights), \
        f"{tag}: combine after plan-reuse dispatch mismatch"

    # --- rejections ---
    if s > 1:
        assert_raises_assertion(
            "tokens", lambda: buffer.dispatch(hidden[: s - 1], plan=plan),
        )
        # topk with a different token count than hidden
        assert_raises_assertion(
            "topk_experts_sk",
            lambda: buffer.dispatch(hidden[: s - 1], weights[: s - 1], topk, tpe),
        )
    assert_raises_assertion(
        "capacity", lambda: buffer.dispatch(hidden[:0], weights[:0], topk[:0], tpe),
    )
    # combine needs NvS or plan.nvs_s rows; s*K rows is neither
    assert plan.nvs_s != s * K and int(ctx["NvS"]) != s * K
    assert_raises_assertion(
        None,
        lambda: buffer.combine(plan=plan, hidden_nvsh=h_snap[: s * K].contiguous()),
    )

    return {
        "s": s,
        "hidden": hidden,
        "weights": weights,
        "plan": plan,
        "total": total,
        "h_snap": h_snap,
        "w_snap": w_snap,
        "out_snap": out_snap,
    }


def _replay_step(buffer, step, tag):
    """Re-dispatch a saved step's hidden with its saved plan after other
    steps (at other s) ran on the same Buffer, and combine with it."""
    plan, total = step["plan"], step["total"]
    h_r, w_r, _, plan_r = buffer.dispatch(step["hidden"], step["weights"], plan=plan)
    torch.cuda.synchronize()
    assert plan_r is plan
    assert torch.equal(h_r[:total], step["h_snap"][:total]), f"{tag}: replayed hidden mismatch"
    assert torch.equal(w_r[:total], step["w_snap"][:total]), f"{tag}: replayed weights mismatch"
    out, gathered, _ = buffer.combine(plan=plan, hidden_nvsh=h_r, route_weights_nvs=w_r)
    torch.cuda.synchronize()
    assert torch.equal(out, step["out_snap"]), f"{tag}: replayed combine mismatch"
    assert torch.equal(gathered, step["weights"]), f"{tag}: replayed weight gather mismatch"


def test_e2e_partial_tokens(dist_env):
    rank, R = dist_env
    # epn=8 keeps the bf16 prefetch and fp32 reduce pool chunks VMM aligned.
    S, H, K, E = 256, 1024, 4, R * 8
    Hp = 128
    buffer = Buffer(S, H, K, E, R, num_sms=32)
    # s < num_sms leaves trailing blocks of the per-token loops empty; s = 1 is
    # the smallest step. Then full capacity on the same Buffer.
    steps = [
        _e2e_partial_tokens(buffer, rank, R, S, H, K, E, Hp, s)
        for s in (S // 2 + 3, 5, 1, S)
    ]
    # Plans saved from earlier steps replay after steps at other s ran on the
    # same Buffer (the backward passes of interleaved micro-batches).
    for step in steps:
        _replay_step(buffer, step, f"replay s={step['s']}")
    # a step larger than the capacity is rejected up front
    hidden, weights, topk, tpe = make_inputs(rank, S + 1, H, K, E, seed=7)
    assert_raises_assertion(
        "capacity", lambda: buffer.dispatch(hidden, weights, topk, tpe),
    )
    if rank == 0:
        print("[test_e2e_partial_tokens] PASS: s <= S dispatch/combine round trips match.")
    buffer.destroy()


def test_e2e_per_rank_tokens(dist_env):
    """Every rank dispatches a different s_r in the same step. The host
    passes the step total; the planner sizes each rank's receive cap as
    ceil_div(total*K, R) rounded per rank, so plan.nvs_s no longer follows
    this rank's own s_r."""
    rank, R = dist_env
    S, H, K, E = 256, 1024, 4, R * 4
    buffer = Buffer(S, H, K, E, R, num_sms=32)
    ctx = buffer._require_ctx()
    s_r = per_rank_tokens(S, rank)
    total = sum(per_rank_tokens(S, r) for r in range(R))
    assert s_r <= total <= R * S
    tag = f"s_r={s_r}, total={total}"
    hidden, weights, topk, tpe = make_inputs(rank, s_r, H, K, E, seed=300 + rank)

    h, w, cu, plan = buffer.dispatch(hidden, weights, topk, tpe, total_num_tokens=total)
    torch.cuda.synchronize()
    cu_total = _check_dispatch_views(
        buffer, plan, h, w, cu, s_r, K, H, tag, total_num_tokens=total
    )
    assert plan.nvs_s == expected_nvs_s(ctx, total)
    assert cu_total <= plan.nvs_s
    pad_rows = assert_zero_fill_rows_zero(plan, h, w, tag)
    cap_r = (total * K) // R + (1 if rank < (total * K) % R else 0)
    assert cu_total - pad_rows <= cap_r, \
        f"{tag}: rank receives {cu_total - pad_rows} real slots > cap {cap_r}"
    # every real slot of the step lands on exactly one rank, so the per-rank
    # counts sum to total*K (with R=8, K=4 the pattern gives total=497,
    # total*K=1988, remainder 4: ranks 0-3 hold one slot more than ranks 4-7)
    real_slots = torch.tensor([cu_total - pad_rows], dtype=torch.int64, device="cuda")
    all_slots = [torch.zeros_like(real_slots) for _ in range(R)]
    dist.all_gather(all_slots, real_slots)
    slot_counts = [int(t.item()) for t in all_slots]
    assert sum(slot_counts) == total * K, \
        f"{tag}: real slots across ranks {slot_counts} do not sum to total*K={total * K}"
    assert all(c <= cap_r_ for c, cap_r_ in zip(
        slot_counts, [(total * K) // R + (1 if r < (total * K) % R else 0) for r in range(R)])), \
        f"{tag}: some rank exceeds its cap: {slot_counts}"
    h_snap, w_snap = h.clone(), w.clone()

    # identity round trip and exact weight gather, [s_r, H] / [s_r, K] outputs
    h_c, w_c, _, _ = buffer.dispatch(hidden, weights, topk, tpe, total_num_tokens=total)
    out, gathered, _ = buffer.combine(plan=plan, hidden_nvsh=h_c, route_weights_nvs=w_c)
    torch.cuda.synchronize()
    assert tuple(out.shape) == (s_r, H), f"{tag}: combine output {tuple(out.shape)}"
    assert tuple(gathered.shape) == (s_r, K), f"{tag}: gathered weights {tuple(gathered.shape)}"
    assert_identity_round_trip(out, hidden, K, rank, R, tag)
    assert torch.equal(gathered, weights), f"{tag}: route weight gather mismatch"

    # plan reuse ignores total_num_tokens and carries s_r and nvs_s along
    h_r, w_r, cu_r, plan_r = buffer.dispatch(hidden, weights, plan=plan)
    torch.cuda.synchronize()
    assert cu_r is None and plan_r is plan
    assert tuple(h_r.shape) == (plan.nvs_s, H) and tuple(w_r.shape) == (plan.nvs_s,)
    assert torch.equal(h_r[:cu_total], h_snap[:cu_total]), f"{tag}: plan-reuse hidden mismatch"
    assert torch.equal(w_r[:cu_total], w_snap[:cu_total]), f"{tag}: plan-reuse weights mismatch"
    out_r, gathered_r, _ = buffer.combine(plan=plan, hidden_nvsh=h_r, route_weights_nvs=w_r)
    torch.cuda.synchronize()
    assert torch.equal(out_r, out) and torch.equal(gathered_r, weights), \
        f"{tag}: combine after plan-reuse dispatch mismatch"

    if rank == 0:
        print("[test_e2e_per_rank_tokens] PASS: per-rank s dispatch/combine round trips match.")
    buffer.destroy()


if __name__ == "__main__":
    env = setup()
    test_e2e(env)
    test_e2e_partial_tokens(env)
    test_e2e_per_rank_tokens(env)
    dist.destroy_process_group()
