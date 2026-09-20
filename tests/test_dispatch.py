"""Dispatch kernel correctness tests.

Run with:
    torchrun --nproc_per_node=8 -m pytest -s tests/test_dispatch.py

Every scatter test also runs at partial step sizes ``s < S``: the Buffer's S
is a capacity, each step routes ``s`` tokens (the same ``s`` on every rank in
these tests) and the plan is sized ``s*K``.
"""

from dataclasses import replace

import pytest
import torch

from tests.kernel_test_utils import (
    KernelCase,
    assert_all_ranks,
    assert_dedup_plan_semantic_equal_all_ranks,
    case_params,
    clone_dedup_plan_fields,
    dedup_plan_fields_equal,
    dedup_plan_invariant_errors,
    gather_tensor,
    init_case,
    make_topk,
    partial_token_counts,
)
from tests.planning_reference import launch_planning_torch_reference


DISPATCH_CASES = [
    KernelCase("balanced", S=256, K=8, epn=16, H=128, num_sms=32, B=4),
    KernelCase(
        "tiny_k1_no_padding",
        S=1,
        K=1,
        epn=1,
        H=8,
        num_sms=1,
        token_padding=1,
    ),
    KernelCase(
        "all_local",
        S=64,
        K=4,
        epn=8,
        H=64,
        num_sms=8,
        B=2,
        token_padding=16,
        routing="all_local",
    ),
    KernelCase(
        "all_remote",
        S=64,
        K=4,
        epn=8,
        H=64,
        num_sms=8,
        B=2,
        token_padding=16,
        routing="all_remote",
        min_R=2,
    ),
    KernelCase(
        "duplicate_topk",
        S=32,
        K=4,
        epn=4,
        H=64,
        num_sms=8,
        B=2,
        token_padding=8,
        routing="duplicate_topk",
        min_R=2,
    ),
    KernelCase(
        "duplicate_topk_k32",
        S=8,
        K=32,
        epn=4,
        H=64,
        num_sms=8,
        B=2,
        token_padding=8,
        routing="duplicate_topk",
        min_R=2,
    ),
    KernelCase(
        "single_expert_split",
        S=9,
        K=2,
        epn=4,
        H=64,
        num_sms=8,
        B=1,
        token_padding=8,
        routing="single_expert",
        min_R=2,
    ),
]

# Extra shapes for the partial-s sweep: K=1 (direct scatter, no dedup) and an
# odd K, both with a real capacity to shrink from.
PARTIAL_DISPATCH_CASES = DISPATCH_CASES + [
    KernelCase("k1_partial", S=32, K=1, epn=4, H=64, num_sms=8, B=1, token_padding=8),
    KernelCase(
        "odd_k3_partial",
        S=40,
        K=3,
        epn=8,
        H=64,
        num_sms=8,
        B=2,
        token_padding=16,
    ),
    KernelCase(
        "odd_k5_duplicate_topk_partial",
        S=24,
        K=5,
        epn=4,
        H=64,
        num_sms=8,
        B=2,
        token_padding=8,
        routing="duplicate_topk",
        min_R=2,
    ),
]

ZERO_FILL_CASES = [
    KernelCase(
        "local_padding",
        S=17,
        K=3,
        epn=8,
        H=64,
        num_sms=8,
        B=2,
        token_padding=16,
        routing="all_local",
    ),
    KernelCase(
        "remote_padding",
        S=17,
        K=3,
        epn=8,
        H=64,
        num_sms=8,
        B=3,
        token_padding=16,
        routing="all_remote",
        min_R=2,
    ),
]

LARGE_DISPATCH_CASES = [
    KernelCase(
        "large_hidden_stride_spotcheck",
        S=8192,
        K=16,
        epn=14,
        H=7168,
        num_sms=32,
        B=4,
    )
]

SAVED_PLAN_CASE = KernelCase(
    "saved_plan",
    S=17,
    K=3,
    epn=8,
    H=64,
    num_sms=8,
    B=2,
    token_padding=16,
    routing="all_remote",
    min_R=2,
)


def _step_tokens(case, s):
    s = case.S if s is None else int(s)
    assert 0 < s <= case.S, f"num_tokens {s} outside [1, S={case.S}]"
    return s


def _traceable_hidden(rank, s, H):
    """[s, H] bf16 whose first two i16 lanes encode (token index, source rank)."""
    hidden = torch.zeros(s, H, dtype=torch.bfloat16, device="cuda")
    hidden_i16 = hidden.view(torch.int16)
    s_idx = torch.arange(s, dtype=torch.int32, device="cuda")
    hidden_i16[:, 0] = s_idx.to(torch.int16)
    if H > 1:
        hidden_i16[:, 1] = torch.full((s,), rank, dtype=torch.int16, device="cuda")
    return hidden


def _traceable_weights(rank, S, K, s=None):
    """[s, K] fp32 bit patterns ``rank*S*K + token*K + k``. The stride stays
    the capacity ``S*K`` for every ``s`` so ``_verify_dispatch_by_dst`` decodes
    (rank, token, k) unambiguously from the value alone."""
    s = S if s is None else int(s)
    weights_i32 = (
        torch.arange(s * K, dtype=torch.int32, device="cuda")
        .reshape(s, K)
        .add_(rank * S * K)
    )
    return weights_i32.view(torch.float32)


def _plan_and_dispatch(
    ctx,
    case,
    rank,
    R,
    hidden,
    weights,
    *,
    hidden_user=None,
    weights_user=None,
    s=None,
):
    """Plan ``s`` tokens (default ``case.S``), run the fresh dispatch builder,
    the in-place duplicate expansion and the zero_copy=False style boundary
    copies. Returns ``(dst, plan, hidden_user, weights_user)``; the user
    tensors are full ``[NvS, ...]`` copies of this rank's shard."""
    from moonep.dispatch import launch_dispatch
    from moonep.dispatch_epilogue import launch_dispatch_epilogue
    from moonep.planning import allocate_planning_outputs, launch_planning

    s = _step_tokens(case, s)
    assert int(hidden.shape[0]) == s, f"hidden has {hidden.shape[0]} rows, expected s={s}"
    topk, tpe = make_topk(case, rank, R, s=s)
    plan, _cu = allocate_planning_outputs(ctx, s)
    assert plan.num_tokens == s and plan.N == s * case.K
    launch_planning(ctx, topk.reshape(-1).contiguous(), tpe, plan=plan, cu_seqlens=_cu)
    _ref_dst, _ref_cu, _ref_etc, _ref_stats, _ref_zfr, ref_dedup_plan = (
        launch_planning_torch_reference(ctx, topk, tpe)
    )
    launch_dispatch(ctx, hidden, weights, plan, build_dedup_map=True)
    assert_dedup_plan_semantic_equal_all_ranks(
        f"dispatch dedup plan[s={s}]", plan, ref_dedup_plan, rank, R
    )
    dedup_errors = dedup_plan_invariant_errors(case, ctx, plan)
    assert_all_ranks(
        not dedup_errors,
        rank,
        R,
        f"{case.name} [s={s}] dispatch dedup plan invariants",
        "; ".join(dedup_errors[:5]),
    )
    # In-place duplicate expansion on the NVL shard, then the master-style
    # zero_copy=False boundary copies into user tensors.
    launch_dispatch_epilogue(ctx, plan)
    if hidden_user is None:
        hidden_user = torch.empty_like(ctx["hidden_buf_local"])
    hidden_user.copy_(ctx["hidden_buf_local"])
    if weights is not None:
        if weights_user is None:
            weights_user = torch.empty(
                (int(ctx["NvS"]),),
                dtype=torch.float32,
                device=hidden_user.device,
            )
        weights_user.copy_(ctx["weights_buf_local"].view(torch.float32))
    else:
        weights_user = None
    return plan.dst, plan, hidden_user, weights_user


def _verify_dispatch_by_dst(ctx, case, rank, R, dst, hidden_user,
                            check_weights=True, weights_user=None,
                            max_checks=None, copy_to_cpu=True, s=None):
    """Decode every (src_rank, token, k) slot that lands on this rank from the
    gathered ``dst`` and check the traceable payload at its local offset."""
    s = _step_tokens(case, s)
    nvs_stride = int(ctx["NvS"])
    all_dst = gather_tensor(dst.reshape(s, case.K).contiguous(), R).cpu()

    if copy_to_cpu:
        hidden_i16 = hidden_user.view(torch.int16).cpu()
        weights_i32 = weights_user.view(torch.int32).cpu() if weights_user is not None else None
    else:
        hidden_i16 = hidden_user.view(torch.int16)
        weights_i32 = weights_user.view(torch.int32) if weights_user is not None else None

    errors = []
    checked = 0
    for src_r in range(R):
        for tok in range(s):
            for k in range(case.K):
                dst_val = int(all_dst[src_r, tok, k].item())
                raw_dst = -dst_val - 1 if dst_val < 0 else dst_val
                dest_rank = raw_dst // nvs_stride
                local_off = raw_dst % nvs_stride
                if dest_rank != rank:
                    continue

                expected_s = tok & 0xFFFF
                if expected_s >= 0x8000:
                    expected_s -= 0x10000
                actual_s = int(hidden_i16[local_off, 0].item())
                actual_r = int(hidden_i16[local_off, 1].item()) if case.H > 1 else src_r
                if actual_s != expected_s or actual_r != src_r:
                    errors.append(
                        f"hidden src=({src_r},{tok},{k}) loff={local_off}: "
                        f"expected s/r={expected_s}/{src_r}, got {actual_s}/{actual_r}"
                    )

                if check_weights:
                    if weights_i32 is None:
                        errors.append("weights_user missing while check_weights=True")
                        continue
                    expected_w = src_r * case.S * case.K + tok * case.K + k
                    actual_w = int(weights_i32[local_off].item())
                    if actual_w != expected_w:
                        errors.append(
                            f"weight src=({src_r},{tok},{k}) loff={local_off}: "
                            f"expected {expected_w}, got {actual_w}"
                        )

                checked += 1
                if max_checks is not None and checked >= max_checks:
                    return errors, checked
    return errors, checked


def _zero_row_errors(zero_fill_ranges, hidden_user, weights_user=None):
    """Every row covered by plan.zero_fill_ranges must be zero (hidden and,
    when given, the matching weight slot). The [last_padded_end, NvS) tail is
    deliberately NOT covered (master behavior: undefined, read via
    cu_seqlens)."""
    hidden_cpu = hidden_user.cpu()
    weights_cpu = weights_user.cpu() if weights_user is not None else None
    ranges = zero_fill_ranges.cpu()

    errors = []
    rows = 0
    for g in range(ranges.shape[0]):
        pad_start = int(ranges[g, 0].item())
        n_pad = int(ranges[g, 1].item())
        for j in range(n_pad):
            local_off = pad_start + j
            rows += 1
            if not torch.all(hidden_cpu[local_off] == 0):
                errors.append(f"hidden padding loff={local_off} is nonzero")
            if weights_cpu is not None and float(weights_cpu[local_off].item()) != 0.0:
                value = int(weights_cpu.view(torch.int32)[local_off].item()) & 0xFFFFFFFF
                errors.append(
                    f"weight padding loff={local_off} is 0x{value:08x}"
                )
    return errors, rows


def _check_scatter(ctx, case, rank, R, s, with_weights, label):
    """One traceable dispatch at ``s`` tokens (weights optional), verified
    slot by slot against the gathered ``dst``."""
    hidden = _traceable_hidden(rank, s, case.H)
    weights = _traceable_weights(rank, case.S, case.K, s=s) if with_weights else None

    dst, plan, hidden_user, weights_user = _plan_and_dispatch(
        ctx, case, rank, R, hidden, weights, s=s
    )
    torch.cuda.synchronize()
    assert tuple(dst.shape) == (s * case.K,), f"dst shape {tuple(dst.shape)}"

    errors, checked = _verify_dispatch_by_dst(
        ctx, case, rank, R, dst, hidden_user=hidden_user,
        check_weights=with_weights, weights_user=weights_user, s=s,
    )
    if weights is None and weights_user is not None:
        errors.append("hidden-only dispatch returned a weights buffer")
    assert_all_ranks(
        checked > 0 and not errors,
        rank,
        R,
        f"{case.name} {label}",
        "; ".join(errors[:5]) or f"checked={checked}",
    )
    return plan


@pytest.mark.parametrize("case", case_params(DISPATCH_CASES))
def test_dispatch_scatters_hidden_and_weights_by_dst(dist_env, case):
    rank, R = dist_env
    ctx = init_case(case, R)
    _check_scatter(ctx, case, rank, R, case.S, True, "dispatch scatter")


@pytest.mark.parametrize("with_weights", [True, False], ids=["weights", "hidden_only"])
@pytest.mark.parametrize("case", case_params(PARTIAL_DISPATCH_CASES))
def test_dispatch_partial_tokens_scatters_by_dst(dist_env, case, with_weights):
    """Fresh plan + dispatch at s in {1, S//2, S-1} on one Buffer of capacity
    S, with and without the weights scatter; every slot of every rank's
    ``s*K`` entries is checked (dedup, K=1 and odd-K cases included)."""
    rank, R = dist_env
    if case.S < 2:
        pytest.skip("partial-token dispatch needs S >= 2")
    ctx = init_case(case, R)
    for s in partial_token_counts(case.S):
        label = f"[s={s}/{case.S}] {'weights' if with_weights else 'hidden-only'} scatter"
        _check_scatter(ctx, case, rank, R, s, with_weights, label)


def _check_padding_zero_fill(ctx, case, rank, R, s):
    hidden = torch.randn(s, case.H, dtype=torch.bfloat16, device="cuda")
    weights = torch.rand(s, case.K, dtype=torch.float32, device="cuda")

    # Poison the shard and the user copies so a skipped zero fill is visible.
    ctx["hidden_buf_local"].fill_(7)
    ctx["weights_buf_local"].fill_(0x55555555)
    hidden_user = torch.empty(
        (int(ctx["NvS"]), case.H),
        dtype=torch.bfloat16,
        device="cuda",
    )
    hidden_user.fill_(7)
    weights_user = torch.empty(
        (int(ctx["NvS"]),),
        dtype=torch.float32,
        device="cuda",
    )
    weights_user.view(torch.int32).fill_(0x55555555)
    torch.cuda.synchronize()

    _dst, plan, hidden_user, weights_user = _plan_and_dispatch(
        ctx,
        case,
        rank,
        R,
        hidden,
        weights,
        hidden_user=hidden_user,
        weights_user=weights_user,
        s=s,
    )
    torch.cuda.synchronize()

    errors, rows = _zero_row_errors(plan.zero_fill_ranges, hidden_user, weights_user)
    assert_all_ranks(
        rows > 0 and not errors,
        rank,
        R,
        f"{case.name} [s={s}/{case.S}] dispatch padding zero-fill",
        "; ".join(errors[:5]) or f"padding_rows={rows}",
    )


@pytest.mark.parametrize("case", case_params(ZERO_FILL_CASES))
def test_dispatch_dedup_plan_clears_padding_with_weights(dist_env, case):
    rank, R = dist_env
    ctx = init_case(case, R)
    _check_padding_zero_fill(ctx, case, rank, R, case.S)


@pytest.mark.parametrize("case", case_params(ZERO_FILL_CASES))
def test_dispatch_partial_tokens_clears_padding_with_weights(dist_env, case):
    """Segment padding rows are zero-filled at every partial s too (S=17,
    K=3, token_padding=16 leaves padding in every group at each s)."""
    rank, R = dist_env
    ctx = init_case(case, R)
    for s in partial_token_counts(case.S):
        _check_padding_zero_fill(ctx, case, rank, R, s)


def _check_saved_plan_hidden_only(ctx, case, rank, R, s):
    """Plan A (all_remote) at ``s`` tokens, then an unrelated plan B on the
    same Buffer, then re-dispatch hidden A with the saved plan A
    (``build_dedup_map=False``, no weights): the scatter must match plan A's
    dst, padding stays zero, the weights shard and A's dedup structures are
    untouched."""
    from moonep.dispatch import launch_dispatch
    from moonep.dispatch_epilogue import launch_dispatch_epilogue
    from moonep.planning import allocate_planning_outputs, launch_planning

    hidden_a = _traceable_hidden(rank, s, case.H)
    weights_a = _traceable_weights(rank, case.S, case.K, s=s)
    topk_a, tpe_a = make_topk(case, rank, R, s=s)
    plan_a, _cu = allocate_planning_outputs(ctx, s)
    launch_planning(ctx, topk_a.reshape(-1).contiguous(), tpe_a, plan=plan_a, cu_seqlens=_cu)
    dst_a = plan_a.dst
    launch_dispatch(ctx, hidden_a, weights_a, plan_a, build_dedup_map=True)
    launch_dispatch_epilogue(ctx, plan_a)
    torch.cuda.synchronize()
    dedup_a_snapshot = clone_dedup_plan_fields(plan_a)

    # The interleaved step B runs at a different s where possible, so the
    # saved plan must carry its own token count.
    case_b = replace(case, routing="all_local")
    s_b = case.S if s != case.S else max(1, case.S // 2)
    hidden_b = torch.randn(s_b, case.H, dtype=torch.bfloat16, device="cuda")
    weights_b = torch.rand(s_b, case.K, dtype=torch.float32, device="cuda")
    topk_b, tpe_b = make_topk(case_b, rank, R, s=s_b)
    plan_b, _cu = allocate_planning_outputs(ctx, s_b)
    launch_planning(ctx, topk_b.reshape(-1).contiguous(), tpe_b, plan=plan_b, cu_seqlens=_cu)
    launch_dispatch(ctx, hidden_b, weights_b, plan_b, build_dedup_map=True)
    launch_dispatch_epilogue(ctx, plan_b)
    torch.cuda.synchronize()
    weights_after_b = ctx["weights_buf_local"].clone()

    launch_dispatch(ctx, hidden_a, None, plan_a, build_dedup_map=False)
    launch_dispatch_epilogue(ctx, plan_a)
    hidden_user = torch.empty_like(ctx["hidden_buf_local"])
    hidden_user.copy_(ctx["hidden_buf_local"])
    torch.cuda.synchronize()

    scatter_errors, checked = _verify_dispatch_by_dst(
        ctx, case, rank, R, dst_a, check_weights=False, hidden_user=hidden_user, s=s
    )
    padding_errors, padding_rows = _zero_row_errors(plan_a.zero_fill_ranges, hidden_user)
    weights_unchanged = torch.equal(ctx["weights_buf_local"], weights_after_b)
    dedup_unchanged = dedup_plan_fields_equal(plan_a, dedup_a_snapshot)
    errors = scatter_errors + padding_errors
    if not weights_unchanged:
        errors.append("weights_buf_local changed during hidden-only dispatch")
    if not dedup_unchanged:
        errors.append("saved plan dedup structures changed during reuse dispatch")

    assert_all_ranks(
        checked > 0 and padding_rows > 0 and not errors,
        rank,
        R,
        f"saved-plan hidden-only dispatch [s={s}/{case.S}]",
        "; ".join(errors[:5]) or f"checked={checked}, padding_rows={padding_rows}",
    )


def test_dispatch_saved_plan_hidden_only_reuses_dst_and_skips_weights(dist_env):
    rank, R = dist_env
    case = SAVED_PLAN_CASE
    ctx = init_case(case, R)
    _check_saved_plan_hidden_only(ctx, case, rank, R, case.S)


def test_dispatch_partial_tokens_saved_plan_reuse(dist_env):
    rank, R = dist_env
    case = SAVED_PLAN_CASE
    ctx = init_case(case, R)
    for s in partial_token_counts(case.S):
        _check_saved_plan_hidden_only(ctx, case, rank, R, s)


@pytest.mark.parametrize("case", case_params(LARGE_DISPATCH_CASES))
def test_dispatch_large_hidden_stride_spotcheck(dist_env, case):
    rank, R = dist_env
    ctx = init_case(case, R)
    hidden = _traceable_hidden(rank, case.S, case.H)
    weights = _traceable_weights(rank, case.S, case.K)

    dst, _plan, hidden_user, weights_user = _plan_and_dispatch(
        ctx, case, rank, R, hidden, weights
    )
    torch.cuda.synchronize()

    errors, checked = _verify_dispatch_by_dst(
        ctx, case, rank, R, dst, hidden_user=hidden_user,
        weights_user=weights_user, max_checks=256, copy_to_cpu=False
    )
    assert_all_ranks(
        checked > 0 and not errors,
        rank,
        R,
        f"{case.name} dispatch large-stride spotcheck",
        "; ".join(errors[:5]) or f"checked={checked}",
    )


def test_dispatch_rejects_bad_inputs(dist_env):
    from moonep.dispatch import launch_dispatch
    from moonep.dispatch_epilogue import launch_dispatch_epilogue
    from moonep.planning import allocate_planning_outputs, launch_planning

    rank, R = dist_env
    case = KernelCase("bad_inputs", S=4, K=2, epn=2, H=8, num_sms=1, token_padding=1)
    ctx = init_case(case, R)
    topk, tpe = make_topk(case, rank, R)
    hidden = _traceable_hidden(rank, case.S, case.H)
    weights = torch.rand(case.S, case.K, dtype=torch.float32, device="cuda")
    plan, _cu = allocate_planning_outputs(ctx)
    launch_planning(ctx, topk.reshape(-1).contiguous(), tpe, plan=plan, cu_seqlens=_cu)

    with pytest.raises(AssertionError, match="hidden_sh"):
        launch_dispatch(ctx, hidden.float(), weights, plan, build_dedup_map=True)
    with pytest.raises(AssertionError, match="hidden_sh"):
        launch_dispatch(
            ctx,
            hidden[: case.S // 2].contiguous(),
            weights,
            plan,
            build_dedup_map=True,
        )
    with pytest.raises(AssertionError, match="route_weights_sk"):
        launch_dispatch(ctx, hidden, weights.double(), plan, build_dedup_map=True)
    with pytest.raises(AssertionError, match="route_weights_sk"):
        launch_dispatch(
            ctx,
            hidden,
            weights[:, : case.K - 1].contiguous(),
            plan,
            build_dedup_map=True,
        )
    # A plan made for s tokens rejects hidden with a different row count.
    s = case.S // 2
    topk_s, tpe_s = make_topk(case, rank, R, s=s)
    plan_s, _cu_s = allocate_planning_outputs(ctx, s)
    launch_planning(ctx, topk_s.reshape(-1).contiguous(), tpe_s, plan=plan_s, cu_seqlens=_cu_s)
    with pytest.raises(AssertionError, match="hidden_sh"):
        launch_dispatch(ctx, hidden, weights[:s].contiguous(), plan_s, build_dedup_map=True)
    with pytest.raises(AssertionError, match="route_weights_sk"):
        launch_dispatch(ctx, hidden[:s].contiguous(), weights, plan_s, build_dedup_map=True)
    bad_plan = plan.clone()
    object.__setattr__(bad_plan, "dst", plan.dst.long())
    with pytest.raises(AssertionError, match="dst"):
        launch_dispatch(ctx, hidden, weights, bad_plan, build_dedup_map=True)
    bad_epilogue_plan = plan.clone()
    object.__setattr__(
        bad_epilogue_plan,
        "dup_loffs",
        plan.dup_loffs.flatten()[:-1].contiguous(),
    )
    with pytest.raises(AssertionError, match="dup_loffs"):
        launch_dispatch_epilogue(ctx, bad_epilogue_plan)
