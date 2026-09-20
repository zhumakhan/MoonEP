"""Combine kernel correctness tests.

Run with:
    torchrun --nproc_per_node=8 -m pytest -s tests/test_combine.py

The round-trip and global-reference tests also run at partial step sizes
``s < S`` on a Buffer of capacity S (same ``s`` on every rank).
"""

import pytest
import torch

from tests.kernel_test_utils import (
    KernelCase,
    assert_close_all_ranks,
    assert_tensor_equal_all_ranks,
    assert_ulp_all_ranks,
    case_params,
    gather_tensor,
    init_case,
    make_topk,
    partial_token_counts,
)


COMBINE_CASES = [
    KernelCase("balanced", S=256, K=8, epn=16, H=128, num_sms=32),
    KernelCase("k1_direct_gather", S=32, K=1, epn=4, H=128, num_sms=8),
    KernelCase(
        "all_remote",
        S=64,
        K=4,
        epn=8,
        H=128,
        num_sms=8,
        routing="all_remote",
        min_R=2,
    ),
    KernelCase(
        "duplicate_topk",
        S=32,
        K=4,
        epn=4,
        H=128,
        num_sms=8,
        routing="duplicate_topk",
        min_R=2,
    ),
    KernelCase(
        "duplicate_topk_cross_stage",
        S=64,
        K=3,
        epn=4,
        H=128,
        num_sms=8,
        routing="duplicate_topk",
        min_R=2,
    ),
]

# Extra shapes for the partial-s sweep: odd K without duplicates and a
# single-expert split (one expert's tokens spread over several ranks).
PARTIAL_COMBINE_CASES = COMBINE_CASES + [
    KernelCase("odd_k5_partial", S=40, K=5, epn=8, H=128, num_sms=8, B=2, token_padding=16),
    KernelCase(
        "single_expert_split_partial",
        S=24,
        K=2,
        epn=4,
        H=128,
        num_sms=8,
        B=1,
        token_padding=8,
        routing="single_expert",
        min_R=2,
    ),
]

GLOBAL_REF_CASES = [
    KernelCase("global_balanced", S=64, K=4, epn=8, H=128, num_sms=8, B=2),
    KernelCase(
        "global_all_remote",
        S=64,
        K=4,
        epn=8,
        H=128,
        num_sms=8,
        B=2,
        routing="all_remote",
        min_R=2,
    ),
]

LARGE_COMBINE_CASES = [
    KernelCase(
        "large_hidden_stride_identity",
        S=8192,
        K=16,
        epn=14,
        H=7168,
        num_sms=32,
        B=4,
    )
]


def _step_tokens(case, s):
    s = case.S if s is None else int(s)
    assert 0 < s <= case.S, f"num_tokens {s} outside [1, S={case.S}]"
    return s


def _random_inputs(case, rank, R, seed=0, s=None):
    s = _step_tokens(case, s)
    dev = "cuda"
    gen = torch.Generator(device=dev).manual_seed(seed + rank)
    hidden = torch.randn(s, case.H, dtype=torch.bfloat16, device=dev, generator=gen)
    weights = torch.rand(s, case.K, dtype=torch.float32, device=dev, generator=gen)
    topk, tpe = make_topk(case, rank, R, s=s)
    return hidden, weights, topk, tpe


def _dispatch_inputs(ctx, case, rank, R, seed=0, s=None):
    """Plan ``s`` tokens (default ``case.S``) and dispatch random inputs;
    returns the full ``[NvS, ...]`` copies of this rank's shard."""
    from moonep.dispatch import launch_dispatch
    from moonep.dispatch_epilogue import launch_dispatch_epilogue
    from moonep.planning import allocate_planning_outputs, launch_planning

    s = _step_tokens(case, s)
    hidden, weights, topk, tpe = _random_inputs(case, rank, R, seed, s=s)
    plan, cu_seqlens = allocate_planning_outputs(ctx, s)
    assert plan.num_tokens == s and plan.N == s * case.K
    launch_planning(ctx, topk.reshape(-1).contiguous(), tpe, plan=plan, cu_seqlens=cu_seqlens)
    dst = plan.dst
    experts_to_copy = plan.experts_to_copy
    expert_ids = _expert_ids_from_experts_to_copy(ctx, cu_seqlens, experts_to_copy[rank])
    launch_dispatch(ctx, hidden, weights, plan, build_dedup_map=True)
    launch_dispatch_epilogue(ctx, plan)
    hidden_user = torch.empty_like(ctx["hidden_buf_local"])
    hidden_user.copy_(ctx["hidden_buf_local"])
    weights_user = torch.empty(
        (int(ctx["NvS"]),), dtype=torch.float32, device=hidden_user.device
    )
    weights_user.copy_(ctx["weights_buf_local"].view(torch.float32))
    return hidden, weights, dst, cu_seqlens, expert_ids, plan, hidden_user, weights_user


def _combine_full(ctx, case, plan, hidden_user, weights_user=None):
    """zero_copy=False style combine: stage the (full or prefix) user tensors
    into the NVL shard, pre-reduce duplicates, then gather. Output rows follow
    the plan's token count."""
    from moonep.combine import launch_combine
    from moonep.combine_prologue import launch_combine_prologue
    from moonep.inter_rank_sync import launch_inter_rank_sync

    s = plan.num_tokens
    output = torch.empty(
        s, case.H, dtype=torch.bfloat16, device=hidden_user.device
    )
    launch_inter_rank_sync(ctx)
    rows = int(hidden_user.shape[0])
    ctx["hidden_buf_local"][:rows].copy_(hidden_user)
    if weights_user is not None:
        w_rows = int(weights_user.shape[0])
        ctx["weights_buf_local"][:w_rows].copy_(weights_user.view(torch.int32))
    launch_combine_prologue(ctx, plan)
    if weights_user is None:
        launch_combine(ctx, output, plan.dst)
        return output, None

    output_sk = torch.empty(
        s, case.K, dtype=torch.float32, device=hidden_user.device
    )
    launch_combine(ctx, output, plan.dst, output_sk=output_sk)
    return output, output_sk


def _expert_ids_from_experts_to_copy(ctx, cu_seqlens, experts_to_copy_row):
    E = int(ctx["E"])
    B = int(ctx["B"])
    expert_ids = torch.full((E + B,), -1, dtype=torch.int32, device=cu_seqlens.device)
    prev = 0
    for group_id in range(E + B):
        cur = int(cu_seqlens[group_id].item())
        if cur > prev:
            if group_id < E:
                expert_ids[group_id] = group_id
            else:
                expert_ids[group_id] = experts_to_copy_row[group_id - E]
        prev = cur
    return expert_ids


def _uniform_scale(buf, cu_seqlens, expert_ids):
    total = int(cu_seqlens[-1].item())
    buf[:total].mul_(3.0)


def _per_expert_scale(buf, cu_seqlens, expert_ids):
    prev = 0
    for group_id in range(cu_seqlens.numel()):
        cur = int(cu_seqlens[group_id].item())
        if cur > prev:
            expert_id = int(expert_ids[group_id].item())
            if expert_id >= 0:
                buf[prev:cur].mul_(float(expert_id + 1))
        prev = cur


# (name, expert_fn, use_ulp): per_expert_scale is slot-precise, so it is
# compared in bf16 ulps.
EXPERT_FNS = [
    ("uniform_scale", _uniform_scale, False),
    ("per_expert_scale", _per_expert_scale, True),
]


def _combine_global_reference(ctx, case, rank, R, hidden, dst, cu_seqlens,
                              experts_to_copy, expert_fn):
    """Slot-precise reference: rebuild every rank's shard from the gathered
    dst, apply ``expert_fn`` per destination rank and gather back. ``s`` is
    taken from ``hidden`` (the same on every rank)."""
    s = int(hidden.shape[0])
    nvs_stride = int(ctx["NvS"])
    NvS_padded = int(ctx["NvS_padded"])
    all_hidden = gather_tensor(hidden.contiguous(), R)
    all_dst = gather_tensor(dst.reshape(s, case.K).contiguous(), R)
    all_cu = gather_tensor(cu_seqlens.contiguous(), R)

    global_buf = torch.zeros(R, NvS_padded, case.H, dtype=torch.bfloat16,
                             device="cuda")
    for src_r in range(R):
        for tok in range(s):
            for k in range(case.K):
                dst_val = int(all_dst[src_r, tok, k].item())
                raw_dst = -dst_val - 1 if dst_val < 0 else dst_val
                dest_rank = raw_dst // nvs_stride
                local_off = raw_dst % nvs_stride
                global_buf[dest_rank, local_off] = all_hidden[src_r, tok]

    for dest_r in range(R):
        expert_ids = _expert_ids_from_experts_to_copy(
            ctx, all_cu[dest_r], experts_to_copy[dest_r]
        )
        expert_fn(global_buf[dest_r], all_cu[dest_r], expert_ids)

    ref = torch.zeros(s, case.H, dtype=torch.float32, device="cuda")
    local_dst = all_dst[rank]
    for tok in range(s):
        for k in range(case.K):
            dst_val = int(local_dst[tok, k].item())
            raw_dst = -dst_val - 1 if dst_val < 0 else dst_val
            dest_rank = raw_dst // nvs_stride
            local_off = raw_dst % nvs_stride
            ref[tok] += global_buf[dest_rank, local_off].float()
    return ref.to(torch.bfloat16)


def _check_identity_round_trip(ctx, case, rank, R, s, label):
    hidden, _weights, _dst, _cu, _expert_ids, plan, hidden_user, _weights_user = _dispatch_inputs(
        ctx, case, rank, R, s=s
    )

    output, _ = _combine_full(ctx, case, plan, hidden_user)
    assert tuple(output.shape) == (s, case.H), f"combine output {tuple(output.shape)}"
    ref = (hidden.float() * case.K).to(torch.bfloat16)
    assert_close_all_ranks(f"{case.name} {label}", output, ref, rank, R)


@pytest.mark.parametrize("case", case_params(COMBINE_CASES))
def test_combine_identity_round_trip(dist_env, case):
    rank, R = dist_env
    ctx = init_case(case, R)
    _check_identity_round_trip(ctx, case, rank, R, case.S, "identity combine")


@pytest.mark.parametrize("case", case_params(PARTIAL_COMBINE_CASES))
def test_combine_partial_tokens_identity_round_trip(dist_env, case):
    """combine(dispatch(x)) == K*x at s in {1, S//2, S-1} on one Buffer of
    capacity S (K=1 direct gather, dedup and odd-K cases included)."""
    rank, R = dist_env
    if case.S < 2:
        pytest.skip("partial-token combine needs S >= 2")
    ctx = init_case(case, R)
    for s in partial_token_counts(case.S):
        _check_identity_round_trip(
            ctx, case, rank, R, s, f"[s={s}/{case.S}] identity combine"
        )


def _check_global_reference(ctx, case, rank, R, s, expert_name, expert_fn, use_ulp):
    hidden, _weights, dst, cu_seqlens, expert_ids, plan, hidden_user, _weights_user = _dispatch_inputs(
        ctx, case, rank, R, s=s
    )

    expert_fn(hidden_user, cu_seqlens, expert_ids)
    torch.cuda.synchronize()
    ref = _combine_global_reference(
        ctx, case, rank, R, hidden, dst, cu_seqlens, plan.experts_to_copy, expert_fn
    )

    output, _ = _combine_full(ctx, case, plan, hidden_user)
    name = f"{case.name} [s={s}/{case.S}] {expert_name} combine"
    if use_ulp:
        assert_ulp_all_ranks(name, output, ref, rank, R, max_ulps=1)
    else:
        assert_close_all_ranks(name, output, ref, rank, R)


@pytest.mark.parametrize("case", case_params(GLOBAL_REF_CASES))
@pytest.mark.parametrize("expert_name,expert_fn,use_ulp", EXPERT_FNS)
def test_combine_matches_global_reference(dist_env, case, expert_name, expert_fn, use_ulp):
    rank, R = dist_env
    ctx = init_case(case, R)
    _check_global_reference(ctx, case, rank, R, case.S, expert_name, expert_fn, use_ulp)


@pytest.mark.parametrize("case", case_params(GLOBAL_REF_CASES))
@pytest.mark.parametrize("expert_name,expert_fn,use_ulp", EXPERT_FNS)
def test_combine_partial_tokens_matches_global_reference(
    dist_env, case, expert_name, expert_fn, use_ulp
):
    """Slot-precise global reference at partial s: each rank's shard is
    rebuilt from the gathered dst, scaled per expert and gathered back."""
    rank, R = dist_env
    ctx = init_case(case, R)
    for s in partial_token_counts(case.S):
        _check_global_reference(ctx, case, rank, R, s, expert_name, expert_fn, use_ulp)


def _check_output_sk_gathers_route_weights(ctx, case, rank, R, s, seed):
    hidden, weights, _dst, _cu, _expert_ids, plan, hidden_user, _weights_user = _dispatch_inputs(
        ctx, case, rank, R, seed=seed, s=s
    )

    out_ref, _ = _combine_full(ctx, case, plan, hidden_user)

    # Re-dispatch because combine reads the mutable NVL buffer.
    _hidden, _weights, _dst, _cu, _expert_ids, plan, hidden_user, weights_user = _dispatch_inputs(
        ctx, case, rank, R, seed=seed, s=s
    )
    out, out_weights = _combine_full(ctx, case, plan, hidden_user, weights_user)

    tag = f"[s={s}/{case.S}]"
    assert tuple(out_weights.shape) == (s, case.K), f"output_sk {tuple(out_weights.shape)}"
    assert_tensor_equal_all_ranks(f"output_sk hidden{tag}", out, out_ref, rank, R)
    assert_tensor_equal_all_ranks(f"output_sk weights{tag}", out_weights, weights, rank, R)
    assert_tensor_equal_all_ranks(f"output_sk hidden input{tag}", _hidden, hidden, rank, R)
    assert_tensor_equal_all_ranks(f"output_sk weight input{tag}", _weights, weights, rank, R)


def test_combine_output_sk_gathers_route_weights(dist_env):
    rank, R = dist_env
    case = KernelCase("output_sk", S=128, K=4, epn=8, H=128, num_sms=8, B=2)
    ctx = init_case(case, R)
    _check_output_sk_gathers_route_weights(ctx, case, rank, R, case.S, seed=100)


def test_combine_partial_tokens_output_sk_gathers_route_weights(dist_env):
    rank, R = dist_env
    case = KernelCase("output_sk_partial", S=128, K=4, epn=8, H=128, num_sms=8, B=2)
    ctx = init_case(case, R)
    for s in partial_token_counts(case.S):
        _check_output_sk_gathers_route_weights(ctx, case, rank, R, s, seed=100 + s)


def test_buffer_combine_stages_external_buffers_and_gathers_weights(dist_env):
    rank, R = dist_env
    case = KernelCase("public_staging", S=128, K=4, epn=8, H=128, num_sms=8, B=2)
    ctx = init_case(case, R)
    buffer = ctx["_buffer"]
    _hidden, weights, _dst, _cu, _expert_ids, plan, hidden_user, weights_user = _dispatch_inputs(
        ctx, case, rank, R, seed=200
    )

    ref_out, ref_weights = _combine_full(ctx, case, plan, hidden_user, weights_user)

    staged_hidden = hidden_user.clone()
    staged_weights = weights_user.clone()
    ctx["hidden_buf_local"].zero_()
    ctx["weights_buf_local"].zero_()

    out, out_weights, _ = buffer.combine(
        plan=plan,
        hidden_nvsh=staged_hidden,
        route_weights_nvs=staged_weights,
    )

    assert_tensor_equal_all_ranks("public staging hidden", out, ref_out, rank, R)
    assert_tensor_equal_all_ranks("public staging weights", out_weights, weights, rank, R)
    assert_tensor_equal_all_ranks("public staging ref weights", ref_weights, weights, rank, R)


def test_buffer_combine_partial_tokens_accepts_prefix_views(dist_env):
    """Buffer.combine takes either the full [NvS, H] shard copy or the
    [plan.nvs_s, H] prefix that dispatch returns at partial s; both must give
    the same [s, H] / [s, K] outputs."""
    rank, R = dist_env
    case = KernelCase("public_prefix", S=128, K=4, epn=8, H=128, num_sms=8, B=2)
    ctx = init_case(case, R)
    buffer = ctx["_buffer"]
    NvS = int(ctx["NvS"])
    extra = int(ctx["token_padding_extra"])
    for s in partial_token_counts(case.S):
        _hidden, weights, _dst, cu_seqlens, _expert_ids, plan, hidden_user, weights_user = (
            _dispatch_inputs(ctx, case, rank, R, seed=200 + s, s=s)
        )
        nvs_s = int(plan.nvs_s)
        tag = f"[s={s}/{case.S}]"
        assert nvs_s == s * case.K + extra, f"{tag} plan.nvs_s={nvs_s}"
        assert int(cu_seqlens[-1].item()) <= nvs_s <= NvS, f"{tag} cu_seqlens[-1] > nvs_s"

        ref_out, ref_weights = _combine_full(ctx, case, plan, hidden_user, weights_user)
        assert tuple(ref_out.shape) == (s, case.H) and tuple(ref_weights.shape) == (s, case.K)

        # Full-capacity copies through the public API.
        ctx["hidden_buf_local"].zero_()
        ctx["weights_buf_local"].zero_()
        out_full, w_full, _ = buffer.combine(
            plan=plan,
            hidden_nvsh=hidden_user.clone(),
            route_weights_nvs=weights_user.clone(),
        )
        # Prefix views of exactly plan.nvs_s rows.
        ctx["hidden_buf_local"].zero_()
        ctx["weights_buf_local"].zero_()
        out_prefix, w_prefix, _ = buffer.combine(
            plan=plan,
            hidden_nvsh=hidden_user[:nvs_s].clone(),
            route_weights_nvs=weights_user[:nvs_s].clone(),
        )
        assert_tensor_equal_all_ranks(f"public full hidden{tag}", out_full, ref_out, rank, R)
        assert_tensor_equal_all_ranks(f"public full weights{tag}", w_full, weights, rank, R)
        assert_tensor_equal_all_ranks(f"public prefix hidden{tag}", out_prefix, ref_out, rank, R)
        assert_tensor_equal_all_ranks(f"public prefix weights{tag}", w_prefix, weights, rank, R)
        assert_tensor_equal_all_ranks(f"public ref weights{tag}", ref_weights, weights, rank, R)


def test_buffer_combine_async_gathers_weights(dist_env):
    rank, R = dist_env
    case = KernelCase("async_weights", S=128, K=4, epn=8, H=128, num_sms=8, B=2)
    ctx = init_case(case, R)
    buffer = ctx["_buffer"]
    _hidden, weights, _dst, _cu, _expert_ids, plan, hidden_user, weights_user = _dispatch_inputs(
        ctx, case, rank, R, seed=300
    )

    ref_out, ref_weights, _ = buffer.combine(
        plan=plan,
        hidden_nvsh=hidden_user,
        route_weights_nvs=weights_user,
    )

    _hidden, _weights, _dst, _cu, _expert_ids, plan, hidden_user, weights_user = _dispatch_inputs(
        ctx, case, rank, R, seed=300
    )
    out, out_weights, event = buffer.combine(
        plan=plan,
        hidden_nvsh=hidden_user,
        route_weights_nvs=weights_user,
        async_finish=True,
    )
    event.wait(torch.cuda.current_stream())
    torch.cuda.synchronize()

    assert_tensor_equal_all_ranks("async combine hidden", out, ref_out, rank, R)
    assert_tensor_equal_all_ranks("async combine weights", out_weights, weights, rank, R)


@pytest.mark.parametrize("case", case_params(LARGE_COMBINE_CASES))
def test_combine_large_hidden_stride_identity(dist_env, case):
    rank, R = dist_env
    ctx = init_case(case, R)
    hidden, _weights, _dst, _cu, _expert_ids, plan, hidden_user, _weights_user = _dispatch_inputs(
        ctx, case, rank, R
    )

    output, _ = _combine_full(ctx, case, plan, hidden_user)
    ref = (hidden.float() * case.K).to(torch.bfloat16)
    assert_close_all_ranks(f"{case.name} combine", output, ref, rank, R)


def test_combine_rejects_bad_inputs(dist_env):
    from moonep.combine import launch_combine

    rank, R = dist_env
    case = KernelCase("bad_inputs", S=4, K=2, epn=2, H=128, num_sms=1)
    ctx = init_case(case, R)
    buffer = ctx["_buffer"]
    _hidden, _weights, dst, _cu, _expert_ids, plan, hidden_user, _weights_user = _dispatch_inputs(
        ctx, case, rank, R
    )
    output = torch.empty(case.S, case.H, dtype=torch.bfloat16, device="cuda")

    with pytest.raises(TypeError, match="hidden_sh"):
        buffer.combine(hidden_sh=output, plan=plan, hidden_nvsh=hidden_user)
    with pytest.raises(AssertionError, match="plan is required"):
        buffer.combine(hidden_nvsh=hidden_user)
    with pytest.raises(AssertionError):
        buffer.combine(plan=plan)
    with pytest.raises(AssertionError):
        buffer.combine(
            plan=plan,
            hidden_nvsh=hidden_user,
            route_weights_nvs=torch.empty(
                case.S, case.K, dtype=torch.float32, device="cuda"
            ),
        )
    # hidden_nvsh must have NvS or plan.nvs_s rows; s*K rows is neither.
    with pytest.raises(AssertionError):
        buffer.combine(
            plan=plan,
            hidden_nvsh=hidden_user[: case.S * case.K].contiguous(),
        )
    with pytest.raises(AssertionError, match="output_sk"):
        bad_output_sk = torch.empty(case.S, case.K, dtype=torch.bfloat16,
                                    device="cuda")
        launch_combine(ctx, output, dst, output_sk=bad_output_sk)
