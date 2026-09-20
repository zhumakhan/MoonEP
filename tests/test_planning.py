"""Planning kernel correctness tests.

Run with:
    torchrun --nproc_per_node=8 -m pytest -s tests/test_planning.py
"""

import pytest
import torch
from tests.kernel_test_utils import (
    DEFAULT_TOKEN_PADDING,
    KernelCase,
    assert_all_ranks,
    assert_tensor_equal_all_ranks,
    case_params,
    gather_tensor,
    init_case,
    make_topk,
    partial_token_counts,
    planning_invariant_errors,
    skip_if_unsupported_world_size,
)
from tests.planning_reference import launch_planning_torch_reference

# Part-2 vblock width of the planning kernel (planning.BLOCK_SIZE_P2): the
# runtime prefix loops run over nvb_rt = ceil(s*K / VBLOCK) vblocks.
VBLOCK = 2048

# Per-rank step sizes for the differing-s tests, indexed by rank (mod 8) and
# clamped into [1, S]. Deterministic on every rank, so the step total needs
# no collective.
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


def per_rank_tokens_total(S, R):
    return sum(per_rank_tokens(S, r) for r in range(R))


def ceil_div(x, y):
    return -(-x // y)


PLANNING_CASES = [
    KernelCase(
        "balanced_epn16",
        S=256,
        K=8,
        epn=16,
        H=128,
        num_sms=32,
        token_padding=DEFAULT_TOKEN_PADDING,
    ),
    KernelCase("tiny_s1_k1", S=1, K=1, epn=1, H=16, num_sms=1, token_padding=8),
    KernelCase(
        "tiny_biased_with_prefetch",
        S=3,
        K=5,
        epn=3,
        H=16,
        num_sms=1,
        B=1,
        token_padding=32,
        routing="biased",
        bias_ratio=1.0,
        seed=9002,
        min_R=2,
    ),
    KernelCase(
        "no_padding",
        S=33,
        K=2,
        epn=2,
        H=16,
        num_sms=1,
        B=1,
        token_padding=1,
        routing="biased",
        bias_ratio=1.0,
        seed=10001,
    ),
    KernelCase(
        "non_power_mild_bias",
        S=17,
        K=3,
        epn=3,
        H=16,
        num_sms=1,
        B=2,
        token_padding=16,
        routing="biased",
        bias_ratio=0.5,
        seed=10501,
    ),
    KernelCase(
        "small_balanced_with_prefetch",
        S=17,
        K=3,
        epn=3,
        H=16,
        num_sms=1,
        B=2,
        token_padding=16,
        seed=11001,
    ),
    KernelCase(
        "near_degenerate_bias",
        S=64,
        K=2,
        epn=5,
        H=32,
        num_sms=32,
        B=3,
        routing="biased",
        bias_ratio=5.0,
        seed=12001,
    ),
    KernelCase(
        "typical_bias",
        S=32,
        K=4,
        epn=4,
        H=32,
        num_sms=32,
        B=2,
        token_padding=64,
        routing="biased",
        bias_ratio=1.0,
        seed=13001,
    ),
    KernelCase(
        "heavy_bias",
        S=96,
        K=3,
        epn=7,
        H=48,
        num_sms=1,
        B=4,
        routing="biased",
        bias_ratio=2.0,
        seed=14001,
    ),
    KernelCase(
        "step1_cross_cta_group",
        S=1024,
        K=1,
        epn=256,
        H=16,
        num_sms=8,
        B=1,
        token_padding=1,
    ),
    KernelCase(
        "step1_multi_chunk_full_tile",
        S=2048,
        K=1,
        epn=320,
        H=16,
        num_sms=1,
        B=1,
        token_padding=1,
    ),
    KernelCase(
        "step1_segment_tail_full_tile",
        S=1536,
        K=1,
        epn=320,
        H=16,
        num_sms=2,
        B=1,
        token_padding=1,
        min_R=4,
    ),
    KernelCase(
        "experts_gt_block_size",
        S=256,
        K=1,
        epn=1025,
        H=16,
        num_sms=8,
        B=1,
        token_padding=1,
        max_R=2,
    ),
    KernelCase(
        "all_local",
        S=32,
        K=4,
        epn=8,
        H=32,
        num_sms=8,
        token_padding=16,
        routing="all_local",
    ),
    KernelCase(
        "all_remote_with_prefetch_slots",
        S=32,
        K=4,
        epn=8,
        H=32,
        num_sms=8,
        B=3,
        token_padding=16,
        routing="all_remote",
        min_R=2,
    ),
    KernelCase(
        "duplicate_topk",
        S=16,
        K=4,
        epn=4,
        H=32,
        num_sms=8,
        B=2,
        token_padding=8,
        routing="duplicate_topk",
        min_R=2,
    ),
    KernelCase(
        "single_expert_exact_padding",
        S=8,
        K=2,
        epn=4,
        H=16,
        num_sms=1,
        B=1,
        token_padding=8,
        routing="single_expert",
    ),
    KernelCase(
        "single_expert_padding_tail",
        S=9,
        K=2,
        epn=4,
        H=16,
        num_sms=1,
        B=1,
        token_padding=8,
        routing="single_expert",
    ),
    # S*K > VBLOCK: the runtime vblock prefix loops run over several vblocks
    # (nvb_rt = 4 at full S, 2 at S//2, 4 with a partial tail at S-1).
    KernelCase(
        "multi_vblock_k8",
        S=1024,
        K=8,
        epn=16,
        H=32,
        num_sms=32,
        B=2,
    ),
    KernelCase(
        "multi_vblock_k1",
        S=8192,
        K=1,
        epn=16,
        H=32,
        num_sms=32,
        B=2,
    ),
    KernelCase(
        "multi_vblock_biased",
        S=1024,
        K=4,
        epn=8,
        H=32,
        num_sms=8,
        B=2,
        token_padding=32,
        routing="biased",
        bias_ratio=1.0,
        seed=15001,
    ),
]

# Cases whose capacity spans at least 4 vblocks, so partial steps can cover
# nvb_rt in {2, 3, 4} with full and partial tail vblocks.
MULTI_VBLOCK_CASES = [case for case in PLANNING_CASES if case.S * case.K >= 4 * VBLOCK]


def _align_up(x, alignment):
    return ((x + alignment - 1) // alignment) * alignment


def _step1_params(case, R):
    E = case.E(R)
    seg_raw = (E + case.num_sms - 1) // case.num_sms
    experts_per_block = _align_up(seg_raw, 32)
    s1_cols = min(experts_per_block, 512)
    work_ctas = (E + experts_per_block - 1) // experts_per_block
    group_spans_ctas = experts_per_block < case.epn
    has_segment_tail = experts_per_block > s1_cols and experts_per_block % s1_cols != 0
    return {
        "E": E,
        "experts_per_block": experts_per_block,
        "s1_cols": s1_cols,
        "work_ctas": work_ctas,
        "group_spans_ctas": group_spans_ctas,
        "has_segment_tail": has_segment_tail,
    }


def test_planning_step1_case_coverage():
    params = [
        _step1_params(case, R)
        for R in (2, 4)
        for case in PLANNING_CASES
        if R >= case.min_R and (case.max_R is None or R <= case.max_R)
    ]

    assert any(p["E"] > 2048 for p in params)
    assert any(p["s1_cols"] > 32 for p in params)
    assert any(p["s1_cols"] == 512 for p in params)
    assert any(p["experts_per_block"] > p["s1_cols"] for p in params)
    assert any(p["has_segment_tail"] for p in params)
    assert any(p["work_ctas"] > 1 and p["group_spans_ctas"] for p in params)

    # The runtime vblock loops must be exercised with more than one vblock at
    # partial s as well as at full S.
    nvb_full = {ceil_div(case.S * case.K, VBLOCK) for case in MULTI_VBLOCK_CASES}
    nvb_partial = {
        ceil_div(s * case.K, VBLOCK)
        for case in MULTI_VBLOCK_CASES
        for s in partial_token_counts(case.S)
    }
    assert MULTI_VBLOCK_CASES and nvb_full >= {4}
    assert nvb_partial >= {1, 2, 4}
    assert any(
        case.S * case.K > VBLOCK and case.S * case.K < 4 * VBLOCK
        for case in PLANNING_CASES
    )


def _compare_plan_with_reference(case, ctx, rank, R, s, plan, cu_seqlens, topk, tpe, tag):
    """Kernel plan vs the torch reference for this rank's ``s`` tokens (the
    reference derives ``s`` from ``topk`` and the receive caps from the
    gathered ``tpe``), plus the structural invariants bounded by ``nvs_s``.
    ``dst`` is compared per rank because its length may differ across ranks."""
    (
        ref_dst,
        ref_cu_seqlens,
        ref_experts_to_copy,
        ref_remote_stats,
        ref_zero_fill_ranges,
        _ref_dedup_plan,
    ) = launch_planning_torch_reference(ctx, topk, tpe)

    assert_tensor_equal_all_ranks(f"cu_seqlens{tag}", cu_seqlens, ref_cu_seqlens, rank, R)
    assert_tensor_equal_all_ranks(
        f"zero_fill_ranges{tag}", plan.zero_fill_ranges, ref_zero_fill_ranges, rank, R
    )
    assert_tensor_equal_all_ranks(
        f"experts_to_copy{tag}", plan.experts_to_copy, ref_experts_to_copy, rank, R
    )
    assert_tensor_equal_all_ranks(
        f"remote_stats{tag}", plan.remote_stats, ref_remote_stats, rank, R
    )
    dst_ok = torch.equal(plan.dst.cpu(), ref_dst.cpu())
    dst_detail = ""
    if not dst_ok:
        diff = (plan.dst.cpu() != ref_dst.cpu()).nonzero().flatten()[:5].tolist()
        dst_detail = f"{len(diff)}+ of {s * case.K} entries differ, first at {diff}"
    assert_all_ranks(dst_ok, rank, R, f"dst{tag}", dst_detail)

    errors = planning_invariant_errors(
        case, ctx, plan.dst, cu_seqlens, plan.experts_to_copy,
        num_tokens=s, nvs_s=plan.nvs_s,
    )
    assert_all_ranks(
        not errors,
        rank,
        R,
        f"{case.name} {tag} planning invariants",
        "; ".join(errors[:5]),
    )


def _check_uniform_partial_plan(case, ctx, rank, R, s):
    """Every rank plans the same ``s`` on a Buffer of capacity S (uniform
    mode, no ``total_num_tokens``): the plan is sized ``s*K`` and the shard
    prefix dispatch may write is exactly ``s*K + token_padding_extra``."""
    from moonep.planning import allocate_planning_outputs, launch_planning

    topk, tpe = make_topk(case, rank, R, s=s)
    plan, cu_seqlens = allocate_planning_outputs(ctx, s)
    assert plan.num_tokens == s and plan.N == s * case.K
    assert plan.nvs_s == s * case.K + int(ctx["token_padding_extra"]), (
        f"plan.nvs_s={plan.nvs_s}, expected s*K + extra = "
        f"{s * case.K + int(ctx['token_padding_extra'])}"
    )
    assert plan.nvs_s <= int(ctx["NvS"])
    launch_planning(ctx, topk.reshape(-1).contiguous(), tpe, plan=plan, cu_seqlens=cu_seqlens)
    tag = f"[s={s}/{case.S}]"
    _compare_plan_with_reference(case, ctx, rank, R, s, plan, cu_seqlens, topk, tpe, tag)
    return plan, cu_seqlens


@pytest.mark.parametrize("case", case_params(PLANNING_CASES))
def test_planning_partial_tokens_matches_reference(dist_env, case):
    """Plan s < S tokens on a Buffer built for S. The kernel loops over the
    runtime s, the torch reference derives s from the routing input, and every
    rank must still end up with exactly s*K tokens (plus segment padding)."""
    rank, R = dist_env
    skip_if_unsupported_world_size(case, R)
    if case.S < 2:
        pytest.skip("partial-token planning needs S >= 2")

    ctx = init_case(case, R)
    # 1 token, half the capacity, and one below it; s below num_sms leaves
    # some planning CTAs with an empty token range.
    for s in partial_token_counts(case.S):
        _check_uniform_partial_plan(case, ctx, rank, R, s)


@pytest.mark.parametrize("case", case_params(MULTI_VBLOCK_CASES))
def test_planning_multi_vblock_partial_tokens(dist_env, case):
    """Partial steps whose s*K spans 2, 3 and 4 vblocks, with exact and
    partial last vblocks, so the runtime vblock prefix loops are exercised
    with nvb_rt > 1 and a tail vblock that is not full."""
    rank, R = dist_env
    skip_if_unsupported_world_size(case, R)

    ctx = init_case(case, R)
    steps = sorted({
        ceil_div(VBLOCK, case.K) + 1,      # 2 vblocks, partial tail
        ceil_div(2 * VBLOCK, case.K),      # exactly 2 full vblocks
        ceil_div(2 * VBLOCK, case.K) + 1,  # 3 vblocks, partial tail
        case.S - 1,                        # 4 vblocks, partial tail
    })
    nvb_seen = set()
    for s in steps:
        assert 0 < s <= case.S
        nvb_seen.add(ceil_div(s * case.K, VBLOCK))
        _check_uniform_partial_plan(case, ctx, rank, R, s)
    assert nvb_seen >= {2, 3, 4}, f"vblock counts covered: {sorted(nvb_seen)}"


@pytest.mark.parametrize("case", case_params(PLANNING_CASES))
def test_planning_per_rank_tokens_matches_reference(dist_env, case):
    """Each rank plans its own s_r this step. The host passes the step total
    (sum over ranks of s_r); the planner sizes this rank's receive cap as
    ceil_div(total*K, R) rounded per rank, and the torch reference derives
    the same caps from the gathered tokens_per_expert."""
    from moonep.planning import allocate_planning_outputs, launch_planning

    rank, R = dist_env
    skip_if_unsupported_world_size(case, R)

    ctx = init_case(case, R)
    s_r = per_rank_tokens(case.S, rank)
    total = per_rank_tokens_total(case.S, R)
    assert s_r <= total <= R * case.S
    extra = int(ctx["token_padding_extra"])

    topk, tpe = make_topk(case, rank, R, s=s_r)
    plan, cu_seqlens = allocate_planning_outputs(ctx, s_r, total_num_tokens=total)
    assert plan.num_tokens == s_r and plan.N == s_r * case.K
    assert plan.nvs_s == ceil_div(total * case.K, R) + extra, (
        f"plan.nvs_s={plan.nvs_s}, expected ceil_div({total}*{case.K}, {R}) + {extra}"
    )
    assert plan.nvs_s <= int(ctx["NvS"])
    launch_planning(
        ctx, topk.reshape(-1).contiguous(), tpe,
        plan=plan, cu_seqlens=cu_seqlens, total_num_tokens=total,
    )
    tag = f"[s_r={s_r}/{case.S}, total={total}]"
    _compare_plan_with_reference(case, ctx, rank, R, s_r, plan, cu_seqlens, topk, tpe, tag)

    # Every rank receives at most its cap (plus segment padding): the padded
    # layout end is bounded by nvs_s, and the sum of real slots over ranks is
    # the step total.
    real_slots = int(cu_seqlens[-1].item()) - int(plan.zero_fill_ranges[:, 1].sum().item())
    cap_r = (total * case.K) // R + (1 if rank < (total * case.K) % R else 0)
    assert real_slots <= cap_r, f"{tag} rank receives {real_slots} > cap {cap_r}"
    slots = torch.tensor([real_slots], dtype=torch.int64, device="cuda")
    all_slots = gather_tensor(slots, R).flatten().cpu()
    assert int(all_slots.sum().item()) == total * case.K, (
        f"{tag} real slots over ranks {all_slots.tolist()} != total*K={total * case.K}"
    )


@pytest.mark.parametrize("case", case_params(PLANNING_CASES))
def test_planning_matches_reference_and_invariants(dist_env, case):
    from moonep.planning import allocate_planning_outputs, launch_planning

    rank, R = dist_env
    skip_if_unsupported_world_size(case, R)

    ctx = init_case(case, R)
    topk, tpe = make_topk(case, rank, R)

    plan, cu_seqlens = allocate_planning_outputs(ctx)
    assert (
        launch_planning(ctx, topk.reshape(-1).contiguous(), tpe, plan=plan, cu_seqlens=cu_seqlens)
        is None
    )
    dst = plan.dst
    experts_to_copy = plan.experts_to_copy
    remote_stats = plan.remote_stats
    (
        ref_dst,
        ref_cu_seqlens,
        ref_experts_to_copy,
        ref_remote_stats,
        ref_zero_fill_ranges,
        _ref_dedup_plan,
    ) = launch_planning_torch_reference(ctx, topk, tpe)

    assert_tensor_equal_all_ranks("cu_seqlens", cu_seqlens, ref_cu_seqlens, rank, R)
    assert_tensor_equal_all_ranks(
        "zero_fill_ranges", plan.zero_fill_ranges, ref_zero_fill_ranges, rank, R
    )
    assert_tensor_equal_all_ranks(
        "experts_to_copy", experts_to_copy, ref_experts_to_copy, rank, R
    )
    assert_tensor_equal_all_ranks(
        "remote_stats", remote_stats, ref_remote_stats, rank, R
    )
    assert_tensor_equal_all_ranks(
        "dst", dst.reshape(case.S, case.K), ref_dst.reshape(case.S, case.K), rank, R
    )

    assert plan.nvs_s == int(ctx["NvS"]), f"full-S plan.nvs_s={plan.nvs_s} != NvS"
    errors = planning_invariant_errors(
        case, ctx, dst, cu_seqlens, experts_to_copy, nvs_s=plan.nvs_s
    )
    assert_all_ranks(
        not errors,
        rank,
        R,
        f"{case.name} planning invariants",
        "; ".join(errors[:5]),
    )
