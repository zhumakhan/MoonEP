"""
ExpertPools correctness test: one set of expert pools shared by every MoE layer.

Run with:
    torchrun --nproc_per_node=8 -m pytest -s tests/test_expert_pools.py
or, without pytest (same process-group setup, every test in order):
    torchrun --nproc_per_node=8 tests/test_expert_pools.py

Coverage axes:
  - layout: shapes, dtypes, strides, rank padding gap, 16-byte alignment,
    zero init, and aliasing (prefetch_buffer[rank] is weights[epn:],
    reduce_buffer[rank] is grads[epn:]), also by data: every rank tags its
    rows and reads every rank's slot rows through the all-rank views.
  - push: real plans from Buffer.dispatch on skewed routing; each used slot
    holds the owner's staging row, unused slots, the staging rows and the
    padding gap are untouched.
  - reduce: local grads = own + every rank's slots holding this rank's
    experts (exact fp32), consumed slots zeroed, the others untouched; two
    plans on the same pools, without a host barrier before reduce.
  - sharing: two "layers" alternate on the pools with their own weights and
    plans; re-pushing the first layer's saved plan after the second layer
    used the pools restores its copies.
  - lifecycle: destroy() is idempotent, the pools refuse use afterwards, and
    new pools on the same Buffer work.
  - validation: bad projection names, dims and groups are rejected.
"""

import os
import sys

import torch
import torch.distributed as dist

if __package__ in (None, ""):
    # Run as a script: make the repo root (moonep, tests) importable.
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from moonep import Buffer, ExpertPools  # noqa: E402
from moonep._C import get_vmm_granularity  # noqa: E402
from moonep.api import _validate_rank_strided_pool  # noqa: E402
from tests.generate_topk_routing import generate_topk_routing  # noqa: E402


# Tokens per rank, top-k, hidden size, expert FFN size, experts per rank. With
# (out, in) = (384, 256) a [2*epn, out, in] chunk is 1.5 MiB bf16 / 3 MiB
# fp32, so both dtypes get a padding gap before the next rank's chunk.
S, K, H, HI, EPN = 256, 4, 256, 384, 4
SHAPES = {"gate": (HI, H), "up": (HI, H), "down": (H, HI)}
NUM_SMS = 16
ROUTING_SKEW = 2.0  # lognormal sigma: hot experts, so the planner copies them
SENTINEL = -123.0

_all_gather_single = getattr(dist, "all_gather_single", None) or dist.all_gather_into_tensor


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _local_device_index() -> int:
    return int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", 0)))


def _setup():
    """Process group for the script entry; mirrors conftest's ``dist_env``."""
    torch.cuda.set_device(_local_device_index())
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    return dist.get_rank(), dist.get_world_size()


def _barrier():
    dist.barrier(device_ids=[torch.cuda.current_device()])


def _assert_all_ranks(errors, label):
    """Fail on every rank together (no rank is left waiting in a collective)."""
    for error in errors[:8]:
        print(f"[rank {dist.get_rank()}] {label}: {error}", flush=True)
    ok = torch.tensor([0 if errors else 1], dtype=torch.int32, device="cuda")
    dist.all_reduce(ok, op=dist.ReduceOp.MIN)
    assert int(ok.item()) == 1, f"{label} failed (per-rank messages above)"


def _make_buffer(R):
    return Buffer(S, H, K, R * EPN, R, num_sms=NUM_SMS, explicitly_destroy=True)


def _teardown(buffer, pools):
    try:
        if pools is not None:
            pools.destroy()
    finally:
        buffer.destroy()


def _dispatch_plan(buffer, rank, R, seed):
    """A real plan: dispatch S tokens on skewed routing (hot experts are
    shared by all ranks, so the planner copies them)."""
    topk, tpe = generate_topk_routing(
        S, K, R * EPN, R, ROUTING_SKEW, "cuda", seed, rank=rank)
    gen = torch.Generator(device="cuda").manual_seed(1000 * seed + rank)
    hidden = torch.randn(S, H, dtype=torch.bfloat16, device="cuda", generator=gen)
    _, _, _, plan = buffer.dispatch(hidden, None, topk, tpe)
    return plan, hidden


def _random_rows(seed, rank, name):
    out, inn = SHAPES[name]
    gen = torch.Generator(device="cuda").manual_seed(
        7919 * seed + 31 * rank + ExpertPools.PROJECTIONS.index(name))
    return torch.randn(EPN, out, inn, dtype=torch.bfloat16, device="cuda", generator=gen)


def _gather_rows(rows):
    """Every rank's [epn, ...] rows -> [R*epn, ...], indexed by global expert."""
    out = torch.empty((dist.get_world_size() * rows.shape[0], *rows.shape[1:]),
                      dtype=rows.dtype, device=rows.device)
    _all_gather_single(out, rows.contiguous())
    return out


def _flat_mapping(view):
    """The whole all-rank mapping behind a pool view, as one flat tensor."""
    numel = view.untyped_storage().nbytes() // view.element_size()
    return view.as_strided((numel,), (1,), 0)


def _own_gap(pool, rank):
    """The padded tail of this rank's chunk (after its 2*epn rows)."""
    chunk = pool.stride(0)
    return _flat_mapping(pool)[rank * chunk + 2 * pool[0].numel():(rank + 1) * chunk]


def _tag(rank, row):
    return float(16 * rank + row + 1)  # exact in bf16 and fp32


def _own_grad_values(rank, name, it):
    out, inn = SHAPES[name]
    expert = (rank * EPN + torch.arange(EPN, dtype=torch.float32, device="cuda")).view(EPN, 1, 1)
    row = torch.arange(out, dtype=torch.float32, device="cuda").view(1, out, 1)
    col = torch.arange(inn, dtype=torch.float32, device="cuda").view(1, 1, inn)
    return 1000.0 * (it + 1) + expert * 17.0 + row * 0.125 + col * 0.0078125


def _slot_grad_values(rank, name, it):
    out, inn = SHAPES[name]
    slot = torch.arange(EPN, dtype=torch.float32, device="cuda").view(EPN, 1, 1)
    row = torch.arange(out, dtype=torch.float32, device="cuda").view(1, out, 1)
    col = torch.arange(inn, dtype=torch.float32, device="cuda").view(1, 1, inn)
    offset = 3.0 * ExpertPools.PROJECTIONS.index(name) + 500.0 * it
    return offset + (rank + 1) * 101.0 + slot * 11.0 + row * 0.03125 + col * 0.00390625


def _check_slots(pools, rank, plan, gathered, label, unused=None):
    """Each slot this rank's plan row uses must hold the owner's row from
    ``gathered``; with ``unused`` given, the other slots must equal it."""
    etc = plan.experts_to_copy.cpu()
    errors = []
    for name in SHAPES:
        slots = pools.prefetch_buffer(name)[rank]
        for b in range(EPN):
            expert = int(etc[rank, b])
            if expert >= 0:
                want = gathered[name][expert]
            elif unused is not None:
                want = torch.full_like(slots[b], unused)
            else:
                continue
            if not torch.equal(slots[b], want):
                errors.append(f"{name} slot {b} (expert {expert}) mismatch")
    _assert_all_ranks(errors, label)
    return etc


def _push_and_check(buffer, pools, rank, R, seed, label):
    """Stage random rows, push a fresh plan, check every slot and the gap."""
    gathered = {}
    for name in SHAPES:
        pools.staging(name).copy_(_random_rows(seed, rank, name))
        pools.prefetch_buffer(name)[rank].fill_(SENTINEL)
        _own_gap(pools.prefetch_buffer(name), rank).fill_(SENTINEL)
        gathered[name] = _gather_rows(pools.staging(name).clone())
    # Peers push into this rank's slots only after it filled them.
    torch.cuda.synchronize()
    _barrier()

    plan, _ = _dispatch_plan(buffer, rank, R, seed)
    pools.push(plan)
    torch.cuda.synchronize()

    etc = _check_slots(pools, rank, plan, gathered, label, unused=SENTINEL)
    copies = int((etc >= 0).sum())
    errors = [] if copies > 0 else ["the skewed routing produced no expert copies"]
    for name in SHAPES:
        own = gathered[name][rank * EPN:(rank + 1) * EPN]
        if not torch.equal(pools.staging(name), own):
            errors.append(f"{name}: push changed the staging rows")
        if not bool((_own_gap(pools.prefetch_buffer(name), rank) == SENTINEL).all()):
            errors.append(f"{name}: push wrote the padding gap")
    _assert_all_ranks(errors, label)
    return copies


def _assertion_message(fn):
    try:
        fn()
    except AssertionError as exc:
        return str(exc)
    return None


# --------------------------------------------------------------------------
# tests
# --------------------------------------------------------------------------


def test_layout_and_aliasing(dist_env):
    rank, R = dist_env
    buffer = _make_buffer(R)
    pools = None
    try:
        pools = ExpertPools(buffer, dist.group.WORLD, SHAPES)
        granularity = int(get_vmm_granularity())
        errors = []

        def expect(cond, what):
            if not cond:
                errors.append(what)

        expect((pools.epn, pools.R, pools.rank) == (EPN, R, rank), "epn/R/rank attributes")
        for name, (out, inn) in SHAPES.items():
            rows = EPN * out * inn
            for kind, dtype, full, own, pool in (
                ("weights", torch.bfloat16, pools.weights(name),
                 pools.staging(name), pools.prefetch_buffer(name)),
                ("grads", torch.float32, pools.grads(name),
                 pools.local_grads(name), pools.reduce_buffer(name)),
            ):
                tag = f"{name}/{kind}"
                chunk = pool.stride(0)
                expect(full.dtype == own.dtype == pool.dtype == dtype, f"{tag}: dtype")
                expect(full.is_cuda and full.device.index == torch.cuda.current_device(),
                       f"{tag}: device")
                expect(tuple(full.shape) == (2 * EPN, out, inn) and full.is_contiguous(),
                       f"{tag}: compute view {tuple(full.shape)}")
                expect(tuple(own.shape) == (EPN, out, inn) and own.is_contiguous()
                       and own.data_ptr() == full.data_ptr(), f"{tag}: own-rows view")
                expect(tuple(pool.shape) == (R, EPN, out, inn), f"{tag}: pool shape")
                expect(tuple(pool.stride()) == (chunk, out * inn, inn, 1),
                       f"{tag}: pool strides {pool.stride()}")
                # One granularity-padded chunk per rank, with a gap for these shapes.
                expect(chunk * dtype.itemsize % granularity == 0
                       and 2 * rows < chunk <= 2 * rows + granularity // dtype.itemsize,
                       f"{tag}: rank stride {chunk} is not the padded chunk of {2 * rows}")
                expect(full.storage_offset() == rank * chunk and pool.storage_offset() == rows,
                       f"{tag}: storage offsets {full.storage_offset()}, {pool.storage_offset()}")
                expect(pool.untyped_storage().nbytes() == R * chunk * dtype.itemsize,
                       f"{tag}: mapping size")
                expect(pool[rank].data_ptr() == full[EPN].data_ptr(),
                       f"{tag}: pool[rank] does not alias rows [epn, 2*epn)")
                expect(all(pool[r].is_contiguous() for r in range(R)), f"{tag}: rank payloads")
                expect(full.data_ptr() % granularity == 0, f"{tag}: chunk start not VMM aligned")
                expect(all(t.data_ptr() % 16 == 0 for t in (full, own, pool, full[EPN:])),
                       f"{tag}: 16-byte alignment")
                try:
                    expect(_validate_rank_strided_pool(pool) == chunk, f"{tag}: pool validator")
                except AssertionError as exc:
                    errors.append(f"{tag}: _validate_rank_strided_pool rejected the pool: {exc}")
                own_chunk = _flat_mapping(pool)[rank * chunk:(rank + 1) * chunk]
                expect(int(torch.count_nonzero(own_chunk)) == 0, f"{tag}: chunk not zeroed")
        _assert_all_ranks(errors, "layout")

        # Data aliasing across ranks: tag every row of this rank's chunks,
        # then read every rank's slot rows through the all-rank views.
        for name in SHAPES:
            for full in (pools.weights(name), pools.grads(name)):
                for row in range(2 * EPN):
                    full[row].fill_(_tag(rank, row))
        torch.cuda.synchronize()
        _barrier()
        errors = []
        for name in SHAPES:
            for kind, pool in (("prefetch_buffer", pools.prefetch_buffer(name)),
                               ("reduce_buffer", pools.reduce_buffer(name))):
                for r in range(R):
                    for b in range(EPN):
                        if not bool((pool[r, b] == _tag(r, EPN + b)).all()):
                            errors.append(
                                f"{name} {kind}[{r}, {b}] is not rank {r}'s row {EPN + b}")
            for kind, own in (("staging", pools.staging(name)),
                              ("local_grads", pools.local_grads(name))):
                for row in range(EPN):
                    if not bool((own[row] == _tag(rank, row)).all()):
                        errors.append(f"{name} {kind}[{row}] lost its tag")
        _assert_all_ranks(errors, "cross-rank aliasing")
        if rank == 0:
            print(f"  [PASS] layout and aliasing: R={R}, epn={EPN}, shapes={SHAPES}, "
                  f"fabric handles={pools._use_fabric}")
    finally:
        _teardown(buffer, pools)


def test_push_real_plan(dist_env):
    rank, R = dist_env
    buffer = _make_buffer(R)
    pools = None
    try:
        pools = ExpertPools(buffer, dist.group.WORLD, SHAPES)
        for it, seed in enumerate((11, 12)):
            copies = _push_and_check(buffer, pools, rank, R, seed, f"push round {it}")
            if rank == 0:
                print(f"  [PASS] push round {it}: {copies} of {R * EPN} slots used")
    finally:
        _teardown(buffer, pools)


def test_reduce_real_plan(dist_env):
    rank, R = dist_env
    buffer = _make_buffer(R)
    pools = None
    try:
        pools = ExpertPools(buffer, dist.group.WORLD, SHAPES)
        lo, hi = rank * EPN, (rank + 1) * EPN
        for it, seed in enumerate((21, 22)):
            plan, _ = _dispatch_plan(buffer, rank, R, seed)
            etc = plan.experts_to_copy.cpu()
            for name in SHAPES:
                pools.local_grads(name).copy_(_own_grad_values(rank, name, it))
                pools.reduce_buffer(name)[rank].copy_(_slot_grad_values(rank, name, it))
                _own_gap(pools.reduce_buffer(name), rank).fill_(SENTINEL)
            # No host barrier: reduce() itself must order every rank's slot
            # writes before any rank reads them.
            pools.reduce(plan)
            torch.cuda.synchronize()

            errors = [] if int((etc >= 0).sum()) > 0 else ["no expert copies"]
            for name in SHAPES:
                # Exact fp32 sums (all values fit 24 bits), so order is irrelevant.
                expected = _own_grad_values(rank, name, it)
                for src in range(R):
                    slot_vals = _slot_grad_values(src, name, it)
                    for b in range(EPN):
                        expert = int(etc[src, b])
                        if lo <= expert < hi:
                            expected[expert - lo] += slot_vals[b]
                if not torch.equal(pools.local_grads(name), expected):
                    errors.append(f"{name}: local grads mismatch")
                mine = _slot_grad_values(rank, name, it)
                slots = pools.reduce_buffer(name)[rank]
                for b in range(EPN):
                    want = torch.zeros_like(mine[b]) if int(etc[rank, b]) >= 0 else mine[b]
                    if not torch.equal(slots[b], want):
                        errors.append(f"{name}: reduce slot {b} (expert {int(etc[rank, b])}) "
                                      "not zeroed / changed")
                if not bool((_own_gap(pools.reduce_buffer(name), rank) == SENTINEL).all()):
                    errors.append(f"{name}: reduce wrote the padding gap")
            _assert_all_ranks(errors, f"reduce round {it}")
            if rank == 0:
                print(f"  [PASS] reduce round {it}: {int((etc >= 0).sum())} consumed slots")
    finally:
        _teardown(buffer, pools)


def test_two_layers_share_pools(dist_env):
    rank, R = dist_env
    buffer = _make_buffer(R)
    pools = None
    try:
        pools = ExpertPools(buffer, dist.group.WORLD, SHAPES)
        # Two layers' persistent experts in ordinary memory, and every rank's
        # rows of each (to know what a slot must hold).
        params, gathered = {}, {}
        for layer, seed in (("A", 31), ("B", 32)):
            params[layer] = {name: _random_rows(seed, rank, name) for name in SHAPES}
            gathered[layer] = {name: _gather_rows(p) for name, p in params[layer].items()}
        for name in SHAPES:
            pools.prefetch_buffer(name)[rank].fill_(SENTINEL)
        torch.cuda.synchronize()
        _barrier()

        def stage(layer):
            for name in SHAPES:
                pools.staging(name).copy_(params[layer][name])

        # Forward of layer A, then of layer B, as a model runs them: stage,
        # dispatch, push. Each dispatch's cross-rank barriers order the push
        # after every rank's reads of the previous layer's slots.
        stage("A")
        plan_a, hidden_a = _dispatch_plan(buffer, rank, R, seed=33)
        pools.push(plan_a)
        torch.cuda.synchronize()
        etc_a = _check_slots(pools, rank, plan_a, gathered["A"], "layer A forward")
        stage("B")
        plan_b, _ = _dispatch_plan(buffer, rank, R, seed=34)
        pools.push(plan_b)
        torch.cuda.synchronize()
        etc_b = _check_slots(pools, rank, plan_b, gathered["B"], "layer B forward")
        both = int(((etc_a >= 0) & (etc_b >= 0)).sum())
        _assert_all_ranks([] if both > 0 else ["the plans share no slot"], "slot overlap")

        # Backward of layer A: combine bwd re-dispatches with A's saved plan;
        # the pools hold B's copies, so re-stage A and re-push its old plan.
        buffer.dispatch(hidden_a, plan=plan_a)
        stage("A")
        pools.push(plan_a)
        torch.cuda.synchronize()
        _check_slots(pools, rank, plan_a, gathered["A"], "layer A re-push")
        errors = []
        for name in SHAPES:
            if not torch.equal(pools.staging(name), params["A"][name]):
                errors.append(f"{name}: staging rows are not layer A's")
            slots = pools.prefetch_buffer(name)[rank]
            for b in range(EPN):
                expert_a, expert_b = int(etc_a[rank, b]), int(etc_b[rank, b])
                # The re-push touches only A's slots; B-only slots keep B's copy.
                if expert_a < 0 and expert_b >= 0 and \
                        not torch.equal(slots[b], gathered["B"][name][expert_b]):
                    errors.append(f"{name}: B-only slot {b} changed by A's re-push")
        _assert_all_ranks(errors, "layer A re-push")
        if rank == 0:
            print(f"  [PASS] two layers: {both} slots used by both plans restored by re-push")
    finally:
        _teardown(buffer, pools)


def test_destroy_is_idempotent(dist_env):
    rank, R = dist_env
    buffer = _make_buffer(R)
    pools = None
    try:
        pools = ExpertPools(buffer, dist.group.WORLD, SHAPES)
        pools.destroy()
        pools.destroy()  # no-op, no collective
        errors = [] if pools.destroyed else ["destroyed is False after destroy()"]
        for label, call in (
            ("weights", lambda: pools.weights("gate")),
            ("reduce_buffer", lambda: pools.reduce_buffer("down")),
            ("push", lambda: pools.push(None)),
            ("reduce", lambda: pools.reduce(None)),
        ):
            msg = _assertion_message(call)
            if msg is None or "destroyed" not in msg:
                errors.append(f"{label} after destroy(): {msg!r}")
        _assert_all_ranks(errors, "destroy")

        # The Buffer is unaffected and new pools on it work.
        pools = ExpertPools(buffer, dist.group.WORLD, SHAPES)
        _push_and_check(buffer, pools, rank, R, seed=41, label="push after re-create")
        if rank == 0:
            print("  [PASS] destroy is idempotent; new pools on the same Buffer work")
    finally:
        _teardown(buffer, pools)


def test_rejects_bad_arguments(dist_env):
    rank, R = dist_env
    buffer = _make_buffer(R)
    try:
        # All of these fail before any collective, identically on every rank.
        cases = (
            ({"gate": (HI, H), "up": (HI, H)}, "gate/up/down"),
            ({**SHAPES, "shared": (H, HI)}, "gate/up/down"),
            ({"gate": (HI, H), "up": (HI, 200), "down": (H, HI)}, "multiples of 128"),
            ({"gate": (HI, H), "up": (HI, H), "down": (H,)}, "(out, in) pair"),
        )
        errors = []
        for shapes, needle in cases:
            msg = _assertion_message(lambda: ExpertPools(buffer, dist.group.WORLD, shapes))
            if msg is None or needle not in msg:
                errors.append(f"{sorted(shapes)}: expected {needle!r}, got {msg!r}")
        if R >= 2:
            half = dist.new_group(ranks=list(range(R // 2)))
            msg = _assertion_message(lambda: ExpertPools(buffer, half, SHAPES))
            if msg is None or "group" not in msg:
                errors.append(f"half group: got {msg!r}")
        _assert_all_ranks(errors, "argument validation")
        if rank == 0:
            print("  [PASS] bad shapes and groups are rejected")
    finally:
        buffer.destroy()


if __name__ == "__main__":
    env = _setup()
    tests = [fn for name, fn in list(globals().items())
             if name.startswith("test_") and callable(fn)]
    for test in tests:
        if env[0] == 0:
            print(f"[test_expert_pools] {test.__name__}", flush=True)
        test(env)
    if env[0] == 0:
        print(f"[test_expert_pools] {len(tests)} passed", flush=True)
    torch.cuda.synchronize()
    _barrier()
    dist.destroy_process_group()
