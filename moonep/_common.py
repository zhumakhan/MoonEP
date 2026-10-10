"""
MoonEP shared CuTe DSL helpers (common to dispatch + combine).

The core works around two cp.async.bulk lowering limitations of
nvidia-cutlass-dsl 4.4.2:
1) the TMA descriptor of ``CopyBulkTensorTileG2SOp`` caps a single-dim box at
   256 elements, so one H=7168 bf16 row is split into 28 UTMA instructions;
2) the plain form of ``CopyBulkG2SOp`` force-inserts mapa.shared::cluster,
   which mismatches the mbar that PipelineTmaAsync allocates in shared::cta,
   causing a runtime hang.

We issue cp.async.bulk PTX directly via ``llvm.inline_asm``, matching the
semantics of CUTLASS C++ ``cute::SM90_BULK_COPY_{G2S,S2G}::copy``.

Note: cp.async.bulk is a single-thread instruction; callers must wrap the
inline asm in ``if cute.arch.lane_idx() == 0:``.
"""

import cutlass
from cutlass import Int32, Int64, Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op
import cutlass.cute as cute


# ============================================================================
# Cross-rank barrier constants and helpers
# ============================================================================

# grid_sync sentinel: per-round whole-grid increments sum to 0x80000000; a flipped top bit means all arrived.
GRID_SYNC_TAG = 0x80000000

# Same value as GRID_SYNC_TAG
WARP_SYNC_TAG = 0x80000000

# spin-wait timeout. Fail-fast on timeout to avoid further polluting the barrier state.
BARRIER_TIMEOUT_CYCLES = 100 * 2_000_000_000


def _leader_inc(tag, n):
    """The leader's arrive increment ``tag - (n - 1)`` as Int32 bits, so the
    n arrivals total ``tag``. Computed in Int64 and truncated: in Int32,
    ``Int32(tag)`` is already -2**31, so a constant ``n`` makes the DSL's
    constant fold overflow (same bits, but a TYPE_INT_LITERAL_OUT_OF_RANGE
    warning on every compile)."""
    return Int32(Int64(tag) - (n - 1))


@dsl_user_op
def clock64(*, loc=None, ip=None) -> Int64:
    """Corresponds to PTX ``mov.u64 ret, %clock64;``."""
    return Int64(
        llvm.inline_asm(
            T.i64(),
            [],
            "mov.u64 $0, %clock64;",
            "=l",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def device_trap(*, loc=None, ip=None) -> None:
    """Corresponds to PTX ``trap;``; used for device-side fail-fast."""
    llvm.inline_asm(
        None,
        [],
        "trap;",
        "",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def atom_add_release_gpu(ptr_i64, val, *, loc=None, ip=None) -> Int32:
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [ptr_i64, Int32(val).ir_value(loc=loc, ip=ip)],
            "atom.add.release.gpu.global.s32 $0, [$1], $2;",
            "=r,l,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def atom_add_relaxed_gpu_s32(ptr_i64, val, *, loc=None, ip=None) -> Int32:
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [ptr_i64, Int32(val).ir_value(loc=loc, ip=ip)],
            "atom.add.relaxed.gpu.global.s32 $0, [$1], $2;",
            "=r,l,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def atom_min_relaxed_gpu_s32(ptr_i64, val, *, loc=None, ip=None) -> Int32:
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [ptr_i64, Int32(val).ir_value(loc=loc, ip=ip)],
            "atom.min.relaxed.gpu.global.s32 $0, [$1], $2;",
            "=r,l,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )

@dsl_user_op
def atom_or_relaxed_gpu_s32(ptr_i64, val, *, loc=None, ip=None) -> Int32:
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [ptr_i64, Int32(val).ir_value(loc=loc, ip=ip)],
            "atom.or.relaxed.gpu.global.b32 $0, [$1], $2;",
            "=r,l,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def atom_or_relaxed_gpu_b32(ptr_i64, val, *, loc=None, ip=None) -> Uint32:
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [ptr_i64, Uint32(val).ir_value(loc=loc, ip=ip)],
            "atom.or.relaxed.gpu.global.b32 $0, [$1], $2;",
            "=r,l,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def popc_b32(val, *, loc=None, ip=None) -> Int32:
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [Uint32(val).ir_value(loc=loc, ip=ip)],
            "popc.b32 $0, $1;",
            "=r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def ctz_b32(val, *, loc=None, ip=None) -> Int32:
    """Count trailing zeros (bit position of lowest set bit).

    PTX ``brev.b32`` + ``clz.b32``: bit-reverse first, then count leading zeros
    = count trailing zeros. Returns 32 for input 0; the caller must guarantee
    non-zero input or handle that boundary.
    """
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [Uint32(val).ir_value(loc=loc, ip=ip)],
            "brev.b32 $0, $1; clz.b32 $0, $0;",
            "=r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def ld_acquire_gpu_s32(ptr_i64, *, loc=None, ip=None) -> Int32:
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [ptr_i64],
            "ld.acquire.gpu.global.s32 $0, [$1];",
            "=r,l",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def red_add_release_sys(ptr_i64, val, *, loc=None, ip=None) -> None:
    """Corresponds to PTX ``red.release.sys.global.add.s32 [ptr], val;`` (no return value)."""
    llvm.inline_asm(
        None,
        [ptr_i64, Int32(val).ir_value(loc=loc, ip=ip)],
        "red.release.sys.global.add.s32 [$0], $1;",
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def ld_acquire_sys_s32(ptr_i64, *, loc=None, ip=None) -> Int32:
    """Corresponds to PTX ``ld.acquire.sys.s32 ret, [ptr];``."""
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [ptr_i64],
            "ld.acquire.sys.global.s32 $0, [$1];",
            "=r,l",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@cute.jit
def grid_sync(bar_ptr, nsm: Int32, tid: Int32):
    """Self-resetting cooperative grid barrier (modeled on DeepGEMM ``grid_sync``).

    ``bar_ptr[0]`` is the arrive counter: each round SM0 adds ``TAG-(nsm-1)``
    and every other SM adds 1, totaling exactly +0x80000000, so the low 31 bits
    stay 0 and no reset is needed; a flipped top bit means all have arrived.
    The fence is just ``bar.sync`` (=sync_threads); cross-SM ordering is
    guaranteed by release/acquire atomics.
    """
    cute.arch.sync_threads()
    if tid == 0:
        b0 = bar_ptr.toint().ir_value()
        pid = cute.arch.block_idx()[0]
        inc = Int32(1)
        if pid == 0:
            inc = _leader_inc(GRID_SYNC_TAG, nsm)
        old = atom_add_release_gpu(b0, inc)
        done = cutlass.Boolean(False)
        while not done:
            new = ld_acquire_gpu_s32(b0)
            done = ((new ^ old) & GRID_SYNC_TAG) != 0
    cute.arch.sync_threads()


@cute.jit
def cross_rank_barrier(
    meta_buf: cute.Tensor,        # int32 view of the merged meta_buf
    meta_stride: Int32,           # meta_chunk_padded
    barrier_off: Int32,           # BARRIER_OFF
    rank: Int32,
    num_ranks: cutlass.Constexpr[int],
    bar_ptr,
    nsm: Int32,
    num_threads: cutlass.Constexpr[int],
    tid: Int32,
):
    """Self-resetting cross-rank barrier (modeled on DeepGEMM ``nvlink_barrier``).

    grid_sync -> only block0 sends the cross-rank +/-1 signal -> grid_sync.
    Each rank reuses 3 slots in the ``barrier_off`` region: ``+0/+1`` are the
    two phase signals (sent by peers), ``+2`` is the local phase/sign state.
    A single atomic with alternating phase/sign self-resets, so an all-zero
    initial state is already correct. The fence is always ``bar.sync``.

    Proxy-bridge contract (all threads execute both fences):
    - Entry ``fence.proxy.alias``: writer-side bridge for the VMM alias proxy.
      A thread's prior ``multimem`` accesses (mc VA) are made coherent with
      unicast-VA accesses before the barrier release publishes them, so peers
      may read the same physical memory through the unicast alias afterwards.
    - Exit ``fence.proxy.async.global``: consumer-side bridge for the async
      proxy. Data acquired into the generic proxy by this barrier becomes
      visible to subsequent TMA (``cp.async.bulk``) accesses issued by the
      executing thread.
    """
    assert num_threads >= num_ranks, (
        "cross_rank_barrier requires blockDim.x >= num_ranks: "
        f"num_threads={num_threads}, num_ranks={num_ranks}"
    )
    # Writer side of the alias bridge: must precede the barrier release so the
    # multicast stores are published in a state coherent for unicast readers.
    cute.arch.fence_proxy("alias")
    grid_sync(bar_ptr, nsm, tid)
    if cute.arch.block_idx()[0] == 0:
        status = meta_buf[rank * meta_stride + barrier_off + 2] & 3
        phase = status & 1
        sign = status >> 1
        # Each rank sends +/-1 to the peer's current phase slot; sign picks the direction for self-reset.
        if tid < num_ranks:
            peer = (meta_buf.iterator + (tid * meta_stride + barrier_off + phase)).toint()
            delta = Int32(1)
            if sign != 0:
                delta = Int32(-1)
            red_add_release_sys(peer.ir_value(), delta)
        cute.arch.sync_threads()
        if tid == 0:
            cute.arch.atomic_add(meta_buf.iterator + (rank * meta_stride + barrier_off + 2), Int32(1), scope="gpu")
            target = Int32(num_ranks)
            if sign != 0:
                target = Int32(0)
            own = (meta_buf.iterator + (rank * meta_stride + barrier_off + phase)).toint()
            done = cutlass.Boolean(False)
            start = clock64()
            while not done:
                cur = ld_acquire_sys_s32(own.ir_value())
                done = cur == target
                if (clock64() - start) >= Int64(BARRIER_TIMEOUT_CYCLES):
                    cute.printf(
                        "MoonEP cross_rank_barrier timeout (100s), trapping: "
                        "rank=%d barrier_off=%d phase=%d sign=%d signal=%d target=%d\n",
                        rank, barrier_off, phase, sign, cur, target,
                    )
                    device_trap()
    grid_sync(bar_ptr, nsm, tid)
    # Consumer side of the async bridge: must follow the barrier acquire so
    # peer stores now visible in the generic proxy are bridged to this
    # thread's subsequent TMA (cp.async.bulk) accesses.
    cute.arch.fence_proxy("async.global")


@cute.jit
def cross_warp_sync(
    bar_ptr,
    nparticipants: cutlass.Constexpr[int],
    leader: Int32,
):
    """All-participating-warp barrier: lane 0 of each participating warp
    arrives once and spin-waits until all have arrived. leader must be non-zero
    on exactly one participating warp (it contributes the large increment that
    makes the barrier self-resetting). Participants may span multiple SMs x
    multiple builder warps."""
    assert nparticipants > 0, (
        "cross_warp_sync requires at least one participating warp"
    )
    assert nparticipants < WARP_SYNC_TAG, (
        "cross_warp_sync participant count must be smaller than WARP_SYNC_TAG: "
        f"nparticipants={nparticipants}, tag={WARP_SYNC_TAG}"
    )
    # 1. Intra-warp sync: writes by the 32 lanes of this warp become visible to each other
    cute.arch.sync_warp()

    b0 = bar_ptr.toint().ir_value()
    lane = cute.arch.lane_idx()

    # 2. Release arrive
    inc = Int32(1)
    if leader != 0:
        inc = _leader_inc(WARP_SYNC_TAG, nparticipants)   # self-reset: total increment = TAG

    # 3. ALL builder warps acquire-wait (not just bidx == 0)
    old = Int32(0)
    if lane == 0:
        old = atom_add_release_gpu(b0, inc)
        done = cutlass.Boolean(False)
        start = clock64()
        while not done:
            new = ld_acquire_gpu_s32(b0)
            done = ((new ^ old) & WARP_SYNC_TAG) != 0
            if (clock64() - start) >= Int64(BARRIER_TIMEOUT_CYCLES):
                cute.printf("MoonEP builder barrier timeout, trapping\n")
                device_trap()
    cute.arch.sync_warp()      # broadcast completion to all 32 lanes


@cute.jit
def pdl_trigger_dependents(tid: Int32):
    """Publish this CTA's writes, then trigger programmatic dependent launch."""
    cute.arch.sync_threads()
    cute.arch.fence_acq_rel_sys()
    cute.arch.sync_threads()
    if tid == 0:
        cute.arch.griddepcontrol_launch_dependents()


@cute.jit
def pdl_wait_predecessor():
    """Wait until the predecessor grid has completed and flushed its writes."""
    cute.arch.griddepcontrol_wait()


# ============================================================================
# cp.async.bulk inline-PTX wrappers
# ============================================================================


@dsl_user_op
def cp_async_bulk_g2s(
    dst_smem_i32, src_gmem_i64, size_i32, mbar_smem_i32,
    *, loc=None, ip=None,
) -> None:
    """Corresponds to PTX ``cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes``.

    On completion the copy automatically transaction-arrives ``size_i32``
    bytes on the mbar. Single-thread instruction; must be called by the
    elected lane.
    """
    llvm.inline_asm(
        None,
        [dst_smem_i32, src_gmem_i64, size_i32, mbar_smem_i32],
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes "
        "[$0], [$1], $2, [$3];",
        "r,l,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc, ip=ip,
    )


@dsl_user_op
def cp_async_bulk_s2g(
    src_smem_i32, dst_gmem_i64, size_i32,
    *, loc=None, ip=None,
) -> None:
    """Corresponds to PTX ``cp.async.bulk.global.shared::cta.bulk_group [dst], [src], size;``.

    bulk_group form -- completion is tracked via ``cp_async_bulk_commit_group``
    / ``cp_async_bulk_wait_group<N>``; no mbar. Also a single-thread
    instruction.
    """
    llvm.inline_asm(
        None,
        [dst_gmem_i64, src_smem_i32, size_i32],
        "cp.async.bulk.global.shared::cta.bulk_group "
        "[$0], [$1], $2;",
        "l,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc, ip=ip,
    )
