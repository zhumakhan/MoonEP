"""
MoonEP Combine — CuTe DSL implementation.

3-stage warp-specialized G2S / fp32 ACC / S2G with cp.async.bulk per-row TMA,
expressed entirely in the CUTLASS Python DSL. Hidden size H is a JIT constexpr;
the load-pipeline depth (kStagesL) is computed automatically inside the
host-side jit function from H plus the device's per-block opt-in dynamic smem
cap, so callers do not need to dispatch on stage count manually.
``launch_combine`` pulls back only the deduped payload rows (``dst >= 0``);
duplicate entries are pre-reduced into their primary slot by the combine
prologue on the destination rank (``moonep/combine_prologue.py``).
The dedup path runs a cross-rank publish barrier at combine entry before
reading peer ranks' staged NVL rows.
"""

import functools

import torch
import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
from cutlass import BFloat16, Float32, Int32, Int64
from cutlass.cute.runtime import make_ptr

from moonep._common import (
    cp_async_bulk_g2s as _cp_async_bulk_g2s,
    cp_async_bulk_s2g as _cp_async_bulk_s2g,
    cross_rank_barrier,
    pdl_wait_predecessor,
)


# ============================================================================
# Combine kernel
# ============================================================================

class CombineKernel:
    """3-stage warp-specialized combine.

    Layout:
      - 6 warps per CTA: 1 G2S producer (warp 0), 4 fp32 ACC consumers
        (warps 1-4), 1 S2G consumer (warp 5).
      - Load pipeline (PipelineTmaAsync, kStagesL stages) hands one bf16 row
        at a time to the ACC warps. kStagesL is auto-picked at JIT time.
      - Output pipeline (PipelineAsync, 2 stages, fixed) hands the bf16 output
        row to warp 5, which TMA-stores it back to gmem.
      - The 4 ACC warps share a NamedBarrier (id=0) for sub-block sync without
        stalling G2S/S2G.

    Dedup contract: non-negative ``dst`` entries pull one NVL row and
    accumulate it. Negative entries encode the same raw destination as
    ``-raw_dst - 1`` for duplicate top-k entries: their contribution was
    pre-reduced into the primary slot by the combine prologue on the
    destination rank, so the hidden path skips them entirely; the weight
    gather decodes the raw slot and reads it as usual (route weights are
    per-topk values, never deduplicated). The cross-rank publish barrier runs
    at combine entry before remote rows are consumed.
    """

    ACC_THREADS = 128             # warps 1..4
    STAGES_O = 2

    def __init__(
        self,
        H: int,
        R: int,
        S: int,
        K: int,
        NvS: int,
        NvS_padded: int,
        meta_stride: int,
        num_sms: int,
        with_weights: bool,
        smem_budget: int,
        pdl_launch: bool,
    ):
        self.H = H
        self.R = R
        self.S = S
        self.K = K
        self.NvS = NvS
        self.NvS_padded = NvS_padded
        self.meta_stride = meta_stride
        self.num_sms = num_sms
        self.with_weights = with_weights
        self.pdl_launch = pdl_launch
        self.stages_l = self._pick_stages_l(H, smem_budget)
        self.num_warps = 6 if not with_weights else 7
        self.num_threads = 32 * self.num_warps
        if self.stages_l == 0:
            raise RuntimeError(
                f"combine: H={H} too large for per-block smem budget "
                f"{smem_budget} B (need at least "
                f"{self._smem_bytes(H, 2)} B)"
            )

    @classmethod
    def _smem_bytes(cls, H: int, stages_l: int) -> int:
        # stage_smem (bf16, kStagesL deep) + out_smem (bf16, STAGES_O deep)
        # + 2 mbarriers per stage for the load pipe and the out pipe.
        # The fp32 accumulator lives entirely in registers — no smem cost.
        # Pad each tensor to 128 B for the byte_alignment we ask SmemAllocator
        # for, plus 256 B headroom for any internal cutlass-allocated state.
        def _round_up(n: int, a: int) -> int:
            return (n + a - 1) // a * a
        return (
            _round_up(stages_l * H * 2, 128)
            + _round_up(cls.STAGES_O * H * 2, 128)
            + _round_up(stages_l * 2 * 8, 16)
            + _round_up(cls.STAGES_O * 2 * 8, 16)
            + 256
        )

    @classmethod
    def _pick_stages_l(cls, H: int, smem_budget: int) -> int:
        for s in (16, 14, 12, 10, 8, 6, 4, 2):
            if cls._smem_bytes(H, s) <= smem_budget:
                return s
        return 0

    # ------------------------------------------------------------------ host

    @cute.jit
    def __call__(
        self,
        # Tensors / pointers
        output_ptr: cute.Pointer,        # bf16 [S, H]
        output_sk_ptr: cute.Pointer,     # int32 view of fp32 [S, K] (or placeholder)
        hidden_ptr: cute.Pointer,        # bf16 [R*NvS_padded, H]
        meta_ptr: cute.Pointer,          # int32 [R*meta_chunk_padded]
        dst_ptr: cute.Pointer,           # int32 [N=S*K]
        bar_ptr: cute.Pointer,           # int32 [1] grid barrier counter
        # Scalars
        rank: Int32,
        weights_off: Int32,
        barrier_off: Int32,
        num_tokens: Int32, # this step's tokens per rank,  <= S
        stream: cuda.CUstream,
    ):
        H = cutlass.const_expr(self.H)
        stages_l = cutlass.const_expr(self.stages_l)
        S = cutlass.const_expr(self.S)
        K = cutlass.const_expr(self.K)
        NvS_padded = cutlass.const_expr(self.NvS_padded)
        meta_stride = cutlass.const_expr(self.meta_stride)
        R = cutlass.const_expr(self.R)

        # 2-D row-major views with int64 stride. R*NvS_padded*H can exceed 2^31
        # in production, so the row-offset arithmetic `srow * H` must be
        # evaluated as int64. The C++ port does the same with
        # `make_shape((int64_t)R * NvS_padded, H64)`.
        in_rows = R * NvS_padded
        H64 = cutlass.Int64(H)
        gmem_in = cute.make_tensor(
            hidden_ptr,
            cute.make_layout((in_rows, H), stride=(H64, cutlass.Int64(1))),
        )
        gmem_out = cute.make_tensor(
            output_ptr,
            cute.make_layout((S, H), stride=(H64, cutlass.Int64(1))),
        )

        # No atoms needed: producer/consumer issue cp.async.bulk via inline asm.

        # dst / meta stay as plain global tensors (read with normal loads).
        dst_tensor = cute.make_tensor(
            dst_ptr, cute.make_layout((S * K,))
        )
        meta_tensor = cute.make_tensor(
            meta_ptr, cute.make_layout((R * meta_stride,))
        )
        # output_sk: when with_weights=False the kernel never reads it; the
        # caller passes a placeholder. We only need a sized tensor when we do.
        if cutlass.const_expr(self.with_weights):
            sk_tensor = cute.make_tensor(
                output_sk_ptr, cute.make_layout((S * K,))
            )
        else:
            sk_tensor = cute.make_tensor(
                output_sk_ptr, cute.make_layout((1,))
            )

        bar_tensor = cute.make_tensor(bar_ptr, cute.make_layout((1,)))

        smem_bytes = self._smem_bytes(H, stages_l)

        self.kernel(
            gmem_in,
            gmem_out,
            dst_tensor,
            meta_tensor,
            sk_tensor,
            bar_tensor,
            rank,
            weights_off,
            barrier_off,
            num_tokens,
        ).launch(
            grid=(self.num_sms, 1, 1),
            block=(self.num_threads, 1, 1),
            smem=smem_bytes,
            stream=stream,
            cooperative=True,
            use_pdl=self.pdl_launch,
        )

    # ----------------------------------------------------------------- kernel

    @cute.kernel
    def kernel(
        self,
        gmem_in: cute.Tensor,
        gmem_out: cute.Tensor,
        dst_tensor: cute.Tensor,
        meta_tensor: cute.Tensor,
        sk_tensor: cute.Tensor,
        bar_tensor: cute.Tensor,
        rank: Int32,
        weights_off: Int32,
        barrier_off: Int32,
        num_tokens: Int32,
    ):
        H = cutlass.const_expr(self.H)
        S = cutlass.const_expr(self.S)
        K = cutlass.const_expr(self.K)
        NvS = cutlass.const_expr(self.NvS)
        NvS_padded = cutlass.const_expr(self.NvS_padded)
        meta_stride = cutlass.const_expr(self.meta_stride)
        stages_l = cutlass.const_expr(self.stages_l)
        STAGES_O = cutlass.const_expr(self.STAGES_O)
        num_ranks = cutlass.const_expr(self.R)
        num_threads = cutlass.const_expr(self.num_threads)

        if cutlass.const_expr(self.pdl_launch):
            pdl_wait_predecessor()

        # ----- entry barrier: every peer rank's combine has reached us。
        tidx0 = cute.arch.thread_idx()[0]
        cross_rank_barrier(
            meta_tensor, meta_stride, barrier_off, rank, num_ranks,
            bar_tensor.iterator, Int32(self.num_sms), num_threads, tidx0,
        )

        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        # ----- shared memory layout
        smem = utils.SmemAllocator()

        # Mbarrier blocks first (16-byte alignment is plenty).
        load_mbar = smem.allocate_array(Int64, num_elems=2 * stages_l)
        out_mbar = smem.allocate_array(Int64, num_elems=2 * STAGES_O)

        stage_smem = smem.allocate_tensor(
            BFloat16,
            cute.make_ordered_layout((H, stages_l), order=(0, 1)),
            byte_alignment=128,
        )
        out_smem = smem.allocate_tensor(
            BFloat16,
            cute.make_ordered_layout((H, STAGES_O), order=(0, 1)),
            byte_alignment=128,
        )

        # ----- pipelines
        # load pipe (TMA G2S):
        #   Producer = warp 0 (32 threads). Producer_acquire's full-mbar arrive
        #   uses elect_one internally → producer_group size = 1.
        #   Consumer = warps 1..4 (128 threads). PipelineTmaAsync.consumer_release
        #   guards arrive with `is_signalling_thread` (= lane 0 of each warp in
        #   the non-cluster case), so 4 arrivals → consumer_group size = 4.
        load_pipe = pipeline.PipelineTmaAsync.create(
            barrier_storage=load_mbar,
            num_stages=stages_l,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 4),
            tx_count=H * 2,  # bf16 row in bytes
        )

        # out pipe (PipelineAsync, plain mbar handshake):
        #   Producer = warps 1..4 (128 threads). All 128 call producer_commit
        #   (count-based arrive_mbarrier) → size = 128.
        #   Consumer = warp 5 (32 threads). All 32 call consumer_release
        #   (count-based arrive) → size = 32.
        out_pipe = pipeline.PipelineAsync.create(
            barrier_storage=out_mbar,
            num_stages=STAGES_O,
            producer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, self.ACC_THREADS
            ),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 32),
        )

        # User NamedBarrier id=0 maps to HW id 8 (id 0..7 are reserved by the
        # CUTLASS pipelines / __syncthreads). 4 ACC warps = 128 threads.
        acc_bar = pipeline.NamedBarrier(barrier_id=8, num_threads=self.ACC_THREADS)

        # ------ per-block token range over this step's num_tokens (<=S)
        # When num_tokens < num_sms, trailing blocks start past the end;
        # clamp so n_tok never goes negative.
        tpb = (num_tokens + self.num_sms - 1) // self.num_sms
        s_beg = bidx * tpb
        s_end = cutlass.max(cutlass.min(s_beg + tpb, num_tokens), s_beg)
        n_tok = s_end - s_beg

        # ============================================
        # Warp 0 — G2S producer
        # ============================================
        if warp_idx == 0:
            load_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, stages_l
            )
            H_BYTES = cutlass.const_expr(H * 2)  # bf16 row size
            for s_local in cutlass.range(n_tok, unroll=1):
                s = s_beg + s_local
                sK = s * K
                for k in cutlass.range(K, unroll=1):
                    dst_val = dst_tensor[sK + k]
                    # Negative dst marks a duplicate entry: its contribution
                    # was pre-reduced into the primary slot on the destination
                    # rank, so no payload row is loaded for it (the raw slot
                    # is only ever decoded by the warp-6 weight gather).
                    load_token = dst_val >= Int32(0)

                    if load_token:
                        drank = dst_val // NvS
                        loff = dst_val % NvS
                        srow = drank * NvS_padded + loff

                        load_pipe.producer_acquire(load_state)

                        # Issue cp.async.bulk via inline asm — see _cp_async_bulk_g2s.
                        # cp.async.bulk is a single-thread instruction; only lane 0
                        # of warp 0 must execute it (otherwise 32 lanes each start a
                        # copy and the mbar transaction-arrive over-decrements).
                        if cute.arch.lane_idx() == 0:
                            g_row_int = (
                                gmem_in.iterator + cutlass.Int64(srow) * cutlass.Int64(H)
                            ).toint()
                            s_row_int = (
                                stage_smem.iterator + load_state.index * H
                            ).toint()
                            mbar_int = load_pipe.producer_get_barrier(load_state).toint()
                            _cp_async_bulk_g2s(
                                s_row_int.ir_value(),
                                g_row_int.ir_value(),
                                Int32(H_BYTES).ir_value(),
                                mbar_int.ir_value(),
                            )

                        load_state.advance()

        # ============================================
        # Warps 1..4 — fp32 ACC consumers (accumulate in registers)
        # ============================================
        elif warp_idx >= 1 and warp_idx <= 4:
            acc_tid = tidx - 32  # 0..127

            # Each ACC thread owns H / ACC_THREADS fp32 registers; the j-th
            # register covers element idx = j * ACC_THREADS + acc_tid. The
            # host-side assert (H % ACC_THREADS == 0) lets us drop the
            # `idx < H` predicate, which otherwise blocks the codegen from
            # software-pipelining LDS→SHF→FADD across j iterations.
            H_PER_THR = cutlass.const_expr(H // self.ACC_THREADS)
            acc_reg = cute.make_rmem_tensor(
                cute.make_layout((H_PER_THR,)), Float32
            )

            load_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, stages_l
            )
            out_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, STAGES_O
            )

            # Two-level loop over (s_local, k): the K-cycle prologue
            # (acquire+zero) and epilogue (convert+commit) hoist outside,
            # letting the codegen drop the `if k == 0` / `if k == K-1`
            # branches and the `s_local = li // K` lowering.
            for s_local in cutlass.range(n_tok, unroll=1):
                s = s_beg + s_local
                sK = s * K
                stg_o = s_local % STAGES_O

                # Prologue: acquire next output slot + zero accumulator regs.
                # Thread-local — no cross-warp barrier needed.
                out_pipe.producer_acquire(out_state)
                for j in cutlass.range_constexpr(H_PER_THR):
                    acc_reg[j] = Float32(0.0)

                for k in cutlass.range(K, unroll=1):
                    dst_val = dst_tensor[sK + k]
                    # Same skip predicate as the producer (both read the same
                    # dst value), so wait/release stay stage-aligned with the
                    # issued loads: per token both sides step exactly once per
                    # non-negative entry. The predicate is uniform across all
                    # 128 ACC threads, keeping the acc_bar counts consistent.
                    accumulate_token = dst_val >= Int32(0)

                    if accumulate_token:
                        # Wait for this stage's TMA load to complete.
                        load_pipe.consumer_wait(load_state)

                        # fp32 accumulate: acc_reg[j] += bf16-as-f32(stage[idx])
                        for j in cutlass.range_constexpr(H_PER_THR):
                            idx = j * self.ACC_THREADS + acc_tid
                            v = Float32(stage_smem[idx, load_state.index])
                            acc_reg[j] = acc_reg[j] + v

                        # Weight gather lives in standalone warp6 (see below):
                        # keeping it on a single ACC thread serialized one global
                        # LDG per K-iter onto the critical path and capped c_bwd
                        # at a small fraction of the weights-free path.

                        # ACC warps must finish reading stage_smem before
                        # consumer_release lets the producer overwrite it.
                        # consumer_release signals from each warp's lane 0 only
                        # (PipelineTmaAsync signalling thread; consumer_group=4),
                        # so a warp-level sync is enough to make that arrive cover
                        # all 32 lanes' reads — cross-warp completion is already
                        # gated by the 4-count empty mbarrier.
                        cute.arch.fence_view_async_shared()
                        cute.arch.sync_warp()
                        load_pipe.consumer_release(load_state)
                        load_state.advance()

                # Epilogue: convert fp32 acc → bf16 out and notify warp 5.
                for j in cutlass.range_constexpr(H_PER_THR):
                    idx = j * self.ACC_THREADS + acc_tid
                    out_smem[idx, stg_o] = BFloat16(acc_reg[j])
                # All 128 ACC threads must finish writing out_smem before
                # warp 5's TMA store proxy can read it.
                acc_bar.arrive_and_wait()
                cute.arch.fence_view_async_shared()
                out_pipe.producer_commit(out_state)
                out_state.advance()

        # ============================================
        # Warp 5 — S2G consumer
        # ============================================
        elif warp_idx == 5:
            use_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, STAGES_O
            )
            rel_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, STAGES_O
            )
            H_BYTES = cutlass.const_expr(H * 2)
            for s_local in cutlass.range(n_tok, unroll=1):
                s = s_beg + s_local

                out_pipe.consumer_wait(use_state)

                # cp.async.bulk is single-thread — only lane 0 of warp 5 issues it.
                if cute.arch.lane_idx() == 0:
                    g_row_int = (
                        gmem_out.iterator + cutlass.Int64(s) * cutlass.Int64(H)
                    ).toint()
                    s_row_int = (
                        out_smem.iterator + use_state.index * H
                    ).toint()
                    _cp_async_bulk_s2g(
                        s_row_int.ir_value(),
                        g_row_int.ir_value(),
                        Int32(H_BYTES).ir_value(),
                    )
                    cute.arch.cp_async_bulk_commit_group()

                use_state.advance()

                if s_local >= Int32(STAGES_O - 1):
                    cute.arch.cp_async_bulk_wait_group(STAGES_O - 1)
                    out_pipe.consumer_release(rel_state)
                    rel_state.advance()

            cute.arch.cp_async_bulk_wait_group(0)
            # Drain remaining (up to STAGES_O-1) outstanding releases.
            for _ in cutlass.range(STAGES_O - 1, unroll=1):
                if rel_state.count < use_state.count:
                    out_pipe.consumer_release(rel_state)
                    rel_state.advance()

        # ============================================
        # Warp 6 — weight-gather epilogue
        # ============================================
        elif warp_idx == 6:
            # Independent route for weight gather (bwd only).
            # The gather has no data dependency on the load/out pipelines, so it runs
            # concurrently in its own warp instead of serializing in warp 5's prologue,
            # which was throttled by the 2-stage out_pipe.
            if cutlass.const_expr(self.with_weights):
                lane = cute.arch.lane_idx()
                sk_base = s_beg * K
                sk_total = n_tok * K
                n_chunks = (sk_total + 31) // 32
                for sk_chunk in cutlass.range(n_chunks, unroll=1):
                    sk_local = sk_chunk * 32 + lane
                    if sk_local < sk_total:
                        sk_global = sk_base + sk_local
                        dst_val = dst_tensor[sk_global]
                        # Duplicate entries (-raw_dst - 1) still own their
                        # per-topk weight slot: decode and gather as usual,
                        # mirroring the dispatch-side weight scatter.
                        raw_dst = dst_val
                        if raw_dst < 0:
                            raw_dst = - raw_dst - Int32(1)
                        drank = raw_dst // NvS
                        loff = raw_dst % NvS
                        wb = meta_tensor[
                            drank * meta_stride + weights_off + loff
                        ]
                        sk_tensor[sk_global] = wb

# ============================================================================
# Host launcher with per-(H, R, with_weights) compile cache
# ============================================================================

@functools.lru_cache(maxsize=None)
def _max_smem_per_block_optin(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).shared_memory_per_block_optin


@functools.lru_cache(maxsize=None)
def _get_compiled(
    H: int,
    R: int,
    S: int,
    K: int,
    NvS: int,
    NvS_padded: int,
    meta_stride: int,
    num_sms: int,
    with_weights: bool,
    device_index: int,
    pdl_launch: bool,
):
    smem_budget = _max_smem_per_block_optin(device_index) - 1024  # match C++ headroom
    kernel = CombineKernel(
        H=H, R=R, S=S, K=K, NvS=NvS, NvS_padded=NvS_padded,
        meta_stride=meta_stride, num_sms=num_sms,
        with_weights=with_weights, smem_budget=smem_budget,
        pdl_launch=pdl_launch,
    )

    # Representative arguments for cute.compile. Pointers carry only their
    # dtype + assumed alignment — actual addresses are bound at call time.
    bf16_ptr = make_ptr(BFloat16, 0, cute.AddressSpace.gmem, assumed_align=16)
    i32_ptr = make_ptr(Int32, 0, cute.AddressSpace.gmem, assumed_align=16)
    stream_arg = cuda.CUstream(0)

    return cute.compile(
        kernel,
        bf16_ptr,  # output
        i32_ptr,   # output_sk (int32 view)
        bf16_ptr,  # hidden
        i32_ptr,   # meta
        i32_ptr,   # dst
        i32_ptr,   # bar (grid barrier counter)
        Int32(0),  # rank
        Int32(0),  # weights_off
        Int32(0),  # barrier_off
        Int32(0),  # num_tokens
        stream_arg,
    )


def launch_combine(
    ctx: dict,
    output_sh,
    dst,
    output_sk=None,
    *,
    pdl_launch: bool = False,
):
    """Launch the combine kernel.

    Args:
        output_sh: [s, H] bf16 output buffer to receive the per-token
            accumulated result, 1 <= s <= S (the Buffer capacity)
        dst: [N=s*K] int32 routing offsets (must match the dispatch that
            populated hidden_buf / weights_buf). Non-negative entries encode
            ``dest_rank * NvS + local_offset`` and are pulled and accumulated.
            Negative entries encode the same raw destination as
            ``-raw_dst - 1`` (duplicate top-k entries, pre-reduced into the
            primary slot by the combine prologue): the hidden path skips
            them, the weights gather decodes and reads them as usual.
        output_sk: [S, K] fp32 buffer to receive gathered route weights, or
            None to skip the weights gather (placeholder tensor is passed to
            satisfy the non-null pointer constraint; kernel ignores it when
            with_weights=False).
    """
    with_weights = output_sk is not None
    if not with_weights:
        # Placeholder; kernel will not dereference.
        output_sk = dst

    assert output_sh.dtype == torch.bfloat16 and output_sh.is_contiguous(), \
        "output_sh must be contiguous bf16"
    assert dst.dtype == torch.int32 and dst.is_contiguous(), \
        "dst must be contiguous int32"
    assert ctx['hidden_buf'].dtype == torch.bfloat16 and ctx['hidden_buf'].is_contiguous()
    assert ctx['meta_buf'].dtype == torch.int32 and ctx['meta_buf'].is_contiguous()
    if with_weights:
        assert output_sk.dtype == torch.float32 and output_sk.is_contiguous(), \
            "output_sk must be contiguous fp32"
    assert ctx['H'] % CombineKernel.ACC_THREADS == 0, \
        f"H must be a multiple of ACC_THREADS={CombineKernel.ACC_THREADS} " \
        "(also covers the 16-B bulk-copy alignment requirement)"

    H = int(ctx['H'])
    R = int(ctx['R'])
    S = int(ctx['S'])
    K = int(ctx['K'])
    NvS = int(ctx['NvS'])
    NvS_padded = int(ctx['NvS_padded'])
    meta_stride = int(ctx['meta_chunk_padded'])
    num_sms = int(ctx['num_sms'])
    device_index = output_sh.device.index

    # output-sh rows = this step's token count (<=S); dst carries K entries
    # per token and output_sk one row per token
    assert output_sh.ndim == 2 and int(output_sh.shape[1]) == H \
        and 0 < int(output_sh.shape[0]) <= S, \
            f"output_sh must be shape [s, H={H}] with 1 <= s <= S={S}, " \
            f"got {tuple(output_sh.shape)}"
    num_tokens = int(output_sh.shape[0])
    assert dst.numel() == num_tokens * K, \
        f"dst must have s*K={num_tokens*K} entries, got {dst.numel()}"
    if with_weights:
        assert tuple(output_sk.shape) == (num_tokens, K), \
            f"output_sk must be shape [s={num_tokens}, K={K}], " \
            f"got {tuple(output_sk.shape)}"
    
    compiled = _get_compiled(
        H, R, S, K, NvS, NvS_padded, meta_stride,
        num_sms, with_weights, device_index,
        bool(pdl_launch),
    )

    # int32 view onto fp32 output_sk so the kernel always sees an int32 ptr
    # (matches the C++ approach: a 4-byte gather, no fp32 arithmetic).
    sk_int = output_sk.view(torch.int32) if with_weights else output_sk

    out_ptr = make_ptr(BFloat16, output_sh.data_ptr(), cute.AddressSpace.gmem,
                       assumed_align=16)
    sk_ptr = make_ptr(Int32, sk_int.data_ptr(), cute.AddressSpace.gmem,
                      assumed_align=16)
    hid_ptr = make_ptr(BFloat16, ctx['hidden_buf'].data_ptr(),
                       cute.AddressSpace.gmem, assumed_align=16)
    meta_ptr = make_ptr(Int32, ctx['meta_buf'].data_ptr(),
                        cute.AddressSpace.gmem, assumed_align=16)
    dst_ptr = make_ptr(Int32, dst.data_ptr(), cute.AddressSpace.gmem,
                       assumed_align=16)
    bar_ptr = make_ptr(Int32, ctx['grid_sync_bar'].data_ptr(),
                       cute.AddressSpace.gmem, assumed_align=16)

    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    compiled(
        out_ptr,
        sk_ptr,
        hid_ptr,
        meta_ptr,
        dst_ptr,
        bar_ptr,
        Int32(int(ctx['rank'])),
        Int32(int(ctx['WEIGHTS_OFF'])),
        Int32(int(ctx['BARRIER_OFF'])),
        Int32(num_tokens),
        stream,
    )
