# Runtime variable sequence length in MoonEP (`s <= S`)

**Status (2026-10-05).** The fork was reset to upstream `33327eb` ("Public
Release 26/09"), and runtime varlen support was then added again on top of it.
The changes are in the working tree and not committed yet. The design is the
one from the earlier fork rounds 1 and 2 (commits `f08da5e`..`d9f044b`, kept on
branch `master-before-rebase`), ported to upstream's compact `[2*epn]` layout
and push prefetch. It adds one change: `plan.nvs_s` is rounded up to
`token_padding`, matching upstream's alignment of `NvS`. Verification so far is
in §4: every test module passed under a multicast-free emulation on 8x H200.
They still have to run natively on a node where NVSwitch multicast works.

## 1. Goal and design

`Buffer(S, H, K, E, R, ...)` used to require exactly `S` tokens per rank on
every `dispatch`. Variable-length training therefore padded every step to
`S`. That padding costs planning, NVLink traffic, grouped-GEMM rows and both
backward passes in proportion to `S - s`. The zero padding rows also all route
to experts `0..K-1`, which skews the load balancer.

- **`S` is a capacity.** It is fixed at construction and sizes every buffer,
  every CuTe layout and stride (`N = S*K`, `NvS`, `NvS_padded`, the meta
  offsets, the dedup key stride `src_rank * S + token`). It also stays in
  every compile cache key, so there is one JIT kernel set and one VMM /
  multicast mapping.
- **`s` (`num_tokens`) is a runtime `Int32` kernel argument.** The host takes
  it from `hidden_sh.shape[0]`, with `1 <= s <= S`. Planning, dispatch and
  combine loop to `s`, and no padding tokens exist. `s = 0` is not supported.
- **Per-rank receive caps.** Rank 0 sums the gathered `tokens_per_expert` into
  per-rank totals `n_r` and forms `total_slots = sum_r n_r`. It gives rank `r`
  the cap `total_slots // R + (1 if r < total_slots % R else 0)`. The balancer
  drives `group_tokens[r] - cap_r` to zero, so the caps are met exactly, and
  ranks stay balanced up to a remainder of `< R` slots. With the same `s`
  everywhere, `cap_r = s*K` on every rank.
- **Two modes, chosen on the host by `total_num_tokens`.**
  - `None` (the default) means every rank passes the same `s`. The kernel gets
    `total_tokens = R*s, uniform = 1`, and rank 0 traps if any `n_r` differs
    from its own.
  - An int means `sum_r s_r`, which the host knows (from the dataloader, or
    one `all_gather` of ints). The kernel gets `uniform = 0`, and rank 0 traps
    if `sum_r n_r != total_tokens*K`. After the PLAN broadcast, every other
    rank re-checks its own `total_tokens` against the step total, so a rank
    given a different `total_num_tokens` traps too.
  - In both modes, every rank traps if its own `sum_e tpe[e] != s*K`.
  - A trap is a CUDA error on that rank, and its peers hang at the next
    cross-rank barrier until `torchrun` kills them. The `cute.printf` message
    names the failed check.
- **Step-sized views.**
  - `plan.nvs_s = planning_nvs_s(ctx, total)` is
    `align_up(ceil_div(total*K, R) + token_padding_extra, token_padding)`,
    computed on the host. It is an upper bound of every rank's cap plus the
    worst-case segment padding, rounded the same way as
    `NvS = align_up(S*K + token_padding_extra, token_padding)`. So
    `nvs_s == NvS` at `s = S`, `nvs_s <= NvS` always, and the row count stays
    a multiple of `token_padding` for fixed-tile consumers.
  - `dispatch` returns `hidden_nvsh[:nvs_s]` and `route_weights_nvs[:nvs_s]`,
    as zero-copy views or prefix copies.
  - `combine` accepts either those or the full `[NvS, ...]` shard.
  - `cu_seqlens[-1] <= nvs_s` is a planner invariant, so
    `torch._grouped_mm(..., offs=cu_seqlens)` needs no change.
  - `nvs_s` is independent of `plan.N = s*K`: a small-`s` rank can receive
    more slots than it sends.
- **The plan remembers the step:** `plan.N == s*K`, `plan.num_tokens`,
  `plan.nvs_s`. Combine and both backward passes (plan reuse) inherit them.
  `total_num_tokens` is ignored on plan reuse; if it is passed there, it must
  imply the same `nvs_s`.
- **`prefetch_weight` and `reduce_grad` do not depend on `s`.** They move
  expert weights and grads for the slots in `plan.experts_to_copy`, which
  planning fills the same way at any `s`. Upstream's push prefetch is
  collective (it ends in a cross-rank barrier), so every EP rank calls it at
  every step, whatever its own `s`.

No device-to-host sync was added: `s`, `total_num_tokens` and `nvs_s` are all
known on the host.

## 2. What changed relative to upstream `33327eb`

Grep for the identifiers below rather than relying on line numbers.

### `moonep/planning.py`
- `MoonEPCommPlan` changes:
  - new field `nvs_s` (checked against `0 < nvs_s <= NvS` in `__post_init__`
    and copied by `clone()`);
  - new property `num_tokens = N // K`.
- `PlanningKernel.__call__` / `kernel` take `rank_tokens` (per-rank entry
  totals, zeroed in kernel) and the runtime `num_tokens, total_tokens,
  uniform`.
- Loop bounds use `n_rt = num_tokens*K` instead of the capacity `N`:
  - `run_c1`: histogram, vblock prefix (a runtime `cutlass.range(nvb_rt)` with
    `cumsum = Int32(0)`) and pass-A scatter;
  - the rank-0 top-k offload (`copy_v4_remote(..., n_rt)`);
  - the `dst` write loop;
  - the dedup canonicalization loop (`num_tokens`).
- `copy_v4_remote` takes a runtime `n` and clamps its scalar head
  (`head = min(head, n)`).
- Rank 1 orders rank 0's top-k with `n_rt_0 = sum_e tp0[e]`, because rank 0
  may have a different `s`.
- The balancer uses `balance[r] = group_tokens[r] - cap_r`, with `cap_r` as
  in §1, in place of upstream's constant `CAP = NvS_capacity`.
- New contract checks, all trapping with a printed message (§1):
  - every rank: its local tpe sum;
  - rank 0: the uniform-mode check, then the total;
  - every rank, after the PLAN multicast barrier: its `total_tokens`.
  Block reductions use `block_sum_i32` with dedicated shared-memory scalars.
- Host side:
  - `planning_recv_cap`, `planning_nvs_s`;
  - `allocate_planning_outputs(ctx, num_tokens=None, total_num_tokens=None)`;
  - `launch_planning(..., *, total_num_tokens=None)`, which asserts that
    `plan.nvs_s == planning_nvs_s(ctx, total)`;
  - `_aligned16_i32` copies a misaligned or non-contiguous `topk`/`tpe`;
  - `_rank_tokens_buffer`.
- Unchanged on purpose:
  - the `src_info` clear over the full `NvS`;
  - the tpe-driven phases (allocation, selection, `cu_seqlens`,
    `experts_to_copy`, zero-fill), which need no token count;
  - upstream's selection rewrite and its Phase-D proxy fence.

### `moonep/dispatch.py`, `moonep/combine.py`
- `__call__` / `kernel` take `num_tokens: Int32`.
- The per-block token range is `tpb = ceil(num_tokens / num_sms)` with
  `s_end = max(min(s_beg + tpb, num_tokens), s_beg)`. The `max` clamp keeps
  `n_tok` non-negative when `s < num_sms`.
- `launch_dispatch` takes `hidden_sh` as `[s, H]` and `route_weights_sk` as
  exactly `[s, K]`, and asserts 16-byte alignment.
- `_check_dispatch_plan` requires `plan.N == s*K`.
- `launch_combine` takes `output_sh` as `[s, H]`, `dst` with `s*K` entries and
  `output_sk` as `[s, K]`.
- The kernels' `[S, ...]` layouts are still declared at capacity. Every access
  is bounded by `num_tokens`, and no TMA descriptor is built from these
  tensors.
- Unchanged: the dedup builder still initializes `primary_packed` / `kmask`
  over `R*S` and scans all `NvS` `src_info` slots. That is int32 work
  proportional to the capacity.

### `moonep/api.py`
- `Buffer.dispatch(..., total_num_tokens=None)`.
- `s = hidden_sh.shape[0]`, asserted in `1 <= s <= S` (the message contains
  "capacity"). `topk_experts_sk` and `route_weights_sk` must be exactly
  `[s, K]`.
- A misaligned or non-contiguous `hidden_sh`, `route_weights_sk` or `topk` is
  copied before launch.
- On plan reuse, `plan.num_tokens == s` is asserted (the message contains
  "tokens"). A `total_num_tokens` passed there must imply `plan.nvs_s`.
- Returned views are `[plan.nvs_s, H]` / `[plan.nvs_s]`. Boundary copies move
  exactly the rows of the view.
- `Buffer.combine` accepts `[plan.nvs_s, ...]` or `[NvS, ...]` inputs and
  returns `[s, H]` / `[s, K]`.
- `ctx['rank_tokens']` (int32 `[R]`) is allocated with the other local temps.
- Docstrings and the module header document `S` vs `s`, the two modes, and
  CUDA-graph bucketing.

### Tests (`tests/`)
- `kernel_test_utils.py`:
  - `expected_nvs_s(ctx, total)` computes the rounded `nvs_s` independently
    of `planning_nvs_s`;
  - `partial_token_counts(S)` returns `{1, S//2, S-1}`;
  - `planning_invariant_errors(..., num_tokens=None, nvs_s=None)` bounds this
    rank's `dst` / `cu_seqlens` by `nvs_s`;
  - the meta-layout and `NvS` expectations use the capacity `S*K`;
  - `make_topk(case, rank, R, s=None)` keeps the first `s` tokens of the
    full routing.
- `planning_reference.py` derives `s` from the routing input and the
  per-rank caps from the gathered tpe. It gathers `dst` padded to `S*K` for
  the dedup reference.
- `test_planning.py`:
  - partial-`s` comparisons to the reference for every case;
  - `MULTI_VBLOCK_CASES` (2, 3 and 4 vblocks, with full and partial tails);
  - per-rank `s` with `total_num_tokens`.
- `test_dispatch.py` / `test_combine.py`:
  - an `s` parameter on the helpers;
  - partial-`s` variants of the scatter, dedup, zero-fill, saved-plan,
    global-reference and `output_sk` tests;
  - extra partial-`s` shapes (`K = 1`, odd `K`, a single expert split over
    ranks);
  - a check that `Buffer.combine` accepts prefix views.
- `test_e2e.py`:
  - `test_e2e_partial_tokens` runs dispatch, sync/async, zero-copy, combine,
    prefetch, `reduce_grad`, plan reuse, replays across steps, and the
    rejections, for `s in (S//2+3, 5, 1, S)`;
  - `test_e2e_per_rank_tokens` runs `total_num_tokens`;
  - the varlen tests use `E = R*8`, which keeps the prefetch and reduce pool
    chunks aligned to VMM granularity.
- Upstream's removal of `B` also applies to the varlen cases: no
  `KernelCase(B=...)`, no `Buffer(B=...)`, and `cu_seqlens` is `[2*epn]`.

### `my_moon_ep.py` (training example)
Ported from the old fork to the upstream API:
- **Memory.** One `MoonEPShared` (one `Buffer` and one `ExpertPools`) serves
  every layer. The pools provide everything MoonEP maps: `weights(n)` (bf16
  staging rows, then prefetch slots) and `grads(n)` (fp32 local-grad rows,
  then reduce slots). Each layer's parameters are only its fp32 masters (the
  router and the `[epn, ...]` experts) in ordinary memory.
- **Sharing.** `MoonEPShared.stage_and_push(plan, masters)` remembers which
  plan the pools hold. A layer's backward re-stages its experts and re-pushes
  its saved plan only when another layer pushed since its forward, which is
  every layer except the last.
- **Weight grads.** `_ExpertFFN` runs the SwiGLU grouped GEMMs. Its forward
  stages the masters as bf16 and calls `pools.push(plan)`. Its backward takes
  dW over `cu_seqlens[2*epn]` into `grads(n)`, calls `pools.reduce(plan)`, and
  returns the local rows as the masters' grads. These are ordinary autograd
  grads: nothing aliases `.grad`, `zero_grad(set_to_none=True)` works, and no
  separate reduce or weight-sync step remains. The router's grad is summed
  over the EP group in its backward (`_AllReduceGrad`). The old `offs_live`
  compaction is not needed: the `[2*epn]` layout has no dead groups.
- **Varlen.** `forward(x, total_num_tokens=None)` takes `[s, H]` for any
  `s <= S`.
- **Correctness check.** `main()` first checks a small 3-layer stack on one
  shared Buffer against a dense reference built from the all-gathered
  experts. It compares output, dx, router grad and expert grads at `s = S`,
  `s = 131`, `s = 1` and per-rank `s`. Each layer's reference gets that
  layer's actual input and output grad, so bf16 differences do not compound
  across layers. It then times a `NUM_LAYERS`-layer training step (default
  2).
- **Fixed defects of the old version:** a global RNG reseed in `__init__`, a
  `last_plan` / `_last_plan` mix-up, and a `reference_grads` that could not
  run.
- **Emulated result on `slinky-0`** (the earlier hand-mapped version, before
  the `ExpertPools` rewrite; not re-run since). All four checks passed:
  out <= 7e-3, dx <= 1.5e-2, router grad <= 2e-7, expert grads <= 5.5e-3
  relative. The E=128 / K=16 / H=4096 / Hi=8192 / S=4096 step took 204.7 ms
  with AdamW. The bf16 copies matched the masters and the router stayed
  identical across ranks.

### `README.md`, `benchmarks/bench_comm.py`
- The README notation, API bullets and code comments use `[s, ...]` /
  `[plan.nvs_s, ...]`. They describe both modes, the traps, the step-sized
  views, `s = 0` and CUDA-graph bucketing.
- `bench_comm.py --num-tokens s` (default `S`) times every operator at `s`
  tokens per rank on a Buffer built for `S`, accounts bytes with `s`, and adds
  an `s` column. It also fixes upstream's failure CSV row, which was missing
  the `epn` column.

## 3. How to verify

### 3.1 Real runs (8 GPUs with working NVSwitch multicast)
```bash
pip install -e .   # builds moonep._C (needs nvcc, a host compiler and Python headers)
torchrun --nproc_per_node=8 -m pytest -s tests/test_planning.py tests/test_e2e.py
torchrun --nproc_per_node=8 -m pytest -s tests/test_dispatch.py tests/test_combine.py
torchrun --nproc_per_node=8 -m pytest -s tests/test_prefetch.py tests/test_grad_reduce.py
torchrun --nproc_per_node=8 benchmarks/bench_comm.py --ep 8 --hidden 7168 --topk 8 \
    --unbalance-ratio 1.0 --num-tokens 1024
```
Likely first failures if something is off:
- the per-rank-`s` planning test on `dst` / `cu_seqlens` (cap rounding);
- an `nvs_s` mismatch (`expected_nvs_s` against the kernel plan);
- a planning trap (a test passing different `s` without `total_num_tokens`;
  the trapping rank prints the message and the others hang);
- a hang at `s = 1` / `s = 5` (per-block ranges with empty blocks).

### 3.2 Without multicast
The current cluster node (`slinky-0`, 8x H200) cannot run MoonEP directly
(§5). §4 used an emulation that still runs one process per GPU under
`torchrun` and runs the repository's own test modules:
- `moonep._C` is replaced by a cuda-python implementation of `nvl_dist_alloc`
  / `nvl_dist_map`. It uses the same `cuMemCreate` + POSIX-fd export, the
  same import, and the same back-to-back mapping with RW access for the local
  device. The fds still travel through MoonEP's own `_exchange_ipc_fds`.
- The meta-buffer multicast view is the base of the all-rank meta mapping,
  and planning's single `multimem.st` (the PLAN broadcast) becomes R unicast
  stores into every rank's meta chunk. The planning compile wrapper sets the
  rank count and chunk stride for the store.
- A small `pytest` stand-in (parametrize, param, raises, skip, the `dist_env`
  fixture, buffer cleanup after each test) replaces pytest, which the venv
  does not have.
- NCCL runs with `NCCL_NVLS_ENABLE=0`, and the cross-rank barrier timeout is
  raised to 5 min so per-process JIT skew cannot trap.

Everything else is MoonEP's code and tests: planning, dispatch, the dedup
epilogue and prologue, combine, prefetch, grad reduce, the `Buffer` API, the
torch planning reference and the invariant checks. The emulation scripts are
not part of the repository.

## 4. Verification results (2026-10-05, `slinky-0`, emulation of §3.2)

All runs used R = 8 under `torchrun --nproc-per-node 8`.

| Suite | Result |
|---|---|
| `tests/test_planning.py` | 62 passed, 4 skipped |
| `tests/test_e2e.py` | 3 passed |
| `tests/test_dispatch.py` | 33 passed, 2 skipped |
| `tests/test_combine.py` | 27 passed |
| `tests/test_prefetch.py` | 20 passed |
| `tests/test_grad_reduce.py` | 20 passed |

- **Skips.** The skips are the `S = 1` cases (no partial step exists) and the
  case limited to `R <= 2`, the same as in the earlier rounds.
- **One fixed test failure.** The first `test_e2e` run failed in
  `test_e2e_partial_tokens` at the prefetch check. The cause was a zero-fill
  versus push ordering bug in the test, not in the library (§6). It is fixed.
- **What the emulation does not cover.** NVSwitch multicast and the C++
  `moonep._C` were replaced, so the real multicast path and the extension are
  still unverified on this tree (§7.1).

**Benchmark.** `benchmarks/bench_comm.py --ep 8 --hidden 7168 --topk 8
--unbalance-ratio 1.0 --hp 128` (S = 8192, E = 896, CUDA graphs), under the
same emulation. Times are in us.

| `s` | plan | dispatch fwd | dispatch bwd | combine fwd | combine bwd | epilogue / prologue |
|---|---|---|---|---|---|---|
| 8192 (= S) | 51 | 1717 | 1687 | 1684 | 1702 | 155 / 202 |
| 1024 | 43 | 290 | 298 | 231 | 234 | 23 / 28 |

- **Data movement scales with `s`.** Dispatch and combine match the earlier
  round-2 numbers on a multicast node (1722 / 1689 us at `s = S`, 290 / 230
  us at `s = 1024`).
- **Planning is the fixed floor.** Here its PLAN broadcast is R unicast
  stores instead of one multicast store, so its time is only indicative.

## 5. Environment facts

- **Node:** `slinky-0` (Slurm, `srun -w slinky-0 -p slinky --gres=gpu:8`),
  8x H200 with NV18 links between all pairs, driver 580.126.09.
- **Multicast is broken on that node (2026-10-05).**
  - `nvidia-smi` reports Fabric State Completed / Success, but
    `cuMulticastBindMem` fails with `CUDA_ERROR_ILLEGAL_STATE` (401), even in a
    single process with any handle type.
  - NCCL's default NVLS init fails the same way; NCCL works with
    `NCCL_NVLS_ENABLE=0`.
  - MoonEP's `Buffer` needs multicast for its meta buffer (`api.py`
    `create_nvl_dist_multicast_tensor` → `csrc/nvl_shared_buffer.cuh`
    `cuMulticastBindMem` under `CUCHECK`), so it exits on construction there.
  - The node also has no host compiler or Python headers to build `moonep._C`.
- **Python:** `/pfs/code/megatron-minimax-experiments/.venv/bin/python` (torch
  2.14.1+cu130, nvidia-cutlass-dsl 4.8.0, cuda-bindings 13.4.3). `setup.py`
  pins nvidia-cutlass-dsl 4.6.2; the kernels compiled and ran with 4.8.0 in
  §4. Use `PYTHONDONTWRITEBYTECODE=1` so runs do not write into that venv.

## 6. Gotchas

- **CuTe DSL loop bounds and scalars.** Runtime loop bounds work with
  `cutlass.range(lo, hi, step)`. Loop-carried scalars must be DSL-typed before
  the loop (`Int32(0)`). `cutlass.min` / `cutlass.max` take runtime `Int32`s.
- **A runtime `n` needs the head clamp in `copy_v4_remote`.** The vector loop
  can no longer be proven to divide `n`.
- **Layouts declared at capacity are harmless as long as every access is
  bounded by the runtime count.** Keep that invariant when adding loops.
- **Kernel input alignment.** The kernels load user `topk` / `route_weights`
  / `hidden` with `assumed_align=16`. A row-offset slice or a non-contiguous
  view breaks that. The API copies such inputs; `launch_*` called directly
  asserts.
- **`nvs_s` depends on the total token count**, not this rank's `s`.
  Deriving it from `plan.N` is wrong as soon as ranks differ.
- **`nvs_s` is rounded up to `token_padding`.** Upstream aligned `NvS`, so an
  unrounded `nvs_s` would stop equalling `NvS` at `s = S`.
  `planning_nvs_s` is the single source of truth; tests use the independent
  `expected_nvs_s`.
- **Traps are the failure mode for contract violations.** They cannot be
  unit-tested under torchrun (the peers hang), so the host-side asserts are
  the testable first line.
- **Push prefetch is collective.** Every EP rank must call `prefetch_weight`
  in the same order relative to dispatch, combine and `reduce_grad`, at every
  step and whatever its `s`.
- **Pool writes race with peers' pushes.** Peers write into this rank's pool
  slice, and prefetch has no entry barrier. Any local write to that slice
  (zero-filling a fresh pool, or a kernel still reading old slots) needs an
  ordering point on every rank before the next prefetch. The first
  `test_e2e_partial_tokens` run showed this: a zero-fill right before
  prefetch erased a peer's push. The test now synchronizes and barriers
  between them.
- **The `[2*epn]` layout.** The planner gives every remote expert with tokens
  a slot. `cu_seqlens` is `[2*epn]` for every `s`, with this rank's experts in
  groups `[0, epn)` and its prefetch slots in `[epn, 2*epn)`.

## 7. Remaining work

1. **Run the 8-GPU pytest suites** (§3.1) on a node with working multicast,
   then commit in logical pieces (kernels; API; tests; README and benchmark;
   this document).
2. **CUDA graphs.** A captured graph is valid for one `(s, total_num_tokens)`,
   so bucket `s`. A single graph for every `s <= S` would need device-scalar
   `num_tokens` / `total_tokens` and capacity-sized views with a device-side
   `nvs_s`, which brings back the capacity-sized activation cost. Not
   started.
3. **Capacity-proportional int32 work.** The `src_info` clear, the
   `primary_packed` / `kmask` init and the `NvS` scans could be bounded by
   `nvs_s` / `s` only if clear and scan use the same bound. This is under
   1 MB of int32 metadata per step, so it is deferred.
4. **Known limits.**
   - `s = 0` is not supported; pass a dummy token.
   - The multi-node fabric path is untested at any `s`.
   - Contract violations trap instead of raising a cross-rank error.
5. **Not carried over from the old fork.** The other examples
   (`moe_moon_ep.py`, `moe*.py`, `bench_head2head.py`) and the Megatron-LM
   MoonEP backend still use the pre-`33327eb` API (`[E+B]` composites, pull
   prefetch, `full_*` tensors). They need the same port as `my_moon_ep.py`.
