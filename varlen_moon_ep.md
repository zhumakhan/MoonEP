# Variable-length tokens in MoonEP (`s <= S`) — handoff

Status as of 2026-09-19 (night). **Round 1** (`s <= S` per call, the same
`s` on every rank) is committed as `f08da5e` + `4818969` + `cbb3940` on
`master` and was verified on an 8x H200 NVLink node: planning + e2e 36
passed / 3 skipped (both files in one pytest session), dispatch + combine 26
passed at full `S`, `moe_moon_ep.py` ran (dense reference skipped for
memory, replay diff 0), and a driver stepping one `MoonEPMoE` through `s in
{512, 300, 33, 7, 1, 511, 512}` matched the dense reference within bf16
tolerance on every rank.

**Round 2** — the items an independent review raised — is **implemented in
the working tree (uncommitted) and verified on the same 8x H200 node on
2026-09-19**: the uniform-`s` contract is now checked on device, per-rank
`s` works via `total_num_tokens` (§6.4), `dispatch` returns step-sized
`[plan.nvs_s, ...]` views (§6.2), partial-`s` dispatch/combine unit tests
and multi-vblock planning cases exist (§6.5), `copy_v4_remote` clamps its
head, user `topk`/`route_weights` fall back to an aligned copy, the `[s, K]`
asserts are exact, stale `[S, H]` docstrings are fixed, and the README has
CUDA-graph bucketing guidance. Results (§4.2): `test_planning.py` 62 passed /
4 skipped (incl. the multi-vblock and per-rank cases), `test_dispatch.py` +
`test_combine.py` 60 passed / 2 skipped (partial-`s` variants included),
`test_e2e.py` 3 passed (uniform partial `s`, per-rank `s`), the direct
`torchrun tests/test_e2e.py` run passes, `moe_moon_ep.py` runs with the
`[nvs_s, H]` views (replay diff 0), and the partial-`s` model driver matches
the dense reference on every rank. The skips are the `S = 1` cases (no
partial step possible) and the `max_R = 2` case. `bench_comm.py --ep 8
--hidden 7168 --topk 8 --unbalance-ratio 1.0` (S = 8192, CUDA graphs):
at `s = 8192` plan 91 us, dispatch fwd 1722 us, combine fwd 1689 us,
boundary copies 204 us; at `--num-tokens 1024` plan 85 us, dispatch fwd
290 us, combine fwd 230 us, copies 30 us. Data movement scales with `s`;
planning is the fixed floor (§5).

## 1. Goal and design

MoonEP's `Buffer(S, H, K, E, R, ...)` used to require exactly `S` tokens per
rank on every `dispatch`. Training with variable sequence lengths therefore
meant padding to `S`, which wastes planning, NVLink traffic, grouped-GEMM
rows and both backward passes in proportion to `S - s`, and the zero padding
rows all route to experts `0..K-1`, skewing the load balancer.

Design now in the tree:

- `S` is a **capacity** fixed at construction. It still sizes every buffer,
  every CuTe layout and stride (`N = S*K`, `NvS`, `NvS_padded`, meta offsets,
  the dedup key stride `src_rank * S + token`), and stays in every compile
  cache key. One JIT kernel set, one VMM/multicast mapping.
- `s` (`num_tokens`) is a **runtime `Int32` kernel argument**, derived on the
  host from `hidden_sh.shape[0]`, `1 <= s <= S`. Planning, dispatch and
  combine loop to `s`. No padding tokens exist. `s = 0` is not supported.
- **Per-rank receive caps.** Rank 0 sums the gathered `tokens_per_expert`
  into per-rank totals `n_r`, forms `total_slots = sum_r n_r`, and gives rank
  `r` the cap `total_slots // R + (1 if r < total_slots % R else 0)`. The
  balancer drives `group_tokens[r] - cap_r` to zero, so the caps are met
  exactly; balance across ranks holds up to a remainder of `< R` slots. With
  the same `s` everywhere this is `cap_r = s*K` on every rank, i.e. exactly
  the round-1 behaviour.
- **Two modes, selected on the host by `total_num_tokens`.** `None`
  (default) means every rank passes the same `s`; the kernel gets
  `total_tokens = R*s, uniform = 1` and rank 0 traps if any `n_r` differs
  from its own. An int means `sum_r s_r`, host-known (dataloader, or one
  `all_gather` of ints); the kernel gets `uniform = 0` and rank 0 traps if
  `sum_r n_r != total_tokens*K`; after the plan multicast every other rank
  re-checks its own host `total_tokens` against the plan's step total, so a
  rank given a different `total_num_tokens` traps too. Every rank also traps
  if its own `sum_e tpe[e] != s*K`. A trap is a CUDA error on that rank and a hang for
  its peers at the next cross-rank barrier (until `torchrun` kills them);
  the `cute.printf` message says which check failed.
- **Step-sized views.** `MoonEPCommPlan.nvs_s = cap_r + token_padding_extra
  <= NvS` is computed on the host (`ceil_div(total*K, R) + extra`, which
  upper-bounds every rank's cap) and is the number of shard rows dispatch
  may write this step. `dispatch` returns `hidden_nvsh[:nvs_s]` /
  `route_weights_nvs[:nvs_s]` (zero-copy views or prefix copies); `combine`
  accepts either those or the full `[NvS, ...]` shard. `cu_seqlens[-1] <=
  nvs_s` is a planner invariant, so `torch._grouped_mm(..., offs=cu_seqlens)`
  needs no change. `nvs_s` is independent of `plan.N = s*K`: a small-`s`
  rank can receive more slots than it sends.
- The plan remembers the step: `plan.N == s*K`, `plan.num_tokens`,
  `plan.nvs_s`. Combine and both backward passes (plan reuse) inherit them;
  `total_num_tokens` is ignored on plan reuse.

No device-to-host sync was added: `s`, `total_num_tokens` and `nvs_s` are
host-known.

## 2. Environment facts

- Python with torch: `/home/ubuntu/MoonEP/.venv/bin/python` (Python 3.12.3, torch 2.14.0+cu132,
  nvidia-cutlass-dsl 4.4.2). Plain `python3` has no torch.
- If `from moonep._C import ...` fails (e.g. missing `FABRIC_HANDLE_BYTES`),
  the extension is stale: `python setup.py build_ext --inplace` (nvcc 13.0 in
  `/usr/local/cuda-13.0/bin`, no ninja, takes ~1 min).
- This box is 8x H200 with NVLink multicast (`nvl_multicast_supported()` is
  True), so the §4.2 torchrun suites run here; the venv needed `pytest`,
  `nvidia-cutlass-dsl==4.4.2` and `ninja` installed and `python3.12-dev` for
  the extension build. Kernel compile checks (§4.1) need only one GPU.

## 3. What changed, file by file

Identifiers, not line numbers: several files were edited concurrently in
round 2, so grep for the names below.

### `moonep/planning.py`
Round 1:
- `MoonEPCommPlan.num_tokens` property (`N // K`).
- `copy_v4_remote(dst, dst_off, src, n, ...)`: `n` is a runtime `Int32`, not
  a `cutlass.Constexpr` (used for the top-k and order copies).
- `PlanningKernel.__call__` / `kernel` take `num_tokens: Int32` after `rank`.
- `run_c1(..., n_rt)`: `nvb_rt = ceil(n_rt / 2048)`; the histogram and passA
  scatter loop over `nvb_rt` vblocks with `off < n_rt` guards; the vblock
  prefix pass is a runtime `cutlass.range(nvb_rt)` loop with `cumsum =
  Int32(0)`.
- dst loop bound `seg = ceil(n_rt / num_sms)`; dedup canonicalization loop
  bound `num_tokens`; `_check_planning_outputs` requires `plan.N % K == 0 and
  0 < plan.N <= S*K`.

Round 2:
- `MoonEPCommPlan.nvs_s: int`, declared right after `K`; `__post_init__`
  asserts `0 < nvs_s <= NvS`; `clone()` copies it.
- `allocate_planning_outputs(ctx, num_tokens=None, total_num_tokens=None)`:
  `s = S` if `num_tokens is None`, `total = R*s` if `total_num_tokens is
  None`; asserts `0 < s <= S` and `s <= total <= R*S`; `plan.nvs_s =
  ceil_div(total*K, R) + token_padding_extra` (asserted `<= NvS`); `dst`
  sized `s*K`, `plan.N = s*K`.
- `launch_planning(ctx, topk_flat, tpe, cu_seqlens, plan, *,
  total_num_tokens=None)`: positional order unchanged from round 1; `api.py`,
  the tests and the benchmarks pass `cu_seqlens`/`plan` by keyword.
  `uniform = total_num_tokens is None`, `total = R*plan.num_tokens` when
  uniform; asserts `plan.nvs_s` matches; forwards
  `num_tokens, total_num_tokens, uniform` to `_launch_planning_kernel`, which
  passes them as runtime `Int32` kernel arguments `(num_tokens, total_tokens,
  uniform)` right after `rank`.
- Kernel: every rank checks `sum_e tpe[e] == num_tokens*K` and traps with a
  `cute.printf` message otherwise. Rank 0 computes per-rank totals `n_r`
  from `tpe_gather` (kept in a small per-rank token-count buffer), checks
  `sum_r n_r == total_tokens*K`, and when `uniform != 0` checks every `n_r ==
  n_rt` (checked first, so a differing `s` prints the message naming the
  rank and the `total_num_tokens` remedy); both trap with a message. After
  the PLAN multicast barrier every rank sums row `R-1` of `tpe_cumsum` and
  traps if it differs from its own `total_tokens*K` (a per-rank
  `total_num_tokens` mismatch). Receive caps `cap_r = base + (r < rem)`
  with `base = total_slots // R`, `rem = total_slots - base*R`; the balancer
  uses `balance[r] = group_tokens[r] - cap_r`.
- Rank 1 orders rank 0's top-k with `n_rt_0 = sum_e tp0[e]` (rank 0's tpe
  copy) instead of its own `n_rt`, and copies `order0` back with `n_rt_0`.
- `copy_v4_remote` clamps its head so a runtime `n` that is not a multiple
  of the vector width never reads or writes past `n`.

Unchanged on purpose: the `src_info` clear loops over the full `NvS` and the
tpe-driven phases (allocation, cu_seqlens, experts_to_copy, zero-fill) need
no token count at all (§6.3).

### `moonep/dispatch.py`
- `__call__` and `kernel` take `num_tokens: Int32`; per-block token range
  `tpb = ceil(num_tokens/num_sms)`, `s_end = max(min(s_beg + tpb,
  num_tokens), s_beg)` (the `max` clamp fixes a latent negative `n_tok` when
  `bidx*tpb > num_tokens`, reachable once `s < num_sms`).
- `_check_dispatch_plan`: `plan.N == hidden_sh.shape[0] * K`, `0 < s <= S`,
  `dst` checked against `plan.N`. `launch_dispatch`: `hidden_sh` is `[s, H]`,
  `route_weights_sk` exactly `[s, K]`.
- Unchanged: the dedup builder still inits `primary_packed`/`kmask` over
  `R*S` and scans all `NvS` `src_info` slots (planning cleared them to -1).
  Correct, int32-only cost proportional to the capacity (§6.3).

### `moonep/combine.py`
- Same pattern: `num_tokens: Int32` in `__call__`/`kernel`; per-block range
  with clamp; `launch_combine`: `output_sh` is `[s, H]`, `dst.numel() ==
  s*K`, `output_sk` is `[s, K]`.
- The `gmem_out` layout in `__call__` is still declared `(S, H)` while the
  real tensor has `s` rows. Safe because only rows `< num_tokens` are touched
  and no TMA descriptor is built from it (raw `cp.async.bulk` addresses).
  Same for dispatch's `gmem_src` and both `dst` layouts.

### `moonep/api.py`
Round 1:
- `Buffer.dispatch`: `num_tokens = hidden_sh.shape[0]`, asserts `1 <= s <=
  S` (message contains "capacity") and `topk.numel() == s*K`; allocates the
  plan with `num_tokens`; on plan reuse asserts `plan.num_tokens == s`
  (message contains "tokens").
- `Buffer.combine`: outputs allocated as `[plan.num_tokens, H]` and
  `[plan.num_tokens, K]`.

Round 2:
- `Buffer.dispatch(..., *, inter_rank_sync=True, zero_copy=False,
  router_weights_zero_copy=False, total_num_tokens=None)`: `total_num_tokens`
  is passed to `allocate_planning_outputs` and `launch_planning` on the
  fresh-planning path and ignored on plan reuse.
- Returned `hidden_nvsh` is `[plan.nvs_s, H]` (`ctx['hidden_buf_local']
  [:plan.nvs_s]` under `zero_copy`, else a fresh tensor filled by a prefix
  copy) and `route_weights_nvs` is `[plan.nvs_s]` fp32 (view prefix or
  fresh).
- `Buffer.combine` accepts `hidden_nvsh.shape[0]` and
  `route_weights_nvs.shape[0]` in `(NvS, plan.nvs_s)`; the boundary copies
  write exactly `shape[0]` rows into the shard prefix; the zero-copy
  `data_ptr()` checks are unchanged (a prefix view shares the base pointer).
- Alignment fallback: the kernels load `topk_experts_sk` /
  `route_weights_sk` with `assumed_align=16`, so inputs whose `data_ptr()`
  is not 16-byte aligned (or not contiguous), e.g. slices of a larger
  tensor, are copied to an aligned contiguous tensor before launch.
- Tightened asserts: `route_weights_sk` / `topk_experts_sk` must be exactly
  `[s, K]` (shape, not just `numel()`).
- `dispatch_epilogue.py` and `combine_prologue.py` assert on
  `ctx['hidden_buf_local']` (still full `NvS`); no change.

### `moe_moon_ep.py` (example)
- `forward` runs any `s <= S` directly; the old pad-to-S wrapper and
  `forward_full_S` are gone.
- The per-layer host sync was removed earlier: the grouped GEMMs take the
  dispatched view with `cu_seqlens` as `offs` instead of slicing to
  `int(cu_seqlens[-1].item())`. That view is now `[plan.nvs_s, H]`;
  `cu_seqlens[-1] <= nvs_s` keeps `offs` in bounds, so no code changed for
  round 2, only comments. Verified earlier on this GPU that
  `torch._grouped_mm` leaves rows past `offs[-1]` as exact zeros in forward
  and that NaN grads on those rows reach neither weight grads nor grouped-row
  input grads; `w_nvs` is `nan_to_num`'d over all rows.
- Both autograd Functions pass the `[nvs_s, ...]` grads straight to
  `combine` / get them back from `dispatch` with the saved plan; nothing in
  the example hard-codes `NvS`.

### Tests
Round 1:
- `tests/planning_reference.py`: derives this rank's `s`/`N` from the routing
  input (`topk.numel() // K`); asserts `S <= ctx["S"]`.
- `tests/kernel_test_utils.py`: `planning_invariant_errors(..., num_tokens=
  None)`; meta-layout expectations use `N_capacity = case.S*K`; the `dst`
  shape check uses the step's `N`.
- `tests/test_planning.py::test_planning_partial_tokens_matches_reference`:
  over all `PLANNING_CASES`, plans `s in {1, S//2, S-1}` on a Buffer built
  for `S` and compares to the reference plus invariants.
- `tests/test_e2e.py::test_e2e_partial_tokens`: for `s in (131, 5, 1, 256)`
  on `S=256, K=4`: fresh plan sized `s`, `s*K <= cu_seqlens[-1] <= s*K +
  token_padding_extra`, sync == async, identity round trip
  `combine(dispatch(x)) ≈ K*x` (within 2 bf16 ULPs via `assert_ulp_all_ranks`; duplicate slots are
  pre-reduced in bf16 in the combine prologue), exact route-weight gather,
  plan reuse, rejection of `hidden[:s-1]` with a mismatched plan, and
  rejection of `s = S+1`. Takes the session-scoped `dist_env` fixture.

Round 2:
- `tests/planning_reference.py`: per-rank caps from the gathered tpe:
  `total_slots = tpe.sum()` over all ranks and experts, `cap = full(R,
  total_slots // R); cap[: total_slots % R] += 1`, `balance = group_tokens -
  cap`, capacity assert against `cap[r]`.
- `tests/test_dispatch.py` / `tests/test_combine.py`: an `s` parameter
  (default `case.S`) on `_traceable_weights`, `_verify_dispatch_by_dst`,
  `_plan_and_dispatch` and the combine helpers (`allocate_planning_outputs
  (ctx, s)`, sliced `make_topk` output with recomputed `tpe`), with
  partial-`s` variants of the scatter / dedup / identity tests.
- `tests/test_planning.py`: `PLANNING_CASES` entries with `S*K >= 4*2048`
  (collected as `MULTI_VBLOCK_CASES`) drive
  `test_planning_multi_vblock_partial_tokens`, which steps through 2, 3 and 4
  vblocks with full and partial tails (every earlier case had `S*K <= 2048`).
- Per-rank `s` tests: ranks pass different `s_r` with `total_num_tokens =
  sum_r s_r`; planning is compared to the reference's per-rank caps and the
  e2e round trip is checked per rank.
- e2e additions: `hidden_nvsh.shape[0] == plan.nvs_s`, `cu_seqlens[-1] <=
  plan.nvs_s`, combine of the `[nvs_s, ...]` views and of the full shard,
  and the `total_num_tokens` path.
- No test violates the `s`/tpe contract: a violation traps (§6.6).

### `README.md`
- Notation and the API-walkthrough bullets document `S` as a capacity, `s <=
  S` per call, the on-device uniform-`s` check and its failure mode,
  `total_num_tokens` for per-rank `s`, the `[plan.nvs_s, ...]` views,
  `s = 0` being unsupported and CUDA-graph bucketing; snippet comments use
  `[s, ...]` / `[plan.nvs_s, ...]` and note that `cu_seqlens` stays `[E+B]`.

### `benchmarks/`
- `bench_comm.py`: `--num-tokens s` (default `S`): `bench_one(...,
  num_tokens=None)` slices the routing to `s` tokens, sizes
  `hidden`/`weights`/`output`/grad tensors with `s`, plans with
  `allocate_planning_outputs(ctx, s)` and accounts bytes with `s`; an `s`
  column was added to the table and CSV. Its `launch_planning` calls pass
  `plan=`/`cu_seqlens=` by keyword.
- `bench_vs_deepep.py`: same keyword style on its two `launch_planning`
  calls; still runs at `s = S` (DeepEP's `num_max_tokens_per_rank` is `S`).

## 4. How to verify

### 4.1 Compile check (works on any Hopper GPU, no NVLink needed)
```python
# /home/ubuntu/MoonEP/.venv/bin/python
from moonep import planning, dispatch, combine
R,E,B,S,K,H,tp,num_sms = 8,64,8,256,8,2048,128,32
epn=E//R; N=S*K; NvS=N+(tp-1)*2*epn; NvS_padded=NvS+64
TPE_OFF=(NvS+3)&~3; PLAN_OFF=(TPE_OFF+R*E+3)&~3
planning_out=3*E*R+R*(E+B)+2*R*(E+B)+B*R+2*R
N4=(N+3)&~3; TOPK0_OFF=(PLAN_OFF+planning_out+3)&~3
ORDER_OFF=TOPK0_OFF+N4; ORDER0_OFF=ORDER_OFF+N4
BARRIER_OFF=ORDER0_OFF+N4; SRC_INFO_OFF=BARRIER_OFF+3
meta_stride=((SRC_INFO_OFF+NvS)*4+2*1024*1024-1)//(2*1024*1024)*(2*1024*1024)//4
planning._get_compiled(R,E,B,S,K,N,NvS,(N+2047)//2048,meta_stride,TPE_OFF,PLAN_OFF,
                       BARRIER_OFF,TOPK0_OFF,ORDER_OFF,ORDER0_OFF,tp,num_sms)
dispatch._get_compiled(H,R,S,K,E+B,NvS,NvS_padded,meta_stride,SRC_INFO_OFF,num_sms,True,True,0,True)
combine._get_compiled(H,R,S,K,NvS,NvS_padded,meta_stride,num_sms,True,0,True)
```
This snippet is from round 1. The runtime `(num_tokens, total_tokens,
uniform)` arguments are placeholders inside `_get_compiled`, so the calls
above should still be valid; if round 2 added a meta offset for the
per-rank token counts, take the argument list from the module's
`_get_compiled` rather than from here. A host-only sanity check of
`allocate_planning_outputs`, `_check_planning_outputs` and
`_check_dispatch_plan` with a fake `ctx` (`{'S','K','E','B','R','NvS',
'token_padding_extra','meta_buf'}`) is the cheapest way to exercise the
`nvs_s` arithmetic (`s in {1,5,131,256}`, `total in {R*s, R*s-3, s}`,
rejection of `0`, `S+1`, `total < s`, `total > R*S`).

### 4.2 Real runs (8 GPUs with NVLink multicast) — round 2 passed 2026-09-19
```bash
torchrun --nproc_per_node=8 -m pytest -s tests/test_planning.py tests/test_e2e.py
torchrun --nproc_per_node=8 -m pytest -s tests/test_dispatch.py tests/test_combine.py
torchrun --nproc_per_node=8 moe_moon_ep.py          # reference compare + replay + timing
torchrun --nproc_per_node=8 benchmarks/bench_comm.py --ep 8 --hidden 7168 --topk 8 \
    --unbalance-ratio 1.0 --num-tokens 1024       # partial-s comm time
```
`moe_moon_ep.py::main` uses ~48 GiB of expert weights per rank; shrink
`S, K, E, H, Hi` in `main` if needed. Expected first failure modes if
something is off in round 2: the per-rank-`s` planning test on `dst` /
`cu_seqlens` (cap rounding: ranks `r < total_slots % R` get the extra slot),
a `combine` shape assert on `[nvs_s, H]` (an `nvs_s` computed from `plan.N`
instead of the receive cap), a planning trap (a test accidentally passing a
different `s` per rank without `total_num_tokens`; look for the printed
message on the trapping rank, the others hang), or a hang in
`test_e2e_partial_tokens` at `s=1`/`s=5` (per-block range with empty
blocks).

## 5. Gotchas learned

- CuTe DSL: runtime loop bounds are fine with `cutlass.range(lo, hi, step)`;
  loop-carried scalars must be DSL-typed before the loop (`Int32(0)`), which
  is why the prefix pass initializes `cumsum = Int32(0)`.
- `cutlass.min`/`cutlass.max` exist and accept two runtime `Int32`s.
- Editing helper signatures: `copy_v4_remote` was `n: cutlass.Constexpr`;
  dropping the annotation is enough for it to take a runtime value — but a
  runtime `n` needs the head clamp, since the vector loop can no longer be
  proven to divide `n`.
- Kernel `__call__` layouts sized by constexpr `S`/`N` over tensors that are
  physically smaller are harmless as long as every access is bounded by the
  runtime count. Keep that invariant when adding loops.
- The kernels load user `topk`/`route_weights` with `assumed_align=16`; a
  `[S, K]` tensor sliced to `[s, K]` is still aligned, but a row-offset
  slice (`x[1:]`) or a non-contiguous view is not. The API copies such
  inputs; do not bypass it with `launch_*` directly on misaligned tensors.
- `nvs_s` is a function of the *total* token count, not of this rank's `s`.
  Deriving it from `plan.N` is wrong as soon as ranks differ.
- Traps are the failure mode for contract violations. They cannot be
  unit-tested under torchrun (the peers hang); keep the host-side asserts as
  the testable first line and treat the device checks as a backstop.
- The example relies on `torch._grouped_mm` ignoring rows outside `offs`
  (verified on torch 2.11 and 2.14); re-verify if torch is upgraded.

## 6. Remaining work, in priority order

### 6.1 Run and fix — done (rounds 1 and 2, 2026-09-19)
Rounds 1 and 2 both ran 2026-09-19 on 8x H200 (see Status for the counts).
Nothing is left to fix from those runs; commit per §6.8.

### 6.2 Shrink the returned views — done
`dispatch` returns `[plan.nvs_s, H]` / `[plan.nvs_s]`, `combine` accepts
them or the full shard (§1, §3 `api.py`). Remaining lever for consumers of
`torch._grouped_mm`: the padding bound `(token_padding - 1) * 2 * E/R` is
independent of `s` and can dominate at small `s`; a smaller `token_padding`
shrinks it.

### 6.3 Bound the capacity-proportional int32 work — deferred
Planning clears `src_info` over all `NvS`; the dispatch builder inits
`primary_packed`/`kmask` over `R*S` and scans `NvS` `src_info` slots three
times. Bounding these by `nvs_s`/`s` is safe only if clear and scan use the
same bound (stale entries past the bound must never be scanned). The traffic
is int32 metadata, under 1 MB per step at the shapes we care about, so this
stays deferred.

### 6.4 Per-rank differing `s` — done
`total_num_tokens` on the host, per-rank caps `base + (r < rem)` in the
kernel, `n_rt_0` from rank 0's tpe copy for rank 1's ordering (§1, §3
`planning.py`). Balance holds up to a remainder of `< R` slots.

### 6.5 Partial-`s` dispatch/combine unit tests — done
`s` parameters in `tests/test_dispatch.py` / `tests/test_combine.py`, a
multi-vblock `PLANNING_CASES` entry, per-rank-`s` tests (§3 Tests). They
have not run on 8 GPUs yet (§6.1).

### 6.6 Known limits
- `s = 0` is not supported on any rank (host assert; the kernels assume at
  least one token). Pass a dummy token.
- The multi-node fabric path (`use_fabric=True`) is untested at any `s`,
  including `s = S`.
- Contract violations (`sum(tpe) != s*K`, differing `s` without
  `total_num_tokens`, wrong `total_num_tokens`) trap the planning kernel;
  there is no graceful error across ranks.

### 6.7 CUDA graphs — follow-up
`s` is a kernel argument and `plan.nvs_s` sizes the returned views, so a
captured graph is valid for one `s` only; the README recommends bucketing
`s` where a framework needs static shapes. To capture one graph for all `s
<= S` the kernels would need `num_tokens` (and `total_tokens`) as device
scalars read at launch, plus capacity-sized views with `nvs_s` published as
a device value, which re-introduces the capacity-sized activation cost §6.2
removed. Not started.

### 6.8 Housekeeping
- Commit in logical pieces once tests pass (kernel arg plumbing; API; tests;
  README/example/benchmark). Do not commit `moonep/_C*.so` or `build/`.
- `moe_moon_ep.py`, `moe.py`, `moe_ep.py`, `moe_reference.py`,
  `bench_head2head.py`, `pyproject.toml` are tracked since `4818969`;
  `moe_moon_ep.py` is part of this diff and belongs in the
  README/example/benchmark commit.
