# Variable-length tokens in MoonEP (`s <= S`) — handoff

Status as of 2026-09-19: **implemented and compile-checked, never executed.**
All changes are uncommitted in the working tree on `master` (base commit
`2bd860b`). The machine this was written on has a single H100 with no NVLink
multicast, so `Buffer(...)` cannot be constructed there; every kernel change
was verified only by JIT-compiling it. First job for whoever continues: run
the tests on an 8-GPU node (commands in §4).

## 1. Goal and design

MoonEP's `Buffer(S, H, K, E, R, ...)` used to require exactly `S` tokens per
rank on every `dispatch`. Training with variable sequence lengths therefore
meant padding to `S`, which wastes planning, NVLink traffic, grouped-GEMM rows
and both backward passes in proportion to `S - s`, and the zero padding rows
all route to experts `0..K-1`, skewing the load balancer.

Design now in the tree:

- `S` is a **capacity** fixed at construction. It still sizes every buffer,
  every CuTe layout and stride (`N = S*K`, `NvS`, `NvS_padded`, meta offsets,
  the dedup key stride `src_rank * S + token`), and stays in every compile
  cache key. One JIT kernel set, one VMM/multicast mapping.
- `s` (`num_tokens`) is a **runtime `Int32` kernel argument**, derived on the
  host from `hidden_sh.shape[0]`. Planning, dispatch and combine loop to `s`.
  No padding tokens exist.
- The planner's per-rank receive target is `cap_rt = s*K` (was the constexpr
  `CAP = S*K`). The sum over ranks is exactly `R*cap_rt`, so the balancer
  still terminates with every rank at exactly `s*K` real tokens. The layout
  bound per rank is therefore `s*K + token_padding_extra <= NvS`.
- The plan remembers `s`: `MoonEPCommPlan.N == s*K`, `plan.num_tokens`
  property. Combine and both backward passes (plan reuse) inherit it.
- **Constraint: every rank must pass the same `s` in a step.** Rank 1 orders
  rank 0's top-k from a copy (`TOPK0_OFF` offload) and uses its own `n_rt`
  for it. Per-rank `s` is a follow-up (§6.4).

No device-to-host sync was added: `s` is a host-known shape.

## 2. Environment facts

- Python with torch: `/home/ubuntu/.venv/bin/python` (torch 2.11.0+cu130,
  nvidia-cutlass-dsl 4.4.2). Plain `python3` has no torch.
- If `from moonep._C import ...` fails (e.g. missing `FABRIC_HANDLE_BYTES`),
  the extension is stale: `python setup.py build_ext --inplace` (nvcc 13.0 in
  `/usr/local/cuda-13.0/bin`, no ninja, takes ~1 min).
- `moonep._C.nvl_multicast_supported()` returns False on this box; that is
  why nothing runs here. Kernel compile checks do work (§4.1).
- Unrelated pre-existing local edit: `setup.py` adds `"torch"` to
  `install_requires` (the user's own change, leave it).

## 3. What changed, file by file

Line numbers refer to the current working tree.

### `moonep/planning.py`
- `MoonEPCommPlan.num_tokens` property (`N // K`) — line 53.
- `copy_v4_remote(dst, dst_off, src, n, ...)` — line 289: `n` is no longer a
  `cutlass.Constexpr`; it now accepts a runtime `Int32` (used for the top-k
  and order copies below). Still works for constexpr `E`.
- `PlanningKernel.__call__` — line 374: new `num_tokens: Int32` arg between
  `rank` and `stream`, forwarded into `self.kernel(..., rank, num_tokens)`
  (line 398).
- `run_c1(..., pid, tid, n_rt)` — line 407: `nvb_rt = ceil(n_rt / 2048)`
  (line 418). Histogram (1a, line 438) and passA scatter (line 477) loop over
  `nvb_rt` vblocks with `off < n_rt` guards (lines 447, 485). The vblock
  prefix pass is now a runtime loop `for vb in cutlass.range(nvb_rt)` with
  `cumsum = Int32(0)` (line 465); rows of `local_hist` beyond `nvb_rt` are
  never read.
- `kernel(..., rank: Int32, num_tokens: Int32)` — line 533. `n_rt =
  num_tokens * K`, `cap_rt = n_rt` (lines 542-543). The constexpr aliases
  `S` and `CAP` inside the kernel body were removed because nothing reads
  them any more (`self.S` / `self.NvS_capacity` are untouched and remain
  cache keys). Restore the aliases if you prefer; they are dead code.
- Phase A top-k copy to rank 1 uses `n_rt` (line 622).
- Balancer: `balance[j] = group_tokens[k] - cap_rt` (line 699).
- All three `run_c1` call sites pass `n_rt`; the order copy back uses `n_rt`
  (lines 982-992).
- dst loop bound `seg = ceil(n_rt / num_sms)` (lines 1068-1069).
- Dedup canonicalization loop bound is `num_tokens` (lines 1103-1104).
- `_get_compiled`: extra `Int32(0)` in the `cute.compile` signature (line
  ~1167).
- `_launch_planning_kernel(..., num_tokens)` — line 1172; passes
  `Int32(num_tokens)` (line 1218).
- `allocate_planning_outputs(ctx, num_tokens=None)` — line 1227; `None`
  means capacity `S`; `dst` sized `s*K`.
- `_check_planning_outputs`: `plan.N % K == 0 and 0 < plan.N <= S*K` (line
  1290).
- `launch_planning`: asserts `topk_experts_flat.numel() == plan.N` and passes
  `plan.num_tokens` (lines 1350-1357).

Unchanged on purpose: the `src_info` clear loops over the full `NvS` (line
~997) and the tpe-driven phases (allocation, cu_seqlens, experts_to_copy,
zero-fill) need no token count at all.

### `moonep/dispatch.py`
- `__call__` (line 167) and `kernel` (line 290) take `num_tokens: Int32`;
  forwarded at line 259.
- Per-block token range (lines 354-360): `tpb = ceil(num_tokens/num_sms)`,
  `s_end = cutlass.max(cutlass.min(s_beg + tpb, num_tokens), s_beg)`. The
  `max` clamp fixes a latent negative `n_tok` when `bidx*tpb > num_tokens`
  (reachable now that `s` can be below `num_sms`).
- `_get_compiled`: extra `Int32(0)` (line 762).
- `_check_dispatch_plan`: `plan.N == hidden_sh.shape[0] * K` and `0 < s <= S`
  (lines 802-808); `dst` checked against `plan.N`.
- `launch_dispatch`: `hidden_sh` may be `[s, H]` with `1 <= s <= S` (line
  ~898); `route_weights_sk` must be `[s, K]` (line 915); passes
  `Int32(num_tokens)` (line 1000).
- Unchanged: the dedup builder still inits `primary_packed`/`kmask` over
  `R*S` and scans all `NvS` `src_info` slots (planning cleared them to -1).
  Correct, int32-only cost proportional to the capacity (§6.3).

### `moonep/combine.py`
- Same pattern: `num_tokens: Int32` in `__call__` (line 140), `kernel` (line
  224), forwarded (line 200); per-block range with clamp (lines 302-308);
  `_get_compiled` extra `Int32(0)` (line 566).
- `launch_combine`: `output_sh` is `[s, H]`, `dst.numel() == s*K`,
  `output_sk` is `[s, K]` (lines 624-636); passes `Int32(num_tokens)` (line
  673).
- The `gmem_out` layout in `__call__` is still declared `(S, H)` while the
  real tensor has `s` rows. That is safe because only rows `< num_tokens` are
  touched and no TMA descriptor is built from it (raw `cp.async.bulk`
  addresses). Same for dispatch's `gmem_src` and both `dst` layouts.

### `moonep/api.py`
- `Buffer.dispatch` (lines 786-813): `num_tokens = hidden_sh.shape[0]`,
  asserts `1 <= s <= S` (message contains "capacity"), `topk.numel() ==
  s*K`, allocates the plan with `num_tokens`; on plan reuse asserts
  `plan.num_tokens == s` (message contains "tokens"). Docstrings updated.
- `Buffer.combine` (lines 1040-1053): outputs allocated as
  `[plan.num_tokens, H]` and `[plan.num_tokens, K]`.
- Still `[NvS, H]`: `hidden_nvsh` returned by dispatch (fresh
  `torch.empty_like(hidden_buf_local)` or the zero-copy view) and the
  `hidden_nvsh.shape == (NvS, H)` assert in `combine` (line ~1004). See §6.2.

### `moe_moon_ep.py` (example, untracked file)
- `forward` (line 175) runs any `s <= S` directly; the old pad-to-S wrapper
  and `forward_full_S` are gone.
- Separately, the per-layer host sync was removed: the grouped GEMMs take the
  whole `[NvS, H]` shard with `cu_seqlens` as `offs` (lines 204-216) instead
  of slicing to `int(cu_seqlens[-1].item())`. Verified on this GPU that
  `torch._grouped_mm` leaves rows past `offs[-1]` as exact zeros in forward
  and that NaN grads on those rows reach neither weight grads nor grouped-row
  input grads. `z_full` is gone; `w_nvs` is `nan_to_num`'d over all rows.

### Tests
- `tests/planning_reference.py` (lines 45-55): derives `S`/`N` from the
  routing input (`topk.numel() // K`) and sets `CAP = N`; asserts
  `S <= ctx["S"]`. The reference now mirrors `cap_rt`.
- `tests/kernel_test_utils.py`: `planning_invariant_errors(..., num_tokens=
  None)` (line 327). Meta-layout expectations use `N_capacity = case.S*K`;
  the `dst` shape check uses the step's `N`.
- `tests/test_planning.py`: new
  `test_planning_partial_tokens_matches_reference` (line 262), parametrized
  over all `PLANNING_CASES`, plans `s in {1, S//2, S-1}` on a Buffer built
  for `S` and compares to the reference plus invariants. Added `import
  torch`.
- `tests/test_e2e.py`: `_e2e_partial_tokens` (line 421) and
  `test_e2e_partial_tokens` (line 476): for `s in (131, 5, 1, 256)` on
  `S=256, K=4`: fresh plan sized `s`, `s*K <= cu_seqlens[-1] <= s*K +
  token_padding_extra`, sync == async, identity round trip
  `combine(dispatch(x)) ≈ K*x` (`allclose rtol=2e-2`, because duplicate slots
  are pre-reduced in bf16 in the combine prologue), exact route-weight
  gather, plan reuse, rejection of `hidden[:s-1]` with a mismatched plan, and
  rejection of `s = S+1`. Also called from `__main__`.
- Not touched: `tests/test_dispatch.py`, `tests/test_combine.py` (they run
  at full `S` and still pass their shape checks; see §6.5 for partial-`s`
  variants).

### `README.md`
- Notation (line 43) and a new bullet in the API walkthrough (line 81)
  document `S` as a capacity and `s <= S` per call; code comments in the
  dispatch/combine snippets use `[s, ...]`.

## 4. How to verify

### 4.1 Compile check (works on any Hopper GPU, no NVLink needed)
```python
# /home/ubuntu/.venv/bin/python
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
All three compiled (and the no-weights / reuse variants) at the time of
writing. A host-only sanity check of `allocate_planning_outputs`,
`_check_planning_outputs` and `_check_dispatch_plan` with a fake `ctx`
(`{'S','K','E','B','R','NvS','meta_buf'}`) also passed for `s in
{1,5,131,256}` and rejected `0`, `S+1` and a plan/hidden mismatch.

### 4.2 Real runs (8 GPUs with NVLink multicast)
```bash
torchrun --nproc_per_node=8 -m pytest -s tests/test_planning.py tests/test_e2e.py
torchrun --nproc_per_node=8 -m pytest -s tests/test_dispatch.py tests/test_combine.py   # regression at full S
torchrun --nproc_per_node=8 moe_moon_ep.py          # reference compare + replay + timing
```
`moe_moon_ep.py::main` uses ~48 GiB of expert weights per rank; shrink
`S, K, E, H, Hi` at line ~300 if needed. Expected first failure modes if
something is off: `test_planning_partial_tokens_matches_reference` on
`dst`/`cu_seqlens` (planner loop bounds), or a hang in `test_e2e_partial_tokens`
at `s=1`/`s=5` (per-block range with empty blocks; check the `cutlass.max`
clamp actually lowers as intended).

## 5. Gotchas learned

- CuTe DSL: runtime loop bounds are fine with `cutlass.range(lo, hi, step)`;
  loop-carried scalars must be DSL-typed before the loop (`Int32(0)`), which
  is why the prefix pass initializes `cumsum = Int32(0)`.
- `cutlass.min`/`cutlass.max` exist and accept two runtime `Int32`s.
- Editing helper signatures: `copy_v4_remote` was `n: cutlass.Constexpr`;
  dropping the annotation is enough for it to take a runtime value.
- Kernel `__call__` layouts sized by constexpr `S`/`N` over tensors that are
  physically smaller are harmless as long as every access is bounded by the
  runtime count. Keep that invariant when adding loops.
- The example relies on `torch._grouped_mm` ignoring rows outside `offs`
  (verified on torch 2.11); re-verify if torch is upgraded.
- The IDE flagged the removed `S`/`CAP` aliases as "not accessed" before
  removal; do not re-add uses of them by accident.

## 6. Remaining work, in priority order

### 6.1 Run and fix
Run §4.2. Nothing in this change has executed on hardware.

### 6.2 Shrink the returned views (biggest perf item)
Today `dispatch` returns `hidden_nvsh`/`route_weights_nvs` of size `NvS`
regardless of `s`, so a consumer FFN allocates and runs elementwise ops over
capacity-sized activations. With `cap_rt = s*K`, the planner places every
slot of this rank inside `[0, s*K + token_padding_extra)`, so:
- In `api.dispatch`: compute `nvs_s = s*K + ctx['token_padding_extra']`,
  return `ctx['hidden_buf_local'][:nvs_s]` (zero-copy) or
  `torch.empty(nvs_s, H)` copied from that prefix; same for the weights view.
- In `api.combine`: accept `hidden_nvsh.shape == (nvs_s, H)` (line ~1004
  asserts `(NvS, H)` today); the zero-copy `data_ptr()` check still works
  for a prefix view; the boundary `copy_` becomes a prefix copy.
- `dispatch_epilogue.py:391` and `combine_prologue.py:572` assert on
  `ctx['hidden_buf_local']` (still full `NvS`), so they need no change.
- Update `tests/test_e2e.py` comparisons that slice `[:total]` accordingly.

### 6.3 Bound the capacity-proportional int32 work
Planning clears `src_info` over all `NvS` (planning.py ~997); the dispatch
builder inits `primary_packed`/`kmask` over `R*S` (dispatch.py ~536) and
scans `NvS` `src_info` slots three times (~556+). Bounding all of them by
`nvs_s`/`s` is safe only if clear and scan use the same bound (stale entries
past the bound must never be scanned). Cost today is a few MB of int32
traffic per step; low priority.

### 6.4 Per-rank differing `s`
Requires rank 0 to publish its `n_rt` next to the top-k copy (`TOPK0_OFF`
region, or a new meta slot) so rank 1 can order it, and `cap_rt` becomes the
average `sum_r(s_r)*K / R` (rank 0 can compute it from `group_tokens`).
Exact balance then holds only up to a remainder of `< R` slots.

### 6.5 Partial-`s` variants of dispatch/combine unit tests
`tests/test_dispatch.py::_traceable_weights` encodes `src_rank * S * K`
strides and `_verify_dispatch_by_dst` reshapes `dst` with `case.S`; add an
`s` parameter (default `case.S`) to both and to `_plan_and_dispatch`
(`allocate_planning_outputs(ctx, s)`, sliced `make_topk` output with
recomputed `tpe`). Mirror for `tests/test_combine.py`.

### 6.6 Housekeeping
- Commit in logical pieces once tests pass (kernel arg plumbing; API; tests;
  README/example). Do not commit `moonep/_C*.so` or `build/`.
- `moe_moon_ep.py`, `moe.py`, `moe_ep.py`, `moe_reference.py`,
  `bench_head2head.py`, `pyproject.toml` are untracked user files; whether
  they belong in the repo is the user's call.
