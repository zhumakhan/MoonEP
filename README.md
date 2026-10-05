# MoonEP

MoonEP is an Expert Parallelism communication library that keeps token loads perfectly balanced across ranks via dynamic redundant experts.

**Notation**: `S` = token capacity per rank, `s` = tokens per call (`1 <= s <= S`), `K` = routed top-k per token.

1. **Perfect balance**: every rank receives the same number of tokens (`s × K` when every rank passes `s` tokens), no matter how skewed the routing is. A small number of redundant experts is planned online from the current router outputs and prefetched before expert computation; their gradients are reduced back to their home ranks in the backward pass.
2. **Online planning**: a near-optimal GPU planning kernel with negligible overhead
3. **Zero copy and static shapes**: fused permute/unpermute — tokens are sent directly to their expert-grouped positions on remote ranks and buffer views are returned to the computation. Only a fixed `S × K` buffer is needed, and every shape depends only on the step's token count `s`, never on the routing, so no per-layer MoE host synchronization is needed.

## Performance

Both benchmarks run on H20 with EP=8, sweeping the router imbalance:

$$\text{maxvio} = \max_e \left( \frac{T_e}{\bar{T}} \right) - 1$$

where $T_e$ is the number of tokens routed to expert $e$, and $\bar{T}$ is the expected tokens per expert under perfect balance (maxvio = 0 means perfectly balanced).

**Communication vs DeepEP v2** ([benchmarks/bench_vs_deepep.py](benchmarks/bench_vs_deepep.py)):

<img src="figure/comm_vs_deepep.png" alt="MoonEP vs DeepEP v2 communication" width="800">

- **Zero copy makes raw communication faster**: tokens are written directly to their final expert-grouped positions on remote ranks — no permute in, no permute out — and views of the communication buffer are handed straight to the computation, eliminating the comm-buffer → user-buffer copy that dominates the epilogue. MoonEP's comm time is consistently below DeepEP v2 at every imbalance level.
- **Perfect balance makes it immune to imbalance**: MoonEP's comm time stays almost flat as maxvio grows, while DeepEP v2 — whose latency is set by the hottest rank — degrades steadily.
- **The comparison counts MoonEP's extra kernels**: MoonEP adds planning and weight-prefetch kernels that DeepEP does not need, and they are already stacked in the bars above. Even with the whole critical path included, total dispatch time is on par with DeepEP v2's dispatch alone and pulls ahead under imbalance, while combine is significantly faster at every level.

**End-to-end training**:

<img src="figure/e2e_vs_deepep.png" alt="MoonEP vs DeepEP e2e training" width="800">

- **DeepEP degrades with imbalance**: the hottest ranks receive more tokens, so iteration time climbs steadily as maxvio grows; meanwhile the ever-changing activation shapes fragment GPU memory, until training OOMs at high imbalance.
- **MoonEP is unaffected**: every rank always computes exactly `s × K` tokens per layer, so iteration time stays flat at every imbalance level; fully static memory shapes mean no fragmentation, and training never OOMs.

## Supported Devices

- NVIDIA GPU
- Zhenwu PPU (under review, coming soon)

## Usage

### Integration

**Notation**: `S` = token capacity per rank (fixed at `Buffer` construction; sizes every buffer), `s` = actual input tokens per rank in a given call (`1 <= s <= S`; the same on every rank unless `total_num_tokens` is passed), `K` = routed top-k per token, `E` = total routed experts in the EP group, `R` = number of EP ranks (EP comm size), `epn = E/R` = local experts and prefetch/reduce slots per rank, `NvS` = dispatched token slots per rank (`S × K` real-token capacity plus per-VM-group padding, rounded up to `token_padding`), `nvs_s` = `plan.nvs_s`, the slots dispatch may fill this step (receive cap plus padding bound, rounded up to `token_padding`, `<= NvS`), `H` = hidden size, `H'` = expert FFN intermediate size.

MoonEP's communication API receives each projection as a local expert tensor `[epn, H, H']` plus an all-rank prefetch-buffer view `[R, epn, H, H']`. The integrating framework exposes this rank's local experts followed by its local prefetch slice as one contiguous `[2*epn, H, H']` compute view for the VM group GEMM; the planner-produced `cu_seqlens[2*epn]` selects the active rows.

#### Weight buffer

<img src="figure/weight_buffer.png" alt="MoonEP weight buffer layout" width="1000">

For each expert projection (gate/up/down), the framework builds two related views:

- **Communication view `[R, epn, H, H']`**: all ranks' prefetch pools mapped directly for `buffer.prefetch_weight`.
- **Compute view `[2*epn, H, H']`**: rows `[0, epn)` alias this rank's local parameter weights; rows `[epn, 2*epn)` alias this rank's prefetch slice. The VM group GEMM and `cu_seqlens` use this compact row order without mapping the other ranks' parameter weights.

Each rank's prefetch pool is process-global and shared by all layers, so the extra physical cost is `epn` expert weights per projection in total, not per layer.

Each rank has `epn` prefetch slots. The planner moves experts from at most one remote home group to each destination rank, so these slots cover every remote expert segment.

#### Gradient buffers (training only)

<img src="figure/grad_buffer.png" alt="MoonEP grad buffer and grad reduce" width="1000">

Training mirrors the compact weight layout in fp32:

- **Local grad `[epn, H, H']`**: this rank's parameter grads.
- **Compute grad view `[2*epn, H, H']`**: the local grad followed by this rank's reduce-buffer slice. The tail contains temporary prefetch-slot grads and stays separate from the framework's own parameter-grad reduction.
- **Reduce buffer**: every rank maps all `R` reduce buffers as one `[R, epn, H, H']` view. `reduce_grad` lets each rank read the slots holding its own experts' grads from every rank's reduce buffer (remote reads over NVLink), accumulate them into its local parameter grad, then zero its own consumed slots for the next microbatch.

### API walkthrough

```python
from moonep import Buffer

buffer = Buffer(S=4096, H=7168, K=8, E=256, num_ep_ranks=8,
                num_sms=32, token_padding=128)
```

- `num_sms=None` defaults to 32. The current implementation derives `epn = E // num_ep_ranks` internally.
- `dispatch` / `combine` / `prefetch_weight` / `reduce_grad` all accept `async_finish=True` to run on the comm stream and return a CUDA event.
- `combine` defaults to `inter_rank_sync=True`, which runs an explicit rank sync before staging. Pass `inter_rank_sync=False` to skip this pre-staging sync; the combine kernel still performs its own entry cross-rank barrier.
- `S` is a capacity, not a per-call shape. Each `dispatch` / `combine` call may pass any `s` tokens with `1 <= s <= S` (`s = hidden_sh.shape[0]`; `route_weights_sk` / `topk_experts_sk` must then be exactly `[s, K]`; non-contiguous or non-16-byte-aligned inputs, e.g. slices, are copied to a contiguous aligned tensor first). Buffers are sized once for `S`; the returned `plan` remembers `s` (`plan.num_tokens`), so `combine` and both backward passes that reuse the plan produce `[s, ...]` outputs and assert that `s` matches. `prefetch_weight` and `reduce_grad` do not depend on `s`. `s = 0` is not supported: pass a dummy token.
  - **Same `s` on every rank (default).** With `total_num_tokens=None`, every rank must pass the same `s` in a step, and the planner verifies this on device: every rank checks `sum(tokens_per_expert) == s × K` and rank 0 compares every rank's total against its own. A violation traps the planning kernel with a printed message: the local `sum(tokens_per_expert) != s × K` check traps on that rank, the cross-rank checks trap on rank 0, which sees every rank's totals (a CUDA error there; the peers hang at the next cross-rank barrier until `torchrun` kills them). Every rank then receives exactly `s × K` slots.
  - **Per-rank `s`.** Pass `total_num_tokens=<sum of s over the EP group>` (host-known, e.g. from the dataloader or one `all_gather` of ints) and each rank may pass its own `s`. Every rank receives `floor(total_num_tokens × K / R)` slots, plus one for the first `(total_num_tokens × K) mod R` ranks; the planner checks the gathered totals against `total_num_tokens × K` on every rank and traps otherwise, so `total_num_tokens` must be identical on all ranks. `total_num_tokens` is ignored when a saved `plan` is reused.
  - **Step-sized views.** `dispatch` returns `hidden_nvsh` as `[plan.nvs_s, H]` and `route_weights_nvs` as `[plan.nvs_s]`, where `plan.nvs_s` = `ceil(total_num_tokens × K / R)` (an upper bound of every rank's receive cap, so the same value on every rank) + the segment padding bound (`(token_padding - 1) × 2 × epn`), rounded up to `token_padding` like `NvS`, so it equals `NvS` at `s = S` and never exceeds it. The planner keeps every slot of this rank, and hence `cu_seqlens[-1]`, inside that prefix, so `torch._grouped_mm(hidden_nvsh, w, offs=cu_seqlens)` works unchanged and the expert FFN's activations scale with the step, not the capacity. `combine` accepts either those `[plan.nvs_s, ...]` views or the full `[NvS, ...]` shard. `cu_seqlens` stays `[2*epn]`. With `torch._grouped_mm` as the consumer, a smaller `token_padding` shrinks the padding bound further.
  - **CUDA graphs / `torch.compile`.** `s` (and `total_num_tokens`) are kernel arguments and `plan.nvs_s` sizes the returned views, so a captured graph (or a static-shape compile) is only valid for the `(s, total_num_tokens)` it was captured with. Run the comm kernels at the exact `s` wherever you can (they loop over `s`; no padding tokens exist) and bucket `s` only where a framework needs static shapes: capture one graph per bucket and pad the input to the bucket's `s`.

#### dispatch fwd

```python
hidden_nvsh, route_weights_nvs, cu_seqlens, plan = buffer.dispatch(
    hidden_sh,          # [s, H] bf16, any 1 <= s <= S
    route_weights_sk,   # [s, K] fp32
    topk_experts_sk,    # [s, K] int32
    tokens_per_expert,  # [E] int32, local count
)
# hidden_nvsh:       [plan.nvs_s, H] bf16 — dispatched tokens in physical VM group order
#                    (nvs_s = receive cap + padding bound, token_padding aligned, <= NvS;
#                    scales with s, not S)
# route_weights_nvs: [plan.nvs_s] fp32
# cu_seqlens:        [2*epn] int32 — padded token end offset per VM group row; stays
#                    [2*epn] for any s, and cu_seqlens[-1] <= plan.nvs_s
# plan:              MoonEPCommPlan — save it for prefetch/combine and both backward passes
#
# per-rank s: buffer.dispatch(..., total_num_tokens=sum_of_s_over_the_EP_group)

buffer.prefetch_weight(
    plan=plan,
    local_gate_weight=local_gate_weight,        # [epn, H, H'] bf16
    local_up_weight=local_up_weight,            # [epn, H, H'] bf16
    local_down_weight=local_down_weight,        # [epn, H, H'] bf16
    gate_prefetch_buffer=gate_prefetch_buffer,  # [R, epn, H, H'] bf16
    up_prefetch_buffer=up_prefetch_buffer,      # [R, epn, H, H'] bf16
    down_prefetch_buffer=down_prefetch_buffer,  # [R, epn, H, H'] bf16
)
# The framework's GEMM view aliases local_*_weight followed by this rank's
# *_prefetch_buffer slice as [2*epn, H, H'].
```

#### dispatch bwd

Backward of dispatch: sum each token's K dispatched grad copies back to token-major — a combine — and reduce duplicated experts' weight grads back to their home ranks.

```python
grad_hidden_sh, _, _ = buffer.combine(
    plan=plan,
    hidden_nvsh=grad_hidden_nvsh,    # [plan.nvs_s, H] bf16 (or the full [NvS, H] shard)
)
# grad_hidden_sh: [s, H] bf16 (s = plan.num_tokens)

buffer.reduce_grad(
    plan=plan,
    local_gate_grad=local_gate_grad,          # [epn, H, H'] fp32
    local_up_grad=local_up_grad,              # [epn, H, H'] fp32
    local_down_grad=local_down_grad,          # [epn, H, H'] fp32
    gate_reduce_buffer=gate_reduce_buffer,  # [R, epn, H, H'] fp32
    up_reduce_buffer=up_reduce_buffer,      # [R, epn, H, H'] fp32
    down_reduce_buffer=down_reduce_buffer,  # [R, epn, H, H'] fp32
)
# The framework's GEMM grad view aliases local_*_grad followed by this rank's
# *_reduce_buffer slice as [2*epn, H, H'].
```

#### combine fwd

```python
output_sh, gathered_route_weights_sk, _ = buffer.combine(
    plan=plan,
    hidden_nvsh=expert_output_nvsh,       # [plan.nvs_s, H] bf16 (or the full [NvS, H] shard)
    route_weights_nvs=route_weights_nvs,  # [plan.nvs_s] fp32 (or [NvS]), optional
)
# output_sh:                 [s, H] bf16 — combined token-major output (s = plan.num_tokens)
# gathered_route_weights_sk: [s, K] fp32 or None — routing weights gathered back to token-major
```

#### combine bwd

Backward of combine: scatter the output grad back to VM group order by re-dispatching with the saved plan — planning is skipped and no prefetch is needed.

```python
grad_expert_output_nvsh, _, _, _ = buffer.dispatch(
    grad_output_sh,    # [s, H] bf16 (s must equal plan.num_tokens)
    plan=plan,
)
# grad_expert_output_nvsh: [plan.nvs_s, H] bf16
```

#### zero_copy

By default `dispatch` returns fresh tensors and `combine` first copies its inputs into the NVL shard. With `zero_copy=True` on both sides, `dispatch` returns views of the communication buffer and the expert FFN reads/writes them in place — no boundary copy at all:

```python
hidden_nvsh, route_weights_nvs, cu_seqlens, plan = buffer.dispatch(
    hidden_sh, route_weights_sk, topk_experts_sk, tokens_per_expert,
    zero_copy=True,
)
# hidden_nvsh / route_weights_nvs are [plan.nvs_s, ...] prefix views of the NVL buffer;
# the expert FFN must write its output in place on hidden_nvsh
output_sh, gathered_route_weights_sk, _ = buffer.combine(
    plan=plan,
    hidden_nvsh=hidden_nvsh,
    route_weights_nvs=route_weights_nvs,
    zero_copy=True,  # asserts the inputs are exactly the views from dispatch
)
```

- The views alias buffer state that the next `dispatch` / `combine` overwrites — do not hold them across communication calls (autograd must not save them for backward; that case requires `zero_copy=False`).

```python
# explicitly release VMM/NVLink resources held by the Buffer before destroying the process group
buffer.destroy()
```

## Build & Test

```bash
pip install -e .

# run tests (requires multiple GPUs + NVLink)
torchrun --nproc_per_node=8 -m pytest tests/test_planning.py
torchrun --nproc_per_node=8 -m pytest tests/test_dispatch.py
torchrun --nproc_per_node=8 -m pytest tests/test_combine.py
torchrun --nproc_per_node=8 -m pytest tests/test_e2e.py
torchrun --nproc_per_node=8 -m pytest tests/test_grad_reduce.py
torchrun --nproc_per_node=8 -m pytest tests/test_prefetch.py
```

## Acknowledgments

This library is inspired by the following works:

- [DeepEP](https://github.com/deepseek-ai/DeepEP)
- [Echo](https://arxiv.org/abs/2603.07685)
- [UltraEP](https://github.com/Dots-Infra/UltraEP)
- AcclEP (Alibaba's EP communication library)

## Citation

```bibtex
@misc{moonep2026,
      title={MoonEP: A Perfectly Balanced Expert Parallelism Library via Dynamic Redundant Experts},
      author={Yutian Chen, Cong Li, Yucheng Wang, Ming Wei},
      year={2026},
      publisher = {GitHub},
      howpublished = {\url{https://github.com/MoonshotAI/MoonEP}},
}
```
