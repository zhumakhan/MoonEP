"""
Expert pools shared by every MoE layer of a model.

MoonEP needs a layer's prefetch slots and reduce slots only while that layer
computes, so one set of pools per process serves all layers: the extra memory is
``epn`` expert copies per projection in total, not per layer. Each layer keeps
its persistent expert weights and grads in the framework's own memory and
borrows the pools while it computes:

  - forward: copy the layer's bf16 experts into ``staging(n)``, dispatch, then
    ``push(plan)``; the grouped GEMM reads ``weights(n)`` (this rank's experts,
    then its prefetch slots, the row order of ``cu_seqlens[2*epn]``).
  - backward: the pools hold whichever layer pushed last. Unless that is still
    this layer, copy its experts into ``staging(n)`` again and ``push`` its
    saved plan before its expert GEMMs. The weight-grad GEMM adds dW into
    ``grads(n)``, and ``reduce(plan)`` sums every rank's slot grads of this
    rank's experts into ``local_grads(n)`` and zeroes the consumed slots. Those
    rows are shared as well: move them into the layer's own grad storage (and
    zero them) before another layer accumulates into ``grads(n)``.

Layout, per projection ``n`` in gate/up/down with per-expert weight shape
``(out, in)``, and per dtype (bf16 weights, fp32 grads): every EP rank allocates
ONE VMM chunk of ``[2*epn, out, in]`` elements, its ``epn`` staging (local grad)
rows followed by its ``epn`` slot rows, rounded up to the VMM granularity; the
padded tail is an unused gap. All ``R`` chunks are mapped back to back into one
virtual range of ``R * P`` elements (``P`` = padded chunk elements)::

    rank r: | [0, epn): staging / local grads | [epn, 2*epn): slots | gap |
            ^ r*P                             ^ r*P + epn*out*in          ^ (r+1)*P

  - ``weights(n)`` / ``grads(n)``: this rank's contiguous ``[2*epn, out, in]``.
  - ``staging(n)`` / ``local_grads(n)``: their first ``epn`` rows.
  - ``prefetch_buffer(n)`` / ``reduce_buffer(n)``: ``[R, epn, out, in]`` views of
    every rank's slot rows (rank stride ``P``, storage offset ``epn*out*in``),
    the rank-strided pools ``Buffer.prefetch_weight`` / ``Buffer.reduce_grad``
    accept. ``prefetch_buffer(n)[rank]`` is ``weights(n)[epn:]`` and
    ``reduce_buffer(n)[rank]`` is ``grads(n)[epn:]``.

Peers write this rank's memory only through ``push`` (its slots) and read it
only through ``reduce`` (its slot grads); staging and local-grad rows are local.
All views of one projection and dtype are views of one mapping, so they share
one autograd version counter: keep them out of ``save_for_backward``.
"""

from collections.abc import Mapping

import torch
import torch.distributed as dist

from ._C import get_vmm_granularity, nvl_dist_alloc, nvl_release_mem_handle
from .api import Buffer
from .buffer import _map_nvl_dist_tensor, _use_fabric_for_group
from .grad_reduce import GradReduceKernel
from .inter_rank_sync import launch_inter_rank_sync
from .planning import MoonEPCommPlan
from .prefetch import PrefetchKernel

# torch 2.14 renamed ``all_gather_into_tensor`` (the old name warns).
_all_gather_single = getattr(dist, "all_gather_single", None) or dist.all_gather_into_tensor

# The prefetch and grad-reduce kernels tile both inner dims in 128 x 128 blocks.
_DIM_MULTIPLE = max(
    PrefetchKernel.M_BLOCK, PrefetchKernel.N_BLOCK,
    GradReduceKernel.M_BLOCK, GradReduceKernel.N_BLOCK,
)


def _group_ranks(group: dist.ProcessGroup | None) -> tuple[int, ...]:
    return tuple(dist.get_process_group_ranks(
        group if group is not None else dist.group.WORLD))


class ExpertPools:
    """Process-global VMM expert weight and grad pools of one MoonEP EP group.

    Construction, ``push``, ``reduce`` and ``destroy`` are collective over the
    Buffer's EP group; every rank calls them in the same order. ``push`` and
    ``reduce`` run synchronously on the current CUDA stream, like the Buffer's
    ``async_finish=False`` calls.
    """

    PROJECTIONS = ("gate", "up", "down")

    def __init__(
        self,
        buffer: Buffer,
        group: dist.ProcessGroup | None,
        shapes: Mapping[str, tuple[int, int]],
    ):
        """Allocate and map the pools. Collective over ``group``.

        Args:
            buffer: the MoonEP Buffer whose plans drive ``push`` / ``reduce``;
                ``epn = E // R`` comes from it.
            group: the Buffer's EP process group (None = the default group,
                as for Buffer).
            shapes: per-expert ``(out, in)`` weight shape of each of
                gate/up/down, e.g. ``gate=(H', H), up=(H', H), down=(H, H')``.
                Both dims must be multiples of 128; MoonEP's kernels do not
                care what they mean.
        """
        assert isinstance(buffer, Buffer), (
            f"ExpertPools: buffer must be a moonep.Buffer, got {type(buffer).__name__}"
        )
        ctx = buffer._require_ctx()
        R, E, rank = int(ctx['R']), int(ctx['E']), int(ctx['rank'])
        self.shapes = self._parse_shapes(shapes)

        assert dist.is_available() and dist.is_initialized(), (
            "ExpertPools: torch.distributed must be initialized"
        )
        group_size = dist.get_world_size(group=group)
        assert group_size == R, (
            f"ExpertPools: group size {group_size} must equal the Buffer's EP size "
            f"R={R} (-1 means this rank is not in the group)"
        )
        assert _group_ranks(group) == _group_ranks(ctx['group']), (
            f"ExpertPools: group ranks {list(_group_ranks(group))} must be the "
            f"Buffer's EP group ranks {list(_group_ranks(ctx['group']))}"
        )

        self.buffer = buffer
        self.group = group
        self.R = R
        self.rank = rank
        self.epn = E // R
        self._destroyed = False
        self._mappings: list[torch.Tensor] = []
        self._keepalives: list[torch.Tensor] = []
        self._weights: dict[str, torch.Tensor] = {}
        self._staging: dict[str, torch.Tensor] = {}
        self._prefetch: dict[str, torch.Tensor] = {}
        self._grads: dict[str, torch.Tensor] = {}
        self._local_grads: dict[str, torch.Tensor] = {}
        self._reduce: dict[str, torch.Tensor] = {}

        granularity = int(get_vmm_granularity())
        self._check_same_config(granularity)
        # Collective (an all-gather) unless MOONEP_MEM_HANDLE_TYPE=fd; decided
        # once for all chunks.
        self._use_fabric = bool(_use_fabric_for_group(group))
        # Every rank maps the same chunks in the same order (the handle
        # exchange is collective).
        for name in self.PROJECTIONS:
            self._weights[name], self._prefetch[name] = self._map_chunk(
                name, torch.bfloat16, granularity)
            self._staging[name] = self._weights[name][:self.epn]
            self._grads[name], self._reduce[name] = self._map_chunk(
                name, torch.float32, granularity)
            self._local_grads[name] = self._grads[name][:self.epn]
        # _map_chunk zeroed this rank's chunks (cuMemCreate memory is not
        # zero-filled; reduce_grad clears only the slots it consumes). Peers
        # touch them only after this barrier.
        torch.cuda.synchronize()
        dist.barrier(group=group)

    # ---- construction helpers ------------------------------------------------

    @classmethod
    def _parse_shapes(cls, shapes) -> dict[str, tuple[int, int]]:
        names = sorted(shapes, key=str) if isinstance(shapes, Mapping) else None
        assert names is not None and set(names) == set(cls.PROJECTIONS) \
            and len(names) == len(cls.PROJECTIONS), (
                f"ExpertPools: shapes must map exactly {'/'.join(cls.PROJECTIONS)} "
                f"to per-expert (out, in) weight shapes, got {names or type(shapes).__name__}"
            )
        parsed = {}
        for name in cls.PROJECTIONS:
            shape = shapes[name]
            assert isinstance(shape, (tuple, list, torch.Size)) and len(shape) == 2, (
                f"ExpertPools: shapes[{name!r}] must be a per-expert (out, in) pair, "
                f"got {shape!r}"
            )
            out, inn = (int(d) for d in shape)
            assert out > 0 and inn > 0 and out % _DIM_MULTIPLE == 0 \
                and inn % _DIM_MULTIPLE == 0, (
                    f"ExpertPools: {name} weight shape (out={out}, in={inn}) must have "
                    f"both dims positive multiples of {_DIM_MULTIPLE} (the tile of "
                    "MoonEP's prefetch and grad-reduce kernels)"
                )
            parsed[name] = (out, inn)
        return parsed

    def _check_same_config(self, granularity: int) -> None:
        """Every rank must map chunks of the same size (collective)."""
        config = [self.epn, granularity]
        for name in self.PROJECTIONS:
            config.extend(self.shapes[name])
        mine = torch.tensor(config, dtype=torch.int64, device="cuda")
        everyone = torch.empty(self.R, mine.numel(), dtype=torch.int64, device="cuda")
        _all_gather_single(everyone, mine.view(1, -1), group=self.group)
        differ = (everyone != mine).any(dim=1).nonzero().flatten().tolist()
        assert not differ, (
            f"ExpertPools: EP ranks {differ} differ from rank {self.rank} in "
            f"(epn, VMM granularity, gate/up/down shapes) = {config}; got "
            f"{everyone.tolist()}"
        )

    def _map_chunk(
        self, name: str, dtype: torch.dtype, granularity: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Allocate this rank's ``[2*epn, out, in]`` chunk of projection
        ``name`` and map every rank's chunk into one range. Returns (this
        rank's chunk view, the all-rank ``[R, epn, out, in]`` slot view)."""
        out, inn = self.shapes[name]
        rows = self.epn * out * inn  # elements of epn expert rows
        nbytes = 2 * rows * dtype.itemsize
        chunk = (nbytes + granularity - 1) // granularity * granularity // dtype.itemsize
        keepalive, shareable, owned = nvl_dist_alloc(
            shape=[chunk], dtype=dtype, use_fabric=self._use_fabric)
        try:
            # Exchanges the handles (closing the fds once mapped) and pins
            # ``keepalive`` on the returned mapping.
            full = _map_nvl_dist_tensor(
                [chunk], dtype, shareable, keepalive,
                self.rank, self.R, self.group, self._use_fabric,
            )
        finally:
            nvl_release_mem_handle(owned)
        self._mappings.append(full)
        self._keepalives.append(keepalive)
        base = self.rank * chunk
        full[base:base + chunk].zero_()
        compute = full[base:base + 2 * rows].view(2 * self.epn, out, inn)
        pool = full.as_strided(
            (self.R, self.epn, out, inn),
            (chunk, out * inn, inn, 1),
            storage_offset=rows,
        )
        return compute, pool

    # ---- accessors -----------------------------------------------------------

    @property
    def destroyed(self) -> bool:
        return self._destroyed

    def _require_live(self) -> Buffer:
        assert not self._destroyed, "ExpertPools has been destroyed"
        return self.buffer

    def _get(self, table: dict[str, torch.Tensor], name: str) -> torch.Tensor:
        self._require_live()
        assert name in table, (
            f"ExpertPools: unknown projection {name!r}, expected one of "
            f"{self.PROJECTIONS}"
        )
        return table[name]

    def weights(self, name: str) -> torch.Tensor:
        """bf16 ``[2*epn, out, in]``: this rank's staging rows, then its
        prefetch slots; the grouped-GEMM weight view."""
        return self._get(self._weights, name)

    def staging(self, name: str) -> torch.Tensor:
        """bf16 ``[epn, out, in]`` = ``weights(name)[:epn]``: where a layer
        copies its own experts before computing; ``push`` reads it."""
        return self._get(self._staging, name)

    def prefetch_buffer(self, name: str) -> torch.Tensor:
        """bf16 ``[R, epn, out, in]``: every rank's prefetch slots;
        ``[rank]`` is ``weights(name)[epn:]``."""
        return self._get(self._prefetch, name)

    def grads(self, name: str) -> torch.Tensor:
        """fp32 ``[2*epn, out, in]``: this rank's local grad rows, then its
        reduce slots; the grouped-GEMM dW view."""
        return self._get(self._grads, name)

    def local_grads(self, name: str) -> torch.Tensor:
        """fp32 ``[epn, out, in]`` = ``grads(name)[:epn]``: ``reduce`` adds
        the peers' slot grads of this rank's experts here."""
        return self._get(self._local_grads, name)

    def reduce_buffer(self, name: str) -> torch.Tensor:
        """fp32 ``[R, epn, out, in]``: every rank's reduce slots; ``[rank]``
        is ``grads(name)[epn:]``."""
        return self._get(self._reduce, name)

    # ---- collectives ---------------------------------------------------------

    def push(self, plan: MoonEPCommPlan) -> None:
        """``Buffer.prefetch_weight``: copy this rank's experts that ``plan``
        copies elsewhere from ``staging`` into those ranks' prefetch slots.

        Collective; on return (stream order) this rank's slots hold the copies
        ``plan.experts_to_copy[rank]`` names and its unused slots are
        untouched. The kernel has no entry barrier: as with
        ``prefetch_weight``, no rank may still be reading the slots being
        overwritten (any dispatch or combine between a rank's last slot read
        and this push guarantees that).
        """
        buffer = self._require_live()
        buffer.prefetch_weight(
            plan=plan,
            local_gate_weight=self._staging["gate"],
            local_up_weight=self._staging["up"],
            local_down_weight=self._staging["down"],
            gate_prefetch_buffer=self._prefetch["gate"],
            up_prefetch_buffer=self._prefetch["up"],
            down_prefetch_buffer=self._prefetch["down"],
        )

    def reduce(self, plan: MoonEPCommPlan) -> None:
        """``Buffer.reduce_grad``: add every rank's slot grads of this rank's
        experts into ``local_grads``, then each rank zeroes its consumed slots.

        Collective. ``plan`` must be the plan whose slots the grads were
        computed for. Starts with a device-side inter-rank sync, because
        ``reduce_grad`` has no entry barrier: every rank must have finished
        writing its slot grads (stream order) before any rank reads them.
        """
        buffer = self._require_live()
        assert isinstance(plan, MoonEPCommPlan), (
            "ExpertPools.reduce: plan must be the MoonEPCommPlan the grads were "
            "computed with"
        )
        launch_inter_rank_sync(buffer._require_ctx())
        buffer.reduce_grad(
            plan=plan,
            local_gate_grad=self._local_grads["gate"],
            local_up_grad=self._local_grads["up"],
            local_down_grad=self._local_grads["down"],
            gate_reduce_buffer=self._reduce["gate"],
            up_reduce_buffer=self._reduce["up"],
            down_reduce_buffer=self._reduce["down"],
        )

    def destroy(self) -> None:
        """Synchronize, barrier over the group, then drop every mapping.

        Collective; call it before destroying the process group. Idempotent.
        Views obtained earlier keep their mapping alive until they are dropped
        and must not be used afterwards.
        """
        if self._destroyed:
            return
        torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier(group=self.group)
        for table in (self._staging, self._local_grads, self._prefetch,
                      self._reduce, self._weights, self._grads):
            table.clear()
        self._mappings.clear()
        self._keepalives.clear()
        self.buffer = None
        self._destroyed = True
