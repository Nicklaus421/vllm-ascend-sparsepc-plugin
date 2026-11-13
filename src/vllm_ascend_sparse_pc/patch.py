#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend-sparse-pc plugin.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Monkey-patch vllm-ascend to activate sparse prefix caching.

This module contains no algorithm logic; it only wires the plugin's
self-contained implementation into vllm-ascend's runtime classes. Everything is
applied idempotently, so ``apply()`` may be called more than once without
double-wrapping.

Three integration points are patched:

1. ``AscendHybridKVCacheCoordinator`` (``patch_kv_cache_coordinator.py``)
   — resolve sparse Mamba hits and drive the DP-optimal placement.
2. ``NPUModelRunner`` (``worker/model_runner_v1.py``)
   — capture recurrent-state checkpoints after a prefill and restore them
   before a recompute prefill.
3. ``AscendWorkerProc`` / ``WorkerProc`` (``patch_multiproc_executor.py``)
   — hand the cross-process Manager proxy to TP workers.

The plugin targets a *pristine* vllm-ascend (one without any sparse-prefix-cache
code). If vllm-ascend already contains that code, do not install this plugin: the
patches would double-apply.
"""

from __future__ import annotations

import logging
from typing import Any

from vllm_ascend_sparse_pc import envs

logger = logging.getLogger(__name__)

# Names of the integration points that have been applied in this process.
# ``apply()`` uses this to stay idempotent while still applying each patch
# independently, so a process-specific import failure does not block the others.
_APPLIED: set[str] = set()


def _is_enabled() -> bool:
    return bool(envs.VLLM_ASCEND_SPARSE_PREFIX_CACHE_ENABLED)


def _preimport_ascend_ops() -> None:
    """Break a latent vllm-ascend circular import before patching.

    ``vllm_ascend.ops.fused_moe.experts_selector`` imports
    ``vllm_ascend.device.device_op`` at module top level, which in turn imports
    ``vllm_ascend.ops.triton.fla`` and therefore re-enters ``vllm_ascend.ops``
    while the package ``__init__`` is still executing::

        ops/__init__            -> import fused_moe
        fused_moe               -> from experts_selector import select_experts
        experts_selector        -> from device_op import DeviceOperator
        device_op               -> from ops.triton.fla.chunk_scaled_dot_kkt import ...
        ops/__init__ (re-entry) -> import fused_moe
        fused_moe               -> from experts_selector import select_experts
                                    ^ experts_selector is still partially
                                      initialized (select_experts undefined)

    This cycle is harmless when ``vllm_ascend.ops`` is imported *first*: the
    triton ``fla`` submodules have no back-edge to ``experts_selector``, so
    ``ops/__init__`` completes and ``select_experts`` is defined before any
    ``vllm_ascend.quantization`` -> ``w4a8_mxfp4`` -> ``experts_selector`` path
    runs. The plugin's entry point fires during vllm's general-plugin load,
    ahead of ``NPUPlatform.pre_register_and_update`` (which imports
    ``vllm_ascend.quantization``), so we eagerly initialise ``ops`` here to
    establish the safe order and avoid leaving a partially-initialised module in
    ``sys.modules``.
    """
    try:
        import vllm_ascend.ops  # noqa: F401
    except (ImportError, AttributeError) as exc:
        # Only reachable outside a vllm runtime (e.g. CPU unit tests). The
        # per-patch import guards below handle that case too.
        logger.debug(
            "vllm-ascend-sparse-pc: could not pre-import vllm_ascend.ops (%s).",
            exc,
        )


# ---------------------------------------------------------------------------
# 1. KV-cache coordinator
# ---------------------------------------------------------------------------
def _resolve_sparse_mamba_hit(overlap_depth: int) -> int | None:
    """Resolve the sparse recurrent-state hit length for a Mamba group.

    Returns ``None`` when sparse prefix caching is disabled (the caller falls
    back to the dense Mamba block pool); otherwise returns the nearest
    checkpoint position ``C <= overlap_depth`` (``0`` when no checkpoint is
    placed for this prefix length).
    """
    from vllm_ascend_sparse_pc.coordinator import get_checkpoint_positions

    if not envs.VLLM_ASCEND_SPARSE_PREFIX_CACHE_ENABLED:
        return None
    positions = get_checkpoint_positions(overlap_depth)
    return positions[-1] if positions else 0


def _patch_kv_cache_coordinator() -> None:
    from vllm.logger import logger
    from vllm.utils.math_utils import cdiv
    from vllm.v1.core.kv_cache_utils import (
        BlockHash,
        BlockHashList,
        KVCacheBlock,
    )
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        KVCacheSpec,
        MambaSpec,
    )

    from vllm_ascend.patch.platform import patch_kv_cache_coordinator as mod
    from vllm_ascend_sparse_pc.shared import (
        init_sparse_shared_state,
        publish_sparse_positions,
    )
    from vllm_ascend_sparse_pc.strategies import HistogramTracker

    cls = mod.AscendHybridKVCacheCoordinator

    # --- __init__: add sparse state after the original init completes ---
    orig_init = cls.__init__

    def patched_init(self, *args: Any, **kwargs: Any) -> None:
        orig_init(self, *args, **kwargs)
        self._sparse_tracker = None
        self._sparse_solved = False
        self._sparse_warmup_requests = 0
        self._sparse_block_size = 0
        self._init_sparse_hist_tracker()

    cls.__init__ = patched_init

    # --- placement state / histogram ---
    def _init_sparse_hist_tracker(self) -> None:
        if not envs.VLLM_ASCEND_SPARSE_PREFIX_CACHE_ENABLED:
            return
        if envs.VLLM_ASCEND_SPARSE_PREFIX_CACHE_STRATEGY != "hist_exp_decay":
            return

        # Create the Manager proxy in this (EngineCore) process; workers
        # re-attach it via process_kwargs (see patch_multiproc_executor.py).
        init_sparse_shared_state()

        self._sparse_tracker = HistogramTracker(
            max_len=self.max_model_len,
            budget=int(envs.VLLM_ASCEND_SPARSE_PREFIX_CACHE_NUM_CHECKPOINTS),
            mode="frozen",
            gamma=float(envs.VLLM_ASCEND_SPARSE_PREFIX_CACHE_GAMMA),
            replan_interval=int(
                envs.VLLM_ASCEND_SPARSE_PREFIX_CACHE_REPLAN_INTERVAL
            ),
            alpha=float(envs.VLLM_ASCEND_SPARSE_PREFIX_CACHE_LAPLACE_ALPHA),
            bin_size=int(envs.VLLM_ASCEND_SPARSE_PREFIX_CACHE_BIN_SIZE),
            # Prefix-consistent backtracking: solve once and freeze a fixed set
            # of positions so capture (worker) and resolve (coordinator) agree
            # across processes with zero per-prefix bookkeeping.
            adaptive_backtrack=False,
            attn_weight=float(
                envs.VLLM_ASCEND_SPARSE_PREFIX_CACHE_RECOMPUTE_ATTN_WEIGHT
            ),
        )
        self._sparse_warmup_requests = int(
            envs.VLLM_ASCEND_SPARSE_PREFIX_CACHE_WARMUP_REQUESTS
        )
        self._sparse_block_size = int(
            envs.VLLM_ASCEND_SPARSE_PREFIX_CACHE_KERNEL_BLOCK_SIZE
        )

    def _observe_sparse_hit(self, overlap_depth: int) -> None:
        tracker = self._sparse_tracker
        if tracker is None or self._sparse_solved:
            return
        tracker.observe(overlap_depth)
        if tracker.n_obs < self._sparse_warmup_requests:
            return
        import numpy as np

        nz = np.argwhere(tracker.counts > 0)
        n_active = len(nz)
        max_bin = int(nz.max()) if n_active else 0
        logger.info(
            "Sparse prefix cache: warmup complete (%d observations, "
            "%d non-empty bins, max overlap depth %d tokens).",
            tracker.n_obs,
            n_active,
            max_bin * tracker.bin_size,
        )
        import time

        t_solve = time.perf_counter()
        tracker.solve()
        t_solve = time.perf_counter() - t_solve
        self._sparse_solved = True
        fixed = tracker._fixed_positions
        t_clip = time.perf_counter()
        positions = self._clip_sparse_positions(fixed) if fixed else []
        t_clip = time.perf_counter() - t_clip
        if not positions:
            # Degenerate/empty histogram: keep the fixed_interval warmup
            # fallback rather than publishing an empty placement.
            logger.warning(
                "Sparse prefix cache: DP solve produced no usable checkpoint "
                "positions; keeping the fixed_interval warmup placement."
            )
            return
        t_publish = time.perf_counter()
        publish_sparse_positions(positions)
        t_publish = time.perf_counter() - t_publish
        logger.info(
            "Sparse prefix cache: solved DP placement after %d observations "
            "(%d checkpoints: %s; solve=%.3fs clip=%.3fs publish=%.3fs).",
            tracker.n_obs,
            len(positions),
            positions,
            t_solve,
            t_clip,
            t_publish,
        )

    def _clip_sparse_positions(self, positions: list[int]) -> list[int]:
        clipped: list[int] = []
        for pos in positions:
            c = (pos // self._sparse_block_size) * self._sparse_block_size
            if c <= 0 or c in clipped:
                continue
            clipped.append(c)
        return sorted(clipped)

    cls._init_sparse_hist_tracker = _init_sparse_hist_tracker
    cls._observe_sparse_hit = _observe_sparse_hit
    cls._clip_sparse_positions = _clip_sparse_positions

    # --- find_longest_cache_hit: full wrap with the sparse Mamba branch ---
    def find_longest_cache_hit(
        self,
        block_hashes: list[BlockHash],
        max_cache_hit_length: int,
    ) -> tuple[tuple[list[KVCacheBlock], ...], int] | tuple[tuple[list[KVCacheBlock], ...], int, int]:
        def _get_block_hashes(kv_cache_spec: KVCacheSpec) -> BlockHashList:
            return block_hashes

        num_groups = len(self.kv_cache_config.kv_cache_groups)
        hit_length = max_cache_hit_length
        longest_hit_length = 0
        hit_blocks_by_group: list[list[KVCacheBlock] | None] = [None] * num_groups
        hit_length_by_group: list[int] = [0] * num_groups

        is_simple_hybrid = len(self.attention_groups) == 2 and isinstance(
            self.attention_groups[0].spec, FullAttentionSpec
        )

        eagle_verified: set[int] = set()

        while True:
            curr_hit_length = hit_length
            for idx, (spec, group_ids, manager_cls, use_eagle) in enumerate(
                self.attention_groups
            ):
                # SPARSE-PREFIX-CACHE: the recurrent state is only checkpointed
                # at a sparse set of positions, so the Mamba group's hit is the
                # nearest checkpoint C <= curr_hit_length, not a dense block run.
                # Rewinding the hybrid hit length to C makes the scheduler emit
                # a chunked prefill over [C, P) with the restored initial state.
                if isinstance(spec, MambaSpec):
                    if self._sparse_tracker is not None and curr_hit_length > 0:
                        self._observe_sparse_hit(curr_hit_length)
                    sparse_hit_length = _resolve_sparse_mamba_hit(curr_hit_length)
                    if sparse_hit_length is not None:
                        curr_hit_length = sparse_hit_length
                        for group_id in group_ids:
                            hit_blocks_by_group[group_id] = []
                            hit_length_by_group[group_id] = curr_hit_length
                        continue

                group_block_size = self._get_effective_block_size(spec)
                first_group_id = group_ids[0]
                cached_blocks = hit_blocks_by_group[first_group_id]
                if isinstance(spec, FullAttentionSpec) and cached_blocks is not None:
                    # Full attention is downward-closed: we only need to look
                    # up cached blocks once; on subsequent iterations just trim
                    # to the (reduced) current hit length.
                    curr_hit_length = min(
                        curr_hit_length, hit_length_by_group[first_group_id]
                    )
                    continue

                drop_eagle_block = (
                    use_eagle
                    and idx not in eagle_verified
                    and not self.skips_eagle_block_drop
                )

                _max_length = curr_hit_length
                if drop_eagle_block and not isinstance(spec, MambaSpec):
                    eagle_margin = (
                        self.hash_block_size
                        if self.enable_partial_hash_hits
                        and manager_cls.supports_fine_grained_hash_lookup
                        and group_block_size > self.hash_block_size
                        else group_block_size
                    )
                    _max_length = min(
                        curr_hit_length + eagle_margin, max_cache_hit_length
                    )
                hit_result = manager_cls.find_longest_cache_hit(
                    block_hashes=_get_block_hashes(spec),
                    max_length=_max_length,
                    kv_cache_group_ids=group_ids,
                    block_pool=self.block_pool,
                    kv_cache_spec=spec,
                    drop_eagle_block=drop_eagle_block,
                    alignment_tokens=self._cache_hit_alignment_tokens,
                    dcp_world_size=self.dcp_world_size,
                    pcp_world_size=1,
                )
                hit_blocks, _new_hit_length = hit_result
                if drop_eagle_block:
                    eagle_verified.add(idx)
                elif _new_hit_length < curr_hit_length:
                    eagle_verified.clear()
                curr_hit_length = _new_hit_length
                for group_id, blocks in zip(group_ids, hit_blocks):
                    hit_blocks_by_group[group_id] = blocks
                    hit_length_by_group[group_id] = _new_hit_length

                longest_hit_length = max(longest_hit_length, curr_hit_length)

            if curr_hit_length >= hit_length:
                break
            hit_length = curr_hit_length
            if is_simple_hybrid:
                break

        # Truncate all full attention groups to the final hit_length.
        for group in self.attention_groups:
            if not isinstance(group.spec, FullAttentionSpec):
                continue
            num_blocks = cdiv(
                hit_length,
                self._get_effective_block_size(group.spec),
            )
            for group_id in group.group_ids:
                if (blks := hit_blocks_by_group[group_id]) is not None:
                    del blks[num_blocks:]
                    hit_length_by_group[group_id] = hit_length

        cache_hit_blocks = tuple(
            blocks if blocks is not None else [] for blocks in hit_blocks_by_group
        )
        return cache_hit_blocks, hit_length, longest_hit_length - hit_length

    cls.find_longest_cache_hit = find_longest_cache_hit


# ---------------------------------------------------------------------------
# 2. Model runner
# ---------------------------------------------------------------------------
def _patch_model_runner() -> None:
    from vllm.logger import logger
    from vllm.v1.kv_cache_interface import MambaSpec

    from vllm_ascend.worker import model_runner_v1 as mod
    from vllm_ascend_sparse_pc.checkpoint import RecurrentCheckpoint
    from vllm_ascend_sparse_pc.coordinator import get_checkpoint_positions
    from vllm_ascend_sparse_pc.runner import (
        capture_checkpoint_state,
        restore_checkpoint_state,
    )

    cls = mod.NPUModelRunner

    # --- __init__: add sparse state ---
    orig_init = cls.__init__

    def patched_init(self, *args: Any, **kwargs: Any) -> None:
        orig_init(self, *args, **kwargs)
        self._sparse_prefix_cache_enabled: bool = False
        self._sparse_prefix_cache_initialized: bool = False
        self._sparse_captured_reqs: set[str] = set()
        self._sparse_state_buffer: dict[tuple, Any] = {}
        self._sparse_checkpoint_interval: int = 0
        self._sparse_block_size: int = 0
        self._sparse_state_copy_funcs: Any = None

    cls.__init__ = patched_init

    # --- sparse worker-side methods ---
    def _init_sparse_prefix_cache(self) -> None:
        if self._sparse_prefix_cache_initialized:
            return
        self._sparse_prefix_cache_initialized = True

        if not envs.VLLM_ASCEND_SPARSE_PREFIX_CACHE_ENABLED:
            return
        if not self.model_config.is_hybrid:
            return

        self._sparse_prefix_cache_enabled = True
        self._sparse_checkpoint_interval = int(
            envs.VLLM_ASCEND_SPARSE_PREFIX_CACHE_CHECKPOINT_INTERVAL
        )
        self._sparse_block_size = int(
            envs.VLLM_ASCEND_SPARSE_PREFIX_CACHE_KERNEL_BLOCK_SIZE
        )
        self._sparse_state_copy_funcs = self.model.get_mamba_state_copy_func()
        logger.info(
            "Sparse prefix cache enabled (interval=%d tokens).",
            self._sparse_checkpoint_interval,
        )

    def _mamba_kv_caches_by_layer(self) -> dict[int, tuple[Any, Any]]:
        forward_context = self.compilation_config.static_forward_context
        result: dict[int, tuple[Any, Any]] = {}
        layer_idx = 0
        for group in self.kv_cache_config.kv_cache_groups:
            if not isinstance(group.kv_cache_spec, MambaSpec):
                continue
            for layer_name in group.layer_names:
                attention = forward_context[layer_name]
                # GDN attention.kv_cache == [conv_state, ssm_state].
                result[layer_idx] = (attention.kv_cache[0], attention.kv_cache[1])
                layer_idx += 1
        return result

    def _sparse_mamba_group_id(self) -> int | None:
        for gid, group in enumerate(self.kv_cache_config.kv_cache_groups):
            if isinstance(group.kv_cache_spec, MambaSpec):
                return gid
        return None

    def _sparse_block_ids(self, req_idx: int) -> list[int] | None:
        mamba_gid = self._sparse_mamba_group_id()
        if mamba_gid is None:
            return None
        block_table = self.input_batch.block_table[mamba_gid].get_numpy_array()
        if req_idx >= block_table.shape[0]:
            return None
        return [int(b) for b in block_table[req_idx]]

    def _maybe_restore_sparse_checkpoints(self) -> None:
        self._init_sparse_prefix_cache()
        if not self._sparse_prefix_cache_enabled:
            return
        kv_caches_by_layer = self._mamba_kv_caches_by_layer()
        if not kv_caches_by_layer:
            return
        block_size = self._sparse_block_size
        for i, req_id in enumerate(self.input_batch.req_ids):
            req_state = self.requests.get(req_id)
            if req_state is None:
                continue
            c = req_state.num_computed_tokens
            if c <= 0:
                continue
            prompt = req_state.prompt_token_ids
            if prompt is None:
                continue
            checkpoint = self._sparse_state_buffer.get(tuple(prompt[:c]))
            if checkpoint is None:
                continue
            block_ids = self._sparse_block_ids(i)
            if block_ids is None:
                continue
            dest_block_idx = c // block_size - 1
            if dest_block_idx < 0 or dest_block_idx >= len(block_ids):
                continue
            restore_checkpoint_state(
                checkpoint,
                kv_caches_by_layer,
                block_ids,
                dest_block_idx,
                self._sparse_state_copy_funcs,
            )

    def _maybe_capture_sparse_checkpoints(self) -> None:
        self._init_sparse_prefix_cache()
        if not self._sparse_prefix_cache_enabled:
            return
        kv_caches_by_layer = self._mamba_kv_caches_by_layer()
        if not kv_caches_by_layer:
            return
        block_size = self._sparse_block_size
        for i, req_id in enumerate(self.input_batch.req_ids):
            req_state = self.requests.get(req_id)
            if req_state is None or req_id in self._sparse_captured_reqs:
                continue
            if req_state.num_computed_tokens < req_state.num_tokens:
                continue  # still prefilling
            prompt = req_state.prompt_token_ids
            if prompt is None:
                continue  # prompt-embed request: no token-id prefix key
            block_ids = self._sparse_block_ids(i)
            if block_ids is None:
                continue
            seq_len = len(prompt)
            for pos in get_checkpoint_positions(seq_len):
                src_block_idx = pos // block_size - 1
                if src_block_idx < 0 or src_block_idx >= len(block_ids):
                    continue
                recurrent, conv = capture_checkpoint_state(
                    kv_caches_by_layer,
                    block_ids,
                    src_block_idx,
                    self._sparse_state_copy_funcs,
                )
                key = tuple(prompt[:pos])
                self._sparse_state_buffer[key] = RecurrentCheckpoint(
                    position=pos,
                    recurrent_states=recurrent,
                    conv_states=conv,
                )
            self._sparse_captured_reqs.add(req_id)

    cls._init_sparse_prefix_cache = _init_sparse_prefix_cache
    cls._mamba_kv_caches_by_layer = _mamba_kv_caches_by_layer
    cls._sparse_mamba_group_id = _sparse_mamba_group_id
    cls._sparse_block_ids = _sparse_block_ids
    cls._maybe_restore_sparse_checkpoints = _maybe_restore_sparse_checkpoints
    cls._maybe_capture_sparse_checkpoints = _maybe_capture_sparse_checkpoints

    # --- capture after prefill ---
    orig_update_after = cls._update_states_after_model_execute

    def patched_update_after(self, output_token_ids, scheduler_output) -> None:
        self._maybe_capture_sparse_checkpoints()
        return orig_update_after(self, output_token_ids, scheduler_output)

    cls._update_states_after_model_execute = patched_update_after

    # --- restore before the forward ---
    #
    # NOTE: the original in-tree code calls ``_maybe_restore_sparse_checkpoints``
    # deep inside ``execute_model`` (after ``_prepare_inputs`` and the
    # mamba-align preprocess, before ``use_compress``/forward). For
    # ``mamba_cache_mode='all'`` — the only mode sparse prefix caching supports
    # — nothing between ``_update_states`` and the forward writes the recurrent
    # state blocks (the align preprocess is skipped), so restoring right after
    # ``_update_states`` (which has already rewound ``num_computed_tokens`` to
    # ``C`` and built ``self.input_batch.block_table``) is equivalent and avoids
    # copying the entire ``execute_model``.
    orig_update_states = cls._update_states

    def patched_update_states(self, scheduler_output):
        result = orig_update_states(self, scheduler_output)
        self._maybe_restore_sparse_checkpoints()
        return result

    cls._update_states = patched_update_states


# ---------------------------------------------------------------------------
# 3. Multiproc executor (cross-process shared state)
# ---------------------------------------------------------------------------
def _sparse_worker_main(**kwargs: Any):
    """Worker entry point substituted for ``WorkerProc.worker_main``.

    Runs in each spawned TP worker before the real worker main. It (1)
    re-attaches the cross-process ``multiprocessing.Manager`` proxy carrying the
    frozen DP checkpoint positions and (2) re-applies the plugin patches in this
    process. vllm loads ``vllm.general_plugins`` in every process (including
    workers), so the patches normally arrive via that path too; the explicit
    ``apply()`` here is a belt-and-suspenders that also makes the patch reliable
    if a deployment uses a non-standard plugin-loading path.

    This must stay a module-level function: it is passed as ``target`` to
    ``multiprocessing.Process`` and therefore has to be picklable (a nested
    function would fail under the ``spawn`` start method).
    """
    from vllm.v1.executor.multiproc_executor import WorkerProc

    from vllm_ascend_sparse_pc.shared import attach_sparse_shared_state

    proxy = kwargs.pop("sparse_shared_state", None)
    if proxy is not None:
        attach_sparse_shared_state(proxy)

    # Idempotent: applies any patch not yet active in this process.
    apply()

    return WorkerProc.worker_main(**kwargs)


def _patch_multiproc_executor() -> None:
    # The cross-process Manager proxy is only needed for the DP (hist_exp_decay)
    # strategy; fixed_interval is prefix-consistent with no shared state.
    # Importing ``patch_multiproc_executor`` also swaps in
    # ``AscendMultiprocExecutor``, so avoid that side effect unless it is
    # actually required.
    if (
        not envs.VLLM_ASCEND_SPARSE_PREFIX_CACHE_ENABLED
        or envs.VLLM_ASCEND_SPARSE_PREFIX_CACHE_STRATEGY != "hist_exp_decay"
    ):
        return

    from vllm.config import VllmConfig
    from vllm.utils.system_utils import get_mp_context

    from vllm.v1.executor.multiproc_executor import UnreadyWorkerProcHandle

    from vllm_ascend.patch.platform import patch_multiproc_executor as mod
    from vllm_ascend_sparse_pc.shared import init_sparse_shared_state

    proc_cls = mod.AscendWorkerProc

    @staticmethod
    def patched_make_worker_process(
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        input_shm_handle,
        shared_worker_lock,
        is_driver_worker: bool = False,
        inherited_fds: list[int] | None = None,
    ) -> UnreadyWorkerProcHandle:
        context = get_mp_context()
        # Ready pipe to communicate readiness from child to parent
        ready_reader, ready_writer = context.Pipe(duplex=False)
        # Death pipe to let child detect parent process exit
        death_reader, death_writer = context.Pipe(duplex=False)
        if inherited_fds is not None:
            inherited_fds = inherited_fds.copy()
            inherited_fds.extend((ready_reader.fileno(), death_writer.fileno()))
        process_kwargs = {
            "vllm_config": vllm_config,
            "local_rank": local_rank,
            "rank": rank,
            "distributed_init_method": distributed_init_method,
            "input_shm_handle": input_shm_handle,
            "ready_pipe": ready_writer,
            "death_pipe": death_reader,
            "shared_worker_lock": shared_worker_lock,
            "is_driver_worker": is_driver_worker,
            # Have the worker close parent end of this worker's pipes too
            "inherited_fds": inherited_fds if inherited_fds is not None else [],
        }

        # Hand the sparse-prefix-cache Manager proxy to the worker so the
        # DP-optimal checkpoint positions are visible in the worker process.
        # ``init_sparse_shared_state`` is idempotent; ``None`` when sparse
        # caching is disabled or the fixed_interval strategy is in use.
        sparse_shared_state = None
        if (
            envs.VLLM_ASCEND_SPARSE_PREFIX_CACHE_ENABLED
            and envs.VLLM_ASCEND_SPARSE_PREFIX_CACHE_STRATEGY == "hist_exp_decay"
        ):
            sparse_shared_state = init_sparse_shared_state()
        process_kwargs["sparse_shared_state"] = sparse_shared_state

        # Run EngineCore busy loop in background process.
        proc = context.Process(
            target=_sparse_worker_main,
            kwargs=process_kwargs,
            name=f"VllmWorker-{rank}",
            daemon=False,
        )
        proc.start()
        # Close child ends of pipes here in the parent
        ready_writer.close()
        death_reader.close()
        # Keep death_writer open in parent - when parent exits,
        # death_reader in child will get EOFError
        return UnreadyWorkerProcHandle(proc, rank, ready_reader, death_writer)

    proc_cls.make_worker_process = patched_make_worker_process


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
_PATCHES: tuple[tuple[str, Any], ...] = (
    ("kv_cache_coordinator", _patch_kv_cache_coordinator),
    ("model_runner", _patch_model_runner),
    ("multiproc_executor", _patch_multiproc_executor),
)


def apply() -> None:
    """Apply all monkey-patches. Idempotent; safe to call more than once.

    Each integration point is applied independently: if one target module is not
    importable in this process (e.g. the worker-only model runner in the
    engine-core process), the others still apply, and the failed one is retried
    on the next ``apply()`` call.
    """
    _preimport_ascend_ops()
    for name, patch_fn in _PATCHES:
        if name in _APPLIED:
            continue
        try:
            patch_fn()
        except (ImportError, AttributeError) as exc:
            logger.warning(
                "vllm-ascend-sparse-pc: skipped %s patch (%s). "
                "It will be retried on the next apply().",
                name,
                exc,
            )
        else:
            _APPLIED.add(name)
