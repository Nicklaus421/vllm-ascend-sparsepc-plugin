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
"""Sparse Prefix Caching for Hybrid and Recurrent LLM Serving.

This package is a monkey-patch plugin for vllm-ascend implementing
*Sparse Prefix Caching for Hybrid and Recurrent LLM Serving* (arXiv:2605.05219).

The algorithm code is fully self-contained here (``strategies``, ``checkpoint``,
``store``, ``coordinator``, ``runner``, ``shared``). Importing this package also
applies the runtime monkey-patches (``patch.apply()``) that wire the algorithm
into vllm-ascend's ``AscendHybridKVCacheCoordinator``, ``NPUModelRunner`` and the
multiprocessing executor.

Usage::

    import vllm_ascend_sparse_pc  # noqa: F401  (applies the patches)
    # ... then start the vLLM engine as usual.
"""

from vllm_ascend_sparse_pc.checkpoint import (
    PrefixCheckpointStore,
    RecurrentCheckpoint,
)
from vllm_ascend_sparse_pc.coordinator import (
    SparseHit,
    SparsePrefixCacheCoordinator,
    get_checkpoint_positions,
)
from vllm_ascend_sparse_pc.runner import (
    SparsePrefixCacheHook,
    capture_checkpoint_state,
    get_sparse_prefix_cache_hook,
    restore_checkpoint_state,
    set_sparse_prefix_cache_hook,
)
from vllm_ascend_sparse_pc.shared import (
    attach_sparse_shared_state,
    get_sparse_positions,
    get_sparse_shared_state,
    init_sparse_shared_state,
    publish_sparse_positions,
)
from vllm_ascend_sparse_pc.store import SparsePrefixCache
from vllm_ascend_sparse_pc.strategies import (
    HistogramTracker,
    backtrack,
    balanced_positions,
    fixed_interval_positions,
    laplace_smoothing,
    solve_dp,
)

__all__ = [
    "HistogramTracker",
    "PrefixCheckpointStore",
    "RecurrentCheckpoint",
    "SparseHit",
    "SparsePrefixCache",
    "SparsePrefixCacheCoordinator",
    "SparsePrefixCacheHook",
    "apply",
    "attach_sparse_shared_state",
    "backtrack",
    "balanced_positions",
    "fixed_interval_positions",
    "capture_checkpoint_state",
    "get_checkpoint_positions",
    "get_sparse_positions",
    "get_sparse_prefix_cache_hook",
    "get_sparse_shared_state",
    "init_sparse_shared_state",
    "laplace_smoothing",
    "publish_sparse_positions",
    "restore_checkpoint_state",
    "set_sparse_prefix_cache_hook",
    "solve_dp",
]


def apply() -> None:
    """Apply the vllm-ascend monkey-patches (idempotent).

    Called automatically on import; may also be called explicitly if the patch
    was reset or needs to be re-applied after a late import of vllm-ascend.
    """
    from vllm_ascend_sparse_pc import patch

    patch.apply()


# Auto-apply the monkey-patches on import. Guarded so that a missing vllm /
# vllm-ascend installation only surfaces as a warning (the algorithm modules
# above remain importable and unit-testable without a full runtime).
try:
    apply()
except ImportError as exc:  # pragma: no cover - depends on runtime env
    import warnings

    warnings.warn(
        f"vllm-ascend-sparse-pc: monkey-patches not applied because a "
        f"dependency is missing ({exc}). Import vllm-ascend first, or call "
        f"``vllm_ascend_sparse_pc.apply()`` after it is available.",
        RuntimeWarning,
        stacklevel=2,
    )
