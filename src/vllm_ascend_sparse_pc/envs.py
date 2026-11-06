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
"""Sparse-prefix-cache environment configuration.

These mirror the ``VLLM_ASCEND_SPARSE_PREFIX_CACHE_*`` variables defined by
vllm-ascend, but are read here so the plugin is fully self-contained and does
not need to import ``vllm_ascend.envs`` (which pulls the whole vllm-ascend
package __init__ at import time).
"""

import os
from collections.abc import Callable
from typing import Any

env_variables: dict[str, Callable[[], Any]] = {
    # Whether to enable sparse prefix caching for hybrid (attention + recurrent)
    # models. When enabled, the GatedDeltaNet recurrent state is checkpointed at
    # a sparse set of DP-optimal positions instead of every block.
    "VLLM_ASCEND_SPARSE_PREFIX_CACHE_ENABLED": lambda: bool(
        int(os.getenv("VLLM_ASCEND_SPARSE_PREFIX_CACHE_ENABLED", "0"))
    ),
    # Checkpoint placement strategy. "fixed_interval" (default) places
    # checkpoints at a fixed token interval and is prefix-consistent.
    # "hist_exp_decay" is the DP-optimal strategy with a cross-process-shared
    # overlap histogram.
    "VLLM_ASCEND_SPARSE_PREFIX_CACHE_STRATEGY": lambda: os.getenv(
        "VLLM_ASCEND_SPARSE_PREFIX_CACHE_STRATEGY", "fixed_interval"
    ),
    # Byte budget (GB) for the GDN recurrent-state checkpoint store.
    "VLLM_ASCEND_SPARSE_PREFIX_CACHE_BUDGET_GB": lambda: float(
        os.getenv("VLLM_ASCEND_SPARSE_PREFIX_CACHE_BUDGET_GB", "1.0")
    ),
    # Bin size (tokens) used to discretize the overlap-depth histogram.
    "VLLM_ASCEND_SPARSE_PREFIX_CACHE_BIN_SIZE": lambda: int(
        os.getenv("VLLM_ASCEND_SPARSE_PREFIX_CACHE_BIN_SIZE", "16")
    ),
    # Exponential decay factor for the drift-following histogram.
    "VLLM_ASCEND_SPARSE_PREFIX_CACHE_GAMMA": lambda: float(
        os.getenv("VLLM_ASCEND_SPARSE_PREFIX_CACHE_GAMMA", "0.99")
    ),
    # Number of observations between DP re-solves.
    "VLLM_ASCEND_SPARSE_PREFIX_CACHE_REPLAN_INTERVAL": lambda: int(
        os.getenv("VLLM_ASCEND_SPARSE_PREFIX_CACHE_REPLAN_INTERVAL", "100")
    ),
    # Laplace smoothing: total pseudo-count mass spread uniformly over the
    # active overlap-depth bins.
    "VLLM_ASCEND_SPARSE_PREFIX_CACHE_LAPLACE_ALPHA": lambda: float(
        os.getenv("VLLM_ASCEND_SPARSE_PREFIX_CACHE_LAPLACE_ALPHA", "1.0")
    ),
    # Block size (tokens) that checkpoint positions are clipped down to.
    "VLLM_ASCEND_SPARSE_PREFIX_CACHE_KERNEL_BLOCK_SIZE": lambda: int(
        os.getenv("VLLM_ASCEND_SPARSE_PREFIX_CACHE_KERNEL_BLOCK_SIZE", "16")
    ),
    # Fixed interval (tokens) between sparse recurrent-state checkpoints in the
    # v1 deterministic strategy.
    "VLLM_ASCEND_SPARSE_PREFIX_CACHE_CHECKPOINT_INTERVAL": lambda: int(
        os.getenv("VLLM_ASCEND_SPARSE_PREFIX_CACHE_CHECKPOINT_INTERVAL", "256")
    ),
    # Number of overlap-depth observations (cache hits) to accumulate before
    # solving the DP once and freezing the hist_exp_decay placement.
    "VLLM_ASCEND_SPARSE_PREFIX_CACHE_WARMUP_REQUESTS": lambda: int(
        os.getenv("VLLM_ASCEND_SPARSE_PREFIX_CACHE_WARMUP_REQUESTS", "100")
    ),
    # Checkpoint budget M for the DP placement.
    "VLLM_ASCEND_SPARSE_PREFIX_CACHE_NUM_CHECKPOINTS": lambda: int(
        os.getenv("VLLM_ASCEND_SPARSE_PREFIX_CACHE_NUM_CHECKPOINTS", "16")
    ),
    # Per-bin attention-cost coefficient for the attention-aware recompute cost
    # model in the DP placement.
    "VLLM_ASCEND_SPARSE_PREFIX_CACHE_RECOMPUTE_ATTN_WEIGHT": lambda: float(
        os.getenv("VLLM_ASCEND_SPARSE_PREFIX_CACHE_RECOMPUTE_ATTN_WEIGHT", "0.0")
    ),
}


def __getattr__(name: str):
    # lazy evaluation of environment variables
    if name in env_variables:
        return env_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(env_variables.keys())
