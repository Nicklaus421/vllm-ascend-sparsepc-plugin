# vllm-ascend-sparse-pc

Sparse Prefix Caching for Hybrid and Recurrent LLM Serving — a monkey-patch
plugin for [vllm-ascend](https://github.com/vllm-project/vllm-ascend).

This package is a self-contained port of the placement algorithm from
*Sparse Prefix Caching for Hybrid and Recurrent LLM Serving* (arXiv:2605.05219)
and its reference implementation. It caches GatedDeltaNet (GDN) recurrent state
at a sparse set of DP-optimal checkpoint positions instead of every block,
shrinking the recurrent-state footprint so more sequences fit in the prefix
cache.

The plugin patches vllm-ascend **at runtime** — it does not require any source
changes to vllm-ascend.

## Requirements

- A **pristine** vllm-ascend (v0.23.x) — i.e. one that does **not** already
  contain the sparse-prefix-cache code. If you have a development branch with
  that code in-tree, revert/remove it first; otherwise the patches here would
  double-apply.
- vllm (the matching version vllm-ascend builds against).
- A hybrid (full-attention + GDN linear-attention) model such as the Qwen3.8 /
  Qwen3.5 GDN family.

## Install

```bash
pip install -e .
```

This installs the `vllm_ascend_sparse_pc` package (and registers the algorithm
modules). The monkey-patches are applied the first time the package is
imported.

## Activate

`pip install -e .` registers a `vllm.general_plugins` entry point
(`ascend_sparse_pc = vllm_ascend_sparse_pc:register`). vllm loads general
plugins when it starts, so the monkey-patches are applied automatically in the
engine-core and worker processes — no launcher code change and no explicit
import are required.

Two things are required for the patch to actually take effect:

1. The master switch must be on (the plugin reads its own env vars):

   ```bash
   export VLLM_ASCEND_SPARSE_PREFIX_CACHE_ENABLED=1
   export VLLM_ASCEND_SPARSE_PREFIX_CACHE_STRATEGY=fixed_interval
   ```

2. The plugin must be installed into the same Python environment vllm runs
   from (`pip install -e .`).

For a manual import (e.g. in a script that then starts vllm), importing the
package applies the patches directly:

```python
import vllm_ascend_sparse_pc  # noqa: F401  applies the monkey-patches
```

`apply()` is idempotent and safe to call repeatedly.

### Verify it took effect

Confirm the entry point is registered:

```bash
python -c "import importlib.metadata as m; print([e.value for e in m.entry_points(group='vllm.general_plugins') if 'sparse' in e.name])"
```

At runtime, the worker log should contain
`Sparse prefix cache enabled (interval=... tokens).`; with
`STRATEGY=hist_exp_decay`, the engine-core log should eventually contain
`Sparse prefix cache: solved DP placement ...` after warmup.

## Configuration

All configuration is via environment variables (read from `os.environ` by the
plugin's own `envs` module, mirroring vllm-ascend's names):

| Variable | Default | Meaning |
|---|---|---|
| `VLLM_ASCEND_SPARSE_PREFIX_CACHE_ENABLED` | `0` | master switch |
| `VLLM_ASCEND_SPARSE_PREFIX_CACHE_STRATEGY` | `fixed_interval` | `fixed_interval` or `hist_exp_decay` |
| `VLLM_ASCEND_SPARSE_PREFIX_CACHE_CHECKPOINT_INTERVAL` | `256` | v1 fixed-interval spacing (tokens) |
| `VLLM_ASCEND_SPARSE_PREFIX_CACHE_KERNEL_BLOCK_SIZE` | `16` | block granularity for clipping |
| `VLLM_ASCEND_SPARSE_PREFIX_CACHE_BIN_SIZE` | `16` | overlap-depth histogram bin |
| `VLLM_ASCEND_SPARSE_PREFIX_CACHE_NUM_CHECKPOINTS` | `16` | DP checkpoint budget `M` |
| `VLLM_ASCEND_SPARSE_PREFIX_CACHE_WARMUP_REQUESTS` | `100` | hits before the DP solve freezes |
| `VLLM_ASCEND_SPARSE_PREFIX_CACHE_GAMMA` | `0.99` | exp-decay drift factor |
| `VLLM_ASCEND_SPARSE_PREFIX_CACHE_REPLAN_INTERVAL` | `100` | re-solve cadence |
| `VLLM_ASCEND_SPARSE_PREFIX_CACHE_LAPLACE_ALPHA` | `1.0` | total Laplace pseudo-count mass |
| `VLLM_ASCEND_SPARSE_PREFIX_CACHE_BUDGET_GB` | `1.0` | budget for the standalone store |
| `VLLM_ASCEND_SPARSE_PREFIX_CACHE_RECOMPUTE_ATTN_WEIGHT` | `0.0` | attention-aware recompute cost |

## Monkey-patch design

Three integration points are patched (see `src/vllm_ascend_sparse_pc/patch.py`):

1. **`AscendHybridKVCacheCoordinator`** (`vllm_ascend.patch.platform.patch_kv_cache_coordinator`)
   — `__init__` is wrapped to initialize the DP histogram state; a sparse Mamba
   branch is injected into `find_longest_cache_hit` so the hybrid hit is the
   nearest checkpoint `C <= P` rather than a dense Mamba block run.

2. **`NPUModelRunner`** (`vllm_ascend.worker.v1.model_runner`)
   — `__init__` is wrapped to add the per-process state buffer; capture runs
   after prefill (`_update_states_after_model_execute`) and restore runs before
   the forward (`_update_states`).

3. **`AscendWorkerProc` / `WorkerProc`** (`vllm_ascend.patch.platform.patch_multiproc_executor`)
   — `make_worker_process` is replaced to hand the cross-process
   `multiprocessing.Manager` proxy (holding the frozen DP positions) to TP
   workers, and the worker entry point re-attaches it.

### Note on restore placement

The in-tree reference code calls `_maybe_restore_sparse_checkpoints` deep inside
`execute_model`, after `_prepare_inputs` and the `mamba_cache_mode == "align"`
preprocess. This plugin instead restores immediately after `_update_states`
(which has already rewound `num_computed_tokens` to `C` and built the
`block_table`). For `mamba_cache_mode="all"` — the only mode sparse prefix
caching supports — nothing between `_update_states` and the forward writes the
recurrent-state blocks, so the two placements are equivalent. This keeps the
patch a single small method wrap instead of a full copy of `execute_model`.

## Tests

```bash
python -m pytest tests/
```

The DP solver tests run on CPU only (NumPy) and do not require an NPU or vllm.

## License

Apache-2.0.
