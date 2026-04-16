#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
simulations.py

Unified simulation driver for the UD-DML study.
This script orchestrates five experiment families described in UD_DML.pdf.
Pre-main ``bgamma_sensitivity`` / ``efficiency_profile`` are currently disabled
(see ``run_profiling_before_experiment_families``). Standalone ``--experiment``
runs still use the functions below; default Monte Carlo replications for
``bgamma_sensitivity`` are 50 unless overridden by ``--replications``.

Main experiment families:

1. Covariate space visualisation (UD vs. UNIF)
2. Subsample-budget comparison across all DGPs
3. Population-size scaling for low/high subsample budgets
4. Double-robustness stress test in observational settings
5. Nuisance-learner sensitivity on OBS-3

The driver coordinates checkpointed Monte Carlo replications,
stores raw outputs, and delegates aggregation/visualisation to
`evaluation.generate_reports`.
"""

import argparse
import contextlib
import csv
import gc
import pickle
import time
import traceback
import gzip
import os
import math
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Set, Tuple


# -----------------------------------------------------------------------------
# Sanitise thread-count environment variables *before* numpy / scipy / lightgbm
# are imported.  libgomp refuses to initialise if any of these is "" or "0"
# and spams "Invalid value for environment variable OMP_NUM_THREADS" on every
# subsequent OpenMP init.  We drop empty / zero / non-integer values so
# libgomp uses its internal default; explicit positive integers are kept.
# -----------------------------------------------------------------------------
def _sanitize_thread_env_main_process() -> None:
    bad = []
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "GOMP_CPU_AFFINITY",   # libgomp also errors on malformed affinity strings
        "UD_GENUD_NUM_THREADS",
    ):
        val = os.environ.get(key)
        if val is None:
            continue
        stripped = val.strip()
        if stripped == "":
            bad.append((key, val))
            os.environ.pop(key, None)
            continue
        # For thread-count keys (not GOMP_CPU_AFFINITY) require positive int.
        if key == "GOMP_CPU_AFFINITY":
            continue
        try:
            if int(stripped, 10) < 1:
                bad.append((key, val))
                os.environ.pop(key, None)
        except ValueError:
            bad.append((key, val))
            os.environ.pop(key, None)
    if bad:
        # Print once so users know we overrode their environment.
        msg = ", ".join(f"{k}={v!r}" for k, v in bad)
        print(
            f"[simulations] sanitised invalid thread-env vars (removed): {msg}",
            flush=True,
        )


_sanitize_thread_env_main_process()

import joblib.parallel as joblib_parallel
import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from joblib.parallel import BatchCompletionCallBack
from tqdm.auto import tqdm


# -----------------------------------------------------------------------------
# Publication-quality matplotlib defaults.
#
# Applied BEFORE ``evaluation`` is imported so every downstream figure
# (simulations.py bgamma sensitivity, evaluation.py report plots) inherits
# the same larger, bolder labels / titles / legends.  Override any single
# key locally with ``plt.rcParams.update(...)`` or per-axes kwargs.
#
# Sizes are calibrated for figsize ≈ (8-10, 4-6 in):
#   * axes.titlesize / figure.titlesize > axes.labelsize > tick/legend
#   * bold titles + labels to stand out on LaTeX / double-column layouts
#   * 300 dpi PNG savefig with tight bbox (also safe for vector exports)
# -----------------------------------------------------------------------------
def _configure_matplotlib_publication() -> None:
    plt.rcParams.update({
        # Global font
        "font.family": "serif",
        "font.size": 13,
        "mathtext.fontset": "stix",
        # Titles & axis labels
        "axes.titlesize": 17,
        "axes.titleweight": "bold",
        "axes.labelsize": 15,
        "axes.labelweight": "bold",
        "figure.titlesize": 18,
        "figure.titleweight": "bold",
        # Ticks
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "xtick.major.size": 5,
        "ytick.major.size": 5,
        "xtick.major.width": 1.2,
        "ytick.major.width": 1.2,
        "xtick.direction": "in",
        "ytick.direction": "in",
        # Legend
        "legend.fontsize": 13,
        "legend.title_fontsize": 14,
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "0.4",
        "legend.borderpad": 0.5,
        # Lines / axes
        "axes.linewidth": 1.2,
        "lines.linewidth": 2.0,
        "lines.markersize": 6,
        "grid.linewidth": 0.7,
        "grid.alpha": 0.35,
        # Figure / save
        "figure.dpi": 110,
        # NB: keep autolayout OFF.  Several evaluation.py plots call
        # ``fig.tight_layout(rect=...)`` explicitly to reserve top/bottom
        # margin for suptitle + figure-level legend; autolayout overrides
        # those rects and the legend ends up overlapping the subplots.
        "figure.autolayout": False,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    })


_configure_matplotlib_publication()

import config
import evaluation
import methods

# =============================================================================
# Constants & paths
# =============================================================================

FAST_DEMO_MODE = False
FAST_DEMO_OVERRIDES = {
    "n_replications": 10,
}
# Standalone UD profiling experiments (efficiency_profile, bgamma_sensitivity): full scale,
# replications = ADDON_EXPERIMENT_REPLICATIONS_FULL in normal runs and FAST_DEMO_OVERRIDES["n_replications"] in --fast-demo.
# (Pre-main hooks are commented out; constants kept for re-enabling.)
ADDON_EXPERIMENT_SCENARIO: str = "OBS-3"
ADDON_EXPERIMENT_N: int = 100_000
ADDON_EXPERIMENT_R_TOTAL: int = 1_000
ADDON_EXPERIMENT_REPLICATIONS_FULL: int = 50

# Standalone profiling outputs (efficiency profile, B_gamma sensitivity) live here.
ANALYSIS_RESULTS_ROOT = Path("analysis_results")

MAX_PARALLEL_JOBS = config.MAX_PARALLEL_JOBS
ENV_MAX_JOBS = os.environ.get("OS_DML_MAX_JOBS")
if ENV_MAX_JOBS:
    try:
        MAX_PARALLEL_JOBS = max(1, min(int(ENV_MAX_JOBS), MAX_PARALLEL_JOBS))
    except ValueError:
        pass

# joblib: number of *batches* to queue ahead (with batch_size=1, bounds in-flight tasks / IPC pressure).
# Override with env ``OS_DML_PRE_DISPATCH`` (e.g. ``n_jobs``, ``2 * n_jobs``). Tighter = lower RAM spikes.
_ENV_PRE_DISPATCH = os.environ.get("OS_DML_PRE_DISPATCH", "").strip()
# Max tasks per ``Parallel(...)`` call when reusing one executor (bounds scheduler + IPC peaks).
# Override with env ``OS_DML_PARALLEL_MEM_CHUNK`` (integer). Smaller = safer for huge *n* DML per replication.
_ENV_MEM_CHUNK = os.environ.get("OS_DML_PARALLEL_MEM_CHUNK", "").strip()
# In joblib worker processes: force nested BLAS/OpenMP thread counts (default 1). Disable with OS_DML_DISABLE_WORKER_THREAD_CAP=1.
# OS_DML_WORKER_NUM_THREADS must be a positive integer (0 / invalid → treated as 1; libgomp rejects OMP_NUM_THREADS=0).
_ENV_DISABLE_WORKER_THREAD_CAP = os.environ.get(
    "OS_DML_DISABLE_WORKER_THREAD_CAP", ""
).strip().lower() in ("1", "true", "yes", "on")

# Per-process cache: ``config.get_experiments()`` rebuilds large dicts; workers call it on every replication.
_SIM_SCENARIOS_METHODS_CACHE: Optional[Tuple[Dict, Dict]] = None


def _get_scenarios_and_methods_cached() -> Tuple[Dict, Dict]:
    """Return (scenarios, all_methods) once per process (avoids repeated catalog construction)."""
    global _SIM_SCENARIOS_METHODS_CACHE
    if _SIM_SCENARIOS_METHODS_CACHE is None:
        s, m, _ = config.get_experiments()
        _SIM_SCENARIOS_METHODS_CACHE = (s, m)
    return _SIM_SCENARIOS_METHODS_CACHE


# =============================================================================
# Helper utilities
# =============================================================================

# tqdm: show fraction, wall elapsed, ETA, rate, and a live postfix (scenario/method/r/mem).
_SIM_TQDM_KWARGS = {
    "unit": "repl",
    "dynamic_ncols": True,
    "mininterval": 0.25,
    "smoothing": 0.15,
    "bar_format": (
        "{l_bar}{bar}| {n_fmt}/{total_fmt} "
        "[elapsed {elapsed} | ETA {remaining} | {rate_fmt}{postfix}]"
    ),
}

# Optional: rich memory readout in the progress bar.
try:
    import psutil as _psutil  # type: ignore[import-not-found]
    _PSUTIL_PROC = _psutil.Process(os.getpid())
except Exception:  # pragma: no cover
    _psutil = None
    _PSUTIL_PROC = None


def _format_mem_postfix() -> Optional[str]:
    """Return a compact ``RSS=xxGB`` string for the driver process (or None)."""
    if _PSUTIL_PROC is None:
        return None
    try:
        rss = _PSUTIL_PROC.memory_info().rss
    except Exception:  # pragma: no cover
        return None
    return f"rss={rss / (1024 ** 3):4.1f}GB"


def _update_tqdm_postfix(
    bar: "tqdm",
    result: Optional[dict],
    include_memory: bool = True,
) -> None:
    """Show last-completed task metadata + optional memory in the tqdm postfix."""
    parts: List[str] = []
    if isinstance(result, dict):
        scen = result.get("scenario")
        meth = result.get("method")
        r_tot = result.get("r_total")
        pop = result.get("population_size")
        if scen and meth:
            tag = f"{scen}/{meth}"
            # FULL method trains DML on full population; r_total was set to
            # population_size upstream, which is misleading in the progress bar.
            # Show n=population for FULL, r=subsample for UD/UNIF.
            if meth == "FULL" and pop is not None:
                tag += f"(n={int(pop)})"
            elif r_tot is not None:
                tag += f"(r={int(r_tot)})"
            parts.append(tag)
    if include_memory:
        mem = _format_mem_postfix()
        if mem:
            parts.append(mem)
    if parts:
        bar.set_postfix_str(" | ".join(parts), refresh=False)


@contextlib.contextmanager
def _tqdm_joblib(tqdm_bar: tqdm) -> Iterator[tqdm]:
    """Route joblib batch completions into a tqdm bar (per-finished-replication updates).

    Also refreshes a live memory postfix so users can watch driver RSS trend
    (helpful when running huge-n replications on shared machines).
    """

    class _TqdmBatchCompletionCallBack(BatchCompletionCallBack):
        def __call__(self, out):
            tqdm_bar.update(self.batch_size)
            mem = _format_mem_postfix()
            if mem:
                tqdm_bar.set_postfix_str(mem, refresh=False)
            return super().__call__(out)

    old_bcb = joblib_parallel.BatchCompletionCallBack
    joblib_parallel.BatchCompletionCallBack = _TqdmBatchCompletionCallBack
    try:
        yield tqdm_bar
    finally:
        joblib_parallel.BatchCompletionCallBack = old_bcb


def _sanitize_token(token):
    """Make tokens filesystem-safe."""
    token_str = str(token)
    return token_str.replace(" ", "_").replace(".", "p").replace("-", "m")


def _trim_for_fast_demo(values):
    return list(values)


def _apply_fast_demo_overrides(variant):
    if not FAST_DEMO_MODE:
        return variant
    variant["n_replications"] = min(
        variant.get("n_replications", config.DEFAULT_REPLICATIONS),
        FAST_DEMO_OVERRIDES["n_replications"],
    )
    return variant


def _prepare_sampling_config(method_name, variant, scenario_name):
    r_total = variant.get("r_total")
    if method_name in ("UD", "UNIF"):
        if r_total is None:
            raise ValueError(f"{method_name} requires 'r_total' in variant configuration.")
    else:  # FULL method
        r_total = variant.get("population_size", config.N_POPULATION)
    return {
        "r_total": r_total,
    }


def _resolve_population_size(variant, scenario_cfg):
    return int(variant.get("population_size") or scenario_cfg["params"].get("n", config.N_POPULATION))


def _resolve_ud_cache_seed(scenario_name, variant, sampling_cfg):
    return methods.get_ud_cache_seed(
        int(sampling_cfg["r_total"]),
        scenario_name=scenario_name,
        population_size=int(variant.get("population_size", config.N_POPULATION)),
        B_gamma=variant.get("B_gamma"),
    )


def _compose_variant_label(method_name, variant, sampling_cfg):
    parts = [variant.get("label", "baseline")]
    if variant.get("misspecification"):
        parts.append(variant["misspecification"])
    if variant.get("learner"):
        parts.append(variant["learner"])
    tokens = [_sanitize_token(part) for part in parts if part]
    return "_".join(tokens)


def _build_checkpoint_path(checkpoint_dir, scenario_name, method_name, sim_id, variant, sampling_cfg):
    label = _compose_variant_label(method_name, variant, sampling_cfg)
    filename = f"sim_{sim_id:04d}_{method_name}_{label}.pkl.gz"
    return checkpoint_dir / scenario_name / filename


def _generate_variant_blueprints(exp_name, exp_config):
    """Expand experiment parameter grids into variant blueprints."""
    params = exp_config["params"]
    base = {
        "n_estimators": params.get("n_estimators", config.LGBM_N_ESTIMATORS),
        "k_folds": params.get("k_folds", config.K_FOLDS),
        "population_size": params.get("population_size", config.N_POPULATION),
        "r_total": params.get("r_total"),
        "n_replications": params.get("n_replications", config.DEFAULT_REPLICATIONS),
        "store_sample": params.get("store_sample", False),
        "learner": params.get("learner", getattr(config, "DEFAULT_NUISANCE_LEARNER", "lgbm")),
    }

    variants = []

    if exp_name == "experiment_visualization":
        variant = base.copy()
        variant.update({"label": "viz"})
        variants.append(_apply_fast_demo_overrides(variant))
    elif exp_name == "experiment_subsample_size":
        for r_total in _trim_for_fast_demo(params.get("r_totals", [])):
            variant = base.copy()
            variant.update({"label": f"r-{int(r_total)}", "r_total": int(r_total)})
            variants.append(_apply_fast_demo_overrides(variant))
    elif exp_name == "experiment_population_size":
        population_sizes = params.get("population_sizes", [])
        r_totals = params.get("r_totals", [])
        if FAST_DEMO_MODE:
            population_sizes = _trim_for_fast_demo(population_sizes)
            r_totals = _trim_for_fast_demo(r_totals)
        for population in population_sizes:
            # FULL baseline (no subsample budget)
            variant_full = base.copy()
            variant_full.update(
                {
                    "label": f"N-{int(population)}_full",
                    "population_size": int(population),
                    "r_total": None,
                    "method_whitelist": {"FULL"},
                }
            )
            variants.append(_apply_fast_demo_overrides(variant_full))
            for r_total in r_totals:
                variant = base.copy()
                variant.update(
                    {
                        "label": f"N-{int(population)}_r-{int(r_total)}",
                        "population_size": int(population),
                        "r_total": int(r_total),
                        "method_whitelist": {"UD", "UNIF"},
                    }
                )
                variants.append(_apply_fast_demo_overrides(variant))
    elif exp_name == "experiment_double_robust":
        for misspec in params.get("misspecification_scenarios", []):
            variant = base.copy()
            variant.update(
                {
                    "label": f"misspec-{misspec}",
                    "misspecification": misspec,
                    "r_total": params["r_total"],
                }
            )
            variants.append(_apply_fast_demo_overrides(variant))
    elif exp_name == "experiment_nuisance_sensitivity":
        r_totals = params.get("r_totals", [])
        learners = params.get("nuisance_learners", [])
        if FAST_DEMO_MODE:
            r_totals = _trim_for_fast_demo(r_totals)
        for learner in learners:
            for r_total in r_totals:
                variant = base.copy()
                variant.update(
                    {
                        "label": f"{learner}_r-{int(r_total)}",
                        "learner": learner,
                        "r_total": int(r_total),
                    }
                )
                variants.append(_apply_fast_demo_overrides(variant))
    else:
        variant = base.copy()
        variant["label"] = "default"
        variants.append(_apply_fast_demo_overrides(variant))

    return variants


# =============================================================================
# Core execution
# =============================================================================


def _worker_nested_thread_count_str() -> str:
    """Parse ``OS_DML_WORKER_NUM_THREADS``; libgomp requires OMP_NUM_THREADS >= 1 (never 0)."""
    raw = os.environ.get("OS_DML_WORKER_NUM_THREADS")
    if raw is None or not str(raw).strip():
        return "1"
    try:
        v = int(str(raw).strip(), 10)
    except ValueError:
        return "1"
    if v < 1:
        return "1"
    return str(min(v, 1024))


def _replication_worker_cap_nested_threads() -> None:
    """Cap BLAS/OpenMP/genUD threads inside joblib workers.

    Outer parallelism is at the replication level (joblib), so each worker
    should use 1 thread internally.  Otherwise n_workers × n_cores threads
    compete for the same hardware, which slows everyone down.
    """
    if _ENV_DISABLE_WORKER_THREAD_CAP:
        return
    try:
        import multiprocessing as mp
    except Exception:
        return
    if mp.current_process().name == "MainProcess":
        return
    n = _worker_nested_thread_count_str()
    # Defensive: if libgomp has already been loaded in the worker with a bad
    # value (inherited empty / "0" from shell), drop GOMP_CPU_AFFINITY too so
    # our freshly-set OMP_NUM_THREADS is honoured.
    os.environ.pop("GOMP_CPU_AFFINITY", None)
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "UD_GENUD_NUM_THREADS",   # pin genUD.dll internal worker pool
    ):
        os.environ[key] = n


def run_single_replication(task):
    """Execute one Monte Carlo replication with checkpoint support."""
    _replication_worker_cap_nested_threads()
    exp_name, scenario_name, method_name, sim_id, variant, checkpoint_dir = task
    variant = variant.copy()
    sampling_cfg = _prepare_sampling_config(method_name, variant, scenario_name)
    checkpoint_file = _build_checkpoint_path(
        checkpoint_dir, scenario_name, method_name, sim_id, variant, sampling_cfg
    )

    if checkpoint_file.exists():
        try:
            with gzip.open(checkpoint_file, "rb") as f:
                result = pickle.load(f)
            if "est_ate" in result and "scenario" in result:
                return result
        except Exception:
            pass  # corrupted; recompute

    try:
        np.random.seed(config.BASE_SEED + int(sim_id))
        scenarios, all_methods = _get_scenarios_and_methods_cached()
        scenario_cfg = scenarios[scenario_name]
        method_cfg = all_methods[method_name]

        data_params = dict(scenario_cfg["params"])
        if variant.get("population_size"):
            data_params["n"] = variant["population_size"]
        data = scenario_cfg["data_gen_func"](**data_params)
        func = method_cfg["func"]

        learner = variant.get("learner", getattr(config, "DEFAULT_NUISANCE_LEARNER", "lgbm"))
        method_kwargs = {
            "k_folds": variant.get("k_folds", config.K_FOLDS),
            "n_estimators": variant.get("n_estimators", config.LGBM_N_ESTIMATORS),
            "sim_seed": config.BASE_SEED + int(sim_id),
            "learner": learner,
        }
        if variant.get("misspecification"):
            method_kwargs["misspecification"] = variant["misspecification"]
        if variant.get("store_sample"):
            method_kwargs["store_sample"] = True

        if method_name in ("UD", "UNIF"):
            method_kwargs["r"] = {"r_total": sampling_cfg["r_total"]}
        if method_name == "UD":
            method_kwargs["cache_seed"] = _resolve_ud_cache_seed(
                scenario_name, variant, sampling_cfg
            )
            method_kwargs["scenario_name"] = scenario_name
            method_kwargs["population_size"] = variant.get(
                "population_size", config.N_POPULATION
            )

        result = func(
            data["X"],
            data["W"],
            data["Y_obs"],
            data["pi_true"],
            scenario_cfg["design"] == "rct",
            **method_kwargs,
        )

        metadata = {
            "exp_name": exp_name,
            "scenario": scenario_name,
            "method": method_name,
            "sim_id": sim_id,
            "true_ate": data["true_ate"],
            "r_total": sampling_cfg["r_total"],
            "population_size": data_params.get("n", config.N_POPULATION),
            "variant_label": _compose_variant_label(method_name, variant, sampling_cfg),
            "n_estimators": variant.get("n_estimators", config.LGBM_N_ESTIMATORS),
            "k_folds": variant.get("k_folds", config.K_FOLDS),
            "learner": learner,
            "store_sample": variant.get("store_sample", False),
            "covariates": scenario_cfg.get("covariates", "x1"),
        }
        if variant.get("misspecification"):
            metadata["misspecification"] = variant["misspecification"]

        result.update(metadata)

        if isinstance(metadata.get("store_sample"), bool) and metadata["store_sample"] and method_name in ("UD", "UNIF"):
            cov_key = metadata["covariates"]
            dims_map = {"x1": (0, 1), "x2": (0, 5), "x3": (0, 5)}
            dims = dims_map.get(cov_key, (0, 1))
            dims = tuple(int(d) for d in dims)
            full_proj = data["X"][:, list(dims)].astype(np.float32, copy=False)
            subs_idx = np.asarray(result.get("subsample_indices", []), dtype=np.int64)
            subs_proj = full_proj[subs_idx] if subs_idx.size else np.empty((0, len(dims)), dtype=np.float32)
            result["full_projection"] = full_proj
            result["subsample_projection"] = subs_proj
            result["projection_dims"] = dims
        if (
            isinstance(metadata.get("store_sample"), bool)
            and metadata["store_sample"]
            and method_name in ("UD", "UNIF")
            and metadata.get("scenario", "").startswith("OBS")
        ):
            result["propensity_full"] = np.asarray(data["pi_true"], dtype=np.float32)
            result["treatment_full"] = np.asarray(data["W"], dtype=np.int8)

        result = _prepare_result_for_storage(result)
        # Drop full-population dict in the worker as soon as outputs are detached (reduces peak RAM).
        del data
        if _task_population(task) >= 300_000:
            gc.collect()

        legacy_file = checkpoint_file.with_suffix("")
        if legacy_file.exists():
            try:
                legacy_file.unlink()
            except Exception:
                pass
        checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(checkpoint_file, "wb") as f:
            pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)

        return result
    except Exception as exc:
        print(
            f"Error in sim {sim_id} ({scenario_name}, {method_name}, "
            f"{variant.get('label', 'variant')}): {exc}"
        )
        traceback.print_exc()
        return None


def _try_load_checkpoint_result(task):
    """If a valid checkpoint exists, load and return it in a single disk read; else None."""
    exp_name, scenario_name, method_name, sim_id, variant, checkpoint_dir = task
    sampling_cfg = _prepare_sampling_config(method_name, variant, scenario_name)
    checkpoint_file = _build_checkpoint_path(
        checkpoint_dir, scenario_name, method_name, sim_id, variant, sampling_cfg
    )
    opener = gzip.open
    if not checkpoint_file.exists():
        legacy_file = checkpoint_file.with_suffix("")
        if legacy_file.exists():
            checkpoint_file = legacy_file
            opener = open
        else:
            return None
    try:
        with opener(checkpoint_file, "rb") as f:
            result = pickle.load(f)
        if "est_ate" in result and "scenario" in result:
            return result
    except Exception:
        pass
    return None


def _prepare_result_for_storage(result):
    compact = result.copy()
    sample = compact.get("probability_sample")
    if sample is not None:
        compact["probability_sample"] = np.asarray(sample, dtype=np.float32)
    for key in ("full_projection", "subsample_projection", "propensity_full"):
        if key in compact and compact[key] is not None:
            compact[key] = np.asarray(compact[key], dtype=np.float32)
    if "subsample_indices" in compact and compact["subsample_indices"] is not None:
        compact["subsample_indices"] = np.asarray(compact["subsample_indices"], dtype=np.int32)
    if "treatment_full" in compact and compact["treatment_full"] is not None:
        compact["treatment_full"] = np.asarray(compact["treatment_full"], dtype=np.int8)
    return compact

def _task_population(task):
    _, scenario_name, _, _, variant, _ = task
    population = variant.get("population_size")
    if population is None:
        population = config.N_POPULATION
    return int(population)


def _parallel_pre_dispatch(max_population: int) -> str:
    """joblib ``pre_dispatch`` expression: fewer queued batches for large *n* (heavy DML per task)."""
    if _ENV_PRE_DISPATCH:
        return _ENV_PRE_DISPATCH
    if max_population >= 500_000:
        return "1"
    if max_population >= 300_000:
        return "n_jobs"
    return "2 * n_jobs"


def _parallel_task_chunk_len(
    capped_jobs: int, max_population: int, total_pending: int
) -> int:
    """How many pending tasks to submit per ``Parallel`` call (same executor, chunked for RAM)."""
    if _ENV_MEM_CHUNK:
        try:
            return max(1, min(int(_ENV_MEM_CHUNK), total_pending))
        except ValueError:
            pass
    if max_population >= 500_000:
        return max(1, min(capped_jobs, total_pending))
    if max_population >= 300_000:
        return max(1, min(2 * capped_jobs, total_pending))
    return max(1, min(4 * capped_jobs, total_pending))


def _task_sort_key(task):
    _, scenario_name, method_name, sim_id, variant, _ = task
    r_total = variant.get("r_total")
    r_total = -1 if r_total is None else int(r_total)
    method_rank = {"UD": 0, "FULL": 1, "UNIF": 2}.get(method_name, 9)
    return (
        -_task_population(task),
        method_rank,
        -r_total,
        scenario_name,
        int(sim_id),
    )


def _collect_unique_ud_specs(experiment_names: List[str]) -> List[Tuple[str, int, int, Optional[int]]]:
    """Collect unique UD skeleton specs from the selected experiments."""
    scenarios, _, experiments = config.get_experiments()
    specs: Set[Tuple[str, int, int, Optional[int]]] = set()
    for exp_name in experiment_names:
        exp_config = experiments.get(exp_name)
        if exp_config is None or "UD" not in exp_config.get("methods", []):
            continue
        variants = _generate_variant_blueprints(exp_name, exp_config)
        for scenario_name in exp_config["scenarios"]:
            scenario_cfg = scenarios[scenario_name]
            for variant in variants:
                allowed_methods = variant.get("method_whitelist")
                if allowed_methods and "UD" not in allowed_methods:
                    continue
                sampling_cfg = _prepare_sampling_config("UD", variant, scenario_name)
                specs.add(
                    (
                        scenario_name,
                        _resolve_population_size(variant, scenario_cfg),
                        int(sampling_cfg["r_total"]),
                        variant.get("B_gamma"),
                    )
                )
    return sorted(specs)


def _infer_ud_working_dimension(X: np.ndarray) -> int:
    """Infer retained PCA dimension q for skeleton precomputation."""
    rho_0 = getattr(config, "UD_VARIANCE_THRESHOLD", 0.85)
    X_tilde = methods._standardise_covariates(X.astype(np.float64))
    _Z_all, _V_q, q = methods._pca_rotate(X_tilde, rho_0)
    return int(q)


def precompute_ud_skeletons(experiment_names: List[str]) -> None:
    """Precompute and persist all UD skeletons needed by selected experiments."""
    specs = _collect_unique_ud_specs(experiment_names)
    if not specs:
        return

    scenarios, _, _ = config.get_experiments()
    _print_banner(
        f"Precomputing UD skeletons ({len(specs)} unique specs)",
        [
            f"UD C backend: {'ENABLED (genUD.dll, ~30-500x faster)' if methods.ud_c_backend_active() else 'DISABLED (Python)'}",
            "keys     : (r_p, q, B_gamma, cache_seed)  — shared disk cache ./ud_skeleton_cache",
        ],
    )
    start = time.perf_counter()
    prepared = 0
    reused = 0
    seen_keys: Set[Tuple[int, int, int, int]] = set()

    warmup_iter = tqdm(
        specs,
        desc="UD skeleton warmup",
        unit="spec",
        dynamic_ncols=True,
        mininterval=0.25,
        bar_format=(
            "{l_bar}{bar}| {n_fmt}/{total_fmt} "
            "[elapsed {elapsed} | ETA {remaining}{postfix}]"
        ),
    )
    for scenario_name, population_size, r_total, B_gamma in warmup_iter:
        scenario_cfg = scenarios[scenario_name]
        data_params = dict(scenario_cfg["params"])
        data_params["n"] = int(population_size)
        np.random.seed(config.BASE_SEED)
        data = scenario_cfg["data_gen_func"](**data_params)
        q = _infer_ud_working_dimension(data["X"])
        treated = int(np.count_nonzero(data["W"] == 1))
        control = int(np.count_nonzero(data["W"] == 0))
        r_p = min(treated, control, int(r_total) // 2)
        if r_p <= 0:
            raise ValueError(
                f"Cannot precompute UD skeleton for {scenario_name} with r_total={r_total}."
            )
        B = int(B_gamma if B_gamma is not None else getattr(config, "UD_MAX_GENERATOR_CANDIDATES", 30))
        cache_seed = methods.get_ud_cache_seed(
            r_total,
            scenario_name=scenario_name,
            population_size=population_size,
            B_gamma=B_gamma,
        )
        key = (int(r_p), int(q), int(B), int(cache_seed))
        warmup_iter.set_postfix_str(
            f"{scenario_name} r_p={r_p} q={q}", refresh=False,
        )
        if key in seen_keys:
            reused += 1
            continue
        _U, from_cache = methods.warm_start_ud_skeleton(
            r_p,
            q,
            B_gamma=B_gamma,
            cache_seed=cache_seed,
        )
        prepared += 1
        if from_cache:
            reused += 1
        seen_keys.add(key)
        del data
    warmup_iter.close()

    elapsed = time.perf_counter() - start
    print(
        f"-> UD skeleton warmup: prepared {prepared} unique keys, "
        f"reused {reused} cached hits in {elapsed:.2f}s.",
        flush=True,
    )


def _ensure_ud_skeleton_cache_dir() -> None:
    """Create the UD skeleton cache directory *without* precomputing entries.

    Workers will populate it lazily on first miss (via the disk-atomic save
    inside ``methods._select_optimal_uniform_design``).  Subsequent workers
    and replications that need the same ``(r_p, q, B_gamma, cache_seed)``
    read the cached skeleton straight from disk.
    """
    cache_root = methods._ud_skeleton_disk_cache_root()
    if cache_root is None:
        _print_banner(
            "UD skeleton cache",
            [
                f"UD C backend: {'ENABLED (genUD.dll)' if methods.ud_c_backend_active() else 'DISABLED (Python fallback)'}",
                "disk cache   : DISABLED (env UD_SKELETON_DISK_CACHE is off)",
                "mode         : lazy, in-memory only (per worker process)",
            ],
        )
        return
    cache_root.mkdir(parents=True, exist_ok=True)
    try:
        existing = sum(1 for _ in cache_root.glob("ud_r*_q*_Bg*_seed*.npy"))
    except OSError:
        existing = 0
    _print_banner(
        "UD skeleton cache",
        [
            f"UD C backend : {'ENABLED (genUD.dll)' if methods.ud_c_backend_active() else 'DISABLED (Python fallback)'}",
            f"cache dir    : {cache_root}",
            f"existing     : {existing} skeleton file(s) on disk",
            "mode         : lazy — workers populate the cache as simulations run",
        ],
    )


def _addon_experiment_replications() -> int:
    """Replications for efficiency_profile / bgamma_sensitivity when invoked from run_all."""
    if FAST_DEMO_MODE:
        return int(FAST_DEMO_OVERRIDES["n_replications"])
    return int(ADDON_EXPERIMENT_REPLICATIONS_FULL)


def run_profiling_before_experiment_families() -> None:
    """Pre-main profiling (disabled). Previously ran ``bgamma_sensitivity`` then ``efficiency_profile``.

    Uncomment the block below and the ``run_profiling_before_experiment_families()`` call
    in ``run_all`` to restore. Replication count: ``ADDON_EXPERIMENT_REPLICATIONS_FULL``
    (full) or ``FAST_DEMO_OVERRIDES`` (``--fast-demo``).
    """
    # reps = _addon_experiment_replications()
    # print(f"\n{'='*80}\nPRE-MAIN: bgamma_sensitivity ({reps} reps, demo={FAST_DEMO_MODE})\n{'='*80}")
    # out_bg = run_bgamma_sensitivity_experiment(
    #     scenario=ADDON_EXPERIMENT_SCENARIO,
    #     n=ADDON_EXPERIMENT_N,
    #     replications=reps,
    #     r_total=ADDON_EXPERIMENT_R_TOTAL,
    # )
    # print(f"-> bgamma_sensitivity -> {out_bg.resolve()}")
    # print(f"\n{'='*80}\nPRE-MAIN: efficiency_profile ({reps} reps, demo={FAST_DEMO_MODE})\n{'='*80}")
    # out_eff = run_efficiency_profile_experiment(
    #     scenario=ADDON_EXPERIMENT_SCENARIO,
    #     n=ADDON_EXPERIMENT_N,
    #     replications=reps,
    #     r_total=ADDON_EXPERIMENT_R_TOTAL,
    # )
    # print(f"-> efficiency_profile -> {out_eff.resolve()}")
    return


PROFILE_STEP_KEYS: List[str] = [
    "standardize_pca",
    "ecdf_sort",
    "design_search",
    "inverse_cdf_map",
    "kd_build",
    "matching",
    "dml",
    "inference",
]

# Publication-friendly stage labels for efficiency-profile tables (LaTeX text mode).
PROFILE_STEP_LABELS_TEX: dict = {
    "standardize_pca": r"Standardize + PCA",
    "ecdf_sort": r"ECDF sort",
    "design_search": r"Design search",
    "inverse_cdf_map": r"Inverse CDF map",
    "kd_build": r"$k$-d tree build",
    "matching": r"Matching",
    "dml": r"DML (nuisance + outcome)",
    "inference": r"Inference",
}


def _require_columns(row: dict, required: set, context: str) -> None:
    missing = required.difference(row.keys())
    if missing:
        raise KeyError(f"{context}: missing columns {sorted(missing)}")


def run_efficiency_profile_experiment(
    scenario: str = "OBS-3",
    n: int = 100_000,
    replications: int = 100,
    r_total: int = 1_000,
    output_root: Optional[Path] = None,
) -> Path:
    """Profile UD-DML wall times per algorithm stage (single scenario, many replications).

    Writes ``profile_rows.csv``, ``profile_shares.csv``, two boxplot figures,
    ``profile_summary.md``, and ``efficiency_profile_table.tex`` under
    ``analysis_results/efficiency_profile/`` (unless ``output_root`` is set).
    """
    scenarios, _, _ = config.get_experiments()
    if scenario not in scenarios:
        raise ValueError(f"Unknown scenario {scenario!r}.")
    scenario_cfg = scenarios[scenario]
    out_dir = output_root or (ANALYSIS_RESULTS_ROOT / "efficiency_profile")
    out_dir.mkdir(parents=True, exist_ok=True)

    header = [
        "replication",
        "scenario",
        "n",
        "r_total",
        "B_gamma",
    ] + PROFILE_STEP_KEYS + ["total"]

    rows = []
    B_gamma = int(getattr(config, "UD_MAX_GENERATOR_CANDIDATES", 30))

    for rep in tqdm(
        range(replications),
        desc="efficiency_profile",
        **_SIM_TQDM_KWARGS,
    ):
        sim_seed = int(config.BASE_SEED) + int(rep)
        np.random.seed(sim_seed)
        data_params = dict(scenario_cfg["params"])
        data_params["n"] = int(n)
        data = scenario_cfg["data_gen_func"](**data_params)
        out = methods.run_ud(
            data["X"],
            data["W"],
            data["Y_obs"],
            data["pi_true"],
            scenario_cfg["design"] == "rct",
            {"r_total": int(r_total)},
            k_folds=config.K_FOLDS,
            sim_seed=sim_seed,
            return_profile=True,
            learner=config.DEFAULT_NUISANCE_LEARNER,
        )
        tb = out.get("time_breakdown")
        if not isinstance(tb, dict):
            raise RuntimeError("run_ud(..., return_profile=True) must return time_breakdown dict.")
        _require_columns(tb, set(PROFILE_STEP_KEYS + ["total"]), "time_breakdown")
        record = {
            "replication": rep,
            "scenario": scenario,
            "n": int(n),
            "r_total": int(r_total),
            "B_gamma": B_gamma,
        }
        for k in PROFILE_STEP_KEYS:
            record[k] = float(tb[k])
        record["total"] = float(tb["total"])
        rows.append(record)

    rows_path = out_dir / "profile_rows.csv"
    with rows_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        w.writerows(rows)

    share_header = header.copy()
    shares_path = out_dir / "profile_shares.csv"
    share_rows = []
    for r in rows:
        tot = float(r["total"])
        if tot <= 0.0:
            raise ValueError("total time must be positive.")
        sr = {k: r[k] for k in ["replication", "scenario", "n", "r_total", "B_gamma"]}
        for k in PROFILE_STEP_KEYS:
            sr[k] = float(r[k]) / tot
        sr["total"] = 1.0
        share_rows.append(sr)

    with shares_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=share_header)
        w.writeheader()
        w.writerows(share_rows)

    # Raw times boxplot
    fig, ax = plt.subplots(figsize=(10, 4))
    data_cols = [ [ float(r[k]) for r in rows ] for k in PROFILE_STEP_KEYS ]
    ax.boxplot(data_cols, labels=PROFILE_STEP_KEYS, showmeans=True)
    ax.set_ylabel("Time (s)")
    ax.set_title("UD-DML stage times (raw)")
    plt.xticks(rotation=35, ha="right")
    fig.tight_layout()
    fig.savefig(out_dir / "time_raw_boxplot.png", dpi=200)
    plt.close(fig)

    # Cumulative share boxplots (stacked-style diagnostic)
    cum = np.zeros((len(rows), len(PROFILE_STEP_KEYS)))
    for i, r in enumerate(rows):
        tot = float(r["total"])
        cumul = 0.0
        for j, k in enumerate(PROFILE_STEP_KEYS):
            cumul += float(r[k]) / tot
            cum[i, j] = cumul

    fig, ax = plt.subplots(figsize=(10, 4))
    positions = np.arange(1, len(PROFILE_STEP_KEYS) + 1)
    ax.boxplot([cum[:, j] for j in range(len(PROFILE_STEP_KEYS))], positions=positions, showmeans=True)
    ax.set_xticks(positions)
    ax.set_xticklabels([f"≤{k}" for k in PROFILE_STEP_KEYS], rotation=35, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Cumulative time share")
    ax.set_title("Cumulative time share by pipeline stage (boxplots)")
    ax.axhline(1.0, color="gray", ls=":", lw=1)
    fig.tight_layout()
    fig.savefig(out_dir / "time_share_stacked_boxplot.png", dpi=200)
    plt.close(fig)

    totals = [float(r["total"]) for r in rows]
    totals_sorted = sorted(totals)
    med_total = totals_sorted[len(totals_sorted) // 2]

    med_share = {}
    mean_share = {}
    for k in PROFILE_STEP_KEYS:
        s = [float(r[k]) / float(r["total"]) for r in rows]
        s_sorted = sorted(s)
        med_share[k] = s_sorted[len(s_sorted) // 2]
        mean_share[k] = float(np.mean(s))

    bottleneck = max(PROFILE_STEP_KEYS, key=lambda kk: med_share[kk])
    ds_med = med_share.get("design_search", 0.0)
    ds_dominates = ds_med >= max(med_share[k] for k in PROFILE_STEP_KEYS if k != "design_search")

    summary_path = out_dir / "profile_summary.md"
    lines = [
        "# UD-DML efficiency profile summary",
        "",
        f"- **Scenario:** {scenario}, **n:** {n}, **r_total:** {r_total}, **replications:** {replications}",
        f"- **Median total runtime (s):** {med_total:.4f}",
        "",
        "## Median / mean time share by stage",
        "",
        "| Stage | Median share | Mean share |",
        "|---|---:|---:|",
    ]
    for k in PROFILE_STEP_KEYS:
        lines.append(f"| {k} | {med_share[k]:.4f} | {mean_share[k]:.4f} |")
    lines.extend(
        [
            "",
            f"- **Largest median share (bottleneck):** `{bottleneck}`",
            f"- **design_search dominates (median share ≥ every other stage):** {ds_dominates}",
            "",
            "## Interpretation",
            "",
            "The breakdown shows where wall time concentrates across Monte Carlo replications. "
            "Stages with the highest median shares drive throughput; if `design_search` is largest, "
            "the budgeted GLP / mixture-discrepancy loop dominates, whereas a large `dml` share "
            "indicates nuisance cross-fitting is the main cost on this configuration.",
        ]
    )
    summary_path.write_text("\n".join(lines), encoding="utf-8")

    n_tex = f"{int(n):,}".replace(",", r"\,")
    rep_word = "replications" if replications != 1 else "replication"
    cap = (
        r"Share of total wall-clock time by pipeline stage for UD-DML on "
        f"{scenario} ($n={n_tex}$, $r_{{\\mathrm{{total}}}}={int(r_total)}$, "
        f"$B_\\gamma={int(B_gamma)}$, {int(replications)} Monte Carlo {rep_word}). "
        f"Median total elapsed time: ${med_total:.4f}$~s. "
        r"For each stage, \emph{Median} and \emph{Mean} are the median and mean, "
        r"over replications, of that stage's fraction of total runtime (in percent)."
    )
    tex_lines = [
        "% Preamble (main LaTeX document): \\usepackage{booktabs}",
        r"\begin{table}[htbp]",
        r"\centering",
        "\\caption{" + cap + "}",
        r"\label{tab:ud_dml_efficiency_profile}",
        r"\begin{tabular}{@{}lrr@{}}",
        r"\toprule",
        r"Stage & Median (\%) & Mean (\%) \\",
        r"\midrule",
    ]
    for k in PROFILE_STEP_KEYS:
        lab = PROFILE_STEP_LABELS_TEX.get(k, k.replace("_", r"\_"))
        pct_med = 100.0 * float(med_share[k])
        pct_mean = 100.0 * float(mean_share[k])
        tex_lines.append(f"{lab} & {pct_med:.2f} & {pct_mean:.2f} \\\\")
    tex_lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )
    (out_dir / "efficiency_profile_table.tex").write_text(
        "\n".join(tex_lines), encoding="utf-8"
    )

    return out_dir


def run_bgamma_sensitivity_experiment(
    scenario: str = "OBS-3",
    n: int = 100_000,
    replications: int = 50,
    r_total: int = 1_000,
    bgamma_grid: Optional[List[int]] = None,
    output_root: Optional[Path] = None,
) -> Path:
    """Monte Carlo sweep over generator budget ``B_gamma`` for UD-DML (single scenario).

    Saves per-replication CSV, an aggregated summary, a LaTeX table, and a two-panel figure.
    """
    scenarios, _, _ = config.get_experiments()
    if scenario not in scenarios:
        raise ValueError(f"Unknown scenario {scenario!r}.")
    scenario_cfg = scenarios[scenario]
    grid = list(bgamma_grid or [10, 20, 30, 40, 60])
    out_dir = output_root or (ANALYSIS_RESULTS_ROOT / "bgamma_sensitivity")
    out_dir.mkdir(parents=True, exist_ok=True)

    detail_header = [
        "scenario",
        "n",
        "replication",
        "B_gamma",
        "est_ate",
        "ci_lower",
        "ci_upper",
        "runtime",
        "subsample_size",
        "subsample_unique",
        "true_ate",
    ]
    detail_rows = []

    for B_gamma in grid:
        for rep in tqdm(
            range(replications),
            desc=f"B_gamma={B_gamma}",
            **_SIM_TQDM_KWARGS,
        ):
            sim_seed = int(config.BASE_SEED) + int(rep) * 10_007 + int(B_gamma)
            np.random.seed(sim_seed)
            data_params = dict(scenario_cfg["params"])
            data_params["n"] = int(n)
            data = scenario_cfg["data_gen_func"](**data_params)
            true_ate = float(data["true_ate"])
            out = methods.run_ud(
                data["X"],
                data["W"],
                data["Y_obs"],
                data["pi_true"],
                scenario_cfg["design"] == "rct",
                {"r_total": int(r_total)},
                k_folds=config.K_FOLDS,
                sim_seed=sim_seed,
                B_gamma=int(B_gamma),
                learner=config.DEFAULT_NUISANCE_LEARNER,
            )
            req = {"est_ate", "ci_lower", "ci_upper", "runtime", "subsample_size", "subsample_unique"}
            _require_columns(out, req, "run_ud output")
            detail_rows.append(
                {
                    "scenario": scenario,
                    "n": int(n),
                    "replication": rep,
                    "B_gamma": int(B_gamma),
                    "est_ate": out["est_ate"],
                    "ci_lower": out["ci_lower"],
                    "ci_upper": out["ci_upper"],
                    "runtime": out["runtime"],
                    "subsample_size": out["subsample_size"],
                    "subsample_unique": out["subsample_unique"],
                    "true_ate": true_ate,
                }
            )

    detail_path = out_dir / "bgamma_sensitivity.csv"
    with detail_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=detail_header)
        w.writeheader()
        w.writerows(detail_rows)

    summary_rows = []
    for B_gamma in grid:
        sub = [r for r in detail_rows if int(r["B_gamma"]) == int(B_gamma)]
        ests = np.array([float(r["est_ate"]) for r in sub], dtype=np.float64)
        truths = np.array([float(r["true_ate"]) for r in sub], dtype=np.float64)
        runtimes = np.array([float(r["runtime"]) for r in sub], dtype=np.float64)
        widths = np.array(
            [float(r["ci_upper"]) - float(r["ci_lower"]) for r in sub],
            dtype=np.float64,
        )
        covered = []
        for r in sub:
            covered.append(
                float(r["ci_lower"]) <= float(r["true_ate"]) <= float(r["ci_upper"])
            )
        rmse = float(math.sqrt(np.mean((ests - truths) ** 2)))
        summary_rows.append(
            {
                "B_gamma": int(B_gamma),
                "mean_runtime": float(np.mean(runtimes)),
                "median_runtime": float(np.median(runtimes)),
                "mean_est_ate": float(np.mean(ests)),
                "sd_est_ate": float(np.std(ests, ddof=1)) if len(ests) > 1 else 0.0,
                "rmse": rmse,
                "mean_ci_width": float(np.mean(widths)),
                "coverage_95": float(np.mean(covered)),
            }
        )

    sum_header = [
        "B_gamma",
        "mean_runtime",
        "median_runtime",
        "mean_est_ate",
        "sd_est_ate",
        "rmse",
        "mean_ci_width",
        "coverage_95",
    ]
    sum_path = out_dir / "bgamma_sensitivity_summary.csv"
    with sum_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=sum_header)
        w.writeheader()
        w.writerows(summary_rows)

    cap_body = (
        f"UD-DML sensitivity to the generator budget $B_\\gamma$ on {scenario} "
        f"($n={int(n)}$)."
    )
    tex_lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        "\\caption{" + cap_body + "}",
        r"\label{tab:bgamma_sensitivity}",
        r"\begin{tabular}{@{}lrrrrr@{}}",
        r"\toprule",
        r"$B_\gamma$ & Mean Runtime & Median Runtime & RMSE & Mean CI Width & Coverage \\",
        r"\midrule",
    ]
    for sr in summary_rows:
        tex_lines.append(
            f"{int(sr['B_gamma'])} & {sr['mean_runtime']:.3f} & {sr['median_runtime']:.3f} & "
            f"{sr['rmse']:.4f} & {sr['mean_ci_width']:.4f} & {sr['coverage_95']:.2f} \\\\"
        )
    tex_lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    (out_dir / "bgamma_sensitivity_table.tex").write_text("\n".join(tex_lines), encoding="utf-8")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    xs = [int(s["B_gamma"]) for s in summary_rows]
    ax1.plot(xs, [s["mean_runtime"] for s in summary_rows],
             "o-", linewidth=2.4, markersize=8, label="Mean")
    ax1.plot(xs, [s["median_runtime"] for s in summary_rows],
             "s--", linewidth=2.4, markersize=8, label="Median")
    ax1.set_xlabel(r"$B_\gamma$")
    ax1.set_ylabel("Runtime (s)")
    ax1.set_title("Runtime vs generator budget")
    ax1.legend(loc="best")
    ax1.grid(True, ls="--")

    ax2.plot(xs, [s["rmse"] for s in summary_rows],
             "o-", color="C1", linewidth=2.4, markersize=8, label="RMSE")
    ax3 = ax2.twinx()
    ax3.plot(xs, [s["coverage_95"] for s in summary_rows],
             "s--", color="C2", linewidth=2.4, markersize=8, label="Coverage")
    ax2.set_xlabel(r"$B_\gamma$")
    ax2.set_ylabel("RMSE", color="C1")
    ax3.set_ylabel("Coverage", color="C2")
    ax2.set_title(r"Precision metrics vs $B_\gamma$")
    ax2.tick_params(axis="y", labelcolor="C1")
    ax3.tick_params(axis="y", labelcolor="C2")
    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = ax3.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, loc="best")
    ax2.grid(True, ls="--")
    fig.suptitle(f"{scenario}, n={n}, r={r_total}")
    fig.savefig(out_dir / "bgamma_sensitivity_plot.png")
    plt.close(fig)

    return out_dir


def _print_banner(title: str, lines: Optional[List[str]] = None) -> None:
    """Structured section header, uniform across experiments."""
    print(f"\n{'='*80}\n{title}\n{'='*80}", flush=True)
    if lines:
        for ln in lines:
            if ln:
                print(f"  {ln}", flush=True)


def run_experiment(exp_name, n_jobs=-1):
    """Run one experiment as defined in config."""
    _print_banner(
        exp_name.upper(),
        [
            f"UD C backend: {'ENABLED (genUD.dll)' if methods.ud_c_backend_active() else 'DISABLED (pure Python fallback)'}",
        ],
    )

    scenarios, all_methods, experiments = config.get_experiments()
    exp_config = experiments[exp_name]
    output_dir = Path(exp_config["base_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    results_file = output_dir / "raw_results.pkl.gz"

    variants = _generate_variant_blueprints(exp_name, exp_config)
    all_tasks = []
    for scenario_name in exp_config["scenarios"]:
        for method_name in exp_config["methods"]:
            for variant in variants:
                allowed_methods = variant.get("method_whitelist")
                if allowed_methods and method_name not in allowed_methods:
                    continue
                variant_for_method = variant.copy()
                variant_for_method.pop("method_whitelist", None)
                n_replications = variant.get("n_replications", config.DEFAULT_REPLICATIONS)
                for sim_id in range(n_replications):
                    all_tasks.append(
                        (
                            exp_name,
                            scenario_name,
                            method_name,
                            sim_id,
                            variant_for_method,
                            checkpoint_dir,
                        )
                    )

    print(f"\nTotal task count: {len(all_tasks)}")

    existing_results = []
    pending_tasks = []
    for task in all_tasks:
        cached = _try_load_checkpoint_result(task)
        if cached is not None:
            existing_results.append(cached)
        else:
            pending_tasks.append(task)

    if existing_results:
        print(f"-> Reusing {len(existing_results)} cached replications")
    if pending_tasks:
        print(f"  Pending simulations: {len(pending_tasks)}")
    else:
        print("  No pending simulations detected.")

    pending_tasks.sort(key=_task_sort_key)

    new_results = []

    if pending_tasks:
        if n_jobs == 1:
            print("Running sequentially...", flush=True)
            with tqdm(
                total=len(pending_tasks),
                desc="Simulations",
                **_SIM_TQDM_KWARGS,
            ) as progress_bar:
                for task in pending_tasks:
                    outcome = run_single_replication(task)
                    progress_bar.update(1)
                    _update_tqdm_postfix(progress_bar, outcome)
                    if outcome is not None:
                        new_results.append(outcome)
                    # For huge-n replications, release per-task allocations now
                    # rather than letting them accumulate until the next GC cycle.
                    if _task_population(task) >= 300_000:
                        gc.collect()
        else:
            from multiprocessing import cpu_count

            raw_jobs = cpu_count() if n_jobs == -1 else n_jobs
            population_cap = MAX_PARALLEL_JOBS
            max_population = max(_task_population(task) for task in pending_tasks)
            # Memory-aware worker cap, tuned for the 32 vCPU EPYC 9654 / 60 GB server.
            # Per-worker budget: n × p × 8 B raw + LGBM internal ~0.5 GB → ~1-2 GB/worker.
            #
            #   n ≥ 1e6 : ≤16 workers  (≥2 GB/worker, stays under 40 GB driver + workers)
            #   n ≥ 5e5 : ≤24 workers  (60 GB / 24 ≈ 2.5 GB each — safe)
            #   n ≥ 3e5 : ≤32 workers  (full-server parallelism)
            #   else    : MAX_PARALLEL_JOBS (32 by default, override via OS_DML_MAX_JOBS)
            if max_population >= 1_000_000:
                population_cap = min(population_cap, 16)
            elif max_population >= 500_000:
                population_cap = min(population_cap, 24)
            elif max_population >= 300_000:
                population_cap = min(population_cap, 32)
            capped_jobs = max(1, min(raw_jobs, population_cap, len(pending_tasks)))
            pre_dispatch = _parallel_pre_dispatch(max_population)
            # Rough disk footprint of existing checkpoints for this experiment.
            try:
                ckpt_bytes = sum(
                    p.stat().st_size
                    for p in checkpoint_dir.rglob("*.pkl*")
                    if p.is_file()
                )
            except OSError:
                ckpt_bytes = 0
            _print_banner(
                "Parallel execution plan",
                [
                    f"workers           : {capped_jobs} "
                    f"(cap={population_cap}, raw={raw_jobs}, max_jobs={MAX_PARALLEL_JOBS})",
                    f"pre-dispatch      : {pre_dispatch!r} (streamed, bounded queue)",
                    f"population (max)  : {max_population:,}",
                    f"pending tasks     : {len(pending_tasks):,}",
                    f"checkpoint dir    : {checkpoint_dir}  ({ckpt_bytes / (1024 ** 3):.2f} GB on disk)",
                    f"UD disk cache     : {os.environ.get('UD_SKELETON_DISK_CACHE', 'default ./ud_skeleton_cache')}",
                ],
            )

            # One executor, streaming consumption:
            #   * generator_unordered  -> start collecting results as they finish
            #   * batch_size=1         -> tqdm updates per-replication (no stall)
            #   * pre_dispatch scaled  -> bounded in-flight tasks => bounded RAM
            with tqdm(
                total=len(pending_tasks),
                desc=f"Simulations ({exp_name})",
                **_SIM_TQDM_KWARGS,
            ) as progress_bar:
                with _tqdm_joblib(progress_bar):
                    with Parallel(
                        n_jobs=capped_jobs,
                        verbose=0,
                        batch_size=1,
                        pre_dispatch=pre_dispatch,
                        return_as="generator_unordered",
                    ) as parallel:
                        stream = parallel(
                            delayed(run_single_replication)(task)
                            for task in pending_tasks
                        )
                        for res in stream:
                            if res is not None:
                                new_results.append(res)
                                _update_tqdm_postfix(progress_bar, res)
                        if max_population >= 300_000:
                            gc.collect()

    combined_results = {}

    def _register(res):
        if res is None:
            return
        key = (
            res.get("exp_name"),
            res.get("scenario"),
            res.get("method"),
            res.get("sim_id"),
            res.get("variant_label"),
        )
        combined_results[key] = res

    for res in existing_results:
        _register(res)
    for res in new_results:
        _register(res)

    results = list(combined_results.values())

    print(f"\n-> Completed {len(results)} successful replications (including cached)")
    legacy_results = results_file.with_suffix("")
    if legacy_results.exists():
        try:
            legacy_results.unlink()
        except Exception:
            pass
    with gzip.open(results_file, "wb") as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"-> Saved raw results to: {results_file}")

    report_info = evaluation.generate_reports(exp_name, results, output_dir)
    if report_info.get("tables_dir"):
        print(f"Tables written to: {report_info['tables_dir']}")
    if report_info.get("figures_dir"):
        print(f"Figures written to: {report_info['figures_dir']}")

    return results


def run_all(experiments=None, n_jobs=-1, fast_demo=False):
    """Run all (or selected) config experiment families.

    Pre-main ``bgamma_sensitivity`` / ``efficiency_profile`` are disabled; see
    ``run_profiling_before_experiment_families``. Then runs the main experiment loop.
    """
    global FAST_DEMO_MODE
    FAST_DEMO_MODE = fast_demo
    _, _, experiment_catalog = config.get_experiments()
    if experiments is None or len(experiments) == 0:
        experiments = list(experiment_catalog.keys())

    start = time.time()
    # run_profiling_before_experiment_families()
    _ensure_ud_skeleton_cache_dir()

    for exp_name in experiments:
        if exp_name not in experiment_catalog:
            print(f"Warning: '{exp_name}' not found in configuration. Skipping.")
            continue
        run_experiment(exp_name, n_jobs=n_jobs)

    elapsed = time.time() - start
    print(f"\nAll requested experiments completed in {elapsed/3600:.2f} hours.")


# =============================================================================
# CLI entry point
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="UD-DML simulation runner for redesigned experiments."
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        choices=("efficiency_profile", "bgamma_sensitivity"),
        help="Run a standalone add-on experiment (efficiency profile or B_gamma sensitivity).",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="OBS-3",
        help="Scenario key for --experiment runs (default: OBS-3).",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=100_000,
        help="Population size n for --experiment runs (default: 100000).",
    )
    parser.add_argument(
        "--replications",
        type=int,
        default=50,
        help="Monte Carlo replications for --experiment runs (default: 50; e.g. bgamma_sensitivity).",
    )
    parser.add_argument(
        "--r-total",
        type=int,
        default=1_000,
        dest="r_total",
        help="UD / UNIF subsample budget r_total for --experiment runs (default: 1000).",
    )
    parser.add_argument(
        "--experiments",
        nargs="*",
        metavar="NAME",
        help="Optional list of experiment keys to run (defaults to all).",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=-1,
        help="Number of parallel jobs (default: -1 for all available cores).",
    )
    parser.add_argument(
        "--fast-demo",
        action="store_true",
        help=(
            "Enable fast-demo mode (reduced grids/replications, dedicated output folder). "
            "If pre-main profiling is re-enabled, it uses FAST_DEMO_OVERRIDES replications."
        ),
    )
    parser.add_argument(
        "--no-ud-disk-cache",
        action="store_true",
        help="Disable on-disk UD GLP skeleton cache (sets env UD_SKELETON_DISK_CACHE=0).",
    )
    parser.add_argument(
        "--ud-disk-cache-dir",
        type=str,
        default=None,
        metavar="DIR",
        help="Directory for UD skeleton .npy cache (sets env UD_SKELETON_DISK_CACHE).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.no_ud_disk_cache:
        os.environ["UD_SKELETON_DISK_CACHE"] = "0"
    elif args.ud_disk_cache_dir:
        os.environ["UD_SKELETON_DISK_CACHE"] = str(
            Path(args.ud_disk_cache_dir).expanduser().resolve()
        )
    if args.experiment == "efficiency_profile":
        out = run_efficiency_profile_experiment(
            scenario=args.scenario,
            n=args.n,
            replications=args.replications,
            r_total=args.r_total,
        )
        print(f"efficiency_profile outputs -> {out.resolve()}")
        return
    if args.experiment == "bgamma_sensitivity":
        out = run_bgamma_sensitivity_experiment(
            scenario=args.scenario,
            n=args.n,
            replications=args.replications,
            r_total=args.r_total,
        )
        print(f"bgamma_sensitivity outputs -> {out.resolve()}")
        return

    run_all(experiments=args.experiments, n_jobs=args.jobs, fast_demo=args.fast_demo)


if __name__ == "__main__":
    main()
