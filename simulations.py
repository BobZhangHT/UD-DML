#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
simulations.py

Unified simulation driver for the UD-DML study.
This script orchestrates five experiment families described in UD_DML.pdf:

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
import pickle
import time
import traceback
import gzip
import os
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from tqdm.auto import tqdm

import config
import evaluation

# =============================================================================
# Constants & paths
# =============================================================================

FAST_DEMO_MODE = False
FAST_DEMO_OVERRIDES = {
    "n_replications": 10,
}
MAX_PARALLEL_JOBS = config.MAX_PARALLEL_JOBS
ENV_MAX_JOBS = os.environ.get("OS_DML_MAX_JOBS")
if ENV_MAX_JOBS:
    try:
        MAX_PARALLEL_JOBS = max(1, min(int(ENV_MAX_JOBS), MAX_PARALLEL_JOBS))
    except ValueError:
        pass


# =============================================================================
# Helper utilities
# =============================================================================

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

def run_single_replication(task):
    """Execute one Monte Carlo replication with checkpoint support."""
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
        scenarios, all_methods, _ = config.get_experiments()
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


def _checkpoint_exists(task):
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
            return False
    try:
        with opener(checkpoint_file, "rb") as f:
            result = pickle.load(f)
        return "est_ate" in result and "scenario" in result
    except Exception:
        return False


def _load_checkpoint_result(task):
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
            return pickle.load(f)
    except Exception:
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


def run_experiment(exp_name, n_jobs=-1):
    """Run one experiment as defined in config."""
    print(f"\n{'='*80}\n{exp_name.upper()}\n{'='*80}")

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
        if _checkpoint_exists(task):
            cached = _load_checkpoint_result(task)
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

    pending_tasks.sort(
        key=lambda task: (
            _task_population(task),
            task[1],  # scenario
            task[3],  # sim_id
            task[2],  # method
        )
    )

    new_results = []

    if pending_tasks:
        if n_jobs == 1:
            print("Running sequentially...")
            for task in tqdm(pending_tasks, desc="Simulations"):
                outcome = run_single_replication(task)
                if outcome is not None:
                    new_results.append(outcome)
        else:
            from multiprocessing import cpu_count

            raw_jobs = cpu_count() if n_jobs == -1 else n_jobs
            population_cap = MAX_PARALLEL_JOBS
            max_population = max(_task_population(task) for task in pending_tasks)
            if max_population >= 500_000:
                population_cap = min(population_cap, 16)
            elif max_population >= 300_000:
                population_cap = min(population_cap, 32)
            capped_jobs = max(1, min(raw_jobs, population_cap, len(pending_tasks)))
            print(f"Running with {capped_jobs} parallel jobs...")

            chunk_size = capped_jobs
            for start in range(0, len(pending_tasks), chunk_size):
                chunk = pending_tasks[start : start + chunk_size]
                chunk_results = Parallel(
                    n_jobs=capped_jobs,
                    verbose=1,
                    pre_dispatch=capped_jobs,
                    batch_size=1,
                )(delayed(run_single_replication)(task) for task in chunk)
                new_results.extend(res for res in chunk_results if res is not None)
                del chunk_results

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
    """Run all (or selected) experiments."""
    global FAST_DEMO_MODE
    FAST_DEMO_MODE = fast_demo
    _, _, experiment_catalog = config.get_experiments()
    if experiments is None or len(experiments) == 0:
        experiments = list(experiment_catalog.keys())

    start = time.time()
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
        help="Enable fast-demo mode (reduced grids/replications, dedicated output folder).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_all(experiments=args.experiments, n_jobs=args.jobs, fast_demo=args.fast_demo)


if __name__ == "__main__":
    main()
