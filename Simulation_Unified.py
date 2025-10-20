#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simulation_Unified.py

Redesigned simulation driver for the OS-DML study.
This script orchestrates four experiment families:

1. Pilot-ratio sweep (OS only)
2. Subsample budget comparison (all methods)
3. Population scaling (all methods)
4. Double robustness (OS under misspecification grid)

The driver coordinates checkpointed Monte Carlo replications,
stores raw outputs, and delegates aggregation/visualisation to
`evaluation.generate_reports`.
"""

import argparse
import json
import pickle
import time
import traceback
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from tqdm.auto import tqdm

import config
import evaluation

# =============================================================================
# Constants & paths
# =============================================================================

EMPIRICAL_PILOT_OPTIMA_FILE = Path(
    "./analysis_results/experiment_pilot_ratio_sweep/tables/pilot_ratio_optima.json"
)


# =============================================================================
# Helper utilities
# =============================================================================

def _sanitize_token(token):
    """Make tokens filesystem-safe."""
    token_str = str(token)
    return token_str.replace(" ", "_").replace(".", "p").replace("-", "m")


def _format_ratio_token(ratio):
    if ratio is None or np.isnan(ratio):
        return None
    return f"p{int(round(ratio * 100)):03d}"


def _split_sample_sizes(r_total, pilot_ratio):
    if r_total is None or r_total <= 1:
        return 0, 0
    pilot_ratio = np.clip(pilot_ratio, 0.01, 0.99)
    r0 = max(1, int(round(r_total * pilot_ratio)))
    r0 = min(r0, r_total - 1)
    r1 = r_total - r0
    return r0, r1


def _load_empirical_pilot_optima():
    if not EMPIRICAL_PILOT_OPTIMA_FILE.exists():
        return {}
    try:
        with open(EMPIRICAL_PILOT_OPTIMA_FILE, "r") as f:
            data = json.load(f)
        return {k: float(v) for k, v in data.items()}
    except Exception:
        return {}


def _clear_empirical_pilot_optima_cache():
    pass


def _resolve_pilot_ratio(variant, scenario_name):
    strategy = variant.get("pilot_ratio_strategy", "explicit")
    if strategy == "empirical":
        optima = _load_empirical_pilot_optima()
        if scenario_name in optima:
            return optima[scenario_name]
    ratio = variant.get("pilot_ratio")
    return config.DEFAULT_PILOT_RATIO if ratio is None else ratio


def _prepare_sampling_config(method_name, variant, scenario_name):
    r_total = variant.get("r_total")
    pilot_ratio = None
    if method_name == "OS":
        pilot_ratio = _resolve_pilot_ratio(variant, scenario_name)
        if r_total is None:
            raise ValueError("OS requires 'r_total' in variant configuration.")
        r0, r1 = _split_sample_sizes(r_total, pilot_ratio)
    elif method_name in ("UNIF", "LSS"):
        if r_total is None:
            raise ValueError(f"{method_name} requires 'r_total' in variant configuration.")
        pilot_ratio = None
        r0, r1 = 0, r_total
    else:  # FULL method
        pilot_ratio = None
        if r_total is None:
            r_total = variant.get("population_size", config.N_POPULATION)
        r0 = r1 = 0
    return {
        "pilot_ratio": pilot_ratio,
        "r0": r0,
        "r1": r1,
        "r_total": r_total,
    }


def _compose_variant_label(method_name, variant, sampling_cfg):
    parts = [variant.get("label", "baseline")]
    ratio_token = _format_ratio_token(sampling_cfg["pilot_ratio"])
    if method_name == "OS" and ratio_token:
        parts.append(ratio_token)
    if variant.get("misspecification"):
        parts.append(variant["misspecification"])
    tokens = [_sanitize_token(part) for part in parts if part]
    return "_".join(tokens)


def _build_checkpoint_path(checkpoint_dir, scenario_name, method_name, sim_id, variant, sampling_cfg):
    label = _compose_variant_label(method_name, variant, sampling_cfg)
    filename = f"sim_{sim_id:04d}_{method_name}_{label}.pkl"
    return checkpoint_dir / scenario_name / filename


def _generate_variant_blueprints(exp_name, exp_config):
    """Expand experiment parameter grids into variant blueprints."""
    params = exp_config["params"]
    base = {
        "delta": params.get("delta", config.DELTA),
        "n_estimators": params.get("n_estimators", config.LGBM_N_ESTIMATORS),
        "k_folds": params.get("k_folds", config.K_FOLDS),
        "population_size": params.get("population_size", config.N_POPULATION),
        "pilot_ratio": params.get("pilot_ratio", config.DEFAULT_PILOT_RATIO),
        "pilot_ratio_strategy": params.get("pilot_ratio_strategy", "explicit"),
        "r_total": params.get("r_total"),
        "n_replications": params.get("n_replications", config.DEFAULT_REPLICATIONS),
    }

    variants = []

    if exp_name == "experiment_pilot_ratio_sweep":
        for pilot_ratio in params.get("pilot_ratios", []):
            variant = base.copy()
            variant.update(
                {
                    "label": f"pilot-{int(round(pilot_ratio * 100)):03d}",
                    "pilot_ratio": float(pilot_ratio),
                    "pilot_ratio_strategy": "explicit",
                    "r_total": params["r_total"],
                }
            )
            variants.append(variant)
    elif exp_name == "experiment_subsample_budget":
        for r_total in params.get("r_totals", []):
            variant = base.copy()
            variant.update(
                {
                    "label": f"r-{int(r_total)}",
                    "r_total": int(r_total),
                    "pilot_ratio": None,
                }
            )
            variants.append(variant)
    elif exp_name == "experiment_population_scaling":
        threshold = params.get(
            "large_population_threshold", config.LARGE_POPULATION_THRESHOLD
        )
        large_reps = params.get(
            "n_replications_large", config.LARGE_POPULATION_REPLICATIONS
        )
        base_reps = params.get("n_replications", config.DEFAULT_REPLICATIONS)
        for population in params.get("population_sizes", []):
            variant = base.copy()
            variant.update(
                {
                    "label": f"N-{int(population)}",
                    "population_size": int(population),
                    "r_total": params.get("r_total"),
                    "n_replications": large_reps if population >= threshold else base_reps,
                    "pilot_ratio": None,
                }
            )
            variants.append(variant)
    elif exp_name == "experiment_double_robustness":
        for misspec in params.get("misspecification_scenarios", []):
            variant = base.copy()
            variant.update(
                {
                    "label": f"misspec-{misspec}",
                    "misspecification": misspec,
                    "r_total": params["r_total"],
                    "pilot_ratio": None,
                }
            )
            variants.append(variant)
    else:
        variant = base.copy()
        variant["label"] = "default"
        variants.append(variant)

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
            with open(checkpoint_file, "rb") as f:
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

        method_kwargs = {
            "k_folds": variant.get("k_folds", config.K_FOLDS),
            "n_estimators": variant.get("n_estimators", config.LGBM_N_ESTIMATORS),
            "delta": variant.get("delta", config.DELTA),
        }
        if variant.get("misspecification"):
            method_kwargs["misspecification"] = variant["misspecification"]

        if method_name == "OS":
            method_kwargs["r"] = {"r0": sampling_cfg["r0"], "r1": sampling_cfg["r1"]}
            method_kwargs["pilot_ratio"] = sampling_cfg["pilot_ratio"]
        elif method_name in ("UNIF", "LSS"):
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
            "pilot_ratio": sampling_cfg["pilot_ratio"],
            "pilot_ratio_strategy": variant.get("pilot_ratio_strategy", "explicit"),
            "r0": sampling_cfg["r0"],
            "r1": sampling_cfg["r1"],
            "r_total": sampling_cfg["r_total"],
            "population_size": data_params.get("n", config.N_POPULATION),
            "delta": variant.get("delta", config.DELTA),
            "variant_label": _compose_variant_label(method_name, variant, sampling_cfg),
            "n_estimators": variant.get("n_estimators", config.LGBM_N_ESTIMATORS),
            "k_folds": variant.get("k_folds", config.K_FOLDS),
        }
        if variant.get("misspecification"):
            metadata["misspecification"] = variant["misspecification"]

        result.update(metadata)

        checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        with open(checkpoint_file, "wb") as f:
            pickle.dump(result, f)

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
    if not checkpoint_file.exists():
        return False
    try:
        with open(checkpoint_file, "rb") as f:
            result = pickle.load(f)
        return "est_ate" in result and "scenario" in result
    except Exception:
        return False


def run_experiment(exp_name, n_jobs=-1):
    """Run one experiment as defined in config."""
    print(f"\n{'='*80}\n{exp_name.upper()}\n{'='*80}")

    scenarios, all_methods, experiments = config.get_experiments()
    exp_config = experiments[exp_name]
    output_dir = Path(exp_config["base_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    results_file = output_dir / "raw_results.pkl"

    variants = _generate_variant_blueprints(exp_name, exp_config)
    all_tasks = []
    for scenario_name in exp_config["scenarios"]:
        for method_name in exp_config["methods"]:
            for variant in variants:
                n_replications = variant.get("n_replications", config.DEFAULT_REPLICATIONS)
                for sim_id in range(n_replications):
                    all_tasks.append(
                        (
                            exp_name,
                            scenario_name,
                            method_name,
                            sim_id,
                            variant.copy(),
                            checkpoint_dir,
                        )
                    )

    print(f"\nTotal task count: {len(all_tasks)}")
    existing = sum(1 for task in all_tasks if _checkpoint_exists(task))
    if existing:
        print(f"✓ Reusing {existing} cached replications")
        print(f"  Pending simulations: {len(all_tasks) - existing}")

    if n_jobs == 1:
        print("Running sequentially...")
        results = []
        for task in tqdm(all_tasks, desc="Simulations"):
            outcome = run_single_replication(task)
            if outcome is not None:
                results.append(outcome)
    else:
        from multiprocessing import cpu_count

        actual_jobs = cpu_count() if n_jobs == -1 else n_jobs
        print(f"Running with {actual_jobs} parallel jobs...")
        results = Parallel(n_jobs=n_jobs, verbose=1)(
            delayed(run_single_replication)(task) for task in all_tasks
        )
        results = [res for res in results if res is not None]

    print(f"\n✓ Completed {len(results)} successful replications")
    with open(results_file, "wb") as f:
        pickle.dump(results, f)
    print(f"✓ Saved raw results to: {results_file}")

    report_info = evaluation.generate_reports(exp_name, results, output_dir)
    if report_info.get("tables_dir"):
        print(f"Tables written to: {report_info['tables_dir']}")
    if report_info.get("figures_dir"):
        print(f"Figures written to: {report_info['figures_dir']}")

    if exp_name == "experiment_pilot_ratio_sweep":
        _clear_empirical_pilot_optima_cache()

    return results


def run_all(experiments=None, n_jobs=-1):
    """Run all (or selected) experiments."""
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
        description="OS-DML simulation runner for redesigned experiments."
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
    return parser.parse_args()


def main():
    args = parse_args()
    run_all(experiments=args.experiments, n_jobs=args.jobs)


if __name__ == "__main__":
    main()
