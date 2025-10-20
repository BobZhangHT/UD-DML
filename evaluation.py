# -*- coding: utf-8 -*-
"""
evaluation.py

Redesigned analysis utilities for the OS-DML simulation suite.
Responsible for:
    - Converting replication outputs to tidy data frames
    - Aggregating metrics required in the redesigned experiments
    - Writing non-overlapping tables
    - Generating complementary figures
    - Persisting empirical pilot-ratio optima for downstream experiments
All content is in English.
"""

import json
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config

plt.switch_backend("Agg")


# =============================================================================
# Core data wrangling
# =============================================================================

METRIC_COLUMNS = ["Bias", "RMSE", "CI_Coverage", "CI_Width", "Runtime"]


def _ensure_directories(base_dir: Path):
    tables_dir = base_dir / "tables"
    figures_dir = base_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    return tables_dir, figures_dir


def prepare_dataframe(results: List[Dict]) -> pd.DataFrame:
    """Convert raw simulation outputs to a tidy DataFrame."""
    if not results:
        return pd.DataFrame()

    records = []
    for res in results:
        ci_lower = res.get("ci_lower")
        ci_upper = res.get("ci_upper")
        est_ate = res.get("est_ate")
        true_ate = res.get("true_ate")

        if est_ate is None or true_ate is None:
            continue

        record = {
            "exp_name": res.get("exp_name"),
            "scenario": res.get("scenario"),
            "method": res.get("method"),
            "sim_id": res.get("sim_id"),
            "pilot_ratio": res.get("pilot_ratio"),
            "pilot_ratio_strategy": res.get("pilot_ratio_strategy"),
            "r0": res.get("r0"),
            "r1": res.get("r1"),
            "r_total": res.get("r_total"),
            "population_size": res.get("population_size"),
            "delta": res.get("delta"),
            "variant_label": res.get("variant_label"),
            "n_estimators": res.get("n_estimators"),
            "k_folds": res.get("k_folds"),
            "misspecification": res.get("misspecification"),
            "runtime": res.get("runtime"),
            "est_ate": est_ate,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "true_ate": true_ate,
        }

        record["Bias"] = est_ate - true_ate
        record["Sq_Error"] = record["Bias"] ** 2
        record["RMSE_rep"] = abs(record["Bias"])  # used for distributional plots
        if ci_lower is not None and ci_upper is not None:
            record["CI_Coverage"] = float(ci_lower <= true_ate <= ci_upper)
            record["CI_Width"] = ci_upper - ci_lower
        else:
            record["CI_Coverage"] = np.nan
            record["CI_Width"] = np.nan
        records.append(record)

    return pd.DataFrame(records)


def _aggregate_metrics(df: pd.DataFrame, group_cols: Iterable[str]) -> pd.DataFrame:
    grouped = df.groupby(list(group_cols), dropna=False)
    summary = grouped.agg(
        Bias=("Bias", "mean"),
        RMSE=("Sq_Error", lambda x: math.sqrt(x.mean()) if len(x) else np.nan),
        CI_Coverage=("CI_Coverage", "mean"),
        CI_Width=("CI_Width", "mean"),
        Runtime=("runtime", "mean"),
        Replications=("sim_id", "count"),
    ).reset_index()
    return summary


# =============================================================================
# Experiment-specific reporting
# =============================================================================

def _pilot_ratio_reports(df: pd.DataFrame, analysis_dir: Path, params: Dict):
    tables_dir, figures_dir = _ensure_directories(analysis_dir)

    coverage_threshold = params.get("coverage_threshold", 0.93)
    summary = _aggregate_metrics(
        df,
        ["scenario", "method", "pilot_ratio"],
    ).sort_values(["scenario", "method", "pilot_ratio"])
    summary.to_csv(tables_dir / "pilot_ratio_summary.csv", index=False)

    optima = []
    for scenario in summary["scenario"].unique():
        subset = summary[(summary["scenario"] == scenario) & (summary["method"] == "OS")]
        feasible = subset[subset["CI_Coverage"] >= coverage_threshold]
        candidate = feasible.sort_values("RMSE").head(1)
        if candidate.empty:
            candidate = subset.sort_values("RMSE").head(1)
        if not candidate.empty:
            row = candidate.iloc[0]
            optima.append({"scenario": scenario, "pilot_ratio": float(row["pilot_ratio"])})

    optima_df = pd.DataFrame(optima)
    optima_df.to_csv(tables_dir / "pilot_ratio_optima.csv", index=False, float_format="%.2f")
    optima_map = {row["scenario"]: row["pilot_ratio"] for _, row in optima_df.iterrows()}
    if optima_map:
        empirical_file = tables_dir / "pilot_ratio_optima.json"
        with open(empirical_file, "w") as f:
            json.dump(optima_map, f, indent=2)

    # Plots
    for metric, ylabel in [("RMSE", "RMSE"), ("CI_Coverage", "Coverage")]:
        plt.figure(figsize=(12, 6))
        for scenario in sorted(df["scenario"].unique()):
            subset = summary[(summary["scenario"] == scenario) & (summary["method"] == "OS")]
            plt.plot(
                subset["pilot_ratio"],
                subset[metric],
                marker="o",
                label=scenario,
            )
        plt.xlabel("Pilot Ratio")
        plt.ylabel(ylabel)
        plt.title(f"Pilot Ratio vs {ylabel}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(figures_dir / f"pilot_ratio_{metric.lower()}.png", dpi=200)
        plt.close()


def _subsample_budget_reports(df: pd.DataFrame, analysis_dir: Path):
    tables_dir, figures_dir = _ensure_directories(analysis_dir)
    summary = _aggregate_metrics(df, ["scenario", "method", "r_total"])

    # Baseline comparison (r_total == 1000 if available)
    baseline = summary[summary["r_total"] == 1000][["scenario", "method", "RMSE", "CI_Coverage"]]
    baseline = baseline.rename(
        columns={"RMSE": "RMSE_baseline", "CI_Coverage": "Coverage_baseline"}
    )
    merged = summary.merge(baseline, on=["scenario", "method"], how="left")
    merged["RMSE_pct_change"] = 100 * (
        (merged["RMSE"] / merged["RMSE_baseline"]) - 1.0
    )
    merged["Coverage_diff"] = merged["CI_Coverage"] - merged["Coverage_baseline"]
    merged["Runtime_per_draw"] = merged["Runtime"] / merged["r_total"].replace(0, np.nan)
    merged.to_csv(tables_dir / "subsample_budget_summary.csv", index=False)

    # Pareto scatter
    plt.figure(figsize=(10, 6))
    for method, marker in zip(["OS", "UNIF", "LSS", "FULL"], ["o", "s", "^", "D"]):
        subset = merged[merged["method"] == method]
        plt.scatter(
            subset["Runtime"],
            subset["RMSE"],
            label=method,
            marker=marker,
        )
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Runtime (s)")
    plt.ylabel("RMSE")
    plt.title("Runtime vs RMSE across subsample budgets")
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "subsample_budget_runtime_rmse.png", dpi=200)
    plt.close()


def _population_scaling_reports(df: pd.DataFrame, analysis_dir: Path):
    tables_dir, figures_dir = _ensure_directories(analysis_dir)
    summary = _aggregate_metrics(df, ["scenario", "method", "population_size"])
    summary.to_csv(tables_dir / "population_scaling_by_method.csv", index=False)

    avg_summary = _aggregate_metrics(df, ["scenario", "population_size"])
    avg_summary = avg_summary.rename(columns={"Bias": "Bias_avg", "RMSE": "RMSE_avg"})
    avg_summary.to_csv(tables_dir / "population_scaling_mean_over_methods.csv", index=False)

    # Deltas relative to baseline population
    baseline_size = 100_000
    baseline = summary[summary["population_size"] == baseline_size][
        ["scenario", "method", "RMSE", "Runtime"]
    ].rename(columns={"RMSE": "RMSE_baseline", "Runtime": "Runtime_baseline"})
    merged = summary.merge(baseline, on=["scenario", "method"], how="left")
    merged["RMSE_log_ratio"] = np.log(merged["RMSE"] / merged["RMSE_baseline"])
    merged["Runtime_log_ratio"] = np.log(merged["Runtime"] / merged["Runtime_baseline"])
    merged.to_csv(tables_dir / "population_scaling_with_deltas.csv", index=False)

    # Plots: RMSE vs N, Runtime vs N
    for method in sorted(summary["method"].unique()):
        subset = summary[summary["method"] == method]
        plt.figure(figsize=(10, 6))
        for scenario in sorted(subset["scenario"].unique()):
            scen_df = subset[subset["scenario"] == scenario].sort_values("population_size")
            plt.plot(
                scen_df["population_size"],
                scen_df["RMSE"],
                marker="o",
                label=scenario,
            )
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Population size (N)")
        plt.ylabel("RMSE")
        plt.title(f"RMSE scaling with N ({method})")
        plt.grid(True, which="both", ls="--", alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(figures_dir / f"population_scaling_rmse_{method}.png", dpi=200)
        plt.close()

    plt.figure(figsize=(10, 6))
    for scenario in sorted(summary["scenario"].unique()):
        scen_df = summary[summary["scenario"] == scenario].sort_values("population_size")
        runtime_mean = scen_df.groupby("population_size")["Runtime"].mean().reset_index()
        plt.plot(
            runtime_mean["population_size"],
            runtime_mean["Runtime"],
            marker="o",
            label=scenario,
        )
    plt.xscale("log")
    plt.xlabel("Population size (N)")
    plt.ylabel("Runtime (s)")
    plt.title("Runtime scaling with N (averaged across methods)")
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "population_scaling_runtime.png", dpi=200)
    plt.close()


def _double_robustness_reports(df: pd.DataFrame, analysis_dir: Path):
    tables_dir, figures_dir = _ensure_directories(analysis_dir)
    summary = _aggregate_metrics(df, ["scenario", "misspecification", "method"])
    summary.to_csv(tables_dir / "double_robustness_summary.csv", index=False)

    # Pivot into Outcome/Propensity grid
    def _split_misspec(code):
        if code == "correct_correct":
            return "Correct", "Correct"
        if code == "correct_wrong":
            return "Correct", "Wrong"
        if code == "wrong_correct":
            return "Wrong", "Correct"
        return "Wrong", "Wrong"

    summary["Outcome"], summary["Propensity"] = zip(
        *summary["misspecification"].map(_split_misspec)
    )
    formatted = summary[
        ["scenario", "Outcome", "Propensity", "method", "Bias", "RMSE", "CI_Coverage", "CI_Width", "Runtime"]
    ]
    formatted.to_csv(tables_dir / "double_robustness_formatted.csv", index=False)

    ww = summary[summary["misspecification"] == "wrong_wrong"]
    ww = ww.sort_values("CI_Coverage")
    ww[["scenario", "method", "CI_Coverage"]].to_csv(
        tables_dir / "double_robustness_coverage_shortfall.csv", index=False
    )

    # Heatmap-style plot using pivot tables
    for scenario in sorted(summary["scenario"].unique()):
        scen_df = summary[summary["scenario"] == scenario]
        pivot = scen_df.pivot_table(
            index="Outcome", columns="Propensity", values="CI_Coverage", aggfunc="mean"
        )
        plt.figure(figsize=(5, 4))
        plt.imshow(pivot.values, vmin=0, vmax=1, cmap="viridis")
        plt.colorbar(label="Coverage")
        plt.xticks(range(len(pivot.columns)), pivot.columns)
        plt.yticks(range(len(pivot.index)), pivot.index)
        plt.title(f"Coverage heatmap ({scenario})")
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                plt.text(
                    j,
                    i,
                    f"{pivot.values[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color="white" if pivot.values[i, j] < 0.5 else "black",
                )
        plt.tight_layout()
        plt.savefig(figures_dir / f"double_robustness_coverage_{scenario}.png", dpi=200)
        plt.close()

    # Bias distribution violin plots (combined)
    plt.figure(figsize=(12, 6))
    order = ["correct_correct", "correct_wrong", "wrong_correct", "wrong_wrong"]
    df["misspecification"] = pd.Categorical(df["misspecification"], categories=order, ordered=True)
    bias_data = df[df["method"] == "OS"]
    bias_summary = bias_data.groupby(["scenario", "misspecification"])["Bias"].apply(list)
    positions = np.arange(len(bias_summary))
    plt.violinplot(
        bias_summary.tolist(),
        positions=positions,
        showmeans=True,
    )
    labels = [f"{sc}\n{miss}" for sc, miss in bias_summary.index]
    plt.xticks(positions, labels, rotation=45, ha="right")
    plt.ylabel("Bias distribution")
    plt.title("Bias distribution across misspecification scenarios (OS)")
    plt.tight_layout()
    plt.savefig(figures_dir / "double_robustness_bias_violin.png", dpi=200)
    plt.close()


# =============================================================================
# Public entry point
# =============================================================================

def generate_reports(exp_name: str, results: List[Dict], output_dir: Path) -> Dict[str, Path]:
    """
    Produce experiment-specific tables and plots.

    Returns
    -------
    dict
        Keys: 'tables_dir', 'figures_dir'
    """
    analysis_dir = Path("./analysis_results") / exp_name
    tables_dir, figures_dir = _ensure_directories(analysis_dir)

    df = prepare_dataframe(results)
    if df.empty:
        print(f"No valid results to analyse for {exp_name}.")
        return {"tables_dir": tables_dir, "figures_dir": figures_dir}

    df.to_csv(analysis_dir / "replication_level_metrics.csv", index=False)

    if exp_name == "experiment_pilot_ratio_sweep":
        # Deduplicate method to OS only
        df = df[df["method"] == "OS"]
        _pilot_ratio_reports(df, analysis_dir, config.get_experiments()[2][exp_name]["params"])
    elif exp_name == "experiment_subsample_budget":
        _subsample_budget_reports(df, analysis_dir)
    elif exp_name == "experiment_population_scaling":
        _population_scaling_reports(df, analysis_dir)
    elif exp_name == "experiment_double_robustness":
        _double_robustness_reports(df, analysis_dir)
    else:
        summary = _aggregate_metrics(df, ["scenario", "method"])
        summary.to_csv(tables_dir / "summary_table.csv", index=False)

    return {"tables_dir": tables_dir, "figures_dir": figures_dir}
