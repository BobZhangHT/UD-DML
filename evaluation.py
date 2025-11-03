# -*- coding: utf-8 -*-
"""
evaluation.py

Analysis utilities for the UD-DML simulation suite (visualisation, subsample-size,
population-scaling, double-robustness, and nuisance-sensitivity experiments).
All content is in English.
"""
import math
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

import config

plt.switch_backend("Agg")

# =============================================================================
# Constants
# =============================================================================

METRIC_COLUMNS = ["Bias", "RMSE", "CI_Coverage", "CI_Width", "Runtime"]
SCENARIO_ORDER = ["RCT-1", "RCT-2", "RCT-3", "OBS-1", "OBS-2", "OBS-3"]
SCENARIO_COLORS = {
    "RCT-1": "#d62728",
    "RCT-2": "#9467bd",
    "RCT-3": "#8c564b",
    "OBS-1": "#1f77b4",
    "OBS-2": "#ff7f0e",
    "OBS-3": "#2ca02c",
}
SCENARIO_MARKERS = {
    "RCT-1": "D",
    "RCT-2": "P",
    "RCT-3": "X",
    "OBS-1": "o",
    "OBS-2": "s",
    "OBS-3": "^",
}
METHOD_COLORS = {
    "UD": "#1f77b4",
    "UNIF": "#ff7f0e",
}
DEFAULT_FIGURE_DPI = 400
BASE_FONT_SIZE = 12
plt.rcParams.update(
    {
        "axes.titlesize": BASE_FONT_SIZE + 2,
        "axes.labelsize": BASE_FONT_SIZE,
        "xtick.labelsize": BASE_FONT_SIZE,
        "ytick.labelsize": BASE_FONT_SIZE,
        "legend.fontsize": BASE_FONT_SIZE,
    }
)

# =============================================================================
# Helper utilities
# =============================================================================


def _set_scenario_category(series: pd.Series) -> pd.Series:
    return pd.Categorical(series, categories=SCENARIO_ORDER, ordered=True)


def _ensure_directories(base_dir: Path):
    tables_dir = base_dir / "tables"
    figures_dir = base_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    return tables_dir, figures_dir


def _save_figure_multi_format(fig, path_stem: Path, dpi: int = DEFAULT_FIGURE_DPI):
    path_stem = Path(path_stem)
    path_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(path_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _aggregate_metrics(df: pd.DataFrame, group_cols: Iterable[str]) -> pd.DataFrame:
    grouped = df.groupby(list(group_cols), dropna=False, observed=False)
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
# Data preparation
# =============================================================================


def prepare_dataframe(results: List[Dict]) -> pd.DataFrame:
    if not results:
        return pd.DataFrame()

    records = []
    for res in results:
        est_ate = res.get("est_ate")
        true_ate = res.get("true_ate")
        if est_ate is None or true_ate is None:
            continue
        ci_lower = res.get("ci_lower")
        ci_upper = res.get("ci_upper")

        record = {
            "exp_name": res.get("exp_name"),
            "scenario": res.get("scenario"),
            "method": res.get("method"),
            "sim_id": res.get("sim_id"),
            "r_total": res.get("r_total"),
            "population_size": res.get("population_size"),
            "learner": res.get("learner"),
            "subsample_size": res.get("subsample_size"),
            "subsample_unique": res.get("subsample_unique"),
            "runtime": res.get("runtime"),
            "misspecification": res.get("misspecification"),
            "est_ate": est_ate,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "true_ate": true_ate,
            "subsample_projection": res.get("subsample_projection"),
            "covariates": res.get("covariates"),
            "projection_dims": res.get("projection_dims"),
        }

        record["Bias"] = est_ate - true_ate
        record["Sq_Error"] = record["Bias"] ** 2
        record["RMSE_rep"] = abs(record["Bias"])
        if ci_lower is not None and ci_upper is not None:
            record["CI_Coverage"] = float(ci_lower <= true_ate <= ci_upper)
            record["CI_Width"] = ci_upper - ci_lower
        else:
            record["CI_Coverage"] = np.nan
            record["CI_Width"] = np.nan
        records.append(record)

    df = pd.DataFrame(records)
    if not df.empty:
        df["scenario"] = _set_scenario_category(df["scenario"])
    return df

# =============================================================================
# Experiment-specific reporting
# =============================================================================


def _visualization_reports(raw_results: List[Dict], analysis_dir: Path):
    _, figures_dir = _ensure_directories(analysis_dir)
    cov_to_full: Dict[str, np.ndarray] = {}
    cov_to_samples: Dict[str, Dict[str, np.ndarray]] = {}
    cov_to_dims: Dict[str, tuple] = {}
    obs_propensity: Dict[str, np.ndarray] = {}
    obs_treatment: Dict[str, np.ndarray] = {}

    for res in raw_results:
        method = res.get("method")
        if method not in METHOD_COLORS:
            continue
        cov_key = res.get("covariates", "x1")
        full_proj = res.get("full_projection")
        if full_proj is not None and cov_key not in cov_to_full:
            cov_to_full[cov_key] = np.asarray(full_proj)
        subs_proj = res.get("subsample_projection")
        if subs_proj is not None:
            cov_to_samples.setdefault(cov_key, {})[method] = np.asarray(subs_proj)
        dims = res.get("projection_dims")
        if dims is not None and cov_key not in cov_to_dims:
            cov_to_dims[cov_key] = tuple(int(d) for d in dims)
        scenario_name = res.get("scenario")
        prop_full = res.get("propensity_full")
        treat_full = res.get("treatment_full")
        if (
            scenario_name
            and scenario_name.startswith("OBS")
            and scenario_name not in obs_propensity
            and prop_full is not None
            and treat_full is not None
        ):
            obs_propensity[scenario_name] = np.asarray(prop_full)
            obs_treatment[scenario_name] = np.asarray(treat_full)

    cov_order = ["x1", "x2", "x3"]
    dgp_labels = {"x1": "DGP-X1", "x2": "DGP-X2", "x3": "DGP-X3"}
    axis_labels = {"x1": ("X1", "X2"), "x2": ("X1", "X6"), "x3": ("X1", "X2")}
    marker_map = {"UD": "*", "UNIF": "^"}

    fig, axes = plt.subplots(
        len(cov_order), len(METHOD_COLORS), figsize=(12, 12), sharex=False, sharey=False
    )
    if len(cov_order) == 1:
        axes = np.expand_dims(axes, axis=0)

    for row, cov_key in enumerate(cov_order):
        full = cov_to_full.get(cov_key)
        samples_dict = cov_to_samples.get(cov_key, {})
        has_any = full is not None or any(samples_dict.get(m) is not None for m in METHOD_COLORS)
        if not has_any:
            for col in range(len(METHOD_COLORS)):
                axes[row, col].set_visible(False)
            continue
        x_label, y_label = axis_labels.get(cov_key, ("X1", "X2"))
        for col, (method, color) in enumerate(METHOD_COLORS.items()):
            ax = axes[row, col]
            if full is not None and full.size:
                ax.scatter(
                    full[:, 0],
                    full[:, 1],
                    s=4,
                    alpha=0.12,
                    color="#999999",
                    edgecolors="none",
                    rasterized=True,
                    label="Full data" if (row == 0 and col == 0) else None,
                )
            subs = samples_dict.get(method)
            if subs is not None and subs.size:
                ax.scatter(
                    subs[:, 0],
                    subs[:, 1],
                    s=30,
                    alpha=0.9,
                    color=color,
                    marker=marker_map.get(method, "o"),
                    edgecolors="black",
                    linewidths=0.4,
                    rasterized=True,
                )
            ax.set_title(f"{dgp_labels.get(cov_key, cov_key)}: {method}")
            ax.set_xlabel(x_label)
            if col == 0:
                ax.set_ylabel(y_label)
            ax.grid(True, ls="--", alpha=0.3)

    fig.suptitle("UD vs UNIF Subsamples (paired UD)", fontsize=BASE_FONT_SIZE + 4, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    _save_figure_multi_format(fig, figures_dir / "visualization_subsamples")

    obs_order = [sc for sc in ["OBS-1", "OBS-2", "OBS-3"] if sc in obs_propensity]
    if obs_order:
        fig, axes = plt.subplots(1, len(obs_order), figsize=(6 * len(obs_order), 4), sharey=True)
        if len(obs_order) == 1:
            axes = [axes]
        colors = {"Treatment": "#d62728", "Control": "#1f77b4"}
        bins = np.linspace(0.0, 1.0, 60)
        for ax, scenario in zip(axes, obs_order):
            prop = obs_propensity[scenario]
            treat = obs_treatment[scenario].astype(bool)
            ax.hist(
                prop[treat],
                bins=bins,
                density=True,
                alpha=0.6,
                color=colors["Treatment"],
                label="Treatment",
            )
            ax.hist(
                prop[~treat],
                bins=bins,
                density=True,
                alpha=0.6,
                color=colors["Control"],
                label="Control",
            )
            ax.set_title(f"{scenario}")
            ax.set_xlabel("Propensity score")
            ax.grid(True, ls="--", alpha=0.3)
        axes[0].set_ylabel("Density")
        handles = [
            Line2D([0], [0], color=colors["Treatment"], lw=6, label="Treatment"),
            Line2D([0], [0], color=colors["Control"], lw=6, label="Control"),
        ]
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.05), ncol=2, frameon=False)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        _save_figure_multi_format(fig, figures_dir / "visualization_propensity_density")
def _subsample_size_reports(df: pd.DataFrame, analysis_dir: Path):
    tables_dir, figures_dir = _ensure_directories(analysis_dir)
    summary = _aggregate_metrics(df, ["scenario", "method", "r_total"])
    summary["scenario"] = _set_scenario_category(summary["scenario"])
    summary.sort_values(["scenario", "method", "r_total"], inplace=True)
    out = summary.copy()
    out["scenario"] = out["scenario"].astype(str)
    out.to_csv(tables_dir / "subsample_size_summary.csv", index=False)

    metrics = [("RMSE", "RMSE"), ("CI_Coverage", "CI Coverage"), ("CI_Width", "CI Width")]
    scenarios_present = [sc for sc in SCENARIO_ORDER if not summary[summary["scenario"] == sc].empty]
    if not scenarios_present:
        return
    fig, axes = plt.subplots(len(scenarios_present), len(metrics), figsize=(14, 4 * len(scenarios_present)), sharex="col")
    if len(scenarios_present) == 1:
        axes = np.expand_dims(axes, axis=0)
    for row, scenario in enumerate(scenarios_present):
        scen_df = summary[summary["scenario"] == scenario]
        for col, (metric, title) in enumerate(metrics):
            ax = axes[row, col]
            for method, color in METHOD_COLORS.items():
                method_df = scen_df[scen_df["method"] == method].sort_values("r_total")
                if method_df.empty:
                    continue
                ax.plot(
                    method_df["r_total"],
                    method_df[metric],
                    marker="o",
                    color=color,
                    label=method if row == 0 else None,
                )
            ax.set_xscale("log")
            ax.set_xlabel("r_total")
            if col == 0:
                ax.set_ylabel(scenario)
            ax.set_title(title)
            ax.grid(True, ls="--", alpha=0.3)
            if metric == "CI_Coverage":
                ax.axhline(0.95, ls=":", color="gray", alpha=0.7)
    handles = [Line2D([0], [0], color=color, marker="o", label=method) for method, color in METHOD_COLORS.items()]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.98), ncol=len(METHOD_COLORS), frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save_figure_multi_format(fig, figures_dir / "subsample_size_metrics")


def _population_size_reports(df: pd.DataFrame, analysis_dir: Path):
    tables_dir, figures_dir = _ensure_directories(analysis_dir)
    summary = _aggregate_metrics(df, ["scenario", "method", "population_size", "r_total"])
    if summary.empty:
        return
    summary["scenario"] = _set_scenario_category(summary["scenario"])
    summary.sort_values(["scenario", "method", "population_size", "r_total"], inplace=True)
    out = summary.copy()
    out["scenario"] = out["scenario"].astype(str)
    out.to_csv(tables_dir / "population_size_summary.csv", index=False)

    metrics = [("RMSE", "RMSE"), ("CI_Coverage", "CI Coverage"), ("CI_Width", "CI Width"), ("Runtime", "Runtime (s)")]
    r_totals = sorted(summary["r_total"].dropna().unique())
    populations = sorted(summary["population_size"].dropna().unique())
    scenarios_present = [sc for sc in SCENARIO_ORDER if not summary[summary["scenario"] == sc].empty]
    for r_total in r_totals:
        fig, axes = plt.subplots(len(scenarios_present), len(metrics), figsize=(16, 4 * len(scenarios_present)), sharex=True)
        if len(scenarios_present) == 1:
            axes = np.expand_dims(axes, axis=0)
        for row, scenario in enumerate(scenarios_present):
            scen_df = summary[(summary["scenario"] == scenario) & (summary["r_total"] == r_total)]
            for col, (metric, title) in enumerate(metrics):
                ax = axes[row, col]
                for method, color in METHOD_COLORS.items():
                    method_df = scen_df[scen_df["method"] == method].sort_values("population_size")
                    if method_df.empty:
                        continue
                    ax.plot(
                        method_df["population_size"],
                        method_df[metric],
                        marker="o",
                        color=color,
                        label=method if row == 0 else None,
                    )
                ax.set_xscale("log")
                ax.set_xlabel("Population size (N)")
                if col == 0:
                    ax.set_ylabel(scenario)
                ax.set_title(f"{title}")
                ax.grid(True, ls="--", alpha=0.3)
                if metric == "CI_Coverage":
                    ax.axhline(0.95, ls=":", color="gray", alpha=0.7)
        handles = [Line2D([0], [0], color=color, marker="o", label=method) for method, color in METHOD_COLORS.items()]
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.98), ncol=len(METHOD_COLORS), frameon=False)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        _save_figure_multi_format(fig, figures_dir / f"population_metrics_r{int(r_total)}")


def _double_robust_reports(df: pd.DataFrame, analysis_dir: Path):
    tables_dir, figures_dir = _ensure_directories(analysis_dir)
    summary = _aggregate_metrics(df, ["scenario", "misspecification", "method"])
    if summary.empty:
        return
    summary["scenario"] = _set_scenario_category(summary["scenario"])
    summary.sort_values(["scenario", "misspecification", "method"], inplace=True)
    out = summary.copy()
    out["scenario"] = out["scenario"].astype(str)
    out.to_csv(tables_dir / "double_robustness_summary.csv", index=False)

    coverage_pivot = summary.pivot_table(
        index=["scenario", "misspecification"],
        columns="method",
        values="CI_Coverage",
    )
    fig, axes = plt.subplots(len(SCENARIO_ORDER), 1, figsize=(8, 3 * len(SCENARIO_ORDER)), sharex=True)
    if len(SCENARIO_ORDER) == 1:
        axes = [axes]
    for ax, scenario in zip(axes, SCENARIO_ORDER):
        subset = summary[summary["scenario"] == scenario]
        if subset.empty:
            ax.set_visible(False)
            continue
        ms_levels = list(subset["misspecification"].unique())
        x = np.arange(len(ms_levels))
        width = 0.35
        for idx, (method, color) in enumerate(METHOD_COLORS.items()):
            method_vals = []
            for misspec in ms_levels:
                entry = subset[(subset["misspecification"] == misspec) & (subset["method"] == method)]
                method_vals.append(entry["CI_Coverage"].iloc[0] if not entry.empty else np.nan)
            ax.bar(x + (idx - 0.5) * width, method_vals, width=width, color=color, alpha=0.85, label=method if scenario == SCENARIO_ORDER[0] else None)
        ax.axhline(0.95, ls=":", color="gray", alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(ms_levels, rotation=15)
        ax.set_ylabel("Coverage")
        ax.set_title(scenario)
        ax.grid(True, axis="y", ls="--", alpha=0.3)
    fig.legend(loc="upper center", bbox_to_anchor=(0.5, 0.98), ncol=len(METHOD_COLORS), frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save_figure_multi_format(fig, figures_dir / "double_robust_coverage")


def _nuisance_sensitivity_reports(df: pd.DataFrame, analysis_dir: Path):
    tables_dir, figures_dir = _ensure_directories(analysis_dir)
    summary = _aggregate_metrics(df, ["scenario", "method", "learner", "r_total"])
    if summary.empty:
        return
    summary["scenario"] = _set_scenario_category(summary["scenario"])
    summary.sort_values(["scenario", "learner", "method", "r_total"], inplace=True)
    out = summary.copy()
    out["scenario"] = out["scenario"].astype(str)
    out.to_csv(tables_dir / "nuisance_sensitivity_summary.csv", index=False)

    metrics = [("RMSE", "RMSE"), ("CI_Coverage", "CI Coverage"), ("CI_Width", "CI Width")]
    scenarios_present = [sc for sc in SCENARIO_ORDER if not summary[summary["scenario"] == sc].empty]
    learners = sorted(summary["learner"].dropna().unique())
    for scenario in scenarios_present:
        scen_df = summary[summary["scenario"] == scenario]
        fig, axes = plt.subplots(len(metrics), 1, figsize=(8, 4 * len(metrics)), sharex=True)
        for ax, (metric, title) in zip(axes, metrics):
            for learner in learners:
                learner_df = scen_df[scen_df["learner"] == learner]
                for method, color in METHOD_COLORS.items():
                    method_df = learner_df[learner_df["method"] == method].sort_values("r_total")
                    if method_df.empty:
                        continue
                    ax.plot(
                        method_df["r_total"],
                        method_df[metric],
                        marker="o",
                        color=color,
                        linestyle="-" if method == "UD" else "--",
                        label=f"{method}-{learner}" if title == "RMSE" else None,
                    )
            ax.set_xscale("log")
            ax.set_xlabel("r_total")
            ax.set_title(title)
            ax.grid(True, ls="--", alpha=0.3)
            if metric == "CI_Coverage":
                ax.axhline(0.95, ls=":", color="gray", alpha=0.7)
        handles = [
            Line2D([0], [0], color=METHOD_COLORS[m], linestyle="-" if m == "UD" else "--", marker="o", label=f"{m}-{learner}")
            for learner in learners for m in METHOD_COLORS
        ]
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.98), ncol=max(1, len(learners) * len(METHOD_COLORS)), frameon=False)
        fig.suptitle(f"Learner sensitivity ({scenario})", fontsize=BASE_FONT_SIZE + 4)
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        _save_figure_multi_format(fig, figures_dir / f"nuisance_sensitivity_{scenario}")

# =============================================================================
# Public entry point
# =============================================================================


def generate_reports(exp_name: str, results: List[Dict], output_dir: Path) -> Dict[str, Path]:
    analysis_dir = Path("./analysis_results") / exp_name
    tables_dir, figures_dir = _ensure_directories(analysis_dir)

    df = prepare_dataframe(results)
    if exp_name == "experiment_visualization":
        _visualization_reports(results, analysis_dir)
    elif exp_name == "experiment_subsample_size":
        _subsample_size_reports(df, analysis_dir)
    elif exp_name == "experiment_population_size":
        _population_size_reports(df, analysis_dir)
    elif exp_name == "experiment_double_robust":
        _double_robust_reports(df, analysis_dir)
    elif exp_name == "experiment_nuisance_sensitivity":
        _nuisance_sensitivity_reports(df, analysis_dir)
    else:
        # Fallback: export raw dataframe if no dedicated report exists.
        if not df.empty:
            df.to_csv(tables_dir / "raw_results.csv", index=False)

    return {"tables_dir": tables_dir, "figures_dir": figures_dir}
