# -*- coding: utf-8 -*-
"""
evaluation.py

Analysis utilities for the UD-DML simulation suite (visualisation, subsample-size,
population-scaling, double-robustness, and nuisance-sensitivity experiments).
All content is in English.
"""
import colorsys
import math
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib import colors as mcolors

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
    "FULL": "#2ca02c",
}
METHOD_DEFAULT_MARKERS = {
    "UD": "o",
    "UNIF": "s",
    "FULL": "D",
}
METHOD_LEARNER_MARKERS = {
    "UD": {"lasso_cv": "o", "rf": "^", "lgbm": "v"},
    "UNIF": {"lasso_cv": "s", "rf": "D", "lgbm": "P"},
    "FULL": {"lasso_cv": "X", "rf": "X", "lgbm": "X"},
}
LEARNER_LIGHTNESS_SHIFT = {"lasso_cv": -0.08, "rf": 0.0, "lgbm": 0.08}
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


def _aggregate_metrics_with_sd(df: pd.DataFrame, group_cols: Iterable[str]) -> pd.DataFrame:
    """Aggregate metrics with standard deviations for error bars."""
    grouped = df.groupby(list(group_cols), dropna=False, observed=False)
    
    # Compute both means and standard deviations
    summary = grouped.agg(
        Bias=("Bias", "mean"),
        Bias_SD=("Bias", "std"),
        RMSE=("Sq_Error", lambda x: math.sqrt(x.mean()) if len(x) else np.nan),
        RMSE_SD=("RMSE_rep", "std"),
        CI_Coverage=("CI_Coverage", "mean"),
        CI_Coverage_SD=("CI_Coverage", "std"),
        CI_Width=("CI_Width", "mean"),
        CI_Width_SD=("CI_Width", "std"),
        Runtime=("runtime", "mean"),
        Runtime_SD=("runtime", "std"),
        Replications=("sim_id", "count"),
    ).reset_index()
    return summary


def _shift_lightness(hex_color: str, shift: float) -> str:
    rgb = mcolors.to_rgb(hex_color)
    h, l, s = colorsys.rgb_to_hls(*rgb)
    l = max(0.0, min(1.0, l + shift))
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return mcolors.to_hex((r, g, b))


def _get_method_color(method: str, learner: str | None = None) -> str:
    base = METHOD_COLORS.get(method, "#444444")
    if learner and method in METHOD_LEARNER_MARKERS:
        shift = LEARNER_LIGHTNESS_SHIFT.get(learner, 0.0)
        if abs(shift) > 1e-9:
            try:
                return _shift_lightness(base, shift)
            except ValueError:
                return base
    return base


def _get_method_marker(method: str, learner: str | None = None) -> str:
    if learner and method in METHOD_LEARNER_MARKERS:
        marker = METHOD_LEARNER_MARKERS[method].get(learner)
        if marker:
            return marker
    return METHOD_DEFAULT_MARKERS.get(method, "o")


def _format_learner_name(learner: str) -> str:
    """Format learner name for display (uppercase)."""
    learner_display = {
        "lasso_cv": "LASSO_CV",
        "lgbm": "LGBM",
        "rf": "RF",
    }
    return learner_display.get(learner, learner.upper())


def _methods_present(df: pd.DataFrame, allowed: Iterable[str] | None = None) -> List[str]:
    present = []
    if df.empty or "method" not in df.columns:
        return present
    unique_methods = set(df["method"].dropna().unique())
    if allowed is not None:
        unique_methods &= set(allowed)
    for method in METHOD_COLORS.keys():
        if method in unique_methods:
            present.append(method)
    return present

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
    axis_labels = {"x1": ("X1", "X2"), "x2": ("X1", "X6"), "x3": ("X1", "X6")}
    method_order = [m for m in ("UD", "UNIF") if m in METHOD_COLORS]

    fig, axes = plt.subplots(
        len(cov_order), len(method_order), figsize=(12, 12), sharex=False, sharey=False
    )
    if len(cov_order) == 1:
        axes = np.expand_dims(axes, axis=0)

    for row, cov_key in enumerate(cov_order):
        full = cov_to_full.get(cov_key)
        samples_dict = cov_to_samples.get(cov_key, {})
        has_any = full is not None or any(samples_dict.get(m) is not None for m in method_order)
        if not has_any:
            for col in range(len(method_order)):
                axes[row, col].set_visible(False)
            continue
        x_label, y_label = axis_labels.get(cov_key, ("X1", "X2"))
        for col, method in enumerate(method_order):
            color = METHOD_COLORS[method]
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
                    marker=_get_method_marker(method),
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
        # Collect subsample propensity scores for UD and UNIF methods
        # Use a single replication (sim_id) for consistency across methods
        ud_propensity: Dict[str, np.ndarray] = {}
        ud_treatment: Dict[str, np.ndarray] = {}
        unif_propensity: Dict[str, np.ndarray] = {}
        unif_treatment: Dict[str, np.ndarray] = {}
        
        # Find a common sim_id for each scenario that exists for both UD and UNIF methods
        scenario_to_sim_id: Dict[str, int] = {}
        for scenario_name in obs_order:
            # Find sim_ids that exist for both UD and UNIF methods
            ud_sim_ids = set()
            unif_sim_ids = set()
            for res in raw_results:
                if res.get("scenario") == scenario_name and res.get("sim_id") is not None:
                    method = res.get("method")
                    sim_id = res.get("sim_id")
                    if method == "UD":
                        ud_sim_ids.add(sim_id)
                    elif method == "UNIF":
                        unif_sim_ids.add(sim_id)
            # Find common sim_id
            common_sim_ids = ud_sim_ids & unif_sim_ids
            if common_sim_ids:
                scenario_to_sim_id[scenario_name] = min(common_sim_ids)  # Use smallest sim_id
        
        for res in raw_results:
            method = res.get("method")
            scenario_name = res.get("scenario")
            sim_id = res.get("sim_id")
            
            if scenario_name not in obs_order:
                continue
            
            # Use the same sim_id for consistency
            if sim_id != scenario_to_sim_id.get(scenario_name):
                continue
            
            prop_full = res.get("propensity_full")
            treat_full = res.get("treatment_full")
            subsample_indices = res.get("subsample_indices")
            
            if prop_full is not None and treat_full is not None and subsample_indices is not None:
                prop_full_arr = np.asarray(prop_full)
                treat_full_arr = np.asarray(treat_full).astype(bool)
                subsample_idx = np.asarray(subsample_indices)
                
                # Extract subsample propensity scores and treatment
                prop_subsample = prop_full_arr[subsample_idx]
                treat_subsample = treat_full_arr[subsample_idx]
                
                if method == "UD":
                    if scenario_name not in ud_propensity:
                        ud_propensity[scenario_name] = prop_subsample
                        ud_treatment[scenario_name] = treat_subsample
                elif method == "UNIF":
                    if scenario_name not in unif_propensity:
                        unif_propensity[scenario_name] = prop_subsample
                        unif_treatment[scenario_name] = treat_subsample
        
        # Create 3x3 subplot grid
        fig, axes = plt.subplots(3, len(obs_order), figsize=(6 * len(obs_order), 12), sharey="row", sharex=True)
        if len(obs_order) == 1:
            axes = np.expand_dims(axes, axis=1)
        
        colors = {"Treatment": "#d62728", "Control": "#1f77b4"}
        bins = np.linspace(0.0, 1.0, 60)
        
        # Row 1: Full data
        for col, scenario in enumerate(obs_order):
            ax = axes[0, col]
            prop = obs_propensity[scenario]
            treat = obs_treatment[scenario].astype(bool)
            ax.hist(
                prop[treat],
                bins=bins,
                density=True,
                alpha=0.6,
                color=colors["Treatment"],
                label="Treatment" if col == 0 else None,
            )
            ax.hist(
                prop[~treat],
                bins=bins,
                density=True,
                alpha=0.6,
                color=colors["Control"],
                label="Control" if col == 0 else None,
            )
            ax.set_title(f"Full Data: {scenario}")
            if col == 0:
                ax.set_ylabel("Density")
            ax.grid(True, ls="--", alpha=0.3)
        
        # Row 2: UD-DML
        for col, scenario in enumerate(obs_order):
            ax = axes[1, col]
            if scenario in ud_propensity:
                prop = ud_propensity[scenario]
                treat = ud_treatment[scenario].astype(bool)
                ax.hist(
                    prop[treat],
                    bins=bins,
                    density=True,
                    alpha=0.6,
                    color=colors["Treatment"],
                )
                ax.hist(
                    prop[~treat],
                    bins=bins,
                    density=True,
                    alpha=0.6,
                    color=colors["Control"],
                )
            ax.set_title(f"UD-DML: {scenario}")
            if col == 0:
                ax.set_ylabel("Density")
            ax.grid(True, ls="--", alpha=0.3)
        
        # Row 3: UNIF-DML
        for col, scenario in enumerate(obs_order):
            ax = axes[2, col]
            if scenario in unif_propensity:
                prop = unif_propensity[scenario]
                treat = unif_treatment[scenario].astype(bool)
                ax.hist(
                    prop[treat],
                    bins=bins,
                    density=True,
                    alpha=0.6,
                    color=colors["Treatment"],
                )
                ax.hist(
                    prop[~treat],
                    bins=bins,
                    density=True,
                    alpha=0.6,
                    color=colors["Control"],
                )
            ax.set_title(f"UNIF-DML: {scenario}")
            ax.set_xlabel("Propensity score")
            if col == 0:
                ax.set_ylabel("Density")
            ax.grid(True, ls="--", alpha=0.3)
        
        # Add legend to the first subplot
        handles = [
            Line2D([0], [0], color=colors["Treatment"], lw=6, label="Treatment"),
            Line2D([0], [0], color=colors["Control"], lw=6, label="Control"),
        ]
        axes[0, 0].legend(handles=handles, loc="upper right", frameon=False)
        fig.tight_layout()
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
    methods_present = _methods_present(summary)
    if not methods_present:
        return
    fig, axes = plt.subplots(len(scenarios_present), len(metrics), figsize=(14, 4 * len(scenarios_present)), sharex="col")
    if len(scenarios_present) == 1:
        axes = np.expand_dims(axes, axis=0)
    for row, scenario in enumerate(scenarios_present):
        scen_df = summary[summary["scenario"] == scenario]
        for col, (metric, title) in enumerate(metrics):
            ax = axes[row, col]
            if metric == "CI_Coverage":
                r_values = sorted(scen_df["r_total"].dropna().unique())
                if not r_values:
                    continue
                x = np.arange(len(r_values))
                bar_width = 0.8 / max(1, len(methods_present))
                for idx, method in enumerate(methods_present):
                    color = METHOD_COLORS[method]
                    method_df = scen_df[scen_df["method"] == method]
                    if method_df.empty:
                        continue
                    heights = []
                    for r_val in r_values:
                        value_series = method_df[method_df["r_total"] == r_val][metric]
                        heights.append(float(value_series.iloc[0]) if not value_series.empty else np.nan)
                    position_shift = (idx - (len(methods_present) - 1) / 2) * bar_width
                    ax.bar(
                        x + position_shift,
                        heights,
                        bar_width,
                        color=color,
                        edgecolor="black",
                        linewidth=0.5,
                        label=method if row == 0 else None,
                    )
                ax.set_xticks(x)
                ax.set_xticklabels([f"{int(r)}" for r in r_values])
                ax.set_xlabel("$r$")
                if col == 0:
                    ax.set_ylabel(scenario)
                ax.set_title(title)
                ax.grid(True, ls="--", alpha=0.3)
                ax.axhline(0.95, ls=":", color="gray", alpha=0.7)
            else:
                for method in methods_present:
                    color = METHOD_COLORS[method]
                    method_df = scen_df[scen_df["method"] == method].sort_values("r_total")
                    if method_df.empty:
                        continue
                    ax.plot(
                        method_df["r_total"],
                        method_df[metric],
                        marker=_get_method_marker(method),
                        color=color,
                        label=method if row == 0 else None,
                    )
                ax.set_xscale("log")
                ax.set_xlabel("$r$")
                if col == 0:
                    ax.set_ylabel(scenario)
                ax.set_title(title)
                ax.grid(True, ls="--", alpha=0.3)
    # Add legend to the first subplot (top-left)
    handles = [
        Line2D([0], [0], color=METHOD_COLORS[method], marker=_get_method_marker(method), label=method)
        for method in methods_present
    ]
    axes[0, 0].legend(
        handles=handles,
        loc="upper right",
        frameon=False,
    )
    fig.tight_layout()
    _save_figure_multi_format(fig, figures_dir / "subsample_size_metrics")


def _population_size_reports(
    df: pd.DataFrame,
    analysis_dir: Path,
    allowed_methods: Iterable[str] | None = None,
    allowed_scenarios: Iterable[str] | None = None,
):
    tables_dir, figures_dir = _ensure_directories(analysis_dir)
    df = df.copy()
    if allowed_methods is not None:
        df = df[df["method"].isin(allowed_methods)]
    if allowed_scenarios is not None:
        allowed_set = {str(s) for s in allowed_scenarios}
        df = df[df["scenario"].astype(str).isin(allowed_set)]

    methods_present_pre = _methods_present(df, allowed_methods)
    if "FULL" in methods_present_pre:
        subsample_r_totals = sorted(
            df[df["method"] != "FULL"]["r_total"].dropna().unique().tolist()
        )
        if subsample_r_totals:
            full_rows = df[df["method"] == "FULL"].copy()
            replicated = []
            for r_total in subsample_r_totals:
                temp = full_rows.copy()
                temp["r_total"] = r_total
                replicated.append(temp)
            if replicated:
                df = pd.concat(
                    [df[df["method"] != "FULL"]] + replicated,
                    ignore_index=True,
                )

    summary = _aggregate_metrics(df, ["scenario", "method", "population_size", "r_total"])
    if summary.empty:
        return
    summary["scenario"] = _set_scenario_category(summary["scenario"])
    summary.sort_values(["scenario", "method", "population_size", "r_total"], inplace=True)
    population_table = summary.copy()
    population_table["scenario"] = population_table["scenario"].astype(str)

    metrics = ["RMSE", "CI_Coverage", "CI_Width", "Runtime"]
    
    # Extract r_totals only from UD/UNIF methods to avoid FULL's population_size values
    non_full_summary = summary[summary["method"].isin(["UD", "UNIF"])]
    if non_full_summary.empty:
        r_totals = sorted(summary["r_total"].dropna().unique())
    else:
        r_totals = sorted(non_full_summary["r_total"].dropna().unique())
    
    populations = sorted(summary["population_size"].dropna().unique())
    scenario_order = allowed_scenarios if allowed_scenarios is not None else SCENARIO_ORDER
    scenarios_present = [sc for sc in scenario_order if not summary[summary["scenario"] == sc].empty]
    methods_present = _methods_present(summary, allowed_methods)
    if not methods_present:
        return

    # Publication table: flatten combinations and compute metric blocks
    # Pre-compute FULL data (should be same for all subsample sizes)
    # Note: FULL method's r_total is set to population_size, so we filter by scenario and population only
    full_data_cache = {}
    for scenario in scenarios_present:
        # Convert scenario to string for comparison with population_table
        scenario_str = str(scenario)
        for population in populations:
            full_slice = population_table[
                (population_table["scenario"] == scenario_str)
                & (population_table["population_size"] == population)
                & (population_table["method"] == "FULL")
            ]
            if not full_slice.empty:
                # Take the first record (FULL data doesn't depend on r_total subsample size)
                # Note: FULL's r_total field contains population_size value, not experiment r_total
                full_record = full_slice.iloc[0]
                full_data_cache[(scenario_str, population)] = {
                    metric: full_record[metric] for metric in metrics
                }
    
    records = []
    for scenario in scenarios_present:
        # Convert scenario to string for consistency
        scenario_str = str(scenario)
        for population in populations:
            for r_total in r_totals:
                row = {
                    "Scenario": scenario_str,
                    "Population": int(population),
                    "Subsample": int(r_total),
                }
                slice_df = population_table[
                    (population_table["scenario"] == scenario_str)
                    & (population_table["population_size"] == population)
                    & (population_table["r_total"] == r_total)
                ]
                
                # Check if there's actual experimental data for this configuration
                # Skip if no UD/UNIF data exists for this r_total
                has_data = False
                for method in methods_present:
                    if method == "FULL":
                        continue  # FULL is handled separately via cache
                    method_data = slice_df[slice_df["method"] == method]
                    if not method_data.empty:
                        # Check if at least one metric has a valid value
                        for metric in metrics:
                            value = method_data[metric].iloc[0] if not method_data.empty else np.nan
                            if pd.notna(value):
                                has_data = True
                                break
                    if has_data:
                        break
                
                # Skip rows with no UD/UNIF experimental data (unrun results)
                if not has_data:
                    continue
                
                # Populate metric values
                for metric in metrics:
                    for method in methods_present:
                        if method == "FULL":
                            # Use cached FULL data (same across all subsample sizes)
                            cache_key = (scenario_str, population)
                            if cache_key in full_data_cache:
                                value = full_data_cache[cache_key][metric]
                            else:
                                value = np.nan
                        else:
                            value_df = slice_df[slice_df["method"] == method][metric]
                            value = float(value_df.iloc[0]) if not value_df.empty else np.nan
                        col_name = f"{metric}_{method}"
                        row[col_name] = value
                records.append(row)

    if records:
        table_df = pd.DataFrame(records).sort_values(
            ["Scenario", "Population", "Subsample"]
        )
        # Truncate digits for readability
        format_map = {col: "{:.3f}" for col in table_df.columns if col.startswith("RMSE")}
        format_map.update({col: "{:.3f}" for col in table_df.columns if col.startswith("CI_Width")})
        format_map.update({col: "{:.2f}" for col in table_df.columns if col.startswith("Runtime")})
        format_map.update({col: "{:.2f}" for col in table_df.columns if col.startswith("CI_Coverage")})

        display_df = table_df.copy()
        for col, fmt in format_map.items():
            if col in display_df:
                display_df[col] = display_df[col].apply(
                    lambda x: fmt.format(x) if pd.notnull(x) else ""
                )
        csv_path = tables_dir / "population_size_publication_table.csv"
        display_df.to_csv(csv_path, index=False)

        # LaTeX table with multi-level header
        method_labels = list(methods_present)
        metric_display = {
            "RMSE": "RMSE",
            "CI_Coverage": "CI Coverage",
            "CI_Width": "CI Width",
            "Runtime": "Runtime",
        }
        metric_headers = []
        for metric in metrics:
            metric_headers.extend([metric_display[metric]] * len(method_labels))
        sub_headers = method_labels * len(metrics)

        latex_lines = []
        latex_lines.append("\\begin{table}[htbp]")
        latex_lines.append("\\centering")
        column_spec = "lll" + "c" * (len(metric_headers))
        latex_lines.append(f"\\begin{{tabular}}{{{column_spec}}}")
        latex_lines.append("\\toprule")
        header_top = ["Scenario", "Population", "Subsample"]
        header_top += [
            f"\\multicolumn{{{len(method_labels)}}}{{c}}{{{metric_display[metric]}}}"
            for metric in metrics
        ]
        latex_lines.append(" & ".join(header_top) + " \\\\")
        # Column separators for metric blocks
        start_col = 4
        for idx in range(len(metrics)):
            left = start_col + idx * len(method_labels)
            right = left + len(method_labels) - 1
            latex_lines.append(f"\\cmidrule(lr){{{left}-{right}}}")
        header_second = ["", "", ""]
        header_second += [method for method in sub_headers]
        latex_lines.append(" & ".join(header_second) + " \\\\")
        latex_lines.append("\\midrule")
        for _, row in display_df.iterrows():
            cells = [
                row["Scenario"],
                f"{int(row['Population']):,}",
                f"{int(row['Subsample']):,}",
            ]
            for metric in ["RMSE", "CI_Coverage", "CI_Width", "Runtime"]:
                for method in method_labels:
                    col = f"{metric}_{method}"
                    cells.append(row.get(col, ""))
            latex_lines.append(" & ".join(cells) + " \\\\")
        latex_lines.append("\\bottomrule")
        latex_lines.append("\\end{tabular}")
        latex_lines.append("\\caption{Population size experiment summary.}")
        latex_lines.append("\\label{tab:population_size}")
        latex_lines.append("\\end{table}")

        tex_path = tables_dir / "population_size_publication_table.tex"
        tex_path.write_text("\n".join(latex_lines), encoding="utf-8")

    for r_total in r_totals:
        fig, axes = plt.subplots(len(scenarios_present), len(metrics), figsize=(16, 4 * len(scenarios_present)), sharex=True)
        if len(scenarios_present) == 1:
            axes = np.expand_dims(axes, axis=0)
        for row, scenario in enumerate(scenarios_present):
            scen_df = summary[(summary["scenario"] == scenario) & (summary["r_total"] == r_total)]
            for col, metric in enumerate(metrics):
                ax = axes[row, col]
                for method in methods_present:
                    color = METHOD_COLORS[method]
                    method_df = scen_df[scen_df["method"] == method].sort_values("population_size")
                    if method_df.empty:
                        continue
                    ax.plot(
                        method_df["population_size"],
                        method_df[metric],
                        marker=_get_method_marker(method),
                        color=color,
                        label=method if row == 0 else None,
                    )
                ax.set_xscale("log")
                ax.set_xlabel("Population size (N)")
                if col == 0:
                    ax.set_ylabel(scenario)
                title = {
                    "RMSE": "RMSE",
                    "CI_Coverage": "CI Coverage",
                    "CI_Width": "CI Width",
                    "Runtime": "Runtime (s)",
                }[metric]
                ax.set_title(title)
                ax.grid(True, ls="--", alpha=0.3)
                if metric == "CI_Coverage":
                    ax.axhline(0.95, ls=":", color="gray", alpha=0.7)
        handles = [
            Line2D(
                [0],
                [0],
                color=METHOD_COLORS[method],
                marker=_get_method_marker(method),
                label=method,
            )
            for method in methods_present
        ]
        fig.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.98),
            ncol=max(1, len(handles)),
            frameon=False,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        _save_figure_multi_format(fig, figures_dir / f"population_metrics_r{int(r_total)}")


def _double_robust_reports(
    df: pd.DataFrame,
    analysis_dir: Path,
    allowed_methods: Iterable[str] | None = None,
    allowed_scenarios: Iterable[str] | None = None,
):
    tables_dir, figures_dir = _ensure_directories(analysis_dir)
    if allowed_methods is not None:
        df = df[df["method"].isin(allowed_methods)]
    if allowed_scenarios is not None:
        allowed_set = {str(s) for s in allowed_scenarios}
        df = df[df["scenario"].astype(str).isin(allowed_set)]
    summary = _aggregate_metrics(df, ["scenario", "misspecification", "method"])
    if summary.empty:
        return
    summary["scenario"] = _set_scenario_category(summary["scenario"])
    summary.sort_values(["scenario", "misspecification", "method"], inplace=True)
    out = summary.copy()
    out["scenario"] = out["scenario"].astype(str)
    out.to_csv(tables_dir / "double_robustness_summary.csv", index=False)

    methods_present = _methods_present(summary, allowed_methods)
    if not methods_present:
        return
    scenario_order = allowed_scenarios if allowed_scenarios is not None else SCENARIO_ORDER
    obs_scenarios = [sc for sc in scenario_order if sc.startswith("OBS")]
    if not obs_scenarios:
        obs_scenarios = [sc for sc in scenario_order if "OBS" in sc]
    metrics = [
        ("RMSE", "RMSE"),
        ("CI_Coverage", "CI Coverage"),
        ("CI_Width", "CI Width"),
    ]

    # Publication-grade table
    def _split_misspec(code: str) -> tuple[str, str]:
        mapping = {
            "correct": "Correct",
            "wrong": "Wrong",
        }
        try:
            outcome, propensity = code.split("_")
        except ValueError:
            return ("", "")
        return (mapping.get(outcome, outcome.title()), mapping.get(propensity, propensity.title()))

    table_records = []
    summary["Outcome"], summary["Propensity"] = zip(*summary["misspecification"].apply(_split_misspec))
    # Only include OBS scenarios in publication table (exclude RCT scenarios)
    for scenario in obs_scenarios:
        scen_df = summary[summary["scenario"] == scenario]
        if scen_df.empty:
            continue
        for misspec in sorted(scen_df["misspecification"].unique()):
            misspec_df = scen_df[scen_df["misspecification"] == misspec]
            if misspec_df.empty:
                continue
            outcome = misspec_df["Outcome"].iloc[0]
            propensity = misspec_df["Propensity"].iloc[0]
            row = {"Scenario": scenario, "Outcome": outcome, "Propensity": propensity}
            for metric_key, _ in metrics:
                for method in methods_present:
                    value = misspec_df[misspec_df["method"] == method][metric_key]
                    col = f"{metric_key}_{method}"
                    row[col] = float(value.iloc[0]) if not value.empty else np.nan
            table_records.append(row)

    if table_records:
        table_df = pd.DataFrame(table_records).sort_values(
            ["Scenario", "Outcome", "Propensity"]
        )
        format_map = {
            key: "{:.3f}" for key in table_df.columns if key.startswith(("RMSE_", "CI_Width_"))
        }
        format_map.update({key: "{:.2f}" for key in table_df.columns if key.startswith("CI_Coverage_")})

        display_df = table_df.copy()
        for col, fmt in format_map.items():
            if col in display_df:
                display_df[col] = display_df[col].apply(
                    lambda x: fmt.format(x) if pd.notnull(x) else ""
                )
        csv_path = tables_dir / "double_robust_publication_table.csv"
        display_df.to_csv(csv_path, index=False)

        metric_keys = [m[0] for m in metrics]
        metric_display = {
            "RMSE": "RMSE",
            "CI_Coverage": "CI Coverage",
            "CI_Width": "CI Width",
        }
        method_labels = list(methods_present)
        latex_lines = []
        latex_lines.append("\\begin{table}[htbp]")
        latex_lines.append("\\centering")
        column_spec = "lll" + "c" * (len(metric_keys) * len(method_labels))
        latex_lines.append(f"\\begin{{tabular}}{{{column_spec}}}")
        latex_lines.append("\\toprule")
        header_top = ["Scenario", "Outcome", "Propensity"]
        header_top += [
            f"\\multicolumn{{{len(method_labels)}}}{{c}}{{{metric_display[m]}}}"
            for m in metric_keys
        ]
        latex_lines.append(" & ".join(header_top) + " \\\\")
        start_col = 4
        for idx in range(len(metric_keys)):
            left = start_col + idx * len(method_labels)
            right = left + len(method_labels) - 1
            latex_lines.append(f"\\cmidrule(lr){{{left}-{right}}}")
        header_second = ["", "", ""]
        header_second += method_labels * len(metric_keys)
        latex_lines.append(" & ".join(header_second) + " \\\\")
        latex_lines.append("\\midrule")
        for _, row in display_df.iterrows():
            cells = [row["Scenario"], row["Outcome"], row["Propensity"]]
            for metric_key in metric_keys:
                for method in method_labels:
                    col = f"{metric_key}_{method}"
                    cells.append(row.get(col, ""))
            latex_lines.append(" & ".join(cells) + " \\\\")
        latex_lines.append("\\bottomrule")
        latex_lines.append("\\end{tabular}")
        latex_lines.append("\\caption{Double robustness experiment summary.}")
        latex_lines.append("\\label{tab:double_robust}")
        latex_lines.append("\\end{table}")
        tex_path = tables_dir / "double_robust_publication_table.tex"
        tex_path.write_text("\n".join(latex_lines), encoding="utf-8")

    fig, axes = plt.subplots(
        len(obs_scenarios),
        len(metrics),
        figsize=(15, 4 * max(1, len(obs_scenarios))),
        sharex=False,
        sharey=False,
    )
    if len(obs_scenarios) == 1:
        axes = np.expand_dims(axes, axis=0)

    for row, scenario in enumerate(obs_scenarios):
        scen_summary = summary[summary["scenario"] == scenario]
        if scen_summary.empty:
            for col in range(len(metrics)):
                axes[row, col].set_visible(False)
            continue
        ms_levels = list(scen_summary["misspecification"].unique())
        for col, (metric, title) in enumerate(metrics):
            ax = axes[row, col]
            box_data = []
            box_positions = []
            labels = []
            pos = 0
            spacing = 1.2
            for misspec in ms_levels:
                misspec_df = scen_summary[scen_summary["misspecification"] == misspec]
                if misspec_df.empty:
                    continue
                for method in methods_present:
                    method_vals = misspec_df[misspec_df["method"] == method][metric]
                    if method_vals.empty:
                        continue
                    box_data.append(method_vals.to_numpy())
                    box_positions.append(pos)
                    labels.append(f"{misspec}\n{method}")
                    pos += spacing
                pos += spacing * 0.5
            if not box_data:
                ax.set_visible(False)
                continue
            bp = ax.boxplot(
                box_data,
                positions=box_positions,
                widths=0.9,
                patch_artist=True,
                manage_ticks=False,
            )
            for patch, label in zip(bp["boxes"], labels):
                # Determine method color from label suffix
                method = label.split("\n")[-1]
                patch.set_facecolor(METHOD_COLORS.get(method, "#999999"))
                patch.set_alpha(0.75)
            for element in ["whiskers", "caps", "medians"]:
                for artist in bp[element]:
                    artist.set_color("#333333")
                    if element == "medians":
                        artist.set_linewidth(1.5)
            ax.set_xticks(box_positions)
            ax.set_xticklabels(labels, rotation=20, ha="right")
            ax.set_ylabel(title if col == 0 else "")
            ax.set_title(f"{scenario} - {title}")
            ax.grid(True, axis="y", ls="--", alpha=0.3)
            if metric == "CI_Coverage":
                ax.axhline(0.95, ls=":", color="gray", alpha=0.7)
    handles = [
        Line2D([0], [0], color=METHOD_COLORS[method], marker="s", linestyle="none", markersize=8, label=method)
        for method in methods_present
    ]
    if handles:
        fig.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.98),
            ncol=max(1, len(handles)),
            frameon=False,
        )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save_figure_multi_format(fig, figures_dir / "double_robust_boxplots")


def _nuisance_sensitivity_reports(
    df: pd.DataFrame,
    analysis_dir: Path,
    allowed_methods: Iterable[str] | None = None,
    allowed_scenarios: Iterable[str] | None = None,
):
    tables_dir, figures_dir = _ensure_directories(analysis_dir)
    if allowed_methods is not None:
        df = df[df["method"].isin(allowed_methods)]
    if allowed_scenarios is not None:
        allowed_set = {str(s) for s in allowed_scenarios}
        df = df[df["scenario"].astype(str).isin(allowed_set)]
    summary = _aggregate_metrics_with_sd(df, ["scenario", "method", "learner", "r_total"])
    if summary.empty:
        return
    summary["scenario"] = _set_scenario_category(summary["scenario"])

    full_runtime = (
        summary[summary["method"] == "FULL"]
        .groupby(["scenario", "learner"], dropna=False)["Runtime"]
        .mean()
    )

    def _runtime_ratio(row):
        if row["method"] == "FULL":
            return 1.0
        baseline = full_runtime.get((row["scenario"], row["learner"]))
        if baseline is not None and baseline > 0:
            return float(row["Runtime"] / baseline)
        return np.nan

    summary["Runtime_Ratio"] = summary.apply(_runtime_ratio, axis=1)
    summary.sort_values(["scenario", "learner", "method", "r_total"], inplace=True)

    # Filter to only include OBS-3 for publication table
    out = summary[summary["scenario"] == "OBS-3"].copy()
    out["scenario"] = out["scenario"].astype(str)
    out.to_csv(tables_dir / "nuisance_sensitivity_summary.csv", index=False)

    # Define metrics with plot type and transformation info
    # (raw_col, agg_col, sd_col, title, log_type, plot_type)
    # log_type: 'log' for log10(x), 'log1p' for log10(x+1), None for no transform
    # plot_type: 'box' for boxplot, 'bar' for barplot with SD
    metrics = [
        ("RMSE_rep", "RMSE", "RMSE_SD", "RMSE", "log", "box"),
        ("CI_Coverage", "CI_Coverage", "CI_Coverage_SD", "CI Coverage", None, "bar"),
        ("CI_Width", "CI_Width", "CI_Width_SD", "CI Width", "log", "box"),
        ("runtime", "Runtime", "Runtime_SD", "Runtime", "log1p", "bar"),
    ]
    scenario_order = allowed_scenarios if allowed_scenarios is not None else SCENARIO_ORDER
    scenarios_present = [sc for sc in scenario_order if not summary[summary["scenario"] == sc].empty]
    
    # Define hatching patterns for different learners to distinguish boxes
    learner_hatches = {"lasso_cv": "//", "rf": "\\\\", "lgbm": ""}
    
    for scenario in scenarios_present:
        scen_summary = summary[summary["scenario"] == scenario]
        if scen_summary.empty:
            continue
        learners = sorted(scen_summary["learner"].dropna().unique())
        methods_present = _methods_present(scen_summary, allowed_methods)
        if not learners or not methods_present:
            continue
        
        # Filter out FULL method (don't compare with real data)
        methods_present = [m for m in methods_present if m != "FULL"]
        if not methods_present:
            continue
        
        # Get raw data for this scenario (not aggregated)
        scen_df = df[df["scenario"] == scenario]
        r_totals = sorted(scen_df["r_total"].dropna().unique())
        
        # Create 2x2 subplot grid for 4 metrics
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = np.atleast_1d(axes).ravel()
        
        legend_handles = []
        legend_labels = []
        
        for ax_idx, (raw_col, agg_col, sd_col, title, log_type, plot_type) in enumerate(metrics):
            ax = axes[ax_idx]
            
            # Group spacing
            group_width = len(methods_present) * len(learners)
            group_spacing = group_width + 1
            
            if plot_type == "box":
                # Boxplot: use raw data from df
                box_data = []
                box_positions = []
                box_colors = []
                box_hatches = []
                current_pos = 0
                
                for r_idx, r_total in enumerate(r_totals):
                    for method in methods_present:
                        for learner in learners:
                            # Get raw data for this method-learner-r_total combination
                            subset = scen_df[
                                (scen_df["method"] == method) & 
                                (scen_df["learner"] == learner) &
                                (scen_df["r_total"] == r_total)
                            ]
                            
                            if subset.empty or raw_col not in subset.columns:
                                current_pos += 1
                                continue
                            
                            # Extract raw values and apply transformation if needed
                            values = subset[raw_col].dropna().values
                            
                            if len(values) > 0:
                                if log_type == "log":
                                    values = np.log10(values[values > 0])
                                elif log_type == "log1p":
                                    values = np.log10(values + 1)
                                
                                if len(values) > 0:
                                    box_data.append(values)
                                    box_positions.append(current_pos)
                                    box_colors.append(_get_method_color(method, learner))
                                    box_hatches.append(learner_hatches.get(learner, ""))
                                    
                                    # Add to legend (only for first metric)
                                    if ax_idx == 0 and r_idx == 0:
                                        legend_labels.append(f"{method}-{_format_learner_name(learner)}")
                            
                            current_pos += 1
                    
                    # Add spacing between r_total groups
                    current_pos = (r_idx + 1) * group_spacing
                
                # Create boxplot
                if box_data:
                    bp = ax.boxplot(
                        box_data,
                        positions=box_positions,
                        widths=0.6,
                        patch_artist=True,
                        showfliers=False,
                        manage_ticks=False,
                    )
                    
                    # Color the boxes
                    for patch, color, hatch in zip(bp["boxes"], box_colors, box_hatches):
                        patch.set_facecolor(color)
                        patch.set_alpha(0.7)
                        patch.set_hatch(hatch)
                        patch.set_edgecolor('black')
                        patch.set_linewidth(0.8)
                    
                    # Style other elements
                    for element in ["whiskers", "caps", "medians"]:
                        for artist in bp[element]:
                            artist.set_color("#333333")
                            if element == "medians":
                                artist.set_linewidth(2)
            
            else:  # plot_type == "bar"
                # Bar plot with SD: use aggregated data from summary
                bar_positions = []
                bar_heights = []
                bar_errors = []
                bar_colors = []
                bar_hatches = []
                current_pos = 0
                
                r_total_positions = np.arange(len(r_totals))
                bar_width = 0.12
                
                combination_idx = 0
                for method in methods_present:
                    for learner in learners:
                        means = []
                        sds = []
                        
                        for r_total in r_totals:
                            # Get aggregated data from summary
                            subset = scen_summary[
                                (scen_summary["method"] == method) & 
                                (scen_summary["learner"] == learner) &
                                (scen_summary["r_total"] == r_total)
                            ]
                            
                            if not subset.empty:
                                mean_val = subset[agg_col].iloc[0]
                                sd_val = subset[sd_col].iloc[0] if sd_col in subset.columns else 0
                                
                                # Apply log transformation if needed
                                if log_type == "log" and pd.notna(mean_val) and mean_val > 0:
                                    log_mean = np.log10(mean_val)
                                    log_sd = sd_val / (mean_val * np.log(10)) if sd_val > 0 else 0
                                    means.append(log_mean)
                                    sds.append(log_sd)
                                elif log_type == "log1p" and pd.notna(mean_val):
                                    log_mean = np.log10(mean_val + 1)
                                    log_sd = sd_val / ((mean_val + 1) * np.log(10)) if sd_val > 0 else 0
                                    means.append(log_mean)
                                    sds.append(log_sd)
                                else:
                                    means.append(mean_val if pd.notna(mean_val) else 0)
                                    sds.append(sd_val if pd.notna(sd_val) else 0)
                            else:
                                means.append(np.nan)
                                sds.append(0)
                        
                        # Calculate bar positions
                        n_combinations = len(methods_present) * len(learners)
                        bar_offset = -(n_combinations - 1) * bar_width / 2
                        positions = r_total_positions + bar_offset + combination_idx * bar_width
                        
                        color = _get_method_color(method, learner)
                        hatch = learner_hatches.get(learner, "")
                        
                        # Plot bars (CI Coverage without error bars)
                        if title == "CI Coverage":
                            bars = ax.bar(
                                positions,
                                means,
                                bar_width,
                                color=color,
                                edgecolor='black',
                                linewidth=0.5,
                                hatch=hatch,
                                alpha=0.7,
                            )
                        else:
                            bars = ax.bar(
                                positions,
                                means,
                                bar_width,
                                yerr=sds,
                                color=color,
                                edgecolor='black',
                                linewidth=0.5,
                                hatch=hatch,
                                capsize=3,
                                error_kw={'linewidth': 1},
                                alpha=0.7,
                            )
                        
                        # Add to legend (only for first metric)
                        if ax_idx == 0:
                            if combination_idx == 0:
                                legend_labels = []
                            legend_labels.append(f"{method}-{_format_learner_name(learner)}")
                        
                        combination_idx += 1
            
            # Set x-axis labels
            if plot_type == "box":
                group_centers = [(i * group_spacing + group_width / 2 - 0.5) for i in range(len(r_totals))]
                ax.set_xticks(group_centers)
            else:  # bar plot
                ax.set_xticks(np.arange(len(r_totals)))
            
            ax.set_xticklabels([f"{int(r)}" for r in r_totals])
            ax.set_xlabel("$r$")
            ax.set_title(title)
            ax.grid(True, axis='y', ls="--", alpha=0.3)
            
            # Set y-axis label for log-transformed metrics
            if log_type == "log":
                ax.set_ylabel(f"log₁₀({title})")
            elif log_type == "log1p":
                ax.set_ylabel(f"log₁₀({title}+1)")
            
            # Add reference line for CI Coverage
            if title == "CI Coverage":
                ax.axhline(0.95, ls=":", color="gray", alpha=0.7, linewidth=2)
        
        # Create legend with patches for method-learner combinations
        if legend_labels:
            legend_handles = []
            for idx, (method, learner) in enumerate(
                [(m, l) for m in methods_present for l in learners]
            ):
                color = _get_method_color(method, learner)
                hatch = learner_hatches.get(learner, "")
                patch = Patch(
                    facecolor=color, 
                    edgecolor='black',
                    hatch=hatch,
                    alpha=0.7,
                    label=f"{method}-{_format_learner_name(learner)}"
                )
                legend_handles.append(patch)
        
        # Add title first, then legend below it
        fig.suptitle(f"Learner Sensitivity ({scenario})", fontsize=BASE_FONT_SIZE + 4, y=0.98)
        
        if legend_handles:
            fig.legend(
                handles=legend_handles,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.94),
                ncol=min(6, len(legend_handles)),
                frameon=False,
            )
        
        fig.tight_layout(rect=[0, 0, 1, 0.90])
        _save_figure_multi_format(fig, figures_dir / f"nuisance_sensitivity_{scenario}")

# =============================================================================
# Public entry point
# =============================================================================


def generate_reports(exp_name: str, results: List[Dict], output_dir: Path) -> Dict[str, Path]:
    # Remove "experiment_" prefix from folder name
    folder_name = exp_name.replace("experiment_", "", 1) if exp_name.startswith("experiment_") else exp_name
    analysis_dir = Path("./analysis_results") / folder_name
    tables_dir, figures_dir = _ensure_directories(analysis_dir)

    df = prepare_dataframe(results)
    _, _, experiments = config.get_experiments()
    allowed_methods = experiments.get(exp_name, {}).get("methods")
    allowed_scenarios = experiments.get(exp_name, {}).get("scenarios")
    if exp_name == "experiment_visualization":
        _visualization_reports(results, analysis_dir)
    elif exp_name == "experiment_subsample_size":
        _subsample_size_reports(df, analysis_dir)
    elif exp_name == "experiment_population_size":
        _population_size_reports(df, analysis_dir, allowed_methods, allowed_scenarios)
    elif exp_name == "experiment_double_robust":
        _double_robust_reports(df, analysis_dir, allowed_methods, allowed_scenarios)
    elif exp_name == "experiment_nuisance_sensitivity":
        _nuisance_sensitivity_reports(df, analysis_dir, allowed_methods, allowed_scenarios)
    else:
        # Fallback: export raw dataframe if no dedicated report exists.
        if not df.empty:
            df.to_csv(tables_dir / "raw_results.csv", index=False)

    return {"tables_dir": tables_dir, "figures_dir": figures_dir}
