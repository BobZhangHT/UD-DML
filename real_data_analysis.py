#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""real_data_analysis.py

Real-data application of UD-DML on the CDC NCHS 2021 US natality file.

Treatment W   : maternal smoking during pregnancy (CIG_REC)
Outcome   Y   : birth weight in grams (DBWT), scaled to [0, 1]
Covariates X  : MAGER, MEDUC, PRECARE, PREVIS, SEX, DMAR, FAGECOMB,
                RF_GDIAB, RF_GHYPE, PRIORTERM   (p = 10)

Unified repeated-working-sample analysis
----------------------------------------
  * FULL-DML run once on the original sample as the reference.
  * For each of B paired repetitions on the same cleaned dataset, rerun
    all four working-sample methods at every budget r in ``--r-grid``. Method-specific
    random streams vary working-sample selection and cross-fitting while the
    observed data and FULL-DML reference remain fixed.
  * The canonical budget (``--canonical-r``, default = 5000) must be in
    the grid — Figure A slices at that budget, Figure B plots the full
    scaling curve.
  * Per-rep results are cached to disk; the script resumes automatically
    and skips reps whose cache file already exists.

Outputs under ``real_data_results/``:
    figures/real_data_application.{png,pdf} -- stability, agreement, and cost
    tables/real_data_application_table.tex  -- interval/design diagnostics
    tables/real_data_canonical_summary.csv  -- exact canonical values
    tables/real_data_scaling_summary.csv    -- exact scaling values
    raw/rep_<b>_<contract>.pkl.gz        -- per-rep cache (cells = r x method)
    raw/full_reference_<contract>.pkl.gz -- FULL-DML reference

Run
---
    python real_data_analysis.py \\
        --data-path Nat2021us/Nat2021US.txt \\
        --reps 100 --r-grid 1000,2500,5000,10000,25000 --canonical-r 5000 \\
        --jobs 16
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import pickle
import platform
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── Thread-env sanitiser BEFORE numpy import ──────────────────────────
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    _v = os.environ.get(_k, "").strip()
    if _v in ("", "0"):
        os.environ.pop(_k, None)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from tqdm.auto import tqdm

import config  # noqa: F401
import methods
from methods import run_full, run_sep_ud, run_stratified_unif, run_unif, run_ud


# ── Publication rcParams (Okabe-Ito) ──────────────────────────────────
METHOD_ORDER = ("UNIF", "STRAT-UNIF", "SEP-UD", "UD")
METHOD_COLORS = {
    "UNIF": "#56B4E9", "STRAT-UNIF": "#E69F00",
    "SEP-UD": "#CC79A7", "UD": "#D55E00", "FULL": "#009E73",
}
METHOD_MARKERS = {
    "UNIF": "s", "STRAT-UNIF": "^", "SEP-UD": "v", "UD": "o", "FULL": "D",
}
REAL_DATA_RESULT_SCHEMA_VERSION = 6


def _learner_backend_name() -> str:
    return (
        "lightgbm" if getattr(methods, "_HAS_LIGHTGBM", False)
        else "sklearn-gradient-boosting-fallback"
    )


def _analysis_cache_tag(
    *, n: int, r_grid: List[int], k_folds: int, learner: str
) -> str:
    contract = {
        "schema": REAL_DATA_RESULT_SCHEMA_VERSION,
        "n": int(n),
        "r_grid": [int(value) for value in sorted(r_grid)],
        "k_folds": int(k_folds),
        "learner": str(learner),
        "learner_backend": _learner_backend_name(),
        "methods": list(METHOD_ORDER),
        "base_seed": int(config.BASE_SEED),
    }
    digest = hashlib.sha256(
        json.dumps(contract, sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]
    return f"schema{REAL_DATA_RESULT_SCHEMA_VERSION}_{digest}"


def _replication_cache_path(cache_dir: Path, rep: int, cache_tag: str) -> Path:
    return cache_dir / f"rep_{rep:04d}_{cache_tag}.pkl.gz"

plt.rcParams.update({
    "font.family": "serif", "font.size": 13, "mathtext.fontset": "stix",
    "axes.titlesize": 17, "axes.titleweight": "bold",
    "axes.labelsize": 15, "axes.labelweight": "bold",
    "figure.titlesize": 18, "figure.titleweight": "bold",
    "xtick.labelsize": 13, "ytick.labelsize": 13,
    "xtick.direction": "in", "ytick.direction": "in",
    "legend.fontsize": 13, "legend.frameon": True, "legend.framealpha": 0.9,
    "legend.edgecolor": "0.4",
    "axes.linewidth": 1.2, "lines.linewidth": 2.0, "lines.markersize": 6,
    "grid.alpha": 0.35,
    "savefig.dpi": 300, "savefig.bbox": "tight",
})


# ═══════════════════════════════════════════════════════════════════════
# 1. Data loading
# ═══════════════════════════════════════════════════════════════════════
_COLSPECS = [
    (74, 76), (123, 124), (223, 225), (237, 239), (474, 475),
    (119, 120), (146, 148), (313, 314), (315, 316), (174, 176),
    (268, 269), (503, 507),
]
_NAMES = [
    "MAGER", "MEDUC", "PRECARE", "PREVIS", "SEX", "DMAR",
    "FAGECOMB", "RF_GDIAB", "RF_GHYPE", "PRIORTERM",
    "CIG_REC", "DBWT",
]
_X_COLS = [
    "MAGER", "MEDUC", "PRECARE", "PREVIS", "SEX", "DMAR",
    "FAGECOMB", "RF_GDIAB", "RF_GHYPE", "PRIORTERM",
]


def load_natality_data(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Data file not found: {p.resolve()}")
    print(f"[data] reading {p} ...")
    t0 = time.perf_counter()
    df = pd.read_fwf(str(p), colspecs=_COLSPECS, names=_NAMES, dtype=str)
    print(f"[data] raw records: {len(df):,}  ({time.perf_counter()-t0:.1f}s)")

    df["CIG_REC"] = df["CIG_REC"].str.strip()
    df = df[df["CIG_REC"].isin(["Y", "N"])].copy()
    df["W"] = df["CIG_REC"].map({"Y": 1, "N": 0}).astype(int)
    df["DBWT"] = pd.to_numeric(df["DBWT"].str.strip(), errors="coerce")
    df = df[df["DBWT"].notna() & (df["DBWT"] > 0) & (df["DBWT"] < 9000)].copy()
    for col in ["MAGER", "MEDUC", "PRECARE", "PREVIS", "PRIORTERM"]:
        df[col] = pd.to_numeric(df[col].str.strip(), errors="coerce")
    df["FAGECOMB"] = pd.to_numeric(df["FAGECOMB"].str.strip(), errors="coerce")
    df.loc[df["FAGECOMB"] == 99, "FAGECOMB"] = np.nan
    df["SEX"] = df["SEX"].str.strip().map({"M": 1, "F": 0})
    df["DMAR"] = pd.to_numeric(df["DMAR"].str.strip(), errors="coerce")
    for col in ["RF_GDIAB", "RF_GHYPE"]:
        df[col] = df[col].str.strip().map({"Y": 1, "N": 0})

    df_clean = df[["DBWT", "W"] + _X_COLS].dropna().copy()
    y_min, y_max = float(df_clean["DBWT"].min()), float(df_clean["DBWT"].max())
    df_clean["Y"] = (df_clean["DBWT"] - y_min) / (y_max - y_min)
    X = df_clean[_X_COLS].values.astype(np.float64)
    W = df_clean["W"].values.astype(np.float64)
    Y = df_clean["Y"].values.astype(np.float64)
    print(f"[data] clean n = {len(df_clean):,}  |  smokers = {int(W.sum()):,}  "
          f"({W.mean()*100:.2f}%)  |  DBWT ∈ [{y_min:.0f}g, {y_max:.0f}g]")
    return X, W, Y, y_min, y_max


# ═══════════════════════════════════════════════════════════════════════
# 2. Per-repetition worker (fixed data + all r × method)
# ═══════════════════════════════════════════════════════════════════════

def _run_method(method: str, X: np.ndarray, W: np.ndarray, Y: np.ndarray,
                *, r_total: Optional[int], seed: int, k_folds: int,
                learner: str = "lgbm") -> Dict:
    pi_val = float(W.mean())
    kw = dict(is_rct=False, pi_true=pi_val, k_folds=k_folds,
              sim_seed=seed, learner=learner)
    if method == "FULL":
        return run_full(X, W, Y, **kw)
    if method == "UNIF":
        return run_unif(X, W, Y, r={"r_total": int(r_total)}, **kw)
    if method == "STRAT-UNIF":
        return run_stratified_unif(X, W, Y, r={"r_total": int(r_total)}, **kw)
    if method == "SEP-UD":
        return run_sep_ud(
            X, W, Y, r={"r_total": int(r_total)},
            cache_seed=int(seed) + 10_007 * int(r_total),
            population_size=int(X.shape[0]), **kw,
        )
    if method == "UD":
        return run_ud(
            X,
            W,
            Y,
            r={"r_total": int(r_total)},
            cache_seed=int(seed) + 10_007 * int(r_total),
            population_size=int(X.shape[0]),
            **kw,
        )
    raise ValueError(method)


def _cap_worker_threads() -> None:
    """Pin BLAS / genUD to 1 thread inside each joblib worker."""
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
              "UD_GENUD_NUM_THREADS"):
        os.environ[k] = "1"


def _process_one_rep(
    rep: int,
    X: np.ndarray, W: np.ndarray, Y: np.ndarray,
    r_grid: List[int], run_seed: int,
    k_folds: int, learner: str,
    cache_dir: Path, cache_tag: str, ate_full: float,
) -> Optional[Path]:
    """One paired repeated-fit run of UD and UNIF on the fixed dataset.

    Each repetition runs the full budget-by-method grid on the fixed dataset.
    The methods share the fixed observed data and repetition identifier. Cached
    repetitions are skipped only when the full run contract matches.
    """
    _cap_worker_threads()
    cache_file = _replication_cache_path(cache_dir, rep, cache_tag)
    if cache_file.exists():
        return cache_file

    X_b = X; W_b = W; Y_b = Y
    if W_b.sum() < 10 or (1 - W_b).sum() < 10:
        # Defensive guard for a malformed fixed analysis dataset.
        with gzip.open(cache_file, "wb", compresslevel=1) as f:
            pickle.dump({"rep": rep, "skipped": True, "cache_tag": cache_tag}, f,
                        protocol=pickle.HIGHEST_PROTOCOL)
        return cache_file

    rows = []
    for r_total in r_grid:
        for method in METHOD_ORDER:
            res = _run_method(
                method, X_b, W_b, Y_b,
                r_total=r_total, seed=int(run_seed), k_folds=k_folds,
                learner=learner,
            )
            diagnostics = res.get("design_diagnostics") or {}
            rows.append({
                "rep": rep, "method": method, "r_total": int(r_total),
                "est_ate": float(res["est_ate"]),
                "ci_lower": float(res["ci_lower"]),
                "ci_upper": float(res["ci_upper"]),
                "ci_width": float(res["ci_upper"] - res["ci_lower"]),
                "runtime": float(res["runtime"]),
                "standard_error": float(res.get("standard_error", np.nan)),
                "variance_method": res.get("variance_method"),
                "smd_mean": diagnostics.get("smd_mean"),
                "smd_max": diagnostics.get("smd_max"),
                "gefd_estimate": diagnostics.get("gefd_estimate"),
                "matching_mean_distance": diagnostics.get("matching_mean_distance"),
                "matching_max_distance": diagnostics.get("matching_max_distance"),
            })
    with gzip.open(cache_file, "wb", compresslevel=1) as f:
        pickle.dump(
            {"rep": rep, "rows": rows, "skipped": False, "cache_tag": cache_tag},
            f,
                    protocol=pickle.HIGHEST_PROTOCOL)
    return cache_file


# ═══════════════════════════════════════════════════════════════════════
# 3. Figure and table emitters (same outputs as old Plan A + Plan B)
# ═══════════════════════════════════════════════════════════════════════

def _emit_application_figure(
    df: pd.DataFrame,
    r_canonical: int,
    ate_full: float,
    ci_full: Tuple[float, float],
    time_full: float,
    fig_dir: Path,
):
    """Emit one non-redundant real-data figure for the manuscript."""
    canonical = df[df["r_total"] == r_canonical].copy()
    rmsref = (
        df.assign(sq_ref=(df["est_ate"] - ate_full) ** 2)
        .groupby(["method", "r_total"], observed=False)["sq_ref"]
        .mean().pow(0.5).rename("rmsref").reset_index()
    )
    scaling = (
        df.groupby(["method", "r_total"], observed=False)
        .agg(repeated_fit_sd=("est_ate", "std"), mean_runtime=("runtime", "mean"))
        .reset_index()
        .merge(rmsref, on=["method", "r_total"], how="left")
    )
    scaling["speedup_vs_full"] = time_full / scaling["mean_runtime"]

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), constrained_layout=True)
    ax = axes[0, 0]
    bp = ax.boxplot(
        [canonical[canonical["method"] == m]["est_ate"].values for m in METHOD_ORDER],
        positions=np.arange(1, len(METHOD_ORDER) + 1), widths=0.55,
        patch_artist=True, showfliers=True,
        medianprops=dict(color="black", linewidth=2),
    )
    for patch, method in zip(bp["boxes"], METHOD_ORDER):
        patch.set_facecolor(METHOD_COLORS[method])
        patch.set_alpha(0.65)
    ax.axhline(ate_full, ls="--", color=METHOD_COLORS["FULL"], linewidth=2.0)
    ax.axhspan(ci_full[0], ci_full[1], color=METHOD_COLORS["FULL"], alpha=0.12)
    ax.set_xticks(np.arange(1, len(METHOD_ORDER) + 1))
    ax.set_xticklabels(METHOD_ORDER, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("ATE (scaled)")
    ax.set_title("(a) Canonical-budget estimates", loc="left")
    ax.grid(True, ls="--", axis="y")

    panels = [
        (axes[0, 1], "rmsref", "RMS reference discrepancy", "(b) Reference agreement"),
        (axes[1, 0], "repeated_fit_sd", r"Repeated-fit SD of $\widehat{\theta}$", "(c) Repeated-fit stability"),
        (axes[1, 1], "speedup_vs_full", "Speed-up vs FULL-DML", "(d) Relative computational cost"),
    ]
    for panel, column, ylabel, title in panels:
        for method in METHOD_ORDER:
            method_data = scaling[scaling["method"] == method].sort_values("r_total")
            panel.plot(
                method_data["r_total"], method_data[column],
                marker=METHOD_MARKERS[method], color=METHOD_COLORS[method],
                linewidth=2.2, markersize=8,
            )
        if column == "speedup_vs_full":
            panel.axhspan(1.0, 500.0, color=METHOD_COLORS["FULL"], alpha=0.06, zorder=0)
            panel.axhspan(0.3, 1.0, color="0.75", alpha=0.12, zorder=0)
            panel.axhline(1.0, ls="--", color=METHOD_COLORS["FULL"], linewidth=2)
            panel.set_ylim(0.3, 500.0)
            panel.text(
                0.97, 0.95, "Working-sample method faster", transform=panel.transAxes,
                color=METHOD_COLORS["FULL"], fontsize=9, va="top", ha="right",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72,
                      "pad": 1.5},
            )
            panel.text(
                0.03, 0.04, "FULL-DML faster", transform=panel.transAxes,
                color="0.35", fontsize=9, va="bottom",
            )
        panel.set_xscale("log")
        panel.set_yscale("log")
        panel.set_xlabel("Working-sample size $r$")
        panel.set_ylabel(ylabel, labelpad=2)
        panel.set_title(title, loc="left")
        panel.grid(True, ls="--", which="both")
        r_sorted = sorted(set(int(r) for r in scaling["r_total"].tolist()))
        panel.set_xticks(r_sorted)
        panel.set_xticklabels(
            [f"{r / 1000:g}k" if r >= 1000 else str(r) for r in r_sorted],
            fontsize=9,
        )
        from matplotlib.ticker import NullFormatter
        panel.xaxis.set_minor_formatter(NullFormatter())
        panel.tick_params(axis="x", which="minor", bottom=False, labelbottom=False)

    handles = [
        plt.Line2D([], [], color=METHOD_COLORS[m], marker=METHOD_MARKERS[m],
                   linewidth=2.2, label=f"{m}-DML")
        for m in METHOD_ORDER
    ]
    handles.append(
        plt.Line2D([], [], color=METHOD_COLORS["FULL"], linestyle="--",
                   linewidth=2.0, label="FULL-DML reference")
    )
    fig.legend(handles=handles, loc="outside upper center", ncol=5, frameon=False)
    for ext in (".png", ".pdf"):
        fig.savefig(fig_dir / f"real_data_application{ext}", bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def _emit_tables(df: pd.DataFrame, r_grid: List[int], r_canonical: int,
                 ate_full: float, ci_full: Tuple[float, float],
                 time_full: float, n_total: int, B_actual: int,
                 y_min: float, y_max: float, tables_dir: Path):
    # ── Table A: canonical-r stability summary ─────────────────────────
    sub = df[df["r_total"] == r_canonical]
    def _agg(s):
        a = s["est_ate"].values
        return pd.Series({
            "Mean ATE": a.mean(), "SD ATE": a.std(ddof=1),
            "Abs. reference difference": abs(a.mean() - ate_full),
            "MS reference discrepancy": ((a - ate_full) ** 2).mean(),
            "CI Width": s["ci_width"].mean(),
            "Runtime": s["runtime"].mean(),
        })
    summary_A = sub.groupby("method").apply(_agg, include_groups=False).reindex(METHOD_ORDER)
    speedups = {
        method: time_full / max(summary_A.loc[method, "Runtime"], 1e-9)
        for method in METHOD_ORDER
    }
    ate_g = {"FULL": ate_full * (y_max - y_min)}
    ate_g.update({
        method: summary_A.loc[method, "Mean ATE"] * (y_max - y_min)
        for method in METHOD_ORDER
    })
    cap_A = (
        f"Repeated working-sample stability of the four working-sample designs at the canonical "
        f"budget $r={r_canonical}$ on the CDC 2021 US natality dataset "
        f"($n={n_total:,}$, $B={B_actual}$ paired repetitions).  "
        f"Columns: mean ATE, Monte-Carlo SD, absolute reference difference "
        f"and mean-squared discrepancy relative to the estimated full-sample "
        f"DML reference, mean 95\\% CI width, mean wall-clock "
        f"runtime, and speed-up relative to FULL on $n$.  The back-transformed "
        f"FULL reference is ${ate_g['FULL']:.1f}$\\,g; working-sample estimates "
        f"are reported in the same scaled units in the table."
    )
    lines_A = [
        r"\begin{table}[htbp]", r"\centering",
        r"\caption{" + cap_A + "}", r"\label{tab:real_data_plan_A}",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{lrrrrrrr}", r"\toprule",
        r"Method & Mean ATE & SD & Abs. Ref. Diff. & MS Ref. Disc. & CI Width & Runtime (s) & Speed-up \\",
        r"\midrule",
        (f"FULL & {ate_full:.5f} & --- & --- & --- & "
         f"{ci_full[1]-ci_full[0]:.5f} & {time_full:.2f} & 1$\\times$ \\\\"),
    ]
    for m in METHOD_ORDER:
        r = summary_A.loc[m]
        lines_A.append(
            f"{m} & {r['Mean ATE']:.5f} & {r['SD ATE']:.5f} & "
            f"{r['Abs. reference difference']:.5f} & "
            f"{r['MS reference discrepancy']:.2e} & "
            f"{r['CI Width']:.5f} & {r['Runtime']:.2f} & {speedups[m]:.0f}$\\times$ \\\\"
        )
    lines_A.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    (tables_dir / "real_data_plan_A.tex").write_text("\n".join(lines_A) + "\n", encoding="utf-8")

    # ── Table B: r-scaling grid (excludes canonical r) ─────────────────
    # MC SD is the SD across paired repeated working-sample fits.
    r_grid_B = [r for r in r_grid if r != r_canonical]
    agg_B = df[df["r_total"].isin(r_grid_B)].groupby(["method", "r_total"]).apply(
        lambda s: pd.Series({
            "rmse": np.sqrt(((s["est_ate"] - ate_full) ** 2).mean()),
            "mc_sd": s["est_ate"].std(ddof=1),
            "mean_ci_width": s["ci_width"].mean(),
            "mean_runtime":  s["runtime"].mean(),
        }), include_groups=False,
    ).reset_index()
    cap_B = (
        f"Subsample-size scaling on the CDC 2021 natality data "
        f"($B={B_actual}$ paired repetitions).  Budgets disjoint "
        f"from Plan A's canonical $r={r_canonical}$.  Columns: root "
        f"mean-squared discrepancy to the estimated FULL reference; "
        f"repeated-fit Monte-Carlo standard deviation of "
        f"$\\widehat{{\\theta}}_{{\\mathrm{{sub}}}}$; mean interval width; "
        f"and mean wall-clock runtime per call.  "
        f"FULL runtime $= {time_full:.2f}$\\,s on $n={n_total:,}$.  "
        f"The repeated-fit SD is descriptive. Because FULL-DML is computed "
        f"once on the fixed dataset, reference discrepancy is not RMSE for "
        f"the unknown causal effect and does not measure superpopulation uncertainty."
    )
    lines_B = [
        r"\begin{table}[htbp]", r"\centering",
        r"\caption{" + cap_B + "}", r"\label{tab:real_data_plan_B}",
        r"\setlength{\tabcolsep}{5pt}",
        r"\begin{tabular}{lrrrrr}", r"\toprule",
        r"Method & $r$ & RMS Ref. Discrepancy & MC SD & CI Width & Runtime (s) \\",
        r"\midrule",
    ]
    for m in METHOD_ORDER:
        for r in r_grid_B:
            row = agg_B[(agg_B["method"] == m) & (agg_B["r_total"] == r)]
            if row.empty:
                continue
            row = row.iloc[0]
            lines_B.append(
                f"{m} & {int(row['r_total'])} & {row['rmse']:.5f} & "
                f"{row['mc_sd']:.5f} & {row['mean_ci_width']:.5f} & "
                f"{row['mean_runtime']:.2f} \\\\"
            )
        lines_B.append(r"\midrule")
    if lines_B[-1] == r"\midrule":
        lines_B.pop()
    lines_B.append(r"\midrule")
    lines_B.append(
        f"FULL & $n={n_total:,}$ & 0 (ref) & --- & --- & {time_full:.2f} \\\\"
    )
    lines_B.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    (tables_dir / "real_data_plan_B.tex").write_text("\n".join(lines_B) + "\n", encoding="utf-8")

    design_summary = (
        sub.groupby("method", observed=False)
        .agg(
            Mean_SMD=("smd_mean", "mean"),
            Mean_Max_SMD=("smd_max", "mean"),
            Mean_GEFD=("gefd_estimate", "mean"),
            Mean_Matching_Radius=("matching_mean_distance", "mean"),
            Mean_Max_Matching_Radius=("matching_max_distance", "mean"),
        )
        .reindex(METHOD_ORDER)
        .reset_index()
        .rename(columns={"method": "Method"})
    )
    design_summary.to_csv(
        tables_dir / "real_data_design_diagnostics.csv", index=False
    )
    (tables_dir / "real_data_design_diagnostics.tex").write_text(
        design_summary.to_latex(
            index=False,
            na_rep="--",
            float_format=lambda value: f"{value:.4f}",
            caption=(
                f"Design diagnostics at the canonical budget $r={r_canonical}$. "
                "GEFD and matching radii apply to SEP-UD-DML and UD-DML; all quantities "
                "are descriptive averages across paired working-sample repetitions."
            ),
            label="tab:real_data_design_diagnostics",
        ),
        encoding="utf-8",
    )


# ═══════════════════════════════════════════════════════════════════════
# 4. Unified analysis driver
# ═══════════════════════════════════════════════════════════════════════

def _emit_nonredundant_tables(
    df: pd.DataFrame,
    r_canonical: int,
    ate_full: float,
    n_total: int,
    B_actual: int,
    tables_dir: Path,
) -> None:
    """Write one unique manuscript table and exact supplemental CSV files."""
    canonical = df[df["r_total"] == r_canonical].copy()
    canonical_summary = canonical.groupby("method", observed=False).apply(
        lambda sample: pd.Series({
            "mean_ate": sample["est_ate"].mean(),
            "repeated_fit_sd": sample["est_ate"].std(ddof=1),
            "rms_reference_discrepancy": np.sqrt(
                ((sample["est_ate"] - ate_full) ** 2).mean()
            ),
            "mean_runtime": sample["runtime"].mean(),
        }),
        include_groups=False,
    ).reindex(METHOD_ORDER).reset_index()
    canonical_summary.to_csv(
        tables_dir / "real_data_canonical_summary.csv", index=False
    )

    scaling_summary = df.groupby(["method", "r_total"], observed=False).apply(
        lambda sample: pd.Series({
            "rms_reference_discrepancy": np.sqrt(
                ((sample["est_ate"] - ate_full) ** 2).mean()
            ),
            "repeated_fit_sd": sample["est_ate"].std(ddof=1),
            "mean_runtime": sample["runtime"].mean(),
        }),
        include_groups=False,
    ).reset_index()
    scaling_summary.to_csv(
        tables_dir / "real_data_scaling_summary.csv", index=False
    )

    diagnostics = canonical.groupby("method", observed=False).agg(
        Mean_CI_Width=("ci_width", "mean"),
        Mean_SMD=("smd_mean", "mean"),
        Maximum_SMD=("smd_max", "mean"),
        Realised_GEFD=("gefd_estimate", "mean"),
        Mean_Matching_Radius=("matching_mean_distance", "mean"),
        Maximum_Matching_Radius=("matching_max_distance", "mean"),
    ).reindex(METHOD_ORDER).reset_index().rename(columns={"method": "Method"})
    diagnostics.to_csv(tables_dir / "real_data_application_table.csv", index=False)

    def fmt(value: float) -> str:
        return "--" if pd.isna(value) else f"{value:.4f}"

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        (r"\caption{Interval and design diagnostics at the canonical working-sample "
         f"budget $r={r_canonical}$ on the fixed natality dataset "
         f"($n={n_total:,}$, $B={B_actual}$ paired repetitions). "
         r"GEFD and matching radii apply only to the two UD-based selectors.}"),
        r"\label{tab:real_data_summary}",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Method & Mean CI width & Mean SMD & Max. SMD & GEFD & Mean radius & Max. radius \\",
        r"\midrule",
    ]
    for _, row in diagnostics.iterrows():
        lines.append(
            f"{row['Method']}-DML & {fmt(row['Mean_CI_Width'])} & "
            f"{fmt(row['Mean_SMD'])} & {fmt(row['Maximum_SMD'])} & "
            f"{fmt(row['Realised_GEFD'])} & {fmt(row['Mean_Matching_Radius'])} & "
            f"{fmt(row['Maximum_Matching_Radius'])}" + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    (tables_dir / "real_data_application_table.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def run_analysis(
    X: np.ndarray, W: np.ndarray, Y: np.ndarray,
    y_min: float, y_max: float,
    *,
    r_grid: List[int],
    r_canonical: int,
    B: int,
    k_folds: int,
    n_jobs: int,
    output_root: Path,
    learner: str = "lgbm",
) -> Dict:
    if r_canonical not in r_grid:
        raise ValueError(
            f"--canonical-r {r_canonical} must be one of the --r-grid values {r_grid}"
        )

    cache_dir   = output_root / "raw";     cache_dir.mkdir(parents=True, exist_ok=True)
    fig_dir     = output_root / "figures"; fig_dir.mkdir(parents=True, exist_ok=True)
    tables_dir  = output_root / "tables";  tables_dir.mkdir(parents=True, exist_ok=True)
    cache_tag = _analysis_cache_tag(
        n=X.shape[0], r_grid=r_grid, k_folds=k_folds, learner=learner
    )
    run_contract = {
        "schema_version": REAL_DATA_RESULT_SCHEMA_VERSION,
        "cache_tag": cache_tag,
        "population_size": int(X.shape[0]),
        "treated_count": int(np.sum(W == 1)),
        "control_count": int(np.sum(W == 0)),
        "r_grid": [int(value) for value in r_grid],
        "canonical_r": int(r_canonical),
        "paired_repetitions": int(B),
        "k_folds": int(k_folds),
        "learner": learner,
        "learner_backend": _learner_backend_name(),
        "base_seed": int(config.BASE_SEED),
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    (output_root / "run_manifest.json").write_text(
        json.dumps(run_contract, indent=2, sort_keys=True), encoding="utf-8"
    )

    # ── FULL reference (cached) ────────────────────────────────────────
    full_cache = cache_dir / f"full_reference_{cache_tag}.pkl.gz"
    if full_cache.exists():
        with gzip.open(full_cache, "rb") as f:
            ref = pickle.load(f)
        ate_full = float(ref["est_ate"])
        ci_full = (float(ref["ci_lower"]), float(ref["ci_upper"]))
        time_full = float(ref["runtime"])
        print(f"[FULL] cached: ATE = {ate_full:.5f} | t = {time_full:.1f}s")
    else:
        print("[FULL] running on full sample (one-off reference)...")
        res_full = _run_method("FULL", X, W, Y, r_total=None,
                               seed=0, k_folds=k_folds, learner=learner)
        ate_full = float(res_full["est_ate"])
        ci_full = (float(res_full["ci_lower"]), float(res_full["ci_upper"]))
        time_full = float(res_full["runtime"])
        with gzip.open(full_cache, "wb", compresslevel=1) as f:
            pickle.dump({"est_ate": ate_full, "ci_lower": ci_full[0],
                         "ci_upper": ci_full[1], "runtime": time_full},
                        f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[FULL] ATE = {ate_full:.5f} | CI = [{ci_full[0]:.5f}, "
              f"{ci_full[1]:.5f}] | t = {time_full:.1f}s")

    # Resumable paired repeated-fit plan.
    run_seeds = np.random.default_rng(config.BASE_SEED + 5003).integers(
        0, 2**31 - 1, size=B
    )
    pending = [
        b
        for b in range(B)
        if not _replication_cache_path(cache_dir, b, cache_tag).exists()
    ]
    done    = B - len(pending)
    print(f"[fits] {B} reps total  |  {done} cached  |  {len(pending)} pending")

    if pending:
        n = X.shape[0]
        # Cap workers: never more than pending reps (avoids 16-worker spawn for 10 tasks).
        eff_jobs = max(1, min(n_jobs, len(pending)))
        # joblib temp folder for memmap of large X/W/Y arrays.
        # Python's Windows multiprocessing resource tracker serialises this
        # path as ASCII.  The repository may live below a Unicode directory,
        # so place transient memmaps below the system temp directory instead.
        mmap_dir = (
            Path(tempfile.gettempdir())
            / "ud_dml_joblib"
            / f"schema{REAL_DATA_RESULT_SCHEMA_VERSION}_{os.getpid()}"
        )
        mmap_dir.mkdir(parents=True, exist_ok=True)
        print(f"[fits] parallel jobs = {eff_jobs} (req {n_jobs})  "
              f"|  fixed n = {n:,}  |  mmap = {mmap_dir}")

        # Wrap _process_one_rep so X/W/Y are explicit args -> joblib auto-memmaps
        # (closure-captured arrays bypass memmap and get pickled per worker).
        def _task(b, X_, W_, Y_):
            return _process_one_rep(
                b, X_, W_, Y_, r_grid, int(run_seeds[b]),
                k_folds, learner, cache_dir, cache_tag, ate_full,
            )

        with tqdm(total=len(pending), desc="paired working-sample fits",
                  unit="rep", dynamic_ncols=True, smoothing=0.0,
                  mininterval=0.5) as bar:
            with Parallel(
                n_jobs=eff_jobs, batch_size=1, pre_dispatch="2*n_jobs",
                return_as="generator_unordered",
                backend="loky", temp_folder=str(mmap_dir),
                max_nbytes="10M", mmap_mode="r",
            ) as par:
                for _ in par(delayed(_task)(b, X, W, Y) for b in pending):
                    bar.update(1)

    # ── Load all cached reps into one DataFrame ────────────────────────
    rows: List[Dict] = []
    skipped = 0
    for b in range(B):
        fp = _replication_cache_path(cache_dir, b, cache_tag)
        if not fp.exists():
            continue
        with gzip.open(fp, "rb") as f:
            payload = pickle.load(f)
        if payload.get("skipped"):
            skipped += 1
            continue
        rows.extend(payload["rows"])
    if skipped:
        print(f"[fits] {skipped} repetitions skipped (pathological "
              f"treated/control imbalance).")
    df = pd.DataFrame(rows)
    B_actual = df["rep"].nunique() if not df.empty else 0
    print(f"[fits] usable reps: {B_actual}")
    if B_actual != B:
        raise RuntimeError(
            f"Expected {B} paired repetitions but found {B_actual}; "
            "refusing to emit incomplete reviewer-facing summaries."
        )
    df.to_csv(output_root / "repeated_fit_results.csv", index=False)

    # ── Emit outputs ───────────────────────────────────────────────────
    n_total = X.shape[0]
    _emit_application_figure(
        df, r_canonical, ate_full, ci_full, time_full, fig_dir
    )
    print(f"[out] figure -> {fig_dir / 'real_data_application.{png,pdf}'}")
    _emit_nonredundant_tables(
        df, r_canonical, ate_full, n_total, B_actual, tables_dir
    )
    print(f"[out] table  -> {tables_dir / 'real_data_application_table.tex'}")
    print(f"[out] exact  -> {tables_dir} / real_data_[canonical,scaling]_summary.csv")

    return {"ate_full": ate_full, "time_full": time_full, "B": B_actual,
            "output_root": str(output_root)}


# ═══════════════════════════════════════════════════════════════════════
# 5. CLI
# ═══════════════════════════════════════════════════════════════════════

def _parse_r_grid(s: str) -> List[int]:
    return sorted(int(x.strip()) for x in s.split(",") if x.strip())


def main():
    ap = argparse.ArgumentParser(description="UD-DML real-data analysis (CDC 2021 natality).")
    ap.add_argument("--data-path", type=str, default="Nat2021us/Nat2021US.txt",
                    help="Path to Nat2021US.txt fixed-width file.")
    ap.add_argument("--reps", type=int, default=100,
                    help="Number of paired working-sample repetitions (default 100).")
    ap.add_argument("--r-grid", type=str, default="1000,2500,5000,10000,25000",
                    help="Comma-separated r values.  Must include --canonical-r.")
    ap.add_argument("--canonical-r", type=int, default=5000,
                    help="Budget used in Figure A (stability slice).")
    ap.add_argument("--k-folds", type=int, default=5,
                    help="DML cross-fitting folds.")
    ap.add_argument("--jobs", type=int, default=-1,
                    help="Parallel jobs (-1 = all CPUs).")
    ap.add_argument("--out", type=str, default="real_data_results",
                    help="Output root directory.")
    ap.add_argument("--clear-cache", action="store_true",
                    help="Delete cached rep_*.pkl.gz before running (forces restart).")
    ap.add_argument("--fast-demo", action="store_true",
                    help="Fast smoke-test mode: force B=10, take a seeded "
                         "100000-row working population, and use r=500,1000,2500. "
                         "Writes outputs under '<out>_fast_demo/' to avoid "
                         "polluting the full-run cache.")
    ap.add_argument(
        "--full",
        action="store_true",
        help="Run the full cleaned dataset, requested repetitions, and r-grid.",
    )
    args = ap.parse_args()
    if args.fast_demo and args.full:
        raise ValueError("Choose exactly one of --fast-demo or --full.")

    if args.fast_demo:
        reps = 10
        out_dir = args.out + "_fast_demo"
        print(f"[mode] FAST-DEMO: B={reps}, output -> {out_dir}/")
    else:
        reps = args.reps
        out_dir = args.out
        print(f"[mode] FULL: B={reps}, output -> {out_dir}/")

    output_root = Path(out_dir)
    if args.clear_cache:
        import shutil
        for sub in ("raw", "figures", "tables"):
            p = output_root / sub
            if p.exists():
                shutil.rmtree(p)
                print(f"[cache] cleared {p}")

    X, W, Y, y_min, y_max = load_natality_data(args.data_path)
    if args.fast_demo and X.shape[0] > 100_000:
        demo_rng = np.random.default_rng(int(config.BASE_SEED) + 7001)
        demo_indices = demo_rng.choice(X.shape[0], size=100_000, replace=False)
        X, W, Y = X[demo_indices], W[demo_indices], Y[demo_indices]
        r_grid = [500, 1_000, 2_500]
        canonical_r = 1_000
        print(
            f"[mode] FAST-DEMO population: n={X.shape[0]:,}, "
            f"smokers={int(W.sum()):,}, r_grid={r_grid}"
        )
    else:
        r_grid = _parse_r_grid(args.r_grid)
        canonical_r = args.canonical_r
    n_jobs = os.cpu_count() if args.jobs == -1 else int(args.jobs)

    run_analysis(
        X, W, Y, y_min, y_max,
        r_grid=r_grid, r_canonical=canonical_r,
        B=reps, k_folds=args.k_folds,
        n_jobs=n_jobs, output_root=output_root,
    )
    print("\n[done] analysis finished.")


if __name__ == "__main__":
    main()
