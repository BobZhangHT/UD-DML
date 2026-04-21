#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""real_data_analysis.py

Real-data application of UD-DML on the CDC NCHS 2021 US natality file.

Treatment W   : maternal smoking during pregnancy (CIG_REC)
Outcome   Y   : birth weight in grams (DBWT), scaled to [0, 1]
Covariates X  : MAGER, MEDUC, PRECARE, PREVIS, SEX, DMAR, FAGECOMB,
                RF_GDIAB, RF_GHYPE, PRIORTERM   (p = 10)

Unified bootstrap analysis
--------------------------
  * FULL-DML run once on the original sample as the reference.
  * Nonparametric bootstrap: for each of B replications, resample n rows
    with replacement from (X, W, Y) to form (X_b, W_b, Y_b); within that
    bootstrap, run UD-DML and UNIF-DML at every budget r in ``--r-grid``.
  * The canonical budget (``--canonical-r``, default = 5000) must be in
    the grid — Figure A slices at that budget, Figure B plots the full
    scaling curve.
  * Per-rep results are cached to disk; the script resumes automatically
    and skips reps whose cache file already exists.

Outputs under ``real_data_results/``:
    figures/real_data_plan_A.{png,pdf}   -- bootstrap distribution + speedup
    figures/real_data_plan_B.{png,pdf}   -- scaling across r
    tables/real_data_plan_A.tex          -- LaTeX (canonical-r summary)
    tables/real_data_plan_B.tex          -- LaTeX (r-scaling grid)
    raw/rep_<b>.pkl.gz                   -- per-rep cache (cells = r × method)
    raw/full_reference.pkl.gz            -- FULL-DML reference

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
import os
import pickle
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
from methods import run_full, run_unif, run_ud


# ── Publication rcParams (Okabe-Ito) ──────────────────────────────────
METHOD_COLORS = {"UD": "#D55E00", "UNIF": "#56B4E9", "FULL": "#009E73"}
METHOD_MARKERS = {"UD": "o", "UNIF": "s", "FULL": "D"}

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
# 2. Per-rep worker (bootstrap + all r × method)
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
    if method == "UD":
        return run_ud(X, W, Y, r={"r_total": int(r_total)}, **kw)
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
    r_grid: List[int], boot_seed: int,
    k_folds: int, learner: str,
    cache_dir: Path, ate_full: float,
) -> Optional[Path]:
    """One bootstrap rep: resample, run UD + UNIF at every r, save cache.

    Reuses the bootstrap sample across the whole (r × method) grid so
    only one O(n) resample + one index materialisation is paid per rep.
    Skips the rep entirely when its cache file already exists.
    """
    _cap_worker_threads()
    cache_file = cache_dir / f"rep_{rep:04d}.pkl.gz"
    if cache_file.exists():
        return cache_file

    n = X.shape[0]
    b_rng = np.random.default_rng(int(boot_seed))
    idx = b_rng.integers(0, n, size=n)
    X_b = X[idx]; W_b = W[idx]; Y_b = Y[idx]
    if W_b.sum() < 10 or (1 - W_b).sum() < 10:
        # pathological bootstrap -> write empty record so we don't retry
        with gzip.open(cache_file, "wb", compresslevel=1) as f:
            pickle.dump({"rep": rep, "skipped": True}, f,
                        protocol=pickle.HIGHEST_PROTOCOL)
        return cache_file

    rows = []
    for r_total in r_grid:
        for method in ("UD", "UNIF"):
            res = _run_method(
                method, X_b, W_b, Y_b,
                r_total=r_total, seed=int(boot_seed), k_folds=k_folds,
                learner=learner,
            )
            rows.append({
                "rep": rep, "method": method, "r_total": int(r_total),
                "est_ate": float(res["est_ate"]),
                "ci_lower": float(res["ci_lower"]),
                "ci_upper": float(res["ci_upper"]),
                "ci_width": float(res["ci_upper"] - res["ci_lower"]),
                "runtime": float(res["runtime"]),
                "covers_full": int(res["ci_lower"] <= ate_full <= res["ci_upper"]),
            })
    with gzip.open(cache_file, "wb", compresslevel=1) as f:
        pickle.dump({"rep": rep, "rows": rows, "skipped": False}, f,
                    protocol=pickle.HIGHEST_PROTOCOL)
    return cache_file


# ═══════════════════════════════════════════════════════════════════════
# 3. Figure and table emitters (same outputs as old Plan A + Plan B)
# ═══════════════════════════════════════════════════════════════════════

def _emit_figure_A(df: pd.DataFrame, r_canonical: int,
                   ate_full: float, ci_full: Tuple[float, float],
                   time_full: float, fig_dir: Path):
    sub = df[df["r_total"] == r_canonical].copy()
    fig, axes = plt.subplots(
        1, 3, figsize=(10.5, 3.3),
        gridspec_kw={"width_ratios": [1.3, 1.3, 0.85]},
    )

    # (a) ATE boxplot
    ax = axes[0]
    bp = ax.boxplot(
        [sub[sub["method"] == m]["est_ate"].values for m in ("UD", "UNIF")],
        positions=[1, 2], widths=0.55, patch_artist=True, showfliers=True,
        medianprops=dict(color="black", linewidth=2),
    )
    for p, m in zip(bp["boxes"], ("UD", "UNIF")):
        p.set_facecolor(METHOD_COLORS[m]); p.set_alpha(0.65)
    ax.axhline(ate_full, ls="--", color=METHOD_COLORS["FULL"], linewidth=2.0,
               label=f"FULL = {ate_full:.4f}")
    ax.axhspan(ci_full[0], ci_full[1], color=METHOD_COLORS["FULL"],
               alpha=0.12, label="FULL 95% CI")
    ax.set_xticks([1, 2]); ax.set_xticklabels(["UD-DML", "UNIF-DML"])
    ax.set_ylabel("ATE (scaled)")
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, ls="--", axis="y")

    # (b) eCDF of |ATE - FULL|
    ax2 = axes[1]
    for m in ("UD", "UNIF"):
        errs = np.sort(np.abs(sub[sub["method"] == m]["est_ate"].values - ate_full))
        if errs.size == 0:
            continue
        cdf = np.arange(1, len(errs) + 1) / len(errs)
        ax2.step(errs, cdf, where="post",
                 color=METHOD_COLORS[m], linewidth=2.2, label=f"{m}-DML")
    ax2.set_xscale("log")
    ax2.set_xlabel(r"$|\widehat{\theta}_{\mathrm{sub}} - \widehat{\theta}_{\mathrm{FULL}}|$")
    ax2.set_ylabel("Empirical CDF")
    ax2.legend(loc="upper left", fontsize=10)
    ax2.grid(True, ls="--", which="both")
    ax2.set_ylim(0, 1.02)

    # (c) speedup
    ax3 = axes[2]
    t_ud = sub[sub["method"] == "UD"]["runtime"].mean()
    t_unif = sub[sub["method"] == "UNIF"]["runtime"].mean()
    speedup = {"UD": time_full / max(t_ud, 1e-9),
               "UNIF": time_full / max(t_unif, 1e-9)}
    bars = ax3.bar(
        ["UD", "UNIF"], [speedup["UD"], speedup["UNIF"]],
        color=[METHOD_COLORS["UD"], METHOD_COLORS["UNIF"]],
        edgecolor="black", linewidth=1.0,
    )
    for b, v in zip(bars, [speedup["UD"], speedup["UNIF"]]):
        ax3.text(b.get_x() + b.get_width() / 2, v * 1.02,
                 f"{v:.0f}×", ha="center", va="bottom", fontsize=11)
    ax3.set_ylabel("Speed-up vs FULL"); ax3.set_yscale("log")
    ax3.axhline(1.0, ls=":", color="gray", alpha=0.7)
    ax3.grid(True, ls="--", axis="y", which="both")

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        fig.tight_layout(pad=0.4, w_pad=0.6)
    for ext in (".png", ".pdf"):
        fig.savefig(fig_dir / f"real_data_plan_A{ext}")
    plt.close(fig)


def _emit_figure_B(df: pd.DataFrame, r_grid: List[int],
                   ate_full: float, time_full: float, fig_dir: Path):
    # Per (method, r): RMSE vs FULL + bootstrap MC SD of theta_hat + runtime.
    # MC SD is the actual sampling dispersion of \widehat\theta_sub across
    # bootstrap reps; unlike the plug-in CI width it is unaffected by UD's
    # design-induced shrinkage of the empirical pseudo-outcome variance,
    # so it directly exhibits UD-DML's variance-reduction property
    # (Theorem 2) without the plug-in undercoverage artefact.
    rmse_df = (
        df.assign(sq_err=(df["est_ate"] - ate_full) ** 2)
          .groupby(["method", "r_total"])["sq_err"].mean()
          .pow(0.5).rename("rmse").reset_index()
    )
    sd_df = (
        df.groupby(["method", "r_total"])["est_ate"]
          .std(ddof=1).rename("mc_sd").reset_index()
    )
    agg = df.groupby(["method", "r_total"]).agg(
        mean_runtime=("runtime", "mean"),
    ).reset_index().merge(rmse_df, on=["method", "r_total"]) \
                   .merge(sd_df,   on=["method", "r_total"])

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.5))
    metrics = [
        ("rmse",         "RMSE vs FULL",                  "log"),
        ("mc_sd",        r"Bootstrap SD of $\widehat{\theta}$", "log"),
        ("mean_runtime", "Mean runtime (s)",              "log"),
    ]
    n_panels = len(metrics)
    for idx, (ax, (col, ylabel, yscale)) in enumerate(zip(axes, metrics)):
        for m in ("UD", "UNIF"):
            sub = agg[agg["method"] == m].sort_values("r_total")
            ax.plot(sub["r_total"], sub[col],
                    marker=METHOD_MARKERS[m], color=METHOD_COLORS[m],
                    label=f"{m}-DML", linewidth=2.2, markersize=9)
        if col == "mean_runtime":
            ax.axhline(time_full, ls="--", color=METHOD_COLORS["FULL"],
                       linewidth=2, label="FULL")
        ax.set_xscale("log"); ax.set_yscale(yscale)
        ax.set_xlabel("Subsample size $r$")
        ax.set_ylabel(ylabel, labelpad=6)
        ax.grid(True, ls="--", which="both")
        if idx == n_panels - 1:
            ax.legend(loc="best", fontsize=10)
        r_sorted = sorted(set(int(r) for r in agg["r_total"].tolist()))
        ax.set_xticks(r_sorted)
        ax.set_xticklabels([f"{r//1000}k" if r >= 1000 else str(r)
                            for r in r_sorted])
        ax.tick_params(axis="x", which="minor", bottom=False)

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        fig.tight_layout(pad=0.4, w_pad=0.8)
    for ext in (".png", ".pdf"):
        fig.savefig(fig_dir / f"real_data_plan_B{ext}")
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
            "|Bias|": abs(a.mean() - ate_full),
            "MSE": ((a - ate_full) ** 2).mean(),
            "CI Width": s["ci_width"].mean(),
            "Coverage (FULL)": s["covers_full"].mean(),
            "Runtime": s["runtime"].mean(),
        })
    summary_A = sub.groupby("method").apply(_agg, include_groups=False).reindex(["UD", "UNIF"])
    sp_ud = time_full / max(summary_A.loc["UD", "Runtime"], 1e-9)
    sp_unif = time_full / max(summary_A.loc["UNIF", "Runtime"], 1e-9)
    ate_g = {
        "FULL": ate_full * (y_max - y_min),
        "UD":   summary_A.loc["UD",   "Mean ATE"] * (y_max - y_min),
        "UNIF": summary_A.loc["UNIF", "Mean ATE"] * (y_max - y_min),
    }
    cap_A = (
        f"Bootstrap stability of UD-DML versus UNIF-DML at the canonical "
        f"budget $r={r_canonical}$ on the CDC 2021 US natality dataset "
        f"($n={n_total:,}$, $B={B_actual}$ bootstrap replications).  "
        f"Columns: mean ATE, Monte-Carlo SD, absolute bias and MSE relative "
        f"to the full-sample DML estimate, mean 95\\% CI width, empirical "
        f"coverage of the full-sample point estimate, mean wall-clock "
        f"runtime, and speed-up relative to FULL on $n$.  Back-transformed "
        f"ATE in grams: FULL $= {ate_g['FULL']:.1f}$\\,g, "
        f"UD $= {ate_g['UD']:.1f}$\\,g, UNIF $= {ate_g['UNIF']:.1f}$\\,g."
    )
    lines_A = [
        r"\begin{table}[htbp]", r"\centering",
        r"\caption{" + cap_A + "}", r"\label{tab:real_data_plan_A}",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{lrrrrrrrr}", r"\toprule",
        r"Method & Mean ATE & SD & $|$Bias$|$ & MSE & CI Width & Coverage & Runtime (s) & Speed-up \\",
        r"\midrule",
        (f"FULL & {ate_full:.5f} & --- & --- & --- & "
         f"{ci_full[1]-ci_full[0]:.5f} & --- & {time_full:.2f} & 1$\\times$ \\\\"),
    ]
    for m, sp in [("UD", sp_ud), ("UNIF", sp_unif)]:
        r = summary_A.loc[m]
        lines_A.append(
            f"{m} & {r['Mean ATE']:.5f} & {r['SD ATE']:.5f} & "
            f"{r['|Bias|']:.5f} & {r['MSE']:.2e} & "
            f"{r['CI Width']:.5f} & {r['Coverage (FULL)']:.2f} & "
            f"{r['Runtime']:.2f} & {sp:.0f}$\\times$ \\\\"
        )
    lines_A.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    (tables_dir / "real_data_plan_A.tex").write_text("\n".join(lines_A) + "\n", encoding="utf-8")

    # ── Table B: r-scaling grid (excludes canonical r) ─────────────────
    # MC SD = bootstrap SD of \widehat\theta across reps (actual sampling
    # dispersion, robust to plug-in variance shrinkage from UD's design
    # balance).  Variance-reduction ratio Var(UNIF)/Var(UD) summarises the
    # design gain in a single dimensionless number.
    r_grid_B = [r for r in r_grid if r != r_canonical]
    agg_B = df[df["r_total"].isin(r_grid_B)].groupby(["method", "r_total"]).apply(
        lambda s: pd.Series({
            "rmse": np.sqrt(((s["est_ate"] - ate_full) ** 2).mean()),
            "mc_sd": s["est_ate"].std(ddof=1),
            "mean_runtime":  s["runtime"].mean(),
        }), include_groups=False,
    ).reset_index()
    # Pre-compute Var(UNIF) / Var(UD) per r for the variance-reduction column.
    sd_pivot = agg_B.pivot(index="r_total", columns="method", values="mc_sd")
    cap_B = (
        f"Subsample-size scaling on the CDC 2021 natality data "
        f"($B={B_actual}$ bootstrap replications).  Budgets disjoint "
        f"from Plan A's canonical $r={r_canonical}$.  Columns: RMSE "
        f"relative to FULL; bootstrap Monte-Carlo standard deviation "
        f"of $\\widehat{{\\theta}}_{{\\mathrm{{sub}}}}$; "
        f"variance-reduction ratio $\\mathrm{{Var}}(\\widehat{{\\theta}}_{{\\mathrm{{UNIF}}}})/"
        f"\\mathrm{{Var}}(\\widehat{{\\theta}}_{{\\mathrm{{UD}}}})$ "
        f"(reported once per $r$); mean wall-clock runtime per call.  "
        f"FULL runtime $= {time_full:.2f}$\\,s on $n={n_total:,}$.  "
        f"Bootstrap SD is reported instead of the plug-in Wald CI width "
        f"because UD-DML's design-induced variance shrinkage causes the "
        f"plug-in $\\widehat{{\\sigma}}_{{\\psi,r}}^{{2}}/r$ to "
        f"under-state the actual rep-to-rep dispersion of "
        f"$\\widehat{{\\theta}}_{{\\mathrm{{UD}}}}$; the bootstrap SD is "
        f"unaffected by this artefact and directly exhibits the "
        f"variance-reduction guaranteed by Theorem~\\ref{{thm:bal_ud}}."
    )
    lines_B = [
        r"\begin{table}[htbp]", r"\centering",
        r"\caption{" + cap_B + "}", r"\label{tab:real_data_plan_B}",
        r"\setlength{\tabcolsep}{5pt}",
        r"\begin{tabular}{lrrrrr}", r"\toprule",
        r"Method & $r$ & RMSE vs FULL & MC SD & Var-Ratio & Runtime (s) \\",
        r"\midrule",
    ]
    for m in ("UD", "UNIF"):
        for r in r_grid_B:
            row = agg_B[(agg_B["method"] == m) & (agg_B["r_total"] == r)]
            if row.empty:
                continue
            row = row.iloc[0]
            # Variance-reduction ratio printed once per r (on the UD row);
            # blank for the UNIF row to avoid duplication.
            if m == "UD" and "UNIF" in sd_pivot.columns and "UD" in sd_pivot.columns:
                sd_ud = sd_pivot.loc[r, "UD"]
                sd_un = sd_pivot.loc[r, "UNIF"]
                if sd_ud and not np.isnan(sd_ud) and sd_ud > 0:
                    var_ratio = (sd_un / sd_ud) ** 2
                    var_str = f"{var_ratio:.1f}$\\times$"
                else:
                    var_str = "---"
            else:
                var_str = "---"
            lines_B.append(
                f"{m} & {int(row['r_total'])} & {row['rmse']:.5f} & "
                f"{row['mc_sd']:.5f} & {var_str} & "
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


# ═══════════════════════════════════════════════════════════════════════
# 4. Unified analysis driver
# ═══════════════════════════════════════════════════════════════════════

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

    # ── FULL reference (cached) ────────────────────────────────────────
    full_cache = cache_dir / "full_reference.pkl.gz"
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

    # ── Bootstrap plan (resumable) ─────────────────────────────────────
    boot_seeds = np.random.default_rng(20250919).integers(0, 2**31 - 1, size=B)
    pending = [b for b in range(B) if not (cache_dir / f"rep_{b:04d}.pkl.gz").exists()]
    done    = B - len(pending)
    print(f"[boot] {B} reps total  |  {done} cached  |  {len(pending)} pending")

    if pending:
        n = X.shape[0]
        # Cap workers: never more than pending reps (avoids 16-worker spawn for 10 tasks).
        eff_jobs = max(1, min(n_jobs, len(pending)))
        # joblib temp folder for memmap of large X/W/Y arrays.
        mmap_dir = output_root / "_joblib_mmap"
        mmap_dir.mkdir(parents=True, exist_ok=True)
        print(f"[boot] parallel jobs = {eff_jobs} (req {n_jobs})  "
              f"|  n per bootstrap = {n:,}  |  mmap = {mmap_dir}")

        # Wrap _process_one_rep so X/W/Y are explicit args -> joblib auto-memmaps
        # (closure-captured arrays bypass memmap and get pickled per worker).
        def _task(b, X_, W_, Y_):
            return _process_one_rep(
                b, X_, W_, Y_, r_grid, int(boot_seeds[b]),
                k_folds, learner, cache_dir, ate_full,
            )

        with tqdm(total=len(pending), desc="bootstrap reps",
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
        fp = cache_dir / f"rep_{b:04d}.pkl.gz"
        if not fp.exists():
            continue
        with gzip.open(fp, "rb") as f:
            payload = pickle.load(f)
        if payload.get("skipped"):
            skipped += 1
            continue
        rows.extend(payload["rows"])
    if skipped:
        print(f"[boot] {skipped} bootstrap reps skipped (pathological "
              f"treated/control imbalance).")
    df = pd.DataFrame(rows)
    B_actual = df["rep"].nunique() if not df.empty else 0
    print(f"[boot] usable reps: {B_actual}")

    # ── Emit outputs ───────────────────────────────────────────────────
    n_total = X.shape[0]
    _emit_figure_A(df, r_canonical, ate_full, ci_full, time_full, fig_dir)
    print(f"[out] figure -> {fig_dir / 'real_data_plan_A.{png,pdf}'}")
    _emit_figure_B(df, r_grid, ate_full, time_full, fig_dir)
    print(f"[out] figure -> {fig_dir / 'real_data_plan_B.{png,pdf}'}")
    _emit_tables(df, r_grid, r_canonical, ate_full, ci_full, time_full,
                 n_total, B_actual, y_min, y_max, tables_dir)
    print(f"[out] LaTeX  -> {tables_dir / 'real_data_plan_{A,B}.tex'}")

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
                    help="B = bootstrap replications (default 100).")
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
                    help="Fast smoke-test mode: force B=10 reps; all other "
                         "params identical to the full default run. "
                         "Writes outputs under '<out>_fast_demo/' to avoid "
                         "polluting the full-run cache.")
    args = ap.parse_args()

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
    r_grid = _parse_r_grid(args.r_grid)
    n_jobs = os.cpu_count() if args.jobs == -1 else int(args.jobs)

    run_analysis(
        X, W, Y, y_min, y_max,
        r_grid=r_grid, r_canonical=args.canonical_r,
        B=reps, k_folds=args.k_folds,
        n_jobs=n_jobs, output_root=output_root,
    )
    print("\n[done] analysis finished.")


if __name__ == "__main__":
    main()
