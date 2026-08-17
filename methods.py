# -*- coding: utf-8 -*-
"""
methods.py — Core estimators for the UD-DML framework.

Implements Algorithm 1 from:

    Qu, Xu & Zhang (2026). "UD-DML: Uniform Design Subsampling for
    Double Machine Learning over Massive Data."

The module provides five estimators for the Average Treatment Effect (ATE),
including two reviewer-facing component baselines:

    * ``run_full``  — Full-data cross-fitted DML (reference estimator).
    * ``run_unif``  — Naive uniform random subsampling + DML (benchmark).
    * ``run_ud``    — Uniform Design subsampling + DML (proposed method).

    * ``run_stratified_unif``: treatment-balanced random subsampling.
    * ``run_sep_ud``: separate-arm uniform-design subsampling.

Algorithm overview (UD-DML)
===========================

**Phase 1 — UD subsampling in the retained PCA-rotated space**

    1. Standardise covariates X̃ = D̂⁻¹(X − X̄).
    2. SVD → retain the first *q* principal directions capturing ≥ ρ₀
       of the total variance.  Compute rotated covariates Z = Vq⊤ X̃.
    3. Construct a low-discrepancy skeleton {u_j} ⊂ [0,1]^q via the
       leave-one-out good lattice point (GLP) method with a power
       generator, selecting the design that minimises the mixture
       discrepancy D²_M.
    4. Map the skeleton to the rotated space through the marginal
       empirical inverse CDFs:  v_j = F̂_Z⁻¹(u_j).
    5. For each skeleton point v_j, select one treated and one control unit
       in Z-space by one-to-one nearest-neighbour assignment.  Candidate
       neighbours are obtained from exact ``cKDTree`` queries and each
       original observation can be selected at most once.

**Phase 2 — Cross-fitted DML on the selected original observations**

    6. Standard K-fold cross-fitting on {(Y_i, W_i, X_i) : i ∈ S}.
    7. Compute the AIPW pseudo-outcomes.

**Phase 3 — Estimation and Wald inference**

    8. Point estimate: θ̂ = (1/r) Σ ψ̂*.
    9. For outcome-blind balanced selections, estimate the conditional
       variance from the cross-fitted inverse-propensity residual component.
   10. Form the corresponding Wald confidence interval.
"""

from __future__ import annotations

import math
import os
import tempfile
import time
import warnings
import zlib
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LassoCV, LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

try:
    import lightgbm as lgb

    _HAS_LIGHTGBM = True
except ImportError:
    lgb = None  # type: ignore[assignment]
    _HAS_LIGHTGBM = False
    from sklearn.ensemble import (
        GradientBoostingClassifier,
        GradientBoostingRegressor,
    )

import config

# Cache for optimal GLP skeleton U* in [0,1]^q (budgeted search, reusable across calls).
_UD_SKELETON_CACHE: Dict[Tuple[int, int, int, int], np.ndarray] = {}

# Per-worker LRU (size 1) for UD-selected indices.  Keyed by Python id(X, W)
# plus subsampling parameters.  Saves ~1-2 s per call on n=5e5 when multiple
# tasks (e.g. 4 misspec variants in double_robust) share the same DGP sample.
_UD_INDICES_CACHE: Optional[Tuple[Tuple, np.ndarray]] = None

# Increment whenever the semantics of the selected-index cache change.  Version
# 2 replaces the original with-replacement 1-NN lookup by one-to-one matching.
_UD_MATCHING_VERSION: int = 2

# ---------------------------------------------------------------------------
# Optional compiled C backend for the GLP search.  Enabled by default when
# ``genUD.dll`` / ``libgenUD.so`` is present next to this file; falls back
# transparently to the pure-Python loop if the library cannot be loaded, or
# if the user opts out via ``UD_USE_C_BACKEND=0``.
# ---------------------------------------------------------------------------
_UD_USE_C_ENV = os.environ.get("UD_USE_C_BACKEND", "").strip().lower()
_UD_C_DISABLED = _UD_USE_C_ENV in ("0", "false", "off", "no")

try:
    import genUD_wrapper as _genUD  # noqa: E402

    _UD_C_AVAILABLE = (not _UD_C_DISABLED) and _genUD.c_genUD_available()
except Exception as _ud_c_exc:  # pragma: no cover — graceful fallback
    _genUD = None  # type: ignore[assignment]
    _UD_C_AVAILABLE = False


def ud_c_backend_active() -> bool:
    """Return True iff the compiled C backend is loaded and enabled."""
    return bool(_UD_C_AVAILABLE)


def _ud_skeleton_disk_cache_root() -> Optional[Path]:
    """Resolve optional on-disk skeleton store (see ``config.UD_SKELETON_DISK_CACHE_DIR``).

    Environment ``UD_SKELETON_DISK_CACHE`` overrides the config directory when set
    to a non-empty path; values ``0`` / ``false`` / ``off`` / ``none`` disable.
    """
    env = os.environ.get("UD_SKELETON_DISK_CACHE")
    if env is not None:
        s = env.strip()
        if s.lower() in ("0", "false", "off", "none", ""):
            return None
        return Path(s).expanduser().resolve()
    root = getattr(config, "UD_SKELETON_DISK_CACHE_DIR", None)
    if root is None:
        return None
    return Path(root).expanduser().resolve()


def _ud_skeleton_disk_path(root: Path, key: Tuple[int, int, int, int]) -> Path:
    r_p, q, B_gamma, cache_seed = key
    name = f"ud_r{r_p}_q{q}_Bg{B_gamma}_seed{cache_seed}.npy"
    return root / name


def _try_load_ud_skeleton_npy(path: Path, r_p: int, q: int) -> Optional[np.ndarray]:
    if not path.is_file():
        return None
    try:
        U = np.load(path, allow_pickle=False)
    except (OSError, ValueError):
        return None
    if not isinstance(U, np.ndarray) or U.shape != (r_p, q):
        return None
    return U.astype(np.float64, copy=False)


def _atomic_save_ud_skeleton_npy(path: Path, U: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(suffix=".npy", dir=str(path.parent))
    os.close(fd)
    tmp_path = Path(tmp)
    try:
        np.save(tmp_path, U.astype(np.float64, copy=False), allow_pickle=False)
        os.replace(tmp_path, path)
    except BaseException:
        try:
            tmp_path.unlink()
        except OSError:
            pass
        raise

# ---------------------------------------------------------------------------
# Silence non-critical sklearn / lightgbm warnings
# ---------------------------------------------------------------------------
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", message=".*feature names.*")

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------
_CI_Z: float = 1.96  # z_{0.975} for 95 % Wald intervals


# ═══════════════════════════════════════════════════════════════════════════
# Section 1 — AIPW Score and Wald Inference
# ═══════════════════════════════════════════════════════════════════════════


def _aipw_score(
    Y: np.ndarray,
    W: np.ndarray,
    mu0: np.ndarray,
    mu1: np.ndarray,
    e: np.ndarray,
) -> np.ndarray:
    """Compute the Augmented Inverse Propensity Weighting (AIPW) pseudo-outcome.

    Implements Equation (1) of the paper (with θ = 0):

        ψ*(O; η) = (m₁(X) − m₀(X))
                    + W·(Y − m₁(X)) / e(X)
                    − (1−W)·(Y − m₀(X)) / (1−e(X))

    Parameters
    ----------
    Y : ndarray of shape (n,)
        Observed outcomes.
    W : ndarray of shape (n,)
        Binary treatment indicators (0 or 1).
    mu0 : ndarray of shape (n,)
        Predicted conditional outcome E[Y | X, W=0].
    mu1 : ndarray of shape (n,)
        Predicted conditional outcome E[Y | X, W=1].
    e : ndarray of shape (n,)
        Predicted propensity scores P(W=1 | X), clipped away from 0 and 1.

    Returns
    -------
    ndarray of shape (n,)
        AIPW pseudo-outcomes.
    """
    return (
        (mu1 - mu0)
        + W * (Y - mu1) / e
        - (1.0 - W) * (Y - mu0) / (1.0 - e)
    )


def _wald_inference(scores: np.ndarray) -> Tuple[float, float, float]:
    """Wald point estimate and 95 % confidence interval from pseudo-outcomes.

    Implements Phase 3 of Algorithm 1:
        θ̂ = (1/r) Σ ψ̂*_i
        σ̂² = (1/(r−1)) Σ (ψ̂*_i − θ̂)²
        CI  = θ̂ ± z_{0.975} · √(σ̂²/r)

    Parameters
    ----------
    scores : ndarray of shape (r,)
        Cross-fitted AIPW pseudo-outcomes on the (sub)sample.

    Returns
    -------
    est_ate : float
        Point estimate of the ATE.
    ci_lower : float
        Lower bound of the 95 % confidence interval.
    ci_upper : float
        Upper bound of the 95 % confidence interval.
    """
    r = scores.shape[0]
    if r <= 1:
        return np.nan, np.nan, np.nan
    est_ate = float(np.mean(scores))
    se = float(np.std(scores, ddof=1) / np.sqrt(r))
    return est_ate, est_ate - _CI_Z * se, est_ate + _CI_Z * se


def _selected_residual_inference(
    scores: np.ndarray,
    Y: np.ndarray,
    W: np.ndarray,
    mu0: np.ndarray,
    mu1: np.ndarray,
    e: np.ndarray,
) -> Tuple[float, float, float, float, float]:
    """Conditional root-r inference for the UD-selected score estimator.

    Conditional on the realised selected covariates and treatments, the
    leading stochastic term is the inverse-propensity weighted outcome
    residual.  Its uncentred second moment estimates the conditional
    triangular-array variance in Theorem 3 of the revised manuscript.
    """
    r = int(scores.shape[0])
    if r <= 1:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    residual_scores = (
        W * (Y - mu1) / e
        - (1.0 - W) * (Y - mu0) / (1.0 - e)
    )
    est_ate = float(np.mean(scores))
    residual_variance = float(np.mean(np.square(residual_scores)))
    standard_error = float(np.sqrt(residual_variance / r))
    return (
        est_ate,
        est_ate - _CI_Z * standard_error,
        est_ate + _CI_Z * standard_error,
        standard_error,
        residual_variance,
    )


def _covariate_smd_diagnostics(X: np.ndarray, W: np.ndarray) -> Dict[str, Any]:
    """Summarise treated-control standardized mean differences.

    The denominator is the square root of the equally weighted within-arm
    variances.  Constant coordinates with equal arm means are assigned zero;
    constant coordinates with unequal means are assigned infinity.
    """
    X_arr = np.asarray(X, dtype=np.float64)
    W_arr = np.asarray(W)
    x1 = X_arr[W_arr == 1]
    x0 = X_arr[W_arr == 0]
    if x1.shape[0] < 2 or x0.shape[0] < 2:
        return {
            "smd_mean": np.nan,
            "smd_max": np.nan,
            "smd_count_above_0p1": np.nan,
        }
    numerator = np.abs(np.mean(x1, axis=0) - np.mean(x0, axis=0))
    denominator = np.sqrt(
        0.5 * (np.var(x1, axis=0, ddof=1) + np.var(x0, axis=0, ddof=1))
    )
    smd = np.divide(
        numerator,
        denominator,
        out=np.where(numerator == 0.0, 0.0, np.inf),
        where=denominator > np.finfo(np.float64).eps,
    )
    return {
        "smd_mean": float(np.mean(smd)),
        "smd_max": float(np.max(smd)),
        "smd_count_above_0p1": int(np.sum(smd > 0.1)),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Section 2 — Cross-fitted Nuisance Estimation (Phase 2)
# ═══════════════════════════════════════════════════════════════════════════


def _fit_nuisance_models(
    X: np.ndarray,
    W: np.ndarray,
    Y: np.ndarray,
    k_folds: int,
    is_rct: bool,
    pi_rct_val: Optional[float] = None,
    misspecification: Optional[str] = None,
    learner: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """K-fold cross-fitted estimation of nuisance functions (m₀, m₁, e).

    For each fold *k*, the nuisance models are trained on all data outside
    fold *k* and evaluated on fold *k*, following the standard DML cross-
    fitting protocol (Chernozhukov et al., 2018).

    Parameters
    ----------
    X : ndarray of shape (n, p)
        Covariate matrix.
    W : ndarray of shape (n,)
        Binary treatment assignments.
    Y : ndarray of shape (n,)
        Observed outcomes.
    k_folds : int
        Number of cross-fitting folds (typically K=2).
    is_rct : bool
        If True, propensity scores are set to a constant (RCT design).
    pi_rct_val : float or None
        Constant propensity score for RCT designs.
    misspecification : str or None
        One of ``'correct_correct'``, ``'correct_wrong'``,
        ``'wrong_correct'``, ``'wrong_wrong'`` for the double-robustness
        experiment (Section 3.3, Experiment 3).
    learner : str or None
        Nuisance learner identifier: ``'lgbm'`` (default), ``'rf'``, or
        ``'lasso_cv'``.

    Returns
    -------
    mu0 : ndarray of shape (n,)
    mu1 : ndarray of shape (n,)
    e   : ndarray of shape (n,)
    """
    n = Y.shape[0]
    mu0_hat = np.zeros(n)
    mu1_hat = np.zeros(n)
    e_hat = np.zeros(n)

    learner = (learner or getattr(config, "DEFAULT_NUISANCE_LEARNER", "lgbm")).lower()
    if learner in ("lasso", "lassocv"):
        learner = "lasso_cv"

    kf = KFold(n_splits=k_folds, shuffle=True, random_state=config.BASE_SEED)

    for train_idx, test_idx in kf.split(X):
        X_tr, Y_tr, W_tr = X[train_idx], Y[train_idx], W[train_idx]
        X_te = X[test_idx]

        # ── Outcome models m̂₀ and m̂₁ ─────────────────────────────────
        mu0_hat[test_idx], mu1_hat[test_idx] = _fit_outcome_models(
            X_tr, Y_tr, W_tr, X_te, learner, misspecification,
        )

        # ── Propensity score ê ────────────────────────────────────────
        if is_rct:
            e_hat[test_idx] = pi_rct_val if pi_rct_val is not None else np.mean(W_tr)
        else:
            e_hat[test_idx] = _fit_propensity_model(
                X_tr, W_tr, X_te, learner, misspecification,
            )

    # Clip propensity scores to [0.01, 0.99] for numerical stability
    # (Section 3.2 of the paper).
    np.clip(e_hat, 0.01, 0.99, out=e_hat)
    return mu0_hat, mu1_hat, e_hat


def _fit_outcome_models(
    X_tr: np.ndarray,
    Y_tr: np.ndarray,
    W_tr: np.ndarray,
    X_te: np.ndarray,
    learner: str,
    misspecification: Optional[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit and predict conditional outcome models m̂₀(x) and m̂₁(x).

    Unified misspecification scheme for the Section 3.3 double-robustness
    experiment: whenever the outcome model is marked as ``wrong``
    (both in ``'wrong_correct'`` and in ``'wrong_wrong'``) we use the
    same wrong specification, namely *arm-specific linear regression of
    Y on the single covariate X^{(5)}*.  X^{(5)} enters neither the
    outcome mechanism g nor the CATE Δ nor the propensity e of any of
    OBS-1, OBS-2, or OBS-3, so this amounts to an arm-specific
    intercept-only model with a spurious regressor.  Under (wrong,
    correct), DR is preserved via the propensity path; under (wrong,
    wrong), AIPW collapses to the subsample difference-in-means.
    """
    if misspecification in ("wrong_correct", "wrong_wrong"):
        from sklearn.linear_model import LinearRegression
        idx = [4]  # X^{(5)}: never used in any DGP's outcome or propensity
        lr0 = LinearRegression().fit(X_tr[W_tr == 0][:, idx], Y_tr[W_tr == 0])
        lr1 = LinearRegression().fit(X_tr[W_tr == 1][:, idx], Y_tr[W_tr == 1])
        return lr0.predict(X_te[:, idx]), lr1.predict(X_te[:, idx])

    return _fit_outcome_pair(X_tr, Y_tr, W_tr, X_te, learner)


def _fit_outcome_pair(
    X_tr: np.ndarray,
    Y_tr: np.ndarray,
    W_tr: np.ndarray,
    X_te: np.ndarray,
    learner: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit correctly-specified outcome models using the chosen learner."""
    mask0, mask1 = W_tr == 0, W_tr == 1

    if learner == "rf":
        params = dict(
            n_estimators=getattr(config, "RF_N_TREES", 100),
            random_state=config.BASE_SEED,
            n_jobs=getattr(config, "RF_N_JOBS", 1),
        )
        rf0 = RandomForestRegressor(**params).fit(X_tr[mask0], Y_tr[mask0])
        rf1 = RandomForestRegressor(**params).fit(X_tr[mask1], Y_tr[mask1])
        return rf0.predict(X_te), rf1.predict(X_te)

    if learner == "lasso_cv":
        cv = getattr(config, "LASSO_CV_FOLDS", 5)
        max_iter = getattr(config, "LASSO_CV_MAX_ITER", 5000)
        preds = []
        for mask in (mask0, mask1):
            scaler = StandardScaler().fit(X_tr[mask])
            model = LassoCV(
                cv=cv, n_jobs=1,
                random_state=config.BASE_SEED, max_iter=max_iter,
            ).fit(scaler.transform(X_tr[mask]), Y_tr[mask])
            preds.append(model.predict(scaler.transform(X_te)))
        return preds[0], preds[1]

    # Default: LightGBM (or sklearn GradientBoosting fallback)
    if _HAS_LIGHTGBM:
        params = dict(
            n_estimators=getattr(config, "LGBM_N_ESTIMATORS", 100),
            max_depth=getattr(config, "LGBM_MAX_DEPTH", 5),
            learning_rate=getattr(config, "LGBM_LEARNING_RATE", 0.1),
            num_leaves=getattr(config, "LGBM_NUM_LEAVES", 31),
            random_state=config.BASE_SEED,
            verbose=-1,
            n_jobs=1,
        )
        m0 = lgb.LGBMRegressor(**params).fit(X_tr[mask0], Y_tr[mask0])
        m1 = lgb.LGBMRegressor(**params).fit(X_tr[mask1], Y_tr[mask1])
    else:
        params = dict(
            n_estimators=getattr(config, "LGBM_N_ESTIMATORS", 100),
            max_depth=getattr(config, "LGBM_MAX_DEPTH", 5),
            learning_rate=getattr(config, "LGBM_LEARNING_RATE", 0.1),
            random_state=config.BASE_SEED,
        )
        m0 = GradientBoostingRegressor(**params).fit(X_tr[mask0], Y_tr[mask0])
        m1 = GradientBoostingRegressor(**params).fit(X_tr[mask1], Y_tr[mask1])
    return m0.predict(X_te), m1.predict(X_te)


def _fit_propensity_model(
    X_tr: np.ndarray,
    W_tr: np.ndarray,
    X_te: np.ndarray,
    learner: str,
    misspecification: Optional[str],
) -> np.ndarray:
    """Fit and predict the propensity score ê(x) = P̂(W=1|X=x).

    Unified misspecification scheme for the Section 3.3 double-robustness
    experiment: whenever the propensity model is marked as ``wrong``
    (both in ``'correct_wrong'`` and in ``'wrong_wrong'``) we use the
    same wrong specification, namely *logistic regression of W on the
    single covariate X^{(5)}*.  X^{(5)} enters no DGP's propensity, so
    this reduces to a near-constant fit at the marginal treatment rate.
    Under (correct, wrong), DR is preserved via the correctly specified
    outcome path; under (wrong, wrong), AIPW collapses to the subsample
    difference-in-means.
    """
    if misspecification in ("correct_wrong", "wrong_wrong"):
        idx = [4]  # X^{(5)}: never used in any DGP's propensity
        lr = LogisticRegression(solver="liblinear", max_iter=1000)
        lr.fit(X_tr[:, idx], W_tr)
        return lr.predict_proba(X_te[:, idx])[:, 1]

    if learner == "rf":
        clf = RandomForestClassifier(
            n_estimators=getattr(config, "RF_N_TREES", 100),
            random_state=config.BASE_SEED,
            n_jobs=getattr(config, "RF_N_JOBS", 1),
        ).fit(X_tr, W_tr)
        return clf.predict_proba(X_te)[:, 1]

    if learner == "lasso_cv":
        scaler = StandardScaler().fit(X_tr)
        clf = LogisticRegressionCV(
            Cs=getattr(config, "LOGIT_CV_CS", None),
            cv=getattr(config, "LASSO_CV_FOLDS", 5),
            penalty="l1",
            solver="saga",
            scoring=getattr(config, "LOGIT_CV_SCORING", "neg_log_loss"),
            max_iter=getattr(config, "LOGIT_CV_MAX_ITER", 5000),
            random_state=config.BASE_SEED,
            n_jobs=1,
        ).fit(scaler.transform(X_tr), W_tr)
        return clf.predict_proba(scaler.transform(X_te))[:, 1]

    # Default: LightGBM (or sklearn GradientBoosting fallback)
    if _HAS_LIGHTGBM:
        clf = lgb.LGBMClassifier(
            n_estimators=getattr(config, "LGBM_N_ESTIMATORS", 100),
            max_depth=getattr(config, "LGBM_MAX_DEPTH", 5),
            learning_rate=getattr(config, "LGBM_LEARNING_RATE", 0.1),
            num_leaves=getattr(config, "LGBM_NUM_LEAVES", 31),
            random_state=config.BASE_SEED,
            verbose=-1,
            n_jobs=1,
        ).fit(X_tr, W_tr)
    else:
        clf = GradientBoostingClassifier(
            n_estimators=getattr(config, "LGBM_N_ESTIMATORS", 100),
            max_depth=getattr(config, "LGBM_MAX_DEPTH", 5),
            learning_rate=getattr(config, "LGBM_LEARNING_RATE", 0.1),
            random_state=config.BASE_SEED,
        ).fit(X_tr, W_tr)
    return clf.predict_proba(X_te)[:, 1]


# ═══════════════════════════════════════════════════════════════════════════
# Section 3 — Uniform Design Subsampling (Phase 1 of Algorithm 1)
# ═══════════════════════════════════════════════════════════════════════════
#
# The pipeline below faithfully implements Section 2.2 of the paper:
#
#   Step 1: Standardise → PCA → retain q dimensions (ρ₀ threshold).
#   Step 2: Good lattice point skeleton in [0,1]^q, optimised via
#           mixture discrepancy.
#   Step 3: Empirical inverse CDF mapping → skeleton in Z-space.
#   Step 4: Paired exact 1-NN matching (treated + control) in Z-space
#           without replacement via one-to-one assignment (Algorithm 1).
#


def _standardise_covariates(X: np.ndarray) -> np.ndarray:
    """Standardise covariates: X̃_i = D̂⁻¹ (X_i − X̄).

    Algorithm 1, Step 2.

    Parameters
    ----------
    X : ndarray of shape (n, p)
        Raw covariate matrix.

    Returns
    -------
    X_tilde : ndarray of shape (n, p)
        Column-centred and column-scaled covariate matrix.
    """
    mean = X.mean(axis=0)
    std = X.std(axis=0, ddof=0)
    # Guard against zero-variance columns.
    std[std < 1e-12] = 1.0
    return (X - mean) / std


def _pca_rotate(
    X_tilde: np.ndarray,
    rho_0: float,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Perform PCA and retain the first *q* components (Algorithm 1, Step 3).

    Computes the SVD of the standardised matrix X̃ = U Σ V⊤, and selects
    the smallest *q* such that the cumulative proportion of variance
    (Σ_{d=1}^{q} σ²_d) / (Σ_{d=1}^{p} σ²_d)  ≥  ρ₀.

    Parameters
    ----------
    X_tilde : ndarray of shape (n, p)
        Standardised covariate matrix.
    rho_0 : float
        Cumulative variance threshold, typically 0.85.

    Returns
    -------
    Z : ndarray of shape (n, q)
        Retained PCA-rotated covariates: Z_i = V_q⊤ X̃_i.
    V_q : ndarray of shape (p, q)
        Matrix of retained right-singular vectors.
    q : int
        Number of retained components.
    """
    n, p = X_tilde.shape
    # Fast path for n >> p (typical for real-data application): eigendecompose
    # the p×p sample covariance.  np.linalg.svd on (n, p) is dominated by an
    # internal QR + bidiagonalisation that scales poorly in n; the eigh route
    # is O(n·p²) for the gram and O(p³) for eigendecomp, ~10× faster at
    # n=2.8M, p=10.
    if p <= 50:
        gram = X_tilde.T @ X_tilde                    # (p, p)
        eigvals, eigvecs = np.linalg.eigh(gram)       # ascending
        order = np.argsort(eigvals)[::-1]
        eigvals = np.maximum(eigvals[order], 0.0)
        eigvecs = eigvecs[:, order]
        total = max(eigvals.sum(), 1e-12)
        cumulative_ratio = np.cumsum(eigvals) / total
        q = int(np.searchsorted(cumulative_ratio, rho_0) + 1)
        q = min(q, p)
        V_q = eigvecs[:, :q]
        Z = X_tilde @ V_q
        return Z, V_q, q

    # Fallback: economy SVD for large p.
    _, S, Vt = np.linalg.svd(X_tilde, full_matrices=False)
    var_explained = S ** 2
    cumulative_ratio = np.cumsum(var_explained) / var_explained.sum()
    q = int(np.searchsorted(cumulative_ratio, rho_0) + 1)
    q = min(q, p)
    V_q = Vt[:q, :].T
    Z = X_tilde @ V_q
    return Z, V_q, q


def _marginal_empirical_cdf_ranks(Z: np.ndarray) -> np.ndarray:
    """Compute per-column ranks (scaled to [0, 1]) of the rotated covariates.

    For each dimension d, the empirical CDF is:
        F̂_{Z^{(d)}}(z) = (rank of z among {Z_i^{(d)}}) / n.

    We store argsorts so that the inverse CDF lookup in
    ``_map_skeleton_to_rotated_space`` is O(1) per query.

    Parameters
    ----------
    Z : ndarray of shape (n, q)
        PCA-rotated covariates.

    Returns
    -------
    Z_sorted : ndarray of shape (n, q)
        Column-wise sorted Z values (the order statistics).
    """
    return np.sort(Z, axis=0)


def _map_skeleton_to_rotated_space(
    U: np.ndarray,
    Z_sorted: np.ndarray,
) -> np.ndarray:
    """Map skeleton points from [0,1]^q to the rotated Z-space (Step 12).

    For each design coordinate u_{jd}, the empirical inverse CDF is:
        v_{jd} = F̂⁻¹_{Z^{(d)}}(u_{jd}) ≈ Z_sorted[⌈u·n⌉ − 1, d].

    Parameters
    ----------
    U : ndarray of shape (r_p, q)
        Uniform design skeleton points in [0, 1]^q.
    Z_sorted : ndarray of shape (n, q)
        Column-wise sorted rotated covariates.

    Returns
    -------
    V : ndarray of shape (r_p, q)
        Skeleton points mapped into the rotated covariate space.
    """
    n = Z_sorted.shape[0]
    U_clipped = np.clip(U, 0.0, 1.0)
    row_idx = np.clip(np.ceil(U_clipped * n).astype(np.intp) - 1, 0, n - 1)
    return np.take_along_axis(Z_sorted, row_idx, axis=0)


# ── Good Lattice Point (GLP) Uniform Design ──────────────────────────────


def _find_admissible_generators(
    r_p: int,
    q: int,
    B_gamma: int,
    rng: Optional[np.random.Generator] = None,
) -> list[int]:
    """Enumerate or subsample admissible power generators for the GLP construction.

    A positive integer α is *admissible* if gcd(α, r_p + 1) = 1 and the
    remainders  α⁰, α¹, …, α^{q−1}  (mod r_p + 1)  are mutually distinct
    (Section 2.2 of the paper).

    When the number of admissible generators exceeds ``B_gamma``, a random
    subset of size ``B_gamma`` is drawn and searched (budgeted quasi-optimal
    search).

    Parameters
    ----------
    r_p : int
        Number of skeleton pairs.
    q : int
        Working dimension (number of retained PCA components).
    B_gamma : int
        Maximum number of generator candidates to evaluate (``B_γ``).
    rng : Generator or None
        Random number generator for subset sampling.

    Returns
    -------
    list of int
        Admissible generator values (full list or subsample, sorted).
    """
    modulus = r_p + 1
    admissible = []

    for alpha in range(2, modulus):
        if math.gcd(alpha, modulus) != 1:
            continue
        powers = set()
        val = 1
        distinct = True
        for _ in range(q):
            remainder = val % modulus
            if remainder in powers:
                distinct = False
                break
            powers.add(remainder)
            val = (val * alpha) % modulus
        if distinct:
            admissible.append(alpha)

    if len(admissible) > B_gamma:
        if rng is None:
            rng = np.random.default_rng(42)
        chosen = rng.choice(admissible, size=B_gamma, replace=False)
        return sorted(chosen.tolist())

    return admissible


def _construct_glp_design(
    r_p: int,
    q: int,
    alpha: int,
) -> np.ndarray:
    """Construct a candidate r_p-run q-factor design via the power generator.

    Implements the formula from Section 2.2:

        u_j^{(α)} = [ mod(j · γ_α,  r_p + 1) / r_p ] − (1 / 2r_p) · 1_q

    where  γ_α = (α⁰, α¹, …, α^{q−1})⊤.

    Parameters
    ----------
    r_p : int
        Number of design runs (skeleton pairs).
    q : int
        Number of factors (retained PCA dimensions).
    alpha : int
        Admissible power generator.

    Returns
    -------
    U : ndarray of shape (r_p, q)
        Candidate design points in [0, 1]^q.
    """
    modulus = r_p + 1
    # Build the power generator vector γ_α = (1, α, α², …, α^{q-1}) mod (r_p+1)
    gamma = np.empty(q, dtype=np.int64)
    val = 1
    for d in range(q):
        gamma[d] = val % modulus
        val = (val * alpha) % modulus

    # j = 1, 2, …, r_p
    j_vals = np.arange(1, r_p + 1, dtype=np.int64)  # shape (r_p,)
    # Outer product:  (j_vals ⊗ gamma) mod modulus  →  shape (r_p, q)
    raw = np.mod(j_vals[:, np.newaxis] * gamma[np.newaxis, :], modulus)
    U = raw.astype(np.float64) / r_p - 1.0 / (2.0 * r_p)
    return U


_DISCREPANCY_MEM_BUDGET: int = 128 * 1024 * 1024  # ~128 MB per worker


def _mixture_discrepancy_squared(U: np.ndarray) -> float:
    """Evaluate the squared mixture discrepancy D²_M of a design in [0,1]^q.

    Implements the closed-form expression derived in Proposition 2
    (Appendix A.1):

        D²_M = (19/12)^q
               − (2/r_p) Σ_j Π_d [ 5/3 − ¼|u_{jd}−½| − ¼(u_{jd}−½)² ]
               + (1/r_p²) Σ_j Σ_k Π_d k_M(u_{jd}, u_{kd})

    where k_M(u,t) = 15/8 − ¼|u−½| − ¼|t−½| − ¾|u−t| + ½(u−t)².

    A smaller D²_M indicates a more uniformly scattered design.

    The pairwise term (Term 3) is evaluated in row chunks with a
    per-dimension accumulation loop so that peak memory is bounded by
    ~``_DISCREPANCY_MEM_BUDGET`` regardless of ``r_p`` and ``q``.

    Parameters
    ----------
    U : ndarray of shape (r_p, q)
        Design points in [0, 1]^q.

    Returns
    -------
    float
        Squared mixture discrepancy.
    """
    r_p, q = U.shape

    # Term 1
    term1 = (19.0 / 12.0) ** q

    # Term 2
    centered = U - 0.5
    A1_vals = 5.0 / 3.0 - 0.25 * np.abs(centered) - 0.25 * centered ** 2
    term2 = -2.0 / r_p * np.sum(np.prod(A1_vals, axis=1))

    # Term 3 — chunked rows × dimension loop to cap memory.
    # Working arrays per chunk: prod_block(cs, r_p), diff_d(cs, r_p), k_d(cs, r_p)
    # ≈ 3 × chunk_size × r_p × 8 bytes.
    abs_centered = np.abs(centered)
    bytes_per_row = 3 * r_p * 8
    chunk_size = max(1, min(r_p, _DISCREPANCY_MEM_BUDGET // max(bytes_per_row, 1)))

    total = 0.0
    for i0 in range(0, r_p, chunk_size):
        i1 = min(i0 + chunk_size, r_p)
        prod_block = np.ones((i1 - i0, r_p), dtype=np.float64)
        for d in range(q):
            u_i = U[i0:i1, d]
            u_all = U[:, d]
            diff_d = u_i[:, np.newaxis] - u_all[np.newaxis, :]
            k_d = (
                15.0 / 8.0
                - 0.25 * abs_centered[i0:i1, d, np.newaxis]
                - 0.25 * abs_centered[:, d][np.newaxis, :]
                - 0.75 * np.abs(diff_d)
                + 0.5 * diff_d * diff_d
            )
            prod_block *= k_d
        total += prod_block.sum()

    term3 = total / (r_p * r_p)
    return term1 + term2 + term3


def _mixture_kernel_pairs(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Evaluate the product mixture kernel for aligned pairs of points."""
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError("left and right must be aligned two-dimensional arrays.")
    diff = left - right
    coordinate_kernel = (
        15.0 / 8.0
        - 0.25 * np.abs(left - 0.5)
        - 0.25 * np.abs(right - 0.5)
        - 0.75 * np.abs(diff)
        + 0.5 * np.square(diff)
    )
    return np.prod(coordinate_kernel, axis=1)


def _approximate_gefd(
    Z: np.ndarray,
    Z_sorted: np.ndarray,
    U_skeleton: np.ndarray,
    *,
    seed: int,
    n_pairs: Optional[int] = None,
) -> Dict[str, Any]:
    """Monte Carlo approximation of the empirical mixture-kernel GEFD.

    Full-sample points are converted to their marginal empirical-CDF
    coordinates only for the randomly drawn pairs.  This avoids constructing
    an additional ``n by q`` rank matrix or an infeasible quadratic kernel
    matrix.  The reported Monte Carlo standard error pertains to the squared
    discrepancy estimate before truncation at zero.
    """
    Z_arr = np.asarray(Z, dtype=np.float64)
    sorted_arr = np.asarray(Z_sorted, dtype=np.float64)
    U_arr = np.asarray(U_skeleton, dtype=np.float64)
    if Z_arr.ndim != 2 or sorted_arr.shape != Z_arr.shape:
        raise ValueError("Z and Z_sorted must have the same two-dimensional shape.")
    if U_arr.ndim != 2 or U_arr.shape[1] != Z_arr.shape[1]:
        raise ValueError("U_skeleton must have the same number of columns as Z.")
    pairs = int(
        n_pairs
        if n_pairs is not None
        else getattr(config, "GEFD_MONTE_CARLO_PAIRS", 20_000)
    )
    if pairs <= 1:
        raise ValueError("GEFD_MONTE_CARLO_PAIRS must exceed one.")
    rng = np.random.default_rng(int(seed))
    n, q = Z_arr.shape
    r_p = U_arr.shape[0]
    full_left_idx = rng.integers(0, n, size=pairs)
    full_right_idx = rng.integers(0, n, size=pairs)
    skeleton_left_idx = rng.integers(0, r_p, size=pairs)
    skeleton_right_idx = rng.integers(0, r_p, size=pairs)

    def empirical_coordinates(indices: np.ndarray) -> np.ndarray:
        coords = np.empty((pairs, q), dtype=np.float64)
        sampled = Z_arr[indices]
        for d in range(q):
            coords[:, d] = (
                np.searchsorted(sorted_arr[:, d], sampled[:, d], side="right") / n
            )
        return coords

    full_left = empirical_coordinates(full_left_idx)
    full_right = empirical_coordinates(full_right_idx)
    contributions = (
        _mixture_kernel_pairs(full_left, full_right)
        - 2.0 * _mixture_kernel_pairs(full_left, U_arr[skeleton_left_idx])
        + _mixture_kernel_pairs(
            U_arr[skeleton_left_idx], U_arr[skeleton_right_idx]
        )
    )
    squared_estimate = float(np.mean(contributions))
    squared_mcse = float(np.std(contributions, ddof=1) / np.sqrt(pairs))
    return {
        "gefd_estimate": float(np.sqrt(max(squared_estimate, 0.0))),
        "gefd_squared_estimate": squared_estimate,
        "gefd_squared_mcse": squared_mcse,
        "gefd_mc_pairs": pairs,
    }


def _select_optimal_uniform_design(
    r_p: int,
    q: int,
    B_gamma: int,
    rng: Optional[np.random.Generator],
    cache_seed: int,
) -> Tuple[np.ndarray, bool]:
    """Select the GLP design with minimum mixture discrepancy (Algorithm 1, Step 11).

    Uses skeleton cache keyed by ``(r_p, q, B_gamma, cache_seed)`` so repeated
    calls with the same design budget reuse the stored optimal ``U*``. When
    ``config.UD_SKELETON_DISK_CACHE_DIR`` (or env ``UD_SKELETON_DISK_CACHE``) is
    set, entries are also read/written as ``.npy`` files so ``joblib`` workers
    and later processes avoid repeating the GLP search.

    Returns
    -------
    U_best : ndarray of shape (r_p, q)
        Optimal uniform design skeleton in [0, 1]^q.
    from_cache : bool
        True if ``U_best`` was retrieved from cache.
    """
    key = (int(r_p), int(q), int(B_gamma), int(cache_seed))
    cached = _UD_SKELETON_CACHE.get(key)
    if cached is not None:
        return cached.copy(), True

    disk_root = _ud_skeleton_disk_cache_root()
    disk_path = _ud_skeleton_disk_path(disk_root, key) if disk_root is not None else None
    if disk_path is not None:
        loaded = _try_load_ud_skeleton_npy(disk_path, r_p, q)
        if loaded is not None:
            U_mem = loaded.copy()
            _UD_SKELETON_CACHE[key] = U_mem
            return U_mem.copy(), True

    generators = _find_admissible_generators(r_p, q, B_gamma, rng)

    if not generators:
        j = np.arange(1, r_p + 1, dtype=np.float64)
        U = np.column_stack([(j - 0.5) / r_p for _ in range(q)])
        _UD_SKELETON_CACHE[key] = U.copy()
        if disk_path is not None:
            _atomic_save_ud_skeleton_npy(disk_path, U)
        return U, False

    best_U: Optional[np.ndarray] = None
    best_disc = np.inf

    if _UD_C_AVAILABLE:
        # Delegate the hot loop (GLP construction + D^2_M scan over all
        # enumerated generators) to the compiled C routine.  The candidate
        # set is identical to the Python baseline, so results match up to
        # floating-point rounding in the pairwise sum (< 1e-10 relative).
        try:
            best_U, _best_alpha, best_disc = _genUD.c_genUD_search(
                generators, r_p, q,
            )
        except Exception:  # pragma: no cover — fall back transparently
            best_U = None

    if best_U is None:
        for alpha in generators:
            U_candidate = _construct_glp_design(r_p, q, alpha)
            disc = _mixture_discrepancy_squared(U_candidate)
            if disc < best_disc:
                best_disc = disc
                best_U = U_candidate

    assert best_U is not None
    _UD_SKELETON_CACHE[key] = best_U.copy()
    if disk_path is not None:
        _atomic_save_ud_skeleton_npy(disk_path, best_U)
    return best_U, False


def get_ud_cache_seed(
    r_total: int,
    *,
    scenario_name: Optional[str] = None,
    population_size: Optional[int] = None,
    B_gamma: Optional[int] = None,
) -> int:
    """Return a stable cache seed for UD skeleton reuse across replications.

    The GLP skeleton depends on ``(r_p, q, B_gamma)`` plus the random subset
    choice inside the admissible-generator budgeted search. That subset should
    be fixed for a given experiment setting, not per Monte Carlo replication,
    otherwise repeated runs cannot reuse the same cached skeleton.
    """
    parts = [
        str(int(r_total)),
        str(int(population_size)) if population_size is not None else "default",
        str(int(B_gamma)) if B_gamma is not None else "default",
        scenario_name or "global",
    ]
    payload = "|".join(parts).encode("utf-8", errors="strict")
    # Fixed 32-bit seed derived from experiment settings only.
    return int(zlib.adler32(payload) & 0x7FFFFFFF) + 1


def warm_start_ud_skeleton(
    r_p: int,
    q: int,
    *,
    B_gamma: Optional[int] = None,
    cache_seed: int,
) -> Tuple[np.ndarray, bool]:
    """Ensure the UD skeleton for ``(r_p, q, B_gamma, cache_seed)`` is cached."""
    B = int(B_gamma if B_gamma is not None else getattr(config, "UD_MAX_GENERATOR_CANDIDATES", 30))
    rng = np.random.default_rng(int(cache_seed))
    return _select_optimal_uniform_design(int(r_p), int(q), B, rng, int(cache_seed))


# ── Paired Nearest-Neighbour Matching ────────────────────────────────────


def _build_kdtree(points: np.ndarray) -> cKDTree:
    """Build a cKDTree spatial index for nearest-neighbour queries.

    Tuned for very large arms (n ~ 1e6+): a larger leafsize plus
    ``balanced_tree=False`` and ``compact_nodes=False`` cuts construction
    time roughly 3-4× versus scipy defaults at the cost of a small
    increase in per-query depth — favourable when r_p << n.
    """
    try:
        return cKDTree(
            points,
            leafsize=32,
            balanced_tree=False,
            compact_nodes=False,
            copy_data=False,
        )
    except TypeError:
        return cKDTree(points)


def _kdtree_query_nearest(tree: cKDTree, point: np.ndarray) -> int:
    """Return index of the exact nearest neighbour (k=1)."""
    try:
        # workers=1: avoid nested thread pools under joblib process parallelism (UD matching loop).
        _, idx = tree.query(point, k=1, workers=1)
    except TypeError:
        _, idx = tree.query(point, k=1)
    return int(np.atleast_1d(idx).ravel()[0])


def _match_without_replacement(
    tree: cKDTree,
    targets: np.ndarray,
    *,
    initial_neighbors: int = 8,
    max_neighbors: int = 256,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Match target points one-to-one to KD-tree observations.

    The routine first obtains exact k-nearest-neighbour candidate edges from
    ``cKDTree`` and then solves the minimum-weight full bipartite matching on
    that sparse graph.  If collisions prevent a full assignment, ``k`` is
    doubled until a solution is found or the explicit candidate cap is
    reached.  Unlike the former batched 1-NN lookup, returned indices are
    guaranteed to be unique.

    Parameters
    ----------
    tree : cKDTree
        Tree built on one treatment arm.
    targets : ndarray of shape (m, q)
        Skeleton points in the same rotated coordinate system.
    initial_neighbors : int
        Initial number of exact neighbours retained per target.
    max_neighbors : int
        Maximum candidate count per target.  Hitting the cap raises an
        informative error instead of silently reusing observations.

    Returns
    -------
    local_indices : ndarray of shape (m,)
        Unique row indices into ``tree.data``.
    distances : ndarray of shape (m,)
        Euclidean matching distances.
    neighbors_used : int
        Candidate count at which the full assignment was found.
    """
    targets = np.asarray(targets, dtype=np.float64)
    if targets.ndim != 2:
        raise ValueError("targets must be a two-dimensional array.")

    n_targets = int(targets.shape[0])
    n_points = int(tree.n)
    if n_targets == 0:
        return np.empty(0, dtype=np.intp), np.empty(0), 0
    if n_targets > n_points:
        raise ValueError(
            "Without-replacement matching requires at least as many arm "
            "observations as skeleton targets."
        )

    cap = min(n_points, max(1, int(max_neighbors)))
    k = min(cap, max(1, int(initial_neighbors)))
    last_error: Optional[Exception] = None

    while True:
        try:
            distances, candidate_idx = tree.query(targets, k=k, workers=1)
        except TypeError:  # scipy < 1.6 compatibility
            distances, candidate_idx = tree.query(targets, k=k)

        distances = np.asarray(distances, dtype=np.float64)
        candidate_idx = np.asarray(candidate_idx, dtype=np.intp)
        if k == 1:
            distances = distances[:, np.newaxis]
            candidate_idx = candidate_idx[:, np.newaxis]

        row_idx = np.repeat(np.arange(n_targets, dtype=np.intp), k)
        col_idx = candidate_idx.reshape(-1)
        edge_weights = distances.reshape(-1)
        valid = (
            np.isfinite(edge_weights)
            & (col_idx >= 0)
            & (col_idx < n_points)
        )
        # Sparse matching treats stored zeros as missing edges.  Adding machine
        # epsilon preserves ordering while keeping exact-zero matches present.
        graph = coo_matrix(
            (
                edge_weights[valid] + np.finfo(np.float64).eps,
                (row_idx[valid], col_idx[valid]),
            ),
            shape=(n_targets, n_points),
        ).tocsr()

        try:
            matched_rows, matched_cols = min_weight_full_bipartite_matching(graph)
            if matched_rows.size != n_targets:
                raise ValueError("candidate graph did not yield a full row matching")
            order = np.argsort(matched_rows)
            local_indices = np.asarray(matched_cols[order], dtype=np.intp)
            if np.unique(local_indices).size != n_targets:
                raise RuntimeError("bipartite matcher returned duplicate columns")
            matched_points = np.asarray(tree.data)[local_indices]
            matched_distances = np.linalg.norm(targets - matched_points, axis=1)
            return local_indices, matched_distances, k
        except ValueError as exc:
            last_error = exc

        if k >= cap:
            raise RuntimeError(
                "Unable to construct a unique UD match within the configured "
                f"candidate cap ({cap} neighbours per target). Increase "
                "config.UD_MATCH_MAX_NEIGHBORS or reduce r_total."
            ) from last_error
        k = min(cap, 2 * k)


# ── Full UD Subsampling Pipeline ─────────────────────────────────────────


def _select_ud_indices(
    X: np.ndarray,
    W: np.ndarray,
    r_total: int,
    rng: np.random.Generator,
    *,
    B_gamma: Optional[int] = None,
    cache_seed: int,
    profile: Optional[Dict[str, float]] = None,
    diagnostics: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """Execute Phase 1 of Algorithm 1: UD subsampling in PCA-rotated Z-space.

    Steps:
        1. Standardise X → X̃ (once).
        2. SVD / PCA: smallest *q* with cumulative variance ≥ ρ₀; Z = X̃ V_q.
        3. Marginal empirical order statistics of Z (inverse-CDF support).
        4. GLP / power-generator candidates in [0,1]^q; search all admissible
           or a random subset of size B_γ; minimise mixture discrepancy D²_M.
        5. Map skeleton U → V in Z-space via empirical inverse CDF.
        6. cKDTree on Z for each arm; one-to-one nearest-neighbour assignment
           without replacement.

    Parameters
    ----------
    profile : dict, optional
        If provided, cumulative wall times (seconds) are written for keys
        ``standardize_pca``, ``ecdf_sort``, ``design_search``,
        ``inverse_cdf_map``, ``kd_build``, ``matching``.
    """
    if r_total <= 0:
        raise ValueError("r_total must be positive for UD subsampling.")
    if r_total > X.shape[0]:
        raise ValueError("r_total cannot exceed population size.")
    if r_total % 2 != 0:
        raise ValueError("UD subsampling requires an even r_total for paired arms.")

    # Keep the final fold order independent of whether the GLP skeleton came
    # from cache (a cache hit consumes no generator-search random numbers).
    shuffle_rng = np.random.default_rng(
        int(rng.integers(0, np.iinfo(np.uint32).max, dtype=np.uint32))
    )

    # ── Per-worker cache: reuse indices for identical (X, W, r, seed) ──
    global _UD_INDICES_CACHE
    cache_key = (
        id(X), id(W), int(r_total), int(cache_seed),
        int(B_gamma) if B_gamma is not None else -1,
        _UD_MATCHING_VERSION,
    )
    if (
        profile is None
        and diagnostics is None
        and _UD_INDICES_CACHE is not None
        and _UD_INDICES_CACHE[0] == cache_key
    ):
        return _UD_INDICES_CACHE[1].copy()

    treated_idx = np.where(W == 1)[0]
    control_idx = np.where(W == 0)[0]

    if treated_idx.size == 0 or control_idx.size == 0:
        raise ValueError("UD subsampling requires both treated and control units.")

    r_p = r_total // 2
    if treated_idx.size < r_p or control_idx.size < r_p:
        raise ValueError(
            "UD subsampling requires at least r_total/2 observations in each arm."
        )

    B = int(B_gamma if B_gamma is not None else getattr(config, "UD_MAX_GENERATOR_CANDIDATES", 30))

    if profile is not None:
        for k in (
            "standardize_pca",
            "ecdf_sort",
            "design_search",
            "inverse_cdf_map",
            "kd_build",
            "matching",
        ):
            profile.setdefault(k, 0.0)

    t0 = time.perf_counter()
    rho_0 = getattr(config, "UD_VARIANCE_THRESHOLD", 0.85)
    # copy=False avoids redundant 40 MB allocation at n=5e5 × p=10 when
    # X is already float64 (the common case from generate_obs_*).
    X_tilde = _standardise_covariates(np.asarray(X, dtype=np.float64))
    Z_all, _V_q, q = _pca_rotate(X_tilde, rho_0)
    if profile is not None:
        profile["standardize_pca"] += time.perf_counter() - t0

    t1 = time.perf_counter()
    Z_sorted = _marginal_empirical_cdf_ranks(Z_all)
    if profile is not None:
        profile["ecdf_sort"] += time.perf_counter() - t1

    t2 = time.perf_counter()
    U_skeleton, _from_cache = _select_optimal_uniform_design(
        r_p, q, B, rng, int(cache_seed),
    )
    if profile is not None:
        profile["design_search"] += time.perf_counter() - t2

    t3 = time.perf_counter()
    V_skeleton = _map_skeleton_to_rotated_space(U_skeleton, Z_sorted)
    if profile is not None:
        profile["inverse_cdf_map"] += time.perf_counter() - t3

    t4 = time.perf_counter()
    Z_treated = Z_all[treated_idx]
    Z_control = Z_all[control_idx]
    tree_treated = _build_kdtree(Z_treated)
    tree_control = _build_kdtree(Z_control)
    if profile is not None:
        profile["kd_build"] += time.perf_counter() - t4

    t5 = time.perf_counter()
    # Exact KD-tree candidate queries followed by sparse one-to-one assignment.
    max_neighbors = int(getattr(config, "UD_MATCH_MAX_NEIGHBORS", 256))
    initial_neighbors = int(getattr(config, "UD_NEAREST_NEIGHBORS", 8))
    t_locals, t_dist, t_k = _match_without_replacement(
        tree_treated,
        V_skeleton,
        initial_neighbors=initial_neighbors,
        max_neighbors=max_neighbors,
    )
    c_locals, c_dist, c_k = _match_without_replacement(
        tree_control,
        V_skeleton,
        initial_neighbors=initial_neighbors,
        max_neighbors=max_neighbors,
    )
    selected_treated = treated_idx[t_locals]
    selected_control = control_idx[c_locals]
    if profile is not None:
        profile["matching"] += time.perf_counter() - t5

    combined = np.concatenate([selected_treated, selected_control])
    if combined.size != r_total or np.unique(combined).size != r_total:
        raise RuntimeError("UD matching failed to return r_total unique observations.")
    if diagnostics is not None:
        all_dist = np.concatenate([t_dist, c_dist])
        diagnostics.update(
            {
                "retained_dimensions": int(q),
                "variance_threshold": float(rho_0),
                "treated_selected": int(selected_treated.size),
                "control_selected": int(selected_control.size),
                "matching_mean_distance": float(np.mean(all_dist)),
                "matching_max_distance": float(np.max(all_dist)),
                "treated_matching_mean_distance": float(np.mean(t_dist)),
                "treated_matching_max_distance": float(np.max(t_dist)),
                "control_matching_mean_distance": float(np.mean(c_dist)),
                "control_matching_max_distance": float(np.max(c_dist)),
                "treated_candidate_neighbors": int(t_k),
                "control_candidate_neighbors": int(c_k),
                "without_replacement": True,
            }
        )
        diagnostics.update(
            _approximate_gefd(
                Z_all,
                Z_sorted,
                U_skeleton,
                seed=cache_seed + 4001,
            )
        )
    shuffle_rng.shuffle(combined)
    # Store pre-shuffle would preserve deterministic index set, but callers
    # only need the unordered selection, so cache the shuffled array.
    _UD_INDICES_CACHE = (cache_key, combined.copy())
    return combined


def _select_separate_arm_ud_indices(
    X: np.ndarray,
    arm_indices: np.ndarray,
    r_arm: int,
    rng: np.random.Generator,
    *,
    B_gamma: Optional[int],
    cache_seed: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Construct and match a UD independently within one treatment arm."""
    if r_arm <= 0 or arm_indices.size < r_arm:
        raise ValueError("Separate-arm UD requires at least r_arm observations.")

    X_arm = np.asarray(X[arm_indices], dtype=np.float64)
    rho_0 = float(getattr(config, "UD_VARIANCE_THRESHOLD", 0.85))
    X_tilde = _standardise_covariates(X_arm)
    Z_arm, _V_q, q = _pca_rotate(X_tilde, rho_0)
    Z_sorted = _marginal_empirical_cdf_ranks(Z_arm)
    B = int(
        B_gamma
        if B_gamma is not None
        else getattr(config, "UD_MAX_GENERATOR_CANDIDATES", 30)
    )
    U_skeleton, _ = _select_optimal_uniform_design(
        r_arm, q, B, rng, int(cache_seed),
    )
    V_skeleton = _map_skeleton_to_rotated_space(U_skeleton, Z_sorted)
    tree = _build_kdtree(Z_arm)
    local_idx, distances, candidate_k = _match_without_replacement(
        tree,
        V_skeleton,
        initial_neighbors=int(getattr(config, "UD_NEAREST_NEIGHBORS", 8)),
        max_neighbors=int(getattr(config, "UD_MATCH_MAX_NEIGHBORS", 256)),
    )
    selected = np.asarray(arm_indices[local_idx], dtype=np.intp)
    if np.unique(selected).size != r_arm:
        raise RuntimeError("Separate-arm UD returned duplicate observations.")
    arm_diagnostics = {
        "retained_dimensions": int(q),
        "variance_threshold": rho_0,
        "matching_mean_distance": float(np.mean(distances)),
        "matching_max_distance": float(np.max(distances)),
        "candidate_neighbors": int(candidate_k),
    }
    arm_diagnostics.update(
        _approximate_gefd(
            Z_arm,
            Z_sorted,
            U_skeleton,
            seed=cache_seed + 4001,
        )
    )
    return selected, arm_diagnostics


# ═══════════════════════════════════════════════════════════════════════════
# Section 4 — Public Estimator Entry Points
# ═══════════════════════════════════════════════════════════════════════════


def run_full(
    X: np.ndarray,
    W: np.ndarray,
    Y_obs: np.ndarray,
    pi_true: Any,
    is_rct: bool,
    k_folds: int = 2,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Full-data cross-fitted DML estimator (full-sample reference).

    Trains nuisance models on the entire dataset using K-fold cross-fitting
    and computes the AIPW-based ATE estimate.

    Parameters
    ----------
    X : ndarray of shape (n, p)
        Covariate matrix.
    W : ndarray of shape (n,)
        Treatment assignments.
    Y_obs : ndarray of shape (n,)
        Observed outcomes.
    pi_true : float or ndarray
        True propensity score(s) — used only as ``pi_rct_val`` in RCTs.
    is_rct : bool
        Whether the design is a randomised controlled trial.
    k_folds : int
        Number of cross-fitting folds.
    **kwargs
        ``misspecification``, ``learner``, ``n_estimators``.

    Returns
    -------
    dict
        Keys: ``est_ate``, ``ci_lower``, ``ci_upper``, ``runtime``,
        ``subsample_size``, ``subsample_unique``, ``learner``.
    """
    start = time.time()
    pi_val = float(pi_true) if np.isscalar(pi_true) else float(np.mean(pi_true))
    learner = kwargs.get("learner", getattr(config, "DEFAULT_NUISANCE_LEARNER", "lgbm"))

    mu0, mu1, e = _fit_nuisance_models(
        X, W, Y_obs, k_folds, is_rct, pi_val,
        misspecification=kwargs.get("misspecification"),
        learner=learner,
    )
    scores = _aipw_score(Y_obs, W, mu0, mu1, e)
    est_ate, ci_lower, ci_upper = _wald_inference(scores)

    return {
        "est_ate": est_ate,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "runtime": time.time() - start,
        "subsample_size": X.shape[0],
        "subsample_unique": X.shape[0],
        "learner": learner,
    }


def run_unif(
    X: np.ndarray,
    W: np.ndarray,
    Y_obs: np.ndarray,
    pi_true: Any,
    is_rct: bool,
    r: Dict[str, int],
    k_folds: int = 2,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Naive uniform random subsampling + DML (benchmark).

    Draws a simple random subsample of size ``r['r_total']`` without
    replacement, then runs cross-fitted DML on the subsample.

    Parameters
    ----------
    X, W, Y_obs, pi_true, is_rct, k_folds
        See ``run_full``.
    r : dict
        Must contain ``'r_total'`` (desired subsample size).
    **kwargs
        ``sim_seed``, ``misspecification``, ``learner``, ``store_sample``.

    Returns
    -------
    dict
        Same keys as ``run_full`` plus ``subsample_projection``,
        ``subsample_indices``.
    """
    start = time.time()
    r_total = int(r["r_total"])
    if r_total <= 0:
        raise ValueError("r_total must be positive.")
    r_total = min(r_total, X.shape[0])

    sim_seed = kwargs.get("sim_seed", config.BASE_SEED)
    rng = np.random.default_rng(sim_seed + 23)
    idx = rng.choice(X.shape[0], size=r_total, replace=False)

    X_sub, W_sub, Y_sub = X[idx], W[idx], Y_obs[idx]
    pi_val = float(pi_true) if np.isscalar(pi_true) else float(np.mean(pi_true))
    learner = kwargs.get("learner", getattr(config, "DEFAULT_NUISANCE_LEARNER", "lgbm"))

    mu0, mu1, e = _fit_nuisance_models(
        X_sub, W_sub, Y_sub, k_folds, is_rct, pi_val,
        misspecification=kwargs.get("misspecification"),
        learner=learner,
    )
    scores = _aipw_score(Y_sub, W_sub, mu0, mu1, e)
    est_ate, ci_lower, ci_upper = _wald_inference(scores)
    score_variance = float(np.var(scores, ddof=1))
    standard_error = float(np.sqrt(score_variance / scores.size))
    design_diagnostics = _covariate_smd_diagnostics(X_sub, W_sub)
    design_diagnostics.update(
        {
            "treated_selected": int(np.sum(W_sub == 1)),
            "control_selected": int(np.sum(W_sub == 0)),
            "without_replacement": True,
        }
    )

    return {
        "est_ate": est_ate,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "standard_error": standard_error,
        "score_variance": score_variance,
        "variance_method": "iid_pseudo_outcome",
        "runtime": time.time() - start,
        "subsample_size": r_total,
        "subsample_unique": r_total,
        "learner": learner,
        "subsample_projection": X_sub[:, :2].copy() if kwargs.get("store_sample") else None,
        "subsample_indices": idx.tolist(),
        "design_diagnostics": design_diagnostics,
    }


def run_stratified_unif(
    X: np.ndarray,
    W: np.ndarray,
    Y_obs: np.ndarray,
    pi_true: Any,
    is_rct: bool,
    r: Dict[str, int],
    k_folds: int = 2,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Treatment-balanced random subsampling followed by cross-fitted DML.

    This diagnostic baseline isolates the contribution of enforcing exactly
    ``r_total / 2`` treated and control observations from the contribution of
    the uniform-design geometry.
    """
    start = time.perf_counter()
    r_total = int(r["r_total"])
    if r_total <= 0 or r_total % 2 != 0:
        raise ValueError("STRAT-UNIF requires a positive even r_total.")
    if r_total > X.shape[0]:
        raise ValueError("r_total cannot exceed population size.")

    treated_idx = np.flatnonzero(W == 1)
    control_idx = np.flatnonzero(W == 0)
    r_arm = r_total // 2
    if treated_idx.size < r_arm or control_idx.size < r_arm:
        raise ValueError("STRAT-UNIF requires at least r_total/2 units per arm.")

    sim_seed = int(kwargs.get("sim_seed", config.BASE_SEED))
    rng = np.random.default_rng(sim_seed + 29)
    idx = np.concatenate(
        [
            rng.choice(treated_idx, size=r_arm, replace=False),
            rng.choice(control_idx, size=r_arm, replace=False),
        ]
    )
    rng.shuffle(idx)

    X_sub, W_sub, Y_sub = X[idx], W[idx], Y_obs[idx]
    pi_val = float(pi_true) if np.isscalar(pi_true) else float(np.mean(pi_true))
    learner = kwargs.get("learner", getattr(config, "DEFAULT_NUISANCE_LEARNER", "lgbm"))
    mu0, mu1, e = _fit_nuisance_models(
        X_sub,
        W_sub,
        Y_sub,
        k_folds,
        is_rct,
        pi_val,
        misspecification=kwargs.get("misspecification"),
        learner=learner,
    )
    scores = _aipw_score(Y_sub, W_sub, mu0, mu1, e)
    est_ate, ci_lower, ci_upper, standard_error, residual_variance = (
        _selected_residual_inference(scores, Y_sub, W_sub, mu0, mu1, e)
    )
    design_diagnostics = _covariate_smd_diagnostics(X_sub, W_sub)
    design_diagnostics.update(
        {
            "treated_selected": r_arm,
            "control_selected": r_arm,
            "without_replacement": True,
        }
    )
    return {
        "est_ate": est_ate,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "standard_error": standard_error,
        "residual_variance": residual_variance,
        "variance_method": "conditional_selected_residual",
        "runtime": time.perf_counter() - start,
        "subsample_size": int(idx.size),
        "subsample_unique": int(np.unique(idx).size),
        "learner": learner,
        "subsample_projection": X_sub[:, :2].copy() if kwargs.get("store_sample") else None,
        "subsample_indices": idx.tolist(),
        "design_diagnostics": design_diagnostics,
    }


def run_sep_ud(
    X: np.ndarray,
    W: np.ndarray,
    Y_obs: np.ndarray,
    pi_true: Any,
    is_rct: bool,
    r: Dict[str, int],
    k_folds: int = 2,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Separate-arm uniform-design subsampling followed by DML.

    A distinct PCA/ECDF transform, GLP skeleton, and one-to-one match are
    constructed inside each treatment arm.  This baseline isolates the value
    of UD-DML's pooled transform and common paired skeleton.
    """
    start = time.perf_counter()
    r_total = int(r["r_total"])
    if r_total <= 0 or r_total % 2 != 0:
        raise ValueError("SEP-UD requires a positive even r_total.")
    if r_total > X.shape[0]:
        raise ValueError("r_total cannot exceed population size.")

    treated_idx = np.flatnonzero(W == 1)
    control_idx = np.flatnonzero(W == 0)
    r_arm = r_total // 2
    if treated_idx.size < r_arm or control_idx.size < r_arm:
        raise ValueError("SEP-UD requires at least r_total/2 units per arm.")

    sim_seed = int(kwargs.get("sim_seed", config.BASE_SEED))
    rng = np.random.default_rng(sim_seed + 37)
    shuffle_rng = np.random.default_rng(
        int(rng.integers(0, np.iinfo(np.uint32).max, dtype=np.uint32))
    )
    B_gamma = kwargs.get("B_gamma")
    if B_gamma is not None:
        B_gamma = int(B_gamma)
    cache_seed = kwargs.get("cache_seed")
    if cache_seed is None:
        cache_seed = get_ud_cache_seed(
            r_total,
            scenario_name=kwargs.get("scenario_name"),
            population_size=kwargs.get("population_size"),
            B_gamma=B_gamma,
        )
    cache_seed = int(cache_seed)

    selected_treated, treated_diag = _select_separate_arm_ud_indices(
        X,
        treated_idx,
        r_arm,
        rng,
        B_gamma=B_gamma,
        cache_seed=cache_seed + 1009,
    )
    selected_control, control_diag = _select_separate_arm_ud_indices(
        X,
        control_idx,
        r_arm,
        rng,
        B_gamma=B_gamma,
        cache_seed=cache_seed + 2003,
    )
    idx = np.concatenate([selected_treated, selected_control])
    if np.unique(idx).size != r_total:
        raise RuntimeError("SEP-UD failed to return r_total unique observations.")
    shuffle_rng.shuffle(idx)

    X_sub, W_sub, Y_sub = X[idx], W[idx], Y_obs[idx]
    pi_val = float(pi_true) if np.isscalar(pi_true) else float(np.mean(pi_true))
    learner = kwargs.get("learner", getattr(config, "DEFAULT_NUISANCE_LEARNER", "lgbm"))
    mu0, mu1, e = _fit_nuisance_models(
        X_sub,
        W_sub,
        Y_sub,
        k_folds,
        is_rct,
        pi_val,
        misspecification=kwargs.get("misspecification"),
        learner=learner,
    )
    scores = _aipw_score(Y_sub, W_sub, mu0, mu1, e)
    est_ate, ci_lower, ci_upper, standard_error, residual_variance = (
        _selected_residual_inference(scores, Y_sub, W_sub, mu0, mu1, e)
    )
    mean_matching_distance = 0.5 * (
        float(treated_diag["matching_mean_distance"])
        + float(control_diag["matching_mean_distance"])
    )
    design_diagnostics = _covariate_smd_diagnostics(X_sub, W_sub)
    design_diagnostics.update(
        {
            "treated_selected": r_arm,
            "control_selected": r_arm,
            "without_replacement": True,
            "matching_mean_distance": mean_matching_distance,
            "matching_max_distance": max(
                float(treated_diag["matching_max_distance"]),
                float(control_diag["matching_max_distance"]),
            ),
            "gefd_estimate": 0.5 * (
                float(treated_diag["gefd_estimate"])
                + float(control_diag["gefd_estimate"])
            ),
            "treated": treated_diag,
            "control": control_diag,
        }
    )
    return {
        "est_ate": est_ate,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "standard_error": standard_error,
        "residual_variance": residual_variance,
        "variance_method": "conditional_selected_residual",
        "runtime": time.perf_counter() - start,
        "subsample_size": int(idx.size),
        "subsample_unique": int(np.unique(idx).size),
        "learner": learner,
        "subsample_projection": X_sub[:, :2].copy() if kwargs.get("store_sample") else None,
        "subsample_indices": idx.tolist(),
        "design_diagnostics": design_diagnostics,
    }


def run_ud(
    X: np.ndarray,
    W: np.ndarray,
    Y_obs: np.ndarray,
    pi_true: Any,
    is_rct: bool,
    r: Dict[str, int],
    k_folds: int = 2,
    return_profile: bool = False,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Uniform Design subsampling + DML (proposed UD-DML estimator).

    Executes the full three-phase UD-DML pipeline (Algorithm 1):

        Phase 1: UD subsampling in retained PCA-rotated space.
        Phase 2: Cross-fitted DML on selected original observations.
        Phase 3: Wald estimation and inference.

    Parameters
    ----------
    X, W, Y_obs, pi_true, is_rct, k_folds
        See ``run_full``.
    r : dict
        Must contain ``'r_total'`` (desired total subsample size = 2 r_p).
    return_profile : bool
        If True, include ``time_breakdown`` with per-phase wall times (seconds).
    **kwargs
        ``sim_seed``, ``misspecification``, ``learner``, ``store_sample``,
        ``B_gamma`` (optional override for ``UD_MAX_GENERATOR_CANDIDATES``).

    Returns
    -------
    dict
        Same keys as ``run_full`` plus ``subsample_projection``,
        ``subsample_indices``.  If ``return_profile`` is True, also
        ``time_breakdown`` with keys ``standardize_pca``, ``ecdf_sort``,
        ``design_search``, ``inverse_cdf_map``, ``kd_build``, ``matching``,
        ``dml``, ``inference``, ``total``.
    """
    t_wall0 = time.perf_counter()
    phase1_prof: Optional[Dict[str, float]] = {} if return_profile else None
    design_diagnostics: Dict[str, Any] = {}

    r_total = int(r["r_total"])
    if r_total <= 0:
        raise ValueError("r_total must be positive for UD-DML.")
    if r_total > X.shape[0]:
        raise ValueError("r_total cannot exceed population size for UD-DML.")

    sim_seed = int(kwargs.get("sim_seed", config.BASE_SEED))
    rng = np.random.default_rng(sim_seed + 31)
    B_gamma = kwargs.get("B_gamma")
    if B_gamma is not None:
        B_gamma = int(B_gamma)
    cache_seed = kwargs.get("cache_seed")
    if cache_seed is None:
        cache_seed = get_ud_cache_seed(
            r_total,
            scenario_name=kwargs.get("scenario_name"),
            population_size=kwargs.get("population_size"),
            B_gamma=B_gamma,
        )
    cache_seed = int(cache_seed)

    subsample_idx = _select_ud_indices(
        X,
        W,
        r_total,
        rng,
        B_gamma=B_gamma,
        cache_seed=cache_seed,
        profile=phase1_prof,
        diagnostics=design_diagnostics,
    )
    unique_count = int(np.unique(subsample_idx).size)

    X_sub, W_sub, Y_sub = X[subsample_idx], W[subsample_idx], Y_obs[subsample_idx]
    pi_val = float(pi_true) if np.isscalar(pi_true) else float(np.mean(pi_true))
    learner = kwargs.get("learner", getattr(config, "DEFAULT_NUISANCE_LEARNER", "lgbm"))

    t_dml0 = time.perf_counter()
    mu0, mu1, e = _fit_nuisance_models(
        X_sub,
        W_sub,
        Y_sub,
        k_folds,
        is_rct,
        pi_val,
        misspecification=kwargs.get("misspecification"),
        learner=learner,
    )
    scores = _aipw_score(Y_sub, W_sub, mu0, mu1, e)
    t_dml1 = time.perf_counter()

    t_inf0 = time.perf_counter()
    est_ate, ci_lower, ci_upper, standard_error, residual_variance = (
        _selected_residual_inference(scores, Y_sub, W_sub, mu0, mu1, e)
    )
    design_diagnostics.update(_covariate_smd_diagnostics(X_sub, W_sub))
    t_inf1 = time.perf_counter()

    total_time = time.perf_counter() - t_wall0

    out: Dict[str, Any] = {
        "est_ate": est_ate,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "standard_error": standard_error,
        "residual_variance": residual_variance,
        "variance_method": "conditional_selected_residual",
        "runtime": total_time,
        "subsample_size": len(subsample_idx),
        "subsample_unique": unique_count,
        "learner": learner,
        "subsample_projection": X_sub[:, :2].copy() if kwargs.get("store_sample") else None,
        "subsample_indices": subsample_idx.tolist(),
        "design_diagnostics": design_diagnostics,
    }
    if return_profile:
        assert phase1_prof is not None
        out["time_breakdown"] = {
            "standardize_pca": phase1_prof["standardize_pca"],
            "ecdf_sort": phase1_prof["ecdf_sort"],
            "design_search": phase1_prof["design_search"],
            "inverse_cdf_map": phase1_prof["inverse_cdf_map"],
            "kd_build": phase1_prof["kd_build"],
            "matching": phase1_prof["matching"],
            "dml": t_dml1 - t_dml0,
            "inference": t_inf1 - t_inf0,
            "total": total_time,
        }
    return out
