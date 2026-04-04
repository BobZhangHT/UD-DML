# -*- coding: utf-8 -*-
"""
methods.py — Core estimators for the UD-DML framework.

Implements Algorithm 1 from:

    Qu, Xu & Zhang (2026). "UD-DML: Uniform Design Subsampling for
    Double Machine Learning over Massive Data."

The module provides three estimators for the Average Treatment Effect (ATE):

    * ``run_full``  — Full-data cross-fitted DML (gold standard).
    * ``run_unif``  — Naive uniform random subsampling + DML (benchmark).
    * ``run_ud``    — Uniform Design subsampling + DML (proposed method).

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
    5. For each skeleton point v_j, find the nearest treated and nearest
       control unit in Z-space **with replacement** via exact ``cKDTree``
       nearest-neighbour queries.

**Phase 2 — Cross-fitted DML on the selected original observations**

    6. Standard K-fold cross-fitting on {(Y_i, W_i, X_i) : i ∈ S}.
    7. Compute the AIPW pseudo-outcomes.

**Phase 3 — Estimation and Wald inference**

    8. Point estimate: θ̂ = (1/r) Σ ψ̂*.
    9. Variance: σ̂² / r  with σ̂² the empirical variance of pseudo-outcomes.
   10. Confidence interval: θ̂ ± z_{1-α/2} √(σ̂²/r).
"""

from __future__ import annotations

import math
import time
import warnings
from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree
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

    Under misspecification ``'wrong_correct'`` or ``'wrong_wrong'``, the
    outcome models are replaced by linear regressions on the first two
    covariates only (Section 3.3, Experiment 3).
    """
    if misspecification in ("wrong_correct", "wrong_wrong"):
        from sklearn.linear_model import LinearRegression
        lr0 = LinearRegression().fit(X_tr[W_tr == 0, :2], Y_tr[W_tr == 0])
        lr1 = LinearRegression().fit(X_tr[W_tr == 1, :2], Y_tr[W_tr == 1])
        return lr0.predict(X_te[:, :2]), lr1.predict(X_te[:, :2])

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
    """Fit and predict the propensity score ê(x) = P̂(W=1|X=x)."""
    if misspecification in ("correct_wrong", "wrong_wrong"):
        lr = LogisticRegression(solver="liblinear", max_iter=1000)
        lr.fit(X_tr[:, :2], W_tr)
        return lr.predict_proba(X_te[:, :2])[:, 1]

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
#           with replacement (Algorithm 1).
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
    # economy SVD — only the first min(n, p) components
    _, S, Vt = np.linalg.svd(X_tilde, full_matrices=False)

    # Squared singular values are proportional to variance
    var_explained = S ** 2
    cumulative_ratio = np.cumsum(var_explained) / var_explained.sum()

    # Smallest q such that cumulative_ratio[q-1] >= rho_0
    q = int(np.searchsorted(cumulative_ratio, rho_0) + 1)
    q = min(q, p)  # cannot exceed p

    V_q = Vt[:q, :].T                     # shape (p, q)
    Z = X_tilde @ V_q                     # shape (n, q)
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


def _select_optimal_uniform_design(
    r_p: int,
    q: int,
    B_gamma: int,
    rng: Optional[np.random.Generator],
    cache_seed: int,
) -> Tuple[np.ndarray, bool]:
    """Select the GLP design with minimum mixture discrepancy (Algorithm 1, Step 11).

    Uses skeleton cache keyed by ``(r_p, q, B_gamma, cache_seed)`` so repeated
    calls with the same design budget reuse the stored optimal ``U*``.

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

    generators = _find_admissible_generators(r_p, q, B_gamma, rng)

    if not generators:
        j = np.arange(1, r_p + 1, dtype=np.float64)
        U = np.column_stack([(j - 0.5) / r_p for _ in range(q)])
        _UD_SKELETON_CACHE[key] = U.copy()
        return U, False

    best_U: Optional[np.ndarray] = None
    best_disc = np.inf

    for alpha in generators:
        U_candidate = _construct_glp_design(r_p, q, alpha)
        disc = _mixture_discrepancy_squared(U_candidate)
        if disc < best_disc:
            best_disc = disc
            best_U = U_candidate

    assert best_U is not None
    _UD_SKELETON_CACHE[key] = best_U.copy()
    return best_U, False


# ── Paired Nearest-Neighbour Matching ────────────────────────────────────


def _build_kdtree(points: np.ndarray) -> cKDTree:
    """Build a cKDTree spatial index for nearest-neighbour queries.

    Parameters
    ----------
    points : ndarray of shape (m, q)
        Points to index (rotated covariates for one treatment arm).

    Returns
    -------
    cKDTree
    """
    try:
        return cKDTree(points, copy_data=False)
    except TypeError:
        return cKDTree(points)


def _kdtree_query_nearest(tree: cKDTree, point: np.ndarray) -> int:
    """Return index of the exact nearest neighbour (k=1)."""
    try:
        _, idx = tree.query(point, k=1, workers=-1)
    except TypeError:
        _, idx = tree.query(point, k=1)
    return int(np.atleast_1d(idx).ravel()[0])


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
) -> np.ndarray:
    """Execute Phase 1 of Algorithm 1: UD subsampling in PCA-rotated Z-space.

    Steps:
        1. Standardise X → X̃ (once).
        2. SVD / PCA: smallest *q* with cumulative variance ≥ ρ₀; Z = X̃ V_q.
        3. Marginal empirical order statistics of Z (inverse-CDF support).
        4. GLP / power-generator candidates in [0,1]^q; search all admissible
           or a random subset of size B_γ; minimise mixture discrepancy D²_M.
        5. Map skeleton U → V in Z-space via empirical inverse CDF.
        6. cKDTree on Z for each arm; **with-replacement** 1-NN matching per v_j.

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

    treated_idx = np.where(W == 1)[0]
    control_idx = np.where(W == 0)[0]

    if treated_idx.size == 0 or control_idx.size == 0:
        raise ValueError("UD subsampling requires both treated and control units.")

    r_p = min(treated_idx.size, control_idx.size, r_total // 2)
    if r_p == 0:
        raise ValueError("Insufficient treated/control units for UD subsampling.")

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
    X_tilde = _standardise_covariates(X.astype(np.float64))
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
    selected_treated = np.empty(r_p, dtype=np.intp)
    selected_control = np.empty(r_p, dtype=np.intp)
    for j in range(r_p):
        v_j = V_skeleton[j]
        t_local = _kdtree_query_nearest(tree_treated, v_j)
        c_local = _kdtree_query_nearest(tree_control, v_j)
        selected_treated[j] = treated_idx[t_local]
        selected_control[j] = control_idx[c_local]
    if profile is not None:
        profile["matching"] += time.perf_counter() - t5

    combined = np.concatenate([selected_treated, selected_control])
    rng.shuffle(combined)
    return combined


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
    """Full-data cross-fitted DML estimator (gold standard).

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

    return {
        "est_ate": est_ate,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "runtime": time.time() - start,
        "subsample_size": r_total,
        "subsample_unique": r_total,
        "learner": learner,
        "subsample_projection": X_sub[:, :2].copy() if kwargs.get("store_sample") else None,
        "subsample_indices": idx.tolist(),
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

    r_total = int(r["r_total"])
    if r_total <= 0:
        raise ValueError("r_total must be positive for UD-DML.")
    r_total = min(r_total, X.shape[0])

    sim_seed = int(kwargs.get("sim_seed", config.BASE_SEED))
    rng = np.random.default_rng(sim_seed + 31)
    cache_seed = sim_seed + 31

    B_gamma = kwargs.get("B_gamma")
    if B_gamma is not None:
        B_gamma = int(B_gamma)

    subsample_idx = _select_ud_indices(
        X,
        W,
        r_total,
        rng,
        B_gamma=B_gamma,
        cache_seed=cache_seed,
        profile=phase1_prof,
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
    est_ate, ci_lower, ci_upper = _wald_inference(scores)
    t_inf1 = time.perf_counter()

    total_time = time.perf_counter() - t_wall0

    out: Dict[str, Any] = {
        "est_ate": est_ate,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "runtime": total_time,
        "subsample_size": len(subsample_idx),
        "subsample_unique": unique_count,
        "learner": learner,
        "subsample_projection": X_sub[:, :2].copy() if kwargs.get("store_sample") else None,
        "subsample_indices": subsample_idx.tolist(),
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
