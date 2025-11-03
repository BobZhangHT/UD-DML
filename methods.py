# -*- coding: utf-8 -*-
"""
methods.py - UD-DML Implementation

Implements Algorithm 1 from UD_DML.pdf: Uniform Design Double Machine Learning (UD-DML)
alongside benchmark routines.

=============================================================================
ALGORITHM OVERVIEW: Uniform Design Double Machine Learning (UD-DML)
=============================================================================

UD-DML targets efficient estimation of causal parameters with double/debiased machine
learning when the full sample is massive. The method first constructs a subsample via
uniform design (UD) using stratified low-discrepancy points and nearest-neighbour matching,
then runs cross-fitted DML on the resulting subset.

PHASE 1: UD-MMD SUBSAMPLING
    1. Map covariates X to [0, 1]^p with empirical CDF transforms (probability integral transform).
    2. Draw Latin-hypercube skeleton points v_j in [0, 1]^p.
    3. For each skeleton point, locate the nearest treated unit and the nearest control unit
       without replacement (paired design) to obtain a balanced subsample.

PHASE 2: CROSS-FITTED DML ON SUBSAMPLE
    4. Perform K-fold cross-fitting on the UD-selected subset.
    5. Compute Neyman-orthogonal pseudo outcomes (AIPW score for the ATE).

INFERENCE
    - Point estimate: sample average of the pseudo outcomes.
    - Variance: empirical variance of pseudo outcomes scaled by 1/r.
    - Confidence interval: Wald interval using asymptotic normality.

BENCHMARK METHODS IMPLEMENTED:
    - FULL: Full-data DML (gold standard)
    - UNIF: Simple uniform subsampling DML (without replacement)
    - UD: Uniform Design subsampling via MMD with stratified uniform draws (proposed)

All content is in English.
"""
import time
import warnings
from typing import Optional, Tuple

import numpy as np
import lightgbm as lgb
from numba import njit
from scipy.spatial import cKDTree
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LassoCV, LogisticRegression, LogisticRegressionCV
from sklearn.preprocessing import StandardScaler

import config

warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.utils.validation')
warnings.filterwarnings('ignore', message='.*feature names.*')
warnings.filterwarnings('ignore', message='X does not have valid feature names')

_CI_Z = 1.96


# =============================================================================
# NUMBA-ACCELERATED UTILITIES FOR UD SUBSAMPLING
# =============================================================================

@njit
def _rank_transform_numba(X: np.ndarray, clip: float) -> np.ndarray:
    """Compute marginal ranks and rescale them to (0, 1) with optional clipping."""
    n, p = X.shape
    out = np.empty((n, p), dtype=np.float64)
    denom = n + 1.0
    for j in range(p):
        col = X[:, j]
        order = np.argsort(col)
        ranks = np.empty(n, dtype=np.int64)
        for pos in range(n):
            ranks[order[pos]] = pos + 1
        for i in range(n):
            val = ranks[i] / denom
            if val < clip:
                val = clip
            elif val > 1.0 - clip:
                val = 1.0 - clip
            out[i, j] = val
    return out

# =============================================================================
# GENERAL DML HELPER FUNCTIONS
# =============================================================================


def _orthogonal_score(Y: np.ndarray, W: np.ndarray, mu0: np.ndarray, mu1: np.ndarray, e: np.ndarray) -> np.ndarray:
    """Compute the AIPW pseudo-outcome for the ATE."""
    return mu1 - mu0 + (W / e) * (Y - mu1) - ((1.0 - W) / (1.0 - e)) * (Y - mu0)


def _fit_nuisance_models(
    X: np.ndarray,
    W: np.ndarray,
    Y: np.ndarray,
    k_folds: int,
    is_rct: bool,
    pi_rct_val: Optional[float] = None,
    sample_weight: Optional[np.ndarray] = None,
    misspecification: Optional[str] = None,
    n_estimators: Optional[int] = None,
    learner: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cross-fitted nuisance estimation mirroring the original OS-DML implementation."""
    n = Y.shape[0]
    mu0_preds = np.zeros(n)
    mu1_preds = np.zeros(n)
    e_preds = np.zeros(n)

    kf = KFold(n_splits=k_folds, shuffle=True, random_state=config.BASE_SEED)

    n_estimators = n_estimators or getattr(config, 'LGBM_N_ESTIMATORS', 100)
    max_depth = getattr(config, 'LGBM_MAX_DEPTH', 5)
    learning_rate = getattr(config, 'LGBM_LEARNING_RATE', 0.1)
    num_leaves = getattr(config, 'LGBM_NUM_LEAVES', 31)
    p_half = X.shape[1] // 2
    learner = (learner or getattr(config, 'DEFAULT_NUISANCE_LEARNER', 'lgbm')).lower()
    if learner in ('lasso', 'lassocv'):
        learner = 'lasso_cv'
    rf_trees = getattr(config, 'RF_N_TREES', 200)
    rf_jobs = getattr(config, 'RF_N_JOBS', 1)
    lasso_cv_folds = getattr(config, 'LASSO_CV_FOLDS', 5)
    lasso_cv_max_iter = getattr(config, 'LASSO_CV_MAX_ITER', 5000)
    logit_cv_max_iter = getattr(config, 'LOGIT_CV_MAX_ITER', 5000)
    logit_cv_scoring = getattr(config, 'LOGIT_CV_SCORING', 'neg_log_loss')
    logit_cv_cs = getattr(config, 'LOGIT_CV_CS', None)

    for train_idx, test_idx in kf.split(X):
        X_train, Y_train, W_train = X[train_idx], Y[train_idx], W[train_idx]
        X_test = X[test_idx]
        weights_train = sample_weight[train_idx] if sample_weight is not None else None

        if misspecification in ['wrong_correct', 'wrong_wrong']:
            from sklearn.linear_model import LinearRegression

            lr0 = LinearRegression().fit(X_train[W_train == 0, :p_half], Y_train[W_train == 0])
            mu0_preds[test_idx] = lr0.predict(X_test[:, :p_half])
            lr1 = LinearRegression().fit(X_train[W_train == 1, :p_half], Y_train[W_train == 1])
            mu1_preds[test_idx] = lr1.predict(X_test[:, :p_half])
        else:
            if learner == 'rf':
                rf_params = dict(
                    n_estimators=rf_trees,
                    random_state=config.BASE_SEED,
                    n_jobs=rf_jobs,
                )
                rf0 = RandomForestRegressor(**rf_params)
                rf0.fit(
                    X_train[W_train == 0],
                    Y_train[W_train == 0],
                    sample_weight=(weights_train[W_train == 0] if weights_train is not None else None),
                )
                mu0_preds[test_idx] = rf0.predict(X_test)

                rf1 = RandomForestRegressor(**rf_params)
                rf1.fit(
                    X_train[W_train == 1],
                    Y_train[W_train == 1],
                    sample_weight=(weights_train[W_train == 1] if weights_train is not None else None),
                )
                mu1_preds[test_idx] = rf1.predict(X_test)
            elif learner == 'lasso_cv':
                scaler0 = StandardScaler()
                x0 = X_train[W_train == 0]
                scaler0.fit(x0)
                model0 = LassoCV(
                    cv=lasso_cv_folds,
                    n_jobs=1,
                    random_state=config.BASE_SEED,
                    max_iter=lasso_cv_max_iter,
                )
                model0.fit(scaler0.transform(x0), Y_train[W_train == 0])
                mu0_preds[test_idx] = model0.predict(scaler0.transform(X_test))

                scaler1 = StandardScaler()
                x1 = X_train[W_train == 1]
                scaler1.fit(x1)
                model1 = LassoCV(
                    cv=lasso_cv_folds,
                    n_jobs=1,
                    random_state=config.BASE_SEED,
                    max_iter=lasso_cv_max_iter,
                )
                model1.fit(scaler1.transform(x1), Y_train[W_train == 1])
                mu1_preds[test_idx] = model1.predict(scaler1.transform(X_test))
            else:
                lgbm0 = lgb.LGBMRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    learning_rate=learning_rate,
                    num_leaves=num_leaves,
                    random_state=config.BASE_SEED,
                    verbose=-1,
                    n_jobs=1,
                )
                lgbm0.fit(
                    X_train[W_train == 0],
                    Y_train[W_train == 0],
                    sample_weight=(weights_train[W_train == 0] if weights_train is not None else None),
                )
                mu0_preds[test_idx] = lgbm0.predict(X_test)

                lgbm1 = lgb.LGBMRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    learning_rate=learning_rate,
                    num_leaves=num_leaves,
                    random_state=config.BASE_SEED,
                    verbose=-1,
                    n_jobs=1,
                )
                lgbm1.fit(
                    X_train[W_train == 1],
                    Y_train[W_train == 1],
                    sample_weight=(weights_train[W_train == 1] if weights_train is not None else None),
                )
                mu1_preds[test_idx] = lgbm1.predict(X_test)

        if is_rct:
            e_preds[test_idx] = pi_rct_val if pi_rct_val is not None else np.mean(W_train)
        elif misspecification in ['correct_wrong', 'wrong_wrong']:
            lr_e = LogisticRegression(solver='liblinear', max_iter=1000).fit(X_train[:, :p_half], W_train)
            e_preds[test_idx] = lr_e.predict_proba(X_test[:, :p_half])[:, 1]
        else:
            if learner == 'rf':
                clf = RandomForestClassifier(
                    n_estimators=rf_trees,
                    random_state=config.BASE_SEED,
                    n_jobs=rf_jobs,
                )
                clf.fit(X_train, W_train, sample_weight=weights_train)
                e_preds[test_idx] = clf.predict_proba(X_test)[:, 1]
            elif learner == 'lasso_cv':
                scaler_e = StandardScaler().fit(X_train)
                X_train_scaled = scaler_e.transform(X_train)
                X_test_scaled = scaler_e.transform(X_test)
                clf = LogisticRegressionCV(
                    Cs=logit_cv_cs,
                    cv=lasso_cv_folds,
                    penalty='l1',
                    solver='saga',
                    scoring=logit_cv_scoring,
                    max_iter=logit_cv_max_iter,
                    random_state=config.BASE_SEED,
                    n_jobs=1,
                )
                clf.fit(X_train_scaled, W_train, sample_weight=weights_train)
                e_preds[test_idx] = clf.predict_proba(X_test_scaled)[:, 1]
            else:
                lgbmc = lgb.LGBMClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    learning_rate=learning_rate,
                    num_leaves=num_leaves,
                    random_state=config.BASE_SEED,
                    verbose=-1,
                    n_jobs=1,
                )
                lgbmc.fit(X_train, W_train, sample_weight=weights_train)
                e_preds[test_idx] = lgbmc.predict_proba(X_test)[:, 1]

    e_preds = np.clip(e_preds, 0.01, 0.99)
    return mu0_preds, mu1_preds, e_preds


def _wald_ci_from_scores(scores: np.ndarray) -> Tuple[float, float, float]:
    r = scores.shape[0]
    if r <= 1:
        return np.nan, np.nan, np.nan
    est = float(np.mean(scores))
    se = float(np.std(scores, ddof=1) / np.sqrt(r))
    ci_lower = est - _CI_Z * se
    ci_upper = est + _CI_Z * se
    return est, ci_lower, ci_upper


# =============================================================================
# UD-DML SUBSAMPLING PIPELINE
# =============================================================================


def _transform_covariates(X: np.ndarray) -> np.ndarray:
    X_contig = np.ascontiguousarray(X, dtype=np.float64)
    clip = getattr(config, 'UD_CDF_CLIP', 1e-6)
    return _rank_transform_numba(X_contig, float(clip))


def _generate_uniform_design_points(n_points: int, dimension: int, rng: np.random.Generator) -> np.ndarray:
    """Generate Latin-hypercube points in [0, 1]^p."""
    if n_points <= 0:
        raise ValueError("n_points must be positive for UD subsampling.")
    points = np.empty((n_points, dimension), dtype=np.float64)
    for d in range(dimension):
        strata = (np.arange(n_points) + rng.random(n_points)) / n_points
        rng.shuffle(strata)
        points[:, d] = strata
    clip = getattr(config, 'UD_CDF_CLIP', 1e-6)
    return np.clip(points, clip, 1.0 - clip)


def _query_available(tree: cKDTree, used: np.ndarray, point: np.ndarray, rng: np.random.Generator) -> int:
    remaining = np.flatnonzero(~used)
    if remaining.size == 0:
        raise RuntimeError("No available units to match.")
    k = 1
    while True:
        k = min(k, remaining.size)
        try:
            _, idx = tree.query(point, k=k, workers=-1)
        except TypeError:
            _, idx = tree.query(point, k=k)
        idx = np.atleast_1d(idx)
        for candidate in idx:
            if not used[candidate]:
                return int(candidate)
        if k >= remaining.size:
            break
        k = min(remaining.size, k * 2)
    return int(rng.choice(remaining))


def _select_ud_indices(X: np.ndarray, W: np.ndarray, r_total: int, rng: np.random.Generator) -> np.ndarray:
    if r_total <= 0:
        raise ValueError("r_total must be positive for UD subsampling.")
    if r_total > X.shape[0]:
        raise ValueError("r_total cannot exceed population size in UD subsampling.")

    treated_idx = np.where(W == 1)[0]
    control_idx = np.where(W == 0)[0]
    if treated_idx.size == 0 or control_idx.size == 0:
        raise ValueError("UD subsampling requires both treated and control units.")

    max_pairs = min(treated_idx.size, control_idx.size, r_total // 2)
    if max_pairs == 0:
        raise ValueError("Insufficient treated/control counts for UD subsampling.")

    transformed = _transform_covariates(X)
    treated_data = transformed[treated_idx]
    control_data = transformed[control_idx]

    treat_tree = cKDTree(treated_data)
    control_tree = cKDTree(control_data)
    used_treat = np.zeros(treated_idx.size, dtype=bool)
    used_control = np.zeros(control_idx.size, dtype=bool)

    skeleton = _generate_uniform_design_points(max_pairs, X.shape[1], rng)
    selected_treat = []
    selected_control = []

    for point in skeleton:
        t_local = _query_available(treat_tree, used_treat, point, rng)
        c_local = _query_available(control_tree, used_control, point, rng)
        used_treat[t_local] = True
        used_control[c_local] = True
        selected_treat.append(treated_idx[t_local])
        selected_control.append(control_idx[c_local])

    combined = np.concatenate([selected_treat, selected_control])
    rng.shuffle(combined)
    return combined


# =============================================================================
# MAIN METHOD IMPLEMENTATIONS
# =============================================================================


def run_full(X, W, Y_obs, pi_true, is_rct, k_folds, **kwargs):
    start_time = time.time()
    misspecification = kwargs.get('misspecification')
    n_estimators_override = kwargs.get('n_estimators', getattr(config, 'LGBM_N_ESTIMATORS', None))
    pi_value = float(pi_true) if np.isscalar(pi_true) else np.mean(pi_true) if pi_true is not None else np.mean(W)
    learner = kwargs.get('learner', getattr(config, 'DEFAULT_NUISANCE_LEARNER', 'lgbm'))

    mu0, mu1, e = _fit_nuisance_models(
        X,
        W,
        Y_obs,
        k_folds,
        is_rct,
        pi_value,
        sample_weight=None,
        misspecification=misspecification,
        n_estimators=n_estimators_override,
        learner=learner,
    )
    scores = _orthogonal_score(Y_obs, W, mu0, mu1, e)
    est_ate, ci_lower, ci_upper = _wald_ci_from_scores(scores)
    return {
        'est_ate': est_ate,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'runtime': time.time() - start_time,
        'subsample_size': X.shape[0],
        'subsample_unique': X.shape[0],
        'learner': learner,
    }


def run_unif(X, W, Y_obs, pi_true, is_rct, r, k_folds, **kwargs):
    start_time = time.time()
    if 'r_total' not in r:
        raise ValueError("run_unif requires 'r_total' in the r dictionary.")
    r_total = int(r['r_total'])
    if r_total <= 0:
        raise ValueError('r_total must be positive in run_unif.')
    if r_total > X.shape[0]:
        r_total = X.shape[0]

    sim_seed = kwargs.get('sim_seed', config.BASE_SEED)
    rng = np.random.default_rng(sim_seed + 23)
    subsample_idx = rng.choice(X.shape[0], size=r_total, replace=False)
    X_sub, W_sub, Y_sub = X[subsample_idx], W[subsample_idx], Y_obs[subsample_idx]

    misspecification = kwargs.get('misspecification')
    n_estimators_override = kwargs.get('n_estimators', getattr(config, 'LGBM_N_ESTIMATORS', None))
    pi_value = float(pi_true) if np.isscalar(pi_true) else np.mean(pi_true) if pi_true is not None else np.mean(W_sub)
    learner = kwargs.get('learner', getattr(config, 'DEFAULT_NUISANCE_LEARNER', 'lgbm'))

    mu0, mu1, e = _fit_nuisance_models(
        X_sub,
        W_sub,
        Y_sub,
        k_folds,
        is_rct,
        pi_value,
        sample_weight=None,
        misspecification=misspecification,
        n_estimators=n_estimators_override,
        learner=learner,
    )
    scores = _orthogonal_score(Y_sub, W_sub, mu0, mu1, e)
    est_ate, ci_lower, ci_upper = _wald_ci_from_scores(scores)
    store_sample = kwargs.get('store_sample')
    sample_payload = X_sub[:, :2].copy() if store_sample else None
    return {
        'est_ate': est_ate,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'runtime': time.time() - start_time,
        'subsample_size': r_total,
        'subsample_unique': r_total,
        'learner': learner,
        'subsample_projection': sample_payload,
        'subsample_indices': subsample_idx.tolist(),
    }


def run_ud(X, W, Y_obs, pi_true, is_rct, r, k_folds, **kwargs):
    start_time = time.time()
    if 'r_total' not in r:
        raise ValueError("run_ud requires 'r_total' in the r dictionary.")
    r_total = int(r['r_total'])
    if r_total <= 0:
        raise ValueError('r_total must be positive for UD-DML.')
    if r_total > X.shape[0]:
        r_total = X.shape[0]

    sim_seed = kwargs.get('sim_seed', config.BASE_SEED)
    rng = np.random.default_rng(sim_seed + 31)
    subsample_idx = _select_ud_indices(X, W, r_total, rng)
    unique_count = int(np.unique(subsample_idx).size)

    X_sub, W_sub, Y_sub = X[subsample_idx], W[subsample_idx], Y_obs[subsample_idx]

    misspecification = kwargs.get('misspecification')
    n_estimators_override = kwargs.get('n_estimators', getattr(config, 'LGBM_N_ESTIMATORS', None))
    pi_value = float(pi_true) if np.isscalar(pi_true) else np.mean(pi_true) if pi_true is not None else np.mean(W_sub)
    learner = kwargs.get('learner', getattr(config, 'DEFAULT_NUISANCE_LEARNER', 'lgbm'))

    mu0, mu1, e = _fit_nuisance_models(
        X_sub,
        W_sub,
        Y_sub,
        k_folds,
        is_rct,
        pi_value,
        sample_weight=None,
        misspecification=misspecification,
        n_estimators=n_estimators_override,
        learner=learner,
    )
    scores = _orthogonal_score(Y_sub, W_sub, mu0, mu1, e)
    est_ate, ci_lower, ci_upper = _wald_ci_from_scores(scores)
    store_sample = kwargs.get('store_sample')
    sample_payload = X_sub[:, :2].copy() if store_sample else None
    return {
        'est_ate': est_ate,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'runtime': time.time() - start_time,
        'subsample_size': len(subsample_idx),
        'subsample_unique': unique_count,
        'learner': learner,
        'subsample_projection': sample_payload,
        'subsample_indices': subsample_idx.tolist(),
    }

