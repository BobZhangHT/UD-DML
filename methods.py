# -*- coding: utf-8 -*-
"""
methods.py

Implements the core algorithms for OS-DML and its benchmarks (UNIF, LSS, FULL).
All content is in English.
"""
import time
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
import config

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _orthogonal_score(Y, W, mu0, mu1, e):
    """Computes the Neyman-orthogonal score for the ATE."""
    return mu1 - mu0 + (W / e) * (Y - mu1) - ((1 - W) / (1 - e)) * (Y - mu0)

def _fit_nuisance_models(X, W, Y, k_folds, is_rct, pi_rct_val=None, 
                         sample_weight=None, misspecification=None):
    """
    Performs K-fold cross-fitting for nuisance functions with importance weights.
    Includes logic for misspecification scenarios from Experiment 3.
    """
    mu0_preds = np.zeros(len(Y))
    mu1_preds = np.zeros(len(Y))
    e_preds = np.zeros(len(Y))
    
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=config.BASE_SEED)
    
    lgbm_params = {'n_jobs': 1, 'random_state': config.BASE_SEED, 'n_estimators': 100, 'num_leaves': 31}
    p_half = X.shape[1] // 2

    for train_idx, test_idx in kf.split(X):
        X_train, Y_train, W_train = X[train_idx], Y[train_idx], W[train_idx]
        X_test = X[test_idx]
        weights_train = sample_weight[train_idx] if sample_weight is not None else None

        # Outcome models
        if misspecification in ['wrong_correct', 'wrong_wrong']:
            from sklearn.linear_model import LinearRegression
            lr0 = LinearRegression().fit(X_train[W_train == 0, :p_half], Y_train[W_train == 0])
            mu0_preds[test_idx] = lr0.predict(X_test[:, :p_half])
            lr1 = LinearRegression().fit(X_train[W_train == 1, :p_half], Y_train[W_train == 1])
            mu1_preds[test_idx] = lr1.predict(X_test[:, :p_half])
        else:
            lgbm0 = lgb.LGBMRegressor(**lgbm_params).fit(X_train[W_train == 0], Y_train[W_train == 0], sample_weight=weights_train[W_train == 0] if weights_train is not None else None)
            mu0_preds[test_idx] = lgbm0.predict(X_test)
            lgbm1 = lgb.LGBMRegressor(**lgbm_params).fit(X_train[W_train == 1], Y_train[W_train == 1], sample_weight=weights_train[W_train == 1] if weights_train is not None else None)
            mu1_preds[test_idx] = lgbm1.predict(X_test)
        
        # Propensity score model
        if is_rct:
            e_preds[test_idx] = pi_rct_val
        elif misspecification in ['correct_wrong', 'wrong_wrong']:
            from sklearn.linear_model import LogisticRegression
            lr_e = LogisticRegression(solver='liblinear').fit(X_train[:, :p_half], W_train)
            e_preds[test_idx] = lr_e.predict_proba(X_test[:, :p_half])[:, 1]
        else:
            lgbm_e = lgb.LGBMClassifier(**lgbm_params).fit(X_train, W_train, sample_weight=weights_train)
            e_preds[test_idx] = lgbm_e.predict_proba(X_test)[:, 1]

    e_preds = np.clip(e_preds, 0.01, 0.99)
    return mu0_preds, mu1_preds, e_preds

def _get_frequentist_hajek_ci(scores, q_j):
    """Calculates a design-based frequentist CI for the Hájek estimator."""
    r = len(scores)
    if r <= 1: return np.nan, np.nan, np.nan
    
    weights = 1.0 / q_j
    w_sum = np.sum(weights)
    est_ate = np.sum(scores * weights) / w_sum if w_sum > 0 else np.nan
    
    var_hat = (1 / (w_sum**2)) * (r / (r - 1)) * np.sum(((scores - est_ate) / q_j)**2) if w_sum > 0 else np.nan
    
    se = np.sqrt(var_hat) if var_hat >= 0 else np.nan
    ci_lower = est_ate - 1.96 * se
    ci_upper = est_ate + 1.96 * se
    
    return est_ate, ci_lower, ci_upper
    
def _get_full_data_ci(scores):
    """Calculates a standard DML CI based on the influence function (IF) variance."""
    n = len(scores)
    est_ate = np.mean(scores)
    se = np.std(scores, ddof=1) / np.sqrt(n)
    ci_lower = est_ate - 1.96 * se
    ci_upper = est_ate + 1.96 * se
    return est_ate, ci_lower, ci_upper

# =============================================================================
# MAIN METHOD IMPLEMENTATIONS
# =============================================================================

def run_full(X, W, Y_obs, pi_true, is_rct, k_folds, **kwargs):
    """Benchmark: Full-Data DML with formula-based IF variance."""
    start_time = time.time()
    misspecification = kwargs.get('misspecification')
    mu0, mu1, e = _fit_nuisance_models(X, W, Y_obs, k_folds, is_rct, np.mean(W), misspecification=misspecification)
    scores = _orthogonal_score(Y_obs, W, mu0, mu1, e)
    est_ate, ci_lower, ci_upper = _get_full_data_ci(scores)
    return {"est_ate": est_ate, "ci_lower": ci_lower, "ci_upper": ci_upper, "runtime": time.time() - start_time}

def run_unif(X, W, Y_obs, pi_true, is_rct, r, k_folds, **kwargs):
    """Benchmark: Uniform Subsampling DML with combined-draw inference."""
    start_time = time.time()
    n = len(Y_obs)
    r_total = r['r0'] + r['r1']
    misspecification = kwargs.get('misspecification')

    sub_idx = np.random.choice(n, size=r_total, replace=True)
    q_j = np.full(r_total, 1/n)
    
    X_sub, W_sub, Y_sub = X[sub_idx], W[sub_idx], Y_obs[sub_idx]
    
    weights = (1/q_j)
    mu0, mu1, e = _fit_nuisance_models(X_sub, W_sub, Y_sub, k_folds, is_rct, np.mean(W_sub), sample_weight=weights, misspecification=misspecification)
    scores = _orthogonal_score(Y_sub, W_sub, mu0, mu1, e)
    
    est_ate, ci_lower, ci_upper = _get_frequentist_hajek_ci(scores, q_j)
    return {"est_ate": est_ate, "ci_lower": ci_lower, "ci_upper": ci_upper, "runtime": time.time() - start_time}

def _run_pps_pipeline(X, W, Y_obs, pi_true, is_rct, r, k_folds, pps_type, **kwargs):
    """Generic helper for all two-phase PPS methods (OS and LSS)."""
    start_time = time.time()
    n = len(Y_obs)
    r0, r1 = r['r0'], r['r1']
    misspecification = kwargs.get('misspecification')

    # Phase 1: Pilot & Probability Construction
    pilot_idx = np.random.choice(n, size=r0, replace=True)
    X_pilot, W_pilot, Y_pilot = X[pilot_idx], W[pilot_idx], Y_obs[pilot_idx]
    
    lgbm_params_pilot = {'n_jobs': 1, 'random_state': config.BASE_SEED, 'n_estimators': 30} # Faster pilot
    
    mu0_model = lgb.LGBMRegressor(**lgbm_params_pilot).fit(X_pilot[W_pilot == 0], Y_pilot[W_pilot == 0])
    mu1_model = lgb.LGBMRegressor(**lgbm_params_pilot).fit(X_pilot[W_pilot == 1], Y_pilot[W_pilot == 1])
    
    if is_rct:
        e_model = lambda x: np.full(x.shape[0], np.mean(W_pilot))
    else:
        e_model_fit = lgb.LGBMClassifier(**lgbm_params_pilot).fit(X_pilot, W_pilot)
        e_model = lambda x: e_model_fit.predict_proba(x)[:, 1]

    # Predict on full data to get probabilities
    mu0_full = mu0_model.predict(X)
    mu1_full = mu1_model.predict(X)
    e_full = np.clip(e_model(X), 0.01, 0.99)

    if pps_type == 'OS':
        scores_full = _orthogonal_score(Y_obs, W, mu0_full, mu1_full, e_full)
        tau_pilot = np.mean(scores_full)
        phi_pilot = scores_full - tau_pilot
        probs = np.abs(phi_pilot)
        
    elif pps_type == 'LSS':
        X_aug = np.c_[np.ones(n), X]
        try:
            Q, _ = np.linalg.qr(X_aug)
            leverages = np.sum(Q**2, axis=1)
            probs = leverages
        except np.linalg.LinAlgError:
            print("Warning: QR decomposition failed for LSS, falling back to uniform probabilities.")
            probs = np.ones(n)

    # Stabilize and normalize probabilities
    c = 0.01 * np.median(probs) if pps_type == 'OS' else 0
    stabilized_probs = np.maximum(probs, c)
    probs_sum = np.sum(stabilized_probs)
    pps_probs = stabilized_probs / probs_sum if probs_sum > 0 else np.full(n, 1/n)

    # Phase 2: Main Subsample Draw
    pps_idx = np.random.choice(n, size=r1, replace=True, p=pps_probs)

    # Combined-draw inference
    combined_idx = np.concatenate([pilot_idx, pps_idx])
    q_j = np.concatenate([np.full(r0, 1/n), pps_probs[pps_idx]])
    
    X_c, W_c, Y_c = X[combined_idx], W[combined_idx], Y_obs[combined_idx]
    
    weights = 1.0 / q_j
    mu0, mu1, e = _fit_nuisance_models(X_c, W_c, Y_c, k_folds, is_rct, np.mean(W_c), sample_weight=weights, misspecification=misspecification)
    scores = _orthogonal_score(Y_c, W_c, mu0, mu1, e)
    
    est_ate, ci_lower, ci_upper = _get_frequentist_hajek_ci(scores, q_j)
    runtime = time.time() - start_time
    return {"est_ate": est_ate, "ci_lower": ci_lower, "ci_upper": ci_upper, "runtime": runtime}

def run_os(X, W, Y_obs, pi_true, is_rct, r, k_folds, **kwargs):
    return _run_pps_pipeline(X, W, Y_obs, pi_true, is_rct, r, k_folds, 'OS', **kwargs)

def run_lss(X, W, Y_obs, pi_true, is_rct, r, k_folds, **kwargs):
    return _run_pps_pipeline(X, W, Y_obs, pi_true, is_rct, r, k_folds, 'LSS', **kwargs)

