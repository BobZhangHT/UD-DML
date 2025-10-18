# -*- coding: utf-8 -*-
"""
methods.py - OS-DML Implementation

Implements Algorithm 1: Two-Stage Optimal Subsampling DML (OS-DML) and benchmark methods.

=============================================================================
ALGORITHM 1 OVERVIEW: Two-Stage Optimal Subsampling DML (OS-DML)
=============================================================================

OS-DML is a computationally efficient method for estimating average treatment effects (ATE)
from large-scale datasets using double/debiased machine learning with optimal subsampling.

KEY INNOVATION:
Instead of uniform subsampling, OS-DML uses importance sampling proportional to the absolute
value of influence functions |φ_i^(0)|, which optimally allocates samples to observations
that contribute most to estimation uncertainty.

ALGORITHM STRUCTURE:

Phase 1: Pilot Estimation and Probability Construction (Steps 1-4)
    1. Draw uniform pilot subsample S_0 of size r_0
    2. Fit cross-fitted nuisance models η^(0) = (μ₀, μ₁, e) on pilot data
    3. Predict pseudo-outcomes φ_i^(0) for ALL N observations in full dataset
    4. Compute centered pseudo-outcomes and construct stabilized sampling probabilities:
       φ̂̄^(0) = N^(-1) * Σ_i φ̂_i^(0) (average over full data)
       p_i ∝ |φ̂_i^(0) - φ̂̄^(0)| + δ (centered and stabilized)

Phase 2: Main Subsampling and Final Estimation (Steps 5-8)
    5. Draw main subsample S_1 of size r_1 using probabilities {p_i}
    6. Form combined subsample S_comb = S_0 ∪ S_1 with importance weights
    7. Fit final cross-fitted nuisance models on combined subsample
    8. Compute estimator (Hájek or Hansen-Hurwitz):
       Hájek: τ̂_HJ = (Σ_t φ̂_{I_t} / q_{I_t}) / (Σ_t 1 / q_{I_t})
       Hansen-Hurwitz: τ_HH = (1/(N*r)) * Σ_t (φ_{I_t} / q_{I_t})
       where q_{I_t} = 1/N for pilot draws, q_{I_t} = p_{I_t} for PPS draws

Inference (Steps 9-10)
    9-10. Compute variance:
       Hájek: Plug-in variance Var(τ̂_HJ) = (1/(r*N²)) * (1/(r-1)) * Σ_t(U_t - Ū)²
             where U_t = (φ̂_t - τ̂_HJ) / q_t and Ū = r^(-1) * Σ_t U_t
       Hansen-Hurwitz: Design-based variance Var(τ_HH) = (1/(N²*r)) * (1/(r-1)) * Σ_t(H_t - H̄)²

THEORETICAL PROPERTIES:
- Design-unbiased for estimated finite-population mean
- √r-consistent with asymptotic normality
- Variance reduction compared to uniform subsampling
- Double robustness inherited from DML framework

BENCHMARK METHODS IMPLEMENTED:
- FULL: Full-data DML (gold standard)
- UNIF: Uniform subsampling DML
- LSS: Leverage score subsampling DML
- OS: Optimal subsampling DML (Algorithm 1)

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
    """
    Computes the Neyman-orthogonal score (pseudo-outcome) φ for the ATE.
    
    Formula: φ(Z; η) = μ₁(X) - μ₀(X) + W/e(X) * (Y - μ₁(X)) - (1-W)/(1-e(X)) * (Y - μ₀(X))
    
    Args:
        Y: Observed outcomes
        W: Treatment indicators
        mu0: Estimated E[Y|X,W=0]
        mu1: Estimated E[Y|X,W=1]
        e: Estimated propensity score P(W=1|X)
    
    Returns:
        Array of pseudo-outcomes (influence functions)
    """
    return mu1 - mu0 + (W / e) * (Y - mu1) - ((1 - W) / (1 - e)) * (Y - mu0)

def _fit_nuisance_models(X, W, Y, k_folds, is_rct, pi_rct_val=None, 
                         sample_weight=None, misspecification=None):
    """
    Performs K-fold cross-fitting for nuisance functions η = (μ₀, μ₁, e).
    
    This is a key component of DML that ensures the Neyman-orthogonality property.
    Cross-fitting eliminates overfitting bias by training on one fold and predicting
    on another.
    
    Args:
        X: Covariates (n x p)
        W: Treatment indicators (n,)
        Y: Observed outcomes (n,)
        k_folds: Number of folds for cross-fitting
        is_rct: If True, propensity score e(X) is known constant
        pi_rct_val: Value of constant propensity score for RCT (if is_rct=True)
        sample_weight: Importance weights for each observation (for subsampling methods)
        misspecification: Scenario for robustness checks ('correct_correct', 
                         'correct_wrong', 'wrong_correct', 'wrong_wrong')
    
    Returns:
        Tuple of (mu0_preds, mu1_preds, e_preds):
        - mu0_preds: E[Y|X,W=0] predictions for all n observations
        - mu1_preds: E[Y|X,W=1] predictions for all n observations
        - e_preds: P(W=1|X) predictions for all n observations (clipped to [0.01, 0.99])
    """
    mu0_preds = np.zeros(len(Y))
    mu1_preds = np.zeros(len(Y))
    e_preds = np.zeros(len(Y))
    
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=config.BASE_SEED)
    
    lgbm_params = {'n_jobs': 1, 'random_state': config.BASE_SEED, 'n_estimators': 100, 
                   'num_leaves': 31, 'verbose': -1}
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

def _get_hajek_ci(scores, q_j, N):
    """
    Calculates Hájek ATE estimator with sample variance-based (linearization) estimator.
    
    Implements Algorithm 1 from the latest OS-DML paper:
    - Point estimation: τ̂_HJ = (Σ_t φ̂_{I_t} / q_{I_t}) / (Σ_t 1 / q_{I_t})
    - Sample variance-based variance: 
      Var(τ̂_HJ) = 1/(r*(Σ_t 1/q_{I_t})²) * (1/(r-1)) * Σ_t(U_t - Ū)²
      where U_t = (φ̂_t - τ̂_HJ) / q_{I_t} and Ū = r^(-1) * Σ_t U_t
    
    Args:
        scores: Pseudo-outcomes φ̂_{I_t} for each draw
        q_j: Selection probabilities q_{I_t} for each draw
        N: Population size (not used in the new formula)
    
    Returns:
        Tuple of (est_ate, ci_lower, ci_upper)
    """
    r = len(scores)
    if r <= 1:
        return np.nan, np.nan, np.nan
    
    # Step 10: Hájek ATE estimator (ratio estimator)
    # τ̂_HJ = (Σ_t φ̂_{I_t} / q_{I_t}) / (Σ_t 1 / q_{I_t})
    numerator = np.sum(scores / q_j)
    denominator = np.sum(1 / q_j)
    est_ate = numerator / denominator
    
    # Step 12: Define residuals
    # U_t = (φ̂_t - τ̂_HJ) / q_{I_t}
    U_t = (scores - est_ate) / q_j
    
    # Ū = (r_0 + r_1)^(-1) * Σ_t U_t
    U_bar = np.mean(U_t)
    
    # Step 13: Sample variance-based (linearization) estimator
    # Var(τ̂_HJ) = 1/(r*(Σ_t 1/q_{I_t})²) * (1/(r-1)) * Σ_t(U_t - Ū)²
    var_hat = (r / (denominator**2)) * (1.0 / (r - 1)) * np.sum((U_t - U_bar)**2)
    
    # Standard error and confidence interval
    se = np.sqrt(var_hat) if var_hat >= 0 else np.nan
    ci_lower = est_ate - 1.96 * se
    ci_upper = est_ate + 1.96 * se
    
    return est_ate, ci_lower, ci_upper


def _get_hansen_hurwitz_ci(scores, q_j, N):
    """
    Calculates design-based CI for the Hansen-Hurwitz estimator following Algorithm 1.
    
    This implements Steps 8-10 of Algorithm 1:
    - Step 8: τ_HH = (1/(N*r)) * Σ_t (φ_{I_t} / q_{I_t})
    - Step 9-10: Design-based variance with scaling factor 1/N²
    
    Args:
        scores: Pseudo-outcomes φ_{I_t}
        q_j: Selection probabilities for each draw
        N: Population size
    
    Returns:
        Tuple of (est_ate, ci_lower, ci_upper)
    """
    r = len(scores)
    if r <= 1:
        return np.nan, np.nan, np.nan
    
    # Step 8: Hansen-Hurwitz estimator
    # τ_HH = (1/(N*r)) * Σ_t (φ_{I_t} / q_{I_t})
    H_t = scores / q_j  # Scaled pseudo-outcomes
    est_ate = np.mean(H_t) / N
    
    # Steps 9-10: Design-based variance
    # Var(τ_HH) = (1/(N²*r)) * (1/(r-1)) * Σ_t(H_t - H̄)²
    H_bar = np.mean(H_t)
    var_hat = (1.0 / (N**2 * r)) * (1.0 / (r - 1)) * np.sum((H_t - H_bar)**2)
    
    se = np.sqrt(var_hat) if var_hat >= 0 else np.nan
    ci_lower = est_ate - 1.96 * se
    ci_upper = est_ate + 1.96 * se
    
    return est_ate, ci_lower, ci_upper


def _get_estimator_ci(scores, q_j, N, estimator_type='hajek'):
    """
    Wrapper function to choose between Hájek and Hansen-Hurwitz estimators.
    
    Args:
        scores: Pseudo-outcomes φ_{I_t}
        q_j: Selection probabilities for each draw
        N: Population size
        estimator_type: 'hajek' or 'hh'
    
    Returns:
        Tuple of (est_ate, ci_lower, ci_upper)
    """
    if estimator_type.lower() == 'hajek':
        return _get_hajek_ci(scores, q_j, N)
    elif estimator_type.lower() == 'hh':
        return _get_hansen_hurwitz_ci(scores, q_j, N)
    else:
        raise ValueError(f"Unknown estimator type: {estimator_type}. Use 'hajek' or 'hh'.")


def _get_full_data_ci(scores):
    """
    Calculates standard DML confidence interval using influence function variance.
    
    For full-data DML, we use the standard asymptotic variance formula based on
    the sample variance of the pseudo-outcomes (influence functions).
    
    Args:
        scores: Pseudo-outcomes φ_i for all n observations
    
    Returns:
        Tuple of (est_ate, ci_lower, ci_upper)
    """
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
    """
    FULL: Full-Data Double Machine Learning.
    
    Benchmark method that uses the entire dataset without subsampling.
    This represents the gold standard that subsampling methods try to approximate.
    Uses standard DML with cross-fitting and influence function-based inference.
    
    Args:
        X: Covariates (N x p)
        W: Treatment indicators (N,)
        Y_obs: Observed outcomes (N,)
        pi_true: True propensity scores (not used)
        is_rct: Whether data comes from RCT
        k_folds: Number of cross-fitting folds
        **kwargs: Additional args including 'misspecification' (for robustness experiments)
    
    Returns:
        Dict with keys: est_ate, ci_lower, ci_upper, runtime
    """
    start_time = time.time()
    misspecification = kwargs.get('misspecification')
    mu0, mu1, e = _fit_nuisance_models(X, W, Y_obs, k_folds, is_rct, np.mean(W), 
                                        misspecification=misspecification)
    scores = _orthogonal_score(Y_obs, W, mu0, mu1, e)
    est_ate, ci_lower, ci_upper = _get_full_data_ci(scores)
    return {"est_ate": est_ate, "ci_lower": ci_lower, "ci_upper": ci_upper, 
            "runtime": time.time() - start_time}

def run_unif(X, W, Y_obs, pi_true, is_rct, r, k_folds, **kwargs):
    """
    Benchmark: Uniform Subsampling DML with Hansen-Hurwitz inference.
    
    This method draws r_total = r0 + r1 samples uniformly with replacement
    and uses design-based inference.
    """
    start_time = time.time()
    N = len(Y_obs)
    
    # Handle both old format (r0, r1) and new format (r_total)
    if 'r_total' in r:
        r_total = r['r_total']
    else:
        r_total = r['r0'] + r['r1']
    
    misspecification = kwargs.get('misspecification')

    # Draw uniform subsample with replacement
    sub_idx = np.random.choice(N, size=r_total, replace=True)
    q_j = np.full(r_total, 1.0 / N)
    
    X_sub, W_sub, Y_sub = X[sub_idx], W[sub_idx], Y_obs[sub_idx]
    
    # Fit nuisance models with importance weights
    weights = 1.0 / q_j
    mu0, mu1, e = _fit_nuisance_models(X_sub, W_sub, Y_sub, k_folds, is_rct, 
                                        np.mean(W_sub), sample_weight=weights, 
                                        misspecification=misspecification)
    scores = _orthogonal_score(Y_sub, W_sub, mu0, mu1, e)
    
    # Estimator and inference (Hájek or Hansen-Hurwitz)
    est_ate, ci_lower, ci_upper = _get_estimator_ci(scores, q_j, N, config.ESTIMATOR_TYPE)
    return {"est_ate": est_ate, "ci_lower": ci_lower, "ci_upper": ci_upper, 
            "runtime": time.time() - start_time}

def _run_pps_pipeline(X, W, Y_obs, pi_true, is_rct, r, k_folds, pps_type, **kwargs):
    """
    Two-phase optimal subsampling pipeline for OS-DML (Algorithm 1).
    
    Implements Algorithm 1 from the OS-DML paper.
    Note: LSS now uses single-stage sampling (see run_lss function).
    
    Args:
        X, W, Y_obs: Full population data
        pi_true: True propensity scores (not used in OS-DML)
        is_rct: Whether the data comes from RCT
        r: Dictionary with 'r0' (pilot size) and 'r1' (main subsample size) OR 'r_total' (total sample size)
        k_folds: Number of folds for cross-fitting
        pps_type: Should be 'OS' (kept for backward compatibility)
        **kwargs: Additional arguments including 'misspecification'
    
    Returns:
        Dictionary with est_ate, ci_lower, ci_upper, runtime
    """
    start_time = time.time()
    N = len(Y_obs)
    
    # Handle both old format (r0, r1) and new format (r_total)
    if 'r_total' in r:
        r_total = r['r_total']
        r0 = int(r_total * config.PILOT_RATIO)
        r1 = r_total - r0
    else:
        r0, r1 = r['r0'], r['r1']
    
    misspecification = kwargs.get('misspecification')
    delta = config.DELTA  # Use global delta parameter

    # =========================================================================
    # PHASE 1: PILOT ESTIMATION AND PROBABILITY CONSTRUCTION (Algorithm 1, Steps 1-4)
    # =========================================================================
    
    # Step 1: Draw uniform pilot subsample S_0 of size r_0 with replacement
    pilot_idx = np.random.choice(N, size=r0, replace=True)
    X_pilot, W_pilot, Y_pilot = X[pilot_idx], W[pilot_idx], Y_obs[pilot_idx]
    
    # Step 2: Fit cross-fitted nuisance models η^(0) on pilot data
    lgbm_params_pilot = {'n_jobs': 1, 'random_state': config.BASE_SEED, 
                         'n_estimators': config.PILOT_N_ESTIMATORS, 'verbose': -1}
    
    mu0_model = lgb.LGBMRegressor(**lgbm_params_pilot).fit(
        X_pilot[W_pilot == 0], Y_pilot[W_pilot == 0])
    mu1_model = lgb.LGBMRegressor(**lgbm_params_pilot).fit(
        X_pilot[W_pilot == 1], Y_pilot[W_pilot == 1])
    
    if is_rct:
        e_model = lambda x: np.full(x.shape[0], np.mean(W_pilot))
    else:
        e_model_fit = lgb.LGBMClassifier(**lgbm_params_pilot).fit(X_pilot, W_pilot)
        e_model = lambda x: e_model_fit.predict_proba(x)[:, 1]

    # Steps 3-4: Compute estimated pseudo-outcomes for ALL i=1,...,N
    mu0_full = mu0_model.predict(X)
    mu1_full = mu1_model.predict(X)
    e_full = np.clip(e_model(X), 0.01, 0.99)
    
    # φ_i^(0) = φ(Z_i; η^(0)) for all i
    phi_pilot_full = _orthogonal_score(Y_obs, W, mu0_full, mu1_full, e_full)

    # Algorithm 1, Step 4: Compute centered pseudo-outcomes and construct probabilities
    # φ̂̄^(0) = N^(-1) * Σ_i φ̂_i^(0) (average over full data)
    phi_bar_pilot = np.mean(phi_pilot_full)
    
    # Step 4: Set stabilized centered PPS probabilities
    # p_i ∝ |φ̂_i^(0) - φ̂̄^(0)| + δ
    centered_phi = phi_pilot_full - phi_bar_pilot
    abs_centered_phi = np.abs(centered_phi)
    numerator = abs_centered_phi + delta
    pps_probs = numerator / np.sum(numerator)

    # =========================================================================
    # PHASE 2: MAIN SUBSAMPLING AND FINAL ESTIMATION (Algorithm 1, Steps 5-8)
    # =========================================================================
    
    # Step 5: Draw main subsample S_1 of size r_1 using probabilities {p_i}
    pps_idx = np.random.choice(N, size=r1, replace=True, p=pps_probs)

    # Step 6: Form combined subsample S_comb = S_0 ∪ S_1
    # Track which indices came from which phase for proper q_j assignment
    combined_idx = np.concatenate([pilot_idx, pps_idx])
    
    # Per-draw selection probabilities:
    # q_{I_t} = 1/N if from S_0 (uniform pilot)
    # q_{I_t} = p_{I_t} if from S_1 (PPS main sample)
    q_j = np.concatenate([
        np.full(r0, 1.0 / N),      # Pilot draws: uniform probability 1/N
        pps_probs[pps_idx]          # Main draws: PPS probabilities
    ])
    
    X_c, W_c, Y_c = X[combined_idx], W[combined_idx], Y_obs[combined_idx]
    
    # Step 6 (continued): Fit final nuisance models with importance weights ω_t ∝ 1/q_{I_t}
    weights = 1.0 / q_j
    mu0, mu1, e = _fit_nuisance_models(X_c, W_c, Y_c, k_folds, is_rct, 
                                        np.mean(W_c), sample_weight=weights, 
                                        misspecification=misspecification)
    
    # Step 7: Compute final estimated pseudo-outcomes
    scores = _orthogonal_score(Y_c, W_c, mu0, mu1, e)
    
    # =========================================================================
    # INFERENCE (Algorithm 1, Steps 8-10)
    # =========================================================================
    
    # Steps 8-10: Calculate estimator and variance (Hájek or Hansen-Hurwitz)
    est_ate, ci_lower, ci_upper = _get_estimator_ci(scores, q_j, N, config.ESTIMATOR_TYPE)
    
    runtime = time.time() - start_time
    return {"est_ate": est_ate, "ci_lower": ci_lower, "ci_upper": ci_upper, 
            "runtime": runtime}

def run_os(X, W, Y_obs, pi_true, is_rct, r, k_folds, **kwargs):
    """
    OS-DML: Optimal Subsampling for Double Machine Learning.
    
    Implements Algorithm 1 from the OS-DML paper. This is the main proposed method
    that uses a two-stage procedure:
    1. Pilot stage: Estimate nuisance functions on uniform subsample
    2. Main stage: Draw probability-proportional-to-size sample based on |φ_i^(0)|
    
    Uses Hájek estimator with plug-in variance estimation by default.
    
    Args:
        X: Covariates (N x p)
        W: Treatment indicators (N,)
        Y_obs: Observed outcomes (N,)
        pi_true: True propensity scores (not used in OS-DML)
        is_rct: Whether data comes from RCT
        r: Dict with 'r0' (pilot size) and 'r1' (main subsample size) OR 'r_total' (total sample size)
        k_folds: Number of cross-fitting folds
        **kwargs: Additional args including 'misspecification' (for robustness experiments)
    
    Returns:
        Dict with keys: est_ate, ci_lower, ci_upper, runtime
    """
    return _run_pps_pipeline(X, W, Y_obs, pi_true, is_rct, r, k_folds, 'OS', **kwargs)

def run_lss(X, W, Y_obs, pi_true, is_rct, r, k_folds, **kwargs):
    """
    LSS: Leverage Score Subsampling for DML (Single-Stage).
    
    Benchmark method that uses leverage scores for sampling probabilities.
    Unlike OS-DML, this is a single-stage method that directly samples
    based on statistical leverage scores without pilot estimation.
    
    Leverage scores: h_i = diagonal elements of H = X(X'X)^{-1}X'
    Sampling probabilities: p_i ∝ h_i
    
    Args:
        X: Covariates (N x p)
        W: Treatment indicators (N,)
        Y_obs: Observed outcomes (N,)
        pi_true: True propensity scores (not used)
        is_rct: Whether data comes from RCT
        r: Dict with 'r0' (pilot size) and 'r1' (main subsample size) OR 'r_total' (total sample size)
        k_folds: Number of cross-fitting folds
        **kwargs: Additional args including 'misspecification'
    
    Returns:
        Dict with keys: est_ate, ci_lower, ci_upper, runtime
    """
    start_time = time.time()
    N = len(Y_obs)
    
    # Handle both old format (r0, r1) and new format (r_total)
    if 'r_total' in r:
        r_total = r['r_total']
    else:
        r_total = r['r0'] + r['r1']
    
    misspecification = kwargs.get('misspecification')
    delta = config.DELTA  # Use global delta parameter
    
    # =========================================================================
    # SINGLE-STAGE LEVERAGE SCORE SAMPLING
    # =========================================================================
    
    # Compute leverage scores from covariate matrix
    X_aug = np.c_[np.ones(N), X]  # Add intercept
    
    try:
        # Compute Q from QR decomposition (more stable than (X'X)^{-1})
        Q, _ = np.linalg.qr(X_aug)
        leverages = np.sum(Q**2, axis=1)  # h_i = ||Q_i||^2
        
        # Stabilize and normalize to get sampling probabilities
        stabilized_leverages = leverages + delta
        pps_probs = stabilized_leverages / np.sum(stabilized_leverages)
        
    except np.linalg.LinAlgError:
        print("Warning: QR decomposition failed for LSS, falling back to uniform sampling")
        pps_probs = np.full(N, 1.0 / N)
    
    # Draw single subsample based on leverage scores
    subsample_idx = np.random.choice(N, size=r_total, replace=True, p=pps_probs)
    q_j = pps_probs[subsample_idx]  # Selection probabilities for each draw
    
    # Extract subsampled data
    X_sub, W_sub, Y_sub = X[subsample_idx], W[subsample_idx], Y_obs[subsample_idx]
    
    # Fit nuisance models with importance weights
    weights = 1.0 / q_j
    mu0, mu1, e = _fit_nuisance_models(X_sub, W_sub, Y_sub, k_folds, is_rct, 
                                        np.mean(W_sub), sample_weight=weights, 
                                        misspecification=misspecification)
    
    # Compute pseudo-outcomes
    scores = _orthogonal_score(Y_sub, W_sub, mu0, mu1, e)
    
    # Estimator and inference (Hájek or Hansen-Hurwitz)
    est_ate, ci_lower, ci_upper = _get_estimator_ci(scores, q_j, N, config.ESTIMATOR_TYPE)
    
    runtime = time.time() - start_time
    return {"est_ate": est_ate, "ci_lower": ci_lower, "ci_upper": ci_upper, 
            "runtime": runtime}

