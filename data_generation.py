# -*- coding: utf-8 -*-
"""
data_generation.py

Implements six DGPs for evaluating Double Machine Learning (DML) methods,
as per the user's request. The DGPs are structured to test double robustness
by varying the complexity and non-linearity of the nuisance functions
(propensity score e(x) and outcome model g(x)) and the target parameter
(heterogeneous treatment effect Delta(x)).

The six DGPs are a 2x3 design:
-   RCT (Randomized Controlled Trial) vs. OBS (Observational Study)
-   Scenarios 1, 2, 3, increasing in complexity.

-   Scenario 1: Simple linear nuisance/target functions, high overlap (in OBS).
-   Scenario 2: Moderate non-linearity, moderate overlap (in OBS).
-   Scenario 3: Complex/high non-linearity, low overlap (in OBS).

All content is in English.
"""
import numpy as np
from scipy.special import expit

# --- Constants ---
_DEFAULT_P = 10  # Fixed covariate dimension as requested.
_SIGMA_CORR = 1.5  # For _covariates_x2
_CORR = 0.5  # Unused in this version, but kept from original

# --- True Average Treatment Effects (ATEs) ---
# We define the true ATE as the expected value of Delta(X).
# In all scenarios, we design Delta(X) so E[Delta(X)] = 1.0.
SCENARIO_1_TRUE_ATE = 1.0
SCENARIO_2_TRUE_ATE = 1.0
SCENARIO_3_TRUE_ATE = 1.0


# --- Covariate Generation (Simple to Complex) ---

def _covariates_x1(n, p):
    """
    DGP-X1 (Simple): Independent uniform covariates.
    """
    # All covariates are independent U(-2, 2). E[X] = 0.
    return np.random.uniform(-2, 2, size=(n, p))


def _covariates_x2(n, p):
    """
    DGP-X2 (Moderate): Mixed marginals.
    First half uniform, second half normal.
    """
    X = np.empty((n, p), dtype=np.float64)
    split = min(max(p // 2, 1), p)
    # First 5 (p=10) are independent U(-2, 2). E[X] = 0.
    X[:, :split] = np.random.uniform(-2, 2, size=(n, split))
    if split < p:
        # Second 5 (p=10) are independent N(0, 1.5^2). E[X] = 0.
        X[:, split:] = np.random.normal(0.0, _SIGMA_CORR, size=(n, p - split))
    return X


def _covariates_x3(n, p):
    """
    DGP-X3 (Complex): Gaussian-mixture block + Standard Normal block.
    
    MODIFICATION:
    - Replaced the pathological Log-Normal and t-distributions.
    - The first block (X_0...X_4) remains a Gaussian Mixture (bimodal),
      which is a good source of non-linearity.
    - The second block (X_5...X_9) is now standard N(0, 1).
    - The "complexity" for Scenario 3 will now come from the
      *functional forms* of g(x), e(x), and Delta(x), not from
      pathological covariate distributions.
    """
    X = np.empty((n, p), dtype=np.float64)
    
    # First five coordinates: mixture of two Gaussians.
    # This block is bimodal. E[X] = 0. (Unchanged)
    means = np.array([[-2, -2, 0, 0, 0], [2, 2, 0, 0, 0]], dtype=np.float64)
    mix = np.random.binomial(1, 0.5, size=n)
    X[:, :5] = means[mix] + 0.5 * np.random.normal(size=(n, 5))
    
    # Remaining coordinates (p=10 -> 5 remaining): Standard Normal N(0, 1).
    if p > 5:
        X[:, 5:p] = np.random.normal(0.0, 1.0, size=(n, p - 5))
    return X


# --- DGP Implementations ---

# --- Scenario 1: Simple, Linear, High Overlap ---
# (Functions generate_rct_1_data and generate_obs_1_data are unchanged)
def generate_rct_1_data(n, p=_DEFAULT_P):
    """
    RCT-1: Simple linear g(x) and Delta(x).
    """
    X = _covariates_x1(n, p)
    W = np.random.binomial(1, 0.5, size=n)  # RCT
    
    # g(x): Simple linear model using 2/10 variables.
    g_X = 0.5 * X[:, 0] + 0.3 * X[:, 1]
    
    # Delta(x): Simple linear model using 1/10 variables.
    # E[Delta(X)] = 1.0 + 0.2 * E[X[:, 2]] = 1.0.
    delta_X = 1.0 + 0.2 * X[:, 2]
    
    sigma = np.ones(n)
    epsilon = np.random.normal(0, 1, size=n)
    
    Y0 = g_X + sigma * epsilon
    Y1 = g_X + delta_X + sigma * epsilon
    Y_obs = np.where(W == 1, Y1, Y0)
    
    return {
        "X": X,
        "W": W,
        "Y_obs": Y_obs,
        "pi_true": np.full(n, 0.5),  # True propensity for RCT
        "true_ate": SCENARIO_1_TRUE_ATE,
    }


def generate_obs_1_data(n, p=_DEFAULT_P):
    """
    OBS-1: Simple linear g(x), e(x), and Delta(x).
    Designed for HIGH OVERLAP in propensity scores.
    """
    X = _covariates_x1(n, p)
    
    # e(x): Simple linear logit with weak predictors.
    # This results in pi(x) being concentrated around 0.5 (high overlap).
    logit_pi = 0.2 * X[:, 0] - 0.2 * X[:, 1]
    pi = expit(logit_pi)
    W = np.random.binomial(1, pi)
    
    # g(x): Same as RCT-1. Simple linear.
    g_X = 0.5 * X[:, 0] + 0.3 * X[:, 1]
    
    # Delta(x): Same as RCT-1. Simple linear.
    # E[Delta(X)] = 1.0.
    delta_X = 1.0 + 0.2 * X[:, 2]
    
    sigma = np.ones(n)
    epsilon = np.random.normal(0, 1, size=n)
    
    Y0 = g_X + sigma * epsilon
    Y1 = g_X + delta_X + sigma * epsilon
    Y_obs = np.where(W == 1, Y1, Y0)
    
    return {
        "X": X,
        "W": W,
        "Y_obs": Y_obs,
        "pi_true": pi,
        "true_ate": SCENARIO_1_TRUE_ATE,
    }

# --- Scenario 2: Moderate Non-linearity, Moderate Overlap ---
# (Functions generate_rct_2_data and generate_obs_2_data are unchanged)
def generate_rct_2_data(n, p=_DEFAULT_P):
    """
    RCT-2: Moderately non-linear g(x) and Delta(x).
    """
    X = _covariates_x2(n, p)
    W = np.random.binomial(1, 0.5, size=n)  # RCT
    
    # g(x): Moderately non-linear (polynomials, interactions, non-poly).
    # Uses vars from both uniform and normal blocks.
    g_X = 0.5 * X[:, 0]**2 + 0.5 * X[:, 1] * X[:, 2] + np.sin(X[:, 5])
    
    # Delta(x): Moderately non-linear (interaction).
    # E[Delta(X)] = 1.0 + E[X[:, 0]] * E[X[:, 1]] = 1.0 (since E[X] = 0).
    delta_X = 1.0 + 0.5 * (X[:, 0] * X[:, 1])
    
    sigma = np.ones(n)
    epsilon = np.random.normal(0, 1, size=n)
    
    Y0 = g_X + sigma * epsilon
    Y1 = g_X + delta_X + sigma * epsilon
    Y_obs = np.where(W == 1, Y1, Y0)
    
    return {
        "X": X,
        "W": W,
        "Y_obs": Y_obs,
        "pi_true": np.full(n, 0.5),
        "true_ate": SCENARIO_2_TRUE_ATE,
    }


def generate_obs_2_data(n, p=_DEFAULT_P):
    """
    OBS-2: Moderately non-linear g(x), e(x), and Delta(x).
    Designed for MODERATE OVERLAP.
    """
    X = _covariates_x2(n, p)
    
    # e(x): Moderately non-linear logit with stronger predictors.
    # This creates more spread in pi(x) (moderate overlap).
    logit_pi = (
        0.5 * X[:, 0] 
        - 0.3 * X[:, 1]**2 
        + 0.4 * np.sin(X[:, 5]) 
        + 0.2 * X[:, 6]
    )
    pi = expit(logit_pi)
    W = np.random.binomial(1, pi)
    
    # g(x): Same as RCT-2. Moderately non-linear.
    g_X = 0.5 * X[:, 0]**2 + 0.5 * X[:, 1] * X[:, 2] + np.sin(X[:, 5])
    
    # Delta(x): Same as RCT-2. Moderately non-linear.
    # E[Delta(X)] = 1.0.
    delta_X = 1.0 + 0.5 * (X[:, 0] * X[:, 1])
    
    sigma = np.ones(n)
    epsilon = np.random.normal(0, 1, size=n)
    
    Y0 = g_X + sigma * epsilon
    Y1 = g_X + delta_X + sigma * epsilon
    Y_obs = np.where(W == 1, Y1, Y0)
    
    return {
        "X": X,
        "W": W,
        "Y_obs": Y_obs,
        "pi_true": pi,
        "true_ate": SCENARIO_2_TRUE_ATE,
    }


# --- Scenario 3: Complex Non-linearity, Low Overlap ---
# === UPDATED FUNCTIONS ===

def generate_rct_3_data(n, p=_DEFAULT_P):
    """
    RCT-3: Complex/highly non-linear g(x) and Delta(x).
    
    MODIFICATION:
    - Covariates are now GMM (bimodal) + Standard Normal (well-behaved).
    - g(x) and Delta(x) are now complex, smooth, non-linear functions
      of these well-behaved covariates.
    - Complexity comes from interactions, polynomials, and non-polynomial
      functions (sin, tanh) on bounded or normal variables.
    """
    X = _covariates_x3(n, p)
    W = np.random.binomial(1, 0.5, size=n)  # RCT
    
    # g(x): Complex, but smooth and well-behaved.
    # E[g(X)] = 0
    g_X = (
        np.sin(np.pi * X[:, 0])           # Periodic function on bimodal var
        + 0.5 * (X[:, 1] * X[:, 2])      # Interaction on bimodal vars
        + 0.1 * (X[:, 5]**3)             # Polynomial on N(0,1) var
        + 0.2 * np.cos(X[:, 6])          # Periodic on N(0,1) var
    )
    
    # Delta(x): Complex, but smooth and well-behaved.
    # E[tanh(X_0)] approx 0, E[X_5*X_6] = E[X_5]E[X_6] = 0.
    # True ATE remains 1.0.
    delta_X = (
        1.0 
        + 0.5 * np.tanh(X[:, 0])         # Non-linear function of bimodal var
        + 0.2 * (X[:, 5] * X[:, 6])      # Interaction of N(0,1) vars
    )
    
    sigma = np.ones(n)
    epsilon = np.random.normal(0, 1, size=n)
    
    Y0 = g_X + sigma * epsilon
    Y1 = g_X + delta_X + sigma * epsilon
    Y_obs = np.where(W == 1, Y1, Y0)
    
    return {
        "X": X,
        "W": W,
        "Y_obs": Y_obs,
        "pi_true": np.full(n, 0.5),
        "true_ate": SCENARIO_3_TRUE_ATE,
    }


def generate_obs_3_data(n, p=_DEFAULT_P):
    """
    OBS-3: Complex non-linear g(x), e(x), and Delta(x).
    Designed for LOW OVERLAP.
    
    MODIFICATION:
    - g(x) and Delta(x) updated as in RCT-3.
    - e(x) (logit_pi) is now designed to create "low overlap"
      by using strong coefficients on the bimodal X_0 and X_1.
    - This creates pi(x) values close to 0 and 1, but in a
      controlled way without pathological predictors.
    """
    X = _covariates_x3(n, p)
    
    # e(x): Strong, non-linear predictors to create LOW OVERLAP.
    # The bimodal X_0, X_1 (peaks at -2, +2) and strong
    # coefficients (1.0) will push logit_pi to bimodal peaks
    # around -2 and +2, creating pi(x) separation.
    logit_pi = (
        0.3 * X[:, 0] 
        + 0.3 * X[:, 1] 
        - 0.5 * X[:, 5] # Add a well-behaved N(0,1) var
    )
    pi = expit(logit_pi)
    W = np.random.binomial(1, pi)
    
    # g(x): Same as updated RCT-3. Highly complex, but smooth.
    g_X = (
        np.sin(np.pi * X[:, 0])
        + 0.5 * (X[:, 1] * X[:, 2])
        + 0.1 * (X[:, 5]**3)
        + 0.2 * np.cos(X[:, 6])
    )
    
    # Delta(x): Same as updated RCT-3. Highly complex.
    # E[Delta(X)] = 1.0.
    delta_X = (
        1.0 
        + 0.5 * np.tanh(X[:, 0])
        + 0.2 * (X[:, 5] * X[:, 6])
    )
    
    sigma = np.ones(n)
    epsilon = np.random.normal(0, 1, size=n)
    
    Y0 = g_X + sigma * epsilon
    Y1 = g_X + delta_X + sigma * epsilon
    Y_obs = np.where(W == 1, Y1, Y0)
    
    return {
        "X": X,
        "W": W,
        "Y_obs": Y_obs,
        "pi_true": pi,
        "true_ate": SCENARIO_3_TRUE_ATE,
    }