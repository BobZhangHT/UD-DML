# -*- coding: utf-8 -*-
"""
data_generation.py

Generates simulated data for the 6 DGPs described in the latest OS-DML manuscript.
Implements RCT-1, RCT-2, RCT-3, OBS-1, OBS-2, OBS-3 scenarios with varying
levels of heterogeneity to test the performance of OS-DML across different
data generating processes.

All content is in English.
"""
import numpy as np
from scipy.special import expit
from scipy.optimize import root_scalar
from scipy.stats import norm, t

def generate_rct_1_data(n, p):
    """
    RCT-1: Low Heterogeneity
    
    Objective: To simulate a large, simple trial with a nearly constant treatment effect 
    and well-behaved outcomes. This scenario represents a baseline where uniform 
    subsampling is expected to be near-optimal.
    
    Data Generating Process:
    - Covariates: p = 20, X ~ U(-2, 2)^p
    - Treatment: W ~ Bern(0.5)
    - Nuisance: g(X) = 2 + 0.5(X_1 + X_2 - X_3)
    - CATE: Δ(X) = 1.0 + 0.1X_1
    - Error: σ_a(X) = 1 (homoscedastic)
    - True ATE: τ_0 = E[Δ(X)] = 1.0
    """
    X = np.random.uniform(-2, 2, size=(n, p))
    W = np.random.binomial(1, 0.5, n)
    
    # Nuisance function
    g_X = 2 + 0.5 * (X[:, 0] + X[:, 1] - X[:, 2])
    
    # CATE (Conditional Average Treatment Effect)
    delta_X = 1.0 + 0.1 * X[:, 0]
    
    # Error structure (homoscedastic)
    sigma_a_X = np.ones(n)
    epsilon_a = np.random.normal(0, 1, n)
    
    Y0 = g_X + sigma_a_X * epsilon_a
    Y1 = g_X + delta_X + sigma_a_X * epsilon_a
    Y_obs = np.where(W == 1, Y1, Y0)
    
    # True ATE: E[1.0 + 0.1X_1] = 1.0 + 0.1 * E[X_1] = 1.0 + 0.1 * 0 = 1.0
    true_ate = 1.0
    
    return {"X": X, "W": W, "Y_obs": Y_obs, "pi_true": np.full(n, 0.5), "true_ate": true_ate}

def generate_rct_2_data(n, p):
    """
    RCT-2: Moderate Heterogeneity
    
    Objective: To model a trial where treatment efficacy varies smoothly across 
    patient subgroups defined by continuous biomarkers. This is a common scenario 
    in modern clinical research.
    
    Data Generating Process:
    - Covariates: p = 50, X ~ U(-2, 2)^p
    - Treatment: W ~ Bern(0.5)
    - Nuisance: g(X) = sin(πX_1) + cos(πX_2)
    - CATE: Δ(X) = 1.5 + sin(πX_1)
    - Error: σ_a(X) = 1 + 0.5|X_3| (moderately heteroscedastic)
    - True ATE: τ_0 = E[Δ(X)] = 1.5
    """
    X = np.random.uniform(-2, 2, size=(n, p))
    W = np.random.binomial(1, 0.5, n)
    
    # Nuisance function
    g_X = np.sin(np.pi * X[:, 0]) + np.cos(np.pi * X[:, 1])
    
    # CATE (Conditional Average Treatment Effect)
    delta_X = 1.5 + np.sin(np.pi * X[:, 0])
    
    # Error structure (moderately heteroscedastic)
    sigma_a_X = 1 + 0.5 * np.abs(X[:, 2])
    epsilon_a = np.random.normal(0, 1, n)
    
    Y0 = g_X + sigma_a_X * epsilon_a
    Y1 = g_X + delta_X + sigma_a_X * epsilon_a
    Y_obs = np.where(W == 1, Y1, Y0)
    
    # True ATE: E[1.5 + sin(πX_1)] = 1.5 + E[sin(πX_1)] = 1.5 + 0 = 1.5
    true_ate = 1.5
    
    return {"X": X, "W": W, "Y_obs": Y_obs, "pi_true": np.full(n, 0.5), "true_ate": true_ate}

def generate_rct_3_data(n, p):
    """
    RCT-3: High Heterogeneity
    
    Objective: To simulate a precision medicine scenario where a treatment provides 
    a substantial benefit but only for a very small, well-defined subgroup (e.g., 
    based on a rare genetic marker), while also accounting for extreme outlier responses.
    
    Data Generating Process:
    - Covariates: p = 100, X ~ U(-2, 2)^p
    - Treatment: W ~ Bern(0.5)
    - Nuisance: g(X) = 0.5(X_1^2 - X_2^2) + cos(πX_3)
    - CATE: Δ(X) = 1.0 + 20 * 1{X_1 > 1.8 and X_2 < -1.8}
    - Error: σ_a(X) = 1, ε_a ~ 0.95N(0,1) + 0.05t_3(0,1) (homoscedastic with heavy tails)
    - True ATE: τ_0 = E[Δ(X)] = 1.0 + 20 × 0.0025 = 1.05
    """
    X = np.random.uniform(-2, 2, size=(n, p))
    W = np.random.binomial(1, 0.5, n)
    
    # Nuisance function
    g_X = 0.5 * (X[:, 0]**2 - X[:, 1]**2) + np.cos(np.pi * X[:, 2])
    
    # CATE (Conditional Average Treatment Effect) - high heterogeneity
    # Large effect for tiny subgroup: X_1 > 1.8 and X_2 < -1.8
    indicator = (X[:, 0] > 1.8) & (X[:, 1] < -1.8)
    delta_X = 1.0 + 20.0 * indicator.astype(float)
    
    # Error structure (homoscedastic with heavy tails)
    sigma_a_X = np.ones(n)
    # Mixture: 95% normal, 5% t-distribution with 3 df
    is_t_dist = np.random.binomial(1, 0.05, n).astype(bool)
    epsilon_a = np.zeros(n)
    epsilon_a[is_t_dist] = t.rvs(df=3, size=np.sum(is_t_dist))
    epsilon_a[~is_t_dist] = np.random.normal(0, 1, size=np.sum(~is_t_dist))
    
    Y0 = g_X + sigma_a_X * epsilon_a
    Y1 = g_X + delta_X + sigma_a_X * epsilon_a
    Y_obs = np.where(W == 1, Y1, Y0)
    
    # True ATE: E[1.0 + 20 * 1{X_1 > 1.8 and X_2 < -1.8}]
    # P(X_1 > 1.8 and X_2 < -1.8) = P(X_1 > 1.8) * P(X_2 < -1.8) = 0.05 * 0.05 = 0.0025
    true_ate = 1.0 + 20.0 * 0.0025
    
    return {"X": X, "W": W, "Y_obs": Y_obs, "pi_true": np.full(n, 0.5), "true_ate": true_ate}

def generate_obs_1_data(n, p):
    """
    OBS-1: Low Heterogeneity
    
    Objective: To simulate a well-controlled observational study with weak confounding 
    and a near-constant treatment effect. In this simulation, propensity scores are 
    designed to be far from the boundaries, and all relationships are simple.
    
    Data Generating Process:
    - Covariates: p = 20, X ~ U(-2, 2)^p
    - Propensity: logit(e(X)) = -0.4 + 0.5X_1 - 0.3X_2
    - Nuisance: g(X) = 2 + X_1 + X_2
    - CATE: Δ(X) = 1.0 + 0.1X_1
    - Error: σ_a(X) = 1 (homoscedastic)
    - True ATE: τ_0 = E[Δ(X)] = 1.0
    """
    X = np.random.uniform(-2, 2, size=(n, p))
    
    # Propensity score model
    logit_pi = -0.4 + 0.5 * X[:, 0] - 0.3 * X[:, 1]
    pi = expit(logit_pi)
    W = np.random.binomial(1, pi)
    
    # Nuisance function
    g_X = 2 + X[:, 0] + X[:, 1]
    
    # CATE (Conditional Average Treatment Effect)
    delta_X = 1.0 + 0.1 * X[:, 0]
    
    # Error structure (homoscedastic)
    sigma_a_X = np.ones(n)
    epsilon_a = np.random.normal(0, 1, n)
    
    Y0 = g_X + sigma_a_X * epsilon_a
    Y1 = g_X + delta_X + sigma_a_X * epsilon_a
    Y_obs = np.where(W == 1, Y1, Y0)
    
    # True ATE: E[1.0 + 0.1X_1] = 1.0 + 0.1 * E[X_1] = 1.0 + 0.1 * 0 = 1.0
    true_ate = 1.0
    
    return {"X": X, "W": W, "Y_obs": Y_obs, "pi_true": pi, "true_ate": true_ate}

def generate_obs_2_data(n, p):
    """
    OBS-2: Moderate Heterogeneity
    
    Objective: To model a common scenario in EHR-based research where treatment 
    decisions follow non-linear patterns, and the treatment effect itself is 
    modified by a primary confounder.
    
    Data Generating Process:
    - Covariates: p = 50, X ~ U(-2, 2)^p
    - Propensity: logit(e(X)) = 0.5X_1 - 0.6X_2^2 + 0.2X_3
    - Nuisance: g(X) = sin(πX_1) + X_2^2
    - CATE: Δ(X) = 2 + 1.5X_2^2
    - Error: σ_a(X) = 1 (homoscedastic)
    - True ATE: τ_0 = E[2 + 1.5X_2^2] = 2 + 1.5 × (4/3) = 4.0
    """
    X = np.random.uniform(-2, 2, size=(n, p))
    
    # Propensity score model
    logit_pi = 0.5 * X[:, 0] - 0.6 * X[:, 1]**2 + 0.2 * X[:, 2]
    pi = expit(logit_pi)
    W = np.random.binomial(1, pi)
    
    # Nuisance function
    g_X = np.sin(np.pi * X[:, 0]) + X[:, 1]**2
    
    # CATE (Conditional Average Treatment Effect)
    delta_X = 2 + 1.5 * X[:, 1]**2
    
    # Error structure (homoscedastic)
    sigma_a_X = np.ones(n)
    epsilon_a = np.random.normal(0, 1, n)
    
    Y0 = g_X + sigma_a_X * epsilon_a
    Y1 = g_X + delta_X + sigma_a_X * epsilon_a
    Y_obs = np.where(W == 1, Y1, Y0)
    
    # True ATE: E[2 + 1.5X_2^2] = 2 + 1.5 * E[X_2^2]
    # For X_2 ~ U(-2, 2), E[X_2^2] = Var(X_2) + E[X_2]^2 = (4^2/12) + 0^2 = 16/12 = 4/3
    true_ate = 2 + 1.5 * (4/3)
    
    return {"X": X, "W": W, "Y_obs": Y_obs, "pi_true": pi, "true_ate": true_ate}

def generate_obs_3_data(n, p):
    """
    OBS-3: High Heterogeneity
    
    Objective: To model a challenging "confounding by indication" scenario where 
    patient severity strongly predicts treatment assignment, leading to a near 
    violation of the positivity assumption.
    
    Data Generating Process:
    - Risk Score: S(X) = (X_1 + X_2 + X_3 + X_4) / 2
    - Covariates: p = 100, X ~ U(-2, 2)^p
    - Propensity: logit(e(X)) = -2.0 + 2.5S(X)
    - Nuisance: g(X) = 2 + 3S(X)^2 + 0.5X_5
    - CATE: Δ(X) = 1.5 - S(X)
    - Error: σ_a(X) = 1 + |S(X)| (heteroscedastic)
    - True ATE: τ_0 = E[1.5 - S(X)] = 1.5
    """
    X = np.random.uniform(-2, 2, size=(n, p))
    
    # Risk score
    S_X = (X[:, 0] + X[:, 1] + X[:, 2] + X[:, 3]) / 2
    
    # Propensity score model (confounding by indication)
    logit_pi = -2.0 + 2.5 * S_X
    pi = expit(logit_pi)
    W = np.random.binomial(1, pi)
    
    # Nuisance function
    g_X = 2 + 3 * S_X**2 + 0.5 * X[:, 4]
    
    # CATE (Conditional Average Treatment Effect)
    delta_X = 1.5 - S_X
    
    # Error structure (heteroscedastic)
    sigma_a_X = 1 + np.abs(S_X)
    epsilon_a = np.random.normal(0, 1, n)
    
    Y0 = g_X + sigma_a_X * epsilon_a
    Y1 = g_X + delta_X + sigma_a_X * epsilon_a
    Y_obs = np.where(W == 1, Y1, Y0)
    
    # True ATE: E[1.5 - S(X)] = 1.5 - E[S(X)]
    # E[S(X)] = E[(X_1 + X_2 + X_3 + X_4)/2] = (0 + 0 + 0 + 0)/2 = 0
    true_ate = 1.5
    
    return {"X": X, "W": W, "Y_obs": Y_obs, "pi_true": pi, "true_ate": true_ate}

