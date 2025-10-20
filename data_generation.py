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
    RCT-3 (Revised): High, Non-Linear Heterogeneity
    
    Objective: To create a scenario with high influence heterogeneity that is explicitly 
    disconnected from covariate leverage, specifically favoring the OS-DML method.
    
    Data Generating Process:
    - Covariates: p = 100, X ~ U(-2, 2)^p
    - Treatment: W ~ Bern(0.5)
    - Nuisance: g(X) = cos(πX_1) + sin(πX_2)
    - CATE: Δ(X) = 1.0 + 10 * 1{|X_1| < 0.2 and |X_2| < 0.2}
    - Error: σ_a(X) = 1 + 4 * 1{|X_3| < 0.2} (heteroscedastic)
    - True ATE: τ_0 = E[Δ(X)] = 1.0 + 10 × (0.4/4) × (0.4/4) = 1.1
    """
    X = np.random.uniform(-2, 2, size=(n, p))
    W = np.random.binomial(1, 0.5, n)
    
    # Nuisance function: cos(πX_1) + sin(πX_2)
    g_X = np.cos(np.pi * X[:, 0]) + np.sin(np.pi * X[:, 1])
    
    # CATE (Conditional Average Treatment Effect) - high non-linear heterogeneity
    # Large effect for central subgroup: |X_1| < 0.2 and |X_2| < 0.2 (1% of population)
    indicator = (np.abs(X[:, 0]) < 0.2) & (np.abs(X[:, 1]) < 0.2)
    delta_X = 1.0 + 10.0 * indicator.astype(float)
    
    # Error structure (heteroscedastic)
    # Higher variance in central region where leverage is typically lowest
    error_indicator = np.abs(X[:, 2]) < 0.2
    sigma_a_X = 1 + 4.0 * error_indicator.astype(float)
    epsilon_a = np.random.normal(0, 1, n)
    
    Y0 = g_X + sigma_a_X * epsilon_a
    Y1 = g_X + delta_X + sigma_a_X * epsilon_a
    Y_obs = np.where(W == 1, Y1, Y0)
    
    # True ATE: E[1.0 + 10 * 1{|X_1| < 0.2 and |X_2| < 0.2}]
    # P(|X_1| < 0.2) = P(|X_2| < 0.2) = 0.4 / 4 = 0.1
    # P(|X_1| < 0.2 and |X_2| < 0.2) = 0.1 × 0.1 = 0.01
    true_ate = 1.0 + 10.0 * 0.01
    
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
    OBS-2 (Revised): Moderate Heterogeneity with Stable Nuisances
    
    Objective: To create a scenario with moderate, structured heterogeneity while 
    ensuring that the variance estimation is stable, leading to reasonable confidence 
    interval coverage for all methods.
    
    Data Generating Process:
    - Covariates: p = 50, X ~ U(-2, 2)^p
    - Propensity: logit(e(X)) = 0.5X_1 - 0.6X_2 + 0.2X_3
    - Nuisance: g(X) = 2 + 0.5(X_1 + X_2)
    - CATE: Δ(X) = 2.0 + 0.75X_2^2
    - Error: σ_a(X) = 2.5 (increased homoscedastic noise)
    - True ATE: τ_0 = E[2.0 + 0.75X_2^2] = 2.0 + 0.75 × (4/3) = 3.0
    """
    X = np.random.uniform(-2, 2, size=(n, p))
    
    # Propensity score model
    logit_pi = 0.5 * X[:, 0] - 0.6 * X[:, 1] + 0.2 * X[:, 2]
    pi = expit(logit_pi)
    W = np.random.binomial(1, pi)
    
    # Nuisance function (simplified and linear)
    g_X = 2 + 0.5 * (X[:, 0] + X[:, 1])
    
    # CATE (Conditional Average Treatment Effect) - simple quadratic function
    delta_X = 2.0 + 0.75 * X[:, 1]**2
    
    # Error structure (increased homoscedastic noise)
    sigma_a_X = np.full(n, 2.5)
    epsilon_a = np.random.normal(0, 1, n)
    
    Y0 = g_X + sigma_a_X * epsilon_a
    Y1 = g_X + delta_X + sigma_a_X * epsilon_a
    Y_obs = np.where(W == 1, Y1, Y0)
    
    # True ATE: E[2.0 + 0.75X_2^2] = 2.0 + 0.75 * E[X_2^2]
    # For X_2 ~ U(-2, 2), E[X_2^2] = Var(X_2) + E[X_2]^2 = (4^2/12) + 0^2 = 16/12 = 4/3
    true_ate = 2.0 + 0.75 * (4/3)
    
    return {"X": X, "W": W, "Y_obs": Y_obs, "pi_true": pi, "true_ate": true_ate}

def generate_obs_3_data(n, p):
    """
    OBS-3 (Revised): High Heterogeneity without Positivity Violation
    
    Objective: To model a challenging scenario with strong confounding and complex 
    interactions, but with well-behaved propensity scores to ensure that inference 
    is valid for all methods.
    
    Data Generating Process:
    - Risk Score: S(X) = (X_1 + X_2 + X_3 + X_4) / 4
    - Covariates: p = 100, X ~ U(-2, 2)^p
    - Propensity: logit(e(X)) = -0.5 + 1.0S(X) (moderated coefficients)
    - Nuisance: g(X) = 2 + 2S(X)^2 + 0.5X_5
    - CATE: Δ(X) = 1.2 + 12.8 * 1{X_6 > 1.5 and X_7 > 1.5}
    - Error: σ_a(X) = 1 + |S(X)| (moderately heteroscedastic)
    - True ATE: τ_0 = E[1.2 + 12.8 * 1{X_6 > 1.5 and X_7 > 1.5}] = 1.4
    """
    X = np.random.uniform(-2, 2, size=(n, p))
    
    # Risk score
    S_X = (X[:, 0] + X[:, 1] + X[:, 2] + X[:, 3]) / 4
    
    # Propensity score model (moderated coefficients to ensure propensity scores in (0.07, 0.82))
    logit_pi = -0.5 + 1.0 * S_X
    pi = expit(logit_pi)
    W = np.random.binomial(1, pi)
    
    # Nuisance function
    g_X = 2 + 2 * S_X**2 + 0.5 * X[:, 4]
    
    # CATE (Conditional Average Treatment Effect) - sparse high-impact interaction
    # Large effect for small subgroup: X_6 > 1.5 and X_7 > 1.5 (disconnected from risk score)
    indicator = (X[:, 5] > 1.5) & (X[:, 6] > 1.5)  # X_6 and X_7 (0-indexed)
    delta_X = 1.2 + 12.8 * indicator.astype(float)
    
    # Error structure (moderately heteroscedastic)
    sigma_a_X = 1 + np.abs(S_X)
    epsilon_a = np.random.normal(0, 1, n)
    
    Y0 = g_X + sigma_a_X * epsilon_a
    Y1 = g_X + delta_X + sigma_a_X * epsilon_a
    Y_obs = np.where(W == 1, Y1, Y0)
    
    # True ATE: E[1.2 + 12.8 * 1{X_6 > 1.5 and X_7 > 1.5}]
    # P(X_6 > 1.5) = P(X_7 > 1.5) = (2 - 1.5) / 4 = 0.5 / 4 = 0.125
    # P(X_6 > 1.5 and X_7 > 1.5) = 0.125 × 0.125 = 0.015625
    true_ate = 1.2 + 12.8 * 0.015625
    
    return {"X": X, "W": W, "Y_obs": Y_obs, "pi_true": pi, "true_ate": true_ate}

