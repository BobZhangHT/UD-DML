# -*- coding: utf-8 -*-
"""
data_generation.py

Generates simulated data for the DGPs described in the latest OS-DML manuscript.
DGPs are now modified to create scenarios where influence functions are highly
variable but leverage scores are relatively uniform, highlighting the advantage of OS.
All content is in English.
"""
import numpy as np
from scipy.special import expit
from scipy.optimize import root_scalar
from scipy.stats import norm, t

def generate_rct_s_data(n, p):
    """
    Generates data for RCT-S scenario.
    Modified for high influence function variability and low leverage score variability.
    """
    # Covariates are now from a uniform distribution to reduce high-leverage points
    X = np.random.uniform(-2, 2, size=(n, p))
    W = np.random.binomial(1, 0.5, n)
    
    g_X = 0.5 * X[:, 0] + 0.5 * X[:, 1]
    # Treatment effect is now highly dependent on a non-linear term, creating variable influence
    delta_X = 0.3 + 1.5 * np.sin(X[:, 0] * np.pi)**2 
    # Heteroskedasticity is tied to a different variable to amplify influence variability
    sigma_a_X = 0.5 + 1.0 * (np.abs(X[:, 2]) > 1.5)
    
    epsilon_a = np.random.normal(0, 1, n)
    
    Y0 = g_X + sigma_a_X * epsilon_a
    Y1 = g_X + delta_X + sigma_a_X * epsilon_a
    Y_obs = np.where(W == 1, Y1, Y0)
    
    # Precise ATE calculation using numerical integration for the sine term
    from scipy.integrate import quad
    # E[sin^2(pi*U)] where U ~ Unif(-2, 2) is 1/2
    true_ate = 0.3 + 1.5 * 0.5 
    
    return {"X": X, "W": W, "Y_obs": Y_obs, "pi_true": np.full(n, 0.5), "true_ate": true_ate}

def generate_rct_c_data(n, p):
    """
    Generates data for RCT-C scenario.
    Modified for high influence function variability.
    """
    X = np.random.uniform(-2, 2, size=(n, p))
    W = np.random.binomial(1, 0.3, n)
    
    g_X = 0.6 * np.sin(X[:, 0]) + 0.3 * (X[:, 1]**2 - 1) + 0.2 * X[:, 2] * X[:, 3]
    # Effect heterogeneity now stronger and more localized
    delta_X = 0.2 + 2.0 * ((X[:, 4] > 1.8) | (X[:, 5] < -1.8))
    sigma_a_X = 0.5 + 1.5 * (np.abs(X[:, 1]) > 1.9)
    
    is_t_dist = np.random.binomial(1, 0.1, n).astype(bool)
    epsilon_a = np.zeros(n)
    epsilon_a[is_t_dist] = t.rvs(df=3, size=np.sum(is_t_dist)) / np.sqrt(3)
    epsilon_a[~is_t_dist] = np.random.normal(0, 1, size=np.sum(~is_t_dist))
    
    Y0 = g_X + sigma_a_X * epsilon_a
    Y1 = g_X + delta_X + sigma_a_X * epsilon_a
    Y_obs = np.where(W == 1, Y1, Y0)
    
    # ATE from Unif(-2, 2)
    prob_event = 1 - ((1.8 - (-2)) / 4) * ((1.8 - (-2)) / 4)
    true_ate = 0.2 + 2.0 * prob_event
    
    return {"X": X, "W": W, "Y_obs": Y_obs, "pi_true": np.full(n, 0.3), "true_ate": true_ate}

def generate_obs_s_data(n, p):
    """
    Generates data for OBS-S scenario.
    Modified for high influence variability.
    """
    X = np.random.uniform(-2, 2, size=(n, p))
    
    # Propensity score with moderate overlap, not strongly related to outcome drivers
    def mean_pi_error(beta0):
        return np.mean(expit(beta0 + 0.5 * X[:, 10] + 0.5 * X[:, 11])) - 0.5
    beta0 = root_scalar(mean_pi_error, bracket=[-5, 5], method='brentq').root
    pi = expit(beta0 + 0.5 * X[:, 10] + 0.5 * X[:, 11])
    W = np.random.binomial(1, pi)
    
    g_X = 0.5 * X[:, 0] + 0.5 * X[:, 1]
    delta_X = 0.4 + 1.8 * (X[:, 2] > 1.5)
    
    Y0 = g_X + np.random.normal(0, 1, n)
    Y1 = g_X + delta_X + np.random.normal(0, 1, n)
    Y_obs = np.where(W == 1, Y1, Y0)
    
    # ATE from Unif(-2, 2)
    true_ate = 0.4 + 1.8 * ((2 - 1.5) / 4)
    
    return {"X": X, "W": W, "Y_obs": Y_obs, "pi_true": pi, "true_ate": true_ate}

def generate_obs_c_data(n, p):
    """
    Generates data for OBS-C scenario.
    Modified for high influence variability.
    """
    X = np.random.uniform(-2, 2, size=(n, p))
    
    def mean_pi_error(beta0):
        return np.mean(expit(beta0 + 0.4 * np.sum(X[:, 10:14], axis=1))) - 0.4
    beta0 = root_scalar(mean_pi_error, bracket=[-5, 5], method='brentq').root
    pi = expit(beta0 + 0.4 * np.sum(X[:, 10:14], axis=1))
    W = np.random.binomial(1, pi)
    
    g_X = 0.4 * np.sin(X[:, 0]) + 0.3 * (X[:, 1]**2 - 1) + 0.2 * X[:, 3] * X[:, 4]
    delta_X = 0.2 + 2.5 * ((np.abs(X[:, 0]) > 1.8) & (X[:, 1] > 1.5))
    sigma_a_X = 0.5 + 1.0 * (np.abs(X[:, 1]) > 1.8) + 1.0 * (X[:, 2] < -1.8)

    Y0 = g_X + sigma_a_X * np.random.normal(0, 1, n)
    Y1 = g_X + delta_X + sigma_a_X * np.random.normal(0, 1, n)
    Y_obs = np.where(W == 1, Y1, Y0)
    
    # ATE from Unif(-2, 2)
    prob_event = (2 * (2 - 1.8) / 4) * ((2 - 1.5) / 4)
    true_ate = 0.2 + 2.5 * prob_event
    
    return {"X": X, "W": W, "Y_obs": Y_obs, "pi_true": pi, "true_ate": true_ate}

