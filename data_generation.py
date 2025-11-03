# -*- coding: utf-8 -*-
"""
data_generation.py

Implements the six DGPs described in UD_DML.pdf:
three covariate structures (DGP-X1/2/3) crossed with RCT/OBS scenarios.
All content is in English.
"""
import numpy as np
from scipy.special import expit

_DEFAULT_P = 10
_SIGMA_CORR = 1.5
_CORR = 0.5


def _covariates_x1(n, p):
    """Independent uniform covariates."""
    return np.random.uniform(-2, 2, size=(n, p))


def _covariates_x2(n, p):
    """Mixed marginals: first half uniform, second half normal."""
    X = np.empty((n, p), dtype=np.float64)
    split = min(max(p // 2, 1), p)
    X[:, :split] = np.random.uniform(-2, 2, size=(n, split))
    if split < p:
        X[:, split:] = np.random.normal(0.0, _SIGMA_CORR, size=(n, p - split))
    return X


def _covariates_x3(n, p):
    """Correlated first block, remaining uniform."""
    X = np.empty((n, p), dtype=np.float64)
    block = min(3, p)
    if block > 0:
        cov = np.full((block, block), _CORR * (_SIGMA_CORR ** 2))
        np.fill_diagonal(cov, _SIGMA_CORR ** 2)
        X[:, :block] = np.random.multivariate_normal(
            mean=np.zeros(block), cov=cov, size=n
        )
    if block < p:
        X[:, block:] = np.random.uniform(-2, 2, size=(n, p - block))
    return X


def _true_ate_scenario3(sample_size: int = 1_000_000, seed: int = 20251021) -> float:
    """Monte Carlo approximation of the Scenario 3 ATE using 10^6 draws."""
    rng = np.random.default_rng(seed)
    cov = np.full((3, 3), _CORR * (_SIGMA_CORR ** 2))
    np.fill_diagonal(cov, _SIGMA_CORR ** 2)
    block = rng.multivariate_normal(mean=np.zeros(3), cov=cov, size=sample_size)
    x1 = block[:, 0]
    x2 = block[:, 1]
    x3 = block[:, 2]
    indicator_term = (x2 > 0).astype(np.float64) * x3
    delta = 2.0 + 1.5 * np.cos(np.pi * x1) + 2.0 * indicator_term
    return float(delta.mean())


SCENARIO3_TRUE_ATE = _true_ate_scenario3()


def generate_rct_1_data(n, p=_DEFAULT_P):
    X = _covariates_x1(n, p)
    W = np.random.binomial(1, 0.5, size=n)
    g_X = 0.5 * X[:, 0] + 0.5 * X[:, 1]
    delta_X = 1.0 + 0.2 * X[:, 0]
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
        "true_ate": 1.0,
    }


def generate_rct_2_data(n, p=_DEFAULT_P):
    X = _covariates_x2(n, p)
    W = np.random.binomial(1, 0.5, size=n)
    x1 = X[:, 0]
    x6 = X[:, 5] if p > 5 else X[:, -1]
    g_X = 0.5 * x1 + 0.3 * x6
    delta_X = 1.5 + 0.5 * x1 + 0.5 * (X[:, 1] * x6)
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
        "true_ate": 1.5,
    }


def generate_rct_3_data(n, p=_DEFAULT_P):
    X = _covariates_x3(n, p)
    W = np.random.binomial(1, 0.5, size=n)
    x1, x2, x3, x4 = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
    g_X = np.sin(np.pi * x1) + (x2 ** 2) + 0.5 * x4
    delta_X = 2.0 + 1.5 * np.cos(np.pi * x1) + 2.0 * ((x2 > 0).astype(float) * x3)
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
        "true_ate": SCENARIO3_TRUE_ATE,
    }


def generate_obs_1_data(n, p=_DEFAULT_P):
    X = _covariates_x1(n, p)
    logit_pi = -0.4 + 0.3 * X[:, 0] - 0.2 * X[:, 1]
    pi = expit(logit_pi)
    W = np.random.binomial(1, pi)
    g_X = 0.5 * X[:, 0] + 0.5 * X[:, 1]
    delta_X = 1.0 + 0.2 * X[:, 0]
    sigma = np.ones(n)
    epsilon = np.random.normal(0, 1, size=n)
    Y0 = g_X + sigma * epsilon
    Y1 = g_X + delta_X + sigma * epsilon
    Y_obs = np.where(W == 1, Y1, Y0)
    return {"X": X, "W": W, "Y_obs": Y_obs, "pi_true": pi, "true_ate": 1.0}


def generate_obs_2_data(n, p=_DEFAULT_P):
    X = _covariates_x2(n, p)
    x6 = X[:, 5] if p > 5 else X[:, -1]
    logit_pi = -0.2 + 0.3 * X[:, 0] - 0.2 * X[:, 1] + 0.3 * x6
    pi = expit(logit_pi)
    W = np.random.binomial(1, pi)
    g_X = 0.5 * X[:, 0] + 0.3 * x6
    delta_X = 1.5 + 0.5 * X[:, 0] + 0.5 * (X[:, 1] * x6)
    sigma = np.ones(n)
    epsilon = np.random.normal(0, 1, size=n)
    Y0 = g_X + sigma * epsilon
    Y1 = g_X + delta_X + sigma * epsilon
    Y_obs = np.where(W == 1, Y1, Y0)
    return {"X": X, "W": W, "Y_obs": Y_obs, "pi_true": pi, "true_ate": 1.5}


def generate_obs_3_data(n, p=_DEFAULT_P):
    X = _covariates_x3(n, p)
    logit_pi = 0.4 * np.sin(np.pi * X[:, 0]) + 0.3 * (X[:, 1] ** 2) - 0.3 * X[:, 3]
    pi = expit(logit_pi)
    W = np.random.binomial(1, pi)
    x1, x2, x3, x4 = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
    g_X = np.sin(np.pi * x1) + (x2 ** 2) + 0.5 * x4
    delta_X = 2.0 + 1.5 * np.cos(np.pi * x1) + 2.0 * ((x2 > 0).astype(float) * x3)
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
        "true_ate": SCENARIO3_TRUE_ATE,
    }
