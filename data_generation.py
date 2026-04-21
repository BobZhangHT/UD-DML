# -*- coding: utf-8 -*-
"""
data_generation.py — Synthetic data-generating processes (DGPs) for UD-DML.

Implements six DGPs structured as a 2 × 3 factorial design (Table 1):

    ┌──────────┬───────────────────────────────────────────────────────┐
    │          │  Scenario 1          Scenario 2        Scenario 3    │
    │          │  Low heterogeneity   Moderate           High          │
    │          │  High overlap        Moderate overlap   Low overlap   │
    ├──────────┼───────────────────────────────────────────────────────┤
    │ RCT      │  RCT-1               RCT-2              RCT-3       │
    │ OBS      │  OBS-1               OBS-2              OBS-3       │
    └──────────┴───────────────────────────────────────────────────────┘

Common settings across all DGPs:
    * Covariate dimension p = 10.
    * Outcome model: Y = g(X) + W · Δ(X) + ε,  ε ~ N(0, 1).
    * True ATE: θ₀ = E[Δ(X)] = 1.0  (analytically, in all scenarios).

Covariate structures (Section 3.1):
    * X1: X^(d) ~ U[-2, 2]  independently for all d.
    * X2: X^(1..5) ~ U[-2, 2],  X^(6..10) ~ N(0, 1.5²).
    * X3: X^(1..5) ~ GMM(μ₁, μ₂; σ=0.5),  X^(6..10) ~ N(0, 1).
"""

from __future__ import annotations

import numpy as np
from scipy.special import expit

# ── Module Constants ──────────────────────────────────────────────────────

_DEFAULT_P: int = 10
"""Fixed covariate dimension as specified in all DGPs (Table 1)."""

_TRUE_ATE: float = 1.0
"""True Average Treatment Effect, common to all six scenarios."""


# ═══════════════════════════════════════════════════════════════════════════
# Covariate Generators
# ═══════════════════════════════════════════════════════════════════════════


def _covariates_x1(n: int, p: int) -> np.ndarray:
    """X1 (Simple): Independent uniform covariates.

    X^(d) ~ U[-2, 2]  for d = 1, …, p.

    Parameters
    ----------
    n : int
        Sample size.
    p : int
        Covariate dimension.

    Returns
    -------
    ndarray of shape (n, p)
    """
    return np.random.uniform(-2, 2, size=(n, p))


def _covariates_x2(n: int, p: int) -> np.ndarray:
    """X2 (Moderate): Mixed marginals — uniform block + normal block.

    X^(1..5) ~ U[-2, 2],  X^(6..10) ~ N(0, 1.5²).

    Parameters
    ----------
    n, p : int
    """
    X = np.empty((n, p), dtype=np.float64)
    split = min(max(p // 2, 1), p)
    X[:, :split] = np.random.uniform(-2, 2, size=(n, split))
    if split < p:
        X[:, split:] = np.random.normal(0.0, 1.5, size=(n, p - split))
    return X


def _covariates_x3(n: int, p: int) -> np.ndarray:
    """X3 (Complex): Gaussian mixture block + standard-normal block.

    X^(1..5) ~ 0.5·N(μ₁, 0.5²I) + 0.5·N(μ₂, 0.5²I)
        with μ₁ = (-2, -2, 0, 0, 0), μ₂ = (2, 2, 0, 0, 0).
    X^(6..10) ~ N(0, I).

    Parameters
    ----------
    n, p : int
    """
    X = np.empty((n, p), dtype=np.float64)

    # First five: two-component Gaussian mixture (bimodal)
    means = np.array([[-2, -2, 0, 0, 0], [2, 2, 0, 0, 0]], dtype=np.float64)
    component = np.random.binomial(1, 0.5, size=n)
    X[:, :5] = means[component] + 0.5 * np.random.normal(size=(n, 5))

    # Remaining: standard normal
    if p > 5:
        X[:, 5:p] = np.random.normal(0.0, 1.0, size=(n, p - 5))
    return X


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 1 — Low Heterogeneity, High Overlap
# ═══════════════════════════════════════════════════════════════════════════
#
# g(X)  = 0.5·X₁ + 0.3·X₂
# Δ(X)  = 1.0 + 0.2·X₃          →  E[Δ] = 1.0
# e(X)  = expit(0.2·X₁ - 0.2·X₂)     (OBS only)
#


def generate_rct_1_data(n: int, p: int = _DEFAULT_P) -> dict:
    """RCT-1: Simple linear outcome and CATE, W ~ Bern(0.5).

    Returns
    -------
    dict with keys ``X``, ``W``, ``Y_obs``, ``pi_true``, ``true_ate``.
    """
    X = _covariates_x1(n, p)
    W = np.random.binomial(1, 0.5, size=n)

    g = 0.5 * X[:, 0] + 0.3 * X[:, 1]
    delta = 1.0 + 0.2 * X[:, 2]
    eps = np.random.normal(0, 1, size=n)

    Y_obs = g + W * delta + eps
    return {"X": X, "W": W, "Y_obs": Y_obs,
            "pi_true": np.full(n, 0.5), "true_ate": _TRUE_ATE}


def generate_obs_1_data(n: int, p: int = _DEFAULT_P) -> dict:
    """OBS-1: Same as RCT-1 but with confounded treatment (high overlap).

    logit(e(X)) = 0.2·X₁ − 0.2·X₂  →  propensity scores near 0.5.
    """
    X = _covariates_x1(n, p)

    logit_e = 0.2 * X[:, 0] - 0.2 * X[:, 1]
    pi = expit(logit_e)
    W = np.random.binomial(1, pi)

    g = 0.5 * X[:, 0] + 0.3 * X[:, 1]
    delta = 1.0 + 0.2 * X[:, 2]
    eps = np.random.normal(0, 1, size=n)

    Y_obs = g + W * delta + eps
    return {"X": X, "W": W, "Y_obs": Y_obs,
            "pi_true": pi, "true_ate": _TRUE_ATE}


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 2 — Moderate Heterogeneity, Moderate Overlap
# ═══════════════════════════════════════════════════════════════════════════
#
# g(X)  = 0.5·X₁² + 0.5·X₂·X₃ + sin(X₆)
# Δ(X)  = 1.0 + 0.5·X₁·X₂      →  E[Δ] = 1.0
# e(X)  = expit(0.5·X₁ − 0.3·X₂² + 0.4·sin(X₆) + 0.2·X₇)  (OBS only)
#


def generate_rct_2_data(n: int, p: int = _DEFAULT_P) -> dict:
    """RCT-2: Moderate non-linearity, W ~ Bern(0.5)."""
    X = _covariates_x2(n, p)
    W = np.random.binomial(1, 0.5, size=n)

    g = 0.5 * X[:, 0] ** 2 + 0.5 * X[:, 1] * X[:, 2] + np.sin(X[:, 5])
    delta = 1.0 + 0.5 * X[:, 0] * X[:, 1]
    eps = np.random.normal(0, 1, size=n)

    Y_obs = g + W * delta + eps
    return {"X": X, "W": W, "Y_obs": Y_obs,
            "pi_true": np.full(n, 0.5), "true_ate": _TRUE_ATE}


def generate_obs_2_data(n: int, p: int = _DEFAULT_P) -> dict:
    """OBS-2: Moderate non-linearity with confounded treatment.

    logit(e(X)) = 0.5·X₁ − 0.3·X₂² + 0.4·sin(X₆) + 0.2·X₇
    →  moderate overlap.
    """
    X = _covariates_x2(n, p)

    logit_e = (
        0.5 * X[:, 0]
        - 0.3 * X[:, 1] ** 2
        + 0.4 * np.sin(X[:, 5])
        + 0.2 * X[:, 6]
    )
    pi = expit(logit_e)
    W = np.random.binomial(1, pi)

    g = 0.5 * X[:, 0] ** 2 + 0.5 * X[:, 1] * X[:, 2] + np.sin(X[:, 5])
    delta = 1.0 + 0.5 * X[:, 0] * X[:, 1]
    eps = np.random.normal(0, 1, size=n)

    Y_obs = g + W * delta + eps
    return {"X": X, "W": W, "Y_obs": Y_obs,
            "pi_true": pi, "true_ate": _TRUE_ATE}


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 3 — High Heterogeneity, Low Overlap
# ═══════════════════════════════════════════════════════════════════════════
#
# g(X)  = sin(π·X₁) + 0.5·X₂·X₃ + 0.1·X₆³ + 0.2·cos(X₇)
# Δ(X)  = 1.0 + 0.5·tanh(X₁) + 0.2·X₆·X₇   →  E[Δ] = 1.0
# e(X)  = expit(0.3·X₁ + 0.3·X₂ − 0.5·X₆)        (OBS only)
#


def generate_rct_3_data(n: int, p: int = _DEFAULT_P) -> dict:
    """RCT-3: Complex non-linear outcome and CATE, W ~ Bern(0.5)."""
    X = _covariates_x3(n, p)
    W = np.random.binomial(1, 0.5, size=n)

    g = (
        np.sin(np.pi * X[:, 0])
        + 0.5 * X[:, 1] * X[:, 2]
        + 0.1 * X[:, 5] ** 3
        + 0.2 * np.cos(X[:, 6])
    )
    delta = 1.0 + 0.5 * np.tanh(X[:, 0]) + 0.2 * X[:, 5] * X[:, 6]
    eps = np.random.normal(0, 1, size=n)

    Y_obs = g + W * delta + eps
    return {"X": X, "W": W, "Y_obs": Y_obs,
            "pi_true": np.full(n, 0.5), "true_ate": _TRUE_ATE}


def generate_obs_3_data(n: int, p: int = _DEFAULT_P) -> dict:
    """OBS-3: Complex non-linearity with severe confounding (low overlap).

    logit(e(X)) = 0.3·X₁ + 0.3·X₂ − 0.5·X₆
    Strong coefficients on the bimodal mixture variables push propensity
    scores towards 0 and 1, creating a challenging low-overlap setting.
    """
    X = _covariates_x3(n, p)

    logit_e = 0.3 * X[:, 0] + 0.3 * X[:, 1] - 0.5 * X[:, 5]
    pi = expit(logit_e)
    W = np.random.binomial(1, pi)

    g = (
        np.sin(np.pi * X[:, 0])
        + 0.5 * X[:, 1] * X[:, 2]
        + 0.1 * X[:, 5] ** 3
        + 0.2 * np.cos(X[:, 6])
    )
    delta = 1.0 + 0.5 * np.tanh(X[:, 0]) + 0.2 * X[:, 5] * X[:, 6]
    eps = np.random.normal(0, 1, size=n)

    Y_obs = g + W * delta + eps
    return {"X": X, "W": W, "Y_obs": Y_obs,
            "pi_true": pi, "true_ate": _TRUE_ATE}


def generate_obs_3_overlap_data(
    n: int,
    p: int = _DEFAULT_P,
    overlap_strength: float = 1.0,
) -> dict:
    """OBS-3 with tuneable overlap severity for the overlap gradient experiment.

    logit(e(X)) = c * (0.3*X1 + 0.3*X2 - 0.5*X6)

    c = 0   -> propensity = 0.5 (perfect overlap)
    c = 1   -> default OBS-3 (low overlap)
    c > 1   -> extreme confounding
    """
    X = _covariates_x3(n, p)
    c = float(overlap_strength)

    logit_e = c * (0.3 * X[:, 0] + 0.3 * X[:, 1] - 0.5 * X[:, 5])
    pi = expit(logit_e)
    W = np.random.binomial(1, pi)

    g = (
        np.sin(np.pi * X[:, 0])
        + 0.5 * X[:, 1] * X[:, 2]
        + 0.1 * X[:, 5] ** 3
        + 0.2 * np.cos(X[:, 6])
    )
    delta = 1.0 + 0.5 * np.tanh(X[:, 0]) + 0.2 * X[:, 5] * X[:, 6]
    eps = np.random.normal(0, 1, size=n)

    Y_obs = g + W * delta + eps
    return {
        "X": X, "W": W, "Y_obs": Y_obs,
        "pi_true": pi, "true_ate": _TRUE_ATE,
        "overlap_strength": c,
    }
