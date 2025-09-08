# data_generation.py
#
# Contains functions for generating simulated data for different models.
# -----------------------------------------------------------------------------

import numpy as np
from scipy.stats import t as t_dist

def generate_linear_data(N, p, beta, error_dist="t", df=3, sigma=1.0, **kwargs):
    """Generates data for a linear regression model."""
    X = np.random.randn(N, p)
    X[:, 0] = 1.0  # Intercept term
    
    mu = X @ beta
    
    if error_dist == "t":
        errors = t_dist.rvs(df, size=N) * sigma
    else: # Default to normal
        errors = np.random.randn(N) * sigma
        
    y = mu + errors
    return {"X": X, "y": y.reshape(-1, 1)}

def generate_logistic_data(N, p, beta, imbalance_offset=0.0, **kwargs):
    """Generates data for a logistic regression model."""
    X = np.random.randn(N, p)
    X[:, 0] = 1.0 # Intercept
    
    eta = X @ beta + imbalance_offset
    probs = 1 / (1 + np.exp(-eta))
    y = np.random.binomial(1, probs)
    return {"X": X, "y": y.reshape(-1, 1)}

def generate_cox_data(N, p, beta, lambda0=0.01, tau_max=50, **kwargs):
    """Generates data for a Cox Proportional Hazards model."""
    X = np.random.randn(N, p)
    X[:, 0] = 1.0 # Intercept equivalent
    
    # Baseline hazard is exponential
    hazard_ratio = np.exp(X @ beta)
    true_event_times = np.random.exponential(1 / (lambda0 * hazard_ratio))
    
    # Censoring time
    censoring_times = np.random.uniform(0, tau_max, N)
    
    # Observed time and event indicator
    observed_times = np.minimum(true_event_times, censoring_times)
    event_indicators = (true_event_times <= censoring_times).astype(int)
    
    # Sort data by observed time (crucial for Cox likelihood calculation)
    sort_idx = np.argsort(observed_times)
    X_sorted = X[sort_idx]
    y_sorted = np.c_[observed_times[sort_idx], event_indicators[sort_idx]]
    
    return {"X": X_sorted, "y": y_sorted}
