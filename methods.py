# methods.py
#
# Contains implementations of all benchmark methods.
# Uses numba for JIT compilation to accelerate computations.
# -----------------------------------------------------------------------------

import numpy as np
import time
from numba import jit
from scipy.special import logsumexp

# --- Numba-accelerated Helper Functions ---

@jit(nopython=True)
def linear_loglik_i(x, y, theta, sigma=1.0):
    mu = np.dot(x, theta)
    return -0.5 * np.log(2 * np.pi * sigma**2) - 0.5 * ((y - mu) / sigma)**2

@jit(nopython=True)
def logistic_loglik_i(x, y, theta):
    eta = np.dot(x, theta)
    # Clamp eta to avoid overflow in exp
    if eta > 30.0:
        return y * eta - eta
    elif eta < -30.0:
        return y * eta
    return y * eta - np.log(1.0 + np.exp(eta))

@jit(nopython=True)
def cox_loglik_i(i, eta, y):
    # Assumes y is sorted by time, and eta corresponds to sorted X
    risk_set_eta = eta[i:]
    max_val = np.max(risk_set_eta)
    log_sum_exp_risk_set = max_val + np.log(np.sum(np.exp(risk_set_eta - max_val)))
    return eta[i] - log_sum_exp_risk_set

@jit(nopython=True)
def calculate_total_loglik(X, y, theta, model):
    N = X.shape[0]
    total_loglik = 0.0
    # Pre-calculate eta for Cox to avoid redundant computation in the loop
    if model == "coxph":
        eta = np.dot(X, theta)
    else:
        # Create a dummy eta to satisfy numba compiler
        eta = np.zeros(N)

    for i in range(N):
        if model == "linear":
            total_loglik += linear_loglik_i(X[i], y[i, 0], theta)
        elif model == "logistic":
            total_loglik += logistic_loglik_i(X[i], y[i, 0], theta)
        elif model == "coxph":
            if y[i, 1] == 1: # Only sum over events
                total_loglik += cox_loglik_i(i, eta, y)
    return total_loglik
    
@jit(nopython=True)
def calculate_weighted_loglik(X, y, theta, weights, model, N_full):
    N_sub = X.shape[0]
    total_loglik = 0.0
    if model == "coxph":
        eta = np.dot(X, theta)
    else:
        eta = np.zeros(N_sub) # Dummy for numba

    for i in range(N_sub):
        if weights[i] > 1e-12: # Check for non-zero weight
            if model == "linear":
                ll = linear_loglik_i(X[i], y[i, 0], theta)
            elif model == "logistic":
                ll = logistic_loglik_i(X[i], y[i, 0], theta)
            elif model == "coxph":
                ll = cox_loglik_i(i, eta, y) if y[i, 1] == 1 else 0.0
            else:
                ll = 0.0 # Should not happen
            total_loglik += N_full * weights[i] * ll
    return total_loglik


# --- Main MCMC Algorithm Implementations (Numba-Jitted) ---

@jit(nopython=True)
def abbs_mcmc_numba(X, y, model, theta_init, r, T, burn_in, c_init, a0, b0):
    N, p = X.shape
    samples = np.zeros((T - burn_in, p))
    theta_curr = theta_init.copy()
    c_curr = c_init
    proposal_sd = np.full(p, 0.1)

    for t in range(T):
        # --- Stability checks ---
        if not np.all(np.isfinite(theta_curr)):
            theta_curr = theta_init.copy() 
        if not np.isfinite(c_curr) or c_curr <= 1e-6:
            c_curr = 1.0

        # 1. Update Subsampling Scores
        loglik_vec = np.zeros(N)
        eta = np.dot(X, theta_curr) if model == "coxph" else np.zeros(N)
        for i in range(N):
            if model == "linear": loglik_vec[i] = linear_loglik_i(X[i], y[i,0], theta_curr)
            elif model == "logistic": loglik_vec[i] = logistic_loglik_i(X[i], y[i,0], theta_curr)
            elif model == "coxph" and y[i,1] == 1: loglik_vec[i] = cox_loglik_i(i, eta, y)
        
        scores = np.abs(loglik_vec)
        sum_scores = np.sum(scores)
        pi_vec = scores / sum_scores if sum_scores > 1e-9 else np.full(N, 1.0/N)
        
        # 2. Sample Indicators and Weights
        alpha_vec = np.minimum(1.0, r * pi_vec)
        delta_vec = np.zeros(N, dtype=np.int64)
        for i in range(N):
            if np.isfinite(alpha_vec[i]):
                delta_vec[i] = np.random.binomial(1, alpha_vec[i])
        
        k = np.sum(delta_vec)
        weights = np.zeros(N)
        sum_log_w_sub = 0.0
        if k > 0:
            w_sub = np.random.gamma(c_curr, 1.0, k)
            w_sub_sum = np.sum(w_sub)
            if w_sub_sum > 1e-9: w_sub /= w_sub_sum
            else: w_sub = np.full(k, 1.0/k)
            
            log_w_sub = np.log(w_sub)
            sum_log_w_sub = np.sum(log_w_sub[np.isfinite(log_w_sub)])
            
            # Place weights back into the full N-dimensional vector
            idx_k = 0
            for i in range(N):
                if delta_vec[i] == 1:
                    weights[i] = w_sub[idx_k]
                    idx_k += 1

        # 3. Update concentration parameter c
        shape_c = a0 + k - 1.0
        rate_c = b0 - sum_log_w_sub
        if k > 1 and np.isfinite(sum_log_w_sub) and shape_c > 0 and rate_c > 0:
            c_curr = np.random.gamma(shape_c, 1.0 / rate_c)
        else:
            c_curr = 1.0 # Reset if values are not valid

        # 4. Update theta (Metropolis-Hastings)
        theta_prop = theta_curr + proposal_sd * np.random.randn(p)
        
        # We only need to compute likelihood over the subsample
        X_sub = X[delta_vec==1]
        y_sub = y[delta_vec==1]
        weights_sub = weights[delta_vec==1]

        if X_sub.shape[0] > 0:
            loglik_prop = calculate_weighted_loglik(X_sub, y_sub, theta_prop, weights_sub, model, float(N))
            loglik_curr = calculate_weighted_loglik(X_sub, y_sub, theta_curr, weights_sub, model, float(N))
        else:
            loglik_prop = 0.0
            loglik_curr = 0.0

        logprior_prop = -0.5 * np.dot(theta_prop, theta_prop) / 100.0
        logprior_curr = -0.5 * np.dot(theta_curr, theta_curr) / 100.0
        log_alpha = (loglik_prop + logprior_prop) - (loglik_curr + logprior_curr)
        
        if np.isfinite(log_alpha) and np.log(np.random.rand()) < log_alpha:
            theta_curr = theta_prop
        
        if t >= burn_in:
            samples[t - burn_in, :] = theta_curr
            
    return samples

@jit(nopython=True)
def blbb_mcmc_numba(X, y, model, theta_init, s, b, T_inner):
    N, p = X.shape
    # Return a 3D array to keep samples from each subset separate
    all_samples = np.zeros((s, T_inner, p))
    indices = np.arange(N)
    np.random.shuffle(indices)
    proposal_sd = np.full(p, 0.1)
    rescale_factor = float(N) / b

    for j in range(s):
        sub_indices = indices[j*b : (j+1)*b]
        X_sub = X[sub_indices]
        y_sub = y[sub_indices]
        
        if model == "coxph":
            sort_order = np.argsort(y_sub[:, 0])
            X_sub = X_sub[sort_order]
            y_sub = y_sub[sort_order]

        theta_curr = theta_init.copy()

        for i in range(T_inner):
            if not np.all(np.isfinite(theta_curr)):
                theta_curr = theta_init.copy()

            weights_sub = np.random.gamma(rescale_factor, 1.0, b)
            w_sub_sum = np.sum(weights_sub)
            if w_sub_sum > 1e-9: weights_sub /= w_sub_sum
            else: weights_sub = np.full(b, 1.0/b)
            
            theta_prop = theta_curr + proposal_sd * np.random.randn(p)
            loglik_prop = calculate_weighted_loglik(X_sub, y_sub, theta_prop, weights_sub, model, float(N))
            loglik_curr = calculate_weighted_loglik(X_sub, y_sub, theta_curr, weights_sub, model, float(N))
            logprior_prop = -0.5 * np.dot(theta_prop, theta_prop) / 100.0
            logprior_curr = -0.5 * np.dot(theta_curr, theta_curr) / 100.0
            log_alpha = (loglik_prop + logprior_prop) - (loglik_curr + logprior_curr)
            
            if np.isfinite(log_alpha) and np.log(np.random.rand()) < log_alpha:
                theta_curr = theta_prop
            
            all_samples[j, i, :] = theta_curr
            
    return all_samples

@jit(nopython=True)
def bmh_mcmc_numba(X, y, model, theta_init, m, k, T, burn_in):
    N, p = X.shape
    samples = np.zeros((T - burn_in, p))
    theta_curr = theta_init.copy()
    proposal_sd = np.full(p, 0.1)
    
    for t in range(T):
        if not np.all(np.isfinite(theta_curr)):
            theta_curr = theta_init.copy()

        theta_prop = theta_curr + proposal_sd * np.random.randn(p)
        loglik_curr_avg = 0.0
        loglik_prop_avg = 0.0
        
        for i in range(k):
            boot_idx = np.random.randint(0, N, m)
            X_boot = X[boot_idx]
            y_boot = y[boot_idx]
            
            if model == "coxph":
                sort_order = np.argsort(y_boot[:, 0])
                X_boot = X_boot[sort_order]
                y_boot = y_boot[sort_order]
            
            loglik_curr_avg += calculate_total_loglik(X_boot, y_boot, theta_curr, model)
            loglik_prop_avg += calculate_total_loglik(X_boot, y_boot, theta_prop, model)
        
        loglik_curr_avg /= k
        loglik_prop_avg /= k
        
        loglik_full_curr_approx = (float(N)/m) * loglik_curr_avg
        loglik_full_prop_approx = (float(N)/m) * loglik_prop_avg
        
        logprior_prop = -0.5 * np.dot(theta_prop, theta_prop) / 100.0
        logprior_curr = -0.5 * np.dot(theta_curr, theta_curr) / 100.0
        log_alpha = (loglik_full_prop_approx + logprior_prop) - (loglik_full_curr_approx + logprior_curr)
        
        if np.isfinite(log_alpha) and np.log(np.random.rand()) < log_alpha:
            theta_curr = theta_prop
        
        if t >= burn_in:
            samples[t - burn_in, :] = theta_curr
    
    return samples

# --- Python Wrappers for Methods ---

def run_abbs(X, y, model, p, r, T, burn_in, c_init, a0, b0):
    start_time = time.time()
    theta_init = np.zeros(p)
    posterior_samples = abbs_mcmc_numba(X, y, model, theta_init, r, T, burn_in, c_init, a0, b0)
    
    point_est = np.mean(posterior_samples, axis=0)
    ci_lower = np.percentile(posterior_samples, 2.5, axis=0)
    ci_upper = np.percentile(posterior_samples, 97.5, axis=0)
    
    return {"point_est": point_est, "ci_lower": ci_lower, "ci_upper": ci_upper}

def run_os_repeated(X, y, model, p, r, reps):
    N = X.shape[0]
    n0 = 200 # Pilot sample size
    
    point_estimates = []
    ci_lowers = []
    ci_uppers = []

    for _ in range(reps):
        # Step 1: Pilot Sample
        pilot_idx = np.random.choice(N, n0, replace=False)
        X_pilot, y_pilot = X[pilot_idx], y[pilot_idx]
        
        # Get a reasonable initial estimate for gradient descent
        beta0 = np.zeros(p)
        if model == 'logistic' or model == 'coxph':
            try:
                # Use a smaller subset to initialize to avoid numerical issues
                init_idx = np.random.choice(n0, min(n0, p*2), replace=False)
                if model == 'logistic':
                    beta0 = np.linalg.lstsq(X_pilot[init_idx], y_pilot[init_idx] - 0.5, rcond=None)[0].flatten()
                else: # Cox
                     beta0 = np.linalg.lstsq(X_pilot[init_idx], y_pilot[init_idx, 0], rcond=None)[0].flatten()
            except:
                 beta0 = np.zeros(p)
        
        if model == 'linear':
            beta0 = np.linalg.lstsq(X_pilot, y_pilot, rcond=None)[0].flatten()
        else: # simplified gradient descent
            lr = 0.01
            for _ in range(10):
                if model == 'logistic':
                    eta = X_pilot @ beta0
                    prob = 1 / (1 + np.exp(-np.clip(eta, -30, 30)))
                    grad = X_pilot.T @ (y_pilot.flatten() - prob) / n0
                    beta0 += lr * grad
                elif model == 'coxph':
                    # Simplified gradient (placeholder, real one is complex)
                    eta = X_pilot @ beta0
                    grad = X_pilot.T @ (y_pilot[:, 1] - np.exp(eta)) / n0
                    beta0 += lr * grad


        # Calculate optimal probabilities (L-optimality)
        if model == 'linear':
            residuals = np.abs(y.flatten() - X @ beta0)
        else: # logistic or coxph
            eta = X @ beta0
            if model == 'logistic':
                 probs = 1 / (1 + np.exp(-np.clip(eta, -30, 30)))
                 residuals = np.abs(y.flatten() - probs)
            else: # coxph (approximate residual)
                 residuals = np.abs(y[:, 1] - np.exp(np.clip(eta, -30, 30)))
            
        norms = np.linalg.norm(X, axis=1)
        pi = residuals * norms
        pi_sum = np.sum(pi)
        pi = pi / pi_sum if pi_sum > 1e-9 else np.full(N, 1.0/N)

        # Step 2: Main Subsample
        subsample_idx = np.random.choice(N, r, replace=True, p=pi)
        X_sub, y_sub = X[subsample_idx], y[subsample_idx]
        
        # Fit main model on subsample
        if model == 'linear':
            beta_hat = np.linalg.lstsq(X_sub, y_sub, rcond=None)[0].flatten()
        else: # logistic or coxph
            beta_hat = beta0.copy() 
            for _ in range(20): 
                if model == 'logistic':
                    eta = X_sub @ beta_hat
                    prob = 1 / (1 + np.exp(-np.clip(eta, -30, 30)))
                    grad = X_sub.T @ (y_sub.flatten() - prob) / r
                    beta_hat += lr * grad
                elif model == 'coxph':
                    eta = X_sub @ beta_hat
                    grad = X_sub.T @ (y_sub[:, 1] - np.exp(eta)) / r
                    beta_hat += lr * grad
        
        # Calculate Asymptotic Variance based on SUBSAMPLES
        try:
            if model == 'linear':
                w_pilot = np.full(n0, N / n0)
                M_X_est = (X_pilot.T * w_pilot) @ X_pilot / N
                
                residuals_sub = (y_sub.flatten() - X_sub @ beta_hat)**2
                pi_sub = pi[subsample_idx]
                w_sub = 1 / (N * pi_sub)
                V_c_est_mat = (X_sub.T * (w_sub * residuals_sub)) @ X_sub
                
            else: # logistic
                eta_pilot = X_pilot @ beta0
                p_pilot = 1 / (1 + np.exp(-np.clip(eta_pilot,-30,30)))
                w_pilot = p_pilot * (1-p_pilot) * (N/n0)
                M_X_est = (X_pilot.T * w_pilot) @ X_pilot / N

                eta_sub = X_sub @ beta_hat
                p_sub = 1 / (1 + np.exp(-np.clip(eta_sub,-30,30)))
                residuals_sub = (y_sub.flatten() - p_sub)**2
                pi_sub = pi[subsample_idx]
                w_sub = 1 / (N*pi_sub)
                V_c_est_mat = (X_sub.T * (w_sub * residuals_sub)) @ X_sub

            M_X_inv = np.linalg.inv(M_X_est)
            V_asym = (M_X_inv @ V_c_est_mat @ M_X_inv) / r
            se = np.sqrt(np.abs(np.diag(V_asym)))
            
            point_estimates.append(beta_hat)
            ci_lowers.append(beta_hat - 1.96 * se)
            ci_uppers.append(beta_hat + 1.96 * se)

        except np.linalg.LinAlgError:
            point_estimates.append(np.full(p, np.nan))
            ci_lowers.append(np.full(p, np.nan))
            ci_uppers.append(np.full(p, np.nan))
            
    return {
        "point_est": np.nanmean(np.array(point_estimates), axis=0),
        "ci_lower": np.nanmean(np.array(ci_lowers), axis=0),
        "ci_upper": np.nanmean(np.array(ci_uppers), axis=0)
    }

def run_blbb(X, y, model, p, s, b, T_inner):
    theta_init = np.zeros(p)
    # The numba function returns a 3D array: (s, T_inner, p)
    posterior_samples_3d = blbb_mcmc_numba(X, y, model, theta_init, s, b, T_inner)

    # Calculate summaries for each subset
    posterior_means_per_subset = np.mean(posterior_samples_3d, axis=1)
    ci_lowers_per_subset = np.percentile(posterior_samples_3d, 2.5, axis=1)
    ci_uppers_per_subset = np.percentile(posterior_samples_3d, 97.5, axis=1)

    # Average the summaries
    point_est = np.mean(posterior_means_per_subset, axis=0)
    ci_lower = np.mean(ci_lowers_per_subset, axis=0)
    ci_upper = np.mean(ci_uppers_per_subset, axis=0)
    
    return {"point_est": point_est, "ci_lower": ci_lower, "ci_upper": ci_upper}

def run_bmh(X, y, model, p, m, k, T, burn_in):
    theta_init = np.zeros(p)
    posterior_samples = bmh_mcmc_numba(X, y, model, theta_init, m, k, T, burn_in)
    
    point_est = np.mean(posterior_samples, axis=0)
    ci_lower = np.percentile(posterior_samples, 2.5, axis=0)
    ci_upper = np.percentile(posterior_samples, 97.5, axis=0)
    
    return {"point_est": point_est, "ci_lower": ci_lower, "ci_upper": ci_upper}

