# OS-DML: Optimal Subsampling for Double Machine Learning

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official implementation of **"Optimal Subsampling for Double Machine Learning"**

---

## Overview

OS-DML is a computationally efficient method for estimating average treatment effects (ATE) from large-scale datasets. It uses **importance sampling proportional to centered influence functions** with optimal sampling probabilities to achieve significant variance reduction compared to uniform subsampling.

**Key Features:**
- Two-phase optimal subsampling with centered probabilities
- Hájek estimator with plug-in variance estimation (default)
- Hansen-Hurwitz estimator also available
- Automated hyperparameter optimization via Bayesian methods
- Compatible with both RCT and observational data
- Significant speedup for large datasets (N > 100,000)

---

## Installation

### Requirements
```bash
pip install -r requirements.txt
```

**Main dependencies:** `numpy`, `scipy`, `lightgbm`, `scikit-learn`, `pandas`, `joblib`, `tqdm`, `matplotlib`, `optuna`

**Note:** For hyperparameter optimization, `optuna` will be automatically installed if not present.

---

## Quick Start

### Basic Usage

```python
import numpy as np
from methods import run_os
from data_generation import generate_rct_s_data

# Generate data
data = generate_rct_s_data(n=100000, p=30)

# Run OS-DML (with global parameters from config.py)
result = run_os(
    X=data['X'], 
    W=data['W'], 
    Y_obs=data['Y_obs'],
    pi_true=data['pi_true'],
    is_rct=True,
    r={'r_total': 10000},  # Uses global PILOT_RATIO to split into r0, r1
    k_folds=2
)

# Or specify r0, r1 explicitly
result = run_os(
    X=data['X'], 
    W=data['W'], 
    Y_obs=data['Y_obs'],
    pi_true=data['pi_true'],
    is_rct=True,
    r={'r0': 3000, 'r1': 7000},  # Pilot: 3000, Main: 7000
    k_folds=2
)

print(f"ATE Estimate: {result['est_ate']:.4f}")
print(f"95% CI: [{result['ci_lower']:.4f}, {result['ci_upper']:.4f}]")
```

### Using Other Methods

```python
from methods import run_unif, run_lss, run_full

# Uniform subsampling
result_unif = run_unif(X, W, Y_obs, pi_true, is_rct, r={'r0': 0, 'r1': 10000}, k_folds=2)

# Leverage score subsampling
result_lss = run_lss(X, W, Y_obs, pi_true, is_rct, r={'r0': 3000, 'r1': 7000}, k_folds=2)

# Full-data DML (baseline)
result_full = run_full(X, W, Y_obs, pi_true, is_rct, k_folds=2)
```

---

## Running Simulation Experiments

The codebase includes a **unified simulation script** that runs all three experiments automatically:

### Run Complete Simulation Study

```bash
# Recommended: Run as Python script
python Simulation_Unified.py
```

This will automatically:
1. **Experiment 1: Sensitivity Analysis** - Find optimal hyperparameters via Bayesian optimization
   - Generates 7 PDF visualization plots
   - Auto-saves progress (resumable if interrupted)
2. **Auto-update config.py** - Apply optimal parameters globally
3. **Experiment 2: Main Comparison** - Compare OS, UNIF, LSS, FULL across 4 DGPs (using optimal params)
   - Parallel computation across all CPUs
   - Checkpoint support (resumable if interrupted)
4. **Experiment 3: Robustness Check** - Test double robustness under misspecification (using optimal params)
   - Parallel computation across all CPUs
   - Checkpoint support (resumable if interrupted)

**Key Features:**
- ⚡ **Parallel Computation**: Automatically uses all CPU cores (6-8× faster)
- 🔄 **Checkpoint Support**: Resume from interruption without data loss
- 📊 **Auto-Visualization**: 7 PDF plots generated for Experiment 1
- 🛡️ **Fault Tolerant**: Safe to interrupt (Ctrl+C) and resume later

### Experiments Included

1. **Experiment 1: Sensitivity Analysis**
   - Bayesian optimization (Optuna) across all scenarios
   - Search space: `n_estimators` [20, 100], `delta` [10^-4, 10^-1], `pilot_ratio` [0.1, 0.9]
   - Finds optimal hyperparameters that minimize RMSE while maintaining 95% coverage
   - Outputs: `optimal_parameters.json`, 7 PDF visualization plots
   - Features: Parallel trial execution, auto-resume from checkpoint

2. **Experiment 2: Main Comparison**
   - Compares OS, UNIF, LSS, FULL across 4 DGPs (RCT-S, RCT-C, OBS-S, OBS-C)
   - Uses optimal hyperparameters from Experiment 1
   - Outputs: Performance metrics (RMSE, Coverage, Bias, Runtime)
   - Features: Parallel computation, per-replication checkpoints

3. **Experiment 3: Robustness Check**
   - Tests OS-DML under model misspecification (OBS-C scenario)
   - Four conditions: correct/wrong outcome model × correct/wrong propensity model
   - Verifies double robustness property
   - Features: Parallel computation, per-replication checkpoints

### Configuration

Edit `config.py` to modify global parameters:

```python
# General settings
N_SIM = 50                     # Monte Carlo replications
N_POPULATION = 1000000         # Population size (updated to 1M)
K_FOLDS = 2                    # Cross-fitting folds

# OS-DML algorithm parameters (auto-updated by Experiment 1)
PILOT_RATIO = 0.3              # Pilot sample ratio
DELTA = 0.01                   # Stabilization constant
PILOT_N_ESTIMATORS = 30        # Pilot GBM complexity

# Estimator choice
ESTIMATOR_TYPE = 'hajek'       # 'hajek' (default) or 'hh' (Hansen-Hurwitz)

# Experiment 1 optimization settings
"experiment_1_sensitivity_analysis": {
    "n_trials": 60,            # Optuna trials
    "n_replications": 10,      # MC reps per trial
    "n_jobs": -1,              # Parallel CPUs (-1 = all)
    "checkpoint": True,        # Enable auto-save
    "resume": True             # Resume from checkpoint
}
```

---

## Project Structure

```
OS-DML/
├── methods.py                # Core algorithms (OS, UNIF, LSS, FULL)
├── data_generation.py        # DGP functions (RCT-S, RCT-C, OBS-S, OBS-C)
├── config.py                 # Global configuration & experiment settings
├── evaluation.py             # Results processing and analysis
├── Simulation_Unified.py     # Unified simulation script (all experiments)
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── simulation_results/       # Output directory (auto-created)
│   ├── exp1_sensitivity_analysis/
│   │   ├── optimal_parameters.json
│   │   ├── trials.csv
│   │   └── optuna_all_scenarios.db
│   ├── exp2_main_comparison/
│   │   ├── raw_results.pkl
│   │   └── summary_table.csv
│   └── exp3_robustness_check/
│       ├── raw_results.pkl
│       └── summary_table.csv
└── analysis_results/         # Processed results (auto-created)
```

---

## Results

After running simulations, results are automatically saved:

### Experiment 1 Outputs
- `optimal_parameters.json` - Best hyperparameters found
- `trials.csv` - All optimization trials
- `optuna_all_scenarios.db` - Optuna study database (auto-resume checkpoint)
- **7 PDF Visualization Plots:**
  - `parameter_importance.pdf` ⭐ - Which hyperparameters matter most
  - `hyperparameter_analysis.pdf` ⭐ - 9-panel comprehensive analysis
  - `optimization_history.pdf` - Convergence over trials
  - `parallel_coordinate.pdf` - Parameter combination space
  - `contour_gbm_delta.pdf` - GBM × Delta interaction
  - `contour_gbm_pilot.pdf` - GBM × Pilot ratio interaction
  - `contour_delta_pilot.pdf` - Delta × Pilot ratio interaction

### Experiments 2 & 3 Outputs
- `raw_results.pkl` - Complete simulation results
- `summary_table.csv` - Performance metrics (RMSE, Coverage, Bias, Runtime)

### Viewing Results

```python
import pandas as pd
import json

# Load optimal parameters
with open('./simulation_results/exp1_sensitivity_analysis/optimal_parameters.json') as f:
    optimal_params = json.load(f)
    print(f"Optimal delta: {optimal_params['delta']:.6f}")
    print(f"Optimal pilot_ratio: {optimal_params['pilot_ratio']:.2f}")

# Load experiment results
exp2_results = pd.read_csv('./simulation_results/exp2_main_comparison/summary_table.csv')
print(exp2_results)

# Compare methods
exp2_results.groupby('method')[['RMSE', 'Coverage', 'Runtime']].mean()
```

---

## Customization

### Using Your Own Data

```python
from methods import run_os

# Prepare your data
# X: (N, p) covariate matrix
# W: (N,) treatment indicators (0/1)
# Y_obs: (N,) observed outcomes
# pi_true: (N,) true/estimated propensity scores

# Run OS-DML
result = run_os(
    X=your_X,
    W=your_W,
    Y_obs=your_Y,
    pi_true=your_pi,  # or None if unavailable
    is_rct=False,     # True for RCT, False for observational
    r={'r0': 3000, 'r1': 7000},
    k_folds=2,
    delta=0.01        # Optional: stabilization constant
)
```

### Changing the ML Algorithm

Edit `methods.py`, function `_fit_nuisance_models()` to replace LightGBM with your preferred estimator:

```python
# Replace this:
lgbm0 = lgb.LGBMRegressor(**lgbm_params).fit(X_train[W_train == 0], Y_train[W_train == 0])

# With your estimator:
from sklearn.ensemble import RandomForestRegressor
rf0 = RandomForestRegressor(n_estimators=100).fit(X_train[W_train == 0], Y_train[W_train == 0])
```

---

## Parameters Guide

### Global Parameters (config.py)

These parameters are automatically optimized by Experiment 1 but can be set manually:

```python
PILOT_RATIO = 0.3          # Pilot sample ratio (r0 / r_total)
DELTA = 0.01               # Stabilization constant for probability construction
PILOT_N_ESTIMATORS = 30    # GBM complexity in pilot stage
ESTIMATOR_TYPE = 'hajek'   # 'hajek' (Hájek estimator) or 'hh' (Hansen-Hurwitz)
```

### Sample Size Selection
- **Pilot size (r₀):** Experiment 1 optimizes, typically 30-50% of total budget
- **Main size (r₁):** Remaining budget
- **Total (r₀ + r₁):** Generally 5-10% of population N

**Example:** For N=1,000,000, use `r_total=10,000` → automatically splits based on `PILOT_RATIO`

### Key Parameters
- `delta`: Stabilization constant for centered optimal sampling probabilities (optimized)
- `pilot_ratio`: Pilot sample allocation (optimized)
- `n_estimators`: Pilot GBM complexity (optimized)
- `k_folds`: Cross-fitting folds (default: 2)
- `is_rct`: Whether treatment is randomized

### Hyperparameter Optimization

Experiment 1 uses Bayesian optimization (Optuna) to find optimal parameters that:
- Minimize RMSE across all scenarios
- Maintain 95% coverage
- Balance pilot quality vs. main sample size

---

## Citation

If you use this code, please cite:

```bibtex
@article{osdml2025,
  title={Optimal Subsampling for Double Machine Learning},
  author={[Author Names]},
  journal={[Journal Name]},
  year={2025},
  doi={[DOI]}
}
```

---

## Algorithm Reference

OS-DML implements a two-phase procedure with centered optimal sampling:

### Algorithm 1: Two-Stage OS-DML with Hájek Estimator

1. **Phase 1 (Pilot Stage):**
   - Draw pilot sample of size r₀ uniformly from population N
   - Fit nuisance models (μ₀, μ₁, e) on pilot data
   - Predict pseudo-outcomes φ̂ᵢ⁽⁰⁾ on full population
   - Compute centered sampling probabilities: pᵢ ∝ |φ̂ᵢ⁽⁰⁾ - φ̄⁽⁰⁾| + δ

2. **Phase 2 (Main Stage):**
   - Draw main sample of size r₁ via importance sampling with probabilities pᵢ
   - Combine pilot and main samples
   - Refit nuisance models with importance weights
   - Compute final pseudo-outcomes φ̂ⱼ

3. **Inference:**
   - Hájek estimator (default): τ̂_HJ = Σⱼ(φ̂ⱼ/qⱼ) / Σⱼ(1/qⱼ)
   - Plug-in variance: Var(τ̂_HJ) = (1/(r·N²)) · (1/(r-1)) · Σₜ(Uₜ - Ū)²
   - Hansen-Hurwitz also available via `ESTIMATOR_TYPE = 'hh'`

### Key Innovations
- **Centered probabilities**: Use |φ̂ᵢ - φ̄| instead of |φ̂ᵢ| for better variance reduction
- **Hájek estimator**: Ratio estimator more efficient than Hansen-Hurwitz
- **Automated tuning**: Bayesian optimization finds optimal δ, pilot_ratio, n_estimators

For detailed algorithm description and theoretical properties, see the paper.

---

## Performance & Advantages

### Computational Efficiency
- **Speed**: 5-10× faster than full-data DML for N > 100,000
- **Memory**: Processes only r ≈ 0.1N samples instead of full N
- **Scalability**: Handles datasets with N > 1,000,000

### Statistical Efficiency
- **Variance reduction**: 30-50% lower RMSE vs. uniform subsampling
- **Coverage**: Maintains nominal 95% coverage across scenarios
- **Robustness**: Double robust to model misspecification

### Automated Optimization
- **Bayesian search**: Efficiently explores hyperparameter space
- **Multi-scenario**: Finds parameters robust across all DGPs
- **Checkpoint**: Resumable optimization for long-running studies

### When to Use OS-DML
- ✅ Large datasets (N > 100,000)
- ✅ Limited computational resources
- ✅ Need for fast inference
- ✅ Both RCT and observational data

### Comparison to Alternatives
| Method | RMSE | Coverage | Runtime | Memory |
|--------|------|----------|---------|--------|
| **OS-DML** | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| UNIF | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| LSS | ⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ |
| FULL | ⭐⭐⭐ | ⭐⭐⭐ | ⭐ | ⭐ |

---

## License

MIT License - see LICENSE file for details.

---

## Contact

- **Issues:** [GitHub Issues](https://github.com/your-username/OS-DML/issues)
- **Email:** zhanght@gdou.edu.cn

---

## FAQ

### Q: How do I resume after interruption?
A: Simply rerun `python Simulation_Unified.py`. All experiments have checkpoint support and will automatically resume from where they stopped.

### Q: Can I adjust parallel computation?
A: Yes! Edit `N_JOBS_EXP2_3` in `Simulation_Unified.py` (line ~682):
```python
N_JOBS_EXP2_3 = -1   # All CPUs (default, fastest)
N_JOBS_EXP2_3 = 8    # Use 8 CPUs
N_JOBS_EXP2_3 = 1    # Sequential (for debugging)
```

### Q: How much faster is parallel execution?
A: With 16 cores, expect 6-8× speedup. Total runtime reduces from ~6 hours (sequential) to ~1 hour (parallel).

### Q: What are the 7 visualization plots?
A: Experiment 1 generates:
1. Parameter importance (shows which hyperparameters matter most)
2. 9-panel analysis (parameter vs. performance relationships)
3. Optimization history (convergence over trials)
4-7. Three 2D contour plots (parameter interactions)

---

## References

1. Chernozhukov, V., et al. (2018). Double/debiased machine learning for treatment and structural parameters. *The Econometrics Journal*, 21(1), C1-C68.

2. Hansen, M. H., & Hurwitz, W. N. (1943). On the theory of sampling from finite populations. *The Annals of Mathematical Statistics*, 14(4), 333-362.

3. Hájek, J. (1971). Comment on "An essay on the logical foundations of survey sampling" by D. Basu. *Foundations of Statistical Inference*, 236.