# OS-DML: Optimal Subsampling for Double Machine Learning

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official implementation of **"Optimal Subsampling for Double Machine Learning"**

---

## Overview

OS-DML is a computationally efficient method for estimating average treatment effects (ATE) from large-scale datasets. It uses **importance sampling proportional to influence functions** to achieve 30-50% variance reduction compared to uniform subsampling.

**Key Features:**
- Two-phase optimal subsampling (pilot + main)
- Hansen-Hurwitz estimator with design-based inference
- Compatible with both RCT and observational data
- Significant speedup for datasets with N > 100,000

---

## Installation

### Requirements
```bash
pip install -r requirements.txt
```

**Main dependencies:** `numpy`, `scipy`, `lightgbm`, `scikit-learn`, `pandas`, `joblib`, `tqdm`

---

## Quick Start

### Basic Usage

```python
import numpy as np
from methods import run_os
from data_generation import generate_rct_s_data

# Generate data
data = generate_rct_s_data(n=100000, p=30)

# Run OS-DML
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

The codebase includes three pre-configured experiments:

### Option 1: Run in Jupyter Notebook
```bash
jupyter notebook Simulation.ipynb
# Then run all cells
```

### Option 2: Run as Python Script
```bash
# Convert notebook to script
jupyter nbconvert --to script Simulation.ipynb

# Run simulation
python Simulation.py
```

### Experiments Included

1. **Experiment 1: Pilot Allocation** - Tests different pilot sample ratios (0.1-0.9)
2. **Experiment 2: Method Comparison** - Compares OS, UNIF, LSS, FULL across 4 DGPs
3. **Experiment 3: Robustness Check** - Tests double robustness under misspecification

**Configuration:** Edit `config.py` to modify:
- `N_SIM`: Number of Monte Carlo replications (default: 50)
- `N_POPULATION`: Population size (default: 100,000)
- Sample sizes (`r0`, `r1`) for each experiment

---

## Project Structure

```
OS-DML/
├── methods.py              # Core algorithms (OS, UNIF, LSS, FULL)
├── data_generation.py      # DGP functions (RCT-S, RCT-C, OBS-S, OBS-C)
├── config.py               # Experiment configuration
├── evaluation.py           # Results analysis
├── Simulation.ipynb        # Main simulation driver
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── simulation_results/    # Output directory (auto-created)
└── analysis_results/      # Summary tables (auto-created)
```

---

## Results

After running simulations, results are saved in:
- **Raw results:** `./simulation_results/exp{1,2,3}_*/`
- **Summary tables:** `./analysis_results/*/summary_table.csv`

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

### Sample Size Selection
- **Pilot size (r₀):** Typically 30-40% of total budget
- **Main size (r₁):** Remaining 60-70%
- **Total (r₀ + r₁):** Generally 5-10% of population N

**Example:** For N=100,000, use r₀=3,000 and r₁=7,000

### Key Parameters
- `delta`: Stabilization constant (default: 0.01)
- `k_folds`: Cross-fitting folds (default: 2)
- `is_rct`: Whether treatment is randomized

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

OS-DML implements a two-phase procedure:
1. **Phase 1:** Draw pilot sample → Estimate nuisances → Compute sampling probabilities ∝ |φᵢ|
2. **Phase 2:** Draw importance sample → Refit nuisances → Calculate Hansen-Hurwitz estimator

For detailed algorithm description, see the paper.

---

## License

MIT License - see LICENSE file for details.

---

## Contact

- **Issues:** [GitHub Issues](https://github.com/your-username/OS-DML/issues)
- **Email:** your.email@domain.com

---

## References

1. Chernozhukov, V., et al. (2018). Double/debiased machine learning for treatment and structural parameters. *The Econometrics Journal*, 21(1), C1-C68.

2. Hansen, M. H., & Hurwitz, W. N. (1943). On the theory of sampling from finite populations. *The Annals of Mathematical Statistics*, 14(4), 333-362.