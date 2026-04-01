# UD-DML: Uniform Design Double Machine Learning

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Python implementation of **Uniform Design Double Machine Learning (UD-DML)** for scalable causal inference on massive datasets.

> **Reference:** Qu, Xu & Zhang (2026). *UD-DML: Uniform Design Subsampling for Double Machine Learning over Massive Data.*

---

## What is UD-DML?

UD-DML estimates the **Average Treatment Effect (ATE)** when working with massive observational datasets where full-data DML is computationally prohibitive. It works in three phases:

1. **Phase 1 — UD Subsampling in PCA-Rotated Space**
   - Standardise covariates and perform PCA to retain the dominant *q* dimensions (ρ₀ = 0.85 cumulative variance threshold).
   - Construct a low-discrepancy skeleton in [0,1]^q via the **good lattice point (GLP)** method with power generators, selecting the design that minimises the **mixture discrepancy** D²_M.
   - Map the skeleton to covariate space through marginal empirical inverse CDFs.
   - Find the nearest available **treated and control unit** for each skeleton point (paired matching without replacement) using `cKDTree` spatial indices.

2. **Phase 2 — Cross-Fitted DML** on the selected original observations.
3. **Phase 3 — Wald Inference** from the AIPW pseudo-outcomes.

The method is compared against **UNIF-DML** (naive uniform subsampling) and **FULL-DML** (gold standard).

---

## Installation

```bash
pip install -r requirements.txt
```

**Dependencies:** numpy, scipy, scikit-learn, lightgbm, pandas, joblib, tqdm, matplotlib.

---

## Quick Start

```python
from data_generation import generate_obs_3_data
from methods import run_ud, run_unif
import config

# Generate synthetic data (10 covariates, 100k observations, low overlap)
data = generate_obs_3_data(n=100_000, p=10)

# UD-DML estimate with 5,000 subsample points
ud_result = run_ud(
    data["X"], data["W"], data["Y_obs"], data["pi_true"],
    is_rct=False,
    r={"r_total": 5_000},
    k_folds=config.K_FOLDS,
)

# Uniform subsampling baseline
unif_result = run_unif(
    data["X"], data["W"], data["Y_obs"], data["pi_true"],
    is_rct=False,
    r={"r_total": 5_000},
    k_folds=config.K_FOLDS,
)

print(f"UD-DML  ATE = {ud_result['est_ate']:.4f}  "
      f"CI = [{ud_result['ci_lower']:.4f}, {ud_result['ci_upper']:.4f}]")
print(f"UNIF    ATE = {unif_result['est_ate']:.4f}  "
      f"CI = [{unif_result['ci_lower']:.4f}, {unif_result['ci_upper']:.4f}]")
```

---

## Data Scenarios

Six DGPs are provided as a 2 × 3 factorial design (Table 1 of the paper):

| Scenario | Covariates | Heterogeneity | Overlap |
|----------|-----------|---------------|---------|
| RCT-1 / OBS-1 | X1: Uniform | Low (linear) | High |
| RCT-2 / OBS-2 | X2: Mixed | Moderate (non-linear) | Moderate |
| RCT-3 / OBS-3 | X3: GMM + Normal | High (complex) | Low |

All scenarios use **p = 10** covariates with true ATE θ₀ = 1.0.

---

## Running Experiments

Run all five experiments:

```bash
python simulations.py
```

Run specific experiments:

```bash
python simulations.py --experiments experiment_subsample_size experiment_population_size --jobs 4
```

Quick test (fewer replications):

```bash
python simulations.py --fast-demo --jobs 1
```

### Available Experiments

| Experiment | Description | Paper Section |
|-----------|-------------|---------------|
| `experiment_visualization` | 2-D covariate coverage diagnostics | — |
| `experiment_subsample_size` | Performance vs subsample size r | §3.3, Exp 1 |
| `experiment_population_size` | Scalability with population n | §3.3, Exp 2 |
| `experiment_double_robust` | Double-robustness stress test | §3.3, Exp 3 |
| `experiment_nuisance_sensitivity` | Learner comparison (LGBM, RF, LASSO) | §3.3, Exp 4 |

---

## Configuration

Edit `config.py` to adjust parameters:

```python
# ── Simulation ──
BASE_SEED = 20250919
DEFAULT_REPLICATIONS = 1000
N_POPULATION = 500_000
K_FOLDS = 2

# ── UD-DML (Algorithm 1) ──
UD_VARIANCE_THRESHOLD = 0.85    # ρ₀ for PCA retention (Step 3)
UD_MAX_GENERATOR_CANDIDATES = 200  # GLP search budget (Step 6)
UD_NEAREST_NEIGHBORS = 5       # Initial k for adaptive k-NN (Steps 16-17)

# ── Nuisance learners ──
DEFAULT_NUISANCE_LEARNER = 'lgbm'
LGBM_N_ESTIMATORS = 100
```

---

## Project Structure

```
UD-DML/
├── config.py             # Configuration and experiment definitions
├── data_generation.py    # Data-generating processes (6 scenarios)
├── methods.py            # UD-DML, UNIF-DML, and FULL-DML estimators
├── evaluation.py         # Post-processing and visualisation
├── simulations.py        # Main experiment driver
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Citation

If you use this code, please cite:

```bibtex
@article{qu2026uddml,
  title   = {UD-DML: Uniform Design Subsampling for Double Machine Learning
             over Massive Data},
  author  = {Qu, Yuanke and Xu, Xiaoya and Zhang, Hengtao},
  year    = {2026}
}
```

---

## License

MIT License
