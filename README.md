# UD-DML: Uniform Design Subsampling for Double Machine Learning over Massive Data

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official implementation of **UD-DML**, a design-based subsampling framework that makes Double Machine Learning (DML) scalable to massive observational datasets while preserving statistical efficiency and inferential validity.

> **Paper:** Qu, Xu & Zhang (2026). *UD-DML: Uniform Design Subsampling for Double Machine Learning over Massive Data.*

---

## Overview

Standard DML yields valid causal inference with flexible nuisance estimation, but its computational cost becomes prohibitive when cross-fitted learners must be trained on millions of observations. Naive uniform subsampling is fast but ignores covariate geometry, producing poor treated–control balance, unstable propensity weights, and inflated variance — especially in low-overlap observational settings.

**UD-DML** resolves this tension in three phases:

| Phase | Description | Complexity |
|-------|-------------|------------|
| **1. UD Subsampling** | Construct a low-discrepancy skeleton in PCA-rotated covariate space via Good Lattice Points (GLP) with mixture discrepancy. Match nearest treated/control units to each skeleton point. | O(B_γ r²_p q) design search + O(r_p q log n) matching |
| **2. Cross-Fitted DML** | Standard K-fold DML on the selected r = 2r_p original observations. | C_DML(r, p, K) |
| **3. Wald Inference** | Point estimate, variance, and confidence interval from AIPW pseudo-outcomes. | O(r) |

The design search (Phase 1) is accelerated by an **optimised C backend** (`genUD.dll` / `libgenUD.so`) that achieves **300–550× speedup** over the pure-Python loop through symmetry exploitation, cache-friendly memory layout, and native multithreading.

### Key Properties (Theorems 1–3)

- **Representativeness**: the UD-selected subsample preserves empirical integration of PCA-induced functionals (Koksma–Hlawka bound).
- **Balance**: treated and control arms are automatically aligned in the dominant latent directions (common-skeleton construction).
- **Asymptotic normality**: the UD-DML estimator is √r-consistent and asymptotically Gaussian under standard DML regularity conditions.

---

## Installation

```bash
git clone https://github.com/BobZhangHT/UD-DML.git
cd UD-DML
pip install -r requirements.txt
```

### C Backend (Optional but Recommended)

The compiled C library speeds up the GLP design search by 300–550×. Build it for your platform:

**Linux (recommended for servers):**
```bash
# Generic (auto-detect CPU)
gcc -O3 -march=native -ffast-math -funroll-loops -fPIC -shared -pthread \
    -o libgenUD.so genUD.c

# AMD EPYC / Zen 4 (explicit AVX-512)
gcc -O3 -march=x86-64-v4 -ffast-math -funroll-loops -fPIC -shared -pthread \
    -o libgenUD.so genUD.c
```

**Windows (MinGW):**
```bash
gcc -O3 -march=native -ffast-math -funroll-loops -shared -static-libgcc \
    -o genUD.dll genUD.c
```

**macOS:**
```bash
gcc -O3 -march=native -ffast-math -funroll-loops -fPIC -shared -pthread \
    -o libgenUD.so genUD.c
```

Verify: `python -c "from genUD_wrapper import c_genUD_available; print(c_genUD_available())"`
→ should print `True`. If the library is absent, all functions fall back transparently to pure Python.

---

## Quick Start

```python
from data_generation import generate_obs_3_data
from methods import run_ud, run_unif
import config

# Generate synthetic data (OBS-3: low overlap, high heterogeneity)
data = generate_obs_3_data(n=100_000, p=10)

# UD-DML with r = 5000 subsample
ud = run_ud(
    data["X"], data["W"], data["Y_obs"], data["pi_true"],
    is_rct=False,
    r={"r_total": 5_000},
    k_folds=config.K_FOLDS,
)

# Naive uniform subsampling baseline
unif = run_unif(
    data["X"], data["W"], data["Y_obs"], data["pi_true"],
    is_rct=False,
    r={"r_total": 5_000},
    k_folds=config.K_FOLDS,
)

print(f"UD-DML   ATE={ud['est_ate']:.4f}  CI=[{ud['ci_lower']:.4f}, {ud['ci_upper']:.4f}]")
print(f"UNIF-DML ATE={unif['est_ate']:.4f}  CI=[{unif['ci_lower']:.4f}, {unif['ci_upper']:.4f}]")
# True ATE = 1.0
```

---

## Data-Generating Processes

Observational DGPs (Section 3.1). Paper focus is observational confounding; RCT variants remain in `data_generation.py` for ablations but are not driven by the simulation suite.

| Scenario | Covariates | g(X) | Δ(X) | Overlap |
|----------|-----------|------|------|---------|
| OBS-1 | X^(d) ~ U[−2,2] | linear | 1 + 0.2·X₃ | High |
| OBS-2 | Uniform block + Normal block | 0.5·X₁² + 0.5·X₂·X₃ + sin(X₆) | 1 + 0.5·X₁·X₂ | Moderate |
| OBS-3 | GMM block + Normal block | sin(π·X₁) + 0.5·X₂·X₃ + 0.1·X₆³ + 0.2·cos(X₇) | 1 + 0.5·tanh(X₁) + 0.2·X₆·X₇ | **Low** |
| OBS-3-overlap | same as OBS-3 | same as OBS-3 | same as OBS-3 | **Tunable** (`overlap_strength` c) |

All scenarios: p = 10, θ₀ = E[Δ(X)] = 1.0, Y = g(X) + W·Δ(X) + ε, ε ~ N(0,1).

`OBS-3-overlap` multiplies the propensity logit by `c ∈ ℝ₊`, sweeping from perfect overlap (c→0) to near-zero overlap (c ≥ 1.5) for the overlap-gradient experiment.

---

## Running the Simulation Suite

```bash
# All experiments (500 replications each, parallel)
python simulations.py --jobs -1

# A subset by name
python simulations.py --experiments experiment_subsample_size experiment_population_size

# Quick smoke test (10 replications)
python simulations.py --fast-demo --jobs 4

# Disable on-disk UD skeleton cache (for reproducibility audits)
python simulations.py --no-ud-disk-cache
```

### Experiments

| Key | Description | Paper Output |
|-----|-------------|:---:|
| `experiment_visualization` | Propensity density, Q-Q normality, SMD love plots. | Figs. 3–4 |
| `experiment_subsample_size` | RMSE / CI coverage / CI width vs subsample size r over OBS-1/2/3. | Fig. 1 |
| `experiment_overlap_gradient` | RMSE / CI metrics vs overlap coefficient c on OBS-3-overlap. | Fig. 2 + Table 1 |
| `experiment_double_robust` | Double-robustness under nuisance misspecification (4 × 3 grid). | Table 2 |
| `experiment_population_size` | Scalability: fixed r, increasing population n; bias–variance decomposition (UD / UNIF / FULL). | Table 3 |

### Standalone Profiling Runs

`simulations.py` additionally exposes two profiling entry points (selected via `--experiment`):

```bash
# C-backend efficiency profile on a single (scenario, n, r)
python simulations.py --experiment efficiency_profile \
    --scenario OBS-3 --n 100000 --r-total 1000 --replications 50

# Sensitivity of the UD estimator to the generator-search budget B_γ
python simulations.py --experiment bgamma_sensitivity \
    --scenario OBS-3 --n 100000 --r-total 1000 --replications 50
```

### Output Structure

```
simulation_results/<experiment>/
├── checkpoints/           # Per-replication .pkl.gz (resumable)
└── raw_results.pkl.gz     # Combined results

analysis_results/<experiment>/
├── tables/                # CSV + LaTeX tables
└── figures/               # PNG + PDF publication figures (300 dpi)
```

---

## Real-Data Analysis (CDC 2021 US Natality)

`real_data_analysis.py` reproduces the natality birth-weight application (Section 4):

- **Treatment** W: maternal smoking during pregnancy (CIG_REC)
- **Outcome** Y: birth weight in grams (DBWT), min–max scaled to [0, 1]
- **Covariates** X (p = 10): MAGER, MEDUC, PRECARE, PREVIS, SEX, DMAR, FAGECOMB, RF_GDIAB, RF_GHYPE, PRIORTERM

### Pipeline

1. **FULL-DML** runs once on the full clean sample (cached under `raw/full_reference.pkl.gz`) as the reference estimate.
2. **Nonparametric bootstrap**: for each of `B` replications, resample n rows with replacement. Within each resample, run UD-DML and UNIF-DML at every budget r in `--r-grid`.
3. The canonical budget `--canonical-r` (default 5000) slices Figure A (stability comparison); the full grid drives Figure B (scaling curve).
4. Per-rep caches (`raw/rep_<b>.pkl.gz`) make the run fully resumable — delete the cache with `--clear-cache` to restart.

### Run

```bash
python real_data_analysis.py \
    --data-path Nat2021us/Nat2021US.txt \
    --reps 100 --r-grid 1000,2500,5000,10000,25000 --canonical-r 5000 \
    --jobs 16

# Smoke test (B = 10 reps, separate output dir)
python real_data_analysis.py --fast-demo --data-path Nat2021us/Nat2021US.txt
```

### Outputs

```
real_data_results/
├── figures/real_data_plan_A.{png,pdf}   # bootstrap distribution + speedup
├── figures/real_data_plan_B.{png,pdf}   # r-scaling curve
├── tables/real_data_plan_A.tex          # canonical-r summary
├── tables/real_data_plan_B.tex          # r-scaling grid
└── raw/                                 # per-rep & FULL-reference caches
```

---

## Configuration

Key parameters in `config.py`:

```python
# Simulation
BASE_SEED = 20250919
DEFAULT_REPLICATIONS = 500       # Monte Carlo replications per variant
N_POPULATION = 500_000           # Full-data size n
K_FOLDS = 2                      # DML cross-fitting folds
MAX_PARALLEL_JOBS = 32           # Outer parallelism cap

# UD-DML (Algorithm 1)
UD_VARIANCE_THRESHOLD = 0.85     # ρ₀: PCA cumulative variance retention
UD_MAX_GENERATOR_CANDIDATES = 30 # B_γ: GLP generator search budget
UD_SKELETON_DISK_CACHE_DIR = Path("ud_skeleton_cache")

# Nuisance learners
DEFAULT_NUISANCE_LEARNER = "lgbm"   # "lgbm" | "rf" | "lasso_cv"
LGBM_N_ESTIMATORS = 100
```

### Environment Variables

| Variable | Effect |
|----------|--------|
| `OS_DML_MAX_JOBS` | Override `MAX_PARALLEL_JOBS` |
| `OS_DML_PRE_DISPATCH` | joblib pre-dispatch (e.g. `2*n_jobs`) |
| `OS_DML_PARALLEL_MEM_CHUNK` | Max tasks per `Parallel(...)` call |
| `UD_SKELETON_DISK_CACHE` | Skeleton cache directory (default `./ud_skeleton_cache`; `0` disables) |
| `UD_USE_C_BACKEND` | Set to `0` to force pure-Python GLP search |
| `UD_GENUD_NUM_THREADS` | Internal C thread count (auto by default; workers use 1) |
| `JOBLIB_TEMP_FOLDER` | joblib scratch (auto-set to `/dev/shm` on Linux) |

---

## Project Structure

```
UD-DML/
├── config.py              # Global configuration and experiment definitions
├── data_generation.py     # 6 DGPs (RCT-1..3, OBS-1..3, OBS-3-overlap)
├── methods.py             # Core: run_ud, run_unif, run_full
├── evaluation.py          # Aggregation, tables, publication figures
├── simulations.py         # Simulation driver (parallel, checkpointed)
├── real_data_analysis.py  # CDC 2021 natality bootstrap study (Section 4)
├── genUD.c                # Optimised C backend for GLP design search
├── genUD_wrapper.py       # ctypes bridge (genUD.dll / libgenUD.so)
├── Nat2021us/             # Natality dataset (not tracked; user-provided)
├── ud_skeleton_cache/     # On-disk GLP skeletons (auto-populated)
├── simulation_results/    # Raw + checkpointed simulation outputs
├── analysis_results/      # Aggregated tables and figures
├── real_data_results/     # Real-data figures, tables, caches
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Performance

Design search speedup (r_p = 5000, q = 8, B_γ = 30):

| Backend | Time | Speedup | D²_M |
|---------|------|---------|------|
| Python (numpy) | 208 s | 1× | 2.29 × 10⁻⁴ |
| C (single thread) | 2.2 s | 94× | identical |
| C (8 threads) | 0.74 s | 281× | identical |
| C (16 threads) | 0.47 s | **443×** | identical |

The C backend selects the **same optimal generator α** as pure Python (U matrix identical to machine epsilon 1.1 × 10⁻¹⁶).

Parallel simulation (n = 500k, 32 vCPU AMD EPYC 9654):
- 32 workers × 1 internal thread each
- UD skeleton lazily cached on disk → first-hit cost ~0.5 s, subsequent hits ~0 s
- Memory: ~1.5 GB per worker at n = 500k (fits 32 workers into 60 GB)

---

## Reproducing Paper Results

```bash
# 1. Build C backend
gcc -O3 -march=native -ffast-math -funroll-loops -fPIC -shared -pthread \
    -o libgenUD.so genUD.c

# 2. Full simulation suite (~6-12 h on a 32-core server)
python simulations.py --jobs -1

# 3. Real-data application (~1-3 h; depends on --reps and --jobs)
python real_data_analysis.py \
    --data-path Nat2021us/Nat2021US.txt \
    --reps 100 --r-grid 1000,2500,5000,10000,25000 --canonical-r 5000 \
    --jobs 16

# 4. Results appear in analysis_results/*/{tables,figures}/
#    and real_data_results/{figures,tables}/
```

Tables and figures match those in the paper (Tables 1–3, Figures 1–4).

---

## Citation

```bibtex
@article{qu2026uddml,
  title   = {UD-DML: Uniform Design Subsampling for Double Machine
             Learning over Massive Data},
  author  = {Qu, Yuanke and Xu, Xiaoya and Zhang, Hengtao},
  year    = {2026},
  journal = {}
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.
