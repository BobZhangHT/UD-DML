# UD-DML

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)

Official implementation of **Uniform-Design Subsampling for Compute-Budgeted Double Machine Learning**.

UD-DML constructs a common low-discrepancy skeleton in retained PCA coordinates, matches each anchor without replacement to one treated and one control observation, and runs cross-fitted DML on the selected original observations. The repository also contains three fixed-budget comparison designs:

- `UNIF-DML`: uniform sampling without arm-size control;
- `STRAT-UNIF-DML`: equal allocation by treatment arm;
- `SEP-UD-DML`: separate arm-specific uniform designs;
- `UD-DML`: pooled transformation and a common treated-control skeleton.

This public repository contains only the code and metadata required to run and validate the method. Raw data, manuscripts, cached designs, logs, and generated results are intentionally excluded.

## Repository contents

```text
UD-DML/
|-- run_all.py               # One-command validation or full workflow
|-- simulations.py           # Checkpointed simulation driver
|-- real_data_analysis.py    # CDC 2021 natality application
|-- methods.py               # Estimators, selection, matching, and inference
|-- data_generation.py       # Simulation data-generating processes
|-- evaluation.py            # Tables and publication figures
|-- config.py                # Experiment and learner configuration
|-- genUD.c                  # Optional native skeleton-search backend
|-- genUD_wrapper.py         # ctypes bridge with Python fallback
|-- build_genud.py           # Cross-platform native build helper
|-- tests/                   # Regression and contract tests
|-- pytest.ini
|-- requirements.txt
|-- LICENSE
`-- README.md
```

## Installation

Python 3.11 or newer is recommended. Create an isolated environment and install the single dependency file:

```bash
python -m venv .venv

# Linux/macOS
source .venv/bin/activate

# Windows PowerShell
# .\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If a private package mirror does not carry the pinned versions, install from PyPI explicitly:

```bash
python -m pip install --index-url https://pypi.org/simple -r requirements.txt
```

Build the optional C backend and run the tests:

```bash
python build_genud.py --if-needed
python -m pytest -q tests
```

`build_genud.py` produces `genUD.dll`, `libgenUD.so`, or `libgenUD.dylib` for the current platform. These binaries are generated locally and are not committed. If no C compiler is available, the verified Python implementation remains available.

Check the active backend with:

```bash
python -c "import methods; print(methods.ud_c_backend_active())"
```

LightGBM is required for submission-scale `--full` runs. Fast-demo mode can use the scikit-learn fallback when LightGBM is unavailable.

## Fast local validation

The fastest end-to-end check does not require the natality data:

```bash
python -u run_all.py --fast-demo --jobs 4 --skip-real-data
```

This command:

1. builds the native backend when possible;
2. runs the regression tests;
3. runs every configured simulation family with reduced demo settings;
4. generates the efficiency profile and generator-budget diagnostic;
5. writes stage logs and a workflow manifest under `run_logs/fast_demo/`.

Fast-demo outputs use dedicated directories and are not submission-scale evidence.

## Full simulation workflow

Run all submission-scale simulations without the private real-data input:

```bash
python -u run_all.py --full --jobs 6 --skip-real-data
```

The example uses six outer workers, which is a conservative setting for a 12-vCPU, 32-GB machine. Each LightGBM fit uses one learner thread; parallelism is across Monte Carlo replications. For a memory-safe background run on Linux:

```bash
mkdir -p tmp/joblib

nohup env \
  OMP_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 \
  MALLOC_ARENA_MAX=2 \
  OS_DML_WORKER_NUM_THREADS=1 \
  UD_GENUD_NUM_THREADS=1 \
  OS_DML_MAX_JOBS=6 \
  OS_DML_PRE_DISPATCH=1 \
  OS_DML_PARALLEL_MEM_CHUNK=6 \
  JOBLIB_TEMP_FOLDER="$PWD/tmp/joblib" \
  TMPDIR="$PWD/tmp" \
  python -u run_all.py --full --jobs 6 --skip-real-data \
  > full_scale_console.log 2>&1 < /dev/null &

echo $! > full_scale.pid
```

Monitor it with:

```bash
ps -fp "$(cat full_scale.pid)"
tail -f full_scale_console.log
```

The simulation runner is checkpointed. Re-running the same command resumes compatible completed cells rather than mixing results from a different result schema.

### Run selected experiments

```bash
# Reduced smoke test
python simulations.py --fast-demo --jobs 4

# Full configured suite
python simulations.py --full --jobs 6

# Selected configured families
python simulations.py --full --jobs 6 \
  --experiments experiment_subsample_size experiment_overlap_gradient

# Standalone profiling experiment
python simulations.py --experiment efficiency_profile \
  --scenario OBS-3 --n 100000 --r-total 1000 --replications 50 \
  --standalone-output-dir analysis_results/efficiency_profile

# Standalone generator-budget sensitivity
python simulations.py --experiment bgamma_sensitivity \
  --scenario OBS-3 --n 100000 --r-total 1000 --replications 50 \
  --standalone-output-dir analysis_results/bgamma_sensitivity
```

Main generated directories are:

```text
simulation_results/    raw results and resumable checkpoints
analysis_results/      aggregated CSV/LaTeX tables and PDF/PNG figures
ud_skeleton_cache/     reusable generated skeletons
run_logs/              one-command workflow logs and manifests
```

All are ignored by Git.

## Real-data application

The application uses the 2021 US Natality fixed-width file from the National Center for Health Statistics. The data are not redistributed. Place the file at:

```text
Nat2021us/Nat2021US.txt
```

Run a local smoke test:

```bash
python -u real_data_analysis.py --fast-demo \
  --data-path Nat2021us/Nat2021US.txt --jobs 4
```

Run the complete fixed-data analysis:

```bash
python -u real_data_analysis.py --full \
  --data-path Nat2021us/Nat2021US.txt \
  --reps 100 \
  --r-grid 1000,2500,5000,10000,25000 \
  --canonical-r 5000 \
  --jobs 4
```

To include it in the one-command workflow:

```bash
python -u run_all.py --full --jobs 4 \
  --data-path Nat2021us/Nat2021US.txt
```

The application writes its cached fits, summaries, and consolidated figure under `real_data_results/`. Use `--clear-cache` only when an intentional clean restart is required.

## Python API example

```python
from data_generation import generate_obs_3_data
from methods import run_ud

data = generate_obs_3_data(n=20_000, p=10)

result = run_ud(
    data["X"],
    data["W"],
    data["Y_obs"],
    data["pi_true"],
    is_rct=False,
    r={"r_total": 1_000},
    k_folds=2,
    sim_seed=20250919,
)

print(result["est_ate"], result["ci_lower"], result["ci_upper"])
```

The total UD-DML budget must be even because the method selects `r_total / 2` distinct observations from each treatment arm.

## Reproducibility controls

Important settings are defined in `config.py`:

- `BASE_SEED`: hierarchical random-seed root;
- `DEFAULT_REPLICATIONS`: submission-scale Monte Carlo replications;
- `N_POPULATION`: default simulated population size;
- `K_FOLDS`: simulation cross-fitting folds;
- `UD_VARIANCE_THRESHOLD`: retained PCA variance threshold;
- `UD_MAX_GENERATOR_CANDIDATES`: skeleton-search budget;
- `DEFAULT_NUISANCE_LEARNER`: nuisance learner used by the main workflow.

Useful environment variables include:

| Variable | Purpose |
|---|---|
| `OS_DML_MAX_JOBS` | Cap outer Monte Carlo workers |
| `OS_DML_PRE_DISPATCH` | Control joblib task pre-dispatch |
| `OS_DML_PARALLEL_MEM_CHUNK` | Limit tasks submitted per parallel chunk |
| `OS_DML_WORKER_NUM_THREADS` | Set numerical-library threads inside workers |
| `UD_GENUD_NUM_THREADS` | Set native skeleton-search threads |
| `UD_USE_C_BACKEND=0` | Force the Python skeleton-search implementation |
| `UD_SKELETON_DISK_CACHE=0` | Disable the on-disk skeleton cache |
| `JOBLIB_TEMP_FOLDER` | Set joblib scratch storage |

## What is intentionally not tracked

The `.gitignore` uses a root-level allowlist. The following remain local:

- the CDC natality file and any other private/raw data;
- manuscripts and compiled paper PDFs;
- native binaries generated from `genUD.c`;
- simulation and real-data results;
- checkpoints, caches, temporary directories, and logs;
- local archives and editor state.

Before publishing, inspect the exact upload set with:

```bash
git status --short
git ls-files
```

## Citation

```bibtex
@article{qu2026uddml,
  title   = {Uniform-Design Subsampling for Compute-Budgeted Double Machine Learning},
  author  = {Qu, Yuanke and Xu, Xiaoya and Zhang, Hengtao},
  year    = {2026}
}
```

## License

This project is released under the [MIT License](LICENSE).
