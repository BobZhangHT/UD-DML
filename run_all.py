#!/usr/bin/env python
"""One-command reproducibility entry point for UD-DML.

Examples
--------
Local validation::

    python run_all.py --fast-demo --jobs 4

Submission-scale server run::

    python run_all.py --full --jobs 12 --data-path Nat2021us/Nat2021US.txt
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def _run(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("[run]", " ".join(command), flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=os.environ.copy(),
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
        return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the complete UD-DML workflow.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--fast-demo", action="store_true")
    mode.add_argument("--full", action="store_true")
    parser.add_argument("--jobs", type=int, default=-1)
    parser.add_argument(
        "--data-path", default="Nat2021us/Nat2021US.txt", help="Natality data path."
    )
    parser.add_argument(
        "--skip-real-data",
        action="store_true",
        help="Run simulations and profiles only; otherwise missing data is fatal.",
    )
    return parser.parse_args()


def main() -> None:
    # Windows terminals may default to GBK while source comments/log messages
    # contain Unicode. Keep the workflow alive and preserve the UTF-8 log.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args()
    try:
        import lightgbm  # type: ignore
        learner_backend = f"lightgbm-{lightgbm.__version__}"
    except ImportError:
        learner_backend = "sklearn-gradient-boosting-fallback"
        if args.full:
            raise RuntimeError(
                "Full mode requires LightGBM. Install requirements.txt "
                "before generating submission-scale results."
            )
    mode_name = "fast_demo" if args.fast_demo else "full"
    logs = ROOT / "run_logs" / mode_name
    started = time.time()
    python = sys.executable

    commands: list[tuple[str, list[str]]] = []
    build_command = [python, "build_genud.py", "--if-needed"]
    commands.append(("build_genud", build_command))
    commands.append(
        (
            "tests",
            [python, "-m", "pytest", "-q", "tests"],
        )
    )
    simulation_command = [python, "simulations.py", f"--{mode_name.replace('_', '-')}", "--jobs", str(args.jobs)]
    commands.append(("simulations", simulation_command))

    if args.fast_demo:
        profile_common = ["--scenario", "OBS-3", "--n", "5000", "--r-total", "500", "--replications", "2"]
    else:
        profile_common = ["--scenario", "OBS-3", "--n", "500000", "--r-total", "5000", "--replications", "50"]
    commands.append(
        (
            "efficiency_profile",
            [
                python, "simulations.py", "--experiment", "efficiency_profile",
                *profile_common, "--standalone-output-dir",
                f"analysis_results/efficiency_profile_{mode_name}",
            ],
        )
    )
    commands.append(
        (
            "bgamma_sensitivity",
            [
                python, "simulations.py", "--experiment", "bgamma_sensitivity",
                *profile_common, "--standalone-output-dir",
                f"analysis_results/bgamma_sensitivity_{mode_name}",
            ],
        )
    )

    data_path = Path(args.data_path)
    if not data_path.is_absolute():
        data_path = ROOT / data_path
    if not args.skip_real_data:
        if not data_path.exists():
            raise FileNotFoundError(
                f"Natality data not found at {data_path}. Upload it or use --skip-real-data."
            )
        real_command = [
            python,
            "real_data_analysis.py",
            "--fast-demo" if args.fast_demo else "--full",
            "--data-path",
            str(data_path),
            "--jobs",
            str(args.jobs),
            "--out",
            "real_data_results",
        ]
        commands.append(("real_data", real_command))

    completed: list[str] = []
    try:
        for label, command in commands:
            _run(command, logs / f"{label}.log")
            completed.append(label)
    finally:
        manifest = {
            "mode": mode_name,
            "started_unix": started,
            "elapsed_seconds": time.time() - started,
            "completed_stages": completed,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "jobs": args.jobs,
            "learner_backend": learner_backend,
            "data_path": None if args.skip_real_data else str(data_path),
        }
        logs.mkdir(parents=True, exist_ok=True)
        (logs / "workflow_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
