#!/usr/bin/env python
"""Build the optional genUD C backend on Windows, Linux, or macOS."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def output_path() -> Path:
    if os.name == "nt":
        return ROOT / "genUD.dll"
    if sys.platform == "darwin":
        return ROOT / "libgenUD.dylib"
    return ROOT / "libgenUD.so"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--if-needed", action="store_true")
    args = parser.parse_args()
    source = ROOT / "genUD.c"
    target = output_path()
    if args.if_needed and target.exists() and target.stat().st_mtime >= source.stat().st_mtime:
        print(f"[build] current backend: {target.name}")
        return
    compiler = next((name for name in ("gcc", "clang", "cc") if shutil.which(name)), None)
    if compiler is None:
        print("[build] no C compiler found; the verified Python fallback remains available")
        return
    common = [compiler, "-O3", "-shared", str(source), "-o", str(target)]
    if os.name != "nt":
        common[1:1] = ["-fPIC"]
    attempts = [common + ["-fopenmp"], common]
    for command in attempts:
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"[build] created {target.name}")
            return
    raise RuntimeError(result.stderr.strip() or "genUD compilation failed")


if __name__ == "__main__":
    main()
