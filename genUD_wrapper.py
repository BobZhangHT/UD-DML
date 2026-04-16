# -*- coding: utf-8 -*-
"""genUD_wrapper.py — ctypes bridge to the compiled C implementation.

Provides a *primitive* interface so that `methods.py` can drive the full
candidate enumeration + caching logic in Python, delegating only the
hot inner loop (GLP construction + mixture discrepancy on a list of
admissible generators) to compiled C (``genUD.dll`` / ``libgenUD.so``).

Public API
----------
    c_genUD_available()  -> bool
    c_genUD_search(alphas, r_p, q) -> (U_best, alpha, D2_M)
    c_mixture_d2(U)      -> float
    c_glp_construct(r_p, q, alpha) -> np.ndarray

The returned skeleton is *bit-identical* to the Python baseline for the
GLP construction (same integer arithmetic), and matches the baseline's
selected alpha under mixture discrepancy up to floating-point rounding
(≲ 1e-10 in D²_M), because the C code uses the same closed-form formula
with a symmetric-triangle summation that is analytically equivalent.

Build (Windows, MinGW UCRT):
    gcc -O3 -march=native -ffast-math -shared -static-libgcc -o genUD.dll genUD.c
Build (Linux / macOS):
    gcc -O3 -march=native -ffast-math -fPIC -shared -pthread -o libgenUD.so genUD.c
"""
from __future__ import annotations

import ctypes
import os
import sys
import threading
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np


_LIB_NAME = "genUD.dll" if os.name == "nt" else "libgenUD.so"
_LIB_PATH = Path(__file__).with_name(_LIB_NAME)

_LIB_LOCK = threading.Lock()
_LIB: Optional[ctypes.CDLL] = None
_LIB_ERR: Optional[str] = None


def _load_lib() -> Optional[ctypes.CDLL]:
    """Load the compiled DLL on first use; cache the handle module-wide.

    Returns the loaded library, or ``None`` if loading fails (e.g., DLL
    not built yet).  Failure reason is stored in ``_LIB_ERR``.
    """
    global _LIB, _LIB_ERR
    if _LIB is not None:
        return _LIB
    with _LIB_LOCK:
        if _LIB is not None:
            return _LIB
        if not _LIB_PATH.is_file():
            _LIB_ERR = (
                f"{_LIB_NAME} not found next to {__file__}. "
                f"Build with the gcc command in the module docstring."
            )
            return None
        try:
            lib = ctypes.CDLL(str(_LIB_PATH))
        except OSError as exc:  # pragma: no cover — diagnostics only
            _LIB_ERR = f"{_LIB_NAME} failed to load: {exc!r}"
            return None
        lib.c_genUD.restype = ctypes.c_int
        lib.c_genUD.argtypes = [
            ctypes.c_int64, ctypes.c_int,
            ctypes.POINTER(ctypes.c_int64), ctypes.c_int,
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
        ]
        lib.c_mixture_d2.restype = ctypes.c_double
        lib.c_mixture_d2.argtypes = [
            ctypes.POINTER(ctypes.c_double), ctypes.c_int64, ctypes.c_int,
        ]
        lib.c_glp_construct.restype = None
        lib.c_glp_construct.argtypes = [
            ctypes.c_int64, ctypes.c_int, ctypes.c_int64,
            ctypes.POINTER(ctypes.c_double),
        ]
        _LIB = lib
        return _LIB


def c_genUD_available() -> bool:
    """True iff the compiled C routine loaded successfully in this process."""
    return _load_lib() is not None


def c_genUD_last_error() -> Optional[str]:
    """Reason (if any) that the compiled routine could not be used."""
    return _LIB_ERR


# ---------------------------------------------------------------------------
#  Primitive C calls
# ---------------------------------------------------------------------------

def c_mixture_d2(U: np.ndarray) -> float:
    """Compute D²_M of a design ``U[r_p, q]`` via the compiled C routine."""
    lib = _load_lib()
    if lib is None:
        raise RuntimeError(f"genUD C library unavailable: {_LIB_ERR}")
    U = np.ascontiguousarray(U, dtype=np.float64)
    r_p, q = U.shape
    return float(lib.c_mixture_d2(
        U.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int64(r_p), ctypes.c_int(q),
    ))


def c_glp_construct(r_p: int, q: int, alpha: int) -> np.ndarray:
    """Build a single GLP candidate via the compiled C routine."""
    lib = _load_lib()
    if lib is None:
        raise RuntimeError(f"genUD C library unavailable: {_LIB_ERR}")
    U = np.empty((int(r_p), int(q)), dtype=np.float64, order="C")
    lib.c_glp_construct(
        ctypes.c_int64(int(r_p)), ctypes.c_int(int(q)),
        ctypes.c_int64(int(alpha)),
        U.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )
    return U


def c_genUD_search(
    alphas: Sequence[int],
    r_p: int,
    q: int,
) -> Tuple[np.ndarray, int, float]:
    """Evaluate a *pre-computed* list of admissible power generators in C.

    The Python caller owns the enumeration (``methods._find_admissible_generators``)
    so that the candidate set is identical to the Python baseline; this
    routine only performs the heavy inner loop (GLP construction +
    mixture-discrepancy scan) in compiled code.

    Parameters
    ----------
    alphas : sequence of int
        Admissible power generators to evaluate (non-empty).
    r_p : int
        Number of skeleton pairs.
    q : int
        Retained PCA dimension (number of factors).

    Returns
    -------
    U_best : ndarray of shape (r_p, q)
        Best GLP skeleton in ``[0, 1]^q``.
    alpha  : int
        Selected power generator (minimum D²_M).
    D2_M   : float
        Squared mixture discrepancy of ``U_best``.
    """
    lib = _load_lib()
    if lib is None:
        raise RuntimeError(f"genUD C library unavailable: {_LIB_ERR}")
    alphas_arr = np.ascontiguousarray(np.asarray(alphas, dtype=np.int64))
    if alphas_arr.size == 0:
        raise ValueError("c_genUD_search requires at least one admissible alpha.")
    U_out = np.empty((int(r_p), int(q)), dtype=np.float64, order="C")
    best_d = ctypes.c_double(np.inf)
    best_idx = lib.c_genUD(
        ctypes.c_int64(int(r_p)), ctypes.c_int(int(q)),
        alphas_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        ctypes.c_int(int(alphas_arr.size)),
        U_out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.byref(best_d),
    )
    if best_idx < 0:
        raise RuntimeError(f"c_genUD failed (returned {best_idx})")
    return U_out, int(alphas_arr[best_idx]), float(best_d.value)


# ---------------------------------------------------------------------------
#  Self-test — compare against the Python baseline (run as a script).
# ---------------------------------------------------------------------------

if __name__ == "__main__":  # pragma: no cover
    import time
    # Import here to avoid a circular dependency at module import time;
    # this block only runs in the standalone smoke-test.
    import methods as _M  # noqa: E402

    if not c_genUD_available():
        print(f"C library not available: {c_genUD_last_error()}")
        sys.exit(1)

    for (r_p, q) in [(300, 6), (500, 8), (1250, 8), (2500, 8)]:
        rng_py = np.random.default_rng(12345)
        rng_c  = np.random.default_rng(12345)
        gens_c = _M._find_admissible_generators(r_p, q, 30, rng_c)
        t0 = time.perf_counter()
        U_c, alpha_c, d_c = c_genUD_search(gens_c, r_p, q)
        t_c = time.perf_counter() - t0

        t1 = time.perf_counter()
        gens_py = _M._find_admissible_generators(r_p, q, 30, rng_py)
        best_py_d = np.inf
        best_py_alpha = -1
        best_py_U = None
        for a in gens_py:
            U = _M._construct_glp_design(r_p, q, a)
            d = _M._mixture_discrepancy_squared(U)
            if d < best_py_d:
                best_py_d = d
                best_py_alpha = a
                best_py_U = U
        t_py = time.perf_counter() - t1

        rel_err = abs(d_c - best_py_d) / max(abs(best_py_d), 1e-300)
        u_diff = (np.max(np.abs(U_c - best_py_U))
                  if best_py_U is not None else float("nan"))
        print(
            f"r_p={r_p:<5d} q={q:<2d} | "
            f"C: alpha={alpha_c:<6d} D2={d_c:.6e}  time={t_c:7.3f}s | "
            f"PY: alpha={best_py_alpha:<6d} D2={best_py_d:.6e}  "
            f"time={t_py:7.3f}s | "
            f"same_alpha={alpha_c == best_py_alpha}  "
            f"relErr(D2)={rel_err:.2e}  max|U_c-U_py|={u_diff:.2e}  "
            f"speedup={t_py/max(t_c,1e-9):6.1f}x"
        )
