"""Linear algebra utilities (port of matrix-linpack.f / asetup LU)."""

from __future__ import annotations

import numpy as np
from scipy.linalg import lu_factor, lu_solve


def ludcmp(a: np.ndarray, n: int, indx: np.ndarray, work: np.ndarray) -> None:
    """LU decomposition with partial pivoting; stores SciPy LAPACK factors in ``a``/``indx``."""
    _ = work
    lu, piv = lu_factor(a[:n, :n], overwrite_a=True)
    a[:n, :n] = lu
    indx[:n] = piv


def baksub(a: np.ndarray, n: int, indx: np.ndarray, b: np.ndarray) -> None:
    """Back-substitution for LU-decomposed system (in-place on ``b``)."""
    b[:n] = lu_solve((a[:n, :n], indx[:n]), b[:n], overwrite_b=True)


def lusolve(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve Ax=b using SciPy LU factorization."""
    lu, piv = lu_factor(a.copy())
    return lu_solve((lu, piv), b.copy())
