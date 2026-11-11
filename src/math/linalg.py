"""Linear algebra utilities (port of matrix-linpack.f / asetup LU)."""

from __future__ import annotations

import numpy as np
from scipy.linalg import lu_factor, lu_solve


def ludcmp(a: np.ndarray, n: int, indx: np.ndarray, work: np.ndarray) -> None:
    """LU decomposition with partial pivoting; stores SciPy LAPACK factors in ``a``/``indx``.

    LAPACK works column-major: a Fortran-ordered ``a[:n, :n]`` is factored in
    place with no copy; any other layout is copied once and written back.
    """
    _ = work
    sub = a[:n, :n]
    lu, piv = lu_factor(sub, overwrite_a=True)
    if not np.shares_memory(lu, sub):
        sub[...] = lu
    indx[:n] = piv


def baksub(a: np.ndarray, n: int, indx: np.ndarray, b: np.ndarray) -> None:
    """Back-substitution for LU-decomposed system (in-place on ``b``).

    ``b[:n]`` may hold one RHS vector or a block of RHS columns. A
    Fortran-ordered ``a[:n, :n]`` is passed to LAPACK without a copy; the
    factors were finite-checked by :func:`ludcmp`, so only ``b`` is checked.
    """
    rhs = np.asarray_chkfinite(b[:n])
    b[:n] = lu_solve((a[:n, :n], indx[:n]), rhs, overwrite_b=True, check_finite=False)


def lusolve(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve Ax=b using SciPy LU factorization."""
    lu, piv = lu_factor(a.copy())
    return lu_solve((lu, piv), b.copy())
