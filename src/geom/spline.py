"""Spline interpolation utilities (port of spline.f)."""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve_banded

NMAX = 1000


def trisol(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray, kk: int) -> np.ndarray:
    """Tridiagonal system solver via ``scipy.linalg.solve_banded``."""
    ab = np.zeros((3, kk), dtype=np.float64)
    ab[0, 1:kk] = c[: kk - 1]
    ab[1, :kk] = a[:kk]
    ab[2, : kk - 1] = b[1:kk]
    d[:kk] = solve_banded((1, 1), ab, d[:kk].copy())
    return d


def _find_interval_index(ss: float, s: np.ndarray, n: int) -> int:
    """Return 1-based interval index matching spline.f convention."""
    ilow = 1
    i = n
    while i - ilow > 1:
        imid = (i + ilow) // 2
        if ss < s[imid - 1]:
            i = imid
        else:
            ilow = imid
    return i


def spline(x: np.ndarray, xs: np.ndarray, s: np.ndarray, n: int) -> np.ndarray:
    """Natural cubic spline second-derivative coefficients."""
    if n > NMAX:
        raise ValueError("SPLINE: array overflow, increase NMAX")

    a = np.zeros(NMAX, dtype=np.float64)
    b = np.zeros(NMAX, dtype=np.float64)
    c = np.zeros(NMAX, dtype=np.float64)
    xs_out = xs.copy()

    for i in range(1, n - 1):
        dsm = s[i] - s[i - 1]
        dsp = s[i + 1] - s[i]
        b[i] = dsp
        a[i] = 2.0 * (dsm + dsp)
        c[i] = dsm
        term1 = (x[i + 1] - x[i]) * (dsm / dsp)
        term2 = (x[i] - x[i - 1]) * (dsp / dsm)
        xs_out[i] = 3.0 * (term1 + term2)

    a[0] = 2.0
    c[0] = 1.0
    xs_out[0] = 3.0 * ((x[1] - x[0]) / (s[1] - s[0]))
    b[n - 1] = 1.0
    a[n - 1] = 2.0
    xs_out[n - 1] = 3.0 * ((x[n - 1] - x[n - 2]) / (s[n - 1] - s[n - 2]))

    xs_out = trisol(a, b, c, xs_out, n)
    xs[:] = xs_out
    return xs_out


def splind(
    x: np.ndarray,
    xs: np.ndarray,
    s: np.ndarray,
    n: int,
    xs1: float,
    xs2: float,
) -> np.ndarray:
    """Cubic spline with specified endpoint slopes."""
    if n > NMAX:
        raise ValueError("SPLIND: array overflow, increase NMAX")

    a = np.zeros(NMAX, dtype=np.float64)
    b = np.zeros(NMAX, dtype=np.float64)
    c = np.zeros(NMAX, dtype=np.float64)
    xs_out = xs.copy()

    for i in range(1, n - 1):
        dsm = s[i] - s[i - 1]
        dsp = s[i + 1] - s[i]
        b[i] = dsp
        a[i] = 2.0 * (dsm + dsp)
        c[i] = dsm
        term1 = (x[i + 1] - x[i]) * (dsm / dsp)
        term2 = (x[i] - x[i - 1]) * (dsp / dsm)
        xs_out[i] = 3.0 * (term1 + term2)

    if xs1 == 999.0:
        a[0] = 2.0
        c[0] = 1.0
        xs_out[0] = 3.0 * ((x[1] - x[0]) / (s[1] - s[0]))
    elif xs1 == -999.0:
        a[0] = 1.0
        c[0] = 1.0
        xs_out[0] = 2.0 * ((x[1] - x[0]) / (s[1] - s[0]))
    else:
        a[0] = 1.0
        c[0] = 0.0
        xs_out[0] = xs1

    if xs2 == 999.0:
        b[n - 1] = 1.0
        a[n - 1] = 2.0
        xs_out[n - 1] = 3.0 * ((x[n - 1] - x[n - 2]) / (s[n - 1] - s[n - 2]))
    elif xs2 == -999.0:
        b[n - 1] = 1.0
        a[n - 1] = 1.0
        xs_out[n - 1] = 2.0 * ((x[n - 1] - x[n - 2]) / (s[n - 1] - s[n - 2]))
    else:
        a[n - 1] = 1.0
        b[n - 1] = 0.0
        xs_out[n - 1] = xs2

    if n == 2 and xs1 == -999.0 and xs2 == -999.0:
        b[n - 1] = 1.0
        a[n - 1] = 2.0
        xs_out[n - 1] = 3.0 * ((x[n - 1] - x[n - 2]) / (s[n - 1] - s[n - 2]))

    xs_out = trisol(a, b, c, xs_out, n)
    xs[:] = xs_out
    return xs_out


def seval(ss: float, x: np.ndarray, xs: np.ndarray, s: np.ndarray, n: int) -> float:
    """Evaluate spline at parameter ss."""
    i = _find_interval_index(ss, s, n)
    ds = s[i - 1] - s[i - 2]
    t = (ss - s[i - 2]) / ds
    cx1 = ds * xs[i - 2] - x[i - 1] + x[i - 2]
    cx2 = ds * xs[i - 1] - x[i - 1] + x[i - 2]
    term = (t - t * t) * ((1.0 - t) * cx1 - t * cx2)
    val = t * x[i - 1] + (1.0 - t) * x[i - 2] + term
    return float(val)


def deval(ss: float, x: np.ndarray, xs: np.ndarray, s: np.ndarray, n: int) -> float:
    """Evaluate spline derivative at parameter ss."""
    i = _find_interval_index(ss, s, n)
    ds = s[i - 1] - s[i - 2]
    t = (ss - s[i - 2]) / ds
    cx1 = ds * xs[i - 2] - x[i - 1] + x[i - 2]
    cx2 = ds * xs[i - 1] - x[i - 1] + x[i - 2]
    val = (
        (x[i - 1] - x[i - 2])
        + (1.0 - 4.0 * t + 3.0 * t * t) * cx1
        + t * (3.0 * t - 2.0) * cx2
    )
    return float(val / ds)
