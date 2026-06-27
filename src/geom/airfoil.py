"""Airfoil camber helpers for geometry construction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from openavl.geom.spacing import akima, nrmliz
from openavl.geom.spline import deval, seval, splind


@dataclass
class AirfoilCamber:
    """Chordwise airfoil tables used during lattice construction.

    ``x`` is chord fraction, ``s`` is camber-line slope ``dz/dx``, ``c`` is
    camber-line height, and ``t`` is thickness.
    """

    x: np.ndarray
    s: np.ndarray
    c: np.ndarray
    t: np.ndarray


def _scalc(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Calculate arc length along an airfoil coordinate polyline."""
    s = np.zeros_like(x)
    for i in range(1, x.size):
        s[i] = s[i - 1] + np.hypot(x[i] - x[i - 1], y[i] - y[i - 1])
    return s


def _segspl(values: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Spline values against arc length, matching AVL's SEGSPL behavior."""
    deriv = np.zeros_like(values)
    start = 0
    n = values.size
    for i in range(1, max(1, n - 2)):
        if s[i] == s[i + 1]:
            count = i - start + 1
            splind(values[start : start + count], deriv[start : start + count], s[start : start + count], count, -999.0, -999.0)
            start = i + 1
    count = n - start
    splind(values[start : start + count], deriv[start : start + count], s[start : start + count], count, -999.0, -999.0)
    return deriv


def _d2val(ss: float, x: np.ndarray, xs: np.ndarray, s: np.ndarray) -> float:
    """Evaluate second derivative for the AVL cubic spline form."""
    n = x.size
    ilow = 0
    i = n - 1
    while i - ilow > 1:
        imid = (i + ilow) // 2
        if ss < s[imid]:
            i = imid
        else:
            ilow = imid
    ds = s[i] - s[i - 1]
    if ds == 0.0:
        return 0.0
    t = (ss - s[i - 1]) / ds
    cx1 = ds * xs[i - 1] - x[i] + x[i - 1]
    cx2 = ds * xs[i] - x[i] + x[i - 1]
    return float(((6.0 * t - 4.0) * cx1 + (6.0 * t - 2.0) * cx2) / (ds * ds))


def _lefind(x: np.ndarray, xp: np.ndarray, y: np.ndarray, yp: np.ndarray, s: np.ndarray) -> float:
    """Find the spline arc-length location of the airfoil leading edge."""
    _ = y, yp
    sle = float(s[0])
    for i in range(1, x.size):
        if x[i] > x[i - 1]:
            sle = float(s[i - 1])
            break

    sref = float(s[-1] - s[0])
    if sref == 0.0:
        return sle
    for _iter in range(20):
        resp = _d2val(sle, x, xp, s)
        if resp == 0.0:
            break
        dsle = -deval(sle, x, xp, s, x.size) / resp
        sle += dsle
        if abs(dsle) / sref < 1.0e-5:
            return float(sle)
    return float(sle)


def _normit(sle: float, x: np.ndarray, y: np.ndarray, s: np.ndarray, xp: np.ndarray) -> float:
    """Normalize airfoil coordinates and arc length to unit chord."""
    xle = seval(sle, x, xp, s, x.size)
    xte = 0.5 * (x[0] + x[-1])
    chord = xte - xle
    if chord == 0.0:
        return sle
    dnorm = 1.0 / chord
    x[:] = (x - xle) * dnorm
    y[:] = y * dnorm
    s[:] = s * dnorm
    return float(sle * dnorm)


def _sinvrt(si: float, xi: float, x: np.ndarray, xs: np.ndarray, s: np.ndarray) -> float:
    """Invert spline x(s) near a supplied initial guess."""
    sref = s[-1] - s[0]
    if sref == 0.0:
        return si
    for _iter in range(10):
        resp = deval(si, x, xs, s, x.size)
        if resp == 0.0:
            break
        ds = -(seval(si, x, xs, s, x.size) - xi) / resp
        si += ds
        if abs(ds / sref) < 1.0e-5:
            return float(si)
    return float(si)


def getcam(
    x_in: np.ndarray,
    y_in: np.ndarray,
    nc: int,
    normalize: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return AVL GETCAM camber and thickness arrays from surface coordinates."""
    return _getcam(x_in, y_in, nc, normalize)


def _getcam(x_in: np.ndarray, y_in: np.ndarray, nc: int, normalize: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return AVL GETCAM camber and thickness arrays from surface coordinates."""
    x = np.asarray(x_in, dtype=np.float64).copy()
    y = np.asarray(y_in, dtype=np.float64).copy()
    n = x.size
    s = _scalc(x, y)
    xp = _segspl(x, s)
    yp = _segspl(y, s)

    sle = _lefind(x, xp, y, yp, s)
    if normalize:
        sle = _normit(sle, x, y, s, xp)
        xp = _segspl(x, s)
        yp = _segspl(y, s)

    xle = seval(sle, x, xp, s, n)
    yle = seval(sle, y, yp, s, n)
    xte = 0.5 * (x[0] + x[-1])

    xc = np.zeros(nc, dtype=np.float64)
    yc = np.zeros(nc, dtype=np.float64)
    tc = np.zeros(nc, dtype=np.float64)
    xc[0] = xle
    yc[0] = yle

    su = sle - 0.01
    sl = sle + 0.01
    for i in range(1, nc):
        xout = xle + (xte - xle) * 0.5 * (1.0 - np.cos(np.pi * i / (nc - 1)))
        su = _sinvrt(su, xout, x, xp, s)
        yu = seval(su, y, yp, s, n)
        sl = _sinvrt(sl, xout, x, xp, s)
        yl = seval(sl, y, yp, s, n)
        xc[i] = xout
        yc[i] = 0.5 * (yu + yl)
        tc[i] = yu - yl
    return xc, yc, tc


def build_naca_slope(code: str = "0000", samples: int = 60) -> AirfoilCamber:
    """Build NACA 4-digit camber, slope, and thickness tables."""
    digits = str(code or "").zfill(4)
    m = int(digits[0]) / 100.0
    p = int(digits[1]) / 10.0
    t_frac = int(digits[2:4]) / 100.0
    xs = np.linspace(0.0, 1.0, samples + 1, dtype=np.float64)
    slope = np.zeros_like(xs)
    camber = np.zeros_like(xs)
    thick = np.zeros_like(xs)
    for i, xf in enumerate(xs):
        if m != 0.0 and p != 0.0:
            if xf < p:
                slope[i] = m * 2.0 * (p - xf) / (p * p)
                camber[i] = m * (2.0 * p * xf - 1.0) * xf / (p * p)
            elif xf > p:
                slope[i] = m * 2.0 * (p - xf) / ((1.0 - p) ** 2)
                camber[i] = m * ((1.0 - 2.0 * p) + (2.0 * p - xf) * xf) / ((1.0 - p) ** 2)
        thick[i] = (
            0.29690 * np.sqrt(max(xf, 0.0))
            - 0.12600 * xf
            - 0.35160 * xf * xf
            + 0.28430 * xf**3
            - 0.10150 * xf**4
        ) * t_frac * 10.0
    return AirfoilCamber(x=xs, s=slope, c=camber, t=thick)


def build_camber_slope(coords: list[list[float]], samples: int = 50) -> AirfoilCamber | None:
    """Build camber slope from coordinate pairs (upper/lower airfoil outline)."""
    if not coords:
        return None
    x = np.array([c[0] for c in coords], dtype=np.float64)
    y = np.array([c[1] for c in coords], dtype=np.float64)
    n = x.size
    if n < 2:
        return None

    n_in = min(samples, n)
    xc, yc, tc = _getcam(x, y, n_in, normalize=True)
    xs = np.linspace(xc[0], xc[-1], n_in, dtype=np.float64)
    slope = np.array([akima(xc, yc, float(xi))[1] for xi in xs], dtype=np.float64)
    camber = np.array([akima(xc, yc, float(xi))[0] for xi in xs], dtype=np.float64)
    thick = np.array([akima(xc, tc, float(xi))[0] for xi in xs], dtype=np.float64)
    nrmliz(xs)
    return AirfoilCamber(x=xs, s=slope, c=camber, t=thick)


def parse_body_coords(text: str) -> list[list[float]]:
    """Parse body cross-section coordinates from a .dat file."""
    coords: list[list[float]] = []
    for raw in text.splitlines():
        trimmed = raw.strip()
        if not trimmed or trimmed.startswith(("#", "!", "%")):
            continue
        parts = trimmed.split()
        if len(parts) >= 2:
            try:
                coords.append([float(parts[0]), float(parts[1])])
            except ValueError:
                continue
    if len(coords) >= 2:
        area = 0.0
        for i in range(len(coords) - 1):
            x0, y0 = coords[i]
            x1, y1 = coords[i + 1]
            area += 0.25 * ((x1 + x0) * (y1 - y0) - (y1 + y0) * (x1 - x0))
        if area < 0.0:
            coords.reverse()
    return coords


def build_body_thread(coords: list[list[float]], samples: int = 50) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Build body centerline and thickness tables via GETCAM (normalize=False)."""
    if not coords or len(coords) < 3:
        return None
    x = np.array([c[0] for c in coords], dtype=np.float64)
    y = np.array([c[1] for c in coords], dtype=np.float64)
    n_in = max(2, int(round(samples)))
    return _getcam(x, y, n_in, normalize=False)
