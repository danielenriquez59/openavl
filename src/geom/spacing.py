"""Panel spacing utilities (port of sgutil.f)."""

from __future__ import annotations

import math

import numpy as np


def akima(x: np.ndarray, y: np.ndarray, xx: float) -> tuple[float, float]:
    """Akima interpolation; returns (value, slope)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = x.size
    if n == 0:
        return 0.0, 0.0
    if x[0] == x[-1]:
        return float(y[0]), 0.0

    xordr = 1.0 if x[0] < x[-1] else -1.0
    xxo = xx * xordr

    ibot = 0
    itop = n - 1
    while itop - ibot > 1:
        i = ibot + (itop - ibot) // 2
        xo = x[i] * xordr
        if xxo >= xo:
            ibot = i
        else:
            itop = i

    i = ibot
    d = np.zeros(5, dtype=np.float64)
    for j in range(5):
        k = i + (j - 2)
        if 0 < k < n:
            d[j] = (y[k] - y[k - 1]) / (x[k] - x[k - 1])

    if n == 2:
        d[1] = d[2]
    if i + 2 >= n:
        d[3] = 2.0 * d[2] - d[1]
    if i + 3 >= n:
        d[4] = 2.0 * d[3] - d[2]
    if i - 1 < 0:
        d[1] = 2.0 * d[2] - d[3]
    if i - 2 < 0:
        d[0] = 2.0 * d[1] - d[2]

    t = np.zeros(2, dtype=np.float64)
    for j in range(2):
        a = abs(d[j + 2] - d[j + 1])
        b = abs(d[j] - d[j - 1])
        if a + b == 0.0:
            a = b = 1.0
        t[j] = (a * d[j + 1] + b * d[j + 2]) / (a + b)

    if xx == x[i]:
        return float(y[i]), float(t[1])

    xint = x[i] - x[i - 1]
    xdif = xx - x[i - 1]
    p0 = y[i - 1]
    p1 = t[0]
    p2 = (3.0 * d[2] - 2.0 * t[0] - t[1]) / xint
    p3 = (t[0] + t[1] - 2.0 * d[2]) / (xint * xint)

    yy = p0 + xdif * (p1 + xdif * (p2 + xdif * p3))
    slp = p1 + xdif * (2.0 * p2 + xdif * 3.0 * p3)
    return float(yy), float(slp)


def nrmliz(x: np.ndarray) -> np.ndarray:
    """Normalize abscissa array to [0, 1]."""
    out = np.asarray(x, dtype=np.float64).copy()
    n = out.size
    if n <= 1:
        return out
    dx = out[-1] - out[0]
    if dx == 0.0:
        dx = 1.0
    x0 = out[0]
    out[:] = (out - x0) / dx
    return out


def spacer(n: int, pspace: float) -> np.ndarray:
    """Cosine/sine/equal/blend panel spacing on [0, 1]."""
    x = np.zeros(n + 1, dtype=np.float64)
    if n <= 0:
        return x

    pi = math.pi
    pabs = abs(pspace)
    nabs = int(pabs) + 1
    if nabs == 1:
        pequ, pcos, psin = 1.0 - pabs, pabs, 0.0
    elif nabs == 2:
        pequ, pcos, psin = 0.0, 2.0 - pabs, pabs - 1.0
    else:
        pequ, pcos, psin = pabs - 2.0, 0.0, 3.0 - pabs

    for k in range(1, n + 1):
        frac = (k - 1) / (n - 1) if n > 1 else 0.0
        theta = frac * pi
        cos_theta = math.cos(theta)
        cos_half = math.cos(theta * 0.5)
        sin_half = math.sin(theta * 0.5)
        if pspace >= 0.0:
            x[k] = (
                pequ * frac
                + pcos * (1.0 - cos_theta) * 0.5
                + psin * (1.0 - cos_half)
            )
        else:
            x[k] = (
                pequ * frac
                + pcos * (1.0 - cos_theta) * 0.5
                + psin * sin_half
            )
    return x


def cspacer(
    nvc: int, cspace: float, claf: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Chordwise vortex/control/source/colocation spacing."""
    xpt = np.zeros(nvc + 2, dtype=np.float64)
    xvr = np.zeros(nvc + 1, dtype=np.float64)
    xsr = np.zeros(nvc + 1, dtype=np.float64)
    xcp = np.zeros(nvc + 1, dtype=np.float64)

    pi = math.pi
    acsp = abs(cspace)
    ncsp = int(acsp)
    if ncsp == 0:
        f0, f1, f2 = 1.0 - acsp, acsp, 0.0
    elif ncsp == 1:
        f0, f1, f2 = 0.0, 2.0 - acsp, acsp - 1.0
    else:
        f0, f1, f2 = acsp - 2.0, 0.0, 3.0 - acsp

    dth1 = pi / (4 * nvc + 2)
    dth2 = 0.5 * pi / (4 * nvc + 1)
    dxc0 = 1.0 / (4 * nvc)

    for ivc in range(1, nvc + 1):
        xc0 = int(4 * ivc - 4) * dxc0
        xpt0 = xc0
        xvr0 = xc0 + dxc0
        xsr0 = xc0 + 2.0 * dxc0
        xcp0 = xc0 + dxc0 + 2.0 * dxc0 * claf

        th1 = int(4 * ivc - 3) * dth1
        xpt1 = 0.5 * (1.0 - math.cos(th1))
        xvr1 = 0.5 * (1.0 - math.cos(th1 + dth1))
        xsr1 = 0.5 * (1.0 - math.cos(th1 + 2.0 * dth1))
        xcp1 = 0.5 * (1.0 - math.cos(th1 + dth1 + 2.0 * dth1 * claf))

        if cspace > 0.0:
            th2 = int(4 * ivc - 3) * dth2
            xpt2 = 1.0 - math.cos(th2)
            xvr2 = 1.0 - math.cos(th2 + dth2)
            xsr2 = 1.0 - math.cos(th2 + 2.0 * dth2)
            xcp2 = 1.0 - math.cos(th2 + dth2 + 2.0 * dth2 * claf)
        else:
            th2 = int(4 * ivc - 4) * dth2
            xpt2 = math.sin(th2)
            xvr2 = math.sin(th2 + dth2)
            xsr2 = math.sin(th2 + 2.0 * dth2)
            xcp2 = math.sin(th2 + dth2 + 2.0 * dth2 * claf)

        xpt[ivc] = f0 * xpt0 + f1 * xpt1 + f2 * xpt2
        xvr[ivc] = f0 * xvr0 + f1 * xvr1 + f2 * xvr2
        xsr[ivc] = f0 * xsr0 + f1 * xsr1 + f2 * xsr2
        xcp[ivc] = f0 * xcp0 + f1 * xcp1 + f2 * xcp2

    xpt[0] = 0.0
    xpt[nvc + 1] = 1.0
    return xpt, xvr, xsr, xcp
