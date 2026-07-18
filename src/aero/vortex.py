"""Horseshoe vortex induced velocity kernel (port of vorvelc.f).

Same as VORVEL, with finite core radius.

Original Scully (AKA Burnham-Hallock) core model::

    Vtan = Gam/(2*pi) * r/(r^2 + rcore^2)

Uses Leishman's R^4 variant of Scully (AKA Burnham-Hallock) core model::

    Vtan = Gam/(2*pi) * r/sqrt(r^4 + rcore^4)
"""

from __future__ import annotations
import math
import numpy as np

# Exact 1/(4*pi), matching aero/aic.py's PI4INV (B5: the previous truncated
# literal 0.079577472 carried ~1e-9 relative error relative to this kernel).
PI4INV = 1 / (4 * math.pi)

try:
    from numba import njit, prange

    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False
    prange = range


@njit(cache=True, parallel=True, fastmath=True) if _HAS_NUMBA else (lambda f: f)
def _vorvelc_mat_core(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    lbound: np.ndarray,
    x1: np.ndarray,
    y1: np.ndarray,
    z1: np.ndarray,
    x2: np.ndarray,
    y2: np.ndarray,
    z2: np.ndarray,
    beta: float,
    rcore: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flat-array horseshoe induced velocity (bound leg + semi-infinite legs)."""
    n = x.size
    u = np.zeros(n, dtype=np.float64)
    v = np.zeros(n, dtype=np.float64)
    w = np.zeros(n, dtype=np.float64)
    inv_bx = PI4INV / beta

    for i in prange(n):
        # Prandtl-Glauert coordinates
        ax1 = (x1[i] - x[i]) / beta
        ay1 = y1[i] - y[i]
        az1 = z1[i] - z[i]
        bx1 = (x2[i] - x[i]) / beta
        by1 = y2[i] - y[i]
        bz1 = z2[i] - z[i]

        asq = ax1 * ax1 + ay1 * ay1 + az1 * az1
        bsq = bx1 * bx1 + by1 * by1 + bz1 * bz1
        amag = np.sqrt(asq)
        bmag = np.sqrt(bsq)

        rcore2 = rcore[i] * rcore[i]
        rcore4 = rcore2 * rcore2

        ui = 0.0
        vi = 0.0
        wi = 0.0

        # contribution from the transverse bound leg
        if lbound[i] and amag * bmag != 0.0:
            axb1 = ay1 * bz1 - az1 * by1
            axb2 = az1 * bx1 - ax1 * bz1
            axb3 = ax1 * by1 - ay1 * bx1
            axbsq = axb1 * axb1 + axb2 * axb2 + axb3 * axb3
            if axbsq != 0.0:
                adb = ax1 * bx1 + ay1 * by1 + az1 * bz1
                alsq = asq + bsq - 2.0 * adb
                # Scully core model:
                # t = (amag + bmag) * (1.0 - adb / (amag * bmag)) / (axbsq + alsq * rcore2)
                # t = ((bsq - adb) / sqrt(bsq + rcore2) + (asq - adb) / sqrt(asq + rcore2))
                #      / (axbsq + alsq * rcore2)
                # Leishman core model
                t1 = bsq - adb
                t2 = asq - adb
                s1 = (bsq * bsq + rcore4) ** 0.25
                s2 = (asq * asq + rcore4) ** 0.25
                num = t1 / s1 + t2 / s2
                den = np.sqrt(axbsq * axbsq + alsq * alsq * rcore4)
                t = num / den
                ui = axb1 * t
                vi = axb2 * t
                wi = axb3 * t

        # trailing leg attached to A
        if amag != 0.0:
            axisq = az1 * az1 + ay1 * ay1
            # Scully core model: t = -(1.0 - ax1 / amag) / (axisq + rcore2)
            # Leishman core model
            t = -(1.0 - ax1 / amag) / np.sqrt(axisq * axisq + rcore4)
            vi += az1 * t
            wi -= ay1 * t

        # trailing leg attached to B
        if bmag != 0.0:
            bxisq = bz1 * bz1 + by1 * by1
            # Scully core model: t = (1.0 - bx1 / bmag) / (bxisq + rcore2)
            # Leishman core model
            t = (1.0 - bx1 / bmag) / np.sqrt(bxisq * bxisq + rcore4)
            vi += bz1 * t
            wi -= by1 * t

        u[i] = ui * inv_bx
        v[i] = vi * PI4INV
        w[i] = wi * PI4INV

    return u, v, w


def _vorvelc_mat_numpy(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    lbound: np.ndarray,
    x1: np.ndarray,
    y1: np.ndarray,
    z1: np.ndarray,
    x2: np.ndarray,
    y2: np.ndarray,
    z2: np.ndarray,
    beta: float,
    rcore: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pure-NumPy fallback for induced velocity (no optional numba dependency)."""
    bx = beta
    # Prandtl-Glauert coordinates
    ax1 = (x1 - x) / bx
    ay1 = y1 - y
    az1 = z1 - z

    bx1 = (x2 - x) / bx
    by1 = y2 - y
    bz1 = z2 - z

    asq = ax1 * ax1 + ay1 * ay1 + az1 * az1
    bsq = bx1 * bx1 + by1 * by1 + bz1 * bz1

    amag = np.sqrt(asq)
    bmag = np.sqrt(bsq)

    rcore2 = rcore * rcore
    rcore4 = rcore2 * rcore2

    u = np.zeros_like(x, dtype=np.float64)
    v = np.zeros_like(x, dtype=np.float64)
    w = np.zeros_like(x, dtype=np.float64)

    # contribution from the transverse bound leg
    bound_active = lbound & (amag * bmag != 0.0)
    axb1 = ay1 * bz1 - az1 * by1
    axb2 = az1 * bx1 - ax1 * bz1
    axb3 = ax1 * by1 - ay1 * bx1
    axbsq = axb1 * axb1 + axb2 * axb2 + axb3 * axb3

    bound_valid = bound_active & (axbsq != 0.0)
    adb = ax1 * bx1 + ay1 * by1 + az1 * bz1
    alsq = asq + bsq - 2.0 * adb
    # Scully core model:
    # t = (amag + bmag) * (1.0 - adb / (amag * bmag)) / (axbsq + alsq * rcore2)
    # t = ((bsq - adb) / sqrt(bsq + rcore2) + (asq - adb) / sqrt(asq + rcore2))
    #      / (axbsq + alsq * rcore2)
    # Leishman core model
    t1 = bsq - adb
    t2 = asq - adb
    s1 = np.sqrt(np.sqrt(bsq * bsq + rcore4))
    s2 = np.sqrt(np.sqrt(asq * asq + rcore4))
    num = t1 / s1 + t2 / s2
    den = np.sqrt(axbsq * axbsq + alsq * alsq * rcore4)
    t_bound = num / den

    u = np.where(bound_valid, axb1 * t_bound, u)
    v = np.where(bound_valid, axb2 * t_bound, v)
    w = np.where(bound_valid, axb3 * t_bound, w)

    # trailing leg attached to A
    leg_a_active = amag != 0.0
    axisq = ay1 * ay1 + az1 * az1
    # Scully core model: t = -(1.0 - ax1 / amag) / (axisq + rcore2)
    # Leishman core model
    t_a = -(1.0 - ax1 / amag) / np.sqrt(axisq * axisq + rcore4)
    v = np.where(leg_a_active, v + az1 * t_a, v)
    w = np.where(leg_a_active, w - ay1 * t_a, w)

    # trailing leg attached to B
    leg_b_active = bmag != 0.0
    bxisq = by1 * by1 + bz1 * bz1
    # Scully core model: t = (1.0 - bx1 / bmag) / (bxisq + rcore2)
    # Leishman core model
    t_b = (1.0 - bx1 / bmag) / np.sqrt(bxisq * bxisq + rcore4)
    v = np.where(leg_b_active, v + bz1 * t_b, v)
    w = np.where(leg_b_active, w - by1 * t_b, w)

    u = u * PI4INV / bx
    v = v * PI4INV
    w = w * PI4INV

    return u, v, w


def vorvelc(
    x: float,
    y: float,
    z: float,
    lbound: bool,
    x1: float,
    y1: float,
    z1: float,
    x2: float,
    y2: float,
    z2: float,
    beta: float,
    rcore: float,
) -> tuple[float, float, float]:
    """Compute induced velocity at (x,y,z) due to a horseshoe vortex segment."""
    u, v, w = vorvelc_mat(
        np.asarray(x, dtype=np.float64),
        np.asarray(y, dtype=np.float64),
        np.asarray(z, dtype=np.float64),
        np.asarray(lbound, dtype=bool),
        np.asarray(x1, dtype=np.float64),
        np.asarray(y1, dtype=np.float64),
        np.asarray(z1, dtype=np.float64),
        np.asarray(x2, dtype=np.float64),
        np.asarray(y2, dtype=np.float64),
        np.asarray(z2, dtype=np.float64),
        beta,
        np.asarray(rcore, dtype=np.float64),
    )
    return float(u), float(v), float(w)


def vorvelc_mat(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    lbound: np.ndarray,
    x1: np.ndarray,
    y1: np.ndarray,
    z1: np.ndarray,
    x2: np.ndarray,
    y2: np.ndarray,
    z2: np.ndarray,
    beta: float,
    rcore: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized induced velocity for many field/vortex pairs (broadcastable arrays)."""
    shape = np.broadcast_shapes(
        x.shape, y.shape, z.shape, lbound.shape,
        x1.shape, y1.shape, z1.shape, x2.shape, y2.shape, z2.shape, rcore.shape,
    )
    xb = np.broadcast_to(x, shape)
    yb = np.broadcast_to(y, shape)
    zb = np.broadcast_to(z, shape)
    lbb = np.broadcast_to(lbound, shape)
    x1b = np.broadcast_to(x1, shape)
    y1b = np.broadcast_to(y1, shape)
    z1b = np.broadcast_to(z1, shape)
    x2b = np.broadcast_to(x2, shape)
    y2b = np.broadcast_to(y2, shape)
    z2b = np.broadcast_to(z2, shape)
    rb = np.broadcast_to(rcore, shape)

    if _HAS_NUMBA:
        u, v, w = _vorvelc_mat_core(
            xb.ravel(), yb.ravel(), zb.ravel(), lbb.ravel(),
            x1b.ravel(), y1b.ravel(), z1b.ravel(),
            x2b.ravel(), y2b.ravel(), z2b.ravel(),
            beta, rb.ravel(),
        )
        return u.reshape(shape), v.reshape(shape), w.reshape(shape)

    return _vorvelc_mat_numpy(
        xb, yb, zb, lbb,
        x1b, y1b, z1b, x2b, y2b, z2b,
        beta, rb,
    )
