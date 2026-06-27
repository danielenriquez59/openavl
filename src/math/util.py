"""Utility linear algebra and rotation helpers (port of autil.f)."""

from __future__ import annotations

import numpy as np


def cross3(u: np.ndarray, v: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
    """Cross product of two 3-vectors."""
    result = np.cross(u, v)
    if w is None:
        return result
    w[:] = result
    return w


def dot3(u: np.ndarray, v: np.ndarray) -> np.float64:
    """Dot product of two 3-vectors."""
    return np.dot(u, v)


def m3inv(a: np.ndarray, ainv: np.ndarray | None = None) -> np.ndarray:
    """Invert a 3x3 matrix with zero-diagonal treated as infinity (M3INV)."""
    if ainv is None:
        ainv = np.zeros((3, 3), dtype=np.float64)
    t = np.zeros((3, 6), dtype=np.float64)

    for k in range(3):
        t[k, 0] = a[k, 0]
        t[k, 1] = a[k, 1]
        t[k, 2] = a[k, 2]
        t[k, 3] = 0.0
        t[k, 4] = 0.0
        t[k, 5] = 0.0

    t[0, 3] = 1.0
    t[1, 4] = 1.0
    t[2, 5] = 1.0

    for n in range(3):
        pivot = t[n, n]
        if pivot == 0.0:
            for col in range(n + 1, 6):
                t[n, col] = 0.0
        else:
            for col in range(n + 1, 6):
                t[n, col] = t[n, col] / pivot

        for row in range(n + 1, 3):
            tel = t[row, n]
            for col in range(n + 1, 6):
                t[row, col] = t[row, col] - tel * t[n, col]

    for n in range(2, 0, -1):
        for row in range(n - 1, -1, -1):
            tel = t[row, n]
            for col in range(3, 6):
                t[row, col] = t[row, col] - tel * t[n, col]

    for k in range(3):
        ainv[k, 0] = t[k, 3]
        ainv[k, 1] = t[k, 4]
        ainv[k, 2] = t[k, 5]

    return ainv


def rateki3(
    a: np.ndarray,
    r: np.ndarray | None = None,
    r_a: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Inverse rotation-rate tensor mapping body rates to Euler-angle rates (RATEKI3)."""
    if r is None:
        r = np.zeros((3, 3), dtype=np.float64)
    if r_a is None:
        r_a = np.zeros((3, 3, 3), dtype=np.float64)

    c1 = np.cos(a[0])
    c2 = np.cos(a[1])
    s1 = np.sin(a[0])
    t2 = np.tan(a[1])

    r[0, 0] = -1.0
    r[1, 0] = 0.0
    r[2, 0] = 0.0

    r[0, 1] = s1 * t2
    r[1, 1] = c1
    r[2, 1] = s1 / c2

    r[0, 2] = -c1 * t2
    r[1, 2] = s1
    r[2, 2] = -c1 / c2

    r_a[0, 0, 0] = 0.0
    r_a[1, 0, 0] = 0.0
    r_a[2, 0, 0] = 0.0

    r_a[0, 1, 0] = c1 * t2
    r_a[1, 1, 0] = -s1
    r_a[2, 1, 0] = c1 / c2

    r_a[0, 2, 0] = s1 * t2
    r_a[1, 2, 0] = c1
    r_a[2, 2, 0] = s1 / c2

    r_a[0, 0, 1] = 0.0
    r_a[1, 0, 1] = 0.0
    r_a[2, 0, 1] = 0.0

    r_a[0, 1, 1] = s1 / (c2 * c2)
    r_a[1, 1, 1] = 0.0
    r_a[2, 1, 1] = (s1 * t2) / c2

    r_a[0, 2, 1] = -c1 / (c2 * c2)
    r_a[1, 2, 1] = 0.0
    r_a[2, 2, 1] = (-c1 * t2) / c2

    r_a[:, :, 2] = 0.0

    return r, r_a


def rotens3(
    a: np.ndarray,
    t: np.ndarray | None = None,
    t_a: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Net rotation tensor from Euler angles phi, theta, psi (ROTENS3)."""
    if t is None:
        t = np.zeros((3, 3), dtype=np.float64)
    if t_a is None:
        t_a = np.zeros((3, 3, 3), dtype=np.float64)

    c1 = np.cos(a[0])
    c2 = np.cos(a[1])
    c3 = np.cos(a[2])
    s1 = np.sin(a[0])
    s2 = np.sin(a[1])
    s3 = np.sin(a[2])

    t[0, 0] = c2 * c3
    t[1, 0] = -c2 * s3
    t[2, 0] = -s2

    t[0, 1] = -s1 * s2 * c3 + c1 * s3
    t[1, 1] = s1 * s2 * s3 + c1 * c3
    t[2, 1] = -s1 * c2

    t[0, 2] = c1 * s2 * c3 + s1 * s3
    t[1, 2] = -c1 * s2 * s3 + s1 * c3
    t[2, 2] = c1 * c2

    t_a[0, 0, 0] = 0.0
    t_a[1, 0, 0] = 0.0
    t_a[2, 0, 0] = 0.0

    t_a[0, 1, 0] = -c1 * s2 * c3 - s1 * s3
    t_a[1, 1, 0] = c1 * s2 * s3 - s1 * c3
    t_a[2, 1, 0] = -c1 * c2

    t_a[0, 2, 0] = -s1 * s2 * c3 + c1 * s3
    t_a[1, 2, 0] = s1 * s2 * s3 + c1 * c3
    t_a[2, 2, 0] = -s1 * c2

    t_a[0, 0, 1] = -s2 * c3
    t_a[1, 0, 1] = s2 * s3
    t_a[2, 0, 1] = -c2

    t_a[0, 1, 1] = -s1 * c2 * c3
    t_a[1, 1, 1] = s1 * c2 * s3
    t_a[2, 1, 1] = s1 * s2

    t_a[0, 2, 1] = c1 * c2 * c3
    t_a[1, 2, 1] = -c1 * c2 * s3
    t_a[2, 2, 1] = -c1 * s2

    t_a[0, 0, 2] = -c2 * s3
    t_a[1, 0, 2] = -c2 * c3
    t_a[2, 0, 2] = 0.0

    t_a[0, 1, 2] = s1 * s2 * s3 + c1 * c3
    t_a[1, 1, 2] = s1 * s2 * c3 - c1 * s3
    t_a[2, 1, 2] = 0.0

    t_a[0, 2, 2] = -c1 * s2 * s3 + s1 * c3
    t_a[1, 2, 2] = -c1 * s2 * c3 - s1 * s3
    t_a[2, 2, 2] = 0.0

    return t, t_a
