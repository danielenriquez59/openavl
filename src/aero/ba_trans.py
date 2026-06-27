"""Body / stability / wind axis transformation matrices (port of ba_trans.f)."""

from __future__ import annotations

import numpy as np


def ba2wa_mat(
    alfa: float,
    beta: float,
    binv: float,
    p: np.ndarray | None = None,
    p_a: np.ndarray | None = None,
    p_b: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build body-to-wind axis transformation matrix and angle derivatives.

    Maps body-axis vectors to wind axes::

        [X_wa]       [   ] [X_ba]
        [Y_wa]   =   [ p ] [Y_ba]
        [Z_wa]       [   ] [Z_ba]
    """
    if p is None:
        p = np.zeros((3, 3), dtype=np.float64)
    if p_a is None:
        p_a = np.zeros((3, 3), dtype=np.float64)
    if p_b is None:
        p_b = np.zeros((3, 3), dtype=np.float64)

    sina = np.sin(alfa)
    cosa = np.cos(alfa)
    sinb = np.sin(beta)
    cosb = np.cos(beta)
    b = binv

    p[0, 0] = cosa * cosb * b
    p[0, 1] = -sinb * b
    p[0, 2] = sina * cosb * b

    p[1, 0] = cosa * sinb
    p[1, 1] = cosb
    p[1, 2] = sina * sinb

    p[2, 0] = -sina
    p[2, 1] = 0.0
    p[2, 2] = cosa

    p_a[0, 0] = -sina * cosb
    p_a[0, 1] = 0.0
    p_a[0, 2] = cosa * cosb

    p_a[1, 0] = -sina * sinb
    p_a[1, 1] = 0.0
    p_a[1, 2] = cosa * sinb

    p_a[2, 0] = -cosa
    p_a[2, 1] = 0.0
    p_a[2, 2] = -sina

    p_b[0, 0] = -cosa * sinb
    p_b[0, 1] = -cosb
    p_b[0, 2] = -sina * sinb

    p_b[1, 0] = cosa * cosb
    p_b[1, 1] = -sinb
    p_b[1, 2] = sina * cosb

    p_b[2, :] = 0.0

    return p, p_a, p_b


def ba2sa_mat(
    alfa: float,
    p: np.ndarray | None = None,
    p_a: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build body-to-stability axis transformation matrix and alpha derivatives.

    Maps body-axis vectors to stability axes::

        [X_sa]       [   ] [X_ba]
        [Y_sa]   =   [ p ] [Y_ba]
        [Z_sa]       [   ] [Z_ba]
    """
    if p is None:
        p = np.zeros((3, 3), dtype=np.float64)
    if p_a is None:
        p_a = np.zeros((3, 3), dtype=np.float64)

    sina = np.sin(alfa)
    cosa = np.cos(alfa)

    p[0, 0] = cosa
    p[0, 1] = 0.0
    p[0, 2] = sina

    p[1, 0] = 0.0
    p[1, 1] = 1.0
    p[1, 2] = 0.0

    p[2, 0] = -sina
    p[2, 1] = 0.0
    p[2, 2] = cosa

    p_a[0, 0] = -sina
    p_a[0, 1] = 0.0
    p_a[0, 2] = cosa

    p_a[1, :] = 0.0

    p_a[2, 0] = -cosa
    p_a[2, 1] = 0.0
    p_a[2, 2] = -sina

    return p, p_a
