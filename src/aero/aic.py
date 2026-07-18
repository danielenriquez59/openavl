"""AIC matrix assembly (port of aic.f).

Builds aerodynamic influence coefficient matrices: horseshoe-vortex induced
velocities (VVOR), body source/doublet line influences (VSRD), and body
strength sensitivities for unit freestream/rotation components (SRDSET).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import math
from openavl.aero.vortex import vorvelc, vorvelc_mat

PI4INV = 1/(4*math.pi)


@dataclass
class _VvorVortexPre:
    """Cached vortex-side geometry shared between repeated VVOR builds."""

    nv: int
    nc_eq_nv: bool
    valid_v: np.ndarray
    rcore_ij: np.ndarray
    x1_r: np.ndarray
    y1_r: np.ndarray
    z1_r: np.ndarray
    x2_r: np.ndarray
    y2_r: np.ndarray
    z2_r: np.ndarray
    x1_y: np.ndarray | None
    y1_y: np.ndarray | None
    z1_y: np.ndarray | None
    x2_y: np.ndarray | None
    y2_y: np.ndarray | None
    z2_y: np.ndarray | None
    x1_z: np.ndarray | None
    y1_z: np.ndarray | None
    z1_z: np.ndarray | None
    x2_z: np.ndarray | None
    y2_z: np.ndarray | None
    z2_z: np.ndarray | None
    x1_yz: np.ndarray | None
    y1_yz: np.ndarray | None
    z1_yz: np.ndarray | None
    x2_yz: np.ndarray | None
    y2_yz: np.ndarray | None
    z2_yz: np.ndarray | None
    xave: np.ndarray | None
    yave: np.ndarray | None
    zave: np.ndarray | None


def _vvor_vortex_pre(
    betm: float,
    iysym: int,
    ysym: float,
    izsym: int,
    zsym: float,
    vrcorec: float,
    vrcorew: float,
    nv: int,
    rv1: np.ndarray,
    rv2: np.ndarray,
    ncompv: np.ndarray,
    chordv: np.ndarray,
    nc: int,
    ncompc: np.ndarray,
) -> _VvorVortexPre:
    """Precompute vortex geometry and core radii reused across VVOR calls."""
    ds_y = rv2[1, :] - rv1[1, :]
    ds_z = rv2[2, :] - rv1[2, :]
    dsyz = np.sqrt(ds_y * ds_y + ds_z * ds_z)
    valid_v = np.isfinite(dsyz) & (dsyz != 0.0)

    # default (non-zero) core size based on spanwise lattice spacing
    rcore_default = 0.0001 * dsyz
    if nc == nv:
        # if field point is not on same component use larger core size
        rcore_cross = np.maximum(vrcorec * chordv, vrcorew * dsyz)
        cross_comp = ncompc[:, np.newaxis] != ncompv[np.newaxis, :]
        rcore_ij = np.where(cross_comp, rcore_cross[np.newaxis, :], rcore_default[np.newaxis, :])
    else:
        rcore_ij = np.broadcast_to(rcore_default[np.newaxis, :], (nc, nv))

    x1_r = rv1[0, np.newaxis, :]
    y1_r = rv1[1, np.newaxis, :]
    z1_r = rv1[2, np.newaxis, :]
    x2_r = rv2[0, np.newaxis, :]
    y2_r = rv2[1, np.newaxis, :]
    z2_r = rv2[2, np.newaxis, :]

    yoff = 2.0 * ysym
    zoff = 2.0 * zsym

    x1_y = y1_y = z1_y = x2_y = y2_y = z2_y = None
    x1_z = y1_z = z1_z = x2_z = y2_z = z2_z = None
    x1_yz = y1_yz = z1_yz = x2_yz = y2_yz = z2_yz = None
    xave = yave = zave = None

    if iysym != 0:
        x1_y = x2_r
        y1_y = yoff - y2_r
        z1_y = z2_r
        x2_y = x1_r
        y2_y = yoff - y1_r
        z2_y = z1_r
        if iysym == 1:
            xave = 0.5 * (rv1[0, :] + rv2[0, :])[np.newaxis, :]
            yave = (yoff - 0.5 * (rv1[1, :] + rv2[1, :]))[np.newaxis, :]
            zave = 0.5 * (rv1[2, :] + rv2[2, :])[np.newaxis, :]

    if izsym != 0:
        x1_z = x2_r
        y1_z = y2_r
        z1_z = zoff - z2_r
        x2_z = x1_r
        y2_z = y1_r
        z2_z = zoff - z1_r

        if iysym != 0:
            x1_yz = x1_r
            y1_yz = yoff - y1_r
            z1_yz = zoff - z1_r
            x2_yz = x2_r
            y2_yz = yoff - y2_r
            z2_yz = zoff - z2_r

    return _VvorVortexPre(
        nv=nv,
        nc_eq_nv=(nc == nv),
        valid_v=valid_v,
        rcore_ij=rcore_ij,
        x1_r=x1_r, y1_r=y1_r, z1_r=z1_r, x2_r=x2_r, y2_r=y2_r, z2_r=z2_r,
        x1_y=x1_y, y1_y=y1_y, z1_y=z1_y, x2_y=x2_y, y2_y=y2_y, z2_y=z2_y,
        x1_z=x1_z, y1_z=y1_z, z1_z=z1_z, x2_z=x2_z, y2_z=y2_z, z2_z=z2_z,
        x1_yz=x1_yz, y1_yz=y1_yz, z1_yz=z1_yz, x2_yz=x2_yz, y2_yz=y2_yz, z2_yz=z2_yz,
        xave=xave, yave=yave, zave=zave,
    )


def _vvor_eval(
    pre: _VvorVortexPre,
    betm: float,
    iysym: int,
    izsym: int,
    nc: int,
    rc: np.ndarray,
    lvtest: bool,
    wc_gam: np.ndarray,
    ncdim: int,
) -> np.ndarray:
    """Evaluate VVOR using precomputed vortex geometry."""
    fysym = float(iysym)
    fzsym = float(izsym)
    nv = pre.nv

    x_f = rc[0, :, np.newaxis]
    y_f = rc[1, :, np.newaxis]
    z_f = rc[2, :, np.newaxis]

    ii = np.arange(nc)[:, np.newaxis]
    jj = np.arange(nv)[np.newaxis, :]
    lbound_real = ~(lvtest & (ii == jj))

    img_x1: list[np.ndarray] = [pre.x1_r]
    img_y1: list[np.ndarray] = [pre.y1_r]
    img_z1: list[np.ndarray] = [pre.z1_r]
    img_x2: list[np.ndarray] = [pre.x2_r]
    img_y2: list[np.ndarray] = [pre.y2_r]
    img_z2: list[np.ndarray] = [pre.z2_r]
    img_lbound: list[np.ndarray] = [lbound_real]
    img_to_u: list[bool] = [False]
    img_scale: list[float] = [1.0]

    if izsym != 0:
        # Calculate the influence of the z-IMAGE vortex
        img_x1.append(pre.x1_z)
        img_y1.append(pre.y1_z)
        img_z1.append(pre.z1_z)
        img_x2.append(pre.x2_z)
        img_y2.append(pre.y2_z)
        img_z2.append(pre.z2_z)
        img_lbound.append(np.ones((nc, nv), dtype=bool))
        img_to_u.append(False)
        img_scale.append(fzsym)

    if iysym != 0:
        # Calculate the influence of the y-IMAGE vortex
        lbound_y = np.ones((nc, nv), dtype=bool)
        if iysym == 1 and pre.xave is not None:
            at_mid = (x_f == pre.xave) & (y_f == pre.yave) & (z_f == pre.zave)
            lbound_y = ~at_mid
        img_x1.append(pre.x1_y)
        img_y1.append(pre.y1_y)
        img_z1.append(pre.z1_y)
        img_x2.append(pre.x2_y)
        img_y2.append(pre.y2_y)
        img_z2.append(pre.z2_y)
        img_lbound.append(lbound_y)
        img_to_u.append(True)
        img_scale.append(fysym)

        if izsym != 0:
            # Calculate the influence of the y,z-IMAGE vortex
            img_x1.append(pre.x1_yz)
            img_y1.append(pre.y1_yz)
            img_z1.append(pre.z1_yz)
            img_x2.append(pre.x2_yz)
            img_y2.append(pre.y2_yz)
            img_z2.append(pre.z2_yz)
            img_lbound.append(np.ones((nc, nv), dtype=bool))
            img_to_u.append(True)
            img_scale.append(fysym * fzsym)

    nimg = len(img_x1)
    if nimg == 1:
        # Calculate the influence of the REAL vortex
        u, v, w = vorvelc_mat(
            x_f, y_f, z_f, img_lbound[0],
            img_x1[0], img_y1[0], img_z1[0], img_x2[0], img_y2[0], img_z2[0],
            betm, pre.rcore_ij,
        )
        ui = vi = wi = np.zeros((nc, nv), dtype=np.float64)
    else:
        u = np.zeros((nc, nv), dtype=np.float64)
        v = np.zeros((nc, nv), dtype=np.float64)
        w = np.zeros((nc, nv), dtype=np.float64)
        ui = np.zeros((nc, nv), dtype=np.float64)
        vi = np.zeros((nc, nv), dtype=np.float64)
        wi = np.zeros((nc, nv), dtype=np.float64)
        for k in range(nimg):
            u_k, v_k, w_k = vorvelc_mat(
                x_f, y_f, z_f, img_lbound[k],
                img_x1[k], img_y1[k], img_z1[k],
                img_x2[k], img_y2[k], img_z2[k],
                betm, pre.rcore_ij,
            )
            scale = img_scale[k]
            if img_to_u[k]:
                ui += u_k * scale
                vi += v_k * scale
                wi += w_k * scale
            else:
                u += u_k * scale
                v += v_k * scale
                w += w_k * scale

    us = u + ui
    vs = v + vi
    ws = w + wi

    invalid = ~pre.valid_v[np.newaxis, :]
    us = np.where(invalid, 0.0, us)
    vs = np.where(invalid, 0.0, vs)
    ws = np.where(invalid, 0.0, ws)
    us = np.where(np.isfinite(us), us, 0.0)
    vs = np.where(np.isfinite(vs), vs, 0.0)
    ws = np.where(np.isfinite(ws), ws, 0.0)

    wc_gam[0, :nc, :nv] = us
    wc_gam[1, :nc, :nv] = vs
    wc_gam[2, :nc, :nv] = ws
    return wc_gam


def cross(u: np.ndarray, v: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
    """Cross product of two 3-vectors with optional output buffer."""
    if w is None:
        return np.cross(u, v)
    # Scalar form avoids np.cross allocation in hot aero/setup loops.
    w[0] = u[1] * v[2] - u[2] * v[1]
    w[1] = u[2] * v[0] - u[0] * v[2]
    w[2] = u[0] * v[1] - u[1] * v[0]
    return w


def dot(u: np.ndarray, v: np.ndarray) -> float:
    """Dot product of two 3-vectors."""
    return float(u[0] * v[0] + u[1] * v[1] + u[2] * v[2])


def srdvelc_batch(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    x1: np.ndarray,
    y1: np.ndarray,
    z1: np.ndarray,
    x2: np.ndarray,
    y2: np.ndarray,
    z2: np.ndarray,
    beta: float,
    rcore: np.ndarray | float,
    uvws: np.ndarray | None = None,
    uvwd: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Batched source/doublet induced velocity kernel (SRDVELC).

    Evaluates the SRDVELC kernel for ``N`` field-point / segment pairs in one
    vectorized pass.  Inputs are 1-D arrays of equal length ``N``; outputs are
    ``uvws (3, N)`` and ``uvwd (3, 3, N)``.
    """
    n = int(np.asarray(x).size)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    x1 = np.asarray(x1, dtype=np.float64)
    y1 = np.asarray(y1, dtype=np.float64)
    z1 = np.asarray(z1, dtype=np.float64)
    x2 = np.asarray(x2, dtype=np.float64)
    y2 = np.asarray(y2, dtype=np.float64)
    z2 = np.asarray(z2, dtype=np.float64)
    rcore = np.asarray(rcore, dtype=np.float64)

    r10 = (x1 - x) / beta
    r11 = y1 - y
    r12 = z1 - z
    r20 = (x2 - x) / beta
    r21 = y2 - y
    r22 = z2 - z

    rcsq = rcore * rcore

    r1sq = r10 * r10 + r11 * r11 + r12 * r12
    r2sq = r20 * r20 + r21 * r21 + r22 * r22

    r1sqeps = r1sq + rcsq
    r2sqeps = r2sq + rcsq

    r1eps = np.sqrt(r1sqeps)
    r2eps = np.sqrt(r2sqeps)

    rdr = r10 * r20 + r11 * r21 + r12 * r22
    rxr0 = r11 * r22 - r12 * r21
    rxr1 = r12 * r20 - r10 * r22
    rxr2 = r10 * r21 - r11 * r20

    xdx = rxr0 * rxr0 + rxr1 * rxr1 + rxr2 * rxr2
    all_ = r1sq + r2sq - 2.0 * rdr
    den = rcsq * all_ + xdx

    ai1 = ((rdr + rcsq) / r1eps - r2eps) / den
    ai2 = ((rdr + rcsq) / r2eps - r1eps) / den

    if uvws is None:
        uvws = np.zeros((3, n), dtype=np.float64)
    if uvwd is None:
        uvwd = np.zeros((3, 3, n), dtype=np.float64)

    r1 = (r10, r11, r12)
    r2 = (r20, r21, r22)

    for k in range(3):
        r1k = r1[k]
        r2k = r2[k]

        # set velocity components for unit source and doublet
        uvws[k] = (r1k * ai1) + (r2k * ai2)

        rr1 = (
            ((r1k + r2k) / r1eps)
            - ((r1k * (rdr + rcsq)) / (r1eps * r1eps * r1eps))
            - (r2k / r2eps)
        )

        rr2 = (
            ((r1k + r2k) / r2eps)
            - ((r2k * (rdr + rcsq)) / (r2eps * r2eps * r2eps))
            - (r1k / r1eps)
        )

        rrt = (2.0 * r1k * (r2sq - rdr)) + (2.0 * r2k * (r1sq - rdr))

        aj1 = (rr1 - (ai1 * rrt)) / den
        aj2 = (rr2 - (ai2 * rrt)) / den

        for j in range(3):
            uvwd[k, j] = (-(aj1 * r1[j]) - (aj2 * r2[j]))

        uvwd[k, k] = uvwd[k, k] - ai1 - ai2

    uvws[0] = (uvws[0] * PI4INV) / beta
    uvws[1] = uvws[1] * PI4INV
    uvws[2] = uvws[2] * PI4INV

    uvwd[0, :] = (uvwd[0, :] * PI4INV) / beta
    uvwd[1, :] = uvwd[1, :] * PI4INV
    uvwd[2, :] = uvwd[2, :] * PI4INV

    return uvws, uvwd


def srdvelc(
    x: float,
    y: float,
    z: float,
    x1: float,
    y1: float,
    z1: float,
    x2: float,
    y2: float,
    z2: float,
    beta: float,
    rcore: float,
    uvws: np.ndarray | None = None,
    uvwd: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Source/doublet panel induced velocity and velocity derivative matrix (SRDVELC).

    Same as SRDVEL, but with finite core radius. Returns UVWS (velocity from
    unit source + unit doublet) and UVWD (3×3 velocity derivative matrix w.r.t.
    doublet orientation components).
    """
    if uvws is None:
        uvws = np.zeros(3, dtype=np.float64)
    if uvwd is None:
        uvwd = np.zeros((3, 3), dtype=np.float64)

    r10 = (x1 - x) / beta
    r11 = y1 - y
    r12 = z1 - z
    r20 = (x2 - x) / beta
    r21 = y2 - y
    r22 = z2 - z

    rcsq = rcore * rcore

    r1sq = r10 * r10 + r11 * r11 + r12 * r12
    r2sq = r20 * r20 + r21 * r21 + r22 * r22

    r1sqeps = r1sq + rcsq
    r2sqeps = r2sq + rcsq

    r1eps = math.sqrt(r1sqeps)
    r2eps = math.sqrt(r2sqeps)

    rdr = r10 * r20 + r11 * r21 + r12 * r22
    rxr0 = r11 * r22 - r12 * r21
    rxr1 = r12 * r20 - r10 * r22
    rxr2 = r10 * r21 - r11 * r20

    xdx = rxr0 * rxr0 + rxr1 * rxr1 + rxr2 * rxr2
    all_ = r1sq + r2sq - 2.0 * rdr
    den = rcsq * all_ + xdx

    ai1 = ((rdr + rcsq) / r1eps - r2eps) / den
    ai2 = ((rdr + rcsq) / r2eps - r1eps) / den

    r1 = (r10, r11, r12)
    r2 = (r20, r21, r22)

    for k in range(3):
        r1k = r1[k]
        r2k = r2[k]

        # set velocity components for unit source and doublet
        uvws[k] = (r1k * ai1) + (r2k * ai2)

        rr1 = (
            ((r1k + r2k) / r1eps)
            - ((r1k * (rdr + rcsq)) / (r1eps * r1eps * r1eps))
            - (r2k / r2eps)
        )

        rr2 = (
            ((r1k + r2k) / r2eps)
            - ((r2k * (rdr + rcsq)) / (r2eps * r2eps * r2eps))
            - (r1k / r1eps)
        )

        rrt = (2.0 * r1k * (r2sq - rdr)) + (2.0 * r2k * (r1sq - rdr))

        aj1 = (rr1 - (ai1 * rrt)) / den
        aj2 = (rr2 - (ai2 * rrt)) / den

        for j in range(3):
            uvwd[k, j] = (-(aj1 * r1[j]) - (aj2 * r2[j]))

        uvwd[k, k] = uvwd[k, k] - ai1 - ai2

    uvws[0] = (uvws[0] * PI4INV) / beta
    uvws[1] = uvws[1] * PI4INV
    uvws[2] = uvws[2] * PI4INV

    for l in range(3):
        uvwd[0, l] = (uvwd[0, l] * PI4INV) / beta
        uvwd[1, l] = uvwd[1, l] * PI4INV
        uvwd[2, l] = uvwd[2, l] * PI4INV

    return uvws, uvwd


def vvor(
    betm: float,
    iysym: int,
    ysym: float,
    izsym: int,
    zsym: float,
    vrcorec: float,
    vrcorew: float,
    nv: int,
    rv1: np.ndarray,
    rv2: np.ndarray,
    ncompv: np.ndarray,
    chordv: np.ndarray,
    nc: int,
    rc: np.ndarray,
    ncompc: np.ndarray,
    lvtest: bool,
    wc_gam: np.ndarray | None = None,
    ncdim: int | None = None,
    vortex_pre: _VvorVortexPre | None = None,
) -> np.ndarray:
    """Assemble velocity influence coefficient matrix for vortex horseshoes (VVOR).

    Calculates the velocity influence matrix for a collection of horseshoe
    vortices and control points. WC_GAM(i,j) is the induced velocity at
    control point i per unit circulation on horseshoe vortex j, including
    symmetry-plane image contributions (real, y-image, z-image, y,z-image).

    Optional ``vortex_pre`` reuses vortex-side geometry from a prior call.
    """
    if ncdim is None:
        ncdim = nc

    if wc_gam is None:
        wc_gam = np.zeros((3, ncdim, nv), dtype=np.float64)

    if vortex_pre is None:
        vortex_pre = _vvor_vortex_pre(
            betm, iysym, ysym, izsym, zsym,
            vrcorec, vrcorew, nv, rv1, rv2, ncompv, chordv, nc, ncompc,
        )

    return _vvor_eval(
        vortex_pre, betm, iysym, izsym, nc, rc, lvtest, wc_gam, ncdim,
    )


def _vsrd_segment_contrib(
    scale: float,
    dbl_sign: np.ndarray,
    uvws: np.ndarray,
    uvwd: np.ndarray,
    src_u_l: np.ndarray,
    dbl_u_l: np.ndarray,
) -> np.ndarray:
    """Accumulate one symmetry image contribution into WC_U slice (3, nc, nu)."""
    dbl_u_eff = dbl_u_l * dbl_sign[:, np.newaxis]
    return scale * (
        uvws[:, :, np.newaxis] * src_u_l[np.newaxis, np.newaxis, :]
        + np.einsum("kji,ju->kiu", uvwd, dbl_u_eff)
    )


def vsrd(
    betm: float,
    iysym: int,
    ysym: float,
    izsym: int,
    zsym: float,
    srcore: float,
    nbody: int,
    lfrst: np.ndarray,
    nldim: int,
    nl: np.ndarray,
    rl: np.ndarray,
    radl: np.ndarray,
    nu: int,
    src_u: np.ndarray,
    dbl_u: np.ndarray,
    nc: int,
    rc: np.ndarray,
    wc_u: np.ndarray | None = None,
    ncdim: int | None = None,
) -> np.ndarray:
    """Assemble body source/doublet velocity influence matrix (VSRD).

    Calculates the velocity influence matrix for a collection of
    source+doublet lines. WC_U(i,iu) is velocity at control point i per unit
    apparent-freestream component iu, summed over all body segments and images.
    """
    fysym = float(iysym)
    fzsym = float(izsym)
    yoff = 2.0 * ysym
    zoff = 2.0 * zsym
    if ncdim is None:
        ncdim = nc

    if wc_u is None:
        wc_u = np.zeros((3, ncdim, nu), dtype=np.float64)
    else:
        wc_u.fill(0.0)

    x_f = rc[0, :nc]
    y_f = rc[1, :nc]
    z_f = rc[2, :nc]
    ones_nc = np.ones(nc, dtype=np.float64)

    for ib in range(nbody):
        for ilseg in range(nl[ib] - 1):
            l1 = int(lfrst[ib] + ilseg)
            l2 = l1 + 1
            l = l1

            ravg = np.sqrt(0.5 * ((radl[l2] * radl[l2]) + (radl[l1] * radl[l1])))
            dx = rl[0, l2] - rl[0, l1]
            dy = rl[1, l2] - rl[1, l1]
            dz = rl[2, l2] - rl[2, l1]
            rlavg = np.sqrt(dx * dx + dy * dy + dz * dz)

            if srcore > 0:
                rcore = srcore * ravg
            else:
                rcore = srcore * rlavg

            images: list[tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = [
                # influence of real segment
                (
                    1.0,
                    np.array([1.0, 1.0, 1.0]),
                    rl[0, l1] * ones_nc,
                    rl[1, l1] * ones_nc,
                    rl[2, l1] * ones_nc,
                    rl[0, l2] * ones_nc,
                    rl[1, l2] * ones_nc,
                    rl[2, l2] * ones_nc,
                ),
            ]
            if iysym != 0:
                # influence of y-image
                images.append((
                    fysym,
                    np.array([1.0, -1.0, 1.0]),
                    rl[0, l1] * ones_nc,
                    (yoff - rl[1, l1]) * ones_nc,
                    rl[2, l1] * ones_nc,
                    rl[0, l2] * ones_nc,
                    (yoff - rl[1, l2]) * ones_nc,
                    rl[2, l2] * ones_nc,
                ))
            if izsym != 0:
                # influence of z-image
                images.append((
                    fzsym,
                    np.array([1.0, 1.0, -1.0]),
                    rl[0, l1] * ones_nc,
                    rl[1, l1] * ones_nc,
                    (zoff - rl[2, l1]) * ones_nc,
                    rl[0, l2] * ones_nc,
                    rl[1, l2] * ones_nc,
                    (zoff - rl[2, l2]) * ones_nc,
                ))
                if iysym != 0:
                    # influence of y,z-image
                    images.append((
                        fysym * fzsym,
                        np.array([1.0, -1.0, -1.0]),
                        rl[0, l1] * ones_nc,
                        (yoff - rl[1, l1]) * ones_nc,
                        (zoff - rl[2, l1]) * ones_nc,
                        rl[0, l2] * ones_nc,
                        (yoff - rl[1, l2]) * ones_nc,
                        (zoff - rl[2, l2]) * ones_nc,
                    ))

            nimg = len(images)
            x_cat = np.tile(x_f, nimg)
            y_cat = np.tile(y_f, nimg)
            z_cat = np.tile(z_f, nimg)
            x1_cat = np.concatenate([img[2] for img in images])
            y1_cat = np.concatenate([img[3] for img in images])
            z1_cat = np.concatenate([img[4] for img in images])
            x2_cat = np.concatenate([img[5] for img in images])
            y2_cat = np.concatenate([img[6] for img in images])
            z2_cat = np.concatenate([img[7] for img in images])

            uvws_all, uvwd_all = srdvelc_batch(
                x_cat, y_cat, z_cat,
                x1_cat, y1_cat, z1_cat,
                x2_cat, y2_cat, z2_cat,
                betm, rcore,
            )

            src_u_l = src_u[l, :nu]
            dbl_u_l = dbl_u[:, l, :nu]
            for im, (scale, dbl_sign, *_rest) in enumerate(images):
                sl = slice(im * nc, (im + 1) * nc)
                wc_u[:, :nc, :] += _vsrd_segment_contrib(
                    scale,
                    dbl_sign,
                    uvws_all[:, sl],
                    uvwd_all[:, :, sl],
                    src_u_l,
                    dbl_u_l,
                )

    return wc_u


def srdset(
    betm: float,
    xyzref: np.ndarray,
    iysym: int,
    nbody: int,
    lfrst: np.ndarray,
    nldim: int,
    nl: np.ndarray,
    rl: np.ndarray,
    radl: np.ndarray,
    src_u: np.ndarray | None = None,
    dbl_u: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Initialize body source and doublet strength sensitivities (SRDSET).

    Sets strengths of source+doublet line segments for six unit flow components:
    three unit (X,Y,Z) freestream and three unit (X,Y,Z) rotations.
    """
    pi = math.pi
    beta = betm

    if src_u is None:
        src_u = np.zeros((nldim, 6), dtype=np.float64)
    if dbl_u is None:
        dbl_u = np.zeros((3, nldim, 6), dtype=np.float64)

    for ib in range(nbody):
        l1b = int(lfrst[ib])
        l2b = l1b + nl[ib] - 1
        # Use the body's actual last node. Fortran SRDSET reads RL(1,L1+NL),
        # one past the last node (out-of-bounds); that only affects the
        # SDFAC=0.5 on-symmetry-plane test in pathological cases.
        blen = np.abs(rl[0, l2b] - rl[0, l1b])
        sdfac = 1.0
        if iysym == 1 and rl[1, l1b] <= 0.001 * blen:
            # body y-image will be added on, so use only half the area
            sdfac = 0.5

        nseg = int(nl[ib]) - 1
        if nseg <= 0:
            continue

        l1_arr = lfrst[ib] + np.arange(nseg, dtype=np.intp)
        l2_arr = l1_arr + 1

        drl0 = (rl[0, l2_arr] - rl[0, l1_arr]) / beta
        drl1 = rl[1, l2_arr] - rl[1, l1_arr]
        drl2 = rl[2, l2_arr] - rl[2, l1_arr]
        drlmag = np.sqrt(drl0 * drl0 + drl1 * drl1 + drl2 * drl2)
        drlmi = np.where(drlmag == 0.0, 0.0, 1.0 / drlmag)

        # unit vector along line segment
        esl0 = drl0 * drlmi
        esl1 = drl1 * drlmi
        esl2 = drl2 * drlmi

        rad1 = radl[l1_arr]
        rad2 = radl[l2_arr]
        adel = pi * (rad2 * rad2 - rad1 * rad1) * sdfac
        aavg = pi * 0.5 * (rad2 * rad2 + rad1 * rad1) * sdfac

        rlref0 = 0.5 * (rl[0, l2_arr] + rl[0, l1_arr]) - xyzref[0]
        rlref1 = 0.5 * (rl[1, l2_arr] + rl[1, l1_arr]) - xyzref[1]
        rlref2 = 0.5 * (rl[2, l2_arr] + rl[2, l1_arr]) - xyzref[2]

        urel = np.zeros((3, nseg, 6), dtype=np.float64)
        urel[0, :, 0] = 1.0 / beta
        urel[1, :, 1] = 1.0
        urel[2, :, 2] = 1.0
        urel[1, :, 3] = rlref2
        urel[2, :, 3] = -rlref1
        urel[0, :, 3] /= beta
        urel[0, :, 4] = -rlref2 / beta
        urel[2, :, 4] = rlref0
        urel[0, :, 5] = rlref1 / beta
        urel[1, :, 5] = -rlref0

        us = (
            urel[0] * esl0[:, np.newaxis]
            + urel[1] * esl1[:, np.newaxis]
            + urel[2] * esl2[:, np.newaxis]
        )

        # U.es; velocity projected on normal plane = U - (U.es) es
        un0 = urel[0] - us * esl0[:, np.newaxis]
        un1 = urel[1] - us * esl1[:, np.newaxis]
        un2 = urel[2] - us * esl2[:, np.newaxis]

        drlmag2 = drlmag * 2.0
        # total source and doublet strength of segment
        src_u[l1_arr, :] = adel[:, np.newaxis] * us
        dbl_u[0, l1_arr, :] = (aavg[:, np.newaxis] * un0) * drlmag2[:, np.newaxis]
        dbl_u[1, l1_arr, :] = (aavg[:, np.newaxis] * un1) * drlmag2[:, np.newaxis]
        dbl_u[2, l1_arr, :] = (aavg[:, np.newaxis] * un2) * drlmag2[:, np.newaxis]

    return src_u, dbl_u
