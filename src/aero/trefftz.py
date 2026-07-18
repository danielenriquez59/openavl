"""Trefftz-plane force integration (port of atpforc.f: TPFORC, PGMAT)."""

from __future__ import annotations

from typing import Any

import numpy as np

def pgmat(
    mach: float,
    alfa: float,
    beta: float,
    p: np.ndarray,
    p_m: np.ndarray,
    p_a: np.ndarray,
    p_b: np.ndarray,
) -> None:
    """Build Prandtl-Glauert transformation matrix (pgmat in atpforc.f).

    Maps body-axis (x, y, z) to compressibility-scaled wind axes (xi, eta, zeta):
        [xi, eta, zeta]^T = P @ [x, y, z]^T

    The Y and Z rows of P are unscaled (incompressible), while row 0 (xi, the
    streamwise direction) is scaled by BINV = 1/sqrt(1-M^2) to account for
    Prandtl-Glauert compressibility. p_m, p_a, p_b are the derivatives of P
    with respect to Mach, alpha, and beta respectively.
    """
    # Prandtl-Glauert compressibility factor: BINV = 1/sqrt(1-M^2)
    binv = (1.0 / np.sqrt((1.0 - (mach * mach))))
    bi_m = ((mach) * (binv * binv * binv))  # d(BINV)/dM
    sina = (np.sin((alfa)))
    cosa = (np.cos((alfa)))
    sinb = (np.sin((beta)))
    cosb = (np.cos((beta)))

    p[0, 0] = (cosa * cosb * binv)
    p[0, 1] = (-sinb * binv)
    p[0, 2] = (sina * cosb * binv)
    p[1, 0] = (cosa * sinb)
    p[1, 1] = cosb
    p[1, 2] = (sina * sinb)
    p[2, 0] = (-sina)
    p[2, 1] = 0.0
    p[2, 2] = cosa

    p_m[0, 0] = (cosa * cosb * bi_m)
    p_m[0, 1] = (-sinb * bi_m)
    p_m[0, 2] = (sina * cosb * bi_m)
    p_m[1, :] = 0.0
    p_m[2, :] = 0.0

    p_a[0, 0] = (-sina * cosb * binv)
    p_a[0, 1] = 0.0
    p_a[0, 2] = (cosa * cosb * binv)
    p_a[1, 0] = (-sina * sinb)
    p_a[1, 1] = 0.0
    p_a[1, 2] = (cosa * sinb)
    p_a[2, 0] = (-cosa)
    p_a[2, 1] = 0.0
    p_a[2, 2] = (-sina)

    p_b[0, 0] = (-cosa * sinb * binv)
    p_b[0, 1] = (-cosb * binv)
    p_b[0, 2] = (-sina * sinb * binv)
    p_b[1, 0] = (cosa * cosb)
    p_b[1, 1] = (-sinb)
    p_b[1, 2] = (sina * cosb)
    p_b[2, :] = 0.0


def vinfab(state: Any) -> Any:
    """Set freestream velocity and its angle derivatives (VINFAB)."""
    sina = (np.sin((state.alfa)))
    cosa = (np.cos((state.alfa)))
    sinb = (np.sin((state.beta)))
    cosb = (np.cos((state.beta)))

    state.vinf[0] = (cosa * cosb)
    state.vinf[1] = (-sinb)
    state.vinf[2] = (sina * cosb)

    state.vinf_a[0] = (-sina * cosb)
    state.vinf_a[1] = 0.0
    state.vinf_a[2] = (cosa * cosb)

    state.vinf_b[0] = (-cosa * sinb)
    state.vinf_b[1] = (-cosb)
    state.vinf_b[2] = (-sina * sinb)
    return state


def _sum_strip_circulations(
    nstrip: int,
    ijfrst: np.ndarray,
    nvstrp: np.ndarray,
    gam: np.ndarray,
    gam_u: np.ndarray,
    gam_d: np.ndarray,
    gam_g: np.ndarray,
    numax: int,
    ncontrol: int,
    ndesign: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sum vortex circulations and sensitivities per strip (GAMS arrays)."""
    gams = np.zeros(nstrip, dtype=np.float64)
    gams_u = np.zeros((nstrip, numax), dtype=np.float64)
    gams_d = np.zeros((nstrip, max(1, ncontrol)), dtype=np.float64)
    gams_g = np.zeros((nstrip, max(1, ndesign)), dtype=np.float64)
    for jc in range(nstrip):
        i1 = int(ijfrst[jc])
        nvc = int(nvstrp[jc])
        if nvc == 0:
            continue
        idx = np.arange(i1, i1 + nvc, dtype=np.intp)
        gams[jc] = np.sum(gam[idx])
        gams_u[jc] = np.sum(gam_u[idx, :numax], axis=0)
        if ncontrol:
            gams_d[jc] = np.sum(gam_d[idx, :ncontrol], axis=0)
        if ndesign:
            gams_g[jc] = np.sum(gam_g[idx, :ndesign], axis=0)
    return gams, gams_u, gams_d, gams_g


def _trefftz_kernel(
    y_f: np.ndarray,
    z_f: np.ndarray,
    y1: np.ndarray,
    z1: np.ndarray,
    y2: np.ndarray,
    z2: np.ndarray,
    rcore: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return 2D Biot-Savart (dterm, yterm) arrays with shape (nc, nv).

    B6 note: when ``rcore`` is given, the squared distance is regularized
    as ``rsq = hypot(r**2, rcore**2) = sqrt(r**4 + rcore**4)``. This is the
    same Leishman r^4 core-model variant documented in
    ``openavl.aero.vortex`` (used there for the horseshoe kernel), rather
    than AVL's ``r**2 + rcore**2`` Scully/Burnham-Hallock form. It changes
    induced-drag values slightly vs. AVL for multi-component configurations
    (the Trefftz core radius is nonzero only across components); see
    ``openavl.aero.vortex`` for the full core-model discussion.
    """
    dy1 = y_f[:, np.newaxis] - y1[np.newaxis, :]
    dy2 = y_f[:, np.newaxis] - y2[np.newaxis, :]
    dz1 = z_f[:, np.newaxis] - z1[np.newaxis, :]
    dz2 = z_f[:, np.newaxis] - z2[np.newaxis, :]
    if rcore is None:
        rsq1 = dy1 * dy1 + dz1 * dz1
        rsq2 = dy2 * dy2 + dz2 * dz2
    else:
        rsq1 = np.hypot(dy1 * dy1 + dz1 * dz1, rcore * rcore)
        rsq2 = np.hypot(dy2 * dy2 + dz2 * dz2, rcore * rcore)
    dterm = (dz1 / rsq1) - (dz2 / rsq2)
    yterm = (-dy1 / rsq1) + (dy2 / rsq2)
    return dterm, yterm


def _contract_trefftz_kernel(
    dterm: np.ndarray,
    yterm: np.ndarray,
    gams: np.ndarray,
    gams_u: np.ndarray,
    gams_d: np.ndarray,
    gams_g: np.ndarray,
    active: np.ndarray,
    hpi: float,
    sign: float,
    numax: int,
    ncontrol: int,
    ndesign: int,
) -> tuple[
    np.ndarray, np.ndarray,
    np.ndarray, np.ndarray,
    np.ndarray, np.ndarray,
    np.ndarray, np.ndarray,
]:
    """Contract kernel matrices with strip circulations and sensitivities."""
    wt = active[np.newaxis, :] * sign
    dterm = dterm * wt
    yterm = yterm * wt
    vy = hpi * np.sum(gams[np.newaxis, :] * dterm, axis=1)
    vz = hpi * np.sum(gams[np.newaxis, :] * yterm, axis=1)
    vy_u = hpi * np.einsum("jn,ij->in", gams_u[:, :numax], dterm, optimize=True)
    vz_u = hpi * np.einsum("jn,ij->in", gams_u[:, :numax], yterm, optimize=True)
    vy_d = vz_d = vy_g = vz_g = None
    if ncontrol:
        vy_d = hpi * np.einsum("jn,ij->in", gams_d[:, :ncontrol], dterm, optimize=True)
        vz_d = hpi * np.einsum("jn,ij->in", gams_d[:, :ncontrol], yterm, optimize=True)
    if ndesign:
        vy_g = hpi * np.einsum("jn,ij->in", gams_g[:, :ndesign], dterm, optimize=True)
        vz_g = hpi * np.einsum("jn,ij->in", gams_g[:, :ndesign], yterm, optimize=True)
    return vy, vz, vy_u, vz_u, vy_d, vz_d, vy_g, vz_g


def _accumulate_trefftz_induced(
    ycntr: np.ndarray,
    zcntr: np.ndarray,
    rt1_y: np.ndarray,
    rt1_z: np.ndarray,
    rt2_y: np.ndarray,
    rt2_z: np.ndarray,
    gams: np.ndarray,
    gams_u: np.ndarray,
    gams_d: np.ndarray,
    gams_g: np.ndarray,
    rcore_ij: np.ndarray,
    active: np.ndarray,
    hpi: float,
    izsym: int,
    iysym: int,
    yoff: float,
    zoff: float,
    numax: int,
    ncontrol: int,
    ndesign: int,
) -> tuple[
    np.ndarray, np.ndarray,
    np.ndarray, np.ndarray,
    np.ndarray, np.ndarray,
    np.ndarray, np.ndarray,
]:
    """Sum Trefftz-plane induced velocities over all wake strips and images."""
    vy = np.zeros(ycntr.shape[0], dtype=np.float64)
    vz = np.zeros(ycntr.shape[0], dtype=np.float64)
    vy_u = np.zeros((ycntr.shape[0], numax), dtype=np.float64)
    vz_u = np.zeros((ycntr.shape[0], numax), dtype=np.float64)
    vy_d = np.zeros((ycntr.shape[0], max(1, ncontrol)), dtype=np.float64) if ncontrol else None
    vz_d = np.zeros((ycntr.shape[0], max(1, ncontrol)), dtype=np.float64) if ncontrol else None
    vy_g = np.zeros((ycntr.shape[0], max(1, ndesign)), dtype=np.float64) if ndesign else None
    vz_g = np.zeros((ycntr.shape[0], max(1, ndesign)), dtype=np.float64) if ndesign else None

    def _add(dterm: np.ndarray, yterm: np.ndarray, sign: float) -> None:
        cvy, cvz, cvy_u, cvz_u, cvy_d, cvz_d, cvy_g, cvz_g = _contract_trefftz_kernel(
            dterm, yterm, gams, gams_u, gams_d, gams_g, active, hpi, sign,
            numax, ncontrol, ndesign,
        )
        vy[:] += cvy
        vz[:] += cvz
        vy_u[:] += cvy_u
        vz_u[:] += cvz_u
        if ncontrol and cvy_d is not None:
            vy_d[:] += cvy_d
            vz_d[:] += cvz_d
        if ndesign and cvy_g is not None:
            vy_g[:] += cvy_g
            vz_g[:] += cvz_g

    dterm, yterm = _trefftz_kernel(ycntr, zcntr, rt1_y, rt1_z, rt2_y, rt2_z, rcore_ij)
    # Sum velocity contributions from wake vortices (real vortex)
    _add(dterm, yterm, 1.0)

    if izsym != 0:
        # Sum velocity contributions from wake vortices (z-image)
        z1z = zoff - rt1_z
        z2z = zoff - rt2_z
        dterm, yterm = _trefftz_kernel(ycntr, zcntr, rt1_y, z1z, rt2_y, z2z)
        _add(dterm, yterm, -float(izsym))

    if iysym != 0:
        # Sum velocity contributions from wake vortices (y-image)
        y1y = yoff - rt1_y
        y2y = yoff - rt2_y
        dterm, yterm = _trefftz_kernel(ycntr, zcntr, y1y, rt1_z, y2y, rt2_z)
        _add(dterm, yterm, -float(iysym))

        if izsym != 0:
            z1z = zoff - rt1_z
            z2z = zoff - rt2_z
            dterm, yterm = _trefftz_kernel(ycntr, zcntr, y1y, z1z, y2y, z2z)
            sym_yz = float(iysym * izsym)
            cvy, cvz, cvy_u, cvz_u, cvy_d, cvz_d, cvy_g, cvz_g = _contract_trefftz_kernel(
                dterm, yterm, gams, gams_u, gams_d, gams_g, active, hpi, sym_yz,
                numax, ncontrol, ndesign,
            )
            vy += cvy
            vz += cvz
            # NOTE: the yz-image contribution to gams/gams_u/gams_d/gams_g all
            # pass through the *same* signed kernel (dterm/yterm already carry
            # sym_yz via `wt` in _contract_trefftz_kernel), so the sensitivity
            # accumulation must use the same sign as the primal velocity
            # accumulation above. An earlier version subtracted these terms,
            # which is inconsistent with the primal (verified by FD in
            # tests/core/test_trefftz.py::test_tpforc_yz_image_sensitivity_fd).
            vy_u += cvy_u
            vz_u += cvz_u
            if ncontrol and cvy_d is not None:
                vy_d += cvy_d
                vz_d += cvz_d
            if ndesign and cvy_g is not None:
                vy_g += cvy_g
                vz_g += cvz_g

    return vy, vz, vy_u, vz_u, vy_d, vz_d, vy_g, vz_g


def tpforc(state: Any) -> None:
    """Trefftz-plane force integration (TPFORC from atpforc.f).

    Calculates far-field forces on the configuration using a Trefftz-plane
    method (kinetic energy integral in the far wake). Only the span-loading
    (gams) and wake geometry (Y, Z coordinates of the trailing vortex legs)
    are required. The normalwash in the cross-plane is evaluated over both
    the real and image sides to enforce symmetry boundary conditions.

    When both Y and Z images are active (``iysym·izsym ≠ 0``), OpenAVL
    accumulates yz-image velocity *and* its ``_u/_d/_g`` sensitivities with
    the same sign. AVL 3.50 adds the primal contribution but subtracts the
    sensitivities — an inconsistency OpenAVL deliberately corrects (see
    ``test_tpforc_yz_image_sensitivity_fd``).

    Outputs written back to state:
        clff    -- far-field lift coefficient
        cyff    -- far-field side-force coefficient
        cdff    -- far-field induced drag coefficient
        spanef  -- span efficiency  e = (CL^2 + CY^2) / (pi * AR * CDi)
        dwwake  -- far-field downwash at each strip centre
        *_u/d/g -- derivatives w.r.t. state, control, and design variables
    """
    numax = state.numax
    ncontrol = state.ncontrol
    ndesign = state.ndesign
    nstrip = state.nstrip
    amach = getattr(state, "amach", state.mach)

    hpi = (1.0 / (2.0 * state.pi))
    p = np.zeros((3, 3), dtype=np.float64)
    p_m = np.zeros((3, 3), dtype=np.float64)
    p_a = np.zeros((3, 3), dtype=np.float64)
    p_b = np.zeros((3, 3), dtype=np.float64)
    # set Prandtl-Glauert transformation matrix
    pgmat(amach, 0.0, 0.0, p, p_m, p_a, p_b)

    yoff = (2.0 * state.ysym)
    zoff = (2.0 * state.zsym)

    gam = state.gam.reshape(-1) if state.gam.ndim > 1 else state.gam
    gam_u = state.gam_u
    gam_d = getattr(state, "gam_d", np.zeros((state.nvor, max(1, ncontrol)), dtype=np.float64))
    gam_g = getattr(state, "gam_g", np.zeros((state.nvor, max(1, ndesign)), dtype=np.float64))

    gams, gams_u, gams_d, gams_g = _sum_strip_circulations(
        nstrip, state.ijfrst, state.nvstrp, gam, gam_u, gam_d, gam_g,
        numax, ncontrol, ndesign,
    )

    ic = (state.ijfrst + state.nvstrp - 1).astype(np.intp)
    # set x,y,z in wind axes (Y,Z are then in Trefftz plane)
    rt1 = p @ state.rv1[:, ic]
    rt2 = p @ state.rv2[:, ic]
    rtc = p @ state.rc[:, ic]

    dyt = rt2[1, :] - rt1[1, :]
    dzt = rt2[2, :] - rt1[2, :]
    dst = np.hypot(dyt, dzt)
    ny = np.divide(-dzt, dst, out=np.zeros_like(dzt), where=dst != 0.0)
    nz = np.divide(dyt, dst, out=np.zeros_like(dyt), where=dst != 0.0)

    dsy = dyt
    dsz = dzt
    dsyz = dst
    comp_jc = state.lncomp[state.lssurf]
    comp_jv = comp_jc
    cross_comp = comp_jc[:, np.newaxis] != comp_jv[np.newaxis, :]
    rcore_cross = np.maximum(state.vrcorec * state.chord[:nstrip], state.vrcorew * dsyz)
    rcore_ij = np.where(cross_comp, rcore_cross[np.newaxis, :], 0.0)

    active = ~np.asarray(state.lstripoff[:nstrip], dtype=bool)
    vy, vz, vy_u, vz_u, vy_d, vz_d, vy_g, vz_g = _accumulate_trefftz_induced(
        rtc[1, :], rtc[2, :],
        rt1[1, :], rt1[2, :], rt2[1, :], rt2[2, :],
        gams, gams_u, gams_d, gams_g,
        rcore_ij, active.astype(np.float64),
        hpi, state.izsym, state.iysym, yoff, zoff,
        numax, ncontrol, ndesign,
    )

    dwwake = -(ny * vy + nz * vz)
    dwwake = np.where(active, dwwake, 0.0)

    lfload = np.asarray(getattr(state, "lfload", np.ones(state.nsurf, dtype=bool)), dtype=bool)
    load_strip = active & lfload[state.lssurf[:nstrip]]
    inv_sref = 1.0 / state.sref

    # Trefftz-plane drag is kinetic energy in crossflow
    clff = float(2.0 * np.sum(gams * dyt * inv_sref, where=load_strip))
    cyff = float(-2.0 * np.sum(gams * dzt * inv_sref, where=load_strip))
    cdff = float(np.sum(gams * (dzt * vy - dyt * vz) * inv_sref, where=load_strip))

    clff_u = np.zeros(numax, dtype=np.float64)
    cyff_u = np.zeros(numax, dtype=np.float64)
    cdff_u = np.zeros(numax, dtype=np.float64)
    clff_d = np.zeros(ncontrol, dtype=np.float64)
    cyff_d = np.zeros(ncontrol, dtype=np.float64)
    cdff_d = np.zeros(ncontrol, dtype=np.float64)
    clff_g = np.zeros(ndesign, dtype=np.float64)
    cyff_g = np.zeros(ndesign, dtype=np.float64)
    cdff_g = np.zeros(ndesign, dtype=np.float64)

    if np.any(load_strip):
        ls = load_strip
        clff_u = 2.0 * np.sum(gams_u[ls, :numax] * dyt[ls, np.newaxis], axis=0) * inv_sref
        cyff_u = -2.0 * np.sum(gams_u[ls, :numax] * dzt[ls, np.newaxis], axis=0) * inv_sref
        cdff_u = (
            np.sum(gams_u[ls, :numax] * (dzt[ls, np.newaxis] * vy[ls, np.newaxis] - dyt[ls, np.newaxis] * vz[ls, np.newaxis]), axis=0)
            + np.sum(gams[ls, np.newaxis] * (dzt[ls, np.newaxis] * vy_u[ls, :numax] - dyt[ls, np.newaxis] * vz_u[ls, :numax]), axis=0)
        ) * inv_sref
        if ncontrol:
            clff_d = 2.0 * np.sum(gams_d[ls, :ncontrol] * dyt[ls, np.newaxis], axis=0) * inv_sref
            cyff_d = -2.0 * np.sum(gams_d[ls, :ncontrol] * dzt[ls, np.newaxis], axis=0) * inv_sref
            cdff_d = (
                np.sum(gams_d[ls, :ncontrol] * (dzt[ls, np.newaxis] * vy[ls, np.newaxis] - dyt[ls, np.newaxis] * vz[ls, np.newaxis]), axis=0)
                + np.sum(gams[ls, np.newaxis] * (dzt[ls, np.newaxis] * vy_d[ls, :ncontrol] - dyt[ls, np.newaxis] * vz_d[ls, :ncontrol]), axis=0)
            ) * inv_sref
        if ndesign:
            clff_g = 2.0 * np.sum(gams_g[ls, :ndesign] * dyt[ls, np.newaxis], axis=0) * inv_sref
            cyff_g = -2.0 * np.sum(gams_g[ls, :ndesign] * dzt[ls, np.newaxis], axis=0) * inv_sref
            cdff_g = (
                np.sum(gams_g[ls, :ndesign] * (dzt[ls, np.newaxis] * vy[ls, np.newaxis] - dyt[ls, np.newaxis] * vz[ls, np.newaxis]), axis=0)
                + np.sum(gams[ls, np.newaxis] * (dzt[ls, np.newaxis] * vy_g[ls, :ndesign] - dyt[ls, np.newaxis] * vz_g[ls, :ndesign]), axis=0)
            ) * inv_sref

    # If case is XZ symmetric (IYSYM=1), add contributions from images,
    # zero out the asymmetric forces and double the symmetric ones
    if state.iysym == 1:
        clff = (2.0 * clff)
        cyff = (0.0)
        cdff = (2.0 * cdff)
        clff_u[:] = (2.0 * clff_u)
        cyff_u[:] = 0.0
        cdff_u[:] = (2.0 * cdff_u)
        clff_d[:] = (2.0 * clff_d)
        cyff_d[:] = 0.0
        cdff_d[:] = (2.0 * cdff_d)
        clff_g[:] = (2.0 * clff_g)
        cyff_g[:] = 0.0
        cdff_g[:] = (2.0 * cdff_g)

    # aspect ratio
    ar = (state.bref * state.bref / state.sref)
    if cdff == 0.0:
        spanef = (0.0)
        spanef_a = (0.0)
        spanef_u = np.zeros(numax, dtype=np.float64)
        spanef_d = np.zeros(ncontrol, dtype=np.float64)
        spanef_g = np.zeros(ndesign, dtype=np.float64)
    else:
        # span efficiency: e = (CL^2 + CY^2) / (pi * AR * CDi)
        spanef = ((clff * clff + cyff * cyff) / (state.pi * ar * cdff))
        # Partial derivatives of span efficiency for sensitivity propagation
        spanef_cl = ((2.0 * clff) / (state.pi * ar * cdff))
        spanef_cy = ((2.0 * cyff) / (state.pi * ar * cdff))
        spanef_cd = (-spanef / cdff)
        spanef_a = (0.0)
        spanef_u = (spanef_cl * clff_u + spanef_cy * cyff_u + spanef_cd * cdff_u)
        spanef_d = (spanef_cl * clff_d + spanef_cy * cyff_d + spanef_cd * cdff_d)
        spanef_g = (spanef_cl * clff_g + spanef_cy * cyff_g + spanef_cd * cdff_g)

    state.clff = clff
    state.cyff = cyff
    state.cdff = cdff
    state.spanef = spanef
    state.spanef_a = spanef_a
    state.dwwake = dwwake
    state.clff_u = clff_u
    state.cyff_u = cyff_u
    state.cdff_u = cdff_u
    state.spanef_u = spanef_u
    state.clff_d = clff_d
    state.cyff_d = cyff_d
    state.cdff_d = cdff_d
    state.spanef_d = spanef_d
    state.clff_g = clff_g
    state.cyff_g = cyff_g
    state.cdff_g = cdff_g
    state.spanef_g = spanef_g


