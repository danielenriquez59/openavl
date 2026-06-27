"""AIC setup and velocity summation (port of asetup.f)."""

from __future__ import annotations

import numpy as np

from openavl.aero.aic import cross, dot, srdset, vsrd, vvor, _vvor_vortex_pre
from openavl.core.state import AVLState
from openavl.math.linalg import baksub, ludcmp


def u_comp(state: AVLState, iu: int) -> float:
    """Return the iu-th unit velocity component (0-based iu in 0..5)."""
    if iu < 3:
        return float(state.vinf[iu])
    return float(state.wrot[iu - 3])


def _unit_velocity(state: AVLState) -> np.ndarray:
    """Return the active unit-velocity vector [vinf, wrot] of length numax."""
    # Shared by gamsum/velsum so body-source weighting uses one u vector.
    return np.concatenate((state.vinf, state.wrot))[: state.numax]


def _solve_aic_column(state: AVLState, col: np.ndarray) -> None:
    """Solve AICN * x = col using pre-factored LU (in-place on col[:nvor])."""
    baksub(state.aicn, state.nvor, state.iapiv, col[: state.nvor])


def _solve_aic_columns(state: AVLState, cols: np.ndarray) -> None:
    """Solve AICN * X = cols for multiple RHS columns (in-place on cols[:nvor])."""
    # One LAPACK call for many columns instead of repeated single-column solves.
    baksub(state.aicn, state.nvor, state.iapiv, cols[: state.nvor])


def _mungea(state: AVLState) -> None:
    """Zero AIC rows for stripped-off vortices and set diagonal to 1."""
    for j in range(state.nstrip):
        if not state.lstripoff[j]:
            continue
        i1 = int(state.ijfrst[j])
        for k in range(state.nvstrp[j]):
            ii = i1 + k
            state.aicn[ii, : state.nvor] = 0.0
            state.aicn[ii, ii] = 1.0


def setup(state: AVLState) -> AVLState:
    """Build AIC matrix, body influence matrices, and velocity kernels."""
    state.amach = state.mach
    state.betm = (np.sqrt((1.0 - (state.amach * state.amach))))

    vortex_pre = None
    if not state.laic or not state.lvel:
        vortex_pre = _vvor_vortex_pre(
            state.betm, state.iysym, state.ysym, state.izsym, state.zsym,
            state.vrcorec, state.vrcorew,
            state.nvor, state.rv1, state.rv2, state.lvcomp, state.chordv,
            state.nvor, state.lvcomp,
        )

    if not state.laic:
        vvor(
            state.betm, state.iysym, state.ysym, state.izsym, state.zsym,
            state.vrcorec, state.vrcorew,
            state.nvor, state.rv1, state.rv2, state.lvcomp, state.chordv,
            state.nvor, state.rc, state.lvcomp, False,
            state.wc_gam, state.nvor,
            vortex_pre=vortex_pre,
        )

        nv = state.nvor
        state.aicn[:nv, :nv] = np.einsum(
            "kij,ki->ij",
            state.wc_gam[:, :nv, :nv],
            state.enc[:, :nv],
            optimize=True,
        )
        state.lvnc[:nv] = True

        for n in range(state.nsurf):
            if state.lfwake[n]:
                continue
            j1 = int(state.jfrst[n])
            jn = j1 + int(state.nj[n]) - 1
            for j in range(j1, jn + 1):
                i1 = int(state.ijfrst[j])
                iv = int(state.ijfrst[j] + state.nvstrp[j] - 1)
                state.aicn[iv, : state.nvor] = 0.0
                state.lvnc[iv] = False
                for jv in range(i1, iv + 1):
                    state.aicn[iv, jv] = 1.0

        _mungea(state)
        ludcmp(state.aicn, state.nvor, state.iapiv, state.work)
        state.laic = True

    if not state.lsrd:
        srdset(
            state.betm, state.xyzref, state.iysym,
            state.nbody, state.lfrst, state.nlmax,
            state.nl, state.rl, state.radl,
            state.src_u, state.dbl_u,
        )

        state.wcsrd_u = vsrd(
            state.betm, state.iysym, state.ysym, state.izsym, state.zsym, state.srcore,
            state.nbody, state.lfrst, state.nlmax,
            state.nl, state.rl, state.radl,
            6, state.src_u, state.dbl_u,
            state.nvor, state.rc,
            state.wcsrd_u, state.nvor,
        )
        state.lsrd = True

    if not state.lvel:
        vvor(
            state.betm, state.iysym, state.ysym, state.izsym, state.zsym,
            state.vrcorec, state.vrcorew,
            state.nvor, state.rv1, state.rv2, state.lvcomp, state.chordv,
            state.nvor, state.rv, state.lvcomp, True,
            state.wv_gam, state.nvor,
            vortex_pre=vortex_pre,
        )

        state.wvsrd_u = vsrd(
            state.betm, state.iysym, state.ysym, state.izsym, state.zsym, state.srcore,
            state.nbody, state.lfrst, state.nlmax,
            state.nl, state.rl, state.radl,
            6, state.src_u, state.dbl_u,
            state.nvor, state.rv,
            state.wvsrd_u, state.nvor,
        )
        state.lvel = True

    return state


def gucalc(state: AVLState) -> AVLState:
    """Compute unit-circulation sensitivities GAM_U_* via AIC back-substitution."""
    nvor = state.nvor
    lvnc = state.lvnc[:nvor]
    lvalbe = state.lvalbe[:nvor]
    enc = state.enc[:, :nvor]
    wcsrd_u = state.wcsrd_u[:, :nvor, :6]
    inactive = ~lvnc

    for iu in range(3):
        vunit = wcsrd_u[:, :, iu].copy()
        vunit[iu, lvalbe & lvnc] += 1.0
        state.gam_u_0[:nvor, iu] = -np.sum(enc * vunit, axis=0)
        if state.ncontrol:
            state.gam_u_d[:nvor, iu, : state.ncontrol] = -np.einsum(
                "kin,ki->in",
                state.enc_d[:, :nvor, : state.ncontrol],
                vunit,
            )
        if state.ndesign:
            state.gam_u_g[:nvor, iu, : state.ndesign] = -np.einsum(
                "kin,ki->in",
                state.enc_g[:, :nvor, : state.ndesign],
                vunit,
            )
        state.gam_u_0[inactive, iu] = 0.0
        if state.ncontrol:
            state.gam_u_d[inactive, iu, : state.ncontrol] = 0.0
        if state.ndesign:
            state.gam_u_g[inactive, iu, : state.ndesign] = 0.0

    rrot = state.rc[:, :nvor] - state.xyzref[:, np.newaxis]
    for iu in range(3, 6):
        wunit = np.zeros((3, nvor), dtype=np.float64)
        wunit[iu - 3, lvalbe & lvnc] = 1.0
        vunit = np.cross(rrot, wunit, axis=0) + wcsrd_u[:, :, iu]
        state.gam_u_0[:nvor, iu] = -np.sum(enc * vunit, axis=0)
        if state.ncontrol:
            state.gam_u_d[:nvor, iu, : state.ncontrol] = -np.einsum(
                "kin,ki->in",
                state.enc_d[:, :nvor, : state.ncontrol],
                vunit,
            )
        if state.ndesign:
            state.gam_u_g[:nvor, iu, : state.ndesign] = -np.einsum(
                "kin,ki->in",
                state.enc_g[:, :nvor, : state.ndesign],
                vunit,
            )
        state.gam_u_0[inactive, iu] = 0.0
        if state.ncontrol:
            state.gam_u_d[inactive, iu, : state.ncontrol] = 0.0
        if state.ndesign:
            state.gam_u_g[inactive, iu, : state.ndesign] = 0.0

    # Batch all unit-mode RHS columns through the factored AIC matrix.
    _solve_aic_columns(state, state.gam_u_0[: state.nvor, :6])
    if state.ncontrol:
        rhs_d = state.gam_u_d[: state.nvor, :6, : state.ncontrol].reshape(
            state.nvor, 6 * state.ncontrol
        )
        _solve_aic_columns(state, rhs_d)
        state.gam_u_d[: state.nvor, :6, : state.ncontrol] = rhs_d.reshape(
            state.nvor, 6, state.ncontrol
        )
    if state.ndesign:
        rhs_g = state.gam_u_g[: state.nvor, :6, : state.ndesign].reshape(
            state.nvor, 6 * state.ndesign
        )
        _solve_aic_columns(state, rhs_g)
        state.gam_u_g[: state.nvor, :6, : state.ndesign] = rhs_g.reshape(
            state.nvor, 6, state.ndesign
        )

    return state


def gdcalc(
    state: AVLState,
    nqdef: int,
    lqdef: np.ndarray,
    enc_q: np.ndarray,
    gam_q: np.ndarray,
) -> AVLState:
    """Compute defined-variable circulation sensitivities GAM_Q."""
    if nqdef == 0:
        return state

    active = [iq for iq in range(nqdef) if lqdef[iq]]
    if not active:
        return state

    nvor = state.nvor
    lvnc = state.lvnc[:nvor]
    lvalbe = state.lvalbe[:nvor]
    inactive = ~lvnc
    rrot = state.rc[:, :nvor] - state.xyzref[:, np.newaxis]
    vrot = np.cross(rrot, state.wrot[:, np.newaxis], axis=0)
    active_be = lvalbe & lvnc
    u = np.concatenate((state.vinf, state.wrot))
    vq = np.einsum("kin,j->ki", state.wcsrd_u[:, :nvor, :6], u)
    vq[:, active_be] += state.vinf[:, np.newaxis] + vrot[:, active_be]

    gam_q[:nvor, active] = -np.einsum(
        "kin,ki->in",
        enc_q[:, :nvor, active],
        vq,
    )
    gam_q[inactive][:, active] = 0.0

    _solve_aic_columns(state, gam_q[: state.nvor, active])

    return state


def gamsum(state: AVLState) -> AVLState:
    """Sum unit-circulation sensitivities into total GAM and body strengths."""
    # Vectorized over active nvor/control/design extents (replaces per-vortex loops).
    nvor = state.nvor
    ncontrol = state.ncontrol
    ndesign = state.ndesign
    nlnode = state.nlnode
    numax = state.numax
    u = _unit_velocity(state)

    state.gam_u[:nvor, :numax] = state.gam_u_0[:nvor, :numax]
    if ncontrol:
        state.gam_u[:nvor, :numax] += np.einsum(
            "ijn,n->ij",
            state.gam_u_d[:nvor, :numax, :ncontrol],
            state.delcon[:ncontrol],
        )
    if ndesign:
        state.gam_u[:nvor, :numax] += np.einsum(
            "ijn,n->ij",
            state.gam_u_g[:nvor, :numax, :ndesign],
            state.deldes[:ndesign],
        )

    if ncontrol:
        state.gam_d[:nvor, :ncontrol] = np.einsum(
            "ijn,j->in",
            state.gam_u_d[:nvor, :numax, :ncontrol],
            u,
        )
    if ndesign:
        state.gam_g[:nvor, :ndesign] = np.einsum(
            "ijn,j->in",
            state.gam_u_g[:nvor, :numax, :ndesign],
            u,
        )

    state.gam[:nvor] = state.gam_u[:nvor, :numax] @ u

    if nlnode:
        state.src[:nlnode] = state.src_u[:nlnode, :numax] @ u
        state.dbl[:, :nlnode] = np.einsum(
            "klj,j->kl",
            state.dbl_u[:, :nlnode, :numax],
            u,
        )

    return state


def velsum(state: AVLState) -> AVLState:
    """Sum induced and body velocities at control points."""
    # Matrix contractions over wc_gam/wv_gam replace O(nvor^2) Python loops.
    nvor = state.nvor
    ncontrol = state.ncontrol
    ndesign = state.ndesign
    numax = state.numax
    ndmax = state.ndmax
    ngmax = state.ngmax
    u = _unit_velocity(state)

    wc = state.wc_gam[:, :nvor, :nvor]
    wv = state.wv_gam[:, :nvor, :nvor]
    gam = state.gam[:nvor]

    state.vc[:, :nvor] = np.einsum("kij,j->ki", wc, gam)
    state.vv[:, :nvor] = np.einsum("kij,j->ki", wv, gam)
    state.vc_u[:, :nvor, :numax] = np.einsum(
        "kij,jn->kin", wc, state.gam_u[:nvor, :numax]
    )
    state.vv_u[:, :nvor, :numax] = np.einsum(
        "kij,jn->kin", wv, state.gam_u[:nvor, :numax]
    )

    if ncontrol:
        state.vc_d[:, :nvor, :ncontrol] = np.einsum(
            "kij,jn->kin", wc, state.gam_d[:nvor, :ncontrol]
        )
        state.vv_d[:, :nvor, :ncontrol] = np.einsum(
            "kij,jn->kin", wv, state.gam_d[:nvor, :ncontrol]
        )
    if ndesign:
        state.vc_g[:, :nvor, :ndesign] = np.einsum(
            "kij,jn->kin", wc, state.gam_g[:nvor, :ndesign]
        )
        state.vv_g[:, :nvor, :ndesign] = np.einsum(
            "kij,jn->kin", wv, state.gam_g[:nvor, :ndesign]
        )

    wcsrd_u = state.wcsrd_u[:, :nvor, :numax]
    wvsrd_u = state.wvsrd_u[:, :nvor, :numax]
    state.wcsrd[:, :nvor] = np.einsum("kin,n->ki", wcsrd_u, u)
    state.wvsrd[:, :nvor] = np.einsum("kin,n->ki", wvsrd_u, u)

    state.wc[:, :nvor] = state.vc[:, :nvor] + state.wcsrd[:, :nvor]
    state.wv[:, :nvor] = state.vv[:, :nvor] + state.wvsrd[:, :nvor]
    state.wc_u[:, :nvor, :numax] = state.vc_u[:, :nvor, :numax] + wcsrd_u
    state.wv_u[:, :nvor, :numax] = state.vv_u[:, :nvor, :numax] + wvsrd_u

    # Preserve Fortran layout: copy derivative slices out to ndmax/ngmax buffers.
    if ndmax:
        state.wc_d[:, :nvor, :ndmax] = state.vc_d[:, :nvor, :ndmax]
        state.wv_d[:, :nvor, :ndmax] = state.vv_d[:, :nvor, :ndmax]
    if ngmax:
        state.wc_g[:, :nvor, :ngmax] = state.vc_g[:, :nvor, :ngmax]
        state.wv_g[:, :nvor, :ngmax] = state.vv_g[:, :nvor, :ngmax]

    return state
