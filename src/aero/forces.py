"""Aerodynamic force integration (port of aero.f: VINFAB, AERO, SFFORC, BDFORC).

Computes inviscid forces from horseshoe vortices, inviscid body source forces,
viscous strip drag from CD(CL) polars, and far-field Trefftz-plane contributions.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from openavl.aero.aic import dot
from openavl.aero.cdcl import cdcl
from openavl.aero.trefftz import tpforc
from openavl.geom.geometry import solver_surface_index, solver_surface_name

# Fortran ICRS/JCRS indexing for explicit cross-product components.
ICRS = (1, 2, 0)
JCRS = (2, 0, 1)


def vinfab(state: Any) -> Any:
    """Set freestream velocity and its angle derivatives (VINFAB).

    Purpose: calculate free stream vector components and sensitivities.

    Input: ALFA (angle of attack), BETA (sideslip, positive wind on right
    cheek facing forward).

    Output: VINF (velocity components), VINF_A (dVINF/dALFA),
    VINF_B (dVINF/dBETA).
    """
    sina = np.sin(state.alfa)
    cosa = np.cos(state.alfa)
    sinb = np.sin(state.beta)
    cosb = np.cos(state.beta)

    # Unit freestream velocity vector in body axes.
    state.vinf[0] = cosa * cosb
    state.vinf[1] = -sinb
    state.vinf[2] = sina * cosb

    state.vinf_a[0] = -sina * cosb
    state.vinf_a[1] = 0.0
    state.vinf_a[2] = cosa * cosb

    state.vinf_b[0] = -cosa * sinb
    state.vinf_b[1] = -cosb
    state.vinf_b[2] = -sina * sinb
    return state


def bdforc(state: Any) -> Any:
    """Integrate body-axis source forces (BDFORC).

    Sums slender-body source-panel forces on each body line segment and adds
    them to the global force/moment totals.
    """
    nbody = getattr(state, "nbody", 0)
    if nbody == 0:
        return state

    numax = state.numax
    # Prandtl-Glauert compressibility factor (x-direction scaling).
    betm = (np.sqrt((1.0 - (state.mach * state.mach))))
    sina = (np.sin((state.alfa)))
    cosa = (np.cos((state.alfa)))

    nl    = state.nl
    lfrst = state.lfrst
    rl    = state.rl
    radl  = state.radl
    src   = state.src
    src_u = state.src_u
    dim_l = getattr(state, "dim_l", nl.shape[0] if hasattr(nl, "shape") else len(nl))
    dcpb  = state.dcpb
    cdbdy = state.cdbdy
    cybdy = state.cybdy
    clbdy = state.clbdy
    cfbdy = state.cfbdy
    cmbdy = state.cmbdy

    for ib in range(nbody):
        # add on body force contributions
        cdbdy[ib] = 0.0
        cybdy[ib] = 0.0
        clbdy[ib] = 0.0
        cfbdy[:, ib] = 0.0
        cmbdy[:, ib] = 0.0

        cdbdy_u = np.zeros(6, dtype=np.float64)
        cybdy_u = np.zeros(6, dtype=np.float64)
        clbdy_u = np.zeros(6, dtype=np.float64)
        cfbdy_u = np.zeros((3, numax), dtype=np.float64)
        cmbdy_u = np.zeros((3, numax), dtype=np.float64)

        nln = int(nl[ib])
        nseg = nln - 1
        if nseg <= 0:
            continue

        l1_arr = lfrst[ib] + np.arange(nseg, dtype=np.intp)
        l2_arr = l1_arr + 1

        drl0 = (rl[0, l2_arr] - rl[0, l1_arr]) / betm
        drl1 = rl[1, l2_arr] - rl[1, l1_arr]
        drl2 = rl[2, l2_arr] - rl[2, l1_arr]
        drlmag = np.sqrt(drl0 * drl0 + drl1 * drl1 + drl2 * drl2)
        drlmi = np.where(drlmag == 0.0, 0.0, 1.0 / drlmag)

        dia = radl[l1_arr] + radl[l2_arr]
        dinv = np.where(dia <= 0.0, 0.0, 1.0 / dia)

        # unit vector along line segment
        esl0 = drl0 * drlmi
        esl1 = drl1 * drlmi
        esl2 = drl2 * drlmi

        rrot0 = 0.5 * (rl[0, l2_arr] + rl[0, l1_arr]) - state.xyzref[0]
        rrot1 = 0.5 * (rl[1, l2_arr] + rl[1, l1_arr]) - state.xyzref[1]
        rrot2 = 0.5 * (rl[2, l2_arr] + rl[2, l1_arr]) - state.xyzref[2]

        vrot0 = rrot1 * state.wrot[2] - rrot2 * state.wrot[1]
        vrot1 = rrot2 * state.wrot[0] - rrot0 * state.wrot[2]
        vrot2 = rrot0 * state.wrot[1] - rrot1 * state.wrot[0]

        veff0 = (state.vinf[0] + vrot0) / betm
        veff1 = state.vinf[1] + vrot1
        veff2 = state.vinf[2] + vrot2

        # VEFF_x = (VINF_x + VROT_x)/BETM, so all x-row sensitivities carry 1/BETM.
        # AVL omits 1/BETM on VEFF_U(1,1) and on the rotation x-entries; OpenAVL
        # uses the consistent calculus (matches JAX AD through the primal).
        veff_u = np.zeros((3, 6, nseg), dtype=np.float64)
        veff_u[0, 0, :] = 1.0 / betm
        veff_u[1, 1, :] = 1.0
        veff_u[2, 2, :] = 1.0
        veff_u[0, 3, :] = 0.0
        veff_u[1, 3, :] = rrot2
        veff_u[2, 3, :] = -rrot1
        veff_u[0, 4, :] = -rrot2 / betm
        veff_u[1, 4, :] = 0.0
        veff_u[2, 4, :] = rrot0
        veff_u[0, 5, :] = rrot1 / betm
        veff_u[1, 5, :] = -rrot0
        veff_u[2, 5, :] = 0.0

        us = veff0 * esl0 + veff1 * esl1 + veff2 * esl2
        # U.es
        us_u = (
            veff_u[0] * esl0[np.newaxis, :]
            + veff_u[1] * esl1[np.newaxis, :]
            + veff_u[2] * esl2[np.newaxis, :]
        )

        un0 = veff0 - us * esl0
        un1 = veff1 - us * esl1
        un2 = veff2 - us * esl2
        # velocity projected on normal plane = U - (U.es) es
        src_l1 = src[l1_arr]

        fb0 = un0 * src_l1
        fb1 = un1 * src_l1
        fb2 = un2 * src_l1

        fb_u = np.zeros((3, 6, nseg), dtype=np.float64)
        for k, (un, esl_k) in enumerate(((un0, esl0), (un1, esl1), (un2, esl2))):
            un_u = veff_u[k] - us_u * esl_k[np.newaxis, :]
            fb_u[k] = un[np.newaxis, :] * src_u[l1_arr, :].T + un_u * src_l1[np.newaxis, :]

        dcpb[0, l1_arr] = fb0 * (2.0 * dinv * drlmi)
        dcpb[1, l1_arr] = fb1 * (2.0 * dinv * drlmi)
        dcpb[2, l1_arr] = fb2 * (2.0 * dinv * drlmi)

        mb0 = rrot1 * fb2 - rrot2 * fb1
        mb1 = rrot2 * fb0 - rrot0 * fb2
        mb2 = rrot0 * fb1 - rrot1 * fb0

        mb_u0 = rrot1[np.newaxis, :] * fb_u[2] - rrot2[np.newaxis, :] * fb_u[1]
        mb_u1 = rrot2[np.newaxis, :] * fb_u[0] - rrot0[np.newaxis, :] * fb_u[2]
        mb_u2 = rrot0[np.newaxis, :] * fb_u[1] - rrot1[np.newaxis, :] * fb_u[0]

        inv_sref = 2.0 / state.sref
        inv_bref = inv_sref / state.bref
        inv_cref = inv_sref / state.cref

        cdbdy[ib] = float(np.sum((fb0 * cosa + fb2 * sina) * inv_sref))
        cybdy[ib] = float(np.sum(fb1 * inv_sref))
        clbdy[ib] = float(np.sum((-fb0 * sina + fb2 * cosa) * inv_sref))
        cfbdy[:, ib] = np.array([np.sum(fb0 * inv_sref), np.sum(fb1 * inv_sref), np.sum(fb2 * inv_sref)])
        cmbdy[0, ib] = float(np.sum(mb0 * inv_bref))
        cmbdy[1, ib] = float(np.sum(mb1 * inv_cref))
        cmbdy[2, ib] = float(np.sum(mb2 * inv_bref))

        cdbdy_u[:] = np.sum((fb_u[0] * cosa + fb_u[2] * sina) * inv_sref, axis=1)
        cybdy_u[:] = np.sum(fb_u[1] * inv_sref, axis=1)
        clbdy_u[:] = np.sum((-fb_u[0] * sina + fb_u[2] * cosa) * inv_sref, axis=1)
        cfbdy_u[:, :] = np.sum(fb_u * inv_sref, axis=2)
        cmbdy_u[0, :] = np.sum(mb_u0 * inv_bref, axis=1)
        cmbdy_u[1, :] = np.sum(mb_u1 * inv_cref, axis=1)
        cmbdy_u[2, :] = np.sum(mb_u2 * inv_bref, axis=1)

        # add body forces and sensitivities to totals
        state.cdtot      = (state.cdtot + cdbdy[ib])
        state.cytot      = (state.cytot + cybdy[ib])
        state.cltot      = (state.cltot + clbdy[ib])
        state.cftot[:]   = (state.cftot + cfbdy[:, ib])
        state.cmtot[:]   = (state.cmtot + cmbdy[:, ib])
        state.cdtot_u[:] = (state.cdtot_u + cdbdy_u)
        state.cytot_u[:] = (state.cytot_u + cybdy_u)
        state.cltot_u[:] = (state.cltot_u + clbdy_u)
        state.cftot_u[:] = (state.cftot_u + cfbdy_u)
        state.cmtot_u[:] = (state.cmtot_u + cmbdy_u)

    return state


def _cross3(a0, a1, a2, b0, b1, b2) -> tuple[float, float, float]:
    """Scalar 3-vector cross product ``a x b``.

    Same products and subtractions, in the same order, as ``np.cross`` on two
    3-vectors, so results are bit-identical without the ``np.cross`` overhead.
    """
    return (a1 * b2 - a2 * b1, a2 * b0 - a0 * b2, a0 * b1 - a1 * b0)


def _cross_axis0(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cross product ``a × b`` for ``[3, ...]`` arrays along axis 0 (no ``np.cross``)."""
    return np.stack(
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ],
        axis=0,
    )


def _batch_veff_u(
    rrot: np.ndarray,
    vind_u: np.ndarray,
    numax: int,
    veff_u: np.ndarray,
) -> None:
    """Fill ``veff_u`` (3, numax, nvor) from induced-velocity sensitivities and rotation."""
    veff_u[:, :, :] = vind_u[:, :, :numax].transpose(0, 2, 1)
    veff_u[0, 0, :] += 1.0
    veff_u[1, 1, :] += 1.0
    veff_u[2, 2, :] += 1.0
    if numax > 3:
        veff_u[1, 3, :] += rrot[2, :]
        veff_u[2, 3, :] -= rrot[1, :]
        veff_u[0, 4, :] -= rrot[2, :]
        veff_u[2, 4, :] += rrot[0, :]
        veff_u[0, 5, :] += rrot[1, :]
        veff_u[1, 5, :] -= rrot[0, :]


class _StripSummer:
    """Sum ``(k, nvor)`` per-vortex rows into ``(nstrip, k)`` per-strip totals.

    Equivalent, bit for bit, to ``np.add.at(zeros((nstrip, k)), v2s, data.T)``:
    one ``np.bincount`` over a flattened (row, strip) index adds every weight
    sequentially in vortex order, starting from zero, just as ``ufunc.at``
    does. No ordering or contiguity of ``vortex_to_strip`` is assumed.
    """

    def __init__(self, vortex_to_strip: np.ndarray, nstrip: int, k: int) -> None:
        self.nstrip = nstrip
        self.k = k
        self.index = (
            vortex_to_strip[np.newaxis, :].astype(np.intp)
            + nstrip * np.arange(k, dtype=np.intp)[:, np.newaxis]
        ).ravel()

    def __call__(self, data: np.ndarray) -> np.ndarray:
        sums = np.bincount(
            self.index, weights=data.ravel(), minlength=self.k * self.nstrip
        )
        return sums.reshape(self.k, self.nstrip).T


def _broadcast_strip_sum(
    vortex_to_strip: np.ndarray, nstrip: int, data: np.ndarray, k: int
) -> np.ndarray:
    """``np.add.at(zeros((nstrip, k)), v2s, data[:, None])``: one strip sum per column."""
    sums = np.bincount(vortex_to_strip, weights=data, minlength=nstrip)
    return np.repeat(sums[:, np.newaxis], k, axis=1)


def _accumulate_all_vortex_forces(
    state: Any,
    numax: int,
    ncontrol: int,
    gam: np.ndarray,
    gam_u: np.ndarray,
    gam_d: np.ndarray,
    vv: np.ndarray,
    vv_u: np.ndarray,
    vv_d: np.ndarray,
    wv: np.ndarray,
    wv_u: np.ndarray,
    wv_d: np.ndarray,
    env_d: np.ndarray,
    lnfld_wv: bool,
) -> tuple[
    tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    ],
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Vectorized bound-vortex force accumulation for all vortices in one pass."""
    nvor = state.nvor
    nstrip = state.nstrip
    # Every returned array is distinct: sfforc updates strip rows in place.
    def zeros_u():
        return np.zeros((nstrip, numax), dtype=np.float64)

    def zeros_d():
        return np.zeros((nstrip, max(1, ncontrol)), dtype=np.float64)

    if nvor == 0:
        empty = (
            np.zeros(nstrip), np.zeros(nstrip), np.zeros(nstrip),
            np.zeros(nstrip), np.zeros(nstrip), np.zeros(nstrip),
            zeros_u(), zeros_u(), zeros_u(), zeros_u(), zeros_u(), zeros_u(),
            zeros_d(), zeros_d(), zeros_d(), zeros_d(), zeros_d(), zeros_d(),
        )
        return empty, np.zeros(nstrip), zeros_u(), zeros_d()

    vortex_to_strip = getattr(state, "vortex_to_strip", None)
    if vortex_to_strip is None or vortex_to_strip.shape[0] != nvor:
        from openavl.core.state import build_vortex_to_strip

        vortex_to_strip = build_vortex_to_strip(state)

    rc4_v = np.empty((3, nvor), dtype=np.float64)
    rc4_v[0] = state.rle[0, vortex_to_strip] + 0.25 * state.chord[vortex_to_strip]
    rc4_v[1] = state.rle[1, vortex_to_strip]
    rc4_v[2] = state.rle[2, vortex_to_strip]

    sr = state.chord[vortex_to_strip] * state.wstrip[vortex_to_strip]
    cr_v = state.chord[vortex_to_strip]
    ensy_v = state.ensy[vortex_to_strip]
    ensz_v = state.ensz[vortex_to_strip]
    active = (
        (~state.lstripoff[vortex_to_strip])
        & (state.wstrip[vortex_to_strip] > 0.0)
    )

    bound_leg = state.rv2[:, :nvor] - state.rv1[:, :nvor]
    r = state.rv[:, :nvor] - rc4_v
    rrot = state.rv[:, :nvor] - state.xyzref[:, np.newaxis]
    vrot = _cross_axis0(rrot, state.wrot[:, np.newaxis])

    if lnfld_wv:
        # forces using total induced velocity WV (h.v.'s + body source+doublets)
        vind = wv[:, :nvor]
        vind_u = wv_u[:, :nvor, :numax]
        vind_d = wv_d[:, :nvor, :ncontrol] if ncontrol else None
    else:
        # forces using h.v. induced velocities VV only (exclude body induced)
        vind = vv[:, :nvor]
        vind_u = vv_u[:, :nvor, :numax]
        vind_d = vv_d[:, :nvor, :ncontrol] if ncontrol else None

    veff = state.vinf[:, np.newaxis] + vrot + vind
    # set VEFF sensitivities to freestream,rotation,induced,controls
    veff_u = np.empty((3, numax, nvor), dtype=np.float64)
    _batch_veff_u(rrot, vind_u, numax, veff_u)

    bound_leg_bc = bound_leg[:, np.newaxis, :]
    # Force coefficient on vortex segment is 2(Veff x Gamma)
    f = _cross_axis0(veff, bound_leg)
    f_u = _cross_axis0(veff_u, bound_leg_bc)

    f_d = None
    if ncontrol:
        veff_d = vind_d.transpose(0, 2, 1)
        f_d = _cross_axis0(veff_d, bound_leg_bc)

    gamma = gam[:nvor]
    fgam = 2.0 * gamma * f
    fgam_u = (
        2.0 * gam_u[:nvor, :].T[np.newaxis, :, :] * f[:, np.newaxis, :]
        + 2.0 * gamma * f_u
    )

    fgam_d = None
    if ncontrol:
        fgam_d = (
            2.0 * gam_d[:nvor, :].T[np.newaxis, :, :] * f[:, np.newaxis, :]
            + 2.0 * gamma * f_d
        )

    env = state.env[:, :nvor]
    dxv_w = state.dxv[:nvor] * state.wstrip[vortex_to_strip]
    # Delta Cp (loading across lifting surface) from vortex
    state.dcp[:nvor] = np.sum(env * fgam, axis=0) / dxv_w
    state.dcp_u[:nvor, :numax] = (np.sum(env[:, np.newaxis, :] * fgam_u, axis=0) / dxv_w).T
    if ncontrol:
        term1 = np.sum(env[:, np.newaxis, :] * fgam_d, axis=0)
        term2 = np.einsum("kin,ki->in", env_d[:, :nvor, :ncontrol], fgam)
        state.dcp_d[:nvor, :ncontrol] = (term1.T + term2) / dxv_w[:, np.newaxis]

    inv_sr = np.where(active, 1.0 / sr, 0.0)
    inv_cr = np.where(active, 1.0 / cr_v, 0.0)
    # vortex contribution to strip forces; forces normalized by strip area
    dcfx = fgam[0, :] * inv_sr
    dcfy = fgam[1, :] * inv_sr
    dcfz = fgam[2, :] * inv_sr
    # moments referred to strip c/4 pt., normalized by strip chord and area
    dcmx = (dcfz * r[1, :] - dcfy * r[2, :]) * inv_cr
    dcmy = (dcfx * r[2, :] - dcfz * r[0, :]) * inv_cr
    dcmz = (dcfy * r[0, :] - dcfx * r[1, :]) * inv_cr
    # accumulate strip spanloading = c*CN
    dcnc = cr_v * (ensy_v * dcfy + ensz_v * dcfz)

    cfx = np.bincount(vortex_to_strip, weights=dcfx, minlength=nstrip)
    cfy = np.bincount(vortex_to_strip, weights=dcfy, minlength=nstrip)
    cfz = np.bincount(vortex_to_strip, weights=dcfz, minlength=nstrip)
    cmx = np.bincount(vortex_to_strip, weights=dcmx, minlength=nstrip)
    cmy = np.bincount(vortex_to_strip, weights=dcmy, minlength=nstrip)
    cmz = np.bincount(vortex_to_strip, weights=dcmz, minlength=nstrip)
    cnc = np.bincount(vortex_to_strip, weights=dcnc, minlength=nstrip)

    # Per-strip sums of the sensitivities.  ``_StripSummer`` accumulates in
    # vortex order exactly like ``np.add.at`` (bit-identical) but is faster.
    dcf_u = fgam_u * inv_sr
    sum_u = _StripSummer(vortex_to_strip, nstrip, numax)
    cfx_u = sum_u(dcf_u[0])
    cfy_u = sum_u(dcf_u[1])
    cfz_u = sum_u(dcf_u[2])
    cmx_u = sum_u((dcf_u[2] * r[1, np.newaxis, :] - dcf_u[1] * r[2, np.newaxis, :]) * inv_cr)
    cmy_u = sum_u((dcf_u[0] * r[2, np.newaxis, :] - dcf_u[2] * r[0, np.newaxis, :]) * inv_cr)
    cmz_u = sum_u((dcf_u[1] * r[0, np.newaxis, :] - dcf_u[0] * r[1, np.newaxis, :]) * inv_cr)
    # NOTE (kept as is): this sums over the sensitivity index n before the
    # strip reduction, so every column of CNC_U holds sum_n dCNC/du_n rather
    # than dCNC/du_n (AVL: CNC_U(J,N) += CR*(ENSY*DCFY_U(N) + ENSZ*DCFZ_U(N))).
    cnc_u = cr_v[np.newaxis, :] * np.sum(
        ensy_v[np.newaxis, :] * dcf_u[1] + ensz_v[np.newaxis, :] * dcf_u[2],
        axis=0,
    )
    cnc_u_acc = _broadcast_strip_sum(vortex_to_strip, nstrip, cnc_u[0], numax)

    cfx_d = cfy_d = cfz_d = cmx_d = cmy_d = cmz_d = None
    cnc_d_acc = None
    if ncontrol:
        dcf_d = fgam_d * inv_sr
        sum_d = _StripSummer(vortex_to_strip, nstrip, ncontrol)
        cfx_d = sum_d(dcf_d[0])
        cfy_d = sum_d(dcf_d[1])
        cfz_d = sum_d(dcf_d[2])
        cmx_d = sum_d((dcf_d[2] * r[1, np.newaxis, :] - dcf_d[1] * r[2, np.newaxis, :]) * inv_cr)
        cmy_d = sum_d((dcf_d[0] * r[2, np.newaxis, :] - dcf_d[2] * r[0, np.newaxis, :]) * inv_cr)
        cmz_d = sum_d((dcf_d[1] * r[0, np.newaxis, :] - dcf_d[0] * r[1, np.newaxis, :]) * inv_cr)
        # Same sum over the control index as CNC_U above (kept as is).
        cnc_d_v = cr_v[np.newaxis, :] * np.sum(
            ensy_v[np.newaxis, :] * dcf_d[1] + ensz_v[np.newaxis, :] * dcf_d[2],
            axis=0,
        )
        cnc_d_acc = _broadcast_strip_sum(vortex_to_strip, nstrip, cnc_d_v[0], ncontrol)

    if ncontrol:
        # hinge moments
        phinge_v = state.phinge[:, vortex_to_strip, :ncontrol]
        vhinge_v = state.vhinge[:, vortex_to_strip, :ncontrol]
        dfac = state.dcontrol[:nvor, :ncontrol] / (state.sref * state.cref)
        rh = state.rv[:, :nvor, np.newaxis] - phinge_v
        mh = _cross_axis0(rh, fgam[:, :, np.newaxis])
        mh_dot_vh = np.sum(mh * vhinge_v, axis=0)
        state.chinge[:ncontrol] = np.sum(mh_dot_vh * dfac, axis=0)
        mh_u = _cross_axis0(
            rh[:, np.newaxis, :, :],
            fgam_u[:, :, :, np.newaxis],
        )
        state.chinge_u[:ncontrol, :numax] = np.sum(
            np.einsum("kvnc,knc->vnc", mh_u, vhinge_v) * dfac[np.newaxis, :, :],
            axis=1,
        ).T
        mh_d = _cross_axis0(
            rh[:, np.newaxis, :, :],
            fgam_d[:, :, :, np.newaxis],
        )
        state.chinge_d[:ncontrol, :ncontrol] = np.sum(
            np.einsum("kvnc,knc->vnc", mh_d, vhinge_v) * dfac[np.newaxis, :, :],
            axis=1,
        ).T

    if cfx_d is None:
        cfx_d, cfy_d, cfz_d, cmx_d, cmy_d, cmz_d = (
            zeros_d(), zeros_d(), zeros_d(), zeros_d(), zeros_d(), zeros_d(),
        )
        cnc_d_acc = zeros_d()

    return (
        cfx, cfy, cfz, cmx, cmy, cmz,
        cfx_u, cfy_u, cfz_u, cmx_u, cmy_u, cmz_u,
        cfx_d, cfy_d, cfz_d, cmx_d, cmy_d, cmz_d,
    ), cnc, cnc_u_acc, cnc_d_acc


class _StripLoads:
    """Loads of the active strips, in body axes about the strip c/4 point.

    Forces are normalised by strip area, moments by strip area and chord.
    Component index first: ``cf[k]`` is a length-``m`` array over the active
    strips, ``cf_u[k]`` is ``(m, numax)`` and ``cf_d[k]`` is ``(m, ncontrol)``.
    """

    __slots__ = ("cf", "cm", "cnc", "cf_u", "cm_u", "cnc_u", "cf_d", "cm_d", "cnc_d")

    def __init__(self, strip_sums, act, numax, ncontrol):
        (
            (
                cfx, cfy, cfz, cmx, cmy, cmz,
                cfx_u, cfy_u, cfz_u, cmx_u, cmy_u, cmz_u,
                cfx_d, cfy_d, cfz_d, cmx_d, cmy_d, cmz_d,
            ),
            cnc, cnc_u, cnc_d,
        ) = strip_sums
        m = act.size
        self.cf = np.stack((cfx[act], cfy[act], cfz[act]))
        self.cm = np.stack((cmx[act], cmy[act], cmz[act]))
        self.cnc = cnc[act]
        self.cf_u = np.stack((cfx_u[act, :numax], cfy_u[act, :numax], cfz_u[act, :numax]))
        self.cm_u = np.stack((cmx_u[act, :numax], cmy_u[act, :numax], cmz_u[act, :numax]))
        self.cnc_u = cnc_u[act, :numax]
        if ncontrol:
            self.cf_d = np.stack((cfx_d[act, :ncontrol], cfy_d[act, :ncontrol], cfz_d[act, :ncontrol]))
            self.cm_d = np.stack((cmx_d[act, :ncontrol], cmy_d[act, :ncontrol], cmz_d[act, :ncontrol]))
            self.cnc_d = cnc_d[act, :ncontrol]
        else:
            self.cf_d = np.zeros((3, m, 0), dtype=np.float64)
            self.cm_d = np.zeros((3, m, 0), dtype=np.float64)
            self.cnc_d = np.zeros((m, 0), dtype=np.float64)


def _strip_lift_direction(vx, vy, vz, ensy, ensz):
    """Unit lift direction of each strip, ULIFT = UDRAG x SPN / |UDRAG x SPN|.

    The "spanwise" vector SPN = (0, ENSZ, -ENSY) is the cross product of the
    strip normal with the X chordline; the wind-axes stream vector UDRAG = VINF
    defines the drag direction (HHY 02272024: was stability axis). Where
    |UDRAG x SPN| = 0 the lift direction defaults to +Z.

    Returns ``(spn, raw, ulmag, ulift)``: SPN, the un-normalised cross
    product, its magnitude and the unit lift direction, each per strip.
    """
    zero = np.zeros_like(ensy)
    spn = (zero, ensz, -ensy)
    raw = _cross3(vx, vy, vz, spn[0], spn[1], spn[2])
    ulmag = np.sqrt(raw[0] * raw[0] + raw[1] * raw[1] + raw[2] * raw[2])
    degenerate = ulmag == 0.0
    safe = np.where(degenerate, 1.0, ulmag)
    ulift = (
        np.where(degenerate, 0.0, raw[0] / safe),
        np.where(degenerate, 0.0, raw[1] / safe),
        np.where(degenerate, 1.0, raw[2] / safe),
    )
    return spn, raw, ulmag, ulift


def _strip_lift_direction_u(spn, raw, ulmag, ulift, numax):
    """ULIFT_U, sensitivity of the unit lift direction, as ``(3, m, numax)``.

    UDRAG_U is the identity on the three freestream components and zero
    elsewhere. Same operations and order as AVL's SFFORC loops; zero where
    the lift direction is the +Z default.
    """
    m = ulmag.shape[0]
    ulift_u = np.empty((3, m, numax), dtype=np.float64)
    for k in range(3):
        ic, jc = ICRS[k], JCRS[k]
        for n in range(numax):
            ulift_u[k, :, n] = (
                (1.0 if ic == n else 0.0) * spn[jc] - (1.0 if jc == n else 0.0) * spn[ic]
            )
    degenerate = ulmag == 0.0
    safe = np.where(degenerate, 1.0, ulmag)[:, np.newaxis]
    ulmag_u = (
        raw[0][:, np.newaxis] * ulift_u[0]
        + raw[1][:, np.newaxis] * ulift_u[1]
        + raw[2][:, np.newaxis] * ulift_u[2]
    ) / safe
    for k in range(3):
        ulift_u[k] = (ulift_u[k] - (ulift[k][:, np.newaxis] * ulmag_u)) / safe
    ulift_u[:, degenerate, :] = 0.0
    return ulift_u


def _add_trailing_leg_forces(state, act, loads, rc4, cr, sr, ensy, ensz,
                             vinf_f, xyzref_f, wrot_f, gam, gam_u, numax):
    """Add h.v. forces from trailing legs lying on the wing surface (LTRFORCE).

    Each bound vortex has two chordwise trailing legs on the wing, from its
    endpoints RV1/RV2 to the strip trailing edge. The contribution of every
    leg is evaluated at once; ``np.add.at`` then adds them to the strip loads
    leg by leg, in AVL's (vortex, leg) order, so the sums round exactly as the
    original loop's. As in the code this was ported from, the legs add no
    control (``_d``) sensitivities.
    """
    vx, vy, vz = vinf_f
    xref, yref, zref = xyzref_f
    wx, wy, wz = wrot_f
    nvc = state.nvstrp[act].astype(np.intp)
    if not nvc.sum():
        return
    # One entry per leg, in AVL's loop order: strip, then chordwise vortex,
    # then the RV1 leg followed by the RV2 leg.
    first = state.ijfrst[act].astype(np.intp)
    vortex = np.concatenate([np.arange(f, f + n) for f, n in zip(first.tolist(), nvc.tolist())])
    i = np.repeat(vortex, 2)
    q = np.repeat(np.arange(act.size), 2 * nvc)
    leg1 = np.tile((True, False), vortex.size)     # leg from RV1, else from RV2
    strip = act[q]
    xte = np.where(
        leg1,
        state.rle1[0, strip] + state.chord1[strip],
        state.rle2[0, strip] + state.chord2[strip],
    )
    p = np.where(leg1, state.rv1[:3, i], state.rv2[:3, i])
    r = (0.5 * (p[0] + xte) - rc4[0][q], p[1] - rc4[1][q], p[2] - rc4[2][q])
    rrot = (0.5 * (p[0] + xte) - xref, p[1] - yref, p[2] - zref)
    gleg = (np.where(leg1, p[0] - xte, xte - p[0]), 0.0, 0.0)
    # set total effective velocity = freestream + rotation
    # (ignores h.v. induced contribution along trailing leg on wing)
    vrot = _cross3(rrot[0], rrot[1], rrot[2], wx, wy, wz)
    leg_veff = (vx + vrot[0], vy + vrot[1], vz + vrot[2])
    # VEFF_U columns: unit vectors for the freestream components,
    # RROT x e_k for the rotation components.
    leg_veff_u = (
        (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0),
        _cross3(rrot[0], rrot[1], rrot[2], 1.0, 0.0, 0.0),
        _cross3(rrot[0], rrot[1], rrot[2], 0.0, 1.0, 0.0),
        _cross3(rrot[0], rrot[1], rrot[2], 0.0, 0.0, 1.0),
    )
    f = _cross3(leg_veff[0], leg_veff[1], leg_veff[2], gleg[0], gleg[1], gleg[2])
    tg = 2.0 * gam[i]
    srq = sr[q]
    crq = cr[q]
    dcf = tuple((tg * f[k]) / srq for k in range(3))
    dcm = (
        ((dcf[2] * r[1]) - (dcf[1] * r[2])) / crq,
        ((dcf[0] * r[2]) - (dcf[2] * r[0])) / crq,
        ((dcf[1] * r[0]) - (dcf[0] * r[1])) / crq,
    )
    for k in range(3):
        np.add.at(loads.cf[k], q, dcf[k])
        np.add.at(loads.cm[k], q, dcm[k])
    np.add.at(loads.cnc, q, crq * (ensy[q] * dcf[1] + ensz[q] * dcf[2]))

    nleg = q.size
    dcf_u = np.empty((3, nleg, numax), dtype=np.float64)
    tgu = 2.0 * gam_u[i, :numax]
    for n in range(numax):
        vu = leg_veff_u[n]
        fu = _cross3(vu[0], vu[1], vu[2], gleg[0], gleg[1], gleg[2])
        for k in range(3):
            dcf_u[k, :, n] = (tgu[:, n] * f[k] + tg * fu[k]) / srq
    crq_c = crq[:, np.newaxis]
    r_c = tuple(c[:, np.newaxis] for c in r)
    dcm_u = (
        ((dcf_u[2] * r_c[1]) - (dcf_u[1] * r_c[2])) / crq_c,
        ((dcf_u[0] * r_c[2]) - (dcf_u[2] * r_c[0])) / crq_c,
        ((dcf_u[1] * r_c[0]) - (dcf_u[0] * r_c[1])) / crq_c,
    )
    for k in range(3):
        np.add.at(loads.cf_u[k], q, dcf_u[k])
        np.add.at(loads.cm_u[k], q, dcm_u[k])
    np.add.at(
        loads.cnc_u, q,
        crq_c * (ensy[q][:, np.newaxis] * dcf_u[1] + ensz[q][:, np.newaxis] * dcf_u[2]),
    )


def _add_viscous_strip_drag(state, act, vis, loads, rc4, cr, ensy, ensz, ulift, ulift_u,
                            vinf_f, xyzref_f, wrot_f, numax, ncontrol):
    """Drag terms due to viscous effects; CD from user-specified CD(CL) polar.

    ``vis`` selects the viscous strips among the active ones. Updates
    ``loads`` in place and returns the strip viscous drag CDV_LSTRP of the
    selected strips.
    """
    vx, vy, vz = vinf_f
    xref, yref, zref = xyzref_f
    wx, wy, wz = wrot_f
    nvis = vis.size
    cr = cr[vis][:, np.newaxis]
    ensy = ensy[vis][:, np.newaxis]
    ensz = ensz[vis][:, np.newaxis]
    ul = tuple(u[vis] for u in ulift)
    ul_u = ulift_u[:, vis, :]
    # The lift direction does not depend on the controls (ULIFT_D = 0).
    ul_d = np.zeros((3, max(1, ncontrol)), dtype=np.float64)
    cfx, cfy, cfz = loads.cf[:, vis]
    cfx_u, cfy_u, cfz_u = loads.cf_u[:, vis, :]
    cfx_d, cfy_d, cfz_d = loads.cf_d[:, vis, :]

    # Onset velocity at strip c/4 = freestream + rotation
    rrot0 = rc4[0][vis] - xref
    rrot1 = rc4[1][vis] - yref
    rrot2 = rc4[2][vis] - zref
    vrot0, vrot1, vrot2 = _cross3(rrot0, rrot1, rrot2, wx, wy, wz)
    ve = (vx + vrot0, vy + vrot1, vz + vrot2)
    veffmag = np.sqrt(ve[0] * ve[0] + ve[1] * ve[1] + ve[2] * ve[2])
    veff_u = np.zeros((3, nvis, numax), dtype=np.float64)
    veff_u[0, :, 0] = 1.0
    veff_u[1, :, 1] = 1.0
    veff_u[2, :, 2] = 1.0
    veff_u[1, :, 3] = rrot2
    veff_u[2, :, 3] = -rrot1
    veff_u[0, :, 4] = -rrot2
    veff_u[2, :, 4] = rrot0
    veff_u[0, :, 5] = rrot1
    veff_u[1, :, 5] = -rrot0
    veffmag_c = veffmag[:, np.newaxis]
    veffmag_u = (
        ve[0][:, np.newaxis] * veff_u[0]
        + ve[1][:, np.newaxis] * veff_u[1]
        + ve[2][:, np.newaxis] * veff_u[2]
    ) / veffmag_c

    clv = (ul[0] * cfx + ul[1] * cfy + ul[2] * cfz)
    clv_u = (
        ul[0][:, np.newaxis] * cfx_u + ul_u[0] * cfx[:, np.newaxis]
        + ul[1][:, np.newaxis] * cfy_u + ul_u[1] * cfy[:, np.newaxis]
        + ul[2][:, np.newaxis] * cfz_u + ul_u[2] * cfz[:, np.newaxis]
    )
    clv_d = (
        ul[0][:, np.newaxis] * cfx_d + ul_d[0] * cfx[:, np.newaxis]
        + ul[1][:, np.newaxis] * cfy_d + ul_d[1] * cfy[:, np.newaxis]
        + ul[2][:, np.newaxis] * cfz_d + ul_d[2] * cfz[:, np.newaxis]
    )
    # Get CD from CLCD function using strip CL as parameter (scalar polar)
    cdv = np.empty(nvis, dtype=np.float64)
    cdv_clv = np.empty(nvis, dtype=np.float64)
    clcd = state.clcd
    for q, strip in enumerate(act[vis].tolist()):
        cdv[q], cdv_clv[q] = cdcl(clcd[strip, :], float(clv[q]))

    # Strip viscous force contribution (per unit strip area)
    dcvf = [(ve[k] * veffmag * cdv) for k in range(3)]
    loads.cf[:, vis] = (cfx + dcvf[0], cfy + dcvf[1], cfz + dcvf[2])
    cdv_strip = (vx * dcvf[0] + vy * dcvf[1] + vz * dcvf[2])
    # d(CD)/d(CL) factor, (VEFF*|VEFF|*CDV_CLV) per component
    clv_factor = [(ve[k] * veffmag * cdv_clv)[:, np.newaxis] for k in range(3)]
    cdv_c = cdv[:, np.newaxis]
    dcvf_u = [
        (veff_u[k] * veffmag_c + ve[k][:, np.newaxis] * veffmag_u) * cdv_c
        + clv_factor[k] * clv_u
        for k in range(3)
    ]
    loads.cf_u[:, vis, :] = (cfx_u + dcvf_u[0], cfy_u + dcvf_u[1], cfz_u + dcvf_u[2])
    loads.cnc_u[vis] = loads.cnc_u[vis] + cr * (ensy * dcvf_u[1] + ensz * dcvf_u[2])
    dcvf_d = [clv_factor[k] * clv_d for k in range(3)]
    loads.cf_d[:, vis, :] = (cfx_d + dcvf_d[0], cfy_d + dcvf_d[1], cfz_d + dcvf_d[2])
    if ncontrol:
        loads.cnc_d[vis] = loads.cnc_d[vis] + cr * (ensy * dcvf_d[1] + ensz * dcvf_d[2])
    return cdv_strip


def _clmax_derivative_correction(state, strip, ulift, ulift_u, numax, ncontrol):
    """Prepare the CLmax clipping derivatives of one strip (OpenAVL extension).

    For s = Clmax / Cl, d(s F) = s (dF - F dCl / Cl).  Form the expression in
    parentheses here, while the local lift direction and all uncapped strip
    loads are still available. The per-surface pass in ``sfforc`` then
    applies s to both loads and derivatives.
    """
    ulift_d = np.zeros((3, max(1, ncontrol)), dtype=np.float64)
    local_cl = state.cl_lstrp[strip]
    force = state.cfstrp[:, strip]
    for suffix, count, lift_deriv in (
        ("u", numax, ulift_u),
        ("d", ncontrol, ulift_d[:, :ncontrol]),
    ):
        if not count:
            continue
        force_deriv = getattr(state, f"cfst_{suffix}")[:, strip, :count]
        relative_cl_deriv = (
            ulift @ force_deriv + force @ lift_deriv
        ) / local_cl
        for deriv_name, load_name in (
            ("cnc", "cnc"), ("cdst", "cdstrp"),
            ("cyst", "cystrp"), ("clst", "clstrp"),
        ):
            getattr(state, f"{deriv_name}_{suffix}")[strip, :count] -= (
                getattr(state, load_name)[strip] * relative_cl_deriv
            )
        force_deriv -= force[:, None] * relative_cl_deriv
        getattr(state, f"cmst_{suffix}")[:, strip, :count] -= (
            state.cmstrp[:, strip, None] * relative_cl_deriv
        )


def _inverse_or_one(x: np.ndarray) -> np.ndarray:
    """``1/x`` elementwise, with 1 where ``x == 0`` (AVL's ``VSQI``/``VPSQI``)."""
    zero = x == 0.0
    return np.where(zero, 1.0, 1.0 / np.where(zero, 1.0, x))


_MISSING = object()


def _attr_or_zeros(state: Any, name: str, shape: tuple[int, ...]) -> Any:
    """Return ``state.<name>``, building a zero array only if the attribute is absent."""
    arr = getattr(state, name, _MISSING)
    if arr is _MISSING:
        arr = np.zeros(shape, dtype=np.float64)
    return arr


def _zero_strip(state, j, nvc, i1, numax, ncontrol):
    """Zero force outputs for an inactive strip."""
    state.cnc[j] = 0.0
    state.cdstrp[j] = 0.0
    state.cystrp[j] = 0.0
    state.clstrp[j] = 0.0
    state.cdst_a[j] = 0.0
    state.cyst_a[j] = 0.0
    state.clst_a[j] = 0.0
    state.cdv_lstrp[j] = 0.0
    state.cl_lstrp[j] = 0.0
    state.cd_lstrp[j] = 0.0
    state.cmc4_lstrp[j] = 0.0
    state.ca_lstrp[j] = 0.0
    state.cn_lstrp[j] = 0.0
    state.clt_lstrp[j] = 0.0
    state.cla_lstrp[j] = 0.0
    state.cmle_lstrp[j] = 0.0
    state.cf_lstrp[:, j] = 0.0
    state.cm_lstrp[:, j] = 0.0
    state.cfstrp[:, j] = 0.0
    state.cmstrp[:, j] = 0.0
    state.cnc_u[j, :numax] = 0.0
    state.cdst_u[j, :numax] = 0.0
    state.cyst_u[j, :numax] = 0.0
    state.clst_u[j, :numax] = 0.0
    state.cfst_u[:, j, :numax] = 0.0
    state.cmst_u[:, j, :numax] = 0.0
    if ncontrol:
        state.cnc_d[j, :ncontrol] = 0.0
        state.cdst_d[j, :ncontrol] = 0.0
        state.cyst_d[j, :ncontrol] = 0.0
        state.clst_d[j, :ncontrol] = 0.0
        state.cfst_d[:, j, :ncontrol] = 0.0
        state.cmst_d[:, j, :ncontrol] = 0.0
    for chord_idx in range(nvc):
        i = i1 + chord_idx
        state.dcp[i] = 0.0
        state.dcp_u[i, :numax] = 0.0
        if ncontrol:
            state.dcp_d[i, :ncontrol] = 0.0


def sfforc(state: Any) -> Any:
    """Integrate strip/surface forces from bound vorticity (SFFORC).

    Purpose: calculate forces on the configuration by vortex, strip, and
    surface. Integrate strip-wise, surface-wise, and into totals.

    Output includes DCP loadings, strip/surface coefficients, span loading
    (CNC), and hinge moments.

    When ``state.clmax_surf[isurf] > 0``, strip forces on that surface are
    scaled so local ``cl_lstrp`` does not exceed the limit (OpenAVL extension).
    Strip sensitivities include the derivative of the clipping scale so
    Newton trim Jacobians and stability derivatives stay consistent with
    the clipped residuals. The clip is
    non-smooth at the boundary (kink in the derivative).
    """
    numax    = state.numax
    ncontrol = state.ncontrol
    sina     = (np.sin((state.alfa)))
    cosa     = (np.cos((state.alfa)))
    ltrforce = bool(getattr(state, "ltrforce", False))
    lnfld_wv = bool(getattr(state, "lnfld_wv", False))
    lfload = getattr(state, "lfload", _MISSING)
    if lfload is _MISSING:
        lfload = np.ones(state.nsurf, dtype=bool)
    lfload = np.asarray(lfload, dtype=bool)

    # Zero fallbacks are only built for minimal hand-made states that lack
    # these fields (AVLState always has them); never allocate them eagerly.
    nvor = state.nvor
    nc1 = max(1, ncontrol)
    gam   = state.gam.reshape(-1) if state.gam.ndim > 1 else state.gam
    gam_u = state.gam_u
    gam_d = _attr_or_zeros(state, "gam_d", (nvor, nc1))
    vv    = _attr_or_zeros(state, "vv", (3, nvor))
    vv_u  = _attr_or_zeros(state, "vv_u", (3, nvor, numax))
    vv_d  = _attr_or_zeros(state, "vv_d", (3, nvor, nc1))
    wv    = _attr_or_zeros(state, "wv", (3, nvor))
    wv_u  = _attr_or_zeros(state, "wv_u", (3, nvor, numax))
    wv_d  = _attr_or_zeros(state, "wv_d", (3, nvor, nc1))
    env_d = _attr_or_zeros(state, "env_d", (3, nvor, nc1))

    strip_sums = _accumulate_all_vortex_forces(
        state, numax, ncontrol,
        gam, gam_u, gam_d,
        vv, vv_u, vv_d,
        wv, wv_u, wv_d,
        env_d, lnfld_wv,
    )

    # Integrate the forces strip-wise, then surface-wise and into totals.
    # All strip quantities below are arrays over the active strips ``act``;
    # strips that are switched off or have zero width are zeroed instead.
    nstrip = state.nstrip
    active = ~np.asarray(state.lstripoff[:nstrip], dtype=bool) & (state.wstrip[:nstrip] != 0.0)
    act = np.flatnonzero(active)
    for strip in np.flatnonzero(~active).tolist():
        _zero_strip(
            state, strip, int(state.nvstrp[strip]), int(state.ijfrst[strip]), numax, ncontrol
        )

    vinf_f = tuple(float(v) for v in state.vinf[:3])
    xyzref_f = tuple(float(v) for v in state.xyzref[:3])
    wrot_f = tuple(float(v) for v in state.wrot[:3])
    vx, vy, vz = vinf_f
    xref, yref, zref = xyzref_f
    wx, wy, wz = wrot_f

    # Strip forces normalized to strip reference quantities
    cr = state.chord[act]
    sr = cr * state.wstrip[act]
    ensy = state.ensy[act]
    ensz = state.ensz[act]
    # Use the strip 1/4 chord location for strip moments
    rc4 = (state.rle[0, act] + (0.25 * cr), state.rle[1, act], state.rle[2, act])

    # Local strip lift and drag directions (drag = wind axes stream vector)
    spn, ulift_raw, ulmag, ulift = _strip_lift_direction(vx, vy, vz, ensy, ensz)
    loads = _StripLoads(strip_sums, act, numax, ncontrol)

    # Add h.v. forces from trailing legs lying on the wing surface
    if ltrforce:
        _add_trailing_leg_forces(
            state, act, loads, rc4, cr, sr, ensy, ensz,
            vinf_f, xyzref_f, wrot_f, gam, gam_u, numax,
        )

    # ULIFT_U is only needed by the viscous and CLmax branches.
    vis = (
        np.flatnonzero(state.lviscstrp[act]) if state.lvisc
        else np.zeros(0, dtype=np.intp)
    )
    clmax = state.clmax_surf[state.lssurf[act]]
    ulift_u = None
    if vis.size or np.any(clmax > 0.0):
        ulift_u = _strip_lift_direction_u(spn, ulift_raw, ulmag, ulift, numax)

    # Drag terms due to viscous effects; CD from user-specified CD(CL) polar
    cdv_lstrp = np.zeros(act.size, dtype=np.float64)
    if vis.size:
        cdv_lstrp[vis] = _add_viscous_strip_drag(
            state, act, vis, loads, rc4, cr, ensy, ensz, ulift, ulift_u,
            vinf_f, xyzref_f, wrot_f, numax, ncontrol,
        )
    state.cdv_lstrp[act] = cdv_lstrp

    # At this point strip forces are in body axes at c/4, normalized by area/chord
    cfx, cfy, cfz = loads.cf
    cmx, cmy, cmz = loads.cm
    cf_u, cm_u = loads.cf_u, loads.cm_u
    cf_d, cm_d = loads.cf_d, loads.cm_d
    state.cnc[act] = loads.cnc
    state.cnc_u[act, :numax] = loads.cnc_u
    if ncontrol:
        state.cnc_d[act, :ncontrol] = loads.cnc_d
    state.cf_lstrp[:, act] = loads.cf
    state.cm_lstrp[:, act] = loads.cm
    state.cfstrp[:, act] = loads.cf
    # Transform strip body axes forces into stability axes
    state.cdstrp[act] = (cfx * cosa + cfz * sina)
    state.cystrp[act] = cfy
    state.clstrp[act] = (-cfx * sina + cfz * cosa)
    state.cdst_a[act] = (-cfx * sina + cfz * cosa)
    state.cyst_a[act] = 0.0
    state.clst_a[act] = (-cfx * cosa - cfz * sina)
    state.cdst_u[act, :numax] = cf_u[0] * cosa + cf_u[2] * sina
    state.cyst_u[act, :numax] = cf_u[1]
    state.clst_u[act, :numax] = -cf_u[0] * sina + cf_u[2] * cosa
    state.cfst_u[:, act, :numax] = cf_u
    if ncontrol:
        state.cdst_d[act, :ncontrol] = cf_d[0] * cosa + cf_d[2] * sina
        state.cyst_d[act, :ncontrol] = cf_d[1]
        state.clst_d[act, :ncontrol] = -cf_d[0] * sina + cf_d[2] * cosa
        state.cfst_d[:, act, :ncontrol] = cf_d

    # vector from chord c/4 reference point to case reference point XYZREF
    rref = (rc4[0] - xref, rc4[1] - yref, rc4[2] - zref)
    # Strip moments in body axes about XYZREF, normalized by strip area/chord
    state.cmstrp[0, act] = (cmx + ((cfz * rref[1]) - (cfy * rref[2])) / cr)
    state.cmstrp[1, act] = (cmy + ((cfx * rref[2]) - (cfz * rref[0])) / cr)
    state.cmstrp[2, act] = (cmz + ((cfy * rref[0]) - (cfx * rref[1])) / cr)
    for deriv, cf_s, cm_s in (("u", cf_u, cm_u), ("d", cf_d, cm_d)):
        count = cf_s.shape[2]
        if not count:
            continue
        r = tuple(c[:, np.newaxis] for c in rref)
        cr_c = cr[:, np.newaxis]
        cmst_s = getattr(state, f"cmst_{deriv}")
        cmst_s[0, act, :count] = cm_s[0] + ((cf_s[2] * r[1]) - (cf_s[1] * r[2])) / cr_c
        cmst_s[1, act, :count] = cm_s[1] + ((cf_s[0] * r[2]) - (cf_s[2] * r[0])) / cr_c
        cmst_s[2, act, :count] = cm_s[2] + ((cf_s[1] * r[0]) - (cf_s[0] * r[1])) / cr_c

    cl_lstrp = (ulift[0] * cfx + ulift[1] * cfy + ulift[2] * cfz)
    state.cl_lstrp[act] = cl_lstrp
    state.cd_lstrp[act] = (vx * cfx + vy * cfy + vz * cfz)
    state.cmc4_lstrp[act] = (ensz * cmy - ensy * cmz)
    # CN,CA forces rotated to be in and normal to strip incidence (HHY bugfix 01102024)
    caxl0 = cfx
    cnrm0 = (ensy * cfy + ensz * cfz)
    sinainc = np.sin(state.ainc[act])
    cosainc = np.cos(state.ainc[act])
    state.ca_lstrp[act] = (caxl0 * cosainc - cnrm0 * sinainc)
    state.cn_lstrp[act] = (cnrm0 * cosainc + caxl0 * sinainc)

    # set total effective velocity = freestream + rotation
    rr0 = state.xsref[act] - xref
    rr1 = state.ysref[act] - yref
    rr2 = state.zsref[act] - zref
    vr0, vr1, vr2 = _cross3(rr0, rr1, rr2, wx, wy, wz)
    ve0 = vx + vr0
    ve1 = vy + vr1
    ve2 = vz + vr2
    vsq = ve0 * ve0 + ve1 * ve1 + ve2 * ve2
    # spanwise and perpendicular velocity components
    es0, es1, es2 = state.ess[0, act], state.ess[1, act], state.ess[2, act]
    vspan = (ve0 * es0 + ve1 * es1 + ve2 * es2)
    vp0 = ve0 - es0 * vspan
    vp1 = ve1 - es1 * vspan
    vp2 = ve2 - es2 * vspan
    vpsq = ((vp0 * vp0) + (vp1 * vp1) + (vp2 * vp2))
    state.clt_lstrp[act] = (cl_lstrp * _inverse_or_one(vpsq))
    state.cla_lstrp[act] = (cl_lstrp * _inverse_or_one(vsq))

    # Moment about strip LE midpoint in direction of LE segment
    rle = (rc4[0] - state.rle[0, act], rc4[1] - state.rle[1, act], rc4[2] - state.rle[2, act])
    dle = state.rle2[:3, act] - state.rle1[:3, act]
    mirrored = state.imags[state.lssurf[act]] < 0
    delx, dely, delz = np.where(mirrored, -dle, dle)
    dmag = np.sqrt((delx * delx) + (dely * dely) + (delz * delz))
    has_le = dmag != 0.0
    dmag = np.where(has_le, dmag, 1.0)
    state.cmle_lstrp[act] = np.where(
        has_le,
        delx / dmag * (cmx + ((cfz * rle[1]) - (cfy * rle[2])) / cr)
        + dely / dmag * (cmy + ((cfx * rle[2]) - (cfz * rle[0])) / cr)
        + delz / dmag * (cmz + ((cfy * rle[0]) - (cfx * rle[1])) / cr),
        0.0,
    )

    # CLmax clipping derivatives, strip by strip, over the strips that clip
    for q in np.flatnonzero((clmax > 0.0) & (cl_lstrp > clmax)).tolist():
        _clmax_derivative_correction(
            state, int(act[q]),
            np.array((ulift[0][q], ulift[1][q], ulift[2][q])),
            np.ascontiguousarray(ulift_u[:, q, :]),
            numax, ncontrol,
        )

    # --- Apply per-surface CLmax capping (OpenAVL extension) ---
    for isurf in range(state.nsurf):
        clmax = state.clmax_surf[isurf]
        if clmax <= 0.0:
            continue
        j0 = int(state.jfrst[isurf])
        nj = int(state.nj[isurf])
        js = np.arange(j0, j0 + nj, dtype=np.intp)
        exceeded = state.cl_lstrp[js] > clmax
        if not np.any(exceeded):
            continue
        warned = getattr(state, "_clmax_clip_warned", None)
        model = getattr(state, "model", None)
        if model is not None:
            model_idx = solver_surface_index(model, isurf)
            surf_name = (
                model.surfaces[model_idx].name
                if 0 <= model_idx < len(model.surfaces)
                else solver_surface_name(model, isurf)
            )
            warn_key = model_idx if model_idx >= 0 else isurf
        else:
            surf_name = f"surface {isurf + 1}"
            warn_key = isurf
        if warned is None or warn_key not in warned:
            print(
                f"Local Cl > Clmax ({clmax:.3f}) on {surf_name}: "
                "Clipping sectional lift force"
            )
            if warned is not None:
                warned.add(warn_key)
        scale = np.ones(js.size, dtype=np.float64)
        scale[exceeded] = clmax / state.cl_lstrp[js[exceeded]]
        state.cfstrp[:, js] *= scale[np.newaxis, :]
        state.clstrp[js] *= scale
        state.cdstrp[js] *= scale
        state.cystrp[js] *= scale
        state.cmstrp[:, js] *= scale[np.newaxis, :]
        state.cl_lstrp[js] = np.minimum(state.cl_lstrp[js], clmax)
        state.cd_lstrp[js] *= scale
        state.cn_lstrp[js] *= scale
        state.ca_lstrp[js] *= scale
        state.clt_lstrp[js] *= scale
        state.cla_lstrp[js] *= scale
        # Complete the product rule prepared in the strip loop above.
        # Explicit alpha derivatives only rotate the output axes; the local
        # lift direction depends on vinf, whose derivatives are in *_u.
        state.cnc[js] *= scale
        state.cdv_lstrp[js] *= scale
        state.cdst_a[js] *= scale
        state.cyst_a[js] *= scale
        state.clst_a[js] *= scale
        state.cnc_u[js, :numax] *= scale[:, np.newaxis]
        state.cdst_u[js, :numax] *= scale[:, np.newaxis]
        state.cyst_u[js, :numax] *= scale[:, np.newaxis]
        state.clst_u[js, :numax] *= scale[:, np.newaxis]
        state.cfst_u[:, js, :numax] *= scale[np.newaxis, :, np.newaxis]
        state.cmst_u[:, js, :numax] *= scale[np.newaxis, :, np.newaxis]
        if ncontrol:
            state.cnc_d[js, :ncontrol] *= scale[:, np.newaxis]
            state.cdst_d[js, :ncontrol] *= scale[:, np.newaxis]
            state.cyst_d[js, :ncontrol] *= scale[:, np.newaxis]
            state.clst_d[js, :ncontrol] *= scale[:, np.newaxis]
            state.cfst_d[:, js, :ncontrol] *= scale[np.newaxis, :, np.newaxis]
            state.cmst_d[:, js, :ncontrol] *= scale[np.newaxis, :, np.newaxis]

    # Surface forces and moments summed from strip forces.  Each sum below
    # reduces the same per-strip products as AVL, grouped so that one NumPy
    # reduction per group replaces one call per coefficient: rows of a
    # C-ordered (q, nj) array reduce exactly like separate 1-D sums, and the
    # leading axis of a C-ordered (nj, ...) array reduces strip by strip.
    for isurf in range(state.nsurf):
        j0 = int(state.jfrst[isurf])
        nj = int(state.nj[isurf])
        js = slice(j0, j0 + nj)
        sr = state.chord[js] * state.wstrip[js]
        cr = state.chord[js]
        scale_sref = sr / state.sref
        scale_ssurf = sr / state.ssurf[isurf]
        scale_cm_b = scale_sref * (cr / state.bref)
        scale_cm_c = scale_sref * (cr / state.cref)
        scale_cm_loc = scale_ssurf * (cr / state.cavesurf[isurf]) / cr

        # Strip c/4 points relative to the surface root LE (local moments)
        rc4s = (state.rle[0, js] + 0.25 * state.chord[js], state.rle[1, js], state.rle[2, js])
        rroot = state.rle1[:, j0] if state.imags[isurf] >= 0 else state.rle2[:, j0]
        r = tuple(rc4s[k] - rroot[k] for k in range(3))
        cf_l = state.cf_lstrp[:, js]
        cm_l = state.cm_lstrp[:, js]

        sums = np.sum(np.stack((
            sr * state.ensy[js],
            sr * state.ensz[js],
            state.cdstrp[js] * scale_sref,
            state.cystrp[js] * scale_sref,
            state.clstrp[js] * scale_sref,
            state.cmstrp[0, js] * scale_cm_b,
            state.cmstrp[1, js] * scale_cm_c,
            state.cmstrp[2, js] * scale_cm_b,
            state.cdv_lstrp[js] * scale_sref,
            state.cdst_a[js] * scale_sref,
            state.cyst_a[js] * scale_sref,
            state.clst_a[js] * scale_sref,
            *(
                scale_cm_loc
                * (cm_l[k] + cf_l[JCRS[k]] * r[ICRS[k]] - cf_l[ICRS[k]] * r[JCRS[k]])
                for k in range(3)
            ),
        )), axis=1)
        enave = np.array((0.0, sums[0], sums[1]))
        (
            state.cdsurf[isurf], state.cysurf[isurf], state.clsurf[isurf],
            state.cmsurf[0, isurf], state.cmsurf[1, isurf], state.cmsurf[2, isurf],
            state.cdvsurf[isurf], state.cds_a[isurf], state.cys_a[isurf], state.cls_a[isurf],
            state.cm_lsrf[0, isurf], state.cm_lsrf[1, isurf], state.cm_lsrf[2, isurf],
        ) = sums[2:].tolist()

        forces = np.sum(np.stack((
            *(state.cfstrp[k, js] * scale_sref for k in range(3)),
            *(cf_l[k] * scale_ssurf for k in range(3)),
        ), axis=1), axis=0)
        state.cfsurf[:, isurf] = forces[:3]
        state.cf_lsrf[:, isurf] = forces[3:]

        s_sref = scale_sref[:, np.newaxis]
        s_b = scale_cm_b[:, np.newaxis]
        s_c = scale_cm_c[:, np.newaxis]
        cfst_u = state.cfst_u[:, js, :numax]
        cmst_u = state.cmst_u[:, js, :numax]
        sums_u = np.sum(np.stack((
            state.cdst_u[js, :numax] * s_sref,
            state.cyst_u[js, :numax] * s_sref,
            state.clst_u[js, :numax] * s_sref,
            cfst_u[0] * s_sref, cfst_u[1] * s_sref, cfst_u[2] * s_sref,
            cmst_u[0] * s_b, cmst_u[1] * s_c, cmst_u[2] * s_b,
        ), axis=1), axis=0)
        state.cds_u[isurf, :numax] = sums_u[0]
        state.cys_u[isurf, :numax] = sums_u[1]
        state.cls_u[isurf, :numax] = sums_u[2]
        state.cfs_u[:, isurf, :numax] = sums_u[3:6]
        state.cms_u[:, isurf, :numax] = sums_u[6:9]
        if ncontrol:
            # Control sums stay one reduction per coefficient: with a single
            # control the (nj, 1) operands reduce pairwise, not strip by strip,
            # so grouping them would change the rounding.
            jsi = np.arange(j0, j0 + nj, dtype=np.intp)
            state.cds_d[isurf, :ncontrol] = np.sum(state.cdst_d[jsi, :ncontrol] * s_sref, axis=0)
            state.cys_d[isurf, :ncontrol] = np.sum(state.cyst_d[jsi, :ncontrol] * s_sref, axis=0)
            state.cls_d[isurf, :ncontrol] = np.sum(state.clst_d[jsi, :ncontrol] * s_sref, axis=0)
            state.cfs_d[:, isurf, :ncontrol] = np.sum(
                state.cfst_d[:, jsi, :ncontrol] * scale_sref[np.newaxis, :, np.newaxis], axis=1
            )
            state.cms_d[0, isurf, :ncontrol] = np.sum(state.cmst_d[0, jsi, :ncontrol] * s_b, axis=0)
            state.cms_d[1, isurf, :ncontrol] = np.sum(state.cmst_d[1, jsi, :ncontrol] * s_c, axis=0)
            state.cms_d[2, isurf, :ncontrol] = np.sum(state.cmst_d[2, jsi, :ncontrol] * s_b, axis=0)

        # To define surface CL and CD we need local lift and drag directions
        # Define a "spanwise" vector: cross product of average surface normal with chordline
        enave[0] = (enave[0] / state.ssurf[isurf])
        enave[1] = (enave[1] / state.ssurf[isurf])
        enave[2] = (enave[2] / state.ssurf[isurf])
        enmag = (np.sqrt(float(dot(enave, enave))))
        if enmag == 0.0:
            enave[2] = 1.0
        else:
            enave[:] = (enave / enmag)
        spn = np.array([0.0, enave[2], (-enave[1])], dtype=np.float64)
        # Wind axes stream vector defines drag direction
        udrag = state.vinf.copy()
        ulift = np.array(_cross3(udrag[0], udrag[1], udrag[2], spn[0], spn[1], spn[2]))
        ulmag = np.linalg.norm(ulift)
        if ulmag == 0.0:
            ulift[2] = 1.0
        else:
            ulift[:] = (ulift / ulmag)
        # Lift direction is vector product of "stream" and spanwise vector
        state.cl_lsrf[isurf] = (dot(ulift, state.cf_lsrf[:, isurf]))
        state.cd_lsrf[isurf] = (dot(udrag, state.cf_lsrf[:, isurf]))

        # Total forces summed from surface forces (normalized to SREF, CREF, BREF)
        if lfload[isurf]:
            state.cftot[:] = (state.cftot + state.cfsurf[:, isurf])
            state.cdtot = (state.cdtot + state.cdsurf[isurf])
            state.cytot = (state.cytot + state.cysurf[isurf])
            state.cltot = (state.cltot + state.clsurf[isurf])
            state.cdvtot = (state.cdvtot + state.cdvsurf[isurf])
            state.cmtot[:] = (state.cmtot + state.cmsurf[:, isurf])
            state.cdtot_a = (state.cdtot_a + state.cds_a[isurf])
            state.cytot_a = (getattr(state, "cytot_a", 0.0) + state.cys_a[isurf])
            state.cltot_a = (state.cltot_a + state.cls_a[isurf])
            state.cdtot_u[:numax] += state.cds_u[isurf, :numax]
            state.cytot_u[:numax] += state.cys_u[isurf, :numax]
            state.cltot_u[:numax] += state.cls_u[isurf, :numax]
            state.cftot_u[:, :numax] += state.cfs_u[:, isurf, :numax]
            state.cmtot_u[:, :numax] += state.cms_u[:, isurf, :numax]
            if ncontrol:
                state.cdtot_d[:ncontrol] += state.cds_d[isurf, :ncontrol]
                state.cytot_d[:ncontrol] += state.cys_d[isurf, :ncontrol]
                state.cltot_d[:ncontrol] += state.cls_d[isurf, :ncontrol]
                state.cftot_d[:, :ncontrol] += state.cfs_d[:, isurf, :ncontrol]
                state.cmtot_d[:, :ncontrol] += state.cms_d[:, isurf, :ncontrol]

    return state


def aero(state: Any) -> Any:
    """Compute total aerodynamic coefficients (AERO).

    Calculate forces: inviscid from horseshoe vortices, inviscid from bodies,
    viscous from drag polars, and far-field Trefftz-plane contributions.
    """
    numax = state.numax
    ncontrol = state.ncontrol

    # Zero force/moment totals and derivative accumulators.
    state.cdtot = 0.0
    state.cytot = 0.0
    state.cltot = 0.0
    state.cftot[:] = 0.0
    state.cmtot[:] = 0.0
    state.cdvtot = 0.0
    state.cdtot_a = 0.0
    state.cltot_a = 0.0
    state.chinge[:ncontrol] = 0.0
    state.cdtot_u[:] = 0.0
    state.cytot_u[:] = 0.0
    state.cltot_u[:] = 0.0
    state.cftot_u[:] = 0.0
    state.cmtot_u[:] = 0.0
    if ncontrol:
        state.chinge_u[:ncontrol, :numax] = 0.0
        state.cdtot_d[:ncontrol] = 0.0
        state.cytot_d[:ncontrol] = 0.0
        state.cltot_d[:ncontrol] = 0.0
        state.cftot_d[:, :ncontrol] = 0.0
        state.cmtot_d[:, :ncontrol] = 0.0
        state.chinge_d[:ncontrol, :ncontrol] = 0.0

    # Evaluate forces on surface, bodies and Trefftz plane
    sfforc(state)
    bdforc(state)
    tpforc(state)

    # If case is XZ symmetric (IYSYM=1), add contributions from images,
    # zero out the asymmetric forces and double the symmetric ones
    if state.iysym == 1:
        state.cdtot = 2.0 * state.cdtot
        state.cytot = 0.0
        state.cltot = 2.0 * state.cltot

        state.cftot[0] = 2.0 * state.cftot[0]
        state.cftot[1] = 0.0
        state.cftot[2] = 2.0 * state.cftot[2]

        state.cmtot[0] = 0.0
        state.cmtot[1] = 2.0 * state.cmtot[1]
        state.cmtot[2] = 0.0

        state.cdvtot = 2.0 * state.cdvtot
        state.cdtot_a = 2.0 * state.cdtot_a
        state.cltot_a = 2.0 * state.cltot_a
        state.cdtot_u[:] = 2.0 * state.cdtot_u
        state.cytot_u[:] = 0.0
        state.cltot_u[:] = 2.0 * state.cltot_u

        state.cftot_u[0, :] = 2.0 * state.cftot_u[0, :]
        state.cftot_u[1, :] = 0.0
        state.cftot_u[2, :] = 2.0 * state.cftot_u[2, :]

        state.cmtot_u[0, :] = 0.0
        state.cmtot_u[1, :] = 2.0 * state.cmtot_u[1, :]
        state.cmtot_u[2, :] = 0.0

        if ncontrol:
            state.cdtot_d[:] = 2.0 * state.cdtot_d
            state.cytot_d[:] = 0.0
            state.cltot_d[:] = 2.0 * state.cltot_d
            state.cftot_d[0, :] = 2.0 * state.cftot_d[0, :]
            state.cftot_d[1, :] = 0.0
            state.cftot_d[2, :] = 2.0 * state.cftot_d[2, :]
            state.cmtot_d[0, :] = 0.0
            state.cmtot_d[1, :] = 2.0 * state.cmtot_d[1, :]
            state.cmtot_d[2, :] = 0.0

    # add baseline reference CD to totals; force in direction of freestream
    vsq = (
        state.vinf[0] * state.vinf[0]
        + state.vinf[1] * state.vinf[1]
        + state.vinf[2] * state.vinf[2]
    )
    vmag = np.sqrt(float(vsq))
    state.cdvtot = state.cdvtot + state.cdref * vsq
    state.cdtot = state.cdtot + state.cdref * vsq
    state.cytot = state.cytot + state.cdref * (state.vinf[1] * vmag)
    for l in range(3):
        state.cftot[l] = state.cftot[l] + state.cdref * (state.vinf[l] * vmag)
        state.cftot_u[l, l] = state.cftot_u[l, l] + state.cdref * vmag
    for iu in range(3):
        state.cdtot_u[iu] = state.cdtot_u[iu] + state.cdref * (2.0 * state.vinf[iu])
        # Deliberate departure from AVL: aero.f adds CDREF*VINF(2)*VMAG to
        # CYTOT but omits the matching CYTOT_U term, leaving CY_b short by
        # CDREF. Differentiate it here the same way as CFTOT_U below.
        state.cytot_u[iu] = (
            state.cytot_u[iu]
            + state.cdref * (state.vinf[1] * state.vinf[iu]) / vmag
        )
        if iu == 1:
            state.cytot_u[iu] = state.cytot_u[iu] + state.cdref * vmag
        for l in range(3):
            state.cftot_u[l, iu] = (
                state.cftot_u[l, iu]
                + state.cdref * (state.vinf[l] * state.vinf[iu]) / vmag
            )
    return state
