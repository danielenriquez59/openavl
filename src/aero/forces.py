"""Aerodynamic force integration (port of aero.f: VINFAB, AERO, SFFORC, BDFORC).

Computes inviscid forces from horseshoe vortices, inviscid body source forces,
viscous strip drag from CD(CL) polars, and far-field Trefftz-plane contributions.
"""

from __future__ import annotations

from typing import Any

import math
import numpy as np

from openavl.aero.aic import dot
from openavl.aero.cdcl import cdcl
from openavl.aero.trefftz import tpforc
from openavl.geom.geometry import solver_surface_index, solver_surface_name

# Fortran ICRS/JCRS indexing for explicit cross-product components.
ICRS = (1, 2, 0)
JCRS = (2, 0, 1)


def _gam_scalar(gam: np.ndarray, i: int) -> np.float64:
    """Return circulation at vortex ``i`` (supports 1D or legacy 2D storage)."""
    if gam.ndim == 1:
        return (gam[i])
    return (gam[i, 0])


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


def _get_sfforc_work(state: Any, numax: int, ncontrol: int) -> dict[str, np.ndarray]:
    """Return strip-force work buffers from state, or local fallbacks for minimal states."""
    nc = max(1, ncontrol)

    def _buf(name: str, shape: tuple[int, ...]) -> np.ndarray:
        arr = getattr(state, name, None)
        if arr is None or arr.shape != shape:
            arr = np.zeros(shape, dtype=np.float64)
            setattr(state, name, arr)
        return arr

    # Only the buffers sfforc still writes into: strip loads and their
    # sensitivities are now held as Python floats / row views of the per-call
    # strip sums, so the remaining ``sfforc_*`` fields on AVLState are unused.
    # ULIFT_D stays all-zero (the lift direction does not depend on controls).
    return {
        "ulift": _buf("sfforc_ulift", (3,)),
        "ulift_u": _buf("sfforc_ulift_u", (3, numax)),
        "ulift_d": _buf("sfforc_ulift_d", (3, nc))[:, :nc],
        "ulmag_u": _buf("sfforc_ulmag_u", (numax,)),
        "veff_u": _buf("sfforc_veff_u", (3, numax)),
        "veffmag_u": _buf("sfforc_veffmag_u", (numax,)),
    }


def _fill_ulift_u(
    ulift_u: np.ndarray,
    ulmag_u: np.ndarray,
    numax: int,
    spn: tuple[float, float, float],
    ulift_raw: tuple[float, float, float] | None,
    ulift: tuple[float, float, float],
    ulmag: float,
) -> None:
    """Fill ULIFT_U, the sensitivity of the unit strip lift direction.

    ULIFT = UDRAG x SPN / |UDRAG x SPN| with UDRAG = VINF, so UDRAG_U is the
    identity on the three freestream components and zero elsewhere.
    ``ulift_raw`` is the un-normalised cross product (``None`` when its
    magnitude is zero, in which case ULIFT_U is zero) and ``ulift`` the
    normalised direction. Same operations and order as AVL's SFFORC loops.
    """
    ulift_u.fill(0.0)
    if ulift_raw is None:
        return
    for k in range(3):
        ic, jc = ICRS[k], JCRS[k]
        for n in range(numax):
            ulift_u[k, n] = (
                (1.0 if ic == n else 0.0) * spn[jc] - (1.0 if jc == n else 0.0) * spn[ic]
            )
    for n in range(numax):
        ulmag_u[n] = (
            (ulift_raw[0] * ulift_u[0, n] + ulift_raw[1] * ulift_u[1, n]
             + ulift_raw[2] * ulift_u[2, n]) / ulmag
        )
    for k in range(3):
        for n in range(numax):
            ulift_u[k, n] = ((ulift_u[k, n] - (ulift[k] * ulmag_u[n])) / ulmag)


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

    work = _get_sfforc_work(state, numax, ncontrol)
    ulift = work["ulift"]
    ulift_u = work["ulift_u"]
    ulift_d = work["ulift_d"]
    ulmag_u = work["ulmag_u"]
    veff_u = work["veff_u"]
    veffmag_u = work["veffmag_u"]

    (
        (
            strip_cfx, strip_cfy, strip_cfz, strip_cmx, strip_cmy, strip_cmz,
            strip_cfx_u, strip_cfy_u, strip_cfz_u, strip_cmx_u, strip_cmy_u, strip_cmz_u,
            strip_cfx_d, strip_cfy_d, strip_cfz_d, strip_cmx_d, strip_cmy_d, strip_cmz_d,
        ),
        strip_cnc,
        strip_cnc_u,
        strip_cnc_d,
    ) = _accumulate_all_vortex_forces(
        state, numax, ncontrol,
        gam, gam_u, gam_d,
        vv, vv_u, vv_d,
        wv, wv_u, wv_d,
        env_d, lnfld_wv,
    )

    # Hoist loop-invariant data out of the strip loop.  ``tolist`` converts
    # float64 to Python floats exactly, so scalar arithmetic on them gives the
    # same IEEE results as on NumPy scalars, only faster.
    nstrip = state.nstrip
    vinf = state.vinf
    # Wind axes stream vector defines drag direction (HHY 02272024: was stability axis)
    vx, vy, vz = float(vinf[0]), float(vinf[1]), float(vinf[2])
    xref, yref, zref = (float(v) for v in state.xyzref[:3])
    wx, wy, wz = (float(v) for v in state.wrot[:3])
    lvisc = state.lvisc
    lviscstrp = state.lviscstrp
    clcd = state.clcd
    rv1 = state.rv1
    rv2 = state.rv2

    lstripoff_l = state.lstripoff[:nstrip].tolist()
    wstrip_l = state.wstrip[:nstrip].tolist()
    chord_l = state.chord[:nstrip].tolist()
    ijfrst_l = state.ijfrst[:nstrip].tolist()
    nvstrp_l = state.nvstrp[:nstrip].tolist()
    ensy_l = state.ensy[:nstrip].tolist()
    ensz_l = state.ensz[:nstrip].tolist()
    rlex_l, rley_l, rlez_l = state.rle[:3, :nstrip].tolist()
    rle1x_l, rle1y_l, rle1z_l = state.rle1[:3, :nstrip].tolist()
    rle2x_l, rle2y_l, rle2z_l = state.rle2[:3, :nstrip].tolist()
    chord1_l = state.chord1[:nstrip].tolist()
    chord2_l = state.chord2[:nstrip].tolist()
    xsref_l = state.xsref[:nstrip].tolist()
    ysref_l = state.ysref[:nstrip].tolist()
    zsref_l = state.zsref[:nstrip].tolist()
    essx_l, essy_l, essz_l = state.ess[:3, :nstrip].tolist()
    ainc = state.ainc[:nstrip]
    sinainc_l = np.sin(ainc).tolist()
    cosainc_l = np.cos(ainc).tolist()
    lssurf = state.lssurf[:nstrip]
    mirrored_l = (state.imags[lssurf] < 0).tolist()
    clmax_l = state.clmax_surf[lssurf].tolist()
    strip_cfx_l = strip_cfx[:nstrip].tolist()
    strip_cfy_l = strip_cfy[:nstrip].tolist()
    strip_cfz_l = strip_cfz[:nstrip].tolist()
    strip_cmx_l = strip_cmx[:nstrip].tolist()
    strip_cmy_l = strip_cmy[:nstrip].tolist()
    strip_cmz_l = strip_cmz[:nstrip].tolist()
    strip_cnc_l = strip_cnc[:nstrip].tolist()

    cnc = state.cnc
    cnc_u = state.cnc_u
    cnc_d = state.cnc_d if ncontrol else None
    cf_lstrp = state.cf_lstrp
    cm_lstrp = state.cm_lstrp
    cfstrp = state.cfstrp
    cmstrp = state.cmstrp
    cdstrp = state.cdstrp
    cystrp = state.cystrp
    clstrp = state.clstrp
    cdst_a = state.cdst_a
    cyst_a = state.cyst_a
    clst_a = state.clst_a
    cdst_u = state.cdst_u
    cyst_u = state.cyst_u
    clst_u = state.clst_u
    cfst_u = state.cfst_u
    cmst_u = state.cmst_u
    if ncontrol:
        cdst_d = state.cdst_d
        cyst_d = state.cyst_d
        clst_d = state.clst_d
        cfst_d = state.cfst_d
        cmst_d = state.cmst_d
    cdv_lstrp = state.cdv_lstrp
    cl_lstrp = state.cl_lstrp
    cd_lstrp = state.cd_lstrp
    cmc4_lstrp = state.cmc4_lstrp
    ca_lstrp = state.ca_lstrp
    cn_lstrp = state.cn_lstrp
    clt_lstrp = state.clt_lstrp
    cla_lstrp = state.cla_lstrp
    cmle_lstrp = state.cmle_lstrp
    empty_d = np.zeros(0, dtype=np.float64)

    # Integrate the forces strip-wise, then surface-wise and into totals
    for strip in range(nstrip):
        vortex_start = ijfrst_l[strip]
        n_chordwise = nvstrp_l[strip]
        if lstripoff_l[strip] or wstrip_l[strip] == 0.0:
            _zero_strip(state, strip, n_chordwise, vortex_start, numax, ncontrol)
            continue

        # Calculate strip forces normalized to strip reference quantities
        cr = chord_l[strip]
        sr = cr * wstrip_l[strip]
        ensy = ensy_l[strip]
        ensz = ensz_l[strip]

        # Define local strip lift and drag directions
        # The "spanwise" vector is cross product of strip normal with X chordline
        spn = (0.0, ensz, -ensy)
        # Lift direction is vector product of "stream" and spanwise vector
        ul0, ul1, ul2 = _cross3(vx, vy, vz, spn[0], spn[1], spn[2])
        ulmag = math.sqrt(ul0 * ul0 + ul1 * ul1 + ul2 * ul2)
        if ulmag == 0.0:
            ulift_raw = None
            ul0, ul1, ul2 = 0.0, 0.0, 1.0
        else:
            ulift_raw = (ul0, ul1, ul2)
            ul0 = ul0 / ulmag
            ul1 = ul1 / ulmag
            ul2 = ul2 / ulmag
        # ULIFT_U is only needed by the viscous and CLmax branches; build lazily.
        ulift_u_ready = False

        # Use the strip 1/4 chord location for strip moments
        rc4x = rlex_l[strip] + (0.25 * cr)
        rc4y = rley_l[strip]
        rc4z = rlez_l[strip]
        cfx = strip_cfx_l[strip]
        cfy = strip_cfy_l[strip]
        cfz = strip_cfz_l[strip]
        cmx = strip_cmx_l[strip]
        cmy = strip_cmy_l[strip]
        cmz = strip_cmz_l[strip]
        cnc[strip] = strip_cnc_l[strip]
        # Row views of this call's freshly built per-strip sums: in-place
        # updates below only touch this strip's row, which nothing else reads.
        cfx_u = strip_cfx_u[strip]
        cfy_u = strip_cfy_u[strip]
        cfz_u = strip_cfz_u[strip]
        cmx_u = strip_cmx_u[strip]
        cmy_u = strip_cmy_u[strip]
        cmz_u = strip_cmz_u[strip]
        if ncontrol:
            cfx_d = strip_cfx_d[strip]
            cfy_d = strip_cfy_d[strip]
            cfz_d = strip_cfz_d[strip]
            cmx_d = strip_cmx_d[strip]
            cmy_d = strip_cmy_d[strip]
            cmz_d = strip_cmz_d[strip]
        else:
            cfx_d = cfy_d = cfz_d = cmx_d = cmy_d = cmz_d = empty_d
        cnc_u[strip, :numax] = strip_cnc_u[strip, :numax]
        if ncontrol:
            cnc_d[strip, :ncontrol] = strip_cnc_d[strip, :ncontrol]

        # Add h.v. forces from trailing legs lying on the wing surface
        if ltrforce:
            xte1 = rle1x_l[strip] + chord1_l[strip]
            xte2 = rle2x_l[strip] + chord2_l[strip]
            cnc_s = cnc[strip]
            cfxu = cfx_u.tolist()
            cfyu = cfy_u.tolist()
            cfzu = cfz_u.tolist()
            cmxu = cmx_u.tolist()
            cmyu = cmy_u.tolist()
            cmzu = cmz_u.tolist()
            cncu = cnc_u[strip, :numax].tolist()
            for chord_idx in range(n_chordwise):
                i = vortex_start + chord_idx
                gamma = _gam_scalar(gam, i)
                gam_u_i = gam_u[i, :numax].tolist()
                for leg in range(2):
                    if leg == 0:
                        px, py, pz = float(rv1[0, i]), float(rv1[1, i]), float(rv1[2, i])
                        r0 = (0.5 * (px + xte1) - rc4x)
                        r1 = (py - rc4y)
                        r2 = (pz - rc4z)
                        rrot0 = (0.5 * (px + xte1) - xref)
                        rrot1 = (py - yref)
                        rrot2 = (pz - zref)
                        gleg0 = (px - xte1)
                    else:
                        px, py, pz = float(rv2[0, i]), float(rv2[1, i]), float(rv2[2, i])
                        r0 = (0.5 * (px + xte2) - rc4x)
                        r1 = (py - rc4y)
                        r2 = (pz - rc4z)
                        rrot0 = (0.5 * (px + xte2) - xref)
                        rrot1 = (py - yref)
                        rrot2 = (pz - zref)
                        gleg0 = (xte2 - px)
                    gleg1 = 0.0
                    gleg2 = 0.0
                    # set total effective velocity = freestream + rotation
                    # (ignores h.v. induced contribution along trailing leg on wing)
                    vrot0, vrot1, vrot2 = _cross3(rrot0, rrot1, rrot2, wx, wy, wz)
                    leg_veff = (vx + vrot0, vy + vrot1, vz + vrot2)
                    # VEFF_U columns: unit vectors for the freestream components,
                    # RROT x e_k for the rotation components, zero beyond.
                    leg_veff_u = [(0.0, 0.0, 0.0)] * numax
                    leg_veff_u[0] = (1.0, 0.0, 0.0)
                    leg_veff_u[1] = (0.0, 1.0, 0.0)
                    leg_veff_u[2] = (0.0, 0.0, 1.0)
                    leg_veff_u[3] = _cross3(rrot0, rrot1, rrot2, 1.0, 0.0, 0.0)
                    leg_veff_u[4] = _cross3(rrot0, rrot1, rrot2, 0.0, 1.0, 0.0)
                    leg_veff_u[5] = _cross3(rrot0, rrot1, rrot2, 0.0, 0.0, 1.0)
                    f0, f1, f2 = _cross3(
                        leg_veff[0], leg_veff[1], leg_veff[2], gleg0, gleg1, gleg2
                    )
                    tg = 2.0 * gamma
                    fgam0 = tg * f0
                    fgam1 = tg * f1
                    fgam2 = tg * f2
                    dcfx = (fgam0 / sr)
                    dcfy = (fgam1 / sr)
                    dcfz = (fgam2 / sr)
                    cfx = (cfx + dcfx)
                    cfy = (cfy + dcfy)
                    cfz = (cfz + dcfz)
                    cmx = (cmx + ((dcfz * r1) - (dcfy * r2)) / cr)
                    cmy = (cmy + ((dcfx * r2) - (dcfz * r0)) / cr)
                    cmz = (cmz + ((dcfy * r0) - (dcfx * r1)) / cr)
                    cnc_s = (cnc_s + (cr * (ensy * dcfy + ensz * dcfz)))
                    for n in range(numax):
                        vu = leg_veff_u[n]
                        fu0, fu1, fu2 = _cross3(vu[0], vu[1], vu[2], gleg0, gleg1, gleg2)
                        tgu = 2.0 * gam_u_i[n]
                        dcfx_u = ((tgu * f0 + tg * fu0) / sr)
                        dcfy_u = ((tgu * f1 + tg * fu1) / sr)
                        dcfz_u = ((tgu * f2 + tg * fu2) / sr)
                        cfxu[n] = (cfxu[n] + dcfx_u)
                        cfyu[n] = (cfyu[n] + dcfy_u)
                        cfzu[n] = (cfzu[n] + dcfz_u)
                        cmxu[n] = (cmxu[n] + ((dcfz_u * r1) - (dcfy_u * r2)) / cr)
                        cmyu[n] = (cmyu[n] + ((dcfx_u * r2) - (dcfz_u * r0)) / cr)
                        cmzu[n] = (cmzu[n] + ((dcfy_u * r0) - (dcfx_u * r1)) / cr)
                        cncu[n] = (cncu[n] + (cr * (ensy * dcfy_u + ensz * dcfz_u)))
            cnc[strip] = cnc_s
            cfx_u[:] = cfxu
            cfy_u[:] = cfyu
            cfz_u[:] = cfzu
            cmx_u[:] = cmxu
            cmy_u[:] = cmyu
            cmz_u[:] = cmzu
            cnc_u[strip, :numax] = cncu

        # Drag terms due to viscous effects; CD from user-specified CD(CL) polar
        cdv_strip = 0.0
        if lvisc and lviscstrp[strip]:
            if not ulift_u_ready:
                _fill_ulift_u(ulift_u, ulmag_u, numax, spn, ulift_raw, (ul0, ul1, ul2), ulmag)
                ulift_u_ready = True
            # Onset velocity at strip c/4 = freestream + rotation
            rrot0 = rc4x - xref
            rrot1 = rc4y - yref
            rrot2 = rc4z - zref
            vrot0, vrot1, vrot2 = _cross3(rrot0, rrot1, rrot2, wx, wy, wz)
            ve0 = vx + vrot0
            ve1 = vy + vrot1
            ve2 = vz + vrot2
            veffmag = math.sqrt(ve0 * ve0 + ve1 * ve1 + ve2 * ve2)
            veff_u.fill(0.0)
            veff_u[0, 0] = 1.0
            veff_u[1, 1] = 1.0
            veff_u[2, 2] = 1.0
            veff_u[1, 3] = rrot2
            veff_u[2, 3] = -rrot1
            veff_u[0, 4] = -rrot2
            veff_u[2, 4] = rrot0
            veff_u[0, 5] = rrot1
            veff_u[1, 5] = -rrot0
            veffmag_u[:] = (ve0 * veff_u[0] + ve1 * veff_u[1] + ve2 * veff_u[2]) / veffmag
            clv = (ul0 * cfx + ul1 * cfy + ul2 * cfz)
            clv_u = (
                ul0 * cfx_u + ulift_u[0] * cfx
                + ul1 * cfy_u + ulift_u[1] * cfy
                + ul2 * cfz_u + ulift_u[2] * cfz
            )
            clv_d = (
                ul0 * cfx_d + ulift_d[0] * cfx
                + ul1 * cfy_d + ulift_d[1] * cfy
                + ul2 * cfz_d + ulift_d[2] * cfz
            )
            # Get CD from CLCD function using strip CL as parameter
            cdv, cdv_clv = cdcl(clcd[strip, :], float(clv))
            # Strip viscous force contribution (per unit strip area)
            dcvfx = (ve0 * veffmag * cdv)
            dcvfy = (ve1 * veffmag * cdv)
            dcvfz = (ve2 * veffmag * cdv)
            cfx = (cfx + dcvfx)
            cfy = (cfy + dcvfy)
            cfz = (cfz + dcvfz)
            cdv_strip = (vx * dcvfx + vy * dcvfy + vz * dcvfz)
            dcvfx_u = (
                (veff_u[0] * veffmag + ve0 * veffmag_u) * cdv
                + ve0 * veffmag * cdv_clv * clv_u
            )
            dcvfy_u = (
                (veff_u[1] * veffmag + ve1 * veffmag_u) * cdv
                + ve1 * veffmag * cdv_clv * clv_u
            )
            dcvfz_u = (
                (veff_u[2] * veffmag + ve2 * veffmag_u) * cdv
                + ve2 * veffmag * cdv_clv * clv_u
            )
            cfx_u += dcvfx_u
            cfy_u += dcvfy_u
            cfz_u += dcvfz_u
            cnc_u[strip, :numax] += cr * (ensy * dcvfy_u + ensz * dcvfz_u)
            dcvfx_d = ve0 * veffmag * cdv_clv * clv_d
            dcvfy_d = ve1 * veffmag * cdv_clv * clv_d
            dcvfz_d = ve2 * veffmag * cdv_clv * clv_d
            cfx_d += dcvfx_d
            cfy_d += dcvfy_d
            cfz_d += dcvfz_d
            if ncontrol:
                cnc_d[strip, :ncontrol] += cr * (ensy * dcvfy_d + ensz * dcvfz_d)
        cdv_lstrp[strip] = cdv_strip

        # At this point strip forces are in body axes at c/4, normalized by area/chord
        cf_lstrp[0, strip], cf_lstrp[1, strip], cf_lstrp[2, strip] = cfx, cfy, cfz
        cm_lstrp[0, strip], cm_lstrp[1, strip], cm_lstrp[2, strip] = cmx, cmy, cmz
        cfstrp[0, strip], cfstrp[1, strip], cfstrp[2, strip] = cfx, cfy, cfz
        # Transform strip body axes forces into stability axes
        cdstrp[strip] = (cfx * cosa + cfz * sina)
        cystrp[strip] = cfy
        clstrp[strip] = (-cfx * sina + cfz * cosa)
        cdst_a[strip] = (-cfx * sina + cfz * cosa)
        cyst_a[strip] = 0.0
        clst_a[strip] = (-cfx * cosa - cfz * sina)
        cdst_u[strip, :numax] = cfx_u * cosa + cfz_u * sina
        cyst_u[strip, :numax] = cfy_u
        clst_u[strip, :numax] = -cfx_u * sina + cfz_u * cosa
        cfst_u[0, strip, :numax] = cfx_u
        cfst_u[1, strip, :numax] = cfy_u
        cfst_u[2, strip, :numax] = cfz_u
        if ncontrol:
            cdst_d[strip, :ncontrol] = cfx_d * cosa + cfz_d * sina
            cyst_d[strip, :ncontrol] = cfy_d
            clst_d[strip, :ncontrol] = -cfx_d * sina + cfz_d * cosa
            cfst_d[0, strip, :ncontrol] = cfx_d
            cfst_d[1, strip, :ncontrol] = cfy_d
            cfst_d[2, strip, :ncontrol] = cfz_d

        # vector from chord c/4 reference point to case reference point XYZREF
        rref0 = rc4x - xref
        rref1 = rc4y - yref
        rref2 = rc4z - zref
        # Strip moments in body axes about XYZREF, normalized by strip area/chord
        cmstrp[0, strip] = (cmx + ((cfz * rref1) - (cfy * rref2)) / cr)
        cmstrp[1, strip] = (cmy + ((cfx * rref2) - (cfz * rref0)) / cr)
        cmstrp[2, strip] = (cmz + ((cfy * rref0) - (cfx * rref1)) / cr)
        cmst_u[0, strip, :numax] = cmx_u + ((cfz_u * rref1) - (cfy_u * rref2)) / cr
        cmst_u[1, strip, :numax] = cmy_u + ((cfx_u * rref2) - (cfz_u * rref0)) / cr
        cmst_u[2, strip, :numax] = cmz_u + ((cfy_u * rref0) - (cfx_u * rref1)) / cr
        if ncontrol:
            cmst_d[0, strip, :ncontrol] = cmx_d + ((cfz_d * rref1) - (cfy_d * rref2)) / cr
            cmst_d[1, strip, :ncontrol] = cmy_d + ((cfx_d * rref2) - (cfz_d * rref0)) / cr
            cmst_d[2, strip, :ncontrol] = cmz_d + ((cfy_d * rref0) - (cfx_d * rref1)) / cr

        local_cl = (ul0 * cfx + ul1 * cfy + ul2 * cfz)
        cl_lstrp[strip] = local_cl
        cd_lstrp[strip] = (vx * cfx + vy * cfy + vz * cfz)
        cmc4_lstrp[strip] = (ensz * cmy - ensy * cmz)
        # CN,CA forces rotated to be in and normal to strip incidence (HHY bugfix 01102024)
        caxl0 = cfx
        cnrm0 = (ensy * cfy + ensz * cfz)
        sinainc = sinainc_l[strip]
        cosainc = cosainc_l[strip]
        ca_lstrp[strip] = (caxl0 * cosainc - cnrm0 * sinainc)
        cn_lstrp[strip] = (cnrm0 * cosainc + caxl0 * sinainc)

        # set total effective velocity = freestream + rotation
        rr0 = xsref_l[strip] - xref
        rr1 = ysref_l[strip] - yref
        rr2 = zsref_l[strip] - zref
        vr0, vr1, vr2 = _cross3(rr0, rr1, rr2, wx, wy, wz)
        ve0 = vx + vr0
        ve1 = vy + vr1
        ve2 = vz + vr2
        vsq = ve0 * ve0 + ve1 * ve1 + ve2 * ve2
        vsqi = (1.0 if vsq == 0.0 else 1.0 / vsq)
        # spanwise and perpendicular velocity components
        es0 = essx_l[strip]
        es1 = essy_l[strip]
        es2 = essz_l[strip]
        vspan = (ve0 * es0 + ve1 * es1 + ve2 * es2)
        vp0 = ve0 - es0 * vspan
        vp1 = ve1 - es1 * vspan
        vp2 = ve2 - es2 * vspan
        vpsq = ((vp0 * vp0) + (vp1 * vp1) + (vp2 * vp2))
        vpsqi = (1.0 if vpsq == 0.0 else 1.0 / vpsq)
        clt_lstrp[strip] = (local_cl * vpsqi)
        cla_lstrp[strip] = (local_cl * vsqi)

        # Moment about strip LE midpoint in direction of LE segment
        rle0 = rc4x - rlex_l[strip]
        rle1 = rc4y - rley_l[strip]
        rle2 = rc4z - rlez_l[strip]
        delx = (rle2x_l[strip] - rle1x_l[strip])
        dely = (rle2y_l[strip] - rle1y_l[strip])
        delz = (rle2z_l[strip] - rle1z_l[strip])
        if mirrored_l[strip]:
            delx = (-delx)
            dely = (-dely)
            delz = (-delz)
        dmag = math.sqrt((delx * delx) + (dely * dely) + (delz * delz))
        cmle = 0.0
        if dmag != 0.0:
            cmle = (
                delx / dmag * (cmx + ((cfz * rle1) - (cfy * rle2)) / cr)
                + dely / dmag * (cmy + ((cfx * rle2) - (cfz * rle0)) / cr)
                + delz / dmag * (cmz + ((cfy * rle0) - (cfx * rle1)) / cr)
            )
        cmle_lstrp[strip] = cmle

        clmax = clmax_l[strip]
        local_cl = cl_lstrp[strip]
        if clmax > 0.0 and local_cl > clmax:
            if not ulift_u_ready:
                _fill_ulift_u(ulift_u, ulmag_u, numax, spn, ulift_raw, (ul0, ul1, ul2), ulmag)
                ulift_u_ready = True
            ulift[0] = ul0
            ulift[1] = ul1
            ulift[2] = ul2
            # For s = Clmax / Cl, d(s F) = s (dF - F dCl / Cl).
            # Form the expression in parentheses here, while the local lift
            # direction and all uncapped strip loads are still available.
            # The surface loop below applies s to both loads and derivatives.
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

    # Surface forces and moments summed from strip forces
    for isurf in range(state.nsurf):
        state.cdsurf[isurf] = 0.0
        state.cysurf[isurf] = 0.0
        state.clsurf[isurf] = 0.0
        state.cfsurf[:, isurf] = 0.0
        state.cmsurf[:, isurf] = 0.0
        state.cdvsurf[isurf] = 0.0
        state.cds_a[isurf] = 0.0
        state.cys_a[isurf] = 0.0
        state.cls_a[isurf] = 0.0
        state.cds_u[isurf, :numax] = 0.0
        state.cys_u[isurf, :numax] = 0.0
        state.cls_u[isurf, :numax] = 0.0
        state.cfs_u[:, isurf, :numax] = 0.0
        state.cms_u[:, isurf, :numax] = 0.0
        if ncontrol:
            state.cds_d[isurf, :ncontrol] = 0.0
            state.cys_d[isurf, :ncontrol] = 0.0
            state.cls_d[isurf, :ncontrol] = 0.0
            state.cfs_d[:, isurf, :ncontrol] = 0.0
            state.cms_d[:, isurf, :ncontrol] = 0.0
        state.cf_lsrf[:, isurf] = 0.0
        state.cm_lsrf[:, isurf] = 0.0
        enave = np.zeros(3, dtype=np.float64)

        j0 = int(state.jfrst[isurf])
        nj = int(state.nj[isurf])
        js = np.arange(j0, j0 + nj, dtype=np.intp)
        sr = state.chord[js] * state.wstrip[js]
        cr = state.chord[js]
        scale_sref = sr / state.sref
        scale_ssurf = sr / state.ssurf[isurf]
        scale_cm_b = scale_sref * (cr / state.bref)
        scale_cm_c = scale_sref * (cr / state.cref)
        scale_cm_loc = scale_ssurf * (cr / state.cavesurf[isurf]) / cr

        enave[0] = 0.0
        enave[1] = float(np.sum(sr * state.ensy[js]))
        enave[2] = float(np.sum(sr * state.ensz[js]))

        state.cdsurf[isurf] = float(np.sum(state.cdstrp[js] * scale_sref))
        state.cysurf[isurf] = float(np.sum(state.cystrp[js] * scale_sref))
        state.clsurf[isurf] = float(np.sum(state.clstrp[js] * scale_sref))
        state.cfsurf[:, isurf] = np.sum(state.cfstrp[:, js] * scale_sref, axis=1)
        state.cmsurf[0, isurf] = float(np.sum(state.cmstrp[0, js] * scale_cm_b))
        state.cmsurf[1, isurf] = float(np.sum(state.cmstrp[1, js] * scale_cm_c))
        state.cmsurf[2, isurf] = float(np.sum(state.cmstrp[2, js] * scale_cm_b))
        state.cdvsurf[isurf] = float(np.sum(state.cdv_lstrp[js] * scale_sref))
        state.cds_a[isurf] = float(np.sum(state.cdst_a[js] * scale_sref))
        state.cys_a[isurf] = float(np.sum(state.cyst_a[js] * scale_sref))
        state.cls_a[isurf] = float(np.sum(state.clst_a[js] * scale_sref))
        state.cds_u[isurf, :numax] = np.sum(state.cdst_u[js, :numax] * scale_sref[:, np.newaxis], axis=0)
        state.cys_u[isurf, :numax] = np.sum(state.cyst_u[js, :numax] * scale_sref[:, np.newaxis], axis=0)
        state.cls_u[isurf, :numax] = np.sum(state.clst_u[js, :numax] * scale_sref[:, np.newaxis], axis=0)
        state.cfs_u[:, isurf, :numax] = np.sum(
            state.cfst_u[:, js, :numax] * scale_sref[np.newaxis, :, np.newaxis], axis=1
        )
        state.cms_u[0, isurf, :numax] = np.sum(
            state.cmst_u[0, js, :numax] * scale_cm_b[:, np.newaxis], axis=0
        )
        state.cms_u[1, isurf, :numax] = np.sum(
            state.cmst_u[1, js, :numax] * scale_cm_c[:, np.newaxis], axis=0
        )
        state.cms_u[2, isurf, :numax] = np.sum(
            state.cmst_u[2, js, :numax] * scale_cm_b[:, np.newaxis], axis=0
        )
        if ncontrol:
            state.cds_d[isurf, :ncontrol] = np.sum(
                state.cdst_d[js, :ncontrol] * scale_sref[:, np.newaxis], axis=0
            )
            state.cys_d[isurf, :ncontrol] = np.sum(
                state.cyst_d[js, :ncontrol] * scale_sref[:, np.newaxis], axis=0
            )
            state.cls_d[isurf, :ncontrol] = np.sum(
                state.clst_d[js, :ncontrol] * scale_sref[:, np.newaxis], axis=0
            )
            state.cfs_d[:, isurf, :ncontrol] = np.sum(
                state.cfst_d[:, js, :ncontrol] * scale_sref[np.newaxis, :, np.newaxis], axis=1
            )
            state.cms_d[0, isurf, :ncontrol] = np.sum(
                state.cmst_d[0, js, :ncontrol] * scale_cm_b[:, np.newaxis], axis=0
            )
            state.cms_d[1, isurf, :ncontrol] = np.sum(
                state.cmst_d[1, js, :ncontrol] * scale_cm_c[:, np.newaxis], axis=0
            )
            state.cms_d[2, isurf, :ncontrol] = np.sum(
                state.cmst_d[2, js, :ncontrol] * scale_cm_b[:, np.newaxis], axis=0
            )

        rc4s = np.stack(
            (
                state.rle[0, js] + 0.25 * state.chord[js],
                state.rle[1, js],
                state.rle[2, js],
            ),
            axis=0,
        )
        if state.imags[isurf] >= 0:
            rroot = state.rle1[:, j0]
        else:
            rroot = state.rle2[:, j0]
        r = rc4s - rroot[:, np.newaxis]
        state.cf_lsrf[:, isurf] = np.sum(state.cf_lstrp[:, js] * scale_ssurf, axis=1)
        for k in range(3):
            ic, jc = ICRS[k], JCRS[k]
            state.cm_lsrf[k, isurf] = float(np.sum(
                scale_cm_loc
                * (state.cm_lstrp[k, js] + state.cf_lstrp[jc, js] * r[ic] - state.cf_lstrp[ic, js] * r[jc])
            ))

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
