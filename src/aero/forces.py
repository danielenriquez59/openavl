"""Aerodynamic force integration (port of aero.f: VINFAB, AERO, SFFORC, BDFORC).

Computes inviscid forces from horseshoe vortices, inviscid body source forces,
viscous strip drag from CD(CL) polars, and far-field Trefftz-plane contributions.
"""

from __future__ import annotations

from typing import Any

import math
import numpy as np

from openavl.aero.aic import cross, dot
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
    sina = (np.sin((state.alfa)))
    cosa = (np.cos((state.alfa)))
    sinb = (np.sin((state.beta)))
    cosb = (np.cos((state.beta)))

    # Unit freestream velocity vector in body axes.
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

        # set VEFF sensitivities to freestream,rotation components
        veff_u = np.zeros((3, 6, nseg), dtype=np.float64)
        veff_u[0, 0, :] = 1.0 / betm
        veff_u[1, 1, :] = 1.0
        veff_u[2, 2, :] = 1.0
        veff_u[0, 3, :] = 0.0
        veff_u[1, 3, :] = rrot2
        veff_u[2, 3, :] = -rrot1
        veff_u[0, 4, :] = -rrot2
        veff_u[1, 4, :] = 0.0
        veff_u[2, 4, :] = rrot0
        veff_u[0, 5, :] = rrot1
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


def _accumulate_all_vortex_forces(
    state: Any,
    numax: int,
    ncontrol: int,
    ndesign: int,
    gam: np.ndarray,
    gam_u: np.ndarray,
    gam_d: np.ndarray,
    gam_g: np.ndarray,
    vv: np.ndarray,
    vv_u: np.ndarray,
    vv_d: np.ndarray,
    vv_g: np.ndarray,
    wv: np.ndarray,
    wv_u: np.ndarray,
    wv_d: np.ndarray,
    wv_g: np.ndarray,
    env_d: np.ndarray,
    env_g: np.ndarray,
    lnfld_wv: bool,
) -> tuple[
    tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    ],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Vectorized bound-vortex force accumulation for all vortices in one pass."""
    nvor = state.nvor
    nstrip = state.nstrip
    z6 = np.zeros((nstrip, numax), dtype=np.float64)
    zd = np.zeros((nstrip, max(1, ncontrol)), dtype=np.float64)
    zg = np.zeros((nstrip, max(1, ndesign)), dtype=np.float64)
    if nvor == 0:
        empty = (
            np.zeros(nstrip), np.zeros(nstrip), np.zeros(nstrip),
            np.zeros(nstrip), np.zeros(nstrip), np.zeros(nstrip),
            z6, z6, z6, z6, z6, z6,
            zd, zd, zd, zd, zd, zd,
            zg, zg, zg, zg, zg, zg,
        )
        return empty, np.zeros(nstrip), z6, zd, zg

    v2s = getattr(state, "vortex_to_strip", None)
    if v2s is None or v2s.shape[0] != nvor:
        from openavl.core.state import build_vortex_to_strip

        v2s = build_vortex_to_strip(state)

    rc4_v = np.empty((3, nvor), dtype=np.float64)
    rc4_v[0] = state.rle[0, v2s] + 0.25 * state.chord[v2s]
    rc4_v[1] = state.rle[1, v2s]
    rc4_v[2] = state.rle[2, v2s]

    sr = state.chord[v2s] * state.wstrip[v2s]
    cr_v = state.chord[v2s]
    ensy_v = state.ensy[v2s]
    ensz_v = state.ensz[v2s]
    active = (~state.lstripoff[v2s]) & (state.wstrip[v2s] > 0.0)

    g = state.rv2[:, :nvor] - state.rv1[:, :nvor]
    r = state.rv[:, :nvor] - rc4_v
    rrot = state.rv[:, :nvor] - state.xyzref[:, np.newaxis]
    vrot = _cross_axis0(rrot, state.wrot[:, np.newaxis])

    if lnfld_wv:
        # forces using total induced velocity WV (h.v.'s + body source+doublets)
        vind = wv[:, :nvor]
        vind_u = wv_u[:, :nvor, :numax]
        vind_d = wv_d[:, :nvor, :ncontrol] if ncontrol else None
        vind_g = wv_g[:, :nvor, :ndesign] if ndesign else None
    else:
        # forces using h.v. induced velocities VV only (exclude body induced)
        vind = vv[:, :nvor]
        vind_u = vv_u[:, :nvor, :numax]
        vind_d = vv_d[:, :nvor, :ncontrol] if ncontrol else None
        vind_g = vv_g[:, :nvor, :ndesign] if ndesign else None

    veff = state.vinf[:, np.newaxis] + vrot + vind
    # set VEFF sensitivities to freestream,rotation,induced,controls,design
    veff_u = np.empty((3, numax, nvor), dtype=np.float64)
    _batch_veff_u(rrot, vind_u, numax, veff_u)

    g_bc = g[:, np.newaxis, :]
    # Force coefficient on vortex segment is 2(Veff x Gamma)
    f = _cross_axis0(veff, g)
    f_u = _cross_axis0(veff_u, g_bc)

    f_d = None
    if ncontrol:
        veff_d = vind_d.transpose(0, 2, 1)
        f_d = _cross_axis0(veff_d, g_bc)

    f_g = None
    if ndesign:
        veff_g = vind_g.transpose(0, 2, 1)
        f_g = _cross_axis0(veff_g, g_bc)

    gi = gam[:nvor]
    fgam = 2.0 * gi * f
    fgam_u = 2.0 * gam_u[:nvor, :].T[np.newaxis, :, :] * f[:, np.newaxis, :] + 2.0 * gi * f_u

    fgam_d = None
    if ncontrol:
        fgam_d = (
            2.0 * gam_d[:nvor, :].T[np.newaxis, :, :] * f[:, np.newaxis, :]
            + 2.0 * gi * f_d
        )

    fgam_g = None
    if ndesign:
        fgam_g = (
            2.0 * gam_g[:nvor, :].T[np.newaxis, :, :] * f[:, np.newaxis, :]
            + 2.0 * gi * f_g
        )

    env = state.env[:, :nvor]
    dxv_w = state.dxv[:nvor] * state.wstrip[v2s]
    # Delta Cp (loading across lifting surface) from vortex
    state.dcp[:nvor] = np.sum(env * fgam, axis=0) / dxv_w
    state.dcp_u[:nvor, :numax] = (np.sum(env[:, np.newaxis, :] * fgam_u, axis=0) / dxv_w).T
    if ncontrol:
        term1 = np.sum(env[:, np.newaxis, :] * fgam_d, axis=0)
        term2 = np.einsum("kin,ki->in", env_d[:, :nvor, :ncontrol], fgam)
        state.dcp_d[:nvor, :ncontrol] = (term1.T + term2) / dxv_w[:, np.newaxis]
    if ndesign:
        term1 = np.sum(env[:, np.newaxis, :] * fgam_g, axis=0)
        # term2 uses env_g which is currently zero (not yet computed in geometry.py);
        # see state.py env_g comment for the fix and the AVL Fortran reference bug.
        term2 = np.einsum("kin,ki->in", env_g[:, :nvor, :ndesign], fgam)
        state.dcp_g[:nvor, :ndesign] = (term1.T + term2) / dxv_w[:, np.newaxis]

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

    cfx = np.bincount(v2s, weights=dcfx, minlength=nstrip)
    cfy = np.bincount(v2s, weights=dcfy, minlength=nstrip)
    cfz = np.bincount(v2s, weights=dcfz, minlength=nstrip)
    cmx = np.bincount(v2s, weights=dcmx, minlength=nstrip)
    cmy = np.bincount(v2s, weights=dcmy, minlength=nstrip)
    cmz = np.bincount(v2s, weights=dcmz, minlength=nstrip)
    cnc = np.bincount(v2s, weights=dcnc, minlength=nstrip)

    dcf_u = fgam_u * inv_sr
    cfx_u = np.zeros((nstrip, numax), dtype=np.float64)
    cfy_u = np.zeros((nstrip, numax), dtype=np.float64)
    cfz_u = np.zeros((nstrip, numax), dtype=np.float64)
    cmx_u = np.zeros((nstrip, numax), dtype=np.float64)
    cmy_u = np.zeros((nstrip, numax), dtype=np.float64)
    cmz_u = np.zeros((nstrip, numax), dtype=np.float64)
    np.add.at(cfx_u, v2s, dcf_u[0].T)
    np.add.at(cfy_u, v2s, dcf_u[1].T)
    np.add.at(cfz_u, v2s, dcf_u[2].T)
    np.add.at(cmx_u, v2s, ((dcf_u[2] * r[1, np.newaxis, :] - dcf_u[1] * r[2, np.newaxis, :]) * inv_cr).T)
    np.add.at(cmy_u, v2s, ((dcf_u[0] * r[2, np.newaxis, :] - dcf_u[2] * r[0, np.newaxis, :]) * inv_cr).T)
    np.add.at(cmz_u, v2s, ((dcf_u[1] * r[0, np.newaxis, :] - dcf_u[0] * r[1, np.newaxis, :]) * inv_cr).T)
    cnc_u = cr_v[np.newaxis, :] * np.sum(
        ensy_v[np.newaxis, :] * dcf_u[1] + ensz_v[np.newaxis, :] * dcf_u[2],
        axis=0,
    )
    cnc_u_acc = np.zeros((nstrip, numax), dtype=np.float64)
    np.add.at(cnc_u_acc, v2s, cnc_u.T)

    cfx_d = cfy_d = cfz_d = cmx_d = cmy_d = cmz_d = None
    cnc_d_acc = None
    if ncontrol:
        dcf_d = fgam_d * inv_sr
        cfx_d = np.zeros((nstrip, ncontrol), dtype=np.float64)
        cfy_d = np.zeros((nstrip, ncontrol), dtype=np.float64)
        cfz_d = np.zeros((nstrip, ncontrol), dtype=np.float64)
        cmx_d = np.zeros((nstrip, ncontrol), dtype=np.float64)
        cmy_d = np.zeros((nstrip, ncontrol), dtype=np.float64)
        cmz_d = np.zeros((nstrip, ncontrol), dtype=np.float64)
        np.add.at(cfx_d, v2s, dcf_d[0].T)
        np.add.at(cfy_d, v2s, dcf_d[1].T)
        np.add.at(cfz_d, v2s, dcf_d[2].T)
        np.add.at(cmx_d, v2s, ((dcf_d[2] * r[1, np.newaxis, :] - dcf_d[1] * r[2, np.newaxis, :]) * inv_cr).T)
        np.add.at(cmy_d, v2s, ((dcf_d[0] * r[2, np.newaxis, :] - dcf_d[2] * r[0, np.newaxis, :]) * inv_cr).T)
        np.add.at(cmz_d, v2s, ((dcf_d[1] * r[0, np.newaxis, :] - dcf_d[0] * r[1, np.newaxis, :]) * inv_cr).T)
        cnc_d_v = cr_v[np.newaxis, :] * np.sum(
            ensy_v[np.newaxis, :] * dcf_d[1] + ensz_v[np.newaxis, :] * dcf_d[2],
            axis=0,
        )
        cnc_d_acc = np.zeros((nstrip, ncontrol), dtype=np.float64)
        np.add.at(cnc_d_acc, v2s, cnc_d_v.T)

    cfx_g = cfy_g = cfz_g = cmx_g = cmy_g = cmz_g = None
    cnc_g_acc = None
    if ndesign:
        dcf_g = fgam_g * inv_sr
        cfx_g = np.zeros((nstrip, ndesign), dtype=np.float64)
        cfy_g = np.zeros((nstrip, ndesign), dtype=np.float64)
        cfz_g = np.zeros((nstrip, ndesign), dtype=np.float64)
        cmx_g = np.zeros((nstrip, ndesign), dtype=np.float64)
        cmy_g = np.zeros((nstrip, ndesign), dtype=np.float64)
        cmz_g = np.zeros((nstrip, ndesign), dtype=np.float64)
        np.add.at(cfx_g, v2s, dcf_g[0].T)
        np.add.at(cfy_g, v2s, dcf_g[1].T)
        np.add.at(cfz_g, v2s, dcf_g[2].T)
        np.add.at(cmx_g, v2s, ((dcf_g[2] * r[1, np.newaxis, :] - dcf_g[1] * r[2, np.newaxis, :]) * inv_cr).T)
        np.add.at(cmy_g, v2s, ((dcf_g[0] * r[2, np.newaxis, :] - dcf_g[2] * r[0, np.newaxis, :]) * inv_cr).T)
        np.add.at(cmz_g, v2s, ((dcf_g[1] * r[0, np.newaxis, :] - dcf_g[0] * r[1, np.newaxis, :]) * inv_cr).T)
        cnc_g_v = cr_v[np.newaxis, :] * np.sum(
            ensy_v[np.newaxis, :] * dcf_g[1] + ensz_v[np.newaxis, :] * dcf_g[2],
            axis=0,
        )
        cnc_g_acc = np.zeros((nstrip, ndesign), dtype=np.float64)
        np.add.at(cnc_g_acc, v2s, cnc_g_v.T)

    if ncontrol:
        # hinge moments
        phinge_v = state.phinge[:, v2s, :ncontrol]
        vhinge_v = state.vhinge[:, v2s, :ncontrol]
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
        if ndesign:
            mh_g = _cross_axis0(
                rh[:, np.newaxis, :, :],
                fgam_g[:, :, :, np.newaxis],
            )
            state.chinge_g[:ncontrol, :ndesign] = np.sum(
                np.einsum("kvnc,knc->vnc", mh_g, vhinge_v) * dfac[np.newaxis, :, :],
                axis=1,
            ).T

    if cfx_d is None:
        cfx_d, cfy_d, cfz_d, cmx_d, cmy_d, cmz_d = zd, zd, zd, zd, zd, zd
        cnc_d_acc = zd
    if cfx_g is None:
        cfx_g, cfy_g, cfz_g, cmx_g, cmy_g, cmz_g = zg, zg, zg, zg, zg, zg
        cnc_g_acc = zg

    return (
        cfx, cfy, cfz, cmx, cmy, cmz,
        cfx_u, cfy_u, cfz_u, cmx_u, cmy_u, cmz_u,
        cfx_d, cfy_d, cfz_d, cmx_d, cmy_d, cmz_d,
        cfx_g, cfy_g, cfz_g, cmx_g, cmy_g, cmz_g,
    ), cnc, cnc_u_acc, cnc_d_acc, cnc_g_acc


def _get_sfforc_work(state: Any, numax: int, ncontrol: int, ndesign: int) -> dict[str, np.ndarray]:
    """Return strip-force work buffers from state, or local fallbacks for minimal states."""
    nc = max(1, ncontrol)
    nd = max(1, ndesign)

    def _buf(name: str, shape: tuple[int, ...]) -> np.ndarray:
        arr = getattr(state, name, None)
        if arr is None or arr.shape != shape:
            arr = np.zeros(shape, dtype=np.float64)
            setattr(state, name, arr)
        return arr

    return {
        "udrag_u": _buf("sfforc_udrag_u", (3, numax)),
        "spn": _buf("sfforc_spn", (3,)),
        "udrag": _buf("sfforc_udrag", (3,)),
        "ulift": _buf("sfforc_ulift", (3,)),
        "ulift_u": _buf("sfforc_ulift_u", (3, numax)),
        "ulift_d": _buf("sfforc_ulift_d", (3, nc))[:, :nc],
        "ulift_g": _buf("sfforc_ulift_g", (3, nd))[:, :nd],
        "ulmag_u": _buf("sfforc_ulmag_u", (numax,)),
        "rc4": _buf("sfforc_rc4", (3,)),
        "cfx_u": _buf("sfforc_cfx_u", (numax,)),
        "cfy_u": _buf("sfforc_cfy_u", (numax,)),
        "cfz_u": _buf("sfforc_cfz_u", (numax,)),
        "cmx_u": _buf("sfforc_cmx_u", (numax,)),
        "cmy_u": _buf("sfforc_cmy_u", (numax,)),
        "cmz_u": _buf("sfforc_cmz_u", (numax,)),
        "cfx_d": _buf("sfforc_cfx_d", (nc,))[:ncontrol],
        "cfy_d": _buf("sfforc_cfy_d", (nc,))[:ncontrol],
        "cfz_d": _buf("sfforc_cfz_d", (nc,))[:ncontrol],
        "cmx_d": _buf("sfforc_cmx_d", (nc,))[:ncontrol],
        "cmy_d": _buf("sfforc_cmy_d", (nc,))[:ncontrol],
        "cmz_d": _buf("sfforc_cmz_d", (nc,))[:ncontrol],
        "cfx_g": _buf("sfforc_cfx_g", (nd,))[:ndesign],
        "cfy_g": _buf("sfforc_cfy_g", (nd,))[:ndesign],
        "cfz_g": _buf("sfforc_cfz_g", (nd,))[:ndesign],
        "cmx_g": _buf("sfforc_cmx_g", (nd,))[:ndesign],
        "cmy_g": _buf("sfforc_cmy_g", (nd,))[:ndesign],
        "cmz_g": _buf("sfforc_cmz_g", (nd,))[:ndesign],
        "veff": _buf("sfforc_veff", (3,)),
        "veff_u": _buf("sfforc_veff_u", (3, numax)),
        "veffmag_u": _buf("sfforc_veffmag_u", (numax,)),
    }


def _zero_strip(state, j, nvc, i1, numax, ncontrol, ndesign):
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
    if ndesign:
        state.cnc_g[j, :ndesign] = 0.0
        state.cdst_g[j, :ndesign] = 0.0
        state.cyst_g[j, :ndesign] = 0.0
        state.clst_g[j, :ndesign] = 0.0
        state.cfst_g[:, j, :ndesign] = 0.0
        state.cmst_g[:, j, :ndesign] = 0.0
    for ii in range(nvc):
        i = i1 + ii
        state.dcp[i] = 0.0
        state.dcp_u[i, :numax] = 0.0
        if ncontrol:
            state.dcp_d[i, :ncontrol] = 0.0
        if ndesign:
            state.dcp_g[i, :ndesign] = 0.0


def sfforc(state: Any) -> Any:
    """Integrate strip/surface forces from bound vorticity (SFFORC).

    Purpose: calculate forces on the configuration by vortex, strip, and
    surface. Integrate strip-wise, surface-wise, and into totals.

    Output includes DCP loadings, strip/surface coefficients, span loading
    (CNC), and hinge moments.

    When ``state.clmax_surf[isurf] > 0``, strip forces on that surface are
    scaled so local ``cl_lstrp`` does not exceed the limit (OpenAVL extension).
    """
    numax    = state.numax
    ncontrol = state.ncontrol
    ndesign  = state.ndesign
    sina     = (np.sin((state.alfa)))
    cosa     = (np.cos((state.alfa)))
    ltrforce = bool(getattr(state, "ltrforce", False))
    lnfld_wv = bool(getattr(state, "lnfld_wv", False))
    lfload   = np.asarray(getattr(state, "lfload", np.ones(state.nsurf, dtype=bool)), dtype=bool)

    gam   = state.gam.reshape(-1) if state.gam.ndim > 1 else state.gam
    gam_u = state.gam_u
    gam_d = getattr(state, "gam_d", np.zeros((state.nvor, max(1, ncontrol)), dtype=np.float64))
    gam_g = getattr(state, "gam_g", np.zeros((state.nvor, max(1, ndesign)), dtype=np.float64))
    vv    = getattr(state, "vv", np.zeros((3, state.nvor), dtype=np.float64))
    vv_u  = getattr(state, "vv_u", np.zeros((3, state.nvor, numax), dtype=np.float64))
    vv_d  = getattr(state, "vv_d", np.zeros((3, state.nvor, max(1, ncontrol)), dtype=np.float64))
    vv_g  = getattr(state, "vv_g", np.zeros((3, state.nvor, max(1, ndesign)), dtype=np.float64))
    wv    = getattr(state, "wv", np.zeros((3, state.nvor), dtype=np.float64))
    wv_u  = getattr(state, "wv_u", np.zeros((3, state.nvor, numax), dtype=np.float64))
    wv_d  = getattr(state, "wv_d", np.zeros((3, state.nvor, max(1, ncontrol)), dtype=np.float64))
    wv_g  = getattr(state, "wv_g", np.zeros((3, state.nvor, max(1, ndesign)), dtype=np.float64))
    env_d = getattr(state, "env_d", np.zeros((3, state.nvor, max(1, ncontrol)), dtype=np.float64))
    env_g = getattr(state, "env_g", np.zeros((3, state.nvor, max(1, ndesign)), dtype=np.float64))

    work = _get_sfforc_work(state, numax, ncontrol, ndesign)
    udrag_u = work["udrag_u"]
    udrag_u.fill(0.0)
    udrag_u[0, 0] = 1.0
    udrag_u[1, 1] = 1.0
    udrag_u[2, 2] = 1.0
    spn = work["spn"]
    udrag = work["udrag"]
    ulift = work["ulift"]
    ulift_u = work["ulift_u"]
    ulift_d = work["ulift_d"]
    ulift_g = work["ulift_g"]
    ulmag_u = work["ulmag_u"]
    rc4 = work["rc4"]
    cfx_u = work["cfx_u"]
    cfy_u = work["cfy_u"]
    cfz_u = work["cfz_u"]
    cmx_u = work["cmx_u"]
    cmy_u = work["cmy_u"]
    cmz_u = work["cmz_u"]
    cfx_d = work["cfx_d"]
    cfy_d = work["cfy_d"]
    cfz_d = work["cfz_d"]
    cmx_d = work["cmx_d"]
    cmy_d = work["cmy_d"]
    cmz_d = work["cmz_d"]
    cfx_g = work["cfx_g"]
    cfy_g = work["cfy_g"]
    cfz_g = work["cfz_g"]
    cmx_g = work["cmx_g"]
    cmy_g = work["cmy_g"]
    cmz_g = work["cmz_g"]
    veff = work["veff"]
    veff_u = work["veff_u"]
    veffmag_u = work["veffmag_u"]

    (
        (
            strip_cfx, strip_cfy, strip_cfz, strip_cmx, strip_cmy, strip_cmz,
            strip_cfx_u, strip_cfy_u, strip_cfz_u, strip_cmx_u, strip_cmy_u, strip_cmz_u,
            strip_cfx_d, strip_cfy_d, strip_cfz_d, strip_cmx_d, strip_cmy_d, strip_cmz_d,
            strip_cfx_g, strip_cfy_g, strip_cfz_g, strip_cmx_g, strip_cmy_g, strip_cmz_g,
        ),
        strip_cnc,
        strip_cnc_u,
        strip_cnc_d,
        strip_cnc_g,
    ) = _accumulate_all_vortex_forces(
        state, numax, ncontrol, ndesign,
        gam, gam_u, gam_d, gam_g,
        vv, vv_u, vv_d, vv_g,
        wv, wv_u, wv_d, wv_g,
        env_d, env_g, lnfld_wv,
    )

    # Integrate the forces strip-wise, then surface-wise and into totals
    for j in range(state.nstrip):
        i1 = int(state.ijfrst[j])
        nvc = int(state.nvstrp[j])
        if state.lstripoff[j] or state.wstrip[j] == 0.0:
            _zero_strip(state, j, nvc, i1, numax, ncontrol, ndesign)
            continue

        # Calculate strip forces normalized to strip reference quantities
        cr = (state.chord[j])
        sr = (state.chord[j] * state.wstrip[j])
        xte1 = (state.rle1[0, j] + state.chord1[j])
        xte2 = (state.rle2[0, j] + state.chord2[j])

        # Define local strip lift and drag directions
        # The "spanwise" vector is cross product of strip normal with X chordline
        spn[0] = 0.0
        spn[1] = state.ensz[j]
        spn[2] = -state.ensy[j]
        # Wind axes stream vector defines drag direction (HHY 02272024: was stability axis)
        udrag[:] = state.vinf

        # Lift direction is vector product of "stream" and spanwise vector
        ulift[:] = cross(udrag, spn)
        ulmag = math.sqrt(ulift[0] * ulift[0] + ulift[1] * ulift[1] + ulift[2] * ulift[2])
        ulift_u.fill(0.0)

        if ulmag == 0.0:
            ulift[0] = 0.0
            ulift[1] = 0.0
            ulift[2] = 1.0
        else:
            for k in range(3):
                ic, jc = ICRS[k], JCRS[k]
                ulift[k] = (udrag[ic] * spn[jc] - udrag[jc] * spn[ic])
                for n in range(numax):
                    ulift_u[k, n] = (udrag_u[ic, n] * spn[jc] - udrag_u[jc, n] * spn[ic])
            ulmag = math.sqrt(ulift[0] * ulift[0] + ulift[1] * ulift[1] + ulift[2] * ulift[2])
            for n in range(numax):
                ulmag_u[n] = (
                    (ulift[0] * ulift_u[0, n] + ulift[1] * ulift_u[1, n] + ulift[2] * ulift_u[2, n]) / ulmag
                )
            for k in range(3):
                ulift[k] = (ulift[k] / ulmag)
                for n in range(numax):
                    ulift_u[k, n] = ((ulift_u[k, n] - (ulift[k] * ulmag_u[n])) / ulmag)

        # Use the strip 1/4 chord location for strip moments
        rc4[0] = state.rle[0, j] + (0.25 * cr)
        rc4[1] = state.rle[1, j]
        rc4[2] = state.rle[2, j]
        cfx = float(strip_cfx[j])
        cfy = float(strip_cfy[j])
        cfz = float(strip_cfz[j])
        cmx = float(strip_cmx[j])
        cmy = float(strip_cmy[j])
        cmz = float(strip_cmz[j])
        state.cnc[j] = strip_cnc[j]
        cfx_u[:] = strip_cfx_u[j, :numax]
        cfy_u[:] = strip_cfy_u[j, :numax]
        cfz_u[:] = strip_cfz_u[j, :numax]
        cmx_u[:] = strip_cmx_u[j, :numax]
        cmy_u[:] = strip_cmy_u[j, :numax]
        cmz_u[:] = strip_cmz_u[j, :numax]
        if ncontrol:
            cfx_d[:] = strip_cfx_d[j, :ncontrol]
            cfy_d[:] = strip_cfy_d[j, :ncontrol]
            cfz_d[:] = strip_cfz_d[j, :ncontrol]
            cmx_d[:] = strip_cmx_d[j, :ncontrol]
            cmy_d[:] = strip_cmy_d[j, :ncontrol]
            cmz_d[:] = strip_cmz_d[j, :ncontrol]
        if ndesign:
            cfx_g[:] = strip_cfx_g[j, :ndesign]
            cfy_g[:] = strip_cfy_g[j, :ndesign]
            cfz_g[:] = strip_cfz_g[j, :ndesign]
            cmx_g[:] = strip_cmx_g[j, :ndesign]
            cmy_g[:] = strip_cmy_g[j, :ndesign]
            cmz_g[:] = strip_cmz_g[j, :ndesign]
        state.cnc_u[j, :numax] = strip_cnc_u[j, :numax]
        if ncontrol:
            state.cnc_d[j, :ncontrol] = strip_cnc_d[j, :ncontrol]
        if ndesign:
            state.cnc_g[j, :ndesign] = strip_cnc_g[j, :ndesign]

        # Add h.v. forces from trailing legs lying on the wing surface
        if ltrforce:
            for ii in range(nvc):
                i = i1 + ii
                for ileg in range(2):
                    r = np.empty(3, dtype=np.float64)
                    rrot = np.empty(3, dtype=np.float64)
                    gleg = np.empty(3, dtype=np.float64)
                    if ileg == 0:
                        r[0] = (0.5 * (state.rv1[0, i] + xte1) - rc4[0])
                        r[1] = (state.rv1[1, i] - rc4[1])
                        r[2] = (state.rv1[2, i] - rc4[2])
                        rrot[0] = (0.5 * (state.rv1[0, i] + xte1) - state.xyzref[0])
                        rrot[1] = (state.rv1[1, i] - state.xyzref[1])
                        rrot[2] = (state.rv1[2, i] - state.xyzref[2])
                        gleg[0] = (state.rv1[0, i] - xte1)
                        gleg[1] = 0.0
                        gleg[2] = 0.0
                    else:
                        r[0] = (0.5 * (state.rv2[0, i] + xte2) - rc4[0])
                        r[1] = (state.rv2[1, i] - rc4[1])
                        r[2] = (state.rv2[2, i] - rc4[2])
                        rrot[0] = (0.5 * (state.rv2[0, i] + xte2) - state.xyzref[0])
                        rrot[1] = (state.rv2[1, i] - state.xyzref[1])
                        rrot[2] = (state.rv2[2, i] - state.xyzref[2])
                        gleg[0] = (xte2 - state.rv2[0, i])
                        gleg[1] = 0.0
                        gleg[2] = 0.0
                    # set total effective velocity = freestream + rotation
                    # (ignores h.v. induced contribution along trailing leg on wing)
                    vrot = cross(rrot, state.wrot)
                    veff = np.array((state.vinf + vrot), dtype=np.float64)
                    veff_u = np.zeros((3, numax), dtype=np.float64)
                    for k in range(3):
                        veff_u[k, k] = 1.0
                    for k in range(3, 6):
                        wrot_u = np.zeros(3, dtype=np.float64)
                        wrot_u[k - 3] = 1.0
                        veff_u[:, k] = cross(rrot, wrot_u)
                    f = cross(veff, gleg)
                    f_u = np.zeros((3, numax), dtype=np.float64)
                    for n in range(numax):
                        f_u[:, n] = cross(veff_u[:, n], gleg)
                    gi = _gam_scalar(gam, i)
                    fgam = (2.0) * gi * f
                    fgam_u = np.zeros((3, numax), dtype=np.float64)
                    for n in range(numax):
                        fgam_u[:, n] = (2.0 * (gam_u[i, n]) * f + 2.0 * gi * f_u[:, n])
                    dcfx = (fgam[0] / sr)
                    dcfy = (fgam[1] / sr)
                    dcfz = (fgam[2] / sr)
                    cfx = (cfx + dcfx)
                    cfy = (cfy + dcfy)
                    cfz = (cfz + dcfz)
                    cmx = (cmx + ((dcfz * r[1]) - (dcfy * r[2])) / cr)
                    cmy = (cmy + ((dcfx * r[2]) - (dcfz * r[0])) / cr)
                    cmz = (cmz + ((dcfy * r[0]) - (dcfx * r[1])) / cr)
                    state.cnc[j] = (state.cnc[j] + (cr * (state.ensy[j] * dcfy + state.ensz[j] * dcfz)))
                    for n in range(numax):
                        dcfx_u = (fgam_u[0, n] / sr)
                        dcfy_u = (fgam_u[1, n] / sr)
                        dcfz_u = (fgam_u[2, n] / sr)
                        cfx_u[n] = (cfx_u[n] + dcfx_u)
                        cfy_u[n] = (cfy_u[n] + dcfy_u)
                        cfz_u[n] = (cfz_u[n] + dcfz_u)
                        cmx_u[n] = (cmx_u[n] + ((dcfz_u * r[1]) - (dcfy_u * r[2])) / cr)
                        cmy_u[n] = (cmy_u[n] + ((dcfx_u * r[2]) - (dcfz_u * r[0])) / cr)
                        cmz_u[n] = (cmz_u[n] + ((dcfy_u * r[0]) - (dcfx_u * r[1])) / cr)
                        state.cnc_u[j, n] = (state.cnc_u[j, n] + (cr * (state.ensy[j] * dcfy_u + state.ensz[j] * dcfz_u)))

        # Drag terms due to viscous effects; CD from user-specified CD(CL) polar
        state.cdv_lstrp[j] = 0.0
        if state.lvisc and state.lviscstrp[j]:
            # Onset velocity at strip c/4 = freestream + rotation
            rrot = rc4 - state.xyzref
            vrot = cross(rrot, state.wrot)
            veff[:] = state.vinf + vrot
            veffmag = math.sqrt(veff[0] * veff[0] + veff[1] * veff[1] + veff[2] * veff[2])
            veff_u.fill(0.0)
            veff_u[0, 0] = 1.0
            veff_u[1, 1] = 1.0
            veff_u[2, 2] = 1.0
            veff_u[1, 3] = rrot[2]
            veff_u[2, 3] = -rrot[1]
            veff_u[0, 4] = -rrot[2]
            veff_u[2, 4] = rrot[0]
            veff_u[0, 5] = rrot[1]
            veff_u[1, 5] = -rrot[0]
            for n in range(numax):
                veffmag_u[n] = (
                    (veff[0] * veff_u[0, n] + veff[1] * veff_u[1, n] + veff[2] * veff_u[2, n]) / veffmag
                )
            clv = (ulift[0] * cfx + ulift[1] * cfy + ulift[2] * cfz)
            clv_u = (
                ulift[0] * cfx_u + ulift_u[0] * cfx
                + ulift[1] * cfy_u + ulift_u[1] * cfy
                + ulift[2] * cfz_u + ulift_u[2] * cfz
            )
            clv_d = (
                ulift[0] * cfx_d + ulift_d[0] * cfx
                + ulift[1] * cfy_d + ulift_d[1] * cfy
                + ulift[2] * cfz_d + ulift_d[2] * cfz
            )
            clv_g = (
                ulift[0] * cfx_g + ulift_g[0] * cfx
                + ulift[1] * cfy_g + ulift_g[1] * cfy
                + ulift[2] * cfz_g + ulift_g[2] * cfz
            )
            # Get CD from CLCD function using strip CL as parameter
            cdv, cdv_clv = cdcl(state.clcd[j, :], float(clv))
            # Strip viscous force contribution (per unit strip area)
            dcvfx = (veff[0] * veffmag * cdv)
            dcvfy = (veff[1] * veffmag * cdv)
            dcvfz = (veff[2] * veffmag * cdv)
            cfx = (cfx + dcvfx)
            cfy = (cfy + dcvfy)
            cfz = (cfz + dcvfz)
            state.cdv_lstrp[j] = (udrag[0] * dcvfx + udrag[1] * dcvfy + udrag[2] * dcvfz)
            dcvfx_u = (
                (veff_u[0] * veffmag + veff[0] * veffmag_u) * cdv
                + veff[0] * veffmag * cdv_clv * clv_u
            )
            dcvfy_u = (
                (veff_u[1] * veffmag + veff[1] * veffmag_u) * cdv
                + veff[1] * veffmag * cdv_clv * clv_u
            )
            dcvfz_u = (
                (veff_u[2] * veffmag + veff[2] * veffmag_u) * cdv
                + veff[2] * veffmag * cdv_clv * clv_u
            )
            cfx_u += dcvfx_u
            cfy_u += dcvfy_u
            cfz_u += dcvfz_u
            state.cnc_u[j, :numax] += cr * (state.ensy[j] * dcvfy_u + state.ensz[j] * dcvfz_u)
            dcvfx_d = veff[0] * veffmag * cdv_clv * clv_d
            dcvfy_d = veff[1] * veffmag * cdv_clv * clv_d
            dcvfz_d = veff[2] * veffmag * cdv_clv * clv_d
            cfx_d += dcvfx_d
            cfy_d += dcvfy_d
            cfz_d += dcvfz_d
            state.cnc_d[j, :ncontrol] += cr * (state.ensy[j] * dcvfy_d + state.ensz[j] * dcvfz_d)
            dcvfx_g = veff[0] * veffmag * cdv_clv * clv_g
            dcvfy_g = veff[1] * veffmag * cdv_clv * clv_g
            dcvfz_g = veff[2] * veffmag * cdv_clv * clv_g
            cfx_g += dcvfx_g
            cfy_g += dcvfy_g
            cfz_g += dcvfz_g
            state.cnc_g[j, :ndesign] += cr * (state.ensy[j] * dcvfy_g + state.ensz[j] * dcvfz_g)

        # At this point strip forces are in body axes at c/4, normalized by area/chord
        state.cf_lstrp[0, j], state.cf_lstrp[1, j], state.cf_lstrp[2, j] = cfx, cfy, cfz
        state.cm_lstrp[0, j], state.cm_lstrp[1, j], state.cm_lstrp[2, j] = cmx, cmy, cmz
        state.cfstrp[0, j], state.cfstrp[1, j], state.cfstrp[2, j] = cfx, cfy, cfz
        # Transform strip body axes forces into stability axes
        state.cdstrp[j] = (cfx * cosa + cfz * sina)
        state.cystrp[j] = cfy
        state.clstrp[j] = (-cfx * sina + cfz * cosa)
        state.cdst_a[j] = (-cfx * sina + cfz * cosa)
        state.cyst_a[j] = 0.0
        state.clst_a[j] = (-cfx * cosa - cfz * sina)
        state.cdst_u[j, :numax] = cfx_u * cosa + cfz_u * sina
        state.cyst_u[j, :numax] = cfy_u
        state.clst_u[j, :numax] = -cfx_u * sina + cfz_u * cosa
        state.cfst_u[0, j, :numax] = cfx_u
        state.cfst_u[1, j, :numax] = cfy_u
        state.cfst_u[2, j, :numax] = cfz_u
        if ncontrol:
            state.cdst_d[j, :ncontrol] = cfx_d * cosa + cfz_d * sina
            state.cyst_d[j, :ncontrol] = cfy_d
            state.clst_d[j, :ncontrol] = -cfx_d * sina + cfz_d * cosa
            state.cfst_d[0, j, :ncontrol] = cfx_d
            state.cfst_d[1, j, :ncontrol] = cfy_d
            state.cfst_d[2, j, :ncontrol] = cfz_d
        if ndesign:
            state.cdst_g[j, :ndesign] = cfx_g * cosa + cfz_g * sina
            state.cyst_g[j, :ndesign] = cfy_g
            state.clst_g[j, :ndesign] = -cfx_g * sina + cfz_g * cosa
            state.cfst_g[0, j, :ndesign] = cfx_g
            state.cfst_g[1, j, :ndesign] = cfy_g
            state.cfst_g[2, j, :ndesign] = cfz_g

        # vector from chord c/4 reference point to case reference point XYZREF
        rref = rc4 - state.xyzref
        # Strip moments in body axes about XYZREF, normalized by strip area/chord
        state.cmstrp[0, j] = (cmx + ((cfz * rref[1]) - (cfy * rref[2])) / cr)
        state.cmstrp[1, j] = (cmy + ((cfx * rref[2]) - (cfz * rref[0])) / cr)
        state.cmstrp[2, j] = (cmz + ((cfy * rref[0]) - (cfx * rref[1])) / cr)
        state.cmst_u[0, j, :numax] = cmx_u + ((cfz_u * rref[1]) - (cfy_u * rref[2])) / cr
        state.cmst_u[1, j, :numax] = cmy_u + ((cfx_u * rref[2]) - (cfz_u * rref[0])) / cr
        state.cmst_u[2, j, :numax] = cmz_u + ((cfy_u * rref[0]) - (cfx_u * rref[1])) / cr
        if ncontrol:
            state.cmst_d[0, j, :ncontrol] = cmx_d + ((cfz_d * rref[1]) - (cfy_d * rref[2])) / cr
            state.cmst_d[1, j, :ncontrol] = cmy_d + ((cfx_d * rref[2]) - (cfz_d * rref[0])) / cr
            state.cmst_d[2, j, :ncontrol] = cmz_d + ((cfy_d * rref[0]) - (cfx_d * rref[1])) / cr
        if ndesign:
            state.cmst_g[0, j, :ndesign] = cmx_g + ((cfz_g * rref[1]) - (cfy_g * rref[2])) / cr
            state.cmst_g[1, j, :ndesign] = cmy_g + ((cfx_g * rref[2]) - (cfz_g * rref[0])) / cr
            state.cmst_g[2, j, :ndesign] = cmz_g + ((cfy_g * rref[0]) - (cfx_g * rref[1])) / cr

        state.cl_lstrp[j] = (ulift[0] * cfx + ulift[1] * cfy + ulift[2] * cfz)
        state.cd_lstrp[j] = (udrag[0] * cfx + udrag[1] * cfy + udrag[2] * cfz)
        state.cmc4_lstrp[j] = (state.ensz[j] * cmy - state.ensy[j] * cmz)
        # CN,CA forces rotated to be in and normal to strip incidence (HHY bugfix 01102024)
        caxl0 = cfx
        cnrm0 = (state.ensy[j] * cfy + state.ensz[j] * cfz)
        sinainc = (np.sin((state.ainc[j])))
        cosainc = (np.cos((state.ainc[j])))
        state.ca_lstrp[j] = (caxl0 * cosainc - cnrm0 * sinainc)
        state.cn_lstrp[j] = (cnrm0 * cosainc + caxl0 * sinainc)

        # set total effective velocity = freestream + rotation
        rrot = np.array([state.xsref[j] - state.xyzref[0], state.ysref[j] - state.xyzref[1], state.zsref[j] - state.xyzref[2]], dtype=np.float64)
        vrot = cross(rrot, state.wrot)
        veff = np.array((state.vinf + vrot), dtype=np.float64)
        vsq = np.dot(veff, veff)
        vsqi = (1.0 if vsq == 0.0 else 1.0 / vsq)
        # spanwise and perpendicular velocity components
        vspan = (veff[0] * state.ess[0, j] + veff[1] * state.ess[1, j] + veff[2] * state.ess[2, j])
        vperp = (veff - state.ess[:, j] * vspan)
        vpsq = ((vperp[0] * vperp[0]) + (vperp[1] * vperp[1]) + (vperp[2] * vperp[2]))
        vpsqi = (1.0 if vpsq == 0.0 else 1.0 / vpsq)
        state.clt_lstrp[j] = (state.cl_lstrp[j] * vpsqi)
        state.cla_lstrp[j] = (state.cl_lstrp[j] * vsqi)

        # Moment about strip LE midpoint in direction of LE segment
        rle = rc4 - state.rle[:, j]
        delx = (state.rle2[0, j] - state.rle1[0, j])
        dely = (state.rle2[1, j] - state.rle1[1, j])
        delz = (state.rle2[2, j] - state.rle1[2, j])
        if state.imags[state.lssurf[j]] < 0:
            delx = (-delx)
            dely = (-dely)
            delz = (-delz)
        dmag = (np.sqrt(float((delx * delx) + (dely * dely) + (delz * delz))))
        state.cmle_lstrp[j] = 0.0
        if dmag != 0.0:
            state.cmle_lstrp[j] = (
                delx / dmag * (cmx + ((cfz * rle[1]) - (cfy * rle[2])) / cr)
                + dely / dmag * (cmy + ((cfx * rle[2]) - (cfz * rle[0])) / cr)
                + delz / dmag * (cmz + ((cfy * rle[0]) - (cfx * rle[1])) / cr)
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
        scale = np.where(exceeded, clmax / state.cl_lstrp[js], 1.0)
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
        if ndesign:
            state.cds_g[isurf, :ndesign] = 0.0
            state.cys_g[isurf, :ndesign] = 0.0
            state.cls_g[isurf, :ndesign] = 0.0
            state.cfs_g[:, isurf, :ndesign] = 0.0
            state.cms_g[:, isurf, :ndesign] = 0.0
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
        if ndesign:
            state.cds_g[isurf, :ndesign] = np.sum(
                state.cdst_g[js, :ndesign] * scale_sref[:, np.newaxis], axis=0
            )
            state.cys_g[isurf, :ndesign] = np.sum(
                state.cyst_g[js, :ndesign] * scale_sref[:, np.newaxis], axis=0
            )
            state.cls_g[isurf, :ndesign] = np.sum(
                state.clst_g[js, :ndesign] * scale_sref[:, np.newaxis], axis=0
            )
            state.cfs_g[:, isurf, :ndesign] = np.sum(
                state.cfst_g[:, js, :ndesign] * scale_sref[np.newaxis, :, np.newaxis], axis=1
            )
            state.cms_g[0, isurf, :ndesign] = np.sum(
                state.cmst_g[0, js, :ndesign] * scale_cm_b[:, np.newaxis], axis=0
            )
            state.cms_g[1, isurf, :ndesign] = np.sum(
                state.cmst_g[1, js, :ndesign] * scale_cm_c[:, np.newaxis], axis=0
            )
            state.cms_g[2, isurf, :ndesign] = np.sum(
                state.cmst_g[2, js, :ndesign] * scale_cm_b[:, np.newaxis], axis=0
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
        ulift = cross(udrag, spn)
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
            if ndesign:
                state.cdtot_g[:ndesign] += state.cds_g[isurf, :ndesign]
                state.cytot_g[:ndesign] += state.cys_g[isurf, :ndesign]
                state.cltot_g[:ndesign] += state.cls_g[isurf, :ndesign]
                state.cftot_g[:, :ndesign] += state.cfs_g[:, isurf, :ndesign]
                state.cmtot_g[:, :ndesign] += state.cms_g[:, isurf, :ndesign]

    return state


def aero(state: Any) -> Any:
    """Compute total aerodynamic coefficients (AERO).

    Calculate forces: inviscid from horseshoe vortices, inviscid from bodies,
    viscous from drag polars, and far-field Trefftz-plane contributions.
    """
    numax = state.numax
    ncontrol = state.ncontrol
    ndesign = state.ndesign

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
    if ndesign:
        state.cdtot_g[:ndesign] = 0.0
        state.cytot_g[:ndesign] = 0.0
        state.cltot_g[:ndesign] = 0.0
        state.cftot_g[:, :ndesign] = 0.0
        state.cmtot_g[:, :ndesign] = 0.0
        state.chinge_g[:ncontrol, :ndesign] = 0.0

    # Evaluate forces on surface, bodies and Trefftz plane
    sfforc(state)
    bdforc(state)
    tpforc(state)

    # If case is XZ symmetric (IYSYM=1), add contributions from images,
    # zero out the asymmetric forces and double the symmetric ones
    if state.iysym == 1:
        state.cdtot = (2.0 * state.cdtot)
        state.cytot = 0.0
        state.cltot = (2.0 * state.cltot)
        state.cftot[0] = (2.0 * state.cftot[0])
        state.cftot[1] = 0.0
        state.cftot[2] = (2.0 * state.cftot[2])
        state.cmtot[0] = 0.0
        state.cmtot[1] = (2.0 * state.cmtot[1])
        state.cmtot[2] = 0.0
        state.cdvtot = (2.0 * state.cdvtot)
        state.cdtot_a = (2.0 * state.cdtot_a)
        state.cltot_a = (2.0 * state.cltot_a)
        state.cdtot_u[:] = (2.0 * state.cdtot_u)
        state.cytot_u[:] = 0.0
        state.cltot_u[:] = (2.0 * state.cltot_u)
        state.cftot_u[0, :] = (2.0 * state.cftot_u[0, :])
        state.cftot_u[1, :] = 0.0
        state.cftot_u[2, :] = (2.0 * state.cftot_u[2, :])
        state.cmtot_u[0, :] = 0.0
        state.cmtot_u[1, :] = (2.0 * state.cmtot_u[1, :])
        state.cmtot_u[2, :] = 0.0
        if ncontrol:
            state.cdtot_d[:] = (2.0 * state.cdtot_d)
            state.cytot_d[:] = 0.0
            state.cltot_d[:] = (2.0 * state.cltot_d)
            state.cftot_d[0, :] = (2.0 * state.cftot_d[0, :])
            state.cftot_d[1, :] = 0.0
            state.cftot_d[2, :] = (2.0 * state.cftot_d[2, :])
            state.cmtot_d[0, :] = 0.0
            state.cmtot_d[1, :] = (2.0 * state.cmtot_d[1, :])
            state.cmtot_d[2, :] = 0.0
        if ndesign:
            state.cdtot_g[:] = (2.0 * state.cdtot_g)
            state.cytot_g[:] = 0.0
            state.cltot_g[:] = (2.0 * state.cltot_g)
            state.cftot_g[0, :] = (2.0 * state.cftot_g[0, :])
            state.cftot_g[1, :] = 0.0
            state.cftot_g[2, :] = (2.0 * state.cftot_g[2, :])
            state.cmtot_g[0, :] = 0.0
            state.cmtot_g[1, :] = (2.0 * state.cmtot_g[1, :])
            state.cmtot_g[2, :] = 0.0

    # add baseline reference CD to totals; force in direction of freestream
    vsq = ((state.vinf[0] * state.vinf[0]) + (state.vinf[1] * state.vinf[1]) + (state.vinf[2] * state.vinf[2]))
    vmag = (np.sqrt(float(vsq)))
    state.cdvtot = (state.cdvtot + (state.cdref * vsq))
    state.cdtot = (state.cdtot + (state.cdref * vsq))
    state.cytot = (state.cytot + (state.cdref * (state.vinf[1] * vmag)))
    for l in range(3):
        state.cftot[l] = (state.cftot[l] + (state.cdref * (state.vinf[l] * vmag)))
        state.cftot_u[l, l] = (state.cftot_u[l, l] + (state.cdref * vmag))
    for iu in range(3):
        state.cdtot_u[iu] = (state.cdtot_u[iu] + (state.cdref * (2.0 * state.vinf[iu])))
        for l in range(3):
            state.cftot_u[l, iu] = (
                state.cftot_u[l, iu] + (state.cdref * (state.vinf[l] * state.vinf[iu]) / vmag)
            )
    return state
