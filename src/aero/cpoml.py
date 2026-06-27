"""CPOML surface pressure coefficient post-processing (port of aoml.f)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from openavl.geom.geometry import solver_surface_name
from openavl.math.linalg import baksub, ludcmp

if TYPE_CHECKING:
    from openavl.core.state import AVLState
    from openavl.fileio.parser import AVLModel

_NCMAX = 256


@dataclass(frozen=True)
class CpomlSurfaceData:
    """Upper/lower OML mesh and absolute Cp for one lifting surface."""

    label: str
    isurf: int
    xyz: np.ndarray
    cp: np.ndarray


def srcpanel(x: float, z: float, x1: float, z1: float, x2: float, z2: float) -> tuple[float, float]:
    """Velocity induced by a 2D unit-strength constant-source panel."""
    den = math.sqrt((x2 - x1) ** 2 + (z2 - z1) ** 2)
    if den == 0.0:
        return 0.0, 0.0
    cs = (x2 - x1) / den
    sn = (z2 - z1) / den
    xp = cs * (x - x1) + sn * (z - z1)
    zp = -sn * (x - x1) + cs * (z - z1)
    pi = math.pi
    r1sq = (xp - 0.0) ** 2 + (zp - 0.0) ** 2
    r2sq = (xp - den) ** 2 + zp * zp
    th1 = math.atan2(zp, xp)
    th2 = math.atan2(zp, xp - den)
    up = math.log(r1sq / r2sq) / (4.0 * pi) if r1sq > 0.0 and r2sq > 0.0 else 0.0
    wp = (th2 - th1) / (2.0 * pi)
    u = cs * up - sn * wp
    w = sn * up + cs * wp
    return u, w


def cpthk(state: AVLState) -> None:
    """Compute thickness-based pressure coefficients ``state.cpt`` (CPTHK)."""
    state.cpt[: state.nvor] = 0.0
    aict = np.zeros((_NCMAX, _NCMAX), dtype=np.float64)
    bict = np.zeros((_NCMAX, _NCMAX), dtype=np.float64)
    rhs = np.zeros(_NCMAX, dtype=np.float64)
    work = np.zeros(_NCMAX, dtype=np.float64)
    indx = np.zeros(_NCMAX, dtype=np.int32)
    qsinf = np.zeros(_NCMAX, dtype=np.float64)
    srcthk = np.zeros(_NCMAX, dtype=np.float64)

    for jstrip in range(state.nstrip):
        i1 = int(state.ijfrst[jstrip])
        nvc = int(state.nvstrp[jstrip])
        if nvc <= 0 or nvc > _NCMAX:
            continue

        xle = 0.5 * (state.rle1[0, jstrip] + state.rle2[0, jstrip])
        zle = 0.5 * (state.rle1[2, jstrip] + state.rle2[2, jstrip])

        for ii in range(nvc):
            i = i1 + ii
            if ii == 0:
                x = 0.5 * (xle + 0.5 * (state.xyn1[0, i] + state.xyn2[0, i])) - xle
                zlo = 0.5 * (zle + 0.5 * (state.zlon1[i] + state.zlon2[i])) - zle
                zup = 0.5 * (zle + 0.5 * (state.zupn1[i] + state.zupn2[i])) - zle
            else:
                i_prev = i - 1
                x = 0.25 * (
                    state.xyn1[0, i_prev]
                    + state.xyn2[0, i_prev]
                    + state.xyn1[0, i]
                    + state.xyn2[0, i]
                ) - xle
                zlo = 0.25 * (
                    state.zlon1[i_prev]
                    + state.zlon2[i_prev]
                    + state.zlon1[i]
                    + state.zlon2[i]
                ) - zle
                zup = 0.25 * (
                    state.zupn1[i_prev]
                    + state.zupn2[i_prev]
                    + state.zupn1[i]
                    + state.zupn2[i]
                ) - zle
            z = 0.5 * (zup - zlo)

            if ii == 0:
                dx = 0.5 * (state.xyn1[0, i] + state.xyn2[0, i]) - xle
                dzlo = 0.5 * (state.zlon1[i] + state.zlon2[i]) - zle
                dzup = 0.5 * (state.zupn1[i] + state.zupn2[i]) - zle
            else:
                i_prev = i - 1
                dx = 0.5 * (
                    state.xyn1[0, i]
                    - state.xyn1[0, i_prev]
                    + state.xyn2[0, i]
                    - state.xyn2[0, i_prev]
                )
                dzlo = 0.5 * (state.zlon1[i] - state.zlon1[i_prev] + state.zlon2[i] - state.zlon2[i_prev])
                dzup = 0.5 * (state.zupn1[i] - state.zupn1[i_prev] + state.zupn2[i] - state.zupn2[i_prev])
            dz = 0.5 * (dzup - dzlo)
            den = math.sqrt(dx * dx + dz * dz)
            if den == 0.0:
                continue
            esx = dx / den
            esz = dz / den
            enx = -esz
            enz = esx

            rhs[ii] = enx * 1.0 + enz * 0.0
            qsinf[ii] = esx * 1.0 + esz * 0.0

            for jj in range(nvc):
                jv = i1 + jj
                if jj == 0:
                    x1 = 0.0
                    z1 = 0.0
                else:
                    j_prev = jv - 1
                    x1 = 0.5 * (state.xyn1[0, j_prev] + state.xyn2[0, j_prev]) - xle
                    z1lo = 0.5 * (state.zlon1[j_prev] + state.zlon2[j_prev]) - zle
                    z1up = 0.5 * (state.zupn1[j_prev] + state.zupn2[j_prev]) - zle
                    z1 = 0.5 * (z1up - z1lo)

                x2 = 0.5 * (state.xyn1[0, jv] + state.xyn2[0, jv]) - xle
                z2lo = 0.5 * (state.zlon1[jv] + state.zlon2[jv]) - zle
                z2up = 0.5 * (state.zupn1[jv] + state.zupn2[jv]) - zle
                z2 = 0.5 * (z2up - z2lo)

                if jj == ii:
                    u = 0.5 * enx
                    w = 0.5 * enz
                else:
                    u, w = srcpanel(x, z, x1, z1, x2, z2)
                aict[ii, jj] = enx * u + enz * w
                bict[ii, jj] = esx * u + esz * w

                u, w = srcpanel(x, z, x1, -z1, x2, -z2)
                aict[ii, jj] += enx * u + enz * w
                bict[ii, jj] += esx * u + esz * w

        ludcmp(aict, nvc, indx, work)
        baksub(aict, nvc, indx, rhs[:nvc])
        srcthk[:nvc] = -rhs[:nvc]

        for ii in range(nvc):
            i = i1 + ii
            qs = qsinf[ii]
            for jj in range(nvc):
                qs += bict[ii, jj] * srcthk[jj]
            state.cpt[i] = 1.0 - qs * qs


def _strip_span_order(state: AVLState, isurf: int) -> tuple[int, int, int]:
    """Return first strip, last strip, and step for a surface."""
    j0 = int(state.jfrst[isurf])
    nj = int(state.nj[isurf])
    if nj <= 1:
        return j0, j0 + nj - 1, 1
    j1 = j0 + nj - 1
    if state.rle1[1, j1] >= state.rle1[1, j0]:
        return j0, j1, 1
    return j1, j0, -1


def _rotate_surface_point(
    xle: float,
    yle: float,
    zle: float,
    x0: float,
    y0: float,
    z0: float,
    csd: float,
    snd: float,
    csa: float,
    sna: float,
) -> tuple[float, float, float]:
    """Rotate an airfoil point into global coordinates (CPDUMP)."""
    ylod = yle + (y0 - yle) * csd - (z0 - zle) * snd
    zlod = zle - (y0 - yle) * snd + (z0 - zle) * csd
    xout = xle + (x0 - xle) * csa + (zlod - zle) * sna
    zout = zle - (x0 - xle) * sna + (zlod - zle) * csa
    return xout, ylod, zout


def cpoml(state: AVLState, model: AVLModel | None = None) -> None:
    """Compute thickness Cp and validate CPOML prerequisites."""
    if model is not None:
        for isurf in range(int(state.nsurf)):
            if not bool(state.lrange[isurf]):
                raise ValueError(
                    "CPOML requires full-chord airfoil definitions on every section."
                )
    cpthk(state)


def collect_cpoml_surfaces(
    state: AVLState,
    model: AVLModel,
    *,
    component: int | None = None,
    load_only: bool = True,
) -> list[CpomlSurfaceData]:
    """Build upper/lower OML meshes and absolute surface Cp values."""
    if not bool(getattr(state, "lsol", False)):
        raise ValueError(
            "CPOML surface Cp requires a solved state; run execute_run() first."
        )

    cpoml(state, model)
    surfaces: list[CpomlSurfaceData] = []

    for isurf in range(int(state.nsurf)):
        if load_only and not bool(state.lfload[isurf]):
            continue
        if component is not None and int(state.lncomp[isurf]) != component:
            continue
        if not bool(state.lrange[isurf]):
            continue

        nj = int(state.nj[isurf])
        nvc = int(state.nk[isurf])
        if nj <= 0 or nvc <= 0:
            continue

        xyz = np.zeros((nj + 1, 2 * nvc + 1, 3), dtype=np.float64)
        cp = np.zeros((nj, 2 * nvc), dtype=np.float64)
        j0, j1, jstep = _strip_span_order(state, isurf)

        idx_strip = 0
        for j in range(j0, j1 + jstep, jstep):
            i1 = int(state.ijfrst[j])
            dyle = state.rle2[1, j] - state.rle1[1, j]
            dzle = state.rle2[2, j] - state.rle1[2, j]
            den_yz = math.sqrt(dyle * dyle + dzle * dzle)
            if den_yz == 0.0:
                csd = 1.0
                snd = 0.0
            else:
                csd = dyle / den_yz
                snd = dzle / den_yz

            csa = math.cos(state.ainc1[j])
            sna = math.sin(state.ainc1[j])
            xle = state.rle1[0, j]
            yle = state.rle1[1, j]
            zle = state.rle1[2, j]
            xyz[idx_strip, nvc, 0] = xle
            xyz[idx_strip, nvc, 1] = yle
            xyz[idx_strip, nvc, 2] = zle

            for ii in range(nvc):
                iv = i1 + ii
                x0 = state.xyn1[0, iv]
                y0 = state.xyn1[1, iv]
                zlo0 = state.zlon1[iv]
                zup0 = state.zupn1[iv]
                xlo, ylo, zlo = _rotate_surface_point(
                    xle, yle, zle, x0, y0, zlo0, csd, snd, csa, sna
                )
                xup, yup, zup = _rotate_surface_point(
                    xle, yle, zle, x0, y0, zup0, csd, snd, csa, sna
                )
                xyz[idx_strip, nvc - (ii + 1), :] = (xup, yup, zup)
                xyz[idx_strip, nvc + (ii + 1), :] = (xlo, ylo, zlo)

            idx_strip += 1

        j = j1
        csa = math.cos(state.ainc2[j])
        sna = math.sin(state.ainc2[j])
        xle = state.rle2[0, j]
        yle = state.rle2[1, j]
        zle = state.rle2[2, j]
        dy = yle - state.rle1[1, j]
        dz = zle - state.rle1[2, j]
        den_yz = math.sqrt(dy * dy + dz * dz)
        if den_yz == 0.0:
            csd = 1.0
            snd = 0.0
        else:
            csd = dy / den_yz
            snd = dz / den_yz
        xyz[idx_strip, nvc, 0] = xle
        xyz[idx_strip, nvc, 1] = yle
        xyz[idx_strip, nvc, 2] = zle
        i1 = int(state.ijfrst[j])
        for ii in range(nvc):
            iv = i1 + ii
            x0 = state.xyn2[0, iv]
            y0 = state.xyn2[1, iv]
            zlo0 = state.zlon2[iv]
            zup0 = state.zupn2[iv]
            xlo, ylo, zlo = _rotate_surface_point(
                xle, yle, zle, x0, y0, zlo0, csd, snd, csa, sna
            )
            xup, yup, zup = _rotate_surface_point(
                xle, yle, zle, x0, y0, zup0, csd, snd, csa, sna
            )
            xyz[idx_strip, nvc - (ii + 1), :] = (xup, yup, zup)
            xyz[idx_strip, nvc + (ii + 1), :] = (xlo, ylo, zlo)

        idx_strip = 0
        for j in range(j0, j1 + jstep, jstep):
            i1 = int(state.ijfrst[j])
            csa = math.cos(state.ainc[j])
            sna = math.sin(state.ainc[j])
            xle = 0.5 * (state.rle1[0, j] + state.rle2[0, j])
            yle = 0.5 * (state.rle1[1, j] + state.rle2[1, j])
            zle = 0.5 * (state.rle1[2, j] + state.rle2[2, j])
            iv0 = i1
            cpt0 = state.cpt[iv0]
            dcp0 = state.dcp[iv0]
            cp[idx_strip, nvc - 1] = cpt0 - 0.5 * dcp0
            cp[idx_strip, nvc] = cpt0 + 0.5 * dcp0

            for ii in range(1, nvc):
                iv = i1 + ii
                cpt_i = state.cpt[iv]
                dcp_i = state.dcp[iv]
                cp[idx_strip, nvc - 1 - ii] = cpt_i - 0.5 * dcp_i
                cp[idx_strip, nvc + ii] = cpt_i + 0.5 * dcp_i
            idx_strip += 1

        surfaces.append(
            CpomlSurfaceData(
                label=solver_surface_name(model, isurf),
                isurf=isurf,
                xyz=xyz,
                cp=cp,
            )
        )

    return surfaces
