"""Differentiable geometry update for the JAX AVL analysis pipeline."""

from __future__ import annotations

import math
import warnings
from typing import Any

import numpy as np

from openavl.geom.spacing import spacer
from openavl.jax.aic import vvor_jax
from openavl.jax.analysis import run_analysis, _make_forces_checkpoint
from openavl.jax.backend import jax, jnp
from openavl.jax.snapshot import _build_vortex_to_strip
from openavl.jax.types import (
    AnalysisGeometry,
    AnalysisResult,
    CirculationGeometry,
    FlowCondition,
    ForceGeometry,
    GeometryDesignParams,
    GeometryTopology,
    ReferenceQuantities,
    TrefftzGeometry,
)


_vvor_jax_remat = jax.checkpoint(
    vvor_jax,
    static_argnums=(0, 1, 2, 3, 4, 5, 6, 13),
)


def _replay_makesurf_topology(state: Any, model: Any) -> dict[str, np.ndarray]:
    """Replay ``makesurf`` strip loops to capture interpolation fractions and mappings."""
    nstrip = state.nstrip
    sec_left = np.zeros(nstrip, dtype=np.int32)
    sec_right = np.zeros(nstrip, dtype=np.int32)
    fc_arr = np.zeros(nstrip, dtype=np.float64)
    f1_arr = np.zeros(nstrip, dtype=np.float64)
    f2_arr = np.zeros(nstrip, dtype=np.float64)
    width_arr = np.zeros(nstrip, dtype=np.float64)
    tanle_slope = np.zeros(nstrip, dtype=np.float64)
    tante_slope = np.zeros(nstrip, dtype=np.float64)
    is_mirror = np.zeros(nstrip, dtype=bool)
    ydup_arr = np.zeros(nstrip, dtype=np.float64)
    imags_neg = np.zeros(nstrip, dtype=bool)
    model_surf_idx = np.zeros(nstrip, dtype=np.int32)
    xyzscal_x = np.ones(nstrip, dtype=np.float64)

    surf_sec_offset: list[int] = []
    surf_nsec: list[int] = []
    surf_xyzscal: list[np.ndarray] = []
    surf_xyztran: list[np.ndarray] = []
    surf_addinc: list[float] = []

    sec_offset = 0
    strip_cursor = 0

    for isurf_model, surf in enumerate(model.surfaces):
        surf_sec_offset.append(sec_offset)
        nsec = len(surf.sections)
        surf_nsec.append(nsec)
        xyzscal = np.asarray(surf.scale, dtype=np.float64)
        xyztran = np.asarray(surf.translate, dtype=np.float64)
        surf_xyzscal.append(xyzscal)
        surf_xyztran.append(xyztran)
        surf_addinc.append((surf.angle_deg or 0.0) * state.dtr)
        sec_offset += nsec

        nsec_surf = nsec
        if nsec_surf < 2:
            continue

        nvs1 = surf.n_span or 0
        sspace = surf.s_space
        nspans = [s.n_span or 0 for s in surf.sections]
        sspaces = [s.s_space if s.s_space is not None else sspace for s in surf.sections]
        xyzles = [(s.xle, s.yle, s.zle) for s in surf.sections]
        chords = [s.chord for s in surf.sections]
        imags = surf.imags if surf.imags else 1

        yzlen = np.zeros(nsec_surf, dtype=np.float64)
        for isec in range(1, nsec_surf):
            dy = xyzles[isec][1] - xyzles[isec - 1][1]
            dz = xyzles[isec][2] - xyzles[isec - 1][2]
            yzlen[isec] = yzlen[isec - 1] + math.sqrt(dy * dy + dz * dz)

        ypt = np.zeros(502, dtype=np.float64)
        ycp = np.zeros(502, dtype=np.float64)
        iptloc = np.zeros(nsec_surf + 1, dtype=np.int32)

        if nvs1 == 0:
            nvs = sum(nspans[: nsec_surf - 1])
            ypt[0] = yzlen[0]
            iptloc[0] = 0
            nvs_acc = 0
            for isec in range(nsec_surf - 1):
                dyzlen = yzlen[isec + 1] - yzlen[isec]
                nvint = nspans[isec]
                nspace = 2 * nvint + 1
                fspace = spacer(nspace, sspaces[isec])
                for n in range(nvint):
                    ivs = nvs_acc + n
                    ycp[ivs] = ypt[nvs_acc] + dyzlen * fspace[2 * n + 2]
                    ypt[ivs + 1] = ypt[nvs_acc] + dyzlen * fspace[2 * n + 3]
                iptloc[isec + 1] = nvs_acc + nvint
                nvs_acc += nvint
            nvs = nvs_acc
        else:
            nspace = 2 * nvs1 + 1
            fspace = spacer(nspace, sspace)
            ypt[0] = yzlen[0]
            for ivs in range(nvs1):
                ycp[ivs] = yzlen[0] + (yzlen[nsec_surf - 1] - yzlen[0]) * fspace[2 * ivs + 2]
                ypt[ivs + 1] = yzlen[0] + (yzlen[nsec_surf - 1] - yzlen[0]) * fspace[2 * ivs + 3]
            npt = nvs1 + 1
            for isec in range(1, nsec_surf - 1):
                best = 1e9
                best_idx = 0
                for ipt in range(npt):
                    d = abs(yzlen[isec] - ypt[ipt])
                    if d < best:
                        best = d
                        best_idx = ipt
                iptloc[isec] = best_idx
            iptloc[0] = 0
            iptloc[nsec_surf - 1] = npt - 1

            for isec in range(1, nsec_surf - 1):
                ipt1 = iptloc[isec - 1]
                ipt2 = iptloc[isec]
                if ipt1 == ipt2:
                    continue
                ypt1 = ypt[ipt1]
                denom = ypt[ipt2] - ypt1
                yscale = (yzlen[isec] - yzlen[isec - 1]) / denom if denom != 0.0 else 1.0
                for ipt in range(ipt1, ipt2):
                    ypt[ipt] = yzlen[isec - 1] + yscale * (ypt[ipt] - ypt1)
                for ivs in range(ipt1, ipt2):
                    ycp[ivs] = yzlen[isec - 1] + yscale * (ycp[ivs] - ypt1)

                ipt1 = iptloc[isec]
                ipt2 = iptloc[isec + 1]
                if ipt1 == ipt2:
                    continue
                ypt1 = ypt[ipt1]
                denom = ypt[ipt2] - ypt1
                yscale = (ypt[ipt2] - yzlen[isec]) / denom if denom != 0.0 else 1.0
                for ipt in range(ipt1, ipt2):
                    ypt[ipt] = yzlen[isec] + yscale * (ypt[ipt] - ypt1)
                for ivs in range(ipt1, ipt2):
                    ycp[ivs] = yzlen[isec] + yscale * (ycp[ivs] - ypt1)

        def _record_strip(
            j: int,
            isec: int,
            f1: float,
            f2: float,
            fc: float,
            width: float,
            xyzle_l: np.ndarray,
            xyzle_r: np.ndarray,
            chord_l: float,
            chord_r: float,
            *,
            mirror: bool,
            ydup: float,
        ) -> None:
            nonlocal strip_cursor
            if j >= nstrip:
                return
            sec_left[j] = surf_sec_offset[isurf_model] + isec
            sec_right[j] = surf_sec_offset[isurf_model] + isec + 1
            f1_arr[j] = f1
            f2_arr[j] = f2
            fc_arr[j] = fc
            width_arr[j] = width
            tanle_slope[j] = (xyzle_r[0] - xyzle_l[0]) / width if width != 0.0 else 0.0
            tante_slope[j] = (xyzle_r[0] + chord_r - xyzle_l[0] - chord_l) / width if width != 0.0 else 0.0
            is_mirror[j] = mirror
            ydup_arr[j] = ydup
            imags_neg[j] = imags < 0
            model_surf_idx[j] = isurf_model
            xyzscal_x[j] = xyzscal[0]
            strip_cursor += 1

        base_strip_start = strip_cursor

        xyzle_l = np.zeros(3, dtype=np.float64)
        xyzle_r = np.zeros(3, dtype=np.float64)

        for isec in range(nsec_surf - 1):
            xyzle_l[0] = xyzscal[0] * xyzles[isec][0] + xyztran[0]
            xyzle_l[1] = xyzscal[1] * xyzles[isec][1] + xyztran[1]
            xyzle_l[2] = xyzscal[2] * xyzles[isec][2] + xyztran[2]
            xyzle_r[0] = xyzscal[0] * xyzles[isec + 1][0] + xyztran[0]
            xyzle_r[1] = xyzscal[1] * xyzles[isec + 1][1] + xyztran[1]
            xyzle_r[2] = xyzscal[2] * xyzles[isec + 1][2] + xyztran[2]
            width = math.sqrt((xyzle_r[1] - xyzle_l[1]) ** 2 + (xyzle_r[2] - xyzle_l[2]) ** 2)
            if not math.isfinite(width) or width == 0.0:
                continue

            chord_l = xyzscal[0] * chords[isec]
            chord_r = xyzscal[0] * chords[isec + 1]

            ipt_l = iptloc[isec]
            ipt_r = iptloc[isec + 1]
            nspan = ipt_r - ipt_l
            if nspan <= 0:
                continue

            for ispan in range(1, nspan + 1):
                ipt1 = ipt_l + ispan - 1
                ipt2 = ipt_l + ispan
                ivs = ipt_l + ispan - 1
                denom = ypt[ipt_r] - ypt[ipt_l]
                f1 = (ypt[ipt1] - ypt[ipt_l]) / denom if denom != 0.0 else 0.0
                f2 = (ypt[ipt2] - ypt[ipt_l]) / denom if denom != 0.0 else 0.0
                fc = (ycp[ivs] - ypt[ipt_l]) / denom if denom != 0.0 else 0.0
                j = strip_cursor
                _record_strip(j, isec, f1, f2, fc, width, xyzle_l, xyzle_r, chord_l, chord_r, mirror=False, ydup=0.0)

        if surf.yduplicate is not None:
            ydup = float(surf.yduplicate)
            nj_base = strip_cursor - base_strip_start
            for jj in range(nj_base):
                j_base = base_strip_start + jj
                j_mirror = strip_cursor + jj
                if j_mirror >= nstrip:
                    break
                sec_left[j_mirror] = sec_left[j_base]
                sec_right[j_mirror] = sec_right[j_base]
                f1_arr[j_mirror] = f1_arr[j_base]
                f2_arr[j_mirror] = f2_arr[j_base]
                fc_arr[j_mirror] = fc_arr[j_base]
                width_arr[j_mirror] = width_arr[j_base]
                tanle_slope[j_mirror] = -tanle_slope[j_base]
                tante_slope[j_mirror] = tante_slope[j_base]
                is_mirror[j_mirror] = True
                ydup_arr[j_mirror] = ydup
                imags_neg[j_mirror] = not imags_neg[j_base]
                model_surf_idx[j_mirror] = isurf_model
                xyzscal_x[j_mirror] = xyzscal_x[j_base]
            strip_cursor += nj_base

    if strip_cursor != nstrip:
        raise ValueError(
            f"topology strip count mismatch: recorded {strip_cursor}, expected {nstrip}"
        )

    return {
        "sec_left": sec_left,
        "sec_right": sec_right,
        "fc": fc_arr,
        "f1": f1_arr,
        "f2": f2_arr,
        "width": width_arr,
        "tanle_slope": tanle_slope,
        "tante_slope": tante_slope,
        "is_mirror": is_mirror,
        "ydup": ydup_arr,
        "imags_neg": imags_neg,
        "model_surf_idx": model_surf_idx,
        "xyzscal_x": xyzscal_x,
        "surf_sec_offset": np.asarray(surf_sec_offset, dtype=np.int32),
        "surf_nsec": np.asarray(surf_nsec, dtype=np.int32),
        "surf_xyzscal": np.asarray(surf_xyzscal, dtype=np.float64),
        "surf_xyztran": np.asarray(surf_xyztran, dtype=np.float64),
        "surf_addinc": np.asarray(surf_addinc, dtype=np.float64),
    }


def _kutta_stripoff_indices(state: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Precompute Kutta-row and strip-off vortex indices for AIC rebuild."""
    kutta_iv: list[int] = []
    kutta_j1: list[int] = []
    kutta_j2: list[int] = []
    stripoff_iv: list[int] = []

    nvor = state.nvor
    for n in range(state.nsurf):
        if state.lfwake[n]:
            continue
        j1 = int(state.jfrst[n])
        jn = j1 + int(state.nj[n]) - 1
        for j in range(j1, jn + 1):
            i1 = int(state.ijfrst[j])
            iv = int(state.ijfrst[j] + state.nvstrp[j] - 1)
            kutta_iv.append(iv)
            kutta_j1.append(i1)
            kutta_j2.append(iv)

    for j in range(state.nstrip):
        if not state.lstripoff[j]:
            continue
        i1 = int(state.ijfrst[j])
        for k in range(int(state.nvstrp[j])):
            stripoff_iv.append(i1 + k)

    return (
        np.asarray(kutta_iv, dtype=np.int32),
        np.asarray(kutta_j1, dtype=np.int32),
        np.asarray(kutta_j2, dtype=np.int32),
        np.asarray(stripoff_iv, dtype=np.int32),
    )


def snapshot_topology(state: Any, model: Any) -> GeometryTopology:
    """Capture fixed topology and interpolation data from a built NumPy solver state."""
    nvor = state.nvor
    nstrip = state.nstrip
    nsurf = state.nsurf

    topo_arrays = _replay_makesurf_topology(state, model)

    vortex_to_strip = _build_vortex_to_strip(state)
    xvr = np.zeros(nvor, dtype=np.float64)
    xcp = np.zeros(nvor, dtype=np.float64)
    for i in range(nvor):
        j = int(vortex_to_strip[i])
        chord1 = float(state.chord1[j])
        chord2 = float(state.chord2[j])
        chord_c = float(state.chord[j])
        if chord1 != 0.0:
            xvr[i] = (float(state.rv1[0, i]) - float(state.rle1[0, j])) / chord1
        if chord_c != 0.0:
            xcp[i] = (float(state.rc[0, i]) - float(state.rle[0, j])) / chord_c

    kutta_iv, kutta_j1, kutta_j2, stripoff_iv = _kutta_stripoff_indices(state)

    chordv_snap = state.chordv[:nvor]
    dxv_frac = np.where(
        chordv_snap != 0.0, state.dxv[:nvor] / np.where(chordv_snap != 0.0, chordv_snap, 1.0), 0.0
    )

    betm = state.betm
    if betm == 0.0:
        betm = float(np.sqrt(1.0 - state.mach * state.mach))

    n_sections = int(topo_arrays["surf_sec_offset"][-1] + topo_arrays["surf_nsec"][-1]) if len(model.surfaces) else 0

    return GeometryTopology(
        nstrip=nstrip,
        nvor=nvor,
        nsurf=nsurf,
        n_sections=n_sections,
        saxfr=float(state.saxfr),
        betm=float(betm),
        iysym=int(state.iysym),
        izsym=int(state.izsym),
        ysym=float(state.ysym),
        zsym=float(state.zsym),
        vrcorec=float(state.vrcorec),
        vrcorew=float(state.vrcorew),
        sec_left=jnp.asarray(topo_arrays["sec_left"][:nstrip], dtype=jnp.int32),
        sec_right=jnp.asarray(topo_arrays["sec_right"][:nstrip], dtype=jnp.int32),
        fc=jnp.asarray(topo_arrays["fc"][:nstrip], dtype=jnp.float64),
        f1=jnp.asarray(topo_arrays["f1"][:nstrip], dtype=jnp.float64),
        f2=jnp.asarray(topo_arrays["f2"][:nstrip], dtype=jnp.float64),
        width=jnp.asarray(topo_arrays["width"][:nstrip], dtype=jnp.float64),
        tanle_slope=jnp.asarray(topo_arrays["tanle_slope"][:nstrip], dtype=jnp.float64),
        tante_slope=jnp.asarray(topo_arrays["tante_slope"][:nstrip], dtype=jnp.float64),
        is_mirror=jnp.asarray(topo_arrays["is_mirror"][:nstrip], dtype=bool),
        ydup=jnp.asarray(topo_arrays["ydup"][:nstrip], dtype=jnp.float64),
        imags_neg=jnp.asarray(topo_arrays["imags_neg"][:nstrip], dtype=bool),
        model_surf_idx=jnp.asarray(topo_arrays["model_surf_idx"][:nstrip], dtype=jnp.int32),
        xyzscal_x=jnp.asarray(topo_arrays["xyzscal_x"][:nstrip], dtype=jnp.float64),
        surf_sec_offset=jnp.asarray(topo_arrays["surf_sec_offset"], dtype=jnp.int32),
        surf_nsec=jnp.asarray(topo_arrays["surf_nsec"], dtype=jnp.int32),
        surf_xyzscal=jnp.asarray(topo_arrays["surf_xyzscal"], dtype=jnp.float64),
        surf_xyztran=jnp.asarray(topo_arrays["surf_xyztran"], dtype=jnp.float64),
        surf_addinc=jnp.asarray(topo_arrays["surf_addinc"], dtype=jnp.float64),
        vortex_to_strip=jnp.asarray(vortex_to_strip, dtype=jnp.int32),
        xvr=jnp.asarray(xvr, dtype=jnp.float64),
        xcp=jnp.asarray(xcp, dtype=jnp.float64),
        slopec=jnp.asarray(state.slopec[:nvor], dtype=jnp.float64),
        slopev=jnp.asarray(state.slopev[:nvor], dtype=jnp.float64),
        lvcomp=jnp.asarray(state.lvcomp[:nvor], dtype=jnp.int32),
        lstripoff=jnp.asarray(state.lstripoff[:nstrip], dtype=bool),
        lfwake=jnp.asarray(state.lfwake[:nsurf], dtype=bool),
        jfrst=jnp.asarray(state.jfrst[:nsurf], dtype=jnp.int32),
        nj=jnp.asarray(state.nj[:nsurf], dtype=jnp.int32),
        ijfrst=jnp.asarray(state.ijfrst[:nstrip], dtype=jnp.int32),
        nvstrp=jnp.asarray(state.nvstrp[:nstrip], dtype=jnp.int32),
        kutta_iv=jnp.asarray(kutta_iv, dtype=jnp.int32),
        kutta_j1=jnp.asarray(kutta_j1, dtype=jnp.int32),
        kutta_j2=jnp.asarray(kutta_j2, dtype=jnp.int32),
        stripoff_iv=jnp.asarray(stripoff_iv, dtype=jnp.int32),
        dxv_frac=jnp.asarray(dxv_frac, dtype=jnp.float64),
    )


def design_params_from_state(state: Any, model: Any) -> GeometryDesignParams:
    """Build baseline section design parameters from a built solver state."""
    aincs: list[float] = []
    chords: list[float] = []
    xles: list[float] = []
    yles: list[float] = []
    zles: list[float] = []
    for surf in model.surfaces:
        for sec in surf.sections:
            aincs.append(sec.ainc_deg * state.dtr)
            chords.append(sec.chord)
            xles.append(sec.xle)
            yles.append(sec.yle)
            zles.append(sec.zle)
    return GeometryDesignParams(
        aincs=jnp.asarray(aincs, dtype=jnp.float64),
        chords=jnp.asarray(chords, dtype=jnp.float64),
        xles=jnp.asarray(xles, dtype=jnp.float64),
        yles=jnp.asarray(yles, dtype=jnp.float64),
        zles=jnp.asarray(zles, dtype=jnp.float64),
    )


def _scaled_section_le(
    params: GeometryDesignParams,
    topo: GeometryTopology,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Scaled/translated leading-edge coordinates for all sections."""
    n_model = topo.surf_nsec.shape[0]
    nsec = params.xles.shape[0]
    x_out = jnp.zeros(nsec, dtype=jnp.float64)
    y_out = jnp.zeros(nsec, dtype=jnp.float64)
    z_out = jnp.zeros(nsec, dtype=jnp.float64)

    for ms in range(n_model):
        off = int(topo.surf_sec_offset[ms])
        n = int(topo.surf_nsec[ms])
        scal = topo.surf_xyzscal[ms]
        tran = topo.surf_xyztran[ms]
        idx = slice(off, off + n)
        x_out = x_out.at[idx].set(scal[0] * params.xles[idx] + tran[0])
        y_out = y_out.at[idx].set(scal[1] * params.yles[idx] + tran[1])
        z_out = z_out.at[idx].set(scal[2] * params.zles[idx] + tran[2])

    return x_out, y_out, z_out


def _interpolate_strips(
    params: GeometryDesignParams,
    topo: GeometryTopology,
) -> tuple[
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
]:
    """Interpolate section parameters to strip-level chords, LE positions, and incidence."""
    sec_l = topo.sec_left
    sec_r = topo.sec_right
    fc = topo.fc
    f1 = topo.f1
    f2 = topo.f2
    scal_x = topo.xyzscal_x
    addinc = topo.surf_addinc[topo.model_surf_idx]

    chord_l = scal_x * params.chords[sec_l]
    chord_r = scal_x * params.chords[sec_r]
    ainc_l = params.aincs[sec_l] + addinc
    ainc_r = params.aincs[sec_r] + addinc
    chsin_l = chord_l * jnp.sin(ainc_l)
    chsin_r = chord_r * jnp.sin(ainc_r)
    chcos_l = chord_l * jnp.cos(ainc_l)
    chcos_r = chord_r * jnp.cos(ainc_r)

    chord1 = (1.0 - f1) * chord_l + f1 * chord_r
    chord2 = (1.0 - f2) * chord_l + f2 * chord_r
    chord = (1.0 - fc) * chord_l + fc * chord_r

    x_sec, y_sec, z_sec = _scaled_section_le(params, topo)
    xyzle_l = jnp.stack([x_sec[sec_l], y_sec[sec_l], z_sec[sec_l]], axis=0)
    xyzle_r = jnp.stack([x_sec[sec_r], y_sec[sec_r], z_sec[sec_r]], axis=0)

    rle1 = (1.0 - f1)[None, :] * xyzle_l + f1[None, :] * xyzle_r
    rle2 = (1.0 - f2)[None, :] * xyzle_l + f2[None, :] * xyzle_r
    rle = (1.0 - fc)[None, :] * xyzle_l + fc[None, :] * xyzle_r

    # IMAGS<0 edge reversal on the original surface half only (not YDUPLICATE mirrors).
    non_mirror = jnp.logical_not(topo.is_mirror)
    swap = jnp.logical_and(topo.imags_neg, non_mirror)
    rle1_nm = jnp.where(swap[None, :], rle2, rle1)
    rle2_nm = jnp.where(swap[None, :], rle1, rle2)
    chord1_nm = jnp.where(swap, chord2, chord1)
    chord2_nm = jnp.where(swap, chord1, chord2)

    # YDUPLICATE mirror strips: replay ``sdupl`` from the base-half interpolation.
    yoff = 2.0 * topo.ydup
    mirror = topo.is_mirror
    rle1_m = jnp.stack([rle2[0], -rle2[1] + yoff, rle2[2]], axis=0)
    rle2_m = jnp.stack([rle1[0], -rle1[1] + yoff, rle1[2]], axis=0)
    rle_m = jnp.stack([rle[0], -rle[1] + yoff, rle[2]], axis=0)
    chord1_m = chord2
    chord2_m = chord1

    rle1 = jnp.where(mirror[None, :], rle1_m, rle1_nm)
    rle2 = jnp.where(mirror[None, :], rle2_m, rle2_nm)
    rle = jnp.where(mirror[None, :], rle_m, rle)
    chord1 = jnp.where(mirror, chord1_m, chord1_nm)
    chord2 = jnp.where(mirror, chord2_m, chord2_nm)

    chsin = chsin_l + fc * (chsin_r - chsin_l)
    chcos = chcos_l + fc * (chcos_r - chcos_l)
    ainc = jnp.arctan2(chsin, chcos)

    # A8: recompute wstrip from the live interpolated strip edges (spanwise
    # distance between rle1/rle2) instead of scaling the frozen baseline
    # `topo.width`, so d(CL, CD)/d(yle, zle) picks up the area/span-growth
    # term when sections move spanwise.
    dy_edge = rle2[1] - rle1[1]
    dz_edge = rle2[2] - rle1[2]
    wstrip = jnp.sqrt(dy_edge * dy_edge + dz_edge * dz_edge)

    # tanle/tante (LE/TE sweep slopes) are carried through from the frozen
    # topology snapshot. They are not currently propagated into ForceGeometry
    # or consumed by the force-integration pipeline, so leaving them fixed
    # has no effect on today's outputs; documented here per A8 rather than
    # recomputed, since they are dead outputs of this function.
    tanle = topo.tanle_slope
    tante = topo.tante_slope

    return rle1, rle2, rle, chord, chord1, chord2, ainc, wstrip, tanle, tante


def _build_vortex_positions(
    rle1: jnp.ndarray,
    rle2: jnp.ndarray,
    rle: jnp.ndarray,
    chord: jnp.ndarray,
    chord1: jnp.ndarray,
    chord2: jnp.ndarray,
    topo: GeometryTopology,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Build vortex horseshoe endpoints and control points from strip geometry."""
    v2s = topo.vortex_to_strip
    xvr = topo.xvr
    xcp = topo.xcp

    rv1 = jnp.zeros((3, topo.nvor), dtype=jnp.float64)
    rv2 = jnp.zeros((3, topo.nvor), dtype=jnp.float64)
    rv = jnp.zeros((3, topo.nvor), dtype=jnp.float64)
    rc = jnp.zeros((3, topo.nvor), dtype=jnp.float64)
    chordv = jnp.zeros(topo.nvor, dtype=jnp.float64)

    c1 = chord1[v2s]
    c2 = chord2[v2s]
    cc = chord[v2s]

    rv1 = rv1.at[0].set(rle1[0, v2s] + xvr * c1)
    rv1 = rv1.at[1].set(rle1[1, v2s])
    rv1 = rv1.at[2].set(rle1[2, v2s])
    rv2 = rv2.at[0].set(rle2[0, v2s] + xvr * c2)
    rv2 = rv2.at[1].set(rle2[1, v2s])
    rv2 = rv2.at[2].set(rle2[2, v2s])
    rv = rv.at[0].set(rle[0, v2s] + xvr * cc)
    rv = rv.at[1].set(rle[1, v2s])
    rv = rv.at[2].set(rle[2, v2s])
    rc = rc.at[0].set(rle[0, v2s] + xcp * cc)
    rc = rc.at[1].set(rle[1, v2s])
    rc = rc.at[2].set(rle[2, v2s])
    chordv = cc

    # A8: dxv (chordwise vortex spacing used to normalize dCp) is a fixed
    # fraction of chordv set by the chordwise mesh spacing at build time;
    # recompute it from the live chordv so it tracks chord design changes.
    dxv = topo.dxv_frac * chordv

    return rv1, rv2, rv, rc, chordv, dxv


def encalc_jax(
    rv1: jnp.ndarray,
    rv2: jnp.ndarray,
    rv: jnp.ndarray,
    ainc: jnp.ndarray,
    slopec: jnp.ndarray,
    slopev: jnp.ndarray,
    wstrip: jnp.ndarray,
    topo: GeometryTopology,
) -> tuple[
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
]:
    """Compute panel normals and strip axes (JAX port of ``encalc``).

    Replaces the original Python for-loop over strips with fully vectorised
    JAX array operations so JAX traces a single compact graph instead of
    nstrip unrolled copies.
    """
    saxfr = jnp.asarray(topo.saxfr, dtype=jnp.float64)

    # --- Strip-level LE and TE vortex indices ---
    i0 = jnp.asarray(topo.ijfrst, dtype=jnp.int32)  # [nstrip]
    i1 = jnp.clip(
        i0 + jnp.asarray(topo.nvstrp, dtype=jnp.int32) - 1, 0, topo.nvor - 1
    )  # [nstrip]

    # Vortex half-span vectors at LE (first vortex) and TE (last vortex) of each strip
    rv1_le = rv1[:, i0]  # [3, nstrip]
    rv2_le = rv2[:, i0]
    rv_le = rv[:, i0]
    rv1_te = rv1[:, i1]  # [3, nstrip]
    rv2_te = rv2[:, i1]
    rv_te = rv[:, i1]

    dxle = rv2_le[0] - rv1_le[0]  # [nstrip]
    dyle = rv2_le[1] - rv1_le[1]
    dzle = rv2_le[2] - rv1_le[2]
    axle = rv_le[0]
    ayle = rv_le[1]
    azle = rv_le[2]

    dxte = rv2_te[0] - rv1_te[0]
    dyte = rv2_te[1] - rv1_te[1]
    dzte = rv2_te[2] - rv1_te[2]
    axte = rv_te[0]
    ayte = rv_te[1]
    azte = rv_te[2]

    dxt = (1.0 - saxfr) * dxle + saxfr * dxte  # [nstrip]
    dyt = (1.0 - saxfr) * dyle + saxfr * dyte
    dzt = (1.0 - saxfr) * dzle + saxfr * dzte
    dmag = jnp.sqrt(dxt * dxt + dyt * dyt + dzt * dzt)
    yzmag = jnp.sqrt(dyt * dyt + dzt * dzt)

    degenerate = jnp.logical_or(
        jnp.logical_or(dmag == 0.0, yzmag == 0.0),
        jnp.logical_or(wstrip == 0.0, jnp.logical_not(jnp.isfinite(dmag))),
    )  # [nstrip]

    dmag_safe = jnp.where(dmag > 0.0, dmag, 1.0)
    yzmag_safe = jnp.where(yzmag > 0.0, yzmag, 1.0)

    ess = jnp.where(
        degenerate[None, :],
        0.0,
        jnp.stack([dxt / dmag_safe, dyt / dmag_safe, dzt / dmag_safe], axis=0),
    )  # [3, nstrip]
    ensy_s = jnp.where(degenerate, 0.0, -dzt / yzmag_safe)  # [nstrip]
    ensz_s = jnp.where(degenerate, 0.0, dyt / yzmag_safe)
    xsref = jnp.where(degenerate, 0.0, (1.0 - saxfr) * axle + saxfr * axte)
    ysref = jnp.where(degenerate, 0.0, (1.0 - saxfr) * ayle + saxfr * ayte)
    zsref = jnp.where(degenerate, 0.0, (1.0 - saxfr) * azle + saxfr * azte)

    # --- Vortex-level normals via vortex_to_strip broadcast ---
    v2s = jnp.asarray(topo.vortex_to_strip, dtype=jnp.int32)  # [nvor]
    deg_v = degenerate[v2s]  # [nvor]
    ensy_v = ensy_s[v2s]
    ensz_v = ensz_s[v2s]
    ainc_v = ainc[v2s]

    # Per-vortex bound-vortex direction
    dxb = rv2[0] - rv1[0]  # [nvor]
    dyb = rv2[1] - rv1[1]
    dzb = rv2[2] - rv1[2]
    emag = jnp.sqrt(dxb * dxb + dyb * dyb + dzb * dzb)
    emag_safe = jnp.where(emag > 0.0, emag, 1.0)
    ebx = dxb / emag_safe
    eby = dyb / emag_safe
    ebz = dzb / emag_safe

    # Control-point normal (enc)
    ang_c = ainc_v - jnp.arctan(slopec)
    sinc_c = jnp.sin(ang_c)
    cosc_c = jnp.cos(ang_c)
    ec_x = cosc_c
    ec_y = -sinc_c * ensy_v
    ec_z = -sinc_c * ensz_v
    cx = ec_y * ebz - ec_z * eby
    cy = ec_z * ebx - ec_x * ebz
    cz = ec_x * eby - ec_y * ebx
    em = jnp.sqrt(cx * cx + cy * cy + cz * cz)
    em_safe = jnp.where(em > 0.0, em, 1.0)
    enc_x = jnp.where(deg_v, 0.0, jnp.where(em > 0.0, cx / em_safe, 0.0))
    enc_y = jnp.where(deg_v, 0.0, jnp.where(em > 0.0, cy / em_safe, 0.0))
    enc_z = jnp.where(deg_v, ensz_v, jnp.where(em > 0.0, cz / em_safe, ensz_v))
    enc = jnp.stack([enc_x, enc_y, enc_z], axis=0)  # [3, nvor]

    # Vortex midpoint normal (env)
    ang_v = ainc_v - jnp.arctan(slopev)
    sinc_v = jnp.sin(ang_v)
    cosc_v = jnp.cos(ang_v)
    ecv_x = cosc_v
    ecv_y = -sinc_v * ensy_v
    ecv_z = -sinc_v * ensz_v
    cxv = ecv_y * ebz - ecv_z * eby
    cyv = ecv_z * ebx - ecv_x * ebz
    czv = ecv_x * eby - ecv_y * ebx
    emv = jnp.sqrt(cxv * cxv + cyv * cyv + czv * czv)
    emv_safe = jnp.where(emv > 0.0, emv, 1.0)
    env_x = jnp.where(deg_v, 0.0, jnp.where(emv > 0.0, cxv / emv_safe, 0.0))
    env_y = jnp.where(deg_v, 0.0, jnp.where(emv > 0.0, cyv / emv_safe, 0.0))
    env_z = jnp.where(deg_v, ensz_v, jnp.where(emv > 0.0, czv / emv_safe, ensz_v))
    env = jnp.stack([env_x, env_y, env_z], axis=0)  # [3, nvor]

    # A7: `lvnc` must additionally be forced False at the Kutta-condition row
    # (Sigma-gamma=0 equation on lfwake=False surfaces) and the strip-off
    # identity rows, mirroring `core/setup.py`'s explicit overrides after its
    # own degeneracy-based `lvnc` pass. These rows get special 0..1..0 /
    # identity treatment in `rebuild_aicn_jax`, so the RHS mask must zero the
    # same rows or the circulation solve enforces the wrong equation there.
    lvnc = jnp.logical_not(deg_v)  # [nvor]
    lvnc = lvnc.at[topo.kutta_iv].set(False)
    lvnc = lvnc.at[topo.stripoff_iv].set(False)

    return enc, env, ess, ensy_s, ensz_s, xsref, ysref, zsref, lvnc


def rebuild_aicn_jax(
    wc_gam: jnp.ndarray,
    enc: jnp.ndarray,
    topo: GeometryTopology,
) -> jnp.ndarray:
    """Rebuild unfactored AIC matrix from influence coefficients and panel normals."""
    aicn = jnp.einsum("kij,ki->ij", wc_gam, enc)
    nv = aicn.shape[1]

    # Kutta rows: replace each row with 0…1…0 spanning j1..j2.
    n_kutta = int(topo.kutta_iv.shape[0])
    if n_kutta > 0:
        kutta_iv = jnp.asarray(topo.kutta_iv, dtype=jnp.int32)
        kutta_j1 = jnp.asarray(topo.kutta_j1, dtype=jnp.int32)
        kutta_j2 = jnp.asarray(topo.kutta_j2, dtype=jnp.int32)
        j_range = jnp.arange(nv)[None, :]  # [1, nv]
        kutta_rows = (
            (j_range >= kutta_j1[:, None]) & (j_range <= kutta_j2[:, None])
        ).astype(jnp.float64)  # [n_kutta, nv]
        aicn = aicn.at[kutta_iv, :].set(kutta_rows)

    # Strip-off rows: identity row (1 at diagonal, 0 elsewhere).
    n_stripoff = int(topo.stripoff_iv.shape[0])
    if n_stripoff > 0:
        stripoff_iv = jnp.asarray(topo.stripoff_iv, dtype=jnp.int32)
        j_range_so = jnp.arange(nv)[None, :]  # [1, nv]
        stripoff_rows = (j_range_so == stripoff_iv[:, None]).astype(
            jnp.float64
        )  # [n_stripoff, nv]
        aicn = aicn.at[stripoff_iv, :].set(stripoff_rows)

    return aicn


def update_geometry(
    topo: GeometryTopology,
    params: GeometryDesignParams,
    baseline: AnalysisGeometry,
    *,
    mach: jnp.ndarray | float | None = None,
) -> AnalysisGeometry:
    """Update analysis geometry from section-level design parameters.

    A8 status of geometry-dependent quantities not recomputed here:
    ``wcsrd_u``/``wvsrd_u`` (body influence at moving control points and
    vortex midpoints) and ``enc_d`` (control-surface normal sensitivities)
    remain frozen at the baseline snapshot -- recomputing them requires
    re-deriving body source/doublet or hinge-normal sensitivities inside the
    traced path, which is out of scope for this fix. A warning is raised
    below when a body or control surface is present so callers know those
    gradients are incomplete.
    ``ssurf``/``cavesurf`` (per-surface area/average-chord, carried in
    ``ForceGeometry`` from the baseline) are likewise left frozen; today's
    JAX force integration only uses their array shape, not their values, so
    this has no effect on current outputs.

    When ``mach`` is supplied, the Prandtl--Glauert factor ``betm`` for the
    vortex influence assembly is derived from that live Mach rather than the
    frozen ``topo.betm`` captured at snapshot time. This keeps the
    geometry-AD path consistent with :func:`run_analysis`, which rebuilds the
    lattice matrices from ``flow.mach`` whenever it differs from the snapshot.
    """
    rle1, rle2, rle, chord, chord1, chord2, ainc, wstrip, tanle, tante = _interpolate_strips(
        params, topo
    )
    rv1, rv2, rv, rc, chordv, dxv = _build_vortex_positions(
        rle1, rle2, rle, chord, chord1, chord2, topo
    )

    if baseline.body.nl.shape[0] > 0:
        warnings.warn(
            "update_geometry: wcsrd_u/wvsrd_u (body source/doublet influence "
            "at control points and vortex midpoints) is frozen at the "
            "baseline snapshot; gradients of body-carrying models w.r.t. "
            "geometry design variables are incomplete.",
            stacklevel=2,
        )
    if int(baseline.circulation.ncontrol) > 0:
        warnings.warn(
            "update_geometry: enc_d (control-surface normal sensitivities) "
            "is frozen at the baseline snapshot; gradients of control "
            "derivatives w.r.t. geometry design variables are incomplete.",
            stacklevel=2,
        )

    enc, env, ess, ensy, ensz, xsref, ysref, zsref, lvnc = encalc_jax(
        rv1, rv2, rv, ainc, topo.slopec, topo.slopev, wstrip, topo
    )

    if mach is None:
        betm = float(topo.betm)
    else:
        betm = jnp.sqrt(1.0 - mach * mach)

    wc_gam = _vvor_jax_remat(
        betm,
        int(topo.iysym),
        float(topo.ysym),
        int(topo.izsym),
        float(topo.zsym),
        float(topo.vrcorec),
        float(topo.vrcorew),
        rv1,
        rv2,
        topo.lvcomp,
        chordv,
        rc,
        topo.lvcomp,
        False,
    )
    wv_gam = _vvor_jax_remat(
        betm,
        int(topo.iysym),
        float(topo.ysym),
        int(topo.izsym),
        float(topo.zsym),
        float(topo.vrcorec),
        float(topo.vrcorew),
        rv1,
        rv2,
        topo.lvcomp,
        chordv,
        rv,
        topo.lvcomp,
        True,
    )
    aicn = rebuild_aicn_jax(wc_gam, enc, topo)

    base_circ = baseline.circulation
    circulation = CirculationGeometry(
        rc=rc,
        enc=enc,
        enc_d=base_circ.enc_d,  # A8: frozen at baseline; see update_geometry docstring.
        aicn=aicn,
        wc_gam=wc_gam,
        wv_gam=wv_gam,
        wcsrd_u=base_circ.wcsrd_u,  # A8: frozen at baseline; see update_geometry docstring.
        wvsrd_u=base_circ.wvsrd_u,  # A8: frozen at baseline, same as wcsrd_u.
        lvnc=lvnc,
        lvalbe=base_circ.lvalbe,
        numax=base_circ.numax,
        ncontrol=base_circ.ncontrol,
    )

    base_force = baseline.force
    smap = base_force.strip_map
    force = ForceGeometry(
        rv1=rv1,
        rv2=rv2,
        rv=rv,
        rc=rc,
        env=env,
        dxv=dxv,
        rle=rle,
        rle1=rle1,
        rle2=rle2,
        chord=chord,
        chord1=chord1,
        chord2=chord2,
        wstrip=wstrip,
        ensy=ensy,
        ensz=ensz,
        ess=ess,
        ainc=ainc,
        xsref=xsref,
        ysref=ysref,
        zsref=zsref,
        ssurf=base_force.ssurf,  # A8: frozen at baseline; see update_geometry docstring.
        cavesurf=base_force.cavesurf,  # A8: frozen at baseline; see update_geometry docstring.
        imags=base_force.imags,
        lfload=base_force.lfload,
        clcd=base_force.clcd,
        strip_map=smap,
        nbody=base_force.nbody,
    )

    base_trefftz = baseline.trefftz
    trefftz = TrefftzGeometry(
        rv1=rv1,
        rv2=rv2,
        rc=rc,
        chord=chord,
        strip_map=smap,
        iysym=base_trefftz.iysym,
        izsym=base_trefftz.izsym,
        ysym=base_trefftz.ysym,
        zsym=base_trefftz.zsym,
        vrcorec=base_trefftz.vrcorec,
        vrcorew=base_trefftz.vrcorew,
        lfload=base_trefftz.lfload,
        amach=base_trefftz.amach,
    )

    return AnalysisGeometry(
        circulation=circulation,
        force=force,
        body=baseline.body,
        trefftz=trefftz,
    )


def run_analysis_with_geometry(
    flow: FlowCondition,
    params: GeometryDesignParams,
    topo: GeometryTopology,
    baseline: AnalysisGeometry,
    refs: ReferenceQuantities,
) -> AnalysisResult:
    """Compose differentiable geometry update with full force analysis."""
    forces_checkpoint = _make_forces_checkpoint(
        iysym=int(topo.iysym),
        include_body=baseline.body.nl.shape[0] > 0,
    )
    geom = update_geometry(topo, params, baseline, mach=flow.mach)
    return run_analysis(
        flow,
        geom,
        refs,
        use_checkpoint=True,
        forces_checkpoint=forces_checkpoint,
    )


__all__ = [
    "design_params_from_state",
    "encalc_jax",
    "rebuild_aicn_jax",
    "run_analysis_with_geometry",
    "snapshot_topology",
    "update_geometry",
]
