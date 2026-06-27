"""Surface and body geometry construction (port of amake.f).

Builds vortex-lattice panels from input sections, optionally Y-duplicates
surfaces, constructs body line nodes, and computes panel normal vectors.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from openavl.geom.spacing import akima, cspacer, spacer

if TYPE_CHECKING:
    from openavl.fileio.parser import BodyDef, SurfaceDef


def makesurf(state, isurf: int, surf: SurfaceDef) -> None:
    """Build vortex lattice panels for one surface (MAKESURF).

    Interpolates between input sections to define strips and horseshoe
    vortices, including control-surface geometry and viscous polar data.
    """
    nsec = len(surf.sections)
    if nsec < 2:
        return

    nvc1 = surf.n_chord or 1
    cspace = surf.c_space
    nvs1 = surf.n_span or 0
    sspace = surf.s_space
    xyzscal = np.asarray(surf.scale, dtype=np.float64)
    xyztran = np.asarray(surf.translate, dtype=np.float64)
    addinc = (surf.angle_deg or 0.0) * state.dtr

    ncontrol = state.ncontrol
    xyzles = [(s.xle, s.yle, s.zle) for s in surf.sections]
    chords = [s.chord for s in surf.sections]
    aincs = [s.ainc_deg * state.dtr for s in surf.sections]
    sspaces = [s.s_space if s.s_space is not None else sspace for s in surf.sections]
    nspans = [s.n_span or 0 for s in surf.sections]

    nasec = []
    xasec = []
    sasec = []
    casec = []
    tasec = []
    for sec in surf.sections:
        cam = sec.airfoil_camber
        if cam is not None and cam.x.size > 1:
            nasec.append(cam.x.size)
            xasec.append(cam.x)
            sasec.append(cam.s)
            casec.append(cam.c)
            tasec.append(cam.t)
        else:
            nasec.append(0)
            xasec.append(np.array([], dtype=np.float64))
            sasec.append(np.array([], dtype=np.float64))
            casec.append(np.array([], dtype=np.float64))
            tasec.append(np.array([], dtype=np.float64))

    # CPOML requires full-chord airfoil definitions on every section.
    state.lrange[isurf] = all(sec.claf >= 0.99 for sec in surf.sections)

    claf = [sec.claf for sec in surf.sections]
    clcdsec = []
    for sec in surf.sections:
        if sec.cdcl and len(sec.cdcl) >= 6:
            clcdsec.append(np.asarray(sec.cdcl[:6], dtype=np.float64))
        else:
            clcdsec.append(np.zeros(6, dtype=np.float64))

    nscon = [len(s.controls) for s in surf.sections]
    icontd = [[c.index for c in s.controls] for s in surf.sections]
    gaind = [[c.gain for c in s.controls] for s in surf.sections]
    xhinged = [[c.xhinge for c in s.controls] for s in surf.sections]
    vhinged = [[c.vhinge for c in s.controls] for s in surf.sections]
    refld = [[c.sgn_dup for c in s.controls] for s in surf.sections]

    # IMAGS: +1 root at edge 1, -1 root at edge 2 (reflected/duplicated surfaces).
    imags = surf.imags if surf.imags else 1
    state.imags[isurf] = imags
    comp_index = surf.component if surf.component else (isurf + 1)
    state.ifrst[isurf] = state.nvor
    state.jfrst[isurf] = state.nstrip
    state.nk[isurf] = nvc1

    # Arc-length positions of sections along wing trace in y-z plane.
    yzlen = np.zeros(nsec, dtype=np.float64)
    for isec in range(1, nsec):
        dy = xyzles[isec][1] - xyzles[isec - 1][1]
        dz = xyzles[isec][2] - xyzles[isec - 1][2]
        yzlen[isec] = yzlen[isec - 1] + math.sqrt(dy * dy + dz * dz)

    nvs = nvs1
    ypt = np.zeros(502, dtype=np.float64)
    ycp = np.zeros(502, dtype=np.float64)
    iptloc = np.zeros(nsec + 1, dtype=np.int32)

    if nvs1 == 0:
        # Per-section-interval spanwise spacing when surface NVS is not set.
        nvs = sum(nspans[: nsec - 1])
        ypt[0] = yzlen[0]
        iptloc[0] = 0
        nvs_acc = 0
        for isec in range(nsec - 1):
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
        # Global surface spanwise spacing, then fudge nodes to align with sections
        # so controls do not bridge vortex strips.
        nspace = 2 * nvs + 1
        fspace = spacer(nspace, sspace)
        ypt[0] = yzlen[0]
        for ivs in range(nvs):
            ycp[ivs] = yzlen[0] + (yzlen[nsec - 1] - yzlen[0]) * fspace[2 * ivs + 2]
            ypt[ivs + 1] = yzlen[0] + (yzlen[nsec - 1] - yzlen[0]) * fspace[2 * ivs + 3]
        npt = nvs + 1
        for isec in range(1, nsec - 1):
            best = 1e9
            best_idx = 0
            for ipt in range(npt):
                d = abs(yzlen[isec] - ypt[ipt])
                if d < best:
                    best = d
                    best_idx = ipt
            iptloc[isec] = best_idx
        iptloc[0] = 0
        iptloc[nsec - 1] = npt - 1

        # Rescale spacing between interior sections so nodes match section LEs exactly.
        for isec in range(1, nsec - 1):
            ipt1 = iptloc[isec - 1]
            ipt2 = iptloc[isec]
            if ipt1 == ipt2:
                raise ValueError(
                    f"makesurf: insufficient spanwise vortices near section {isec + 1} "
                    f"on surface {surf.name}"
                )
            ypt1 = ypt[ipt1]
            denom = ypt[ipt2] - ypt[ipt1]
            yscale = (yzlen[isec] - yzlen[isec - 1]) / denom if denom != 0.0 else 1.0
            for ipt in range(ipt1, ipt2):
                ypt[ipt] = yzlen[isec - 1] + yscale * (ypt[ipt] - ypt1)
            for ivs in range(ipt1, ipt2):
                ycp[ivs] = yzlen[isec - 1] + yscale * (ycp[ivs] - ypt1)

            ipt1 = iptloc[isec]
            ipt2 = iptloc[isec + 1]
            if ipt1 == ipt2:
                raise ValueError(
                    f"makesurf: insufficient spanwise vortices near section {isec + 1} "
                    f"on surface {surf.name}"
                )
            ypt1 = ypt[ipt1]
            denom = ypt[ipt2] - ypt[ipt1]
            yscale = (ypt[ipt2] - yzlen[isec]) / denom if denom != 0.0 else 1.0
            for ipt in range(ipt1, ipt2):
                ypt[ipt] = yzlen[isec] + yscale * (ypt[ipt] - ypt1)
            for ivs in range(ipt1, ipt2):
                ycp[ivs] = yzlen[isec] + yscale * (ycp[ivs] - ypt1)

    state.nj[isurf] = 0

    # Reusable workspace (avoid per-section / per-strip allocations).
    xyzle_l = np.zeros(3, dtype=np.float64)
    xyzle_r = np.zeros(3, dtype=np.float64)
    iscon_l = np.zeros(ncontrol + 1, dtype=np.int32)
    iscon_r = np.zeros(ncontrol + 1, dtype=np.int32)
    vh = np.zeros(3, dtype=np.float64)
    gainda = np.zeros(ncontrol, dtype=np.float64)
    xled_arr = np.zeros(ncontrol, dtype=np.float64)
    xted_arr = np.zeros(ncontrol, dtype=np.float64)

    # --- Define strips between input sections ---
    loop_guard = 0
    for isec in range(nsec - 1):
        # Left/right section LE positions with scale and translation applied.
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
        claf_l = claf[isec]
        claf_r = claf[isec + 1]
        ainc_l = aincs[isec] + addinc
        ainc_r = aincs[isec + 1] + addinc
        chsin_l = chord_l * math.sin(ainc_l)
        chsin_r = chord_r * math.sin(ainc_r)
        chcos_l = chord_l * math.cos(ainc_l)
        chcos_r = chord_r * math.cos(ainc_r)

        # Map control declarations on left/right sections to global control indices.
        iscon_l.fill(0)
        iscon_r.fill(0)
        for n in range(1, ncontrol + 1):
            for iscon in range(nscon[isec]):
                if icontd[isec][iscon] == n - 1:
                    iscon_l[n] = iscon + 1
            for iscon in range(nscon[isec + 1]):
                if icontd[isec + 1][iscon] == n - 1:
                    iscon_r[n] = iscon + 1

        ipt_l = iptloc[isec]
        ipt_r = iptloc[isec + 1]
        nspan = ipt_r - ipt_l
        if nspan <= 0:
            continue

        for ispan in range(1, nspan + 1):
            loop_guard += 1
            if loop_guard > 1_000_000:
                raise RuntimeError("makesurf loop guard tripped")

            # Spanwise interpolation fractions between section LE arc positions.
            ipt1 = ipt_l + ispan - 1
            ipt2 = ipt_l + ispan
            ivs = ipt_l + ispan - 1
            denom = ypt[ipt_r] - ypt[ipt_l]
            f1 = (ypt[ipt1] - ypt[ipt_l]) / denom if denom != 0.0 else 0.0
            f2 = (ypt[ipt2] - ypt[ipt_l]) / denom if denom != 0.0 else 0.0
            fc = (ycp[ivs] - ypt[ipt_l]) / denom if denom != 0.0 else 0.0

            j = state.nstrip
            state.nstrip += 1
            state.nj[isurf] += 1

            rle1 = (1.0 - f1) * xyzle_l + f1 * xyzle_r
            rle2 = (1.0 - f2) * xyzle_l + f2 * xyzle_r
            rle = (1.0 - fc) * xyzle_l + fc * xyzle_r
            if imags < 0:
                # Reverse strip edges so positive Gamma sense is preserved.
                rle1, rle2 = rle2.copy(), rle1.copy()

            state.rle1[:, j] = rle1
            state.rle2[:, j] = rle2
            state.rle[:, j] = rle

            chord1 = (1.0 - f1) * chord_l + f1 * chord_r
            chord2 = (1.0 - f2) * chord_l + f2 * chord_r
            if imags < 0:
                chord1, chord2 = chord2, chord1
            state.chord1[j] = chord1
            state.chord2[j] = chord2
            state.chord[j] = (1.0 - fc) * chord_l + fc * chord_r

            state.wstrip[j] = abs(f2 - f1) * width
            state.tanle[j] = (xyzle_r[0] - xyzle_l[0]) / width
            if imags < 0:
                state.tanle[j] = -state.tanle[j]
            state.tante[j] = (xyzle_r[0] + chord_r - xyzle_l[0] - chord_l) / width

            # Incidence from ATAN of chord projections, not linear AINC interpolation.
            chsin = chsin_l + fc * (chsin_r - chsin_l)
            chcos = chcos_l + fc * (chcos_r - chcos_l)
            state.ainc[j] = math.atan2(chsin, chcos)
            chsin1 = (1.0 - f1) * chsin_l + f1 * chsin_r
            chcos1 = (1.0 - f1) * chcos_l + f1 * chcos_r
            chsin2 = (1.0 - f2) * chsin_l + f2 * chsin_r
            chcos2 = (1.0 - f2) * chcos_l + f2 * chcos_r
            state.ainc1[j] = math.atan2(chsin1, chcos1)
            state.ainc2[j] = math.atan2(chsin2, chcos2)

            state.ijfrst[j] = state.nvor
            state.nvstrp[j] = nvc1
            state.lssurf[j] = isurf

            chord_c = state.chord[j]
            claf_c = (1.0 - fc) * (chord_l / chord_c) * claf_l + fc * (chord_r / chord_c) * claf_r
            # Chordwise vortex/control/source point spacing fractions.
            xpt, xvr, xsr, xcp_arr = cspacer(nvc1, cspace, claf_c)

            # Strip-level control-surface geometry (independent of chordwise index).
            gainda.fill(0.0)
            xled_arr.fill(0.0)
            xted_arr.fill(0.0)
            for n in range(1, ncontrol + 1):
                icl = iscon_l[n]
                icr = iscon_r[n]
                if icl == 0 or icr == 0:
                    continue
                ni = n - 1
                gainda[ni] = (
                    gaind[isec][icl - 1] * (1.0 - fc)
                    + gaind[isec + 1][icr - 1] * fc
                )
                xhd = (
                    chord_l * xhinged[isec][icl - 1] * (1.0 - fc)
                    + chord_r * xhinged[isec + 1][icr - 1] * fc
                )
                if xhd >= 0.0:
                    # TE control surface: hinge at xhd, deflects to trailing edge.
                    xled_arr[ni], xted_arr[ni] = xhd, chord_c
                else:
                    # LE control surface: hinge at -xhd, deflects from leading edge.
                    xled_arr[ni], xted_arr[ni] = 0.0, -xhd
                vh[:] = np.asarray(vhinged[isec][icl - 1], dtype=np.float64) * xyzscal
                vsq = float(np.dot(vh, vh))
                if vsq == 0.0:
                    # Default hinge vector along hingeline between sections.
                    vh[0] = (
                        xyzles[isec + 1][0]
                        + abs(chord_r * xhinged[isec + 1][icr - 1])
                        - (xyzles[isec][0] + abs(chord_l * xhinged[isec][icl - 1]))
                    ) * xyzscal[0]
                    vh[1] = (xyzles[isec + 1][1] - xyzles[isec][1]) * xyzscal[1]
                    vh[2] = (xyzles[isec + 1][2] - xyzles[isec][2]) * xyzscal[2]
                    vsq = float(np.dot(vh, vh))
                vmod = math.sqrt(vsq) if vsq > 0.0 else 1.0
                state.vhinge[:, j, ni] = vh / vmod
                state.vrefl[j, ni] = refld[isec][icl - 1]
                if xhd >= 0.0:
                    state.phinge[0, j, ni] = state.rle[0, j] + xhd
                    state.phinge[1, j, ni] = state.rle[1, j]
                    state.phinge[2, j, ni] = state.rle[2, j]
                else:
                    state.phinge[0, j, ni] = state.rle[0, j] - xhd
                    state.phinge[1, j, ni] = state.rle[1, j]
                    state.phinge[2, j, ni] = state.rle[2, j]

            # Interpolate CD-CL polar data from input sections to strip.
            state.clcd[j, :] = (1.0 - fc) * clcdsec[isec] + fc * clcdsec[isec + 1]
            state.lviscstrp[j] = state.clcd[j, 3] != 0.0

            # All chordwise vortices on this strip at once.
            ivc_idx = np.arange(1, nvc1 + 1, dtype=np.int32)
            i_start = state.nvor
            i_arr = np.arange(i_start, i_start + nvc1, dtype=np.int32)
            state.nvor += nvc1

            xvr_v = xvr[ivc_idx]
            xcp_v = xcp_arr[ivc_idx]
            xsr_v = xsr[ivc_idx]

            # Horseshoe endpoints, midpoint, control point, and source point.
            state.rv1[0, i_arr] = state.rle1[0, j] + xvr_v * state.chord1[j]
            state.rv1[1, i_arr] = state.rle1[1, j]
            state.rv1[2, i_arr] = state.rle1[2, j]
            state.rv2[0, i_arr] = state.rle2[0, j] + xvr_v * state.chord2[j]
            state.rv2[1, i_arr] = state.rle2[1, j]
            state.rv2[2, i_arr] = state.rle2[2, j]
            state.rv[0, i_arr] = state.rle[0, j] + xvr_v * chord_c
            state.rv[1, i_arr] = state.rle[1, j]
            state.rv[2, i_arr] = state.rle[2, j]
            state.rc[0, i_arr] = state.rle[0, j] + xcp_v * chord_c
            state.rc[1, i_arr] = state.rle[1, j]
            state.rc[2, i_arr] = state.rle[2, j]
            state.rs[0, i_arr] = state.rle[0, j] + xsr_v * chord_c
            state.rs[1, i_arr] = state.rle[1, j]
            state.rs[2, i_arr] = state.rle[2, j]

            # Camber slope at control point and vortex midpoint (Akima interp).
            s_l = np.zeros(nvc1, dtype=np.float64)
            s_r = np.zeros(nvc1, dtype=np.float64)
            sv_l = np.zeros(nvc1, dtype=np.float64)
            sv_r = np.zeros(nvc1, dtype=np.float64)
            if nasec[isec] > 1:
                for k, ivc in enumerate(ivc_idx):
                    s_l[k], _ = akima(xasec[isec], sasec[isec], xcp_arr[ivc])
                    _, sv_l[k] = akima(xasec[isec], sasec[isec], xvr[ivc])
            if nasec[isec + 1] > 1:
                for k, ivc in enumerate(ivc_idx):
                    s_r[k], _ = akima(xasec[isec + 1], sasec[isec + 1], xcp_arr[ivc])
                    _, sv_r[k] = akima(xasec[isec + 1], sasec[isec + 1], xvr[ivc])
            scl_l = (chord_l / chord_c) * (1.0 - fc)
            scl_r = (chord_r / chord_c) * fc
            state.slopec[i_arr] = scl_l * s_l + scl_r * s_r
            state.slopev[i_arr] = scl_l * sv_l + scl_r * sv_r

            # CPOML aft-node coordinates at panel trailing edges (xpt[ivc+1]).
            xpt_nodes = xpt[ivc_idx + 1]
            zl_l = np.zeros(nvc1, dtype=np.float64)
            zu_l = np.zeros(nvc1, dtype=np.float64)
            zl_r = np.zeros(nvc1, dtype=np.float64)
            zu_r = np.zeros(nvc1, dtype=np.float64)
            if nasec[isec] > 1:
                zlasec_l = casec[isec] - 0.5 * tasec[isec]
                zuasec_l = casec[isec] + 0.5 * tasec[isec]
                for k, xp in enumerate(xpt_nodes):
                    zl_l[k], _ = akima(xasec[isec], zlasec_l, float(xp))
                    zu_l[k], _ = akima(xasec[isec], zuasec_l, float(xp))
            if nasec[isec + 1] > 1:
                zlasec_r = casec[isec + 1] - 0.5 * tasec[isec + 1]
                zuasec_r = casec[isec + 1] + 0.5 * tasec[isec + 1]
                for k, xp in enumerate(xpt_nodes):
                    zl_r[k], _ = akima(xasec[isec + 1], zlasec_r, float(xp))
                    zu_r[k], _ = akima(xasec[isec + 1], zuasec_r, float(xp))

            state.xyn1[0, i_arr] = state.rle1[0, j] + xpt_nodes * state.chord1[j]
            state.xyn1[1, i_arr] = state.rle1[1, j]
            state.xyn2[0, i_arr] = state.rle2[0, j] + xpt_nodes * state.chord2[j]
            state.xyn2[1, i_arr] = state.rle2[1, j]
            state.zlon1[i_arr] = state.rle1[2, j] + zl_l * state.chord1[j]
            state.zupn1[i_arr] = state.rle1[2, j] + zu_l * state.chord1[j]
            state.zlon2[i_arr] = state.rle2[2, j] + zl_r * state.chord2[j]
            state.zupn2[i_arr] = state.rle2[2, j] + zu_r * state.chord2[j]

            dxoc = xpt[ivc_idx + 1] - xpt[ivc_idx]
            state.dxv[i_arr] = dxoc * chord_c
            state.chordv[i_arr] = chord_c
            state.lvcomp[i_arr] = comp_index
            state.lvalbe[i_arr] = surf.lvalbe

            if ncontrol > 0:
                # Scale control gain by fraction of element on control surface (0..1).
                xpt_v = xpt[ivc_idx]
                with np.errstate(divide="ignore", invalid="ignore"):
                    fracle = (xled_arr[np.newaxis, :] / chord_c - xpt_v[:, np.newaxis]) / dxoc[
                        :, np.newaxis
                    ]
                    fracte = (xted_arr[np.newaxis, :] / chord_c - xpt_v[:, np.newaxis]) / dxoc[
                        :, np.newaxis
                    ]
                zero_dx = dxoc[:, np.newaxis] == 0.0
                fracle = np.where(zero_dx, 0.0, fracle)
                fracte = np.where(zero_dx, 0.0, fracte)
                fracl = np.clip(fracle, 0.0, 1.0)
                fract = np.clip(fracte, 0.0, 1.0)
                state.dcontrol[np.ix_(i_arr, np.arange(ncontrol))] = gainda[np.newaxis, :] * (
                    fract - fracl
                )

    # Wetted surface area (one side) and mean chord.
    sum_area = 0.0
    wtot = 0.0
    for jj in range(state.nj[isurf]):
        j = state.jfrst[isurf] + jj
        astrp = state.wstrip[j] * state.chord[j]
        sum_area += astrp
        wtot += state.wstrip[j]
    state.ssurf[isurf] = sum_area
    state.cavesurf[isurf] = sum_area / wtot if wtot != 0.0 else 0.0


def sdupl(state, base_surf: int, ydup: float, new_surf: int) -> None:
    """Y-duplicate a surface about y = ydup (SDUPL).

    Creates a reflected image with reversed strip edges (IMAGS flipped) so
    positive circulation sense is preserved; hinge Y components are reversed.
    """
    yoff = 2.0 * ydup

    state.lncomp[new_surf] = state.lncomp[base_surf]
    state.lfwake[new_surf] = state.lfwake[base_surf]
    state.lfload[new_surf] = state.lfload[base_surf]
    state.ssurf[new_surf] = state.ssurf[base_surf]
    state.cavesurf[new_surf] = state.cavesurf[base_surf]
    state.clmax_surf[new_surf] = state.clmax_surf[base_surf]
    state.imags[new_surf] = -state.imags[base_surf]
    state.lrange[new_surf] = state.lrange[base_surf]

    state.ifrst[new_surf] = state.nvor
    state.jfrst[new_surf] = state.nstrip
    state.nj[new_surf] = state.nj[base_surf]
    state.nk[new_surf] = state.nk[base_surf]

    nvs = state.nj[new_surf]
    nvc = state.nk[new_surf]

    # Image strips: swap edges 1/2 and reflect Y to maintain Gamma sign convention.
    for ivs in range(nvs):
        jji = state.jfrst[new_surf] + ivs
        jj = state.jfrst[base_surf] + ivs

        state.rle1[0, jji] = state.rle2[0, jj]
        state.rle1[1, jji] = -state.rle2[1, jj] + yoff
        state.rle1[2, jji] = state.rle2[2, jj]
        state.chord1[jji] = state.chord2[jj]

        state.rle2[0, jji] = state.rle1[0, jj]
        state.rle2[1, jji] = -state.rle1[1, jj] + yoff
        state.rle2[2, jji] = state.rle1[2, jj]
        state.chord2[jji] = state.chord1[jj]

        state.rle[0, jji] = state.rle[0, jj]
        state.rle[1, jji] = -state.rle[1, jj] + yoff
        state.rle[2, jji] = state.rle[2, jj]
        state.chord[jji] = state.chord[jj]
        state.wstrip[jji] = state.wstrip[jj]
        state.tanle[jji] = -state.tanle[jj]
        state.ainc[jji] = state.ainc[jj]
        state.ainc1[jji] = state.ainc1[jj]
        state.ainc2[jji] = state.ainc2[jj]
        state.lssurf[jji] = new_surf

        if state.ndesign > 0:
            state.ainc_g[jji, :state.ndesign] = state.ainc_g[jj, :state.ndesign]

        state.clcd[jji, :] = state.clcd[jj, :]
        state.lviscstrp[jji] = state.lviscstrp[jj]

        state.nstrip += 1
        state.ijfrst[jji] = state.nvor
        state.nvstrp[jji] = nvc

        for ivc in range(nvc):
            iii = state.ijfrst[jji] + ivc
            ii = state.ijfrst[jj] + ivc

            state.rv1[0, iii] = state.rv2[0, ii]
            state.rv1[1, iii] = -state.rv2[1, ii] + yoff
            state.rv1[2, iii] = state.rv2[2, ii]
            state.rv2[0, iii] = state.rv1[0, ii]
            state.rv2[1, iii] = -state.rv1[1, ii] + yoff
            state.rv2[2, iii] = state.rv1[2, ii]
            state.rv[0, iii] = state.rv[0, ii]
            state.rv[1, iii] = -state.rv[1, ii] + yoff
            state.rv[2, iii] = state.rv[2, ii]
            state.rc[0, iii] = state.rc[0, ii]
            state.rc[1, iii] = -state.rc[1, ii] + yoff
            state.rc[2, iii] = state.rc[2, ii]
            state.rs[0, iii] = state.rs[0, ii]
            state.rs[1, iii] = -state.rs[1, ii] + yoff
            state.rs[2, iii] = state.rs[2, ii]

            state.slopec[iii] = state.slopec[ii]
            state.slopev[iii] = state.slopev[ii]
            state.xyn1[0, iii] = state.xyn2[0, ii]
            state.xyn1[1, iii] = -state.xyn2[1, ii] + yoff
            state.xyn2[0, iii] = state.xyn1[0, ii]
            state.xyn2[1, iii] = -state.xyn1[1, ii] + yoff
            state.zlon1[iii] = state.zlon2[ii]
            state.zlon2[iii] = state.zlon1[ii]
            state.zupn1[iii] = state.zupn2[ii]
            state.zupn2[iii] = state.zupn1[ii]
            state.dxv[iii] = state.dxv[ii]
            state.chordv[iii] = state.chordv[ii]
            state.lvcomp[iii] = state.lncomp[new_surf]
            state.lvalbe[iii] = state.lvalbe[ii]
            state.lvnc[iii] = state.lvnc[ii]

            for n in range(state.ncontrol):
                # Reverse control deflection sign on image surface.
                rsgn = state.vrefl[jj, n]
                state.dcontrol[iii, n] = -state.dcontrol[ii, n] * rsgn
                state.vrefl[jji, n] = state.vrefl[jj, n]
                state.vhinge[0, jji, n] = state.vhinge[0, jj, n]
                state.vhinge[1, jji, n] = -state.vhinge[1, jj, n]
                state.vhinge[2, jji, n] = state.vhinge[2, jj, n]
                state.phinge[0, jji, n] = state.phinge[0, jj, n]
                state.phinge[1, jji, n] = -state.phinge[1, jj, n] + yoff
                state.phinge[2, jji, n] = state.phinge[2, jj, n]

            state.nvor += 1


def _encalc_control_derivatives(state) -> None:
    """Rotate panel normals for linearized control-surface deflection.

    Vectorized over all (vortex, control) pairs with nonzero ``dcontrol``.
    """
    nvor = state.nvor
    ncontrol = state.ncontrol
    if nvor == 0 or ncontrol == 0:
        return

    strip_of = np.empty(nvor, dtype=np.int32)
    for j in range(state.nstrip):
        i0 = state.ijfrst[j]
        nv = state.nvstrp[j]
        strip_of[i0 : i0 + nv] = j

    vh = state.vhinge[:, strip_of, :ncontrol]
    angddc = state.dtr * state.dcontrol[:nvor, :ncontrol]
    mask = angddc != 0.0

    enc = state.enc[:, :nvor]
    end = np.einsum("ij,ijk->jk", enc, vh)
    ep = enc[:, :, np.newaxis] - end[np.newaxis, :, :] * vh
    eq = np.cross(vh, ep, axis=0)
    state.enc_d[:, :nvor, :ncontrol] = np.where(
        mask[np.newaxis, :, :], eq * angddc[np.newaxis, :, :], 0.0
    )

    env = state.env[:, :nvor]
    end = np.einsum("ij,ijk->jk", env, vh)
    epv = env[:, :, np.newaxis] - end[np.newaxis, :, :] * vh
    eqv = np.cross(vh, epv, axis=0)
    state.env_d[:, :nvor, :ncontrol] = np.where(
        mask[np.newaxis, :, :], eqv * angddc[np.newaxis, :, :], 0.0
    )


def encalc(state) -> None:
    """Compute panel normal vectors (ENCALC).

    Sets strip spanwise/normal directions (ESS, ENSY, ENSZ) and per-vortex
    normals at control points (ENC) and bound-vortex midpoints (ENV),
    including linearized control-surface deflection derivatives.
    """
    for j in range(state.nstrip):
        i0 = state.ijfrst[j]
        # Strip spanwise direction from LE to TE chord lines (blended by SAXFR).
        dxle = state.rv2[0, i0] - state.rv1[0, i0]
        dyle = state.rv2[1, i0] - state.rv1[1, i0]
        dzle = state.rv2[2, i0] - state.rv1[2, i0]
        axle = state.rv[0, i0]
        ayle = state.rv[1, i0]
        azle = state.rv[2, i0]

        i1 = state.ijfrst[j] + state.nvstrp[j] - 1
        dxte = state.rv2[0, i1] - state.rv1[0, i1]
        dyte = state.rv2[1, i1] - state.rv1[1, i1]
        dzte = state.rv2[2, i1] - state.rv1[2, i1]
        axte = state.rv[0, i1]
        ayte = state.rv[1, i1]
        azte = state.rv[2, i1]

        saxfr = state.saxfr
        dxt = (1.0 - saxfr) * dxle + saxfr * dxte
        dyt = (1.0 - saxfr) * dyle + saxfr * dyte
        dzt = (1.0 - saxfr) * dzle + saxfr * dzte
        dmag = math.sqrt(dxt * dxt + dyt * dyt + dzt * dzt)
        yzmag = math.sqrt(dyt * dyt + dzt * dzt)

        if (
            not math.isfinite(dmag)
            or dmag == 0.0
            or not math.isfinite(yzmag)
            or yzmag == 0.0
            or state.wstrip[j] == 0.0
        ):
            # Degenerate strip: mark inactive and zero normals.
            state.lstripoff[j] = True
            state.ess[:, j] = 0.0
            state.ensy[j] = 0.0
            state.ensz[j] = 0.0
            i0 = state.ijfrst[j]
            nv = state.nvstrp[j]
            iv = np.arange(i0, i0 + nv, dtype=np.int32)
            state.enc[:, iv] = 0.0
            state.env[:, iv] = 0.0
            state.lvnc[iv] = False
            continue

        state.ess[0, j] = dxt / dmag
        state.ess[1, j] = dyt / dmag
        state.ess[2, j] = dzt / dmag
        # Strip normal in y-z plane (ENSX = 0).
        state.ensy[j] = -dzt / yzmag
        state.ensz[j] = dyt / yzmag
        state.xsref[j] = (1.0 - saxfr) * axle + saxfr * axte
        state.ysref[j] = (1.0 - saxfr) * ayle + saxfr * ayte
        state.zsref[j] = (1.0 - saxfr) * azle + saxfr * azte

        state.lstripoff[j] = False

        i0 = state.ijfrst[j]
        nv = state.nvstrp[j]
        iv = np.arange(i0, i0 + nv, dtype=np.int32)
        es_y = state.ensy[j]
        es_z = state.ensz[j]
        es_vec = np.array([0.0, es_y, es_z], dtype=np.float64)[:, np.newaxis]
        ainc_j = state.ainc[j]

        # Unit vectors along bound vortex legs for all elements on the strip.
        dxb = state.rv2[0, iv] - state.rv1[0, iv]
        dyb = state.rv2[1, iv] - state.rv1[1, iv]
        dzb = state.rv2[2, iv] - state.rv1[2, iv]
        emag = np.sqrt(dxb * dxb + dyb * dyb + dzb * dzb)
        eb = np.stack([dxb / emag, dyb / emag, dzb / emag])

        # Control-point normals: camber slope + incidence, crossed with bound leg.
        ang_c = ainc_j - np.arctan(state.slopec[iv])
        sinc_c = np.sin(ang_c)
        cosc_c = np.cos(ang_c)
        ec = np.stack([cosc_c, -sinc_c * es_y, -sinc_c * es_z])
        ecxb = np.cross(ec, eb, axis=0)
        em = np.linalg.norm(ecxb, axis=0)
        state.enc[:, iv] = np.where((em != 0.0)[np.newaxis, :], ecxb / em, es_vec)

        # Bound-vortex midpoint normals (uses SLOPEV instead of SLOPEC).
        ang_v = ainc_j - np.arctan(state.slopev[iv])
        sinc_v = np.sin(ang_v)
        cosc_v = np.cos(ang_v)
        ecv = np.stack([cosc_v, -sinc_v * es_y, -sinc_v * es_z])
        ecxbv = np.cross(ecv, eb, axis=0)
        emv = np.linalg.norm(ecxbv, axis=0)
        state.env[:, iv] = np.where((emv != 0.0)[np.newaxis, :], ecxbv / emv, es_vec)

        state.lvnc[iv] = True

    _encalc_control_derivatives(state)


def _body_node_count(body: BodyDef) -> int:
    """Return the number of body line nodes when the body is solvable."""
    n = int(round(body.n_body or 0))
    if n < 2:
        return 0
    if not body.body_thread_x or len(body.body_thread_x) < 2:
        return 0
    if not body.body_thread_y or len(body.body_thread_y) < 2:
        return 0
    if not body.body_thread_t or len(body.body_thread_t) < 2:
        return 0
    return n


def makebody(state, ibody: int, body: BodyDef, node_cursor: int) -> int:
    """Build body line nodes for one body definition (MAKEBODY).

    Returns the updated node cursor (0-based index of next free node slot).
    """
    n_body = _body_node_count(body)
    if n_body < 2:
        return node_cursor

    xb = np.asarray(body.body_thread_x, dtype=np.float64)
    yb = np.asarray(body.body_thread_y, dtype=np.float64)
    tb = np.asarray(body.body_thread_t, dtype=np.float64)
    if xb.size < 2 or yb.size < 2 or tb.size < 2:
        return node_cursor

    x_min = float(xb[0])
    x_max = float(xb[-1])
    if not np.isfinite(x_min) or not np.isfinite(x_max):
        return node_cursor
    if node_cursor + n_body > state.nlmax:
        return node_cursor

    scale = np.asarray(body.scale if body.scale else [1.0, 1.0, 1.0], dtype=np.float64)
    translate = np.asarray(body.translate if body.translate else [0.0, 0.0, 0.0], dtype=np.float64)
    x_scale = float(scale[0]) if scale.size else 1.0
    y_scale = float(scale[1]) if scale.size > 1 else 1.0
    z_scale = float(scale[2]) if scale.size > 2 else 1.0
    radius_scale = math.sqrt(y_scale * z_scale)
    tx = float(translate[0]) if translate.size else 0.0
    ty = float(translate[1]) if translate.size > 1 else 0.0
    tz = float(translate[2]) if translate.size > 2 else 0.0

    fspace = spacer(n_body, body.b_space)
    xpt = np.zeros(n_body + 1, dtype=np.float64)
    for i in range(1, n_body + 1):
        xpt[i] = fspace[i]
    xpt[1] = 0.0
    xpt[n_body] = 1.0

    state.lfrst[ibody] = node_cursor
    state.nl[ibody] = n_body

    for i in range(1, n_body + 1):
        l = node_cursor + i - 1
        s = xpt[i]
        x_local = x_min + (x_max - x_min) * s
        y_center = akima(xb, yb, x_local)[0]
        thickness = akima(xb, tb, x_local)[0]
        state.rl[0, l] = x_scale * x_local + tx
        state.rl[1, l] = ty
        state.rl[2, l] = z_scale * y_center + tz
        state.radl[l] = 0.5 * thickness * radius_scale

    return node_cursor + n_body


def solver_surface_index(model, isurf: int) -> int:
    """Map a solver surface index to ``model.surfaces`` index.

    Symmetric surfaces with ``yduplicate`` occupy two solver indices (original
    half and mirror); both map back to the same model surface index.
    """
    idx = 0
    for i, surf in enumerate(model.surfaces):
        if idx == isurf:
            return i
        idx += 1
        if surf.yduplicate is not None:
            if idx == isurf:
                return i
            idx += 1
    return -1


def solver_surface_name(model, isurf: int) -> str:
    """Return a human-readable label for a solver surface index."""
    idx = 0
    for surf in model.surfaces:
        if idx == isurf:
            return surf.name
        idx += 1
        if surf.yduplicate is not None:
            if idx == isurf:
                return f"{surf.name} (mirror)"
            idx += 1
    return f"Surface {isurf + 1}"


def build_geometry(state, model) -> None:
    """Build full geometry for all surfaces in the model."""
    state.nvor = 0
    state.nstrip = 0
    surf_index = 0
    for surf in model.surfaces:
        isurf = surf_index
        comp = surf.component if surf.component else (isurf + 1)
        state.lncomp[isurf] = comp
        state.lfwake[isurf] = 0 if surf.nowake else 1
        state.lfload[isurf] = 0 if surf.noload else 1
        state.clmax_surf[isurf] = surf.clmax
        makesurf(state, isurf, surf)
        surf_index += 1
        if surf.yduplicate is not None:
            # Optional YDUPLICATE: reflect surface about specified y plane.
            sdupl(state, isurf, surf.yduplicate, surf_index)
            surf_index += 1
    state.nsurf = surf_index

    body_count = 0
    node_cursor = 0
    for body in model.bodies:
        if _body_node_count(body) < 2:
            continue
        if body_count >= state.lfrst.size:
            break
        node_cursor = makebody(state, body_count, body, node_cursor)
        body_count += 1
    state.nbody = body_count
    state.nlnode = node_cursor

    encalc(state)

    from openavl.core.state import build_vortex_to_strip

    build_vortex_to_strip(state)
