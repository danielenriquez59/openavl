"""JAX aerodynamic force integration (SFFORC, BDFORC, AERO)."""

from __future__ import annotations

from functools import partial
from typing import Any

import numpy as np

from openavl.jax.backend import jax, jnp
from openavl.jax.cdcl import cdcl_jax
from openavl.jax.freestream import vinfab as vinfab_jax
from openavl.jax.trefftz import tpforc_jax, tpforc_jax_jit
from openavl.jax.types import (
    BodyForces,
    BodyGeometry,
    FlowCondition,
    ForceGeometry,
    ForceResult,
    InviscidForces,
    ReferenceQuantities,
    StripForces,
    StripMap,
    SurfaceForces,
    TrefftzGeometry,
    Velocities,
)


def _cross(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """Cross product ``a × b`` for 3-vectors or batched ``[3, N]`` arrays."""
    if a.ndim == 1 and b.ndim == 1:
        return jnp.cross(a, b)
    if b.ndim == 1:
        b = b[:, None]
    return jnp.cross(a, b, axis=0)


def _normalize(v: jnp.ndarray, fallback: jnp.ndarray | None = None) -> jnp.ndarray:
    """Normalize vectors; use fallback direction when magnitude is zero."""
    mag = jnp.linalg.norm(v, axis=0 if v.ndim > 1 else None)
    if v.ndim == 1:
        safe = jnp.where(mag > 0.0, mag, 1.0)
        out = v / safe
        if fallback is not None:
            out = jnp.where(mag > 0.0, out, fallback)
        return out
    safe = jnp.where(mag > 0.0, mag, 1.0)
    out = v / safe
    if fallback is not None:
        fb = fallback[:, None]
        out = jnp.where(mag > 0.0, out, fb)
    return out


def _strip_lift_drag_dirs(
    ensy: jnp.ndarray,
    ensz: jnp.ndarray,
    vinf: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Local strip lift and drag unit vectors."""
    spn = jnp.stack([jnp.zeros_like(ensy), ensy, -ensz], axis=0)
    udrag = vinf
    ulift = _cross(udrag, spn)
    ulift = _normalize(ulift, jnp.array([0.0, 0.0, 1.0]))
    return ulift, udrag


def _vortex_forces(
    geom: ForceGeometry,
    gamma: jnp.ndarray,
    velocities: Velocities,
    flow: FlowCondition,
    refs: ReferenceQuantities,
    *,
    lnfld_wv: bool,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Per-vortex bound-vortex force contributions and delta Cp."""
    smap = geom.strip_map
    v2s = smap.vortex_to_strip
    nstrip = geom.chord.shape[0]

    rc4 = jnp.stack(
        [geom.rle[0] + 0.25 * geom.chord, geom.rle[1], geom.rle[2]],
        axis=0,
    )
    rc4_v = rc4[:, v2s]
    wstrip_v = geom.wstrip[v2s]
    cr_v = geom.chord[v2s]
    sr = cr_v * wstrip_v

    # Precompute the activity mask before any division so unselected
    # (turned-off or zero-width) strips never divide by zero: safe-where
    # forces those denominators to a harmless 1 up front, matching the
    # `inv_sr = where(active, 1/sr, 0)` idiom used elsewhere (see A9).
    active = (~smap.lstripoff[v2s]) & (wstrip_v > 0.0)
    sr_safe = jnp.where(active, sr, 1.0)
    cr_safe = jnp.where(active, cr_v, 1.0)
    dxw_safe = jnp.where(active, geom.dxv * wstrip_v, 1.0)

    r = geom.rv - rc4_v
    rrot = geom.rv - refs.xyzref[:, None]
    vrot = _cross(rrot, flow.wrot)

    vinf = vinfab_jax(flow.alfa, flow.beta)
    vind = jnp.where(lnfld_wv, velocities.wv, velocities.vv)
    veff = vinf[:, None] + vrot + vind

    g = geom.rv2 - geom.rv1
    f = _cross(veff, g)
    fgam = 2.0 * gamma[None, :] * f

    dcp = jnp.sum(geom.env * fgam, axis=0) / dxw_safe

    dcfx = fgam[0] / sr_safe
    dcfy = fgam[1] / sr_safe
    dcfz = fgam[2] / sr_safe
    dcmx = ((dcfz * r[1]) - (dcfy * r[2])) / cr_safe
    dcmy = ((dcfx * r[2]) - (dcfz * r[0])) / cr_safe
    dcmz = ((dcfy * r[0]) - (dcfx * r[1])) / cr_safe

    ensy_v = geom.ensy[v2s]
    ensz_v = geom.ensz[v2s]
    dcnc = cr_v * (ensy_v * dcfy + ensz_v * dcfz)

    z = jnp.array(0.0)
    dcfx = jnp.where(active, dcfx, z)
    dcfy = jnp.where(active, dcfy, z)
    dcfz = jnp.where(active, dcfz, z)
    dcmx = jnp.where(active, dcmx, z)
    dcmy = jnp.where(active, dcmy, z)
    dcmz = jnp.where(active, dcmz, z)
    dcnc = jnp.where(active, dcnc, z)
    dcp = jnp.where(active, dcp, z)

    cfx = jax.ops.segment_sum(dcfx, v2s, num_segments=nstrip)
    cfy = jax.ops.segment_sum(dcfy, v2s, num_segments=nstrip)
    cfz = jax.ops.segment_sum(dcfz, v2s, num_segments=nstrip)
    cmx = jax.ops.segment_sum(dcmx, v2s, num_segments=nstrip)
    cmy = jax.ops.segment_sum(dcmy, v2s, num_segments=nstrip)
    cmz = jax.ops.segment_sum(dcmz, v2s, num_segments=nstrip)
    cnc = jax.ops.segment_sum(dcnc, v2s, num_segments=nstrip)

    return jnp.stack([cfx, cfy, cfz]), jnp.stack([cmx, cmy, cmz]), cnc, dcp


def _apply_viscous_strips(
    cf_body: jnp.ndarray,
    geom: ForceGeometry,
    flow: FlowCondition,
    refs: ReferenceQuantities,
    *,
    lvisc: bool,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Add viscous CD(CL) drag to strip body-axis force coefficients."""
    nstrip = geom.chord.shape[0]
    smap = geom.strip_map
    active = (~smap.lstripoff) & (geom.wstrip > 0.0)
    visc_on = lvisc & smap.lviscstrp & active

    vinf = vinfab_jax(flow.alfa, flow.beta)
    ulift, udrag = _strip_lift_drag_dirs(geom.ensy, geom.ensz, vinf)

    cfx, cfy, cfz = cf_body[0], cf_body[1], cf_body[2]
    clv = ulift[0] * cfx + ulift[1] * cfy + ulift[2] * cfz

    cdv = jax.vmap(cdcl_jax)(geom.clcd, clv)

    rc4 = jnp.stack(
        [geom.rle[0] + 0.25 * geom.chord, geom.rle[1], geom.rle[2]],
        axis=0,
    )
    rrot = rc4 - refs.xyzref[:, None]
    vrot = _cross(rrot, flow.wrot)
    veff = vinf[:, None] + vrot
    veffmag = jnp.linalg.norm(veff, axis=0)

    dcvfx = veff[0] * veffmag * cdv
    dcvfy = veff[1] * veffmag * cdv
    dcvfz = veff[2] * veffmag * cdv
    cfx = jnp.where(visc_on, cfx + dcvfx, cfx)
    cfy = jnp.where(visc_on, cfy + dcvfy, cfy)
    cfz = jnp.where(visc_on, cfz + dcvfz, cfz)
    cdv_lstrp = jnp.where(
        visc_on,
        udrag[0] * dcvfx + udrag[1] * dcvfy + udrag[2] * dcvfz,
        0.0,
    )
    return jnp.stack([cfx, cfy, cfz]), cdv_lstrp


def _stability_transform(
    cf_body: jnp.ndarray,
    alfa: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Transform strip body-axis forces to stability axes."""
    sina = jnp.sin(alfa)
    cosa = jnp.cos(alfa)
    cfx, cfy, cfz = cf_body[0], cf_body[1], cf_body[2]
    cdstrp = cfx * cosa + cfz * sina
    cystrp = cfy
    clstrp = -cfx * sina + cfz * cosa
    return cdstrp, cystrp, clstrp


def _surface_accumulation(
    geom: ForceGeometry,
    refs: ReferenceQuantities,
    cdstrp: jnp.ndarray,
    cystrp: jnp.ndarray,
    clstrp: jnp.ndarray,
    cfstrp: jnp.ndarray,
    cmstrp: jnp.ndarray,
    cdv_lstrp: jnp.ndarray,
) -> SurfaceForces:
    """Sum strip coefficients into surface and configuration totals."""
    smap = geom.strip_map
    nsurf = geom.ssurf.shape[0]
    sr = geom.chord * geom.wstrip
    cr = geom.chord

    w = sr / refs.sref
    cds = jax.ops.segment_sum(cdstrp * w, smap.strip_to_surface, num_segments=nsurf)
    cys = jax.ops.segment_sum(cystrp * w, smap.strip_to_surface, num_segments=nsurf)
    cls = jax.ops.segment_sum(clstrp * w, smap.strip_to_surface, num_segments=nsurf)
    cdv = jax.ops.segment_sum(cdv_lstrp * w, smap.strip_to_surface, num_segments=nsurf)

    cfs = jnp.stack(
        [
            jax.ops.segment_sum(cfstrp[k] * w, smap.strip_to_surface, num_segments=nsurf)
            for k in range(3)
        ]
    )
    cm_scale_xz = w * cr / refs.bref
    cm_scale_y = w * cr / refs.cref
    cms = jnp.stack(
        [
            jax.ops.segment_sum(cmstrp[0] * cm_scale_xz, smap.strip_to_surface, num_segments=nsurf),
            jax.ops.segment_sum(cmstrp[1] * cm_scale_y, smap.strip_to_surface, num_segments=nsurf),
            jax.ops.segment_sum(cmstrp[2] * cm_scale_xz, smap.strip_to_surface, num_segments=nsurf),
        ]
    )

    return SurfaceForces(cdsurf=cds, cysurf=cys, clsurf=cls, cfsurf=cfs, cmsurf=cms, cdvsurf=cdv)


def sfforc_jax(
    geom: ForceGeometry,
    gamma: jnp.ndarray,
    velocities: Velocities,
    flow: FlowCondition,
    refs: ReferenceQuantities,
    *,
    lnfld_wv: bool = False,
    lvisc: bool = False,
    ltrforce: bool = False,
) -> InviscidForces:
    """Vectorized strip/surface force integration from bound vorticity."""
    if ltrforce:
        raise NotImplementedError("JAX force integration does not yet include trailing-leg forces")
    smap = geom.strip_map
    nstrip = geom.chord.shape[0]
    nsurf = geom.ssurf.shape[0]
    active_strip = (~smap.lstripoff) & (geom.wstrip > 0.0)

    cf_body, cm_body, cnc, dcp = _vortex_forces(
        geom, gamma, velocities, flow, refs, lnfld_wv=lnfld_wv
    )

    if lvisc:
        cf_body, cdv_lstrp = _apply_viscous_strips(
            cf_body, geom, flow, refs, lvisc=lvisc
        )
    else:
        cdv_lstrp = jnp.zeros(nstrip)

    cf_body = jnp.where(active_strip[None, :], cf_body, 0.0)
    cm_body = jnp.where(active_strip[None, :], cm_body, 0.0)
    cnc = jnp.where(active_strip, cnc, 0.0)
    cdv_lstrp = jnp.where(active_strip, cdv_lstrp, 0.0)

    rc4 = jnp.stack(
        [geom.rle[0] + 0.25 * geom.chord, geom.rle[1], geom.rle[2]],
        axis=0,
    )
    rref = rc4 - refs.xyzref[:, None]
    cmstrp = cm_body + jnp.stack(
        [
            (cf_body[2] * rref[1] - cf_body[1] * rref[2]) / geom.chord,
            (cf_body[0] * rref[2] - cf_body[2] * rref[0]) / geom.chord,
            (cf_body[1] * rref[0] - cf_body[0] * rref[1]) / geom.chord,
        ],
        axis=0,
    )

    cdstrp, cystrp, clstrp = _stability_transform(cf_body, flow.alfa)
    cfstrp = cf_body

    surfaces = _surface_accumulation(
        geom, refs, cdstrp, cystrp, clstrp, cfstrp, cmstrp, cdv_lstrp
    )

    load = geom.lfload.astype(gamma.dtype)
    cltot = jnp.sum(surfaces.clsurf * load)
    cdtot = jnp.sum(surfaces.cdsurf * load)
    cytot = jnp.sum(surfaces.cysurf * load)
    cdvtot = jnp.sum(surfaces.cdvsurf * load)
    cftot = jnp.sum(surfaces.cfsurf * load[None, :], axis=1)
    cmtot = jnp.sum(surfaces.cmsurf * load[None, :], axis=1)

    strips = StripForces(
        cdstrp=cdstrp,
        cystrp=cystrp,
        clstrp=clstrp,
        cfstrp=cfstrp,
        cmstrp=cmstrp,
        cdv_lstrp=cdv_lstrp,
        cnc=cnc,
        dcp=dcp,
    )
    return InviscidForces(
        CL=cltot,
        CD=cdtot,
        CY=cytot,
        CM=cmtot,
        CF=cftot,
        CDV=cdvtot,
        strips=strips,
        surfaces=surfaces,
    )


def bdforc_jax(
    body: BodyGeometry,
    flow: FlowCondition,
    refs: ReferenceQuantities,
) -> BodyForces:
    """Integrate slender-body source forces over body line segments.

    The source strength is formed live from the flow condition (the same
    six-component unit-flow contraction as the NumPy ``gucalc``/``gamsum``
    routines: ``src = src_u @ [vinf, wrot]``) rather than reused from the
    snapshot flow condition, so the body force stays fully differentiable
    w.r.t. ``alfa``/``beta``/``wrot`` at any flow condition.
    """
    nseg = int(body.seg_i1.shape[0])
    if nseg == 0:
        z = jnp.array(0.0)
        return BodyForces(CL=z, CD=z, CY=z, CM=jnp.zeros(3), CF=jnp.zeros(3))

    betm = jnp.sqrt(1.0 - flow.mach * flow.mach)
    sina = jnp.sin(flow.alfa)
    cosa = jnp.cos(flow.alfa)
    vinf = vinfab_jax(flow.alfa, flow.beta)
    u = jnp.concatenate((vinf, flow.wrot))
    src = body.src_u @ u

    l1 = body.seg_i1
    l2 = body.seg_i2
    drl = jnp.stack(
        [
            (body.rl[0, l2] - body.rl[0, l1]) / betm,
            body.rl[1, l2] - body.rl[1, l1],
            body.rl[2, l2] - body.rl[2, l1],
        ],
        axis=0,
    )
    drlmag = jnp.linalg.norm(drl, axis=0)
    drlmi = jnp.where(drlmag > 0.0, 1.0 / drlmag, 0.0)
    esl = drl * drlmi

    rrot = 0.5 * (body.rl[:, l2] + body.rl[:, l1]) - refs.xyzref[:, None]
    vrot = _cross(rrot, flow.wrot[:, None])
    veff = jnp.stack(
        [
            (vinf[0] + vrot[0]) / betm,
            vinf[1] + vrot[1],
            vinf[2] + vrot[2],
        ],
        axis=0,
    )

    us = jnp.sum(veff * esl, axis=0)
    un = veff - esl * us
    fb = un * src[l1]
    mb = _cross(rrot, fb)
    scale = 2.0 / refs.sref

    cdbdy = jnp.sum((fb[0] * cosa + fb[2] * sina) * scale)
    cybdy = jnp.sum(fb[1] * scale)
    clbdy = jnp.sum((-fb[0] * sina + fb[2] * cosa) * scale)
    cfbdy = jnp.sum(fb * scale, axis=1)
    cmbdy = jnp.array(
        [
            jnp.sum(mb[0] * scale / refs.bref),
            jnp.sum(mb[1] * scale / refs.cref),
            jnp.sum(mb[2] * scale / refs.bref),
        ]
    )

    return BodyForces(CL=clbdy, CD=cdbdy, CY=cybdy, CM=cmbdy, CF=cfbdy)


def _apply_symmetry(result: ForceResult, iysym: int) -> ForceResult:
    """Apply XZ-symmetry doubling like the NumPy AERO routine."""
    if iysym != 1:
        return result
    # Trefftz integration applies its own symmetry because wake-image filaments
    # contribute to induced drag before far-field totals are formed.
    cm = result.CM.at[0].set(0.0).at[1].set(2.0 * result.CM[1]).at[2].set(0.0)
    cf = result.CF.at[0].set(2.0 * result.CF[0]).at[1].set(0.0).at[2].set(2.0 * result.CF[2])
    return result._replace(
        CL=2.0 * result.CL,
        CD=2.0 * result.CD,
        CY=jnp.array(0.0),
        CM=cm,
        CF=cf,
        CDV=2.0 * result.CDV,
        CLFF=result.CLFF,
        CYFF=result.CYFF,
        CDFF=result.CDFF,
    )


def _cdref_contribution(flow: FlowCondition, refs: ReferenceQuantities) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Baseline reference drag aligned with the freestream direction."""
    vinf = vinfab_jax(flow.alfa, flow.beta)
    vsq = jnp.dot(vinf, vinf)
    vmag = jnp.sqrt(vsq)
    cd_add = refs.cdref * vsq
    cy_add = refs.cdref * vinf[1] * vmag
    cf_add = refs.cdref * vinf * vmag
    cdv_add = refs.cdref * vsq
    return cd_add, cy_add, cf_add, cdv_add


def _compute_forces_eager(
    geom: ForceGeometry,
    gamma: jnp.ndarray,
    velocities: Velocities,
    flow: FlowCondition,
    refs: ReferenceQuantities,
    body: BodyGeometry | None = None,
    tgeom: TrefftzGeometry | None = None,
    *,
    lnfld_wv: bool = False,
    lvisc: bool = False,
    ltrforce: bool = False,
) -> ForceResult:
    """Eager force integration for differentiable analysis (preserves Python branches)."""
    inviscid = sfforc_jax(
        geom,
        gamma,
        velocities,
        flow,
        refs,
        lnfld_wv=lnfld_wv,
        lvisc=lvisc,
        ltrforce=ltrforce,
    )

    body_forces = None
    cl = inviscid.CL
    cd = inviscid.CD
    cy = inviscid.CY
    cm = inviscid.CM
    cf = inviscid.CF
    cdv = inviscid.CDV

    if body is not None and body.nl.shape[0] > 0:
        body_forces = bdforc_jax(body, flow, refs)
        cl = cl + body_forces.CL
        cd = cd + body_forces.CD
        cy = cy + body_forces.CY
        cm = cm + body_forces.CM
        cf = cf + body_forces.CF

    trefftz = None
    clff = jnp.array(0.0)
    cyff = jnp.array(0.0)
    cdff = jnp.array(0.0)
    spanef = jnp.array(0.0)
    if tgeom is not None:
        trefftz = tpforc_jax(gamma, tgeom, refs)
        clff = trefftz.CL
        cyff = trefftz.CY
        cdff = trefftz.CDi
        spanef = trefftz.spanef

    result = ForceResult(
        CL=cl,
        CD=cd,
        CY=cy,
        CM=cm,
        CF=cf,
        CDV=cdv,
        CLFF=clff,
        CYFF=cyff,
        CDFF=cdff,
        SPANEF=spanef,
        inviscid=inviscid,
        body=body_forces,
        trefftz=trefftz,
    )
    result = _apply_symmetry(result, refs.iysym)

    # cdref must be added after symmetry doubling, matching NumPy AERO's
    # operation order: doubling first, then the baseline-drag contribution.
    cd_add, cy_add, cf_add, cdv_add = _cdref_contribution(flow, refs)
    return result._replace(
        CD=result.CD + cd_add,
        CY=result.CY + cy_add,
        CF=result.CF + cf_add,
        CDV=result.CDV + cdv_add,
    )


def _compute_forces_impl(
    geom: ForceGeometry,
    gamma: jnp.ndarray,
    velocities: Velocities,
    flow: FlowCondition,
    refs: ReferenceQuantities,
    body: BodyGeometry,
    tgeom: TrefftzGeometry,
    *,
    lnfld_wv: bool,
    lvisc: bool,
    ltrforce: bool,
    include_body: bool,
    include_trefftz: bool,
    iysym: int,
) -> ForceResult:
    """Traceable force integration for the JIT-compiled entry point."""
    inviscid = sfforc_jax(
        geom,
        gamma,
        velocities,
        flow,
        refs,
        lnfld_wv=lnfld_wv,
        lvisc=lvisc,
        ltrforce=ltrforce,
    )

    cl = inviscid.CL
    cd = inviscid.CD
    cy = inviscid.CY
    cm = inviscid.CM
    cf = inviscid.CF
    cdv = inviscid.CDV

    body_forces = bdforc_jax(body, flow, refs)
    cl = cl + jnp.where(include_body, body_forces.CL, 0.0)
    cd = cd + jnp.where(include_body, body_forces.CD, 0.0)
    cy = cy + jnp.where(include_body, body_forces.CY, 0.0)
    cm = cm + jnp.where(include_body, body_forces.CM, 0.0)
    cf = cf + jnp.where(include_body, body_forces.CF, 0.0)

    trefftz = tpforc_jax_jit(gamma, tgeom, refs)
    clff = jnp.where(include_trefftz, trefftz.CL, 0.0)
    cyff = jnp.where(include_trefftz, trefftz.CY, 0.0)
    cdff = jnp.where(include_trefftz, trefftz.CDi, 0.0)
    spanef = jnp.where(include_trefftz, trefftz.spanef, 0.0)

    result = ForceResult(
        CL=cl,
        CD=cd,
        CY=cy,
        CM=cm,
        CF=cf,
        CDV=cdv,
        CLFF=clff,
        CYFF=cyff,
        CDFF=cdff,
        SPANEF=spanef,
        inviscid=inviscid,
        body=body_forces,
        trefftz=trefftz,
    )
    result = _apply_symmetry(result, iysym)

    # cdref must be added after symmetry doubling, matching NumPy AERO's
    # operation order: doubling first, then the baseline-drag contribution.
    cd_add, cy_add, cf_add, cdv_add = _cdref_contribution(flow, refs)
    return result._replace(
        CD=result.CD + cd_add,
        CY=result.CY + cy_add,
        CF=result.CF + cf_add,
        CDV=result.CDV + cdv_add,
    )


_compute_forces_jit = partial(
    jax.jit,
    static_argnames=(
        "lnfld_wv",
        "lvisc",
        "ltrforce",
        "include_body",
        "include_trefftz",
        "iysym",
    ),
)(_compute_forces_impl)


def compute_forces(
    geom: ForceGeometry,
    gamma: jnp.ndarray,
    velocities: Velocities,
    flow: FlowCondition,
    refs: ReferenceQuantities,
    body: BodyGeometry | None = None,
    tgeom: TrefftzGeometry | None = None,
    *,
    lnfld_wv: bool = False,
    lvisc: bool = False,
    ltrforce: bool = False,
) -> ForceResult:
    """Combine inviscid, body, Trefftz, and baseline drag into total coefficients."""
    include_body = body is not None and body.nl.shape[0] > 0
    include_trefftz = tgeom is not None
    return _compute_forces_jit(
        geom,
        gamma,
        velocities,
        flow,
        refs,
        body if body is not None else _EMPTY_BODY,
        tgeom if tgeom is not None else _EMPTY_TREFFTZ,
        lnfld_wv=lnfld_wv,
        lvisc=lvisc,
        ltrforce=ltrforce,
        include_body=include_body,
        include_trefftz=include_trefftz,
        iysym=int(refs.iysym),
    )


_EMPTY_BODY = BodyGeometry(
    nl=jnp.zeros(0, dtype=jnp.int32),
    lfrst=jnp.zeros(0, dtype=jnp.int32),
    rl=jnp.zeros((3, 0)),
    radl=jnp.zeros(0),
    src=jnp.zeros(0),
    src_u=jnp.zeros((0, 6)),
    seg_i1=jnp.zeros(0, dtype=jnp.int32),
    seg_i2=jnp.zeros(0, dtype=jnp.int32),
)

_EMPTY_TREFFTZ = TrefftzGeometry(
    rv1=jnp.zeros((3, 0)),
    rv2=jnp.zeros((3, 0)),
    rc=jnp.zeros((3, 0)),
    chord=jnp.zeros(0),
    strip_map=StripMap(
        vortex_to_strip=jnp.zeros(0, dtype=jnp.int32),
        ijfrst=jnp.zeros(0, dtype=jnp.int32),
        nvstrp=jnp.zeros(0, dtype=jnp.int32),
        strip_to_surface=jnp.zeros(0, dtype=jnp.int32),
        jfrst=jnp.zeros(0, dtype=jnp.int32),
        nj=jnp.zeros(0, dtype=jnp.int32),
        lstripoff=jnp.zeros(0, dtype=bool),
        lviscstrp=jnp.zeros(0, dtype=bool),
        lssurf=jnp.zeros(0, dtype=jnp.int32),
        lncomp=jnp.zeros(0, dtype=jnp.int32),
    ),
    iysym=0,
    izsym=0,
    ysym=jnp.array(0.0),
    zsym=jnp.array(0.0),
    vrcorec=jnp.array(0.0),
    vrcorew=jnp.array(0.0),
    lfload=jnp.zeros(0, dtype=bool),
    amach=jnp.array(0.0),
)


def _build_vortex_to_strip(state: Any) -> np.ndarray:
    """Map each vortex index to its parent strip."""
    nvor = state.nvor
    v2s = np.zeros(nvor, dtype=np.int32)
    for j in range(state.nstrip):
        i1 = int(state.ijfrst[j])
        nvc = int(state.nvstrp[j])
        v2s[i1 : i1 + nvc] = j
    return v2s


def force_geometry_from_state(state: Any) -> ForceGeometry:
    """Extract JAX force geometry from a solved NumPy ``AVLState`` (for tests)."""
    clmax_surf = np.asarray(state.clmax_surf[: state.nsurf], dtype=np.float64)
    if np.any(clmax_surf > 0.0):
        raise NotImplementedError(
            "JAX force integration does not yet support clmax_surf sectional-lift clipping"
        )
    v2s = _build_vortex_to_strip(state)
    strip_to_surface = np.asarray(state.lssurf, dtype=np.int32)
    smap = StripMap(
        vortex_to_strip=jnp.asarray(v2s),
        ijfrst=jnp.asarray(state.ijfrst),
        nvstrp=jnp.asarray(state.nvstrp),
        strip_to_surface=jnp.asarray(strip_to_surface),
        jfrst=jnp.asarray(state.jfrst),
        nj=jnp.asarray(state.nj),
        lstripoff=jnp.asarray(state.lstripoff),
        lviscstrp=jnp.asarray(state.lviscstrp),
        lssurf=jnp.asarray(state.lssurf),
        lncomp=jnp.asarray(state.lncomp),
    )
    return ForceGeometry(
        rv1=jnp.asarray(state.rv1),
        rv2=jnp.asarray(state.rv2),
        rv=jnp.asarray(state.rv),
        rc=jnp.asarray(state.rc),
        env=jnp.asarray(state.env),
        dxv=jnp.asarray(state.dxv),
        rle=jnp.asarray(state.rle),
        rle1=jnp.asarray(state.rle1),
        rle2=jnp.asarray(state.rle2),
        chord=jnp.asarray(state.chord),
        chord1=jnp.asarray(state.chord1),
        chord2=jnp.asarray(state.chord2),
        wstrip=jnp.asarray(state.wstrip),
        ensy=jnp.asarray(state.ensy),
        ensz=jnp.asarray(state.ensz),
        ess=jnp.asarray(state.ess),
        ainc=jnp.asarray(state.ainc),
        xsref=jnp.asarray(state.xsref),
        ysref=jnp.asarray(state.ysref),
        zsref=jnp.asarray(state.zsref),
        ssurf=jnp.asarray(state.ssurf),
        cavesurf=jnp.asarray(state.cavesurf),
        imags=jnp.asarray(state.imags),
        lfload=jnp.asarray(state.lfload),
        clcd=jnp.asarray(state.clcd),
        strip_map=smap,
        nbody=int(state.nbody),
    )


def body_geometry_from_state(state: Any) -> BodyGeometry:
    """Extract body geometry arrays from ``AVLState``."""
    seg_i1: list[int] = []
    seg_i2: list[int] = []
    for ib in range(int(state.nbody)):
        nln = int(state.nl[ib])
        base = int(state.lfrst[ib])
        for ilseg in range(nln - 1):
            seg_i1.append(base + ilseg)
            seg_i2.append(base + ilseg + 1)
    return BodyGeometry(
        nl=jnp.asarray(state.nl),
        lfrst=jnp.asarray(state.lfrst),
        rl=jnp.asarray(state.rl),
        radl=jnp.asarray(state.radl),
        src=jnp.asarray(state.src),
        src_u=jnp.asarray(state.src_u),
        seg_i1=jnp.asarray(seg_i1, dtype=jnp.int32),
        seg_i2=jnp.asarray(seg_i2, dtype=jnp.int32),
    )


def trefftz_geometry_from_state(state: Any) -> TrefftzGeometry:
    """Extract Trefftz geometry from ``AVLState``."""
    v2s = _build_vortex_to_strip(state)
    smap = StripMap(
        vortex_to_strip=jnp.asarray(v2s),
        ijfrst=jnp.asarray(state.ijfrst),
        nvstrp=jnp.asarray(state.nvstrp),
        strip_to_surface=jnp.asarray(state.lssurf),
        jfrst=jnp.asarray(state.jfrst),
        nj=jnp.asarray(state.nj),
        lstripoff=jnp.asarray(state.lstripoff),
        lviscstrp=jnp.asarray(state.lviscstrp),
        lssurf=jnp.asarray(state.lssurf),
        lncomp=jnp.asarray(state.lncomp),
    )
    return TrefftzGeometry(
        rv1=jnp.asarray(state.rv1),
        rv2=jnp.asarray(state.rv2),
        rc=jnp.asarray(state.rc),
        chord=jnp.asarray(state.chord),
        strip_map=smap,
        iysym=int(state.iysym),
        izsym=int(state.izsym),
        ysym=jnp.asarray(state.ysym),
        zsym=jnp.asarray(state.zsym),
        vrcorec=jnp.asarray(state.vrcorec),
        vrcorew=jnp.asarray(state.vrcorew),
        lfload=jnp.asarray(state.lfload),
        amach=jnp.asarray(getattr(state, "amach", state.mach)),
    )


def flow_from_state(state: Any) -> FlowCondition:
    """Build ``FlowCondition`` from ``AVLState``."""
    return FlowCondition(
        alfa=jnp.asarray(state.alfa),
        beta=jnp.asarray(state.beta),
        wrot=jnp.asarray(state.wrot),
        mach=jnp.asarray(state.mach),
        delcon=jnp.asarray(state.delcon[: state.ncontrol] if state.ncontrol else np.zeros(0)),
    )


def refs_from_state(state: Any) -> ReferenceQuantities:
    """Build reference quantities from ``AVLState``."""
    return ReferenceQuantities(
        sref=jnp.asarray(state.sref),
        cref=jnp.asarray(state.cref),
        bref=jnp.asarray(state.bref),
        xyzref=jnp.asarray(state.xyzref),
        cdref=jnp.asarray(state.cdref),
        iysym=int(state.iysym),
    )


def velocities_from_state(state: Any) -> Velocities:
    """Extract induced velocities from ``AVLState``."""
    return Velocities(vv=jnp.asarray(state.vv), wv=jnp.asarray(state.wv))
