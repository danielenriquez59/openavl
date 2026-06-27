"""Trefftz-plane force integration (JAX port of atpforc.f)."""

from __future__ import annotations

from openavl.jax.backend import jnp
from openavl.jax.types import ReferenceQuantities, TrefftzForces, TrefftzGeometry


def pgmat_jax(mach: jnp.ndarray) -> jnp.ndarray:
    """Build Prandtl-Glauert transformation matrix at zero incidence."""
    binv = 1.0 / jnp.sqrt(1.0 - mach * mach)
    return jnp.array(
        [
            [binv, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


def _filament_velocity(
    ycntr: jnp.ndarray,
    zcntr: jnp.ndarray,
    y1: jnp.ndarray,
    y2: jnp.ndarray,
    z1: jnp.ndarray,
    z2: jnp.ndarray,
    gams: jnp.ndarray,
    rcore: jnp.ndarray,
    hpi: jnp.ndarray,
    sign: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Biot-Savart velocity from semi-infinite filaments (vectorized over sources)."""
    dy1 = ycntr[..., None] - y1
    dy2 = ycntr[..., None] - y2
    dz1 = zcntr[..., None] - z1
    dz2 = zcntr[..., None] - z2
    rsq1 = jnp.hypot(dy1 * dy1 + dz1 * dz1, rcore * rcore)
    rsq2 = jnp.hypot(dy2 * dy2 + dz2 * dz2, rcore * rcore)
    vy = sign * hpi * gams * ((dz1 / rsq1) - (dz2 / rsq2))
    vz = sign * hpi * gams * ((-dy1 / rsq1) + (dy2 / rsq2))
    return vy, vz


def tpforc_jax(
    gamma: jnp.ndarray,
    tgeom: TrefftzGeometry,
    refs: ReferenceQuantities,
) -> TrefftzForces:
    """Trefftz-plane far-field force integration (primal-only, vectorized).

    Y-symmetry is applied here, not by the near-field force aggregator, because
    wake-image filaments affect induced drag before Trefftz totals are summed.
    """
    smap = tgeom.strip_map
    nstrip = smap.ijfrst.shape[0]
    chord = tgeom.chord[:nstrip]
    hpi = jnp.array(1.0 / (2.0 * jnp.pi))
    active = (~smap.lstripoff).astype(gamma.dtype)

    p = pgmat_jax(tgeom.amach)
    yoff = 2.0 * tgeom.ysym
    zoff = 2.0 * tgeom.zsym

    gams = jnp.zeros(nstrip).at[smap.vortex_to_strip].add(gamma) * active

    last_vortex = smap.ijfrst + smap.nvstrp - 1
    rt1 = p @ tgeom.rv1[:, last_vortex]
    rt2 = p @ tgeom.rv2[:, last_vortex]
    rtc = p @ tgeom.rc[:, last_vortex]

    dyt = rt2[1] - rt1[1]
    dzt = rt2[2] - rt1[2]
    dst = jnp.maximum(jnp.hypot(dyt, dzt), 1e-30)
    ny = -dzt / dst
    nz = dyt / dst
    ycntr = rtc[1]
    zcntr = rtc[2]

    dsy = rt2[1, None, :] - rt1[1, None, :]
    dsz = rt2[2, None, :] - rt1[2, None, :]
    dsyz = jnp.sqrt(dsy * dsy + dsz * dsz)
    comp_strip = smap.lncomp[smap.lssurf]
    same_comp = comp_strip[:, None] == comp_strip[None, :]
    rcore = jnp.where(
        same_comp,
        0.0,
        jnp.maximum(tgeom.vrcorec * chord[None, :], tgeom.vrcorew * dsyz),
    )
    gams_jv = gams[None, :] * active[None, :]

    vy, vz = _filament_velocity(
        ycntr,
        zcntr,
        rt1[1, None, :],
        rt2[1, None, :],
        rt1[2, None, :],
        rt2[2, None, :],
        gams_jv,
        rcore,
        hpi,
        jnp.array(1.0),
    )
    vy = jnp.sum(vy, axis=1)
    vz = jnp.sum(vz, axis=1)

    if tgeom.izsym != 0:
        vy_z, vz_z = _filament_velocity(
            ycntr,
            zcntr,
            rt1[1, None, :],
            rt2[1, None, :],
            zoff - rt1[2, None, :],
            zoff - rt2[2, None, :],
            gams_jv,
            jnp.zeros_like(rcore),
            hpi,
            jnp.array(-float(tgeom.izsym)),
        )
        vy = vy + jnp.sum(vy_z, axis=1)
        vz = vz + jnp.sum(vz_z, axis=1)

    if tgeom.iysym != 0:
        vy_y, vz_y = _filament_velocity(
            ycntr,
            zcntr,
            yoff - rt1[1, None, :],
            yoff - rt2[1, None, :],
            rt1[2, None, :],
            rt2[2, None, :],
            gams_jv,
            jnp.zeros_like(rcore),
            hpi,
            jnp.array(-float(tgeom.iysym)),
        )
        vy = vy + jnp.sum(vy_y, axis=1)
        vz = vz + jnp.sum(vz_y, axis=1)

        if tgeom.izsym != 0:
            vy_yz, vz_yz = _filament_velocity(
                ycntr,
                zcntr,
                yoff - rt1[1, None, :],
                yoff - rt2[1, None, :],
                zoff - rt1[2, None, :],
                zoff - rt2[2, None, :],
                gams_jv,
                jnp.zeros_like(rcore),
                hpi,
                jnp.array(float(tgeom.iysym * tgeom.izsym)),
            )
            vy = vy + jnp.sum(vy_yz, axis=1)
            vz = vz + jnp.sum(vz_yz, axis=1)

    dwwake = -(ny * vy + nz * vz) * active

    load_mask = tgeom.lfload[smap.lssurf].astype(gamma.dtype) * active

    clff = jnp.sum(2.0 * gams * dyt / refs.sref * load_mask)
    cyff = jnp.sum(-2.0 * gams * dzt / refs.sref * load_mask)
    cdff = jnp.sum(gams * ((dzt * vy) - (dyt * vz)) / refs.sref * load_mask)

    if tgeom.iysym == 1:
        clff = 2.0 * clff
        cyff = jnp.array(0.0)
        cdff = 2.0 * cdff

    ar = refs.bref * refs.bref / refs.sref
    spanef = jnp.where(cdff == 0.0, 0.0, (clff * clff + cyff * cyff) / (jnp.pi * ar * cdff))

    return TrefftzForces(CL=clff, CY=cyff, CDi=cdff, spanef=spanef, dwwake=dwwake)


def tpforc_jax_jit(
    gamma: jnp.ndarray,
    tgeom: TrefftzGeometry,
    refs: ReferenceQuantities,
) -> TrefftzForces:
    """JIT-safe Trefftz integration (symmetry branches use ``jnp.where``)."""
    smap = tgeom.strip_map
    nstrip = smap.ijfrst.shape[0]
    chord = tgeom.chord[:nstrip]
    hpi = jnp.array(1.0 / (2.0 * jnp.pi))
    active = (~smap.lstripoff).astype(gamma.dtype)

    p = pgmat_jax(tgeom.amach)
    yoff = 2.0 * tgeom.ysym
    zoff = 2.0 * tgeom.zsym

    gams = jnp.zeros(nstrip).at[smap.vortex_to_strip].add(gamma) * active

    last_vortex = smap.ijfrst + smap.nvstrp - 1
    rt1 = p @ tgeom.rv1[:, last_vortex]
    rt2 = p @ tgeom.rv2[:, last_vortex]
    rtc = p @ tgeom.rc[:, last_vortex]

    dyt = rt2[1] - rt1[1]
    dzt = rt2[2] - rt1[2]
    dst = jnp.maximum(jnp.hypot(dyt, dzt), 1e-30)
    ny = -dzt / dst
    nz = dyt / dst
    ycntr = rtc[1]
    zcntr = rtc[2]

    dsy = rt2[1, None, :] - rt1[1, None, :]
    dsz = rt2[2, None, :] - rt1[2, None, :]
    dsyz = jnp.sqrt(dsy * dsy + dsz * dsz)
    comp_strip = smap.lncomp[smap.lssurf]
    same_comp = comp_strip[:, None] == comp_strip[None, :]
    rcore = jnp.where(
        same_comp,
        0.0,
        jnp.maximum(tgeom.vrcorec * chord[None, :], tgeom.vrcorew * dsyz),
    )
    gams_jv = gams[None, :] * active[None, :]

    vy, vz = _filament_velocity(
        ycntr,
        zcntr,
        rt1[1, None, :],
        rt2[1, None, :],
        rt1[2, None, :],
        rt2[2, None, :],
        gams_jv,
        rcore,
        hpi,
        jnp.array(1.0),
    )
    vy = jnp.sum(vy, axis=1)
    vz = jnp.sum(vz, axis=1)

    vy_z, vz_z = _filament_velocity(
        ycntr,
        zcntr,
        rt1[1, None, :],
        rt2[1, None, :],
        zoff - rt1[2, None, :],
        zoff - rt2[2, None, :],
        gams_jv,
        jnp.zeros_like(rcore),
        hpi,
        -tgeom.izsym,
    )
    vy = vy + jnp.where(tgeom.izsym != 0, jnp.sum(vy_z, axis=1), 0.0)
    vz = vz + jnp.where(tgeom.izsym != 0, jnp.sum(vz_z, axis=1), 0.0)

    vy_y, vz_y = _filament_velocity(
        ycntr,
        zcntr,
        yoff - rt1[1, None, :],
        yoff - rt2[1, None, :],
        rt1[2, None, :],
        rt2[2, None, :],
        gams_jv,
        jnp.zeros_like(rcore),
        hpi,
        -tgeom.iysym,
    )
    vy = vy + jnp.where(tgeom.iysym != 0, jnp.sum(vy_y, axis=1), 0.0)
    vz = vz + jnp.where(tgeom.iysym != 0, jnp.sum(vz_y, axis=1), 0.0)

    vy_yz, vz_yz = _filament_velocity(
        ycntr,
        zcntr,
        yoff - rt1[1, None, :],
        yoff - rt2[1, None, :],
        zoff - rt1[2, None, :],
        zoff - rt2[2, None, :],
        gams_jv,
        jnp.zeros_like(rcore),
        hpi,
        tgeom.iysym * tgeom.izsym,
    )
    both_sym = (tgeom.iysym != 0) & (tgeom.izsym != 0)
    vy = vy + jnp.where(both_sym, jnp.sum(vy_yz, axis=1), 0.0)
    vz = vz + jnp.where(both_sym, jnp.sum(vz_yz, axis=1), 0.0)

    dwwake = -(ny * vy + nz * vz) * active

    load_mask = tgeom.lfload[smap.lssurf].astype(gamma.dtype) * active

    clff = jnp.sum(2.0 * gams * dyt / refs.sref * load_mask)
    cyff = jnp.sum(-2.0 * gams * dzt / refs.sref * load_mask)
    cdff = jnp.sum(gams * ((dzt * vy) - (dyt * vz)) / refs.sref * load_mask)

    clff = jnp.where(tgeom.iysym == 1, 2.0 * clff, clff)
    cyff = jnp.where(tgeom.iysym == 1, jnp.array(0.0), cyff)
    cdff = jnp.where(tgeom.iysym == 1, 2.0 * cdff, cdff)

    ar = refs.bref * refs.bref / refs.sref
    spanef = jnp.where(cdff == 0.0, 0.0, (clff * clff + cyff * cyff) / (jnp.pi * ar * cdff))

    return TrefftzForces(CL=clff, CY=cyff, CDi=cdff, spanef=spanef, dwwake=dwwake)
