"""AIC matrix assembly (JAX port of aero/aic.py)."""

from __future__ import annotations

from openavl.jax.backend import jnp
from openavl.jax.vortex import vorvelc_jax

PI4INV = 1.0 / (4.0 * jnp.pi)


def cross_jax(u: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
    """Cross product of two 3-vectors."""
    return jnp.array(
        [
            u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0],
        ]
    )


def dot_jax(u: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
    """Dot product of two 3-vectors."""
    return jnp.dot(u, v)


def srdvelc_jax(
    x: jnp.ndarray,
    y: jnp.ndarray,
    z: jnp.ndarray,
    x1: jnp.ndarray,
    y1: jnp.ndarray,
    z1: jnp.ndarray,
    x2: jnp.ndarray,
    y2: jnp.ndarray,
    z2: jnp.ndarray,
    beta: jnp.ndarray,
    rcore: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Source/doublet panel induced velocity and velocity derivative matrix."""
    r1 = jnp.array([(x1 - x) / beta, y1 - y, z1 - z])
    r2 = jnp.array([(x2 - x) / beta, y2 - y, z2 - z])

    rcsq = rcore * rcore
    r1sq = jnp.dot(r1, r1)
    r2sq = jnp.dot(r2, r2)
    r1sqeps = r1sq + rcsq
    r2sqeps = r2sq + rcsq
    r1eps = jnp.sqrt(r1sqeps)
    r2eps = jnp.sqrt(r2sqeps)

    rdr = jnp.dot(r1, r2)
    rxr = jnp.cross(r1, r2)
    rxr1, rxr2, rxr3 = rxr[0], rxr[1], rxr[2]

    xdx = rxr1 * rxr1 + rxr2 * rxr2 + rxr3 * rxr3
    all_ = r1sq + r2sq - 2.0 * rdr
    den = rcsq * all_ + xdx

    ai1 = ((rdr + rcsq) / r1eps - r2eps) / den
    ai2 = ((rdr + rcsq) / r2eps - r1eps) / den

    uvws = jnp.zeros(3, dtype=jnp.float64)
    uvwd = jnp.zeros((3, 3), dtype=jnp.float64)

    for k in range(3):
        r1k = r1[k]
        r2k = r2[k]

        uvws = uvws.at[k].set((r1k * ai1) + (r2k * ai2))

        rr1 = (
            ((r1k + r2k) / r1eps)
            - ((r1k * (rdr + rcsq)) / (r1eps * r1eps * r1eps))
            - (r2k / r2eps)
        )
        rr2 = (
            ((r1k + r2k) / r2eps)
            - ((r2k * (rdr + rcsq)) / (r2eps * r2eps * r2eps))
            - (r1k / r1eps)
        )
        rrt = (2.0 * r1k * (r2sq - rdr)) + (2.0 * r2k * (r1sq - rdr))

        aj1 = ((rr1 - (ai1 * rrt)) / den)
        aj2 = ((rr2 - (ai2 * rrt)) / den)

        for j in range(3):
            uvwd = uvwd.at[k, j].set(-(aj1 * r1[j]) - (aj2 * r2[j]))
        uvwd = uvwd.at[k, k].set(uvwd[k, k] - ai1 - ai2)

    uvws = uvws.at[0].set((uvws[0] * PI4INV) / beta)
    uvws = uvws.at[1].set(uvws[1] * PI4INV)
    uvws = uvws.at[2].set(uvws[2] * PI4INV)

    scale = jnp.array([PI4INV / beta, PI4INV, PI4INV])
    uvwd = uvwd * scale[:, None]

    return uvws, uvwd


def _vvor_pair(
    x: jnp.ndarray,
    y: jnp.ndarray,
    z: jnp.ndarray,
    i: jnp.ndarray,
    j: jnp.ndarray,
    betm: jnp.ndarray,
    iysym: jnp.ndarray,
    ysym: jnp.ndarray,
    izsym: jnp.ndarray,
    zsym: jnp.ndarray,
    vrcorec: jnp.ndarray,
    vrcorew: jnp.ndarray,
    nc: jnp.ndarray,
    nv: jnp.ndarray,
    rv1: jnp.ndarray,
    rv2: jnp.ndarray,
    ncompv: jnp.ndarray,
    ncompc: jnp.ndarray,
    chordv: jnp.ndarray,
    lvtest: bool,
) -> jnp.ndarray:
    """Induced velocity at control point i due to vortex j."""
    ds_y = rv2[1, j] - rv1[1, j]
    ds_z = rv2[2, j] - rv1[2, j]
    dsyz = jnp.sqrt(ds_y * ds_y + ds_z * ds_z)

    rcore_default = 0.0001 * dsyz
    rc1 = vrcorec * chordv[j]
    rc2 = vrcorew * dsyz
    rcore_comp = jnp.maximum(rc1, rc2)
    use_comp_core = jnp.logical_and(nc == nv, ncompc[i] != ncompv[j])
    rcore = jnp.where(use_comp_core, rcore_comp, rcore_default)

    yoff = 2.0 * ysym
    zoff = 2.0 * zsym
    fysym = jnp.asarray(iysym, dtype=jnp.float64)
    fzsym = jnp.asarray(izsym, dtype=jnp.float64)

    lbound_real = jnp.logical_not(jnp.logical_and(lvtest, i == j))
    u, v, w = vorvelc_jax(
        x, y, z, lbound_real,
        rv1[0, j], rv1[1, j], rv1[2, j],
        rv2[0, j], rv2[1, j], rv2[2, j],
        betm, rcore,
    )

    ui = jnp.array(0.0, dtype=jnp.float64)
    vi = jnp.array(0.0, dtype=jnp.float64)
    wi = jnp.array(0.0, dtype=jnp.float64)

    xave = 0.5 * (rv1[0, j] + rv2[0, j])
    yave = yoff - 0.5 * (rv1[1, j] + rv2[1, j])
    zave = 0.5 * (rv1[2, j] + rv2[2, j])
    at_image_mid = jnp.logical_and(
        jnp.logical_and(x == xave, y == yave),
        z == zave,
    )
    lbound_img = jnp.logical_not(jnp.logical_and(iysym == 1, at_image_mid))

    iu, iv, iw = vorvelc_jax(
        x, y, z, lbound_img,
        rv2[0, j], yoff - rv2[1, j], rv2[2, j],
        rv1[0, j], yoff - rv1[1, j], rv1[2, j],
        betm, rcore,
    )
    ui = jnp.where(iysym != 0, iu * fysym, ui)
    vi = jnp.where(iysym != 0, iv * fysym, vi)
    wi = jnp.where(iysym != 0, iw * fysym, wi)

    zu, zv, zw = vorvelc_jax(
        x, y, z, True,
        rv2[0, j], rv2[1, j], zoff - rv2[2, j],
        rv1[0, j], rv1[1, j], zoff - rv1[2, j],
        betm, rcore,
    )
    u = jnp.where(izsym != 0, u + zu * fzsym, u)
    v = jnp.where(izsym != 0, v + zv * fzsym, v)
    w = jnp.where(izsym != 0, w + zw * fzsym, w)

    yzu, yzv, yzw = vorvelc_jax(
        x, y, z, True,
        rv1[0, j], yoff - rv1[1, j], zoff - rv1[2, j],
        rv2[0, j], yoff - rv2[1, j], zoff - rv2[2, j],
        betm, rcore,
    )
    both_sym = jnp.logical_and(iysym != 0, izsym != 0)
    ui = jnp.where(both_sym, ui + yzu * fysym * fzsym, ui)
    vi = jnp.where(both_sym, vi + yzv * fysym * fzsym, vi)
    wi = jnp.where(both_sym, wi + yzw * fysym * fzsym, wi)

    us = u + ui
    vs = v + vi
    ws = w + wi

    valid = jnp.logical_and(jnp.isfinite(dsyz), dsyz != 0.0)
    us = jnp.where(valid, us, 0.0)
    vs = jnp.where(valid, vs, 0.0)
    ws = jnp.where(valid, ws, 0.0)

    us = jnp.where(jnp.isfinite(us), us, 0.0)
    vs = jnp.where(jnp.isfinite(vs), vs, 0.0)
    ws = jnp.where(jnp.isfinite(ws), ws, 0.0)

    return jnp.array([us, vs, ws])


def vvor_jax(
    betm: float,
    iysym: int,
    ysym: float,
    izsym: int,
    zsym: float,
    vrcorec: float,
    vrcorew: float,
    rv1: jnp.ndarray,
    rv2: jnp.ndarray,
    ncompv: jnp.ndarray,
    chordv: jnp.ndarray,
    rc: jnp.ndarray,
    ncompc: jnp.ndarray,
    lvtest: bool,
) -> jnp.ndarray:
    """Assemble velocity influence coefficient matrix for vortex horseshoes.

    Replaces the original nested per-pair vmap with a single batched kernel
    operating on full [nc, nv] grids, eliminating O(nc*nv) vmap dispatch
    overhead.  ``vorvelc_jax`` broadcasts over array inputs unchanged.
    """
    nc = rc.shape[1]
    nv = rv1.shape[1]

    # --- Vortex half-span geometry: [nv] quantities broadcast across nc ---
    ds_y = rv2[1] - rv1[1]  # [nv]
    ds_z = rv2[2] - rv1[2]
    dsyz = jnp.sqrt(ds_y * ds_y + ds_z * ds_z)  # [nv]

    rcore_default = 0.0001 * dsyz  # [nv]
    rc1 = vrcorec * chordv  # [nv]
    rc2 = vrcorew * dsyz
    rcore_comp = jnp.maximum(rc1, rc2)

    # Component-matching pairs use the larger core radius
    use_comp_core = jnp.logical_and(
        nc == nv, ncompc[:, None] != ncompv[None, :]
    )  # [nc, nv]
    rcore = jnp.where(use_comp_core, rcore_comp[None, :], rcore_default[None, :])  # [nc, nv]

    # --- Control-point coordinates broadcast to [nc, nv] ---
    x = rc[0, :, None]  # [nc, 1]
    y = rc[1, :, None]
    z = rc[2, :, None]

    # Vortex endpoints broadcast to [nc, nv]
    x1 = rv1[0, None, :]  # [1, nv]
    y1 = rv1[1, None, :]
    z1 = rv1[2, None, :]
    x2 = rv2[0, None, :]
    y2 = rv2[1, None, :]
    z2 = rv2[2, None, :]

    # Bound-vortex flag: suppress self-influence on the diagonal when lvtest=True
    i_idx = jnp.arange(nc, dtype=jnp.int32)[:, None]  # [nc, 1]
    j_idx = jnp.arange(nv, dtype=jnp.int32)[None, :]  # [1, nv]
    lbound_real = jnp.logical_not(
        jnp.logical_and(lvtest, i_idx == j_idx)
    )  # [nc, nv]

    u, v, w = vorvelc_jax(x, y, z, lbound_real, x1, y1, z1, x2, y2, z2, betm, rcore)
    # u, v, w: [nc, nv]

    yoff = 2.0 * ysym
    zoff = 2.0 * zsym

    ui = jnp.zeros((nc, nv), dtype=jnp.float64)
    vi = jnp.zeros((nc, nv), dtype=jnp.float64)
    wi = jnp.zeros((nc, nv), dtype=jnp.float64)

    # Y-symmetry image vortex
    if iysym != 0:
        xave = 0.5 * (rv1[0] + rv2[0])  # [nv]
        yave = yoff - 0.5 * (rv1[1] + rv2[1])
        zave = 0.5 * (rv1[2] + rv2[2])
        at_image_mid = (
            (x == xave[None, :]) & (y == yave[None, :]) & (z == zave[None, :])
        )  # [nc, nv]
        lbound_img = jnp.logical_not(
            jnp.logical_and(iysym == 1, at_image_mid)
        )  # [nc, nv]
        iu, iv_s, iw = vorvelc_jax(
            x, y, z, lbound_img,
            rv2[0, None, :], yoff - rv2[1, None, :], rv2[2, None, :],
            rv1[0, None, :], yoff - rv1[1, None, :], rv1[2, None, :],
            betm, rcore,
        )
        ui = iu * float(iysym)
        vi = iv_s * float(iysym)
        wi = iw * float(iysym)

    # Z-symmetry image vortex
    if izsym != 0:
        zu, zv, zw = vorvelc_jax(
            x, y, z, True,
            rv2[0, None, :], rv2[1, None, :], zoff - rv2[2, None, :],
            rv1[0, None, :], rv1[1, None, :], zoff - rv1[2, None, :],
            betm, rcore,
        )
        u = u + zu * float(izsym)
        v = v + zv * float(izsym)
        w = w + zw * float(izsym)

    # Combined Y+Z symmetry corner image
    if iysym != 0 and izsym != 0:
        yzu, yzv, yzw = vorvelc_jax(
            x, y, z, True,
            rv1[0, None, :], yoff - rv1[1, None, :], zoff - rv1[2, None, :],
            rv2[0, None, :], yoff - rv2[1, None, :], zoff - rv2[2, None, :],
            betm, rcore,
        )
        ui = ui + yzu * float(iysym) * float(izsym)
        vi = vi + yzv * float(iysym) * float(izsym)
        wi = wi + yzw * float(iysym) * float(izsym)

    us = u + ui  # [nc, nv]
    vs = v + vi
    ws = w + wi

    valid = jnp.logical_and(jnp.isfinite(dsyz), dsyz != 0.0)  # [nv]
    us = jnp.where(valid[None, :], us, 0.0)
    vs = jnp.where(valid[None, :], vs, 0.0)
    ws = jnp.where(valid[None, :], ws, 0.0)
    us = jnp.where(jnp.isfinite(us), us, 0.0)
    vs = jnp.where(jnp.isfinite(vs), vs, 0.0)
    ws = jnp.where(jnp.isfinite(ws), ws, 0.0)

    return jnp.stack([us, vs, ws], axis=0)  # [3, nc, nv]


def srdset_jax(
    betm: float,
    xyzref: jnp.ndarray,
    iysym: int,
    nbody: int,
    lfrst: jnp.ndarray,
    nl: jnp.ndarray,
    rl: jnp.ndarray,
    radl: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Initialize body source and doublet strength sensitivities."""
    pi = 3.14159265
    beta = betm
    nldim = rl.shape[1]

    src_u = jnp.zeros((nldim, 6), dtype=jnp.float64)
    dbl_u = jnp.zeros((3, nldim, 6), dtype=jnp.float64)

    for ib in range(nbody):
        l1b = int(lfrst[ib])
        l2b = l1b + int(nl[ib]) - 1
        blen = jnp.abs(rl[0, l2b] - rl[0, l1b])
        sdfac = jnp.where(
            jnp.logical_and(iysym == 1, rl[1, l1b] <= 0.001 * blen),
            0.5,
            1.0,
        )

        for ilseg in range(int(nl[ib]) - 1):
            l1 = int(lfrst[ib] + ilseg)
            l2 = l1 + 1
            l = l1

            drl = jnp.array([
                (rl[0, l2] - rl[0, l1]) / beta,
                rl[1, l2] - rl[1, l1],
                rl[2, l2] - rl[2, l1],
            ])
            drlmag = jnp.sqrt(jnp.dot(drl, drl))
            drlmi = jnp.where(drlmag == 0.0, 0.0, 1.0 / drlmag)
            esl = drl * drlmi

            adel = pi * (radl[l2] * radl[l2] - radl[l1] * radl[l1]) * sdfac
            aavg = pi * 0.5 * (radl[l2] * radl[l2] + radl[l1] * radl[l1]) * sdfac

            rlref = jnp.array([
                0.5 * (rl[0, l2] + rl[0, l1]) - xyzref[0],
                0.5 * (rl[1, l2] + rl[1, l1]) - xyzref[1],
                0.5 * (rl[2, l2] + rl[2, l1]) - xyzref[2],
            ])

            for iu in range(6):
                urel = jnp.zeros(3, dtype=jnp.float64)
                wrot = jnp.zeros(3, dtype=jnp.float64)
                if iu < 3:
                    urel = urel.at[iu].set(1.0)
                else:
                    wrot = wrot.at[iu - 3].set(1.0)
                    urel = cross_jax(rlref, wrot)
                urel = urel.at[0].set(urel[0] / beta)

                us = dot_jax(urel, esl)
                un = urel - us * esl

                src_u = src_u.at[l, iu].set(adel * us)
                dbl_u = dbl_u.at[0, l, iu].set(aavg * un[0] * drlmag * 2.0)
                dbl_u = dbl_u.at[1, l, iu].set(aavg * un[1] * drlmag * 2.0)
                dbl_u = dbl_u.at[2, l, iu].set(aavg * un[2] * drlmag * 2.0)

    return src_u, dbl_u


def vsrd_jax(
    betm: float,
    iysym: int,
    ysym: float,
    izsym: int,
    zsym: float,
    srcore: float,
    nbody: int,
    lfrst: jnp.ndarray,
    nl: jnp.ndarray,
    rl: jnp.ndarray,
    radl: jnp.ndarray,
    nu: int,
    src_u: jnp.ndarray,
    dbl_u: jnp.ndarray,
    rc: jnp.ndarray,
) -> jnp.ndarray:
    """Assemble body source/doublet velocity influence matrix."""
    nc = rc.shape[1]
    wc_u = jnp.zeros((3, nc, nu), dtype=jnp.float64)
    fysym = jnp.asarray(iysym, dtype=jnp.float64)
    fzsym = jnp.asarray(izsym, dtype=jnp.float64)
    yoff = 2.0 * ysym
    zoff = 2.0 * zsym

    for ib in range(nbody):
        for ilseg in range(int(nl[ib]) - 1):
            l1 = int(lfrst[ib] + ilseg)
            l2 = l1 + 1
            l = l1

            ravg = jnp.sqrt(0.5 * (radl[l2] * radl[l2] + radl[l1] * radl[l1]))
            dx = rl[0, l2] - rl[0, l1]
            dy = rl[1, l2] - rl[1, l1]
            dz = rl[2, l2] - rl[2, l1]
            rlavg = jnp.sqrt(dx * dx + dy * dy + dz * dz)
            rcore = jnp.where(srcore > 0, srcore * ravg, srcore * rlavg)

            for i in range(nc):
                uvws, uvwd = srdvelc_jax(
                    rc[0, i], rc[1, i], rc[2, i],
                    rl[0, l1], rl[1, l1], rl[2, l1],
                    rl[0, l2], rl[1, l2], rl[2, l2],
                    betm, rcore,
                )

                for iu in range(nu):
                    contrib = (
                        uvws * src_u[l, iu]
                        + uvwd[0, :] * dbl_u[0, l, iu]
                        + uvwd[1, :] * dbl_u[1, l, iu]
                        + uvwd[2, :] * dbl_u[2, l, iu]
                    )
                    wc_u = wc_u.at[:, i, iu].add(contrib)

                if iysym != 0:
                    vsrc_y, vdbl_y = srdvelc_jax(
                        rc[0, i], rc[1, i], rc[2, i],
                        rl[0, l1], yoff - rl[1, l1], rl[2, l1],
                        rl[0, l2], yoff - rl[1, l2], rl[2, l2],
                        betm, rcore,
                    )
                    for iu in range(nu):
                        contrib = (
                            vsrc_y * src_u[l, iu]
                            + vdbl_y[0, :] * dbl_u[0, l, iu]
                            - vdbl_y[1, :] * dbl_u[1, l, iu]
                            + vdbl_y[2, :] * dbl_u[2, l, iu]
                        )
                        wc_u = wc_u.at[:, i, iu].add(contrib * fysym)

                if izsym != 0:
                    vsrc_z, vdbl_z = srdvelc_jax(
                        rc[0, i], rc[1, i], rc[2, i],
                        rl[0, l1], rl[1, l1], zoff - rl[2, l1],
                        rl[0, l2], rl[1, l2], zoff - rl[2, l2],
                        betm, rcore,
                    )
                    for iu in range(nu):
                        contrib = (
                            vsrc_z * src_u[l, iu]
                            + vdbl_z[0, :] * dbl_u[0, l, iu]
                            + vdbl_z[1, :] * dbl_u[1, l, iu]
                            - vdbl_z[2, :] * dbl_u[2, l, iu]
                        )
                        wc_u = wc_u.at[:, i, iu].add(contrib * fzsym)

                    if iysym != 0:
                        vsrc_yz, vdbl_yz = srdvelc_jax(
                            rc[0, i], rc[1, i], rc[2, i],
                            rl[0, l1], yoff - rl[1, l1], zoff - rl[2, l1],
                            rl[0, l2], yoff - rl[1, l2], zoff - rl[2, l2],
                            betm, rcore,
                        )
                        for iu in range(nu):
                            contrib = (
                                vsrc_yz * src_u[l, iu]
                                + vdbl_yz[0, :] * dbl_u[0, l, iu]
                                - vdbl_yz[1, :] * dbl_u[1, l, iu]
                                - vdbl_yz[2, :] * dbl_u[2, l, iu]
                            )
                            wc_u = wc_u.at[:, i, iu].add(contrib * fysym * fzsym)

    return wc_u
