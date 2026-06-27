"""Horseshoe vortex induced velocity kernel (JAX port of aero/vortex.py)."""

from __future__ import annotations

from openavl.jax.backend import jnp

PI4INV = 0.079577472


def vorvelc_jax(
    x: jnp.ndarray,
    y: jnp.ndarray,
    z: jnp.ndarray,
    lbound: jnp.ndarray,
    x1: jnp.ndarray,
    y1: jnp.ndarray,
    z1: jnp.ndarray,
    x2: jnp.ndarray,
    y2: jnp.ndarray,
    z2: jnp.ndarray,
    beta: jnp.ndarray,
    rcore: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute induced velocity at (x,y,z) due to a horseshoe vortex segment."""
    bx = beta
    ax1 = (x1 - x) / bx
    ay1 = y1 - y
    az1 = z1 - z

    bx1 = (x2 - x) / bx
    by1 = y2 - y
    bz1 = z2 - z

    asq = ax1 * ax1 + ay1 * ay1 + az1 * az1
    bsq = bx1 * bx1 + by1 * by1 + bz1 * bz1

    amag = jnp.sqrt(asq)
    bmag = jnp.sqrt(bsq)

    rcore2 = rcore * rcore
    rcore4 = rcore2 * rcore2

    u = jnp.array(0.0, dtype=jnp.result_type(x, beta))
    v = jnp.array(0.0, dtype=jnp.result_type(x, beta))
    w = jnp.array(0.0, dtype=jnp.result_type(x, beta))

    bound_active = jnp.logical_and(lbound, amag * bmag != 0.0)
    axb1 = ay1 * bz1 - az1 * by1
    axb2 = az1 * bx1 - ax1 * bz1
    axb3 = ax1 * by1 - ay1 * bx1
    axbsq = axb1 * axb1 + axb2 * axb2 + axb3 * axb3

    bound_valid = jnp.logical_and(bound_active, axbsq != 0.0)
    adb = ax1 * bx1 + ay1 * by1 + az1 * bz1
    alsq = asq + bsq - 2.0 * adb

    t1 = bsq - adb
    t2 = asq - adb
    s1 = jnp.sqrt(jnp.sqrt(bsq * bsq + rcore4))
    s2 = jnp.sqrt(jnp.sqrt(asq * asq + rcore4))
    num = t1 / s1 + t2 / s2
    den = jnp.sqrt(axbsq * axbsq + alsq * alsq * rcore4)
    t_bound = num / den

    u = jnp.where(bound_valid, axb1 * t_bound, u)
    v = jnp.where(bound_valid, axb2 * t_bound, v)
    w = jnp.where(bound_valid, axb3 * t_bound, w)

    leg_a_active = amag != 0.0
    axisq = az1 * az1 + ay1 * ay1
    adx = ax1
    rsq = axisq
    t_a = -(1.0 - adx / amag) / jnp.sqrt(rsq * rsq + rcore4)
    v = jnp.where(leg_a_active, v + az1 * t_a, v)
    w = jnp.where(leg_a_active, w - ay1 * t_a, w)

    leg_b_active = bmag != 0.0
    bxisq = bz1 * bz1 + by1 * by1
    bdx = bx1
    rsq_b = bxisq
    t_b = (1.0 - bdx / bmag) / jnp.sqrt(rsq_b * rsq_b + rcore4)
    v = jnp.where(leg_b_active, v + bz1 * t_b, v)
    w = jnp.where(leg_b_active, w - by1 * t_b, w)

    u = (u * PI4INV) / bx
    v = v * PI4INV
    w = w * PI4INV

    return u, v, w
