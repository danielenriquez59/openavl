"""Horseshoe vortex induced velocity kernel (JAX port of aero/vortex.py).

Same as ``openavl.aero.vortex``, with finite core radius.

Original Scully (AKA Burnham-Hallock) core model::

    Vtan = Gam/(2*pi) * r/(r^2 + rcore^2)

Uses Leishman's R^4 variant of Scully (AKA Burnham-Hallock) core model::

    Vtan = Gam/(2*pi) * r/sqrt(r^4 + rcore^4)
"""

from __future__ import annotations

from openavl.jax.backend import jnp

# Exact 1/(4*pi), matching jax/aic.py's PI4INV (B5: the previous truncated
# literal 0.079577472 carried ~1e-9 relative error relative to this kernel).
PI4INV = 1.0 / (4.0 * jnp.pi)


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

    rcore2 = rcore * rcore
    rcore4 = rcore2 * rcore2

    u = jnp.array(0.0, dtype=jnp.result_type(x, beta))
    v = jnp.array(0.0, dtype=jnp.result_type(x, beta))
    w = jnp.array(0.0, dtype=jnp.result_type(x, beta))

    # Safe-where (A9): clamp every sqrt's *argument* to a harmless nonzero
    # value on the branch that jnp.where will discard, rather than clamping
    # the sqrt's output afterward. jnp.sqrt has an infinite local derivative
    # at a zero argument, so merely masking its output still lets a 0
    # (from the discarded-branch cotangent) multiply that infinite local
    # gradient into NaN; clamping the argument keeps the local derivative
    # finite everywhere so 0 x finite = 0, not NaN.
    leg_a_active = asq != 0.0
    leg_b_active = bsq != 0.0
    bound_active = jnp.logical_and(lbound, jnp.logical_and(leg_a_active, leg_b_active))

    asq_safe = jnp.where(leg_a_active, asq, 1.0)
    bsq_safe = jnp.where(leg_b_active, bsq, 1.0)
    amag = jnp.sqrt(asq_safe)
    bmag = jnp.sqrt(bsq_safe)

    axb1 = ay1 * bz1 - az1 * by1
    axb2 = az1 * bx1 - ax1 * bz1
    axb3 = ax1 * by1 - ay1 * bx1
    axbsq = axb1 * axb1 + axb2 * axb2 + axb3 * axb3

    bound_valid = jnp.logical_and(bound_active, axbsq != 0.0)
    adb = ax1 * bx1 + ay1 * by1 + az1 * bz1
    alsq = asq + bsq - 2.0 * adb

    t1 = bsq - adb
    t2 = asq - adb
    s1_arg = jnp.where(bound_valid, bsq * bsq + rcore4, 1.0)
    s2_arg = jnp.where(bound_valid, asq * asq + rcore4, 1.0)
    s1 = jnp.sqrt(jnp.sqrt(s1_arg))
    s2 = jnp.sqrt(jnp.sqrt(s2_arg))
    num = t1 / s1 + t2 / s2
    den_arg = jnp.where(bound_valid, axbsq * axbsq + alsq * alsq * rcore4, 1.0)
    den = jnp.sqrt(den_arg)
    t_bound = num / den

    u = jnp.where(bound_valid, axb1 * t_bound, u)
    v = jnp.where(bound_valid, axb2 * t_bound, v)
    w = jnp.where(bound_valid, axb3 * t_bound, w)

    axisq = az1 * az1 + ay1 * ay1
    adx = ax1
    rsq = axisq
    rdenom_a_arg = jnp.where(leg_a_active, rsq * rsq + rcore4, 1.0)
    rdenom_a = jnp.sqrt(rdenom_a_arg)
    t_a = -(1.0 - adx / amag) / rdenom_a
    v = jnp.where(leg_a_active, v + az1 * t_a, v)
    w = jnp.where(leg_a_active, w - ay1 * t_a, w)

    bxisq = bz1 * bz1 + by1 * by1
    bdx = bx1
    rsq_b = bxisq
    rdenom_b_arg = jnp.where(leg_b_active, rsq_b * rsq_b + rcore4, 1.0)
    rdenom_b = jnp.sqrt(rdenom_b_arg)
    t_b = (1.0 - bdx / bmag) / rdenom_b
    v = jnp.where(leg_b_active, v + bz1 * t_b, v)
    w = jnp.where(leg_b_active, w - by1 * t_b, w)

    u = (u * PI4INV) / bx
    v = v * PI4INV
    w = w * PI4INV

    return u, v, w
