"""Freestream velocity from flow angles (JAX port of VINFAB)."""

from __future__ import annotations

from openavl.jax.backend import jnp


def stability_rates_to_body(alfa, rates, refs, lnasa_sa):
    """Convert AVL stability-axis (pb/2V, qc/2V, rb/2V) to body omega/V.

    Keep this conversion inside AD: holding stability rates fixed while
    varying alpha also changes the body-axis rotation vector.
    """
    direction = -1.0 if lnasa_sa else 1.0
    p = rates[0] * (2.0 / refs.bref) * direction
    q = rates[1] * (2.0 / refs.cref)
    r = rates[2] * (2.0 / refs.bref) * direction
    ca, sa = jnp.cos(alfa), jnp.sin(alfa)
    return jnp.array([p * ca - r * sa, q, p * sa + r * ca])


def vinfab(alfa: jnp.ndarray, beta: jnp.ndarray) -> jnp.ndarray:
    """Return unit freestream velocity vector from angle of attack and sideslip.

    Parameters
    ----------
    alfa:
        Angle of attack (radians).
    beta:
        Sideslip angle (radians).

    Returns
    -------
    jnp.ndarray
        Freestream velocity direction ``[3]`` in body axes.
    """
    sina = jnp.sin(alfa)
    cosa = jnp.cos(alfa)
    sinb = jnp.sin(beta)
    cosb = jnp.cos(beta)
    return jnp.array([cosa * cosb, -sinb, sina * cosb])
