"""Freestream velocity from flow angles (JAX port of VINFAB)."""

from __future__ import annotations

from openavl.jax.backend import jnp


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
