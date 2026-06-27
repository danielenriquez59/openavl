"""Viscous drag polar lookup (JAX port of cdcl.f)."""

from __future__ import annotations

from openavl.jax.backend import jnp


def cdcl_jax(cdclpol: jnp.ndarray, cl: jnp.ndarray) -> jnp.ndarray:
    """Return CD from a piecewise-quadratic drag polar.

    Parameters
    ----------
    cdclpol:
        Six polar breakpoints ``[clmin, cdmin, cl0, cd0, clmax, cdmax]``.
    cl:
        Lift coefficient(s) at which to evaluate CD.

    Returns
    -------
    jnp.ndarray
        Profile drag coefficient CD (same shape as ``cl``).
    """
    clmin = cdclpol[0]
    cdmin = cdclpol[1]
    cl0 = cdclpol[2]
    cd0 = cdclpol[3]
    clmax = cdclpol[4]
    cdmax = cdclpol[5]

    clinc = jnp.array(0.2)
    cdinc = jnp.array(0.05)
    clfac = 1.0 / clinc
    cdx1 = 2.0 * (cdmin - cd0) * (clmin - cl0) / ((clmin - cl0) ** 2)
    cdx2 = 2.0 * (cdmax - cd0) * (clmax - cl0) / ((clmax - cl0) ** 2)

    clv = cl
    cd_below = (
        cdmin
        + cdinc * clfac * clfac * (clv - clmin) ** 2
        + cdx1 * (1.0 - (clv - cl0) / (clmin - cl0))
    )
    cd_mid_lo = cd0 + (cdmin - cd0) * (clv - cl0) ** 2 / ((clmin - cl0) ** 2)
    cd_mid_hi = cd0 + (cdmax - cd0) * (clv - cl0) ** 2 / ((clmax - cl0) ** 2)
    cd_above = (
        cdmax
        + cdinc * clfac * clfac * (clv - clmax) ** 2
        - cdx2 * (1.0 - (clv - cl0) / (clmax - cl0))
    )

    cd = jnp.where(
        clv < clmin,
        cd_below,
        jnp.where(clv < cl0, cd_mid_lo, jnp.where(clv < clmax, cd_mid_hi, cd_above)),
    )
    valid = (clmax > cl0) & (cl0 > clmin)
    return jnp.where(valid, cd, jnp.nan)
