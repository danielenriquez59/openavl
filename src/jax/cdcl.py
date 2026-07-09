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

    valid = (clmax > cl0) & (cl0 > clmin)

    # Safe-where: force the (clmin-cl0)/(clmax-cl0) denominators to a
    # harmless nonzero value whenever the polar is degenerate, so the
    # masked-out branch never carries a NaN/Inf gradient (see A10).
    dlo = clmin - cl0
    dhi = clmax - cl0
    dlo_safe = jnp.where(dlo != 0.0, dlo, 1.0)
    dhi_safe = jnp.where(dhi != 0.0, dhi, 1.0)

    clinc = jnp.array(0.2)
    cdinc = jnp.array(0.05)
    clfac = 1.0 / clinc
    # Matching parameters, ported verbatim from AVL's cdcl.f (matches AVL
    # 3.52 bit-for-bit). Note: dCD/dCL is only continuous at the stall joins
    # when |CLMIN-CL0| = 1 (respectively |CLMAX-CL0| = 1) -- see the NumPy
    # port (src/aero/cdcl.py) for the full explanation. CD itself is always
    # continuous.
    cdx1 = 2.0 * (cdmin - cd0) * dlo / (dlo_safe * dlo_safe)
    cdx2 = 2.0 * (cdmax - cd0) * dhi / (dhi_safe * dhi_safe)

    clv = cl
    cd_below = (
        cdmin
        + cdinc * clfac * clfac * (clv - clmin) ** 2
        + cdx1 * (1.0 - (clv - cl0) / dlo_safe)
    )
    cd_mid_lo = cd0 + (cdmin - cd0) * (clv - cl0) ** 2 / (dlo_safe * dlo_safe)
    cd_mid_hi = cd0 + (cdmax - cd0) * (clv - cl0) ** 2 / (dhi_safe * dhi_safe)
    cd_above = (
        cdmax
        + cdinc * clfac * clfac * (clv - clmax) ** 2
        - cdx2 * (1.0 - (clv - cl0) / dhi_safe)
    )

    cd = jnp.where(
        clv < clmin,
        cd_below,
        jnp.where(clv < cl0, cd_mid_lo, jnp.where(clv < clmax, cd_mid_hi, cd_above)),
    )
    # Invalid (no-polar) strips return zero drag instead of NaN, so they
    # stay finite and gradient-safe when vmapped over every strip.
    return jnp.where(valid, cd, 0.0)
