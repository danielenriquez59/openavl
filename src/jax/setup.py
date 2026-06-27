"""Circulation RHS assembly and velocity recovery (JAX port of GUCALC/GAMSUM/VELSUM)."""

from __future__ import annotations

from openavl.jax.backend import jnp
from openavl.jax.freestream import vinfab
from openavl.jax.solve import solve_circulation, solve_from_lu
from openavl.jax.types import CirculationGeometry, FlowCondition, ReferenceQuantities


def build_rhs(
    geom: CirculationGeometry,
    flow: FlowCondition,
    refs: ReferenceQuantities,
) -> jnp.ndarray:
    """Build the AIC right-hand side for a specific flow condition.

    Replaces the six-column unit-solve approach with a single RHS vector built
    from freestream, rotation, body-source, and control-deflection contributions.
    """
    vinf = vinfab(flow.alfa, flow.beta)
    u = jnp.concatenate((vinf, flow.wrot))
    wcsrd = jnp.einsum("kin,n->ki", geom.wcsrd_u, u)

    rrot = geom.rc - refs.xyzref[:, None]
    vrot = jnp.cross(rrot.T, flow.wrot).T
    v_freestream = jnp.where(geom.lvalbe[None, :], vinf[:, None] + vrot, 0.0)
    v = v_freestream + wcsrd

    rhs = -jnp.sum(geom.enc * v, axis=0)
    enc_d_eff = jnp.einsum("ijn,n->ij", geom.enc_d, flow.delcon)
    rhs = rhs - jnp.sum(enc_d_eff * v, axis=0)

    return jnp.where(geom.lvnc, rhs, 0.0)


def compute_circulation(
    geom: CirculationGeometry,
    flow: FlowCondition,
    refs: ReferenceQuantities,
) -> jnp.ndarray:
    """Solve for horseshoe circulation strengths ``gamma``."""
    rhs = build_rhs(geom, flow, refs)
    return solve_circulation(geom.aicn, rhs)


def compute_circulation_from_lu(
    lu_piv: tuple[jnp.ndarray, jnp.ndarray],
    geom: CirculationGeometry,
    flow: FlowCondition,
    refs: ReferenceQuantities,
) -> jnp.ndarray:
    """Solve for ``gamma`` using a precomputed LU factorization of ``geom.aicn``."""
    rhs = build_rhs(geom, flow, refs)
    return solve_from_lu(lu_piv, rhs)


def compute_velocities(
    geom: CirculationGeometry, gamma: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Recover induced velocities at control and vortex points from ``gamma``."""
    vc = jnp.einsum("kij,j->ki", geom.wc_gam, gamma)
    vv = jnp.einsum("kij,j->ki", geom.wv_gam, gamma)
    return vc, vv
