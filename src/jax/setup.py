"""Circulation RHS assembly and velocity recovery (JAX port of GUCALC/GAMSUM/VELSUM)."""

from __future__ import annotations

from openavl.jax.aic import rebuild_aicn_from_wc_gam, vvor_jax
from openavl.jax.backend import jnp
from openavl.jax.freestream import vinfab
from openavl.jax.solve import solve_circulation, solve_from_lu
from openavl.jax.types import CirculationGeometry, FlowCondition, ReferenceQuantities


def rebuild_circulation_geometry(
    geom: CirculationGeometry, flow: FlowCondition
) -> CirculationGeometry:
    """Rebuild ``aicn``/``wc_gam``/``wv_gam`` from a live, differentiable Mach (A2).

    ``CirculationGeometry.aicn``/``wc_gam``/``wv_gam`` are otherwise baked-in
    constants captured at whatever Mach the geometry snapshot was built at, so
    ``d(gamma)/d(mach)`` (and everything downstream of it) is silently zero.
    This recomputes ``betm = sqrt(1 - mach**2)`` from ``flow.mach`` and reruns
    the vortex influence-coefficient assembly (``vvor_jax``) with it, so the
    lattice AIC tracks Mach through the traced pipeline at the cost of
    redoing that O(nvor^2) assembly on every call.

    When ``geom.rv1`` is ``None`` (raw lattice geometry not captured, e.g. the
    geometry-design-variable AD path in ``geom_jax.update_geometry``, which
    rebuilds these matrices itself), ``geom`` is returned unchanged.

    When ``flow.mach`` equals the captured ``snapshot_mach`` (concrete scalar
    comparison at trace time), the baked-in matrices are returned unchanged to
    avoid a redundant and numerically noisy O(nvor^2) rebuild.
    """
    if geom.rv1 is None:
        return geom

    snap_mach = geom.snapshot_mach
    if snap_mach is not None:
        try:
            if float(flow.mach) == float(snap_mach):
                return geom
        except (TypeError, ValueError):
            pass

    betm = jnp.sqrt(1.0 - flow.mach * flow.mach)
    wc_gam = vvor_jax(
        betm,
        geom.iysym,
        geom.ysym,
        geom.izsym,
        geom.zsym,
        geom.vrcorec,
        geom.vrcorew,
        geom.rv1,
        geom.rv2,
        geom.lvcomp,
        geom.chordv,
        geom.rc,
        geom.lvcomp,
        False,
    )
    wv_gam = vvor_jax(
        betm,
        geom.iysym,
        geom.ysym,
        geom.izsym,
        geom.zsym,
        geom.vrcorec,
        geom.vrcorew,
        geom.rv1,
        geom.rv2,
        geom.lvcomp,
        geom.chordv,
        geom.rv,
        geom.lvcomp,
        True,
    )
    aicn = rebuild_aicn_from_wc_gam(
        wc_gam, geom.enc, geom.kutta_iv, geom.kutta_j1, geom.kutta_j2, geom.stripoff_iv
    )
    return geom._replace(aicn=aicn, wc_gam=wc_gam, wv_gam=wv_gam)


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
    geom: CirculationGeometry, gamma: jnp.ndarray, flow: FlowCondition
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Recover induced velocities at control and vortex points from ``gamma``.

    Mirrors the NumPy ``velsum`` routine: ``vc``/``vv`` are the gamma-only
    contributions at control points and bound-vortex midpoints, while ``wv``
    additionally includes the body source/doublet contribution
    (``wvsrd_u`` contracted with the unit-flow vector), which is required to
    reproduce ``LNFLD_WV`` bound-vortex force integration on body-carrying
    models.
    """
    vc = jnp.einsum("kij,j->ki", geom.wc_gam, gamma)
    vv = jnp.einsum("kij,j->ki", geom.wv_gam, gamma)
    vinf = vinfab(flow.alfa, flow.beta)
    u = jnp.concatenate((vinf, flow.wrot))
    wvsrd = jnp.einsum("kin,n->ki", geom.wvsrd_u, u)
    wv = vv + wvsrd
    return vc, vv, wv
