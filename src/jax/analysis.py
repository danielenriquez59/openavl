"""Top-level differentiable AVL analysis (Phase 3I)."""

from __future__ import annotations

from functools import partial
from typing import Callable

from openavl.jax.backend import jax
from openavl.jax.forces import (
    _compute_forces_eager,
    _compute_forces_impl,
    _EMPTY_BODY,
    _EMPTY_TREFFTZ,
)
from openavl.jax.setup import (
    compute_circulation,
    compute_circulation_from_lu,
    compute_velocities,
)
from openavl.jax.solve import lu_factor_aicn
from openavl.jax.types import (
    AnalysisGeometry,
    AnalysisResult,
    FlowCondition,
    ForceResult,
    ReferenceQuantities,
    Velocities,
)


def _make_forces_checkpoint(*, iysym: int, include_body: bool) -> Callable[..., ForceResult]:
    """Return a checkpointed force integrator with static flags bound."""
    return jax.checkpoint(
        partial(
            _compute_forces_impl,
            lnfld_wv=False,
            lvisc=False,
            ltrforce=False,
            include_body=include_body,
            include_trefftz=True,
            iysym=iysym,
        ),
    )


def _integrate_forces(
    geom: AnalysisGeometry,
    gamma,
    velocities: Velocities,
    flow: FlowCondition,
    refs: ReferenceQuantities,
    *,
    lvisc: bool,
    lnfld_wv: bool,
    use_checkpoint: bool,
    forces_checkpoint: Callable[..., ForceResult] | None = None,
):
    """Force integration; optional checkpointing for reverse-mode AD."""
    if not use_checkpoint:
        return _compute_forces_eager(
            geom.force, gamma, velocities, flow, refs,
            geom.body, geom.trefftz,
            lnfld_wv=lnfld_wv, lvisc=lvisc,
        )
    body = geom.body if geom.body is not None else _EMPTY_BODY
    tgeom = geom.trefftz if geom.trefftz is not None else _EMPTY_TREFFTZ
    checkpoint = forces_checkpoint or _make_forces_checkpoint(
        iysym=int(refs.iysym),
        include_body=geom.body.nl.shape[0] > 0,
    )
    return checkpoint(
        geom.force, gamma, velocities, flow, refs,
        body, tgeom,
    )


def _run_analysis_impl(
    flow: FlowCondition,
    geom: AnalysisGeometry,
    refs: ReferenceQuantities,
    lu_piv: tuple | None,
    *,
    lvisc: bool,
    lnfld_wv: bool,
    use_checkpoint: bool,
    forces_checkpoint: Callable[..., ForceResult] | None = None,
) -> AnalysisResult:
    """Shared analysis path for eager AD and fixed-geometry JIT runners."""
    if lu_piv is None:
        gamma = compute_circulation(geom.circulation, flow, refs)
    else:
        gamma = compute_circulation_from_lu(lu_piv, geom.circulation, flow, refs)
    _vc, vv = compute_velocities(geom.circulation, gamma)
    velocities = Velocities(vv=vv, wv=_vc)
    forces = _integrate_forces(
        geom, gamma, velocities, flow, refs,
        lvisc=lvisc, lnfld_wv=lnfld_wv, use_checkpoint=use_checkpoint,
        forces_checkpoint=forces_checkpoint,
    )
    return AnalysisResult(CL=forces.CL, CD=forces.CD, CY=forces.CY, CM=forces.CM)


def run_analysis(
    flow: FlowCondition,
    geom: AnalysisGeometry,
    refs: ReferenceQuantities,
    *,
    lvisc: bool = False,
    lnfld_wv: bool = False,
    use_checkpoint: bool = False,
    forces_checkpoint: Callable[..., ForceResult] | None = None,
) -> AnalysisResult:
    """Run the full primal analysis: circulation, velocities, and forces.

    Parameters
    ----------
    flow:
        Differentiable flight condition (angles, rates, controls, Mach).
    geom:
        Bundled circulation, force, body, and Trefftz geometry snapshots.
    refs:
        Reference lengths, moment center, and baseline drag.
    lvisc:
        Include viscous CD(CL) strip drag when True.
    lnfld_wv:
        Use wake-normal velocity field for bound-vortex force integration.

    Returns
    -------
    AnalysisResult
        Total lift, drag, sideforce, and moment coefficients.
    """
    return _run_analysis_impl(
        flow, geom, refs, None,
        lvisc=lvisc, lnfld_wv=lnfld_wv, use_checkpoint=use_checkpoint,
        forces_checkpoint=forces_checkpoint,
    )


def make_run_analysis_jit(
    geom: AnalysisGeometry,
    refs: ReferenceQuantities,
    *,
    lvisc: bool = False,
    lnfld_wv: bool = False,
):
    """Return a JIT-compiled runner with fixed geometry/reference PyTrees."""
    lu_piv = lu_factor_aicn(geom.circulation.aicn)
    return jax.jit(
        lambda flow: _run_analysis_impl(
            flow,
            geom,
            refs,
            lu_piv,
            lvisc=lvisc,
            lnfld_wv=lnfld_wv,
            use_checkpoint=False,
        )
    )


def run_analysis_jit(
    flow: FlowCondition,
    geom: AnalysisGeometry,
    refs: ReferenceQuantities,
    *,
    lvisc: bool = False,
    lnfld_wv: bool = False,
) -> AnalysisResult:
    """JIT-compiled version of :func:`run_analysis` (geometry held constant)."""
    return make_run_analysis_jit(geom, refs, lvisc=lvisc, lnfld_wv=lnfld_wv)(flow)
