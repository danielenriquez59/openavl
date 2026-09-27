"""Top-level differentiable AVL analysis (Phase 3I)."""

from __future__ import annotations

from functools import partial
from typing import Callable

from openavl.jax.backend import jax
from openavl.jax.body import rebuild_body
from openavl.jax.forces import (
    _compute_forces_eager,
    _compute_forces_impl,
    _EMPTY_BODY,
    _EMPTY_TREFFTZ,
)
from openavl.jax.setup import (
    compute_circulation,
    compute_velocities,
    rebuild_circulation_geometry,
)
from openavl.jax.types import (
    AnalysisGeometry,
    AnalysisResult,
    FlowCondition,
    ForceResult,
    ReferenceQuantities,
    Velocities,
)


def _make_forces_checkpoint(
    *, iysym: int, include_body: bool, lvisc: bool = False, lnfld_wv: bool = False
) -> Callable[..., ForceResult]:
    """Return a checkpointed force integrator with static flags bound.

    The caller's ``lvisc``/``lnfld_wv`` flags are bound into the checkpoint
    (they are static Python bools, so ``jax.checkpoint`` handles them like
    any other static argument). The returned callable also carries these
    flags as attributes so ``_integrate_forces`` can verify a pre-built
    checkpoint matches the flags requested for a given run.
    """
    checkpoint = jax.checkpoint(
        partial(
            _compute_forces_impl,
            lnfld_wv=lnfld_wv,
            lvisc=lvisc,
            ltrforce=False,
            include_body=include_body,
            include_trefftz=True,
            iysym=iysym,
        ),
    )
    checkpoint.lvisc = lvisc
    checkpoint.lnfld_wv = lnfld_wv
    return checkpoint


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
    tgeom = geom.trefftz if geom.trefftz is not None else _EMPTY_TREFFTZ
    # A2: the Trefftz-plane Prandtl-Glauert scaling (`pgmat_jax`) must track the
    # live, differentiable Mach rather than the value frozen into the geometry
    # snapshot at build time, so override it here regardless of source.
    tgeom = tgeom._replace(amach=flow.mach)
    if not use_checkpoint:
        return _compute_forces_eager(
            geom.force, gamma, velocities, flow, refs,
            geom.body, tgeom,
            lnfld_wv=lnfld_wv, lvisc=lvisc,
        )
    body = geom.body if geom.body is not None else _EMPTY_BODY
    if forces_checkpoint is not None:
        bound_lvisc = getattr(forces_checkpoint, "lvisc", None)
        bound_lnfld_wv = getattr(forces_checkpoint, "lnfld_wv", None)
        if bound_lvisc is not None:
            assert bound_lvisc == lvisc and bound_lnfld_wv == lnfld_wv, (
                f"forces_checkpoint was built with lvisc={bound_lvisc}, "
                f"lnfld_wv={bound_lnfld_wv}, but the caller requested "
                f"lvisc={lvisc}, lnfld_wv={lnfld_wv}"
            )
        checkpoint = forces_checkpoint
    else:
        checkpoint = _make_forces_checkpoint(
            iysym=int(refs.iysym),
            include_body=body.nl.shape[0] > 0,
            lvisc=lvisc,
            lnfld_wv=lnfld_wv,
        )
    return checkpoint(
        geom.force, gamma, velocities, flow, refs,
        body, tgeom,
    )


def _run_analysis_impl(
    flow: FlowCondition,
    geom: AnalysisGeometry,
    refs: ReferenceQuantities,
    *,
    lvisc: bool,
    lnfld_wv: bool,
    use_checkpoint: bool,
    forces_checkpoint: Callable[..., ForceResult] | None = None,
) -> AnalysisResult:
    """Shared analysis path for eager AD and fixed-geometry JIT runners."""
    geom = rebuild_body(geom, flow, refs)
    circ = rebuild_circulation_geometry(geom.circulation, flow)
    gamma = compute_circulation(circ, flow, refs)
    _vc, vv, wv = compute_velocities(circ, gamma, flow)
    velocities = Velocities(vv=vv, wv=wv)
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
        flow, geom, refs,
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
    """JIT runner with fixed lattice coordinates and live Mach dependence."""
    return jax.jit(
        lambda flow: _run_analysis_impl(
            flow,
            geom,
            refs,
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
