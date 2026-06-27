"""Derivative validation: hand-coded, JAX AD, and finite differences (Phase 4A)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tests.jax_backend.require_jax import require_jax

jax = require_jax()
import jax.numpy as jnp

from openavl.analysis.deriv import StabilityDerivatives, compute_stability_derivatives
from openavl.jax.analysis import run_analysis
from openavl.jax.snapshot import snapshot_analysis_geometry, snapshot_flow, snapshot_refs
from openavl.jax.types import FlowCondition
from openavl.solver import AVLSolver

from tests.helpers import GEOMETRIES_DIR

PLANE_AVL = GEOMETRIES_DIR / "plane.avl"
B737_AVL = GEOMETRIES_DIR / "b737.avl"
SUPRA_AVL = GEOMETRIES_DIR / "supra.avl"

JAX_TOL = 1e-6
JAX_TOL_BODY = 5e-5
FD_TOL = 1e-5
FD_STEP = 1e-7

GEOMETRY_CASES = [
    pytest.param(PLANE_AVL, {"alpha": 5.0, "beta": 0.0}, id="plane"),
    pytest.param(B737_AVL, {"alpha": 3.0, "beta": 0.0}, id="b737"),
    pytest.param(SUPRA_AVL, {"alpha": 2.0, "beta": 0.0}, id="supra"),
]


pytestmark = pytest.mark.integration


def _build_solver(avl_path: Path, alpha: float, beta: float) -> AVLSolver:
    if not avl_path.is_file():
        pytest.skip(f"{avl_path.name} not found: {avl_path}")
    solver = AVLSolver(avl_path)
    solver.set_variable("alpha", alpha)
    solver.set_variable("beta", beta)
    solver.execute_run(max_iter=1)
    return solver


def _stability_moments(flow: FlowCondition, result_cm: jnp.ndarray, lnasa_sa: bool) -> jnp.ndarray:
    """Transform body-axis CM to stability-axis roll/pitch/yaw moments."""
    dir_ = -1.0 if lnasa_sa else 1.0
    ca = jnp.cos(flow.alfa)
    sa = jnp.sin(flow.alfa)
    cl_roll = dir_ * (result_cm[0] * ca + result_cm[2] * sa)
    cm_pitch = result_cm[1]
    cn_yaw = dir_ * (result_cm[2] * ca - result_cm[0] * sa)
    return jnp.array([cl_roll, cm_pitch, cn_yaw])


def _perturb_flow(flow: FlowCondition, field: str, index: int | None, delta: float) -> FlowCondition:
    """Return a copy of ``flow`` with one component perturbed."""
    if field == "alfa":
        return flow._replace(alfa=flow.alfa + delta)
    if field == "beta":
        return flow._replace(beta=flow.beta + delta)
    if field == "wrot":
        wrot = flow.wrot.at[index].add(delta)
        return flow._replace(wrot=wrot)
    if field == "delcon":
        delcon = flow.delcon.at[index].add(delta)
        return flow._replace(delcon=delcon)
    if field == "mach":
        return flow._replace(mach=flow.mach + delta)
    raise KeyError(field)


def _central_difference(
    geom,
    refs,
    flow: FlowCondition,
    lnasa_sa: bool,
    output: str,
    field: str,
    index: int | None = None,
    eps: float = FD_STEP,
) -> float:
    """Central finite-difference derivative of a scalar output w.r.t. one flow input."""

    def scalar(f: FlowCondition) -> float:
        result = run_analysis(f, geom, refs)
        if output == "CL":
            return float(result.CL)
        if output == "CD":
            return float(result.CD)
        if output == "CY":
            return float(result.CY)
        if output in ("Cl", "Cm", "Cn"):
            moments = _stability_moments(f, result.CM, lnasa_sa)
            idx = {"Cl": 0, "Cm": 1, "Cn": 2}[output]
            return float(moments[idx])
        raise KeyError(output)

    fp = _perturb_flow(flow, field, index, eps)
    fm = _perturb_flow(flow, field, index, -eps)
    return (scalar(fp) - scalar(fm)) / (2.0 * eps)


def _jax_partial(jac, output: str, field: str, index: int | None = None, *, state=None) -> float:
    """Extract one JAX Jacobian entry from ``jacrev(run_analysis)``."""
    out = getattr(jac, output)
    if field == "alfa":
        return float(out.alfa)
    if field == "beta":
        return float(out.beta)
    if field == "wrot":
        val = float(out.wrot[index])
        if state is not None and index == 1:
            val *= 2.0 / float(state.cref)
        return val
    if field == "delcon":
        return float(out.delcon[index])
    if field == "mach":
        return float(out.mach)
    raise KeyError(field)


def _scale_rate_derivative(value: float, field: str, index: int | None, state) -> float:
    """Apply AVL ``q`` scaling for pitch-rate finite-difference checks."""
    if field == "wrot" and index == 1:
        return value * (2.0 / float(state.cref))
    return value


def _hand_coded_value(derivs: StabilityDerivatives, name: str) -> float:
    return float(getattr(derivs, name))


def _compare_derivative_set(solver: AVLSolver, *, label: str) -> None:
    """Three-way compare for stability-axis force/moment derivatives."""
    state = solver.state
    geom = snapshot_analysis_geometry(state)
    flow = snapshot_flow(state)
    refs = snapshot_refs(state)
    derivs = compute_stability_derivatives(state)
    jac = jax.jacrev(run_analysis)(flow, geom, refs)

    # Full FD triple-check only on the smallest model (plane). For larger models,
    # validate JAX vs hand-coded only. wrot-p/r has no hand reference, so we
    # keep a two-case spot-check for CL_p and CL_r on all geometries.
    _full_fd = (label == "plane")
    _spot_wrot_pr = {("CL", "wrot", 0), ("CL", "wrot", 2)}

    dir_ = -1.0 if state.lnasa_sa else 1.0
    ca = float(np.cos(state.alfa))
    sa = float(np.sin(state.alfa))

    def stability_jac_wrt(field: str, index: int | None = None) -> tuple[float, float, float]:
        cm_jac = jac.CM
        if field == "alfa":
            body = np.array([float(cm_jac.alfa[0]), float(cm_jac.alfa[1]), float(cm_jac.alfa[2])])
        elif field == "beta":
            body = np.array([float(cm_jac.beta[0]), float(cm_jac.beta[1]), float(cm_jac.beta[2])])
        elif field == "wrot":
            body = np.array(
                [
                    float(cm_jac.wrot[0][index]),
                    float(cm_jac.wrot[1][index]),
                    float(cm_jac.wrot[2][index]),
                ]
            )
        elif field == "delcon":
            body = np.array(
                [
                    float(cm_jac.delcon[index][0]),
                    float(cm_jac.delcon[index][1]),
                    float(cm_jac.delcon[index][2]),
                ]
            )
        else:
            raise KeyError(field)
        cl = dir_ * (body[0] * ca + body[2] * sa)
        cm = body[1]
        cn = dir_ * (body[2] * ca - body[0] * sa)
        return cl, cm, cn

    scalar_cases = [
        ("CL", "CL_a", "alfa", None),
        ("CL", "CL_b", "beta", None),
        ("CL", "CL_p", "wrot", 0),
        ("CL", "CL_q", "wrot", 1),
        ("CL", "CL_r", "wrot", 2),
        ("CD", "CD_a", "alfa", None),
        ("CD", "CD_b", "beta", None),
        ("CD", "CD_p", "wrot", 0),
        ("CD", "CD_q", "wrot", 1),
        ("CD", "CD_r", "wrot", 2),
        ("CY", "CY_a", "alfa", None),
        ("CY", "CY_b", "beta", None),
        ("CY", "CY_p", "wrot", 0),
        ("CY", "CY_q", "wrot", 1),
        ("CY", "CY_r", "wrot", 2),
    ]
    moment_cases = [
        ("Cl", "Cl_a", "alfa", None),
        ("Cl", "Cl_b", "beta", None),
        ("Cl", "Cl_p", "wrot", 0),
        ("Cl", "Cl_q", "wrot", 1),
        ("Cl", "Cl_r", "wrot", 2),
        ("Cm", "Cm_a", "alfa", None),
        ("Cm", "Cm_b", "beta", None),
        ("Cm", "Cm_p", "wrot", 0),
        ("Cm", "Cm_q", "wrot", 1),
        ("Cm", "Cm_r", "wrot", 2),
        ("Cn", "Cn_a", "alfa", None),
        ("Cn", "Cn_b", "beta", None),
        ("Cn", "Cn_p", "wrot", 0),
        ("Cn", "Cn_q", "wrot", 1),
        ("Cn", "Cn_r", "wrot", 2),
    ]

    for out, ref_name, field, index in scalar_cases:
        hand = _hand_coded_value(derivs, ref_name)
        jax_val = _jax_partial(jac, out, field, index, state=state)
        tol = JAX_TOL_BODY if label == "supra" else JAX_TOL
        if field == "wrot" and index in (0, 2):
            if _full_fd or (out, field, index) in _spot_wrot_pr:
                fd_val = _scale_rate_derivative(
                    _central_difference(geom, refs, flow, state.lnasa_sa, out, field, index),
                    field,
                    index,
                    state,
                )
                assert jax_val == pytest.approx(fd_val, abs=FD_TOL, rel=FD_TOL), (
                    f"{label} {ref_name}: JAX {jax_val} vs FD {fd_val}"
                )
            continue
        fd_tol = JAX_TOL_BODY if label == "supra" else FD_TOL
        assert jax_val == pytest.approx(hand, abs=tol, rel=tol), (
            f"{label} {ref_name}: JAX {jax_val} vs hand {hand}"
        )
        if _full_fd:
            fd_val = _scale_rate_derivative(
                _central_difference(geom, refs, flow, state.lnasa_sa, out, field, index),
                field,
                index,
                state,
            )
            assert fd_val == pytest.approx(hand, abs=fd_tol, rel=fd_tol), (
                f"{label} {ref_name}: FD {fd_val} vs hand {hand}"
            )

    moment_map = {"Cl": 0, "Cm": 1, "Cn": 2}
    q_scale = 2.0 / float(state.cref)
    for out, ref_name, field, index in moment_cases:
        hand = _hand_coded_value(derivs, ref_name)
        jax_cl, jax_cm, jax_cn = stability_jac_wrt(field, index)
        jax_val = (jax_cl, jax_cm, jax_cn)[moment_map[out]]
        if field == "wrot" and index == 1:
            jax_val *= q_scale
        tol = JAX_TOL_BODY if label == "supra" else JAX_TOL
        if field == "wrot" and index in (0, 2):
            if _full_fd:
                fd_val = _scale_rate_derivative(
                    _central_difference(geom, refs, flow, state.lnasa_sa, out, field, index),
                    field,
                    index,
                    state,
                )
                assert jax_val == pytest.approx(fd_val, abs=FD_TOL, rel=FD_TOL)
            continue
        fd_tol = JAX_TOL_BODY if label == "supra" else FD_TOL
        assert jax_val == pytest.approx(hand, abs=tol, rel=tol), (
            f"{label} {ref_name}: JAX {jax_val} vs hand {hand}"
        )
        if _full_fd:
            fd_val = _scale_rate_derivative(
                _central_difference(geom, refs, flow, state.lnasa_sa, out, field, index),
                field,
                index,
                state,
            )
            assert fd_val == pytest.approx(hand, abs=fd_tol, rel=fd_tol), (
                f"{label} {ref_name}: FD {fd_val} vs hand {hand}"
            )

    rtd = 180.0 / np.pi
    for n, name in enumerate(state.control_names[: state.ncontrol]):
        for out, deriv_dict in [
            ("CL", derivs.CL_d),
            ("CD", derivs.CD_d),
            ("CY", derivs.CY_d),
        ]:
            hand = float(deriv_dict[name])
            jax_val = _jax_partial(jac, out, "delcon", n) * rtd
            assert jax_val == pytest.approx(hand, abs=JAX_TOL, rel=JAX_TOL)
            if _full_fd or n == 0:
                fd_val = _central_difference(geom, refs, flow, state.lnasa_sa, out, "delcon", n) * rtd
                assert fd_val == pytest.approx(hand, abs=FD_TOL, rel=FD_TOL)


@pytest.mark.parametrize("avl_path, kwargs", GEOMETRY_CASES)
@pytest.mark.reference
def test_derivatives_three_way(avl_path: Path, kwargs: dict) -> None:
    """Hand-coded, JAX, and FD derivatives agree on multiple geometries."""
    solver = _build_solver(avl_path, kwargs["alpha"], kwargs["beta"])
    _compare_derivative_set(solver, label=avl_path.stem)
