"""Stability derivative extraction from raw AVL sensitivity arrays."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from openavl.core.state import AVLState

ControlAxis = Literal["stability", "body"]


@dataclass
class StabilityDerivatives:
    """Classical stability-axis force, moment, and control derivatives.

    Perturbation derivatives with respect to ``alpha``, ``beta``, body rates,
    and control deflections are returned in per-radian units (for example
    ``CL_a`` is dCL/dα and ``Cm_d["elevator"]`` is dCm/dδ_elevator).
    """

    CL_a: float = 0.0
    CL_b: float = 0.0
    CL_p: float = 0.0
    CL_q: float = 0.0
    CL_r: float = 0.0
    CD_a: float = 0.0
    CD_b: float = 0.0
    CD_p: float = 0.0
    CD_q: float = 0.0
    CD_r: float = 0.0
    CY_a: float = 0.0
    CY_b: float = 0.0
    CY_p: float = 0.0
    CY_q: float = 0.0
    CY_r: float = 0.0
    Cl_a: float = 0.0
    Cl_b: float = 0.0
    Cl_p: float = 0.0
    Cl_q: float = 0.0
    Cl_r: float = 0.0
    Cm_a: float = 0.0
    Cm_b: float = 0.0
    Cm_p: float = 0.0
    Cm_q: float = 0.0
    Cm_r: float = 0.0
    Cn_a: float = 0.0
    Cn_b: float = 0.0
    Cn_p: float = 0.0
    Cn_q: float = 0.0
    Cn_r: float = 0.0
    CL_d: dict[str, float] = field(default_factory=dict)
    CD_d: dict[str, float] = field(default_factory=dict)
    CY_d: dict[str, float] = field(default_factory=dict)
    Cl_d: dict[str, float] = field(default_factory=dict)
    Cm_d: dict[str, float] = field(default_factory=dict)
    Cn_d: dict[str, float] = field(default_factory=dict)


def _chain_alpha(
    arr6: np.ndarray,
    vinf_a: np.ndarray,
    wrot_a: np.ndarray,
    extra: float = 0.0,
) -> float:
    """Chain-rule derivative with respect to angle of attack."""
    return float(
        arr6[0] * vinf_a[0]
        + arr6[1] * vinf_a[1]
        + arr6[2] * vinf_a[2]
        + arr6[3] * wrot_a[0]
        + arr6[4] * wrot_a[1]
        + arr6[5] * wrot_a[2]
        + extra
    )


def _chain_beta(arr6: np.ndarray, vinf_b: np.ndarray) -> float:
    """Chain-rule derivative with respect to sideslip angle."""
    return float(
        arr6[0] * vinf_b[0]
        + arr6[1] * vinf_b[1]
        + arr6[2] * vinf_b[2]
    )


def _chain_rx(arr6: np.ndarray, wrot_rx: np.ndarray) -> float:
    """Roll-rate contribution through stability-axis rate transform."""
    return float(arr6[3] * wrot_rx[0] + arr6[5] * wrot_rx[2])


def _chain_ry(arr6: np.ndarray) -> float:
    """Pitch-rate contribution."""
    return float(arr6[4])


def _chain_rz(arr6: np.ndarray, wrot_rz: np.ndarray) -> float:
    """Yaw-rate contribution through stability-axis rate transform."""
    return float(arr6[5] * wrot_rz[2] + arr6[3] * wrot_rz[0])


def _control_deriv_per_rad(value: float, state: AVLState) -> float:
    """Convert a raw control derivative from per-degree to per-radian."""
    return float(value) / float(state.dtr)


def compute_stability_derivatives(state: AVLState) -> StabilityDerivatives:
    """Extract classical stability derivatives from raw ``_u`` and ``_d`` arrays.

    Angle and control derivatives are returned per radian. Rate derivatives
    ``*_p``, ``*_q``, and ``*_r`` are non-dimensional (pb/2V, qc/2V, rb/2V).
    """
    dir_ = -1.0 if state.lnasa_sa else 1.0
    ca = float(np.cos(state.alfa))
    sa = float(np.sin(state.alfa))
    w0 = float(state.wrot[0])
    w2 = float(state.wrot[2])

    rx = (w0 * ca + w2 * sa) * dir_
    rz = (w2 * ca - w0 * sa) * dir_
    wrot_rx = np.array([ca * dir_, 0.0, sa * dir_], dtype=np.float64)
    wrot_rz = np.array([-sa * dir_, 0.0, ca * dir_], dtype=np.float64)
    wrot_a = np.array([-rx * sa - rz * ca, 0.0, -rz * sa + rx * ca], dtype=np.float64)

    cl_u = state.cltot_u
    cy_u = state.cytot_u
    cd_u = state.cdtot_u
    cm_u = state.cmtot_u

    crsax_u = cm_u[0, :] * ca + cm_u[2, :] * sa
    cmsax_u = cm_u[1, :].copy()
    cnsax_u = cm_u[2, :] * ca - cm_u[0, :] * sa
    crsax_a = -float(state.cmtot[0]) * sa + float(state.cmtot[2]) * ca
    cnsax_a = -float(state.cmtot[2]) * sa - float(state.cmtot[0]) * ca

    bref = float(state.bref)
    cref = float(state.cref)
    p_scale = 2.0 / bref
    q_scale = 2.0 / cref
    r_scale = 2.0 / bref

    cl_al = _chain_alpha(cl_u, state.vinf_a, wrot_a, float(state.cltot_a))
    cy_al = _chain_alpha(cy_u, state.vinf_a, wrot_a)
    cd_al = _chain_alpha(cd_u, state.vinf_a, wrot_a, float(state.cdtot_a))
    cr_al = _chain_alpha(crsax_u, state.vinf_a, wrot_a, crsax_a)
    cm_al = _chain_alpha(cmsax_u, state.vinf_a, wrot_a)
    cn_al = _chain_alpha(cnsax_u, state.vinf_a, wrot_a, cnsax_a)

    derivs = StabilityDerivatives(
        CL_a=cl_al,
        CL_b=_chain_beta(cl_u, state.vinf_b),
        CL_p=_chain_rx(cl_u, wrot_rx) * p_scale,
        CL_q=_chain_ry(cl_u) * q_scale,
        CL_r=_chain_rz(cl_u, wrot_rz) * r_scale,
        CY_a=cy_al,
        CY_b=_chain_beta(cy_u, state.vinf_b),
        CY_p=_chain_rx(cy_u, wrot_rx) * p_scale,
        CY_q=_chain_ry(cy_u) * q_scale,
        CY_r=_chain_rz(cy_u, wrot_rz) * r_scale,
        CD_a=cd_al,
        CD_b=_chain_beta(cd_u, state.vinf_b),
        CD_p=_chain_rx(cd_u, wrot_rx) * p_scale,
        CD_q=_chain_ry(cd_u) * q_scale,
        CD_r=_chain_rz(cd_u, wrot_rz) * r_scale,
        Cl_a=dir_ * cr_al,
        Cl_b=dir_ * _chain_beta(crsax_u, state.vinf_b),
        Cl_p=dir_ * _chain_rx(crsax_u, wrot_rx) * p_scale,
        Cl_q=dir_ * _chain_ry(crsax_u) * q_scale,
        Cl_r=dir_ * _chain_rz(crsax_u, wrot_rz) * r_scale,
        Cm_a=cm_al,
        Cm_b=_chain_beta(cmsax_u, state.vinf_b),
        Cm_p=_chain_rx(cmsax_u, wrot_rx) * p_scale,
        Cm_q=_chain_ry(cmsax_u) * q_scale,
        Cm_r=_chain_rz(cmsax_u, wrot_rz) * r_scale,
        Cn_a=dir_ * cn_al,
        Cn_b=dir_ * _chain_beta(cnsax_u, state.vinf_b),
        Cn_p=dir_ * _chain_rx(cnsax_u, wrot_rx) * p_scale,
        Cn_q=dir_ * _chain_ry(cnsax_u) * q_scale,
        Cn_r=dir_ * _chain_rz(cnsax_u, wrot_rz) * r_scale,
    )

    for n in range(state.ncontrol):
        name = state.control_names[n] if n < len(state.control_names) else f"d{n + 1}"
        crs_d = float(state.cmtot_d[0, n] * ca + state.cmtot_d[2, n] * sa)
        cms_d = float(state.cmtot_d[1, n])
        cns_d = float(state.cmtot_d[2, n] * ca - state.cmtot_d[0, n] * sa)
        derivs.CL_d[name] = _control_deriv_per_rad(state.cltot_d[n], state)
        derivs.CY_d[name] = _control_deriv_per_rad(state.cytot_d[n], state)
        derivs.CD_d[name] = _control_deriv_per_rad(state.cdtot_d[n], state)
        derivs.Cl_d[name] = _control_deriv_per_rad(dir_ * crs_d, state)
        derivs.Cm_d[name] = _control_deriv_per_rad(cms_d, state)
        derivs.Cn_d[name] = _control_deriv_per_rad(dir_ * cns_d, state)

    return derivs


@dataclass
class BodyAxisDerivatives:
    """Geometry-axis force and moment derivatives (AVL ``DERMATB`` output).

    Rows ``u``–``r`` follow AVL's normalized velocity and rate perturbations.
    Control rows are per radian of deflection.
    """

    rows: list[str] = field(default_factory=list)
    cols: list[str] = field(default_factory=lambda: ["CX", "CY", "CZ", "Cl", "Cm", "Cn"])
    values: list[list[float]] = field(default_factory=list)


def compute_body_axis_derivatives(state: AVLState) -> BodyAxisDerivatives:
    """Extract body-axis derivative matrix from raw ``_u`` and ``_d`` arrays.

    Control-surface rows are converted from AVL's per-degree sensitivities to
    per-radian units. Velocity and rate rows are unchanged from ``DERMATB``.
    """
    dir_ = -1.0 if state.lnasa_sa else 1.0
    bref = float(state.bref)
    cref = float(state.cref)
    p_scale = (2.0 / bref) if abs(bref) > 1e-12 else 0.0
    q_scale = (2.0 / cref) if abs(cref) > 1e-12 else 0.0
    r_scale = p_scale

    cx_u = state.cftot_u[0, :]
    cy_u = state.cftot_u[1, :]
    cz_u = state.cftot_u[2, :]
    cl_u = state.cmtot_u[0, :]
    cm_u = state.cmtot_u[1, :]
    cn_u = state.cmtot_u[2, :]

    rows: list[str] = ["u", "v", "w", "p", "q", "r"]
    values: list[list[float]] = [
        [
            -float(cx_u[0]),
            -dir_ * float(cy_u[0]),
            -float(cz_u[0]),
            -float(cl_u[0]),
            -dir_ * float(cm_u[0]),
            -float(cn_u[0]),
        ],
        [
            -dir_ * float(cx_u[1]),
            -float(cy_u[1]),
            -dir_ * float(cz_u[1]),
            -dir_ * float(cl_u[1]),
            -float(cm_u[1]),
            -dir_ * float(cn_u[1]),
        ],
        [
            -float(cx_u[2]),
            -dir_ * float(cy_u[2]),
            -float(cz_u[2]),
            -float(cl_u[2]),
            -dir_ * float(cm_u[2]),
            -float(cn_u[2]),
        ],
        [
            float(cx_u[3]) * p_scale,
            dir_ * float(cy_u[3]) * p_scale,
            float(cz_u[3]) * p_scale,
            float(cl_u[3]) * p_scale,
            dir_ * float(cm_u[3]) * p_scale,
            float(cn_u[3]) * p_scale,
        ],
        [
            dir_ * float(cx_u[4]) * q_scale,
            float(cy_u[4]) * q_scale,
            dir_ * float(cz_u[4]) * q_scale,
            dir_ * float(cl_u[4]) * q_scale,
            float(cm_u[4]) * q_scale,
            dir_ * float(cn_u[4]) * q_scale,
        ],
        [
            float(cx_u[5]) * r_scale,
            dir_ * float(cy_u[5]) * r_scale,
            float(cz_u[5]) * r_scale,
            float(cl_u[5]) * r_scale,
            dir_ * float(cm_u[5]) * r_scale,
            float(cn_u[5]) * r_scale,
        ],
    ]

    ncontrol = int(state.ncontrol)
    for n in range(ncontrol):
        rows.append(f"d{n + 1}")
        values.append(
            [
                _control_deriv_per_rad(dir_ * float(state.cftot_d[0, n]), state),
                _control_deriv_per_rad(float(state.cftot_d[1, n]), state),
                _control_deriv_per_rad(dir_ * float(state.cftot_d[2, n]), state),
                _control_deriv_per_rad(dir_ * float(state.cmtot_d[0, n]), state),
                _control_deriv_per_rad(float(state.cmtot_d[1, n]), state),
                _control_deriv_per_rad(dir_ * float(state.cmtot_d[2, n]), state),
            ]
        )

    return BodyAxisDerivatives(rows=rows, values=values)


@dataclass
class ControlDerivatives:
    """Control-surface force and moment derivatives in one axis system.

    Rows are control names. Columns are stability-axis
    ``CL, CD, CY, Cl, Cm, Cn`` or body-axis ``CX, CY, CZ, Cl, Cm, Cn``.
    All values are per radian of control deflection.
    """

    axis: ControlAxis = "stability"
    rows: list[str] = field(default_factory=list)
    cols: list[str] = field(default_factory=list)
    values: list[list[float]] = field(default_factory=list)


def compute_control_derivatives(
    state: AVLState,
    axis: ControlAxis = "stability",
) -> ControlDerivatives:
    """Extract control-surface derivatives in body or stability axes.

    Parameters
    ----------
    state:
        Solver state with populated ``*_d`` sensitivity arrays from the latest
        run.
    axis:
        ``"stability"`` for ``CL, CD, CY, Cl, Cm, Cn`` (default), or
        ``"body"`` for ``CX, CY, CZ, Cl, Cm, Cn``.

    Returns
    -------
    ControlDerivatives
        Matrix with one row per control surface. Values are per radian of
        deflection. Stability-axis roll/yaw moments use the same
        body-to-stability transform as :func:`compute_stability_derivatives`;
        body-axis rows match the control block of
        :func:`compute_body_axis_derivatives`.
    """
    if axis == "stability":
        stab = compute_stability_derivatives(state)
        cols = ["CL", "CD", "CY", "Cl", "Cm", "Cn"]
        rows = list(stab.CL_d.keys())
        values = [
            [
                stab.CL_d[name],
                stab.CD_d[name],
                stab.CY_d[name],
                stab.Cl_d[name],
                stab.Cm_d[name],
                stab.Cn_d[name],
            ]
            for name in rows
        ]
        return ControlDerivatives(axis="stability", rows=rows, cols=cols, values=values)

    if axis == "body":
        body = compute_body_axis_derivatives(state)
        cols = list(body.cols)
        ncontrol = int(state.ncontrol)
        rows = [
            state.control_names[n] if n < len(state.control_names) else f"d{n + 1}"
            for n in range(ncontrol)
        ]
        values = [list(row) for row in body.values[6 : 6 + ncontrol]]
        return ControlDerivatives(axis="body", rows=rows, cols=cols, values=values)

    raise ValueError(f"axis must be 'stability' or 'body', got {axis!r}")
