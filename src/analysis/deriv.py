"""Stability derivative extraction from raw AVL sensitivity arrays."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import partial
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

    ``xnp`` is the neutral-point x-location in geometry length units (same as
    ``xcg`` / ``XYZREF``), matching AVL ``aoutput.f``. ``sm`` is the static
    margin as a fraction of ``cref``: ``sm = (xnp - xcg) / cref = -Cm_a / CL_a``.
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
    xnp: float = math.nan
    sm: float = math.nan
    CL_d: dict[str, float] = field(default_factory=dict)
    CD_d: dict[str, float] = field(default_factory=dict)
    CY_d: dict[str, float] = field(default_factory=dict)
    Cl_d: dict[str, float] = field(default_factory=dict)
    Cm_d: dict[str, float] = field(default_factory=dict)
    Cn_d: dict[str, float] = field(default_factory=dict)


def compute_neutral_point(
    xcg: float,
    cref: float,
    cl_a: float,
    cm_a: float,
) -> tuple[float, float]:
    """Compute neutral point and static margin from longitudinal derivatives.

    Matches AVL ``aoutput.f``::

        Xnp = Xcg - Cref * Cm_a / CL_a
        SM  = (Xnp - Xcg) / Cref = -Cm_a / CL_a

    Parameters
    ----------
    xcg:
        Moment-reference / CG x-location in geometry length units.
    cref:
        Reference chord in geometry length units.
    cl_a:
        Lift-curve slope ``dCL/dα`` (per radian).
    cm_a:
        Pitch-stiffness derivative ``dCm/dα`` (per radian), taken about ``xcg``.

    Returns
    -------
    tuple[float, float]
        ``(xnp, sm)`` where ``xnp`` is in geometry length units and ``sm`` is
        a fraction of ``cref``. Both are ``nan`` when ``cl_a`` is zero.
    """
    if cl_a == 0.0:
        return math.nan, math.nan
    sm = -float(cm_a) / float(cl_a)
    xnp = float(xcg) + sm * float(cref)
    return xnp, sm


_STAB_COEFS = ("CL", "CY", "CD", "Cl", "Cm", "Cn")


def _stability_rows(cl, cy, cd, cm, ca: float, sa: float, dir_: float) -> np.ndarray:
    """Stack CL, CY, CD, Cl', Cm, Cn' from geometry-axis moments ``cm`` (3, ...).

    Cl' and Cn' are the roll and yaw moments about the stability axes: the body
    x and z axes turned by alpha about y, with AVL's DIR sign.
    """
    return np.array([
        cl, cy, cd,
        dir_ * (cm[0] * ca + cm[2] * sa),
        cm[1],
        dir_ * (cm[2] * ca - cm[0] * sa),
    ], dtype=np.float64)


def compute_stability_derivatives(state: AVLState) -> StabilityDerivatives:
    """Extract classical stability derivatives from raw ``_u`` and ``_d`` arrays.

    Angle and control derivatives are returned per radian. Rate derivatives
    ``*_p``, ``*_q``, and ``*_r`` are non-dimensional (pb/2V, qc/2V, rb/2V),
    with the roll and yaw rates p', r' about the stability axes.

    The math (AVL ``aoutput.f`` DERMATS). The solver gives each coefficient's
    sensitivity ``U`` to the six unknowns u = (Vinf_x, Vinf_y, Vinf_z, wrot_x,
    wrot_y, wrot_z), all in geometry axes. A stability derivative is the chain
    rule ``dC/dx = U[:, :3] @ dVinf/dx + U[:, 3:] @ dwrot/dx``:

    * alpha: ``dVinf/dalpha`` is ``vinf_a``. The stability-axis rates are held,
      so the body rates turn with alpha, wrot_x = DIR (p' cos a - r' sin a) and
      wrot_z = DIR (p' sin a + r' cos a), giving
      ``dwrot/dalpha = (-wrot_z, 0, wrot_x)``. The moments' own axes turn too,
      which adds ``dCl'/dalpha = Cn'`` and ``dCn'/dalpha = -Cl'``, plus the
      direct ``CL_a``/``CD_a`` terms from the force routine.
    * beta: ``dVinf/dbeta`` is ``vinf_b``; the rates do not change.
    * p', r': ``dwrot/dp' = DIR (cos a, 0, sin a)`` and
      ``dwrot/dr' = DIR (-sin a, 0, cos a)``; q: ``dwrot/dq = (0, 1, 0)``.
      Times 2/bref (2/cref for q), since wrot = 2/bref * pb/2V.

    ``dwrot/dalpha`` deliberately departs from AVL, whose formula carries an
    extra DIR and so flips sign under NASA axes, contradicting its own source
    comments ("= -WROT(3)", "= WROT(1)"). That made CL_a and Cm_a in a turn
    depend on the output axes (0.8% / 1% on Supra at 30 deg bank).
    """
    dir_ = -1.0 if state.lnasa_sa else 1.0
    ca, sa = math.cos(state.alfa), math.sin(state.alfa)
    wx, _, wz = (float(w) for w in state.wrot)
    rows = partial(_stability_rows, ca=ca, sa=sa, dir_=dir_)

    sens = rows(state.cltot_u, state.cytot_u, state.cdtot_u, state.cmtot_u)[:, :6]
    totals = rows(state.cltot, state.cytot, state.cdtot, state.cmtot)
    axes_turn = np.array([state.cltot_a, 0.0, state.cdtot_a, totals[5], 0.0, -totals[3]])
    dvel, drot = sens[:, :3], sens[:, 3:]
    per_unknown = {
        "a": dvel @ state.vinf_a + drot @ np.array([-wz, 0.0, wx]) + axes_turn,
        "b": dvel @ state.vinf_b,
        "p": drot @ np.array([dir_ * ca, 0.0, dir_ * sa]) * (2.0 / state.bref),
        "q": drot[:, 1] * (2.0 / state.cref),
        "r": drot @ np.array([-dir_ * sa, 0.0, dir_ * ca]) * (2.0 / state.bref),
    }
    derivs = StabilityDerivatives(**{
        f"{coef}_{x}": float(column[i])
        for x, column in per_unknown.items()
        for i, coef in enumerate(_STAB_COEFS)
    })

    # Control sensitivities are per degree of deflection.
    ncontrol = int(state.ncontrol)
    controls = rows(state.cltot_d, state.cytot_d, state.cdtot_d, state.cmtot_d)[:, :ncontrol] / state.dtr
    for n in range(ncontrol):
        name = state.control_names[n] if n < len(state.control_names) else f"d{n + 1}"
        for i, coef in enumerate(_STAB_COEFS):
            getattr(derivs, f"{coef}_d")[name] = float(controls[i, n])

    derivs.xnp, derivs.sm = compute_neutral_point(
        xcg=float(state.xyzref[0]),
        cref=float(state.cref),
        cl_a=derivs.CL_a,
        cm_a=derivs.Cm_a,
    )
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

    The math (AVL ``aoutput.f`` DERMATB). Entry [row, col] is
    d(coefficient col)/d(perturbation row), built from the geometry-axis
    sensitivities ``U`` (CX..Cn by Vinf_x..wrot_z) as

        value = (k_row * f_row) * f_col * U[col, row]

    * ``f = (DIR, 1, DIR)`` for both x/y/z triples flips x and z from geometry
      axes to the output axes, on the coefficient and on the perturbation.
    * ``k = -1`` for u, v, w: a body moving at (u, v, w) sees the freestream
      ``Vinf = -(u, v, w) / V``.
    * ``k = 2/bref, 2/cref, 2/bref`` for p, q, r, since wrot = 2/bref * pb/2V.

    Control rows are ``f_col * U_d[col, n]``, converted from AVL's per-degree
    sensitivities to per radian.
    """
    dir_ = -1.0 if state.lnasa_sa else 1.0
    bref = float(state.bref)
    cref = float(state.cref)
    p_scale = (2.0 / bref) if abs(bref) > 1e-12 else 0.0
    q_scale = (2.0 / cref) if abs(cref) > 1e-12 else 0.0

    flip = np.array([dir_, 1.0, dir_, dir_, 1.0, dir_])
    k = np.array([-1.0, -1.0, -1.0, p_scale, q_scale, p_scale])
    sens = np.vstack((state.cftot_u, state.cmtot_u))[:, :6]  # (coefficient, unknown)
    values = np.outer(k * flip, flip) * sens.T

    ncontrol = int(state.ncontrol)
    sens_d = np.vstack((state.cftot_d, state.cmtot_d))[:, :ncontrol]
    controls = (flip[:, np.newaxis] * sens_d / state.dtr).T

    rows = ["u", "v", "w", "p", "q", "r"] + [f"d{n + 1}" for n in range(ncontrol)]
    return BodyAxisDerivatives(rows=rows, values=np.vstack((values, controls)).tolist())


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
