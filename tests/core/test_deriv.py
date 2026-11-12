"""Tests for stability derivative extraction.

``plane_fortran_derivatives.json`` holds every derivative Fortran AVL 3.52
(optvl 2.5.0) prints for plane.avl at alpha 5, beta 4, Mach 0.3, elevator 2,
zero rates: 30 stability-axis, 36 body-axis and all control derivatives in
both axes, stored in OpenAVL's names and units (per rad). Regenerate with
optvl: set the four constraints above and the three rate constraints to 0,
``execute_run()``, then read ``get_stab_derivs()``,
``get_control_stab_derivs()`` (per degree there) and
``get_body_axis_derivs()``. Rates stay zero because with rates AVL's alpha
derivatives are wrong (see ``test_turn_rates.py``).
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from openavl.analysis.deriv import compute_neutral_point
from openavl.solver import AVLSolver

from tests.helpers import FIXTURES_DIR, GEOMETRIES_DIR

PLANE_AVL = GEOMETRIES_DIR / "plane.avl"
FIXTURE_PATH = FIXTURES_DIR / "plane_fortran_derivatives.json"
TOL = 1e-6

pytestmark = pytest.mark.core


@pytest.fixture(scope="module")
def plane() -> AVLSolver:
    """plane.avl solved once at the fixture's flight condition."""
    conditions = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))["meta"]["conditions"]
    solver = AVLSolver(PLANE_AVL, mach=conditions["mach"])
    for name in ("alpha", "beta", "elevator"):
        solver.set_constraint(name, name, conditions[name])
    solver.execute_run(max_iter=20)
    assert solver.state.lsol
    return solver


def test_compute_neutral_point_matches_avl_formula():
    """Neutral point and static margin follow AVL aoutput.f."""
    xcg, cref, cl_a, cm_a = 0.25, 1.0, 5.0, -1.0
    xnp, sm = compute_neutral_point(xcg, cref, cl_a, cm_a)
    assert sm == pytest.approx(0.2)
    assert xnp == pytest.approx(0.45)

    xnp_nan, sm_nan = compute_neutral_point(xcg, cref, 0.0, cm_a)
    assert math.isnan(xnp_nan)
    assert math.isnan(sm_nan)


@pytest.mark.reference
def test_every_derivative_matches_fortran(plane: AVLSolver):
    """All stability, body-axis and control derivatives match AVL 3.52."""
    refs = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    derivs = plane.get_stability_derivatives()
    for name, ref in refs["stability"].items():
        assert getattr(derivs, name) == pytest.approx(ref, rel=TOL, abs=TOL), name
    for name, per_control in refs["stability_controls"].items():
        assert getattr(derivs, name) == pytest.approx(per_control, rel=TOL, abs=TOL), name

    body = plane.get_body_axis_derivatives()
    values = np.asarray(body.values)
    names = list(plane.state.control_names)
    rows = {**{r: i for i, r in enumerate(body.rows[:6])},
            **{n: body.rows.index(f"d{k + 1}") for k, n in enumerate(names)}}
    for group in ("body", "body_controls"):
        for key, ref in refs[group].items():
            coef, var = key.split("_", 1)
            ours = float(values[rows[var], body.cols.index(coef)])
            assert ours == pytest.approx(ref, rel=TOL, abs=TOL), key

    assert derivs.sm == pytest.approx(-derivs.Cm_a / derivs.CL_a)
    assert derivs.xnp == pytest.approx(
        float(plane.state.xyzref[0]) + derivs.sm * float(plane.state.cref)
    )


def test_control_derivative_tables_match_stability_derivatives(plane: AVLSolver):
    """The control tables are keyed by control name and carry the same values."""
    derivs = plane.get_stability_derivatives()
    names = list(plane.state.control_names)
    assert list(derivs.CL_d) == names

    stab = plane.get_control_derivatives(axis="stability")
    assert stab.cols == ["CL", "CD", "CY", "Cl", "Cm", "Cn"]
    assert stab.rows == names
    for i, name in enumerate(names):
        for j, coef in enumerate(stab.cols):
            assert stab.values[i][j] == pytest.approx(getattr(derivs, f"{coef}_d")[name])

    control_rows = plane.get_control_derivatives(axis="body")
    body = plane.get_body_axis_derivatives()
    assert control_rows.cols == body.cols == ["CX", "CY", "CZ", "Cl", "Cm", "Cn"]
    assert body.rows == ["u", "v", "w", "p", "q", "r"] + [f"d{k + 1}" for k in range(len(names))]
    np.testing.assert_allclose(control_rows.values, np.asarray(body.values)[6:])

    with pytest.raises(ValueError, match="axis must be"):
        plane.get_control_derivatives(axis="wind")  # type: ignore[arg-type]
