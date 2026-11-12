"""The GUI Total Forces panel reports non-dimensional body rates like AVL."""

from __future__ import annotations

import pytest

from openavl import constants as C
from openavl.core.reporting import nasa_dir
from openavl.core.solver import AVLSolver
from openavl.web.session import _build_results_extras

from tests.helpers import GEOMETRIES_DIR

PLANE_AVL = GEOMETRIES_DIR / "plane.avl"

pytestmark = pytest.mark.ui

RATES = {"pb/2v": 0.05, "qc/2v": 0.02, "rb/2v": -0.03}


@pytest.mark.skipif(not PLANE_AVL.is_file(), reason="plane.avl not found")
def test_results_extras_report_nondimensional_rates():
    solver = AVLSolver(PLANE_AVL)
    solver.set_constraint("alpha", "alpha", 0.0)
    for name, value in RATES.items():
        solver.set_constraint(name, name, value)
    solver.execute_run()

    state = solver.state
    # Sanity check: the reference lengths are not 2, so the unscaled
    # internal rates differ from pb/2V and the test can tell them apart.
    assert state.bref != pytest.approx(2.0)
    assert state.cref != pytest.approx(2.0)

    extras = _build_results_extras(solver)
    dir_ = nasa_dir(state)
    # Rate constraints are given in the displayed convention, so at alpha = 0
    # (body and stability axes coincide) the panel shows the requested values.
    assert extras["pb2V"] == pytest.approx(RATES["pb/2v"])
    assert extras["qc2V"] == pytest.approx(RATES["qc/2v"])
    assert extras["rb2V"] == pytest.approx(RATES["rb/2v"])
    # AVL totals print DIR*p*b/2V, q*c/2V, DIR*r*b/2V (aoutput.f OUTTOT);
    # the run case stores the rates without DIR.
    assert extras["pb2V"] == pytest.approx(dir_ * state.parval[C.IPROTX, 0])
    assert extras["qc2V"] == pytest.approx(state.parval[C.IPROTY, 0])
    assert extras["rb2V"] == pytest.approx(dir_ * state.parval[C.IPROTZ, 0])
