"""AVLSolver rejects inputs it would otherwise silently mishandle.

Constructor options must apply with or without a mass file, unknown options
must raise, and a run case that imposes one constraint on two variables must
fail with a clear error before the Newton solve, as in AVL.
"""

from __future__ import annotations

import shutil
import warnings

import pytest

from openavl import constants as C
from openavl.solver import AVLSolver

from tests.helpers import GEOMETRIES_DIR

PLANE_AVL = GEOMETRIES_DIR / "plane.avl"
SUPRA_AVL = GEOMETRIES_DIR / "supra.avl"
SUPRA_MASS = GEOMETRIES_DIR.parent / "mass" / "supra.mass"

pytestmark = pytest.mark.core


# ---- Constructor options -------------------------------------------------


def test_parameter_options_apply_without_mass_file():
    solver = AVLSolver(PLANE_AVL, mach=0.3, velocity=50.0, mass=2.5)
    s = solver.state
    assert s.mach == pytest.approx(0.3)
    assert s.parval[C.IPMACH, 0] == pytest.approx(0.3)
    assert s.parval[C.IPVEE, 0] == pytest.approx(50.0)
    assert s.parval[C.IPMASS, 0] == pytest.approx(2.5)


def test_parameter_options_still_override_mass_file():
    solver = AVLSolver(SUPRA_AVL, mass_file=SUPRA_MASS, mach=0.2, xcg=3.6)
    s = solver.state
    assert s.mach == pytest.approx(0.2)
    assert s.parval[C.IPXCG, 0] == pytest.approx(3.6)


def test_mach_option_changes_the_solution():
    """The option reaches the solve: compressibility raises the lift slope."""
    incompressible = AVLSolver(PLANE_AVL, alpha=4.0)
    compressible = AVLSolver(PLANE_AVL, alpha=4.0, mach=0.6)
    for solver in (incompressible, compressible):
        solver.set_constraint("alpha", "alpha", 4.0)
        solver.execute_run()
    assert compressible.get_results()["CL"] > incompressible.get_results()["CL"] * 1.05


def test_unknown_option_raises_type_error():
    with pytest.raises(TypeError, match="Unknown solver option.*mahc"):
        AVLSolver(PLANE_AVL, mahc=0.3)


def test_out_of_range_option_value_raises():
    with pytest.raises(ValueError, match="Mach number must satisfy"):
        AVLSolver(PLANE_AVL, mach=1.2)


def test_base_dir_is_used_for_geometry_files(tmp_path):
    """Airfoil files resolve against base_dir when the .avl lives elsewhere."""
    shutil.copy(SUPRA_AVL, tmp_path / "supra.avl")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        AVLSolver(tmp_path / "supra.avl", base_dir=GEOMETRIES_DIR)
    assert not [w for w in caught if "Airfoil file not found" in str(w.message)]


# ---- Duplicate constraints -----------------------------------------------


def test_constraint_imposed_twice_raises_before_solving():
    solver = AVLSolver(PLANE_AVL)
    solver.set_constraint("alpha", "cl", 0.5)
    solver.set_constraint("elevator", "cl", 0.5)
    with pytest.raises(ValueError, match=r"more than once: CL is imposed by alpha, elevator"):
        solver.execute_run()


def test_reassigning_constraints_one_at_a_time_is_allowed():
    """A duplicate may exist between calls; only the run case being solved is checked."""
    solver = AVLSolver(PLANE_AVL)
    solver.set_constraint("elevator", "cl", 0.5)  # alpha is also on CL by default
    solver.set_constraint("alpha", "cm", 0.0)     # now each constraint is used once
    solver.execute_run()
    assert solver.get_results()["CL"] == pytest.approx(0.5, abs=1e-4)
