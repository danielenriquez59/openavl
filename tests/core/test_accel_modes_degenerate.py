"""Rigid-body outputs stay physical, and unusual models solve without crashing.

* ``get_aero_accel`` must give the same physical result whatever output axes
  are chosen: with NASA axes (x forward, z down) the vectors are those of the
  geometry-axis run with x and z flipped, including the gyroscopic term.
* ``eigenvalues`` re-solves first when something changed, as AVL's MODE
  command does, so a parameter edit cannot leave the modes stale; an untrimmed
  (``max_iter=0``) point stays untrimmed, and an unchanged point is not solved again.
* More than 30 controls, and a model with bodies but no surfaces, both solve.
"""

from __future__ import annotations

import numpy as np
import pytest

from openavl import constants as C
from openavl.solver import AVLSolver

from tests.helpers import GEOMETRIES_DIR

SUPRA_AVL = GEOMETRIES_DIR / "supra.avl"
SUPRA_MASS = GEOMETRIES_DIR.parent / "mass" / "supra.mass"
SUPRAF_AVL = GEOMETRIES_DIR / "supraf.avl"

pytestmark = pytest.mark.core

needs_supra = pytest.mark.skipif(
    not SUPRA_AVL.is_file() or not SUPRA_MASS.is_file(),
    reason="supra geometry or mass file not found",
)


def _accel(lnasa_sa: bool) -> dict:
    solver = AVLSolver(SUPRA_AVL, mass_file=SUPRA_MASS, alpha=5.0, rho=1.225, velocity=12.0)
    s = solver.state
    s.lnasa_sa = lnasa_sa
    s.wrot[:] = (0.02, -0.01, 0.03)  # geometry-axis rates: rolling, pitching, yawing
    s.parval[C.IPIXY, 0] = 0.05 * s.parval[C.IPIXX, 0]  # products that change sign
    s.parval[C.IPIYZ, 0] = 0.04 * s.parval[C.IPIYY, 0]  # between the two axis sets
    solver.execute_run(max_iter=0)
    return solver.get_aero_accel()


@needs_supra
def test_aero_accel_is_the_same_physics_in_either_output_axes():
    geom = _accel(lnasa_sa=False)
    nasa = _accel(lnasa_sa=True)
    flip = np.array([-1.0, 1.0, -1.0])

    for key in ("force_body", "moment_body", "linear_acceleration_body", "rotational_acceleration_body"):
        np.testing.assert_allclose(nasa[key], flip * geom[key], rtol=1e-12, atol=1e-12, err_msg=key)
    np.testing.assert_allclose(nasa["inertia"], np.outer(flip, flip) * geom["inertia"], rtol=1e-12)
    # The gyroscopic term is what the axis mix-up got wrong: make sure it matters here.
    omega = geom["rotational_acceleration_body"]
    no_gyro = np.linalg.solve(geom["inertia"], geom["moment_body"])
    assert not np.allclose(omega, no_gyro, rtol=1e-6)


def _sorted_eigenvalues(solver):
    return np.array(sorted(solver.eigenvalues().eigenvalues, key=lambda z: (z.real, z.imag)))


@needs_supra
@pytest.mark.parametrize(
    ("name", "value"),
    [("xcg", None), ("velocity", None), ("cl", 0.9)],
)
def test_eigenvalues_after_a_parameter_edit_match_a_fresh_solve(name, value):
    """Without the re-solve the roots were off by 1.7-3.4% of the largest one."""
    reused = AVLSolver(SUPRA_AVL, mass_file=SUPRA_MASS)
    reused.setup_trim(1)
    reused.execute_run()
    if name == "xcg":
        value = float(reused.state.parval[C.IPXCG, 0]) + 0.1
    elif name == "velocity":
        value = 1.3 * float(reused.state.parval[C.IPVEE, 0])
    reused.set_parameter(name, value)
    edited = _sorted_eigenvalues(reused)

    fresh = AVLSolver(SUPRA_AVL, mass_file=SUPRA_MASS)
    fresh.setup_trim(1)
    fresh.set_parameter(name, value)
    fresh.execute_run()
    expected = _sorted_eigenvalues(fresh)

    assert expected.size > 0, "fixture should have flight modes"
    np.testing.assert_allclose(edited, expected, rtol=1e-8, atol=1e-10)


@needs_supra
def test_eigenvalues_keep_an_untrimmed_operating_point():
    """After execute_run(max_iter=0) the modes must not re-trim alpha to the CL constraint."""
    solver = AVLSolver(SUPRA_AVL, mass_file=SUPRA_MASS, velocity=12.0)
    solver.set_variable("alpha", 6.0)
    solver.execute_run(max_iter=0)
    solver.set_parameter("xcg", float(solver.state.parval[C.IPXCG, 0]) + 0.1)
    solver.eigenvalues()
    assert np.degrees(solver.state.alfa) == pytest.approx(6.0, abs=1e-12)  # was moved to 2.67


@needs_supra
def test_eigenvalues_do_not_re_solve_an_unchanged_point(monkeypatch):
    solver = AVLSolver(SUPRA_AVL, mass_file=SUPRA_MASS)
    solver.setup_trim(1)
    solver.execute_run()
    calls = []
    monkeypatch.setattr(solver, "execute_run", lambda **kw: calls.append(kw))
    solver.eigenvalues()
    solver.get_system_matrices()
    assert calls == []


def _many_controls_avl(n: int) -> str:
    controls = "".join(f"CONTROL\nc{i} 1.0 0.7 0 0 0 1\n" for i in range(n))
    section = lambda y: f"SECTION\n0 {y} 0 1 0\n{controls}"  # noqa: E731
    return (
        "many controls\n0.0\n0 0 0\n10 1 10\n0 0 0\n"
        "SURFACE\nWing\n8 1.0 12 1.0\nYDUPLICATE\n0.0\n" + section(0) + section(5)
    )


def test_trim_with_more_than_30_controls(tmp_path):
    path = tmp_path / "many.avl"
    path.write_text(_many_controls_avl(31))
    solver = AVLSolver(path)
    assert solver.state.ncontrol == 31
    solver.execute_run()
    assert solver.get_results()["CL"] == pytest.approx(0.6, abs=1e-6)


@pytest.mark.skipif(not SUPRAF_AVL.is_file(), reason="supraf.avl not found")
def test_body_only_model_solves_with_zero_trefftz_loads():
    solver = AVLSolver(SUPRAF_AVL, alpha=3.0)
    assert solver.state.nstrip == 0 and solver.state.nbody > 0
    solver.execute_run(max_iter=0)
    s = solver.state
    assert (s.clff, s.cyff, s.cdff, s.spanef) == (0.0, 0.0, 0.0, 0.0)
    assert np.isfinite(solver.get_results()["CD"])
