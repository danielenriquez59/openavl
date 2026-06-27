"""Tests for CPOML surface Cp post-processing."""

from __future__ import annotations

import numpy as np
import pytest

from openavl import AVLSolver
from openavl.aero.cpoml import collect_cpoml_surfaces, cpthk
from openavl.geometry import Aircraft
from openavl.plotting.cp_plot import collect_cp_surfaces

pytestmark = pytest.mark.core


def _build_rect_wing(n_chord: int = 8, n_span: int = 16) -> Aircraft:
    span = 10.0
    chord = 1.0
    aircraft = Aircraft(name="Cp OML", sref=span * chord, cref=chord, bref=span)
    wing = aircraft.add_wing(
        "Wing",
        n_chord=n_chord,
        n_span=n_span,
        symmetric=True,
        component=1,
    )
    wing.add_section(xyzle=[0.0, 0.0, 0.0], chord=chord).set_airfoil_naca("0012")
    wing.add_section(xyzle=[0.0, span / 2.0, 0.0], chord=chord).set_airfoil_naca("0012")
    return aircraft


def test_collect_cpoml_requires_solved_state():
    solver = AVLSolver(_build_rect_wing(), alpha=5.0)
    with pytest.raises(ValueError, match="solved state"):
        collect_cpoml_surfaces(solver.state, solver.model)


def test_cpoml_surface_shapes_and_negative_upper_cp():
    solver = AVLSolver(_build_rect_wing(), alpha=8.0)
    solver.execute_run(max_iter=20)

    surfaces = collect_cp_surfaces(solver.state, solver.model, component=1, mode="surface")
    assert len(surfaces) == 2
    n_chord = 8
    for item in surfaces:
        assert item.xyz.shape == (item.cp.shape[0] + 1, 2 * n_chord + 1, 3)
        assert item.cp.shape[1] == 2 * n_chord
        assert float(np.min(item.cp)) < 0.0

    cpthk(solver.state)
    assert np.any(solver.state.cpt[: solver.state.nvor] != 0.0)


def test_delta_mode_preserves_lattice_dcp():
    solver = AVLSolver(_build_rect_wing(), alpha=8.0)
    solver.execute_run(max_iter=20)

    surfaces = collect_cp_surfaces(solver.state, solver.model, component=1, mode="delta")
    assert len(surfaces) == 2
    for item in surfaces:
        assert item.xyz.shape[0] == item.cp.shape[0] + 1
        assert item.xyz.shape[1] == item.cp.shape[1] + 1
        i0 = int(solver.state.ijfrst[int(solver.state.jfrst[0])])
        nvc = int(solver.state.nvstrp[int(solver.state.jfrst[0])])
        np.testing.assert_allclose(
            item.cp[0, :],
            solver.state.dcp[i0 : i0 + nvc],
        )


def test_solver_get_cp_data_surface_mode():
    solver = AVLSolver(_build_rect_wing(), alpha=8.0)
    solver.execute_run(max_iter=20)
    data = solver.get_cp_data(component=1, mode="surface")
    assert len(data) == 2
    assert data[0]["cp"].shape[1] == 16
