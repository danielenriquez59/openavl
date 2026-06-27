"""Tests for pressure-coefficient plotting helpers."""

from __future__ import annotations

import pytest

from openavl import AVLSolver
from openavl.geometry import Aircraft
from openavl.plotting.cp_plot import collect_cp_surfaces

pytestmark = pytest.mark.core


def _build_rect_wing() -> Aircraft:
    span = 10.0
    chord = 1.0
    aircraft = Aircraft(name="Cp plot", sref=span * chord, cref=chord, bref=span)
    wing = aircraft.add_wing("Wing", n_chord=8, n_span=16, symmetric=True, component=1)
    wing.add_section(xyzle=[0.0, 0.0, 0.0], chord=chord).set_airfoil_naca("0012")
    wing.add_section(xyzle=[0.0, span / 2.0, 0.0], chord=chord).set_airfoil_naca("0012")
    return aircraft


def test_collect_cp_surfaces_requires_solved_state():
    aircraft = _build_rect_wing()
    solver = AVLSolver(aircraft, alpha=5.0)
    with pytest.raises(ValueError, match="solved state"):
        collect_cp_surfaces(solver.state, solver.model)


def test_plot_cp_after_run():
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    from openavl.plotting.cp_plot import plot_cp

    solver = AVLSolver(_build_rect_wing(), alpha=8.0)
    solver.execute_run(max_iter=20)

    surfaces = collect_cp_surfaces(solver.state, solver.model, component=1, mode="surface")
    assert len(surfaces) == 2
    for item in surfaces:
        assert item.xyz.ndim == 3
        assert item.xyz.shape[0] == item.cp.shape[0] + 1
        assert item.xyz.shape[1] == item.cp.shape[1] + 1
        assert item.xyz.shape[2] == 3

    fig, ax = plot_cp(solver, component=1, show=False)
    assert len(ax.collections) >= 2
    fig.clear()


def test_plot_cp_delta_mode():
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    from openavl.plotting.cp_plot import plot_cp

    solver = AVLSolver(_build_rect_wing(), alpha=8.0)
    solver.execute_run(max_iter=20)
    fig, ax = plot_cp(solver, component=1, mode="delta", show=False)
    assert len(ax.collections) >= 2
    fig.clear()


def test_solver_plot_cp_delegates():
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")

    solver = AVLSolver(_build_rect_wing(), alpha=8.0)
    solver.execute_run(max_iter=20)

    fig, ax = solver.plot_cp(component=1, show=False)
    assert len(ax.collections) >= 2
    fig.clear()
