"""Tests for spanwise lift distribution plotting helpers."""

from __future__ import annotations

import pytest

from openavl import AVLSolver
from openavl.geometry import Aircraft
from openavl.plotting.lift_distribution import collect_lift_distribution

pytestmark = pytest.mark.core


def _build_rect_wing(*, clmax: float = 0.0) -> Aircraft:
    span = 10.0
    chord = 1.0
    aircraft = Aircraft(name="Lift plot", sref=span * chord, cref=chord, bref=span)
    wing = aircraft.add_wing("Wing", n_chord=8, n_span=16, symmetric=True, component=1)
    wing.clmax = clmax
    wing.add_section(xyzle=[0.0, 0.0, 0.0], chord=chord).set_airfoil_naca("0012")
    wing.add_section(xyzle=[0.0, span / 2.0, 0.0], chord=chord).set_airfoil_naca("0012")
    return aircraft


def test_plot_aircraft_on_solver():
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    from openavl.plotting.aircraft3d import plot_aircraft_3d

    solver = AVLSolver(_build_rect_wing(clmax=0.0), alpha=5.0)

    fig, ax = plot_aircraft_3d(solver, show=False)
    n_lines_direct = len(ax.lines)
    fig.clear()

    fig, ax = solver.plot_aircraft(show=False)
    assert len(ax.lines) == n_lines_direct
    fig.clear()

    fig, ax = solver.plot_geom(show=False)
    assert len(ax.lines) == n_lines_direct
    fig.clear()


def test_collect_lift_distribution_requires_solved_state():
    aircraft = _build_rect_wing()
    solver = AVLSolver(aircraft, alpha=5.0)
    with pytest.raises(ValueError, match="solved state"):
        collect_lift_distribution(solver.state, solver.model)


def test_plot_lift_distribution_after_run():
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    from openavl.plotting.lift_distribution import plot_lift_distribution

    solver = AVLSolver(_build_rect_wing(clmax=1.0), alpha=8.0)
    solver.execute_run(max_iter=20)

    series = collect_lift_distribution(solver.state, solver.model, component=1)
    assert len(series) == 2
    assert all(item.y.size == item.cl.size for item in series)
    assert all(item.cl.max() <= 1.0 + 1e-10 for item in series)

    fig, ax = plot_lift_distribution(solver, component=1, show=False)
    assert len(ax.lines) >= 2
    assert ax.get_ylabel() == "Local Cl"
    fig.clear()

    fig, ax = solver.plot_lift_distribution(component=1, show=False)
    assert len(ax.lines) >= 2
    fig.clear()
