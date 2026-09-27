"""Tests for per-wing sectional CLmax capping (OpenAVL extension)."""

from __future__ import annotations

import numpy as np
import pytest

from openavl import AVLSolver
from openavl.geometry import Aircraft

pytestmark = pytest.mark.core


def _build_rect_wing(*, clmax: float = 0.0) -> Aircraft:
    """Simple half-span rectangular wing for CLmax tests."""
    span = 10.0
    chord = 1.0
    aircraft = Aircraft(name="CLmax Test", sref=span * chord, cref=chord, bref=span)
    wing = aircraft.add_wing("Wing", n_chord=8, n_span=16, s_space=1.0)
    wing.clmax = clmax
    root = wing.add_section(xyzle=[0.0, 0.0, 0.0], chord=chord)
    root.set_airfoil_naca("0012")
    tip = wing.add_section(xyzle=[0.0, span / 2.0, 0.0], chord=chord)
    tip.set_airfoil_naca("0012")
    return aircraft


def _run_at_alpha(aircraft: Aircraft, alpha_deg: float):
    """Fixed-alpha force evaluation (alpha constrained to itself, no CL trim)."""
    solver = AVLSolver(aircraft, alpha=alpha_deg, beta=0.0, mach=0.0)
    solver.set_constraint("alpha", "alpha", alpha_deg)
    solver.execute_run(max_iter=5)
    return solver.state, solver.get_results()


def _wing_strip_cl(state, isurf: int = 0) -> np.ndarray:
    j0 = int(state.jfrst[isurf])
    nj = int(state.nj[isurf])
    return state.cl_lstrp[j0 : j0 + nj].copy()


def test_clmax_caps_sectional_lift():
    """Sectional CLs must not exceed clmax; integrated lift is reduced."""
    clmax = 1.0
    alpha_deg = 15.0

    uncapped_state, uncapped = _run_at_alpha(_build_rect_wing(clmax=0.0), alpha_deg)
    state_capped, capped = _run_at_alpha(_build_rect_wing(clmax=clmax), alpha_deg)

    strip_cl_uncapped = _wing_strip_cl(uncapped_state)
    strip_cl_capped = _wing_strip_cl(state_capped)

    assert np.any(strip_cl_uncapped > clmax + 1e-6), "test setup: alpha should exceed CLmax"
    assert np.all(strip_cl_capped <= clmax + 1e-10)
    assert capped["CL"] < uncapped["CL"]


def test_clmax_disabled_by_default():
    """clmax=0.0 leaves sectional CLs unchanged."""
    alpha_deg = 15.0
    state, _ = _run_at_alpha(_build_rect_wing(clmax=0.0), alpha_deg)
    assert state.clmax_surf[0] == pytest.approx(0.0)
    assert np.any(_wing_strip_cl(state) > 1.0)


def test_clmax_symmetric_mirror_uses_same_limit():
    """Y-duplicated halves share the parent wing clmax; other wings stay uncapped."""
    span = 10.0
    chord = 1.0
    alpha_deg = 15.0

    aircraft = Aircraft(name="Two-wing", sref=2 * span * chord, cref=chord, bref=2 * span)
    inner = aircraft.add_wing("Inner", n_chord=8, n_span=8, symmetric=True)
    inner.clmax = 1.0
    inner.add_section(xyzle=[0.0, 0.0, 0.0], chord=chord).set_airfoil_naca("0012")
    inner.add_section(xyzle=[0.0, span / 2.0, 0.0], chord=chord).set_airfoil_naca("0012")

    outer = aircraft.add_wing("Outer", n_chord=8, n_span=8, symmetric=True)
    outer.add_section(xyzle=[0.0, span / 2.0, 0.0], chord=chord).set_airfoil_naca("0012")
    outer.add_section(xyzle=[0.0, span, 0.0], chord=chord).set_airfoil_naca("0012")

    state, _ = _run_at_alpha(aircraft, alpha_deg)

    assert state.clmax_surf[0] == pytest.approx(1.0)
    assert state.clmax_surf[1] == pytest.approx(1.0)
    assert state.clmax_surf[2] == pytest.approx(0.0)
    assert state.clmax_surf[3] == pytest.approx(0.0)

    for isurf in (0, 1):
        assert np.all(_wing_strip_cl(state, isurf) <= 1.0 + 1e-10)


@pytest.mark.parametrize("clmax", [0.0, 1.0])
def test_clmax_alpha_derivatives_match_finite_difference(clmax):
    """Differentiate actual capped loads, including the local lift direction."""
    alpha = 20.0
    step = 1.0e-5  # radians; derivative arrays use radians
    state, _ = _run_at_alpha(_build_rect_wing(clmax=clmax), alpha)
    plus, _ = _run_at_alpha(_build_rect_wing(clmax=clmax), alpha + np.rad2deg(step))
    minus, _ = _run_at_alpha(_build_rect_wing(clmax=clmax), alpha - np.rad2deg(step))
    nstrip = state.nstrip
    if clmax:
        # Keep finite differences on one smooth branch of the clipping law.
        assert np.any(state.cl_lstrp[:nstrip] == clmax)
        np.testing.assert_array_equal(
            plus.cl_lstrp[:nstrip] == clmax,
            minus.cl_lstrp[:nstrip] == clmax,
        )
    for load, deriv in (("clstrp", "clst"), ("cdstrp", "cdst"), ("cystrp", "cyst")):
        analytic = (
            getattr(state, deriv + "_u")[:nstrip, :3] @ state.vinf_a
            + getattr(state, deriv + "_a")[:nstrip]
        )
        finite_difference = (
            getattr(plus, load)[:nstrip] - getattr(minus, load)[:nstrip]
        ) / (2.0 * step)
        np.testing.assert_allclose(analytic, finite_difference, rtol=2e-6, atol=2e-8)
    for load, deriv in (("cfstrp", "cfst"), ("cmstrp", "cmst")):
        analytic = getattr(state, deriv + "_u")[:, :nstrip, :3] @ state.vinf_a
        finite_difference = (
            getattr(plus, load)[:, :nstrip] - getattr(minus, load)[:, :nstrip]
        ) / (2.0 * step)
        np.testing.assert_allclose(analytic, finite_difference, rtol=2e-6, atol=2e-8)
    for load in ("cltot", "cdtot"):
        analytic = getattr(state, load + "_u")[:3] @ state.vinf_a + getattr(state, load + "_a")
        finite_difference = (getattr(plus, load) - getattr(minus, load)) / (2.0 * step)
        assert analytic == pytest.approx(finite_difference, rel=2e-6, abs=2e-8)


@pytest.mark.parametrize("clmax", [0.0, 1.0])
def test_clmax_control_derivatives_match_finite_difference(clmax):
    """Capped control force and moment derivatives follow the load response."""
    def run(deflection):
        aircraft = _build_rect_wing(clmax=clmax)
        for section in aircraft.wings[0].sections:
            section.add_control("flap", gain=1.0, xhinge=0.75)
        solver = AVLSolver(aircraft, alpha=20.0)
        solver.set_variable("flap", deflection)
        solver.execute_run(max_iter=0)
        return solver.state

    step = 1e-3  # control derivatives use degrees
    state = run(0.0)
    plus = run(step)
    minus = run(-step)
    nstrip = state.nstrip
    if clmax:
        assert np.any(state.cl_lstrp[:nstrip] == clmax)
        np.testing.assert_array_equal(
            plus.cl_lstrp[:nstrip] == clmax,
            minus.cl_lstrp[:nstrip] == clmax,
        )
    for load, deriv in (("clstrp", "clst"), ("cdstrp", "cdst"), ("cnc", "cnc")):
        finite_difference = (
            getattr(plus, load)[:nstrip] - getattr(minus, load)[:nstrip]
        ) / (2.0 * step)
        np.testing.assert_allclose(
            getattr(state, deriv + "_d")[:nstrip, 0], finite_difference,
            rtol=2e-6, atol=2e-8,
        )
    for load, deriv in (("cfstrp", "cfst"), ("cmstrp", "cmst")):
        finite_difference = (
            getattr(plus, load)[:, :nstrip] - getattr(minus, load)[:, :nstrip]
        ) / (2.0 * step)
        np.testing.assert_allclose(
            getattr(state, deriv + "_d")[:, :nstrip, 0], finite_difference,
            rtol=2e-6, atol=2e-8,
        )
