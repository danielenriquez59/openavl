"""Tests for per-wing sectional CLmax capping (OpenAVL extension)."""

from __future__ import annotations

import numpy as np
import pytest

from openavl import AVLSolver
from openavl.geometry import Aircraft

pytestmark = pytest.mark.core


def _build_rect_wing(*, clmax: float = 0.0, dihedral: bool = False) -> Aircraft:
    """Simple half-span rectangular wing for CLmax tests."""
    span = 10.0
    chord = 1.0
    aircraft = Aircraft(name="CLmax Test", sref=span * chord, cref=chord, bref=span)
    wing = aircraft.add_wing("Wing", n_chord=8, n_span=16, s_space=1.0)
    wing.clmax = clmax
    root = wing.add_section(xyzle=[0.0, 0.0, 0.0], chord=chord)
    root.set_airfoil_naca("0012")
    tip_z = 2.0 if dihedral else 0.0
    tip = wing.add_section(xyzle=[0.0, span / 2.0, tip_z], chord=chord)
    tip.set_airfoil_naca("0012")
    return aircraft


def _run_at_alpha(aircraft: Aircraft, alpha_deg: float):
    solver = AVLSolver(aircraft, alpha=alpha_deg, beta=0.0, mach=0.0)
    solver.execute_run(max_iter=30)
    return solver.state, solver.get_results()


def _wing_strip_cl(state, isurf: int = 0) -> np.ndarray:
    j0 = int(state.jfrst[isurf])
    nj = int(state.nj[isurf])
    return state.cl_lstrp[j0 : j0 + nj].copy()


def test_clmax_caps_sectional_lift():
    """Sectional CLs must not exceed clmax; integrated lift is reduced."""
    clmax = 1.0
    alpha_deg = 12.0

    uncapped_state, uncapped = _run_at_alpha(_build_rect_wing(clmax=0.0), alpha_deg)
    state_capped, capped = _run_at_alpha(_build_rect_wing(clmax=clmax), alpha_deg)

    strip_cl_uncapped = _wing_strip_cl(uncapped_state)
    strip_cl_capped = _wing_strip_cl(state_capped)

    assert np.any(strip_cl_uncapped > clmax + 1e-6), "test setup: alpha should exceed CLmax"
    assert np.all(strip_cl_capped <= clmax + 1e-10)
    assert capped["CL"] < uncapped["CL"]


def test_clmax_disabled_by_default():
    """clmax=0.0 leaves sectional CLs unchanged."""
    alpha_deg = 12.0
    state, _ = _run_at_alpha(_build_rect_wing(clmax=0.0), alpha_deg)
    assert state.clmax_surf[0] == pytest.approx(0.0)
    assert np.any(_wing_strip_cl(state) > 1.0)


def test_clmax_dihedral_wing():
    """CLmax capping works on a wing with dihedral."""
    clmax = 1.0
    alpha_deg = 12.0

    uncapped_state, uncapped = _run_at_alpha(_build_rect_wing(clmax=0.0, dihedral=True), alpha_deg)
    capped_state, capped = _run_at_alpha(_build_rect_wing(clmax=clmax, dihedral=True), alpha_deg)

    strip_uncapped = _wing_strip_cl(uncapped_state)
    strip_capped = _wing_strip_cl(capped_state)

    assert np.any(strip_uncapped > clmax + 1e-6)
    assert np.all(strip_capped <= clmax + 1e-10)
    assert capped["CL"] < uncapped["CL"]


def test_clmax_symmetric_mirror_uses_same_limit():
    """Y-duplicated halves share the parent wing clmax; other wings stay uncapped."""
    span = 10.0
    chord = 1.0
    alpha_deg = 12.0

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
