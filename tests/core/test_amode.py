"""Tests for eigenvalue analysis."""

from __future__ import annotations

import math
import re
import subprocess
from pathlib import Path

import numpy as np
import pytest

from openavl import constants as C
from openavl.amode import build_appmat, build_sysmat, identify_modes, runchk, solve_eigenvalues
from openavl.analysis.amode import compute_eigenmode_metrics
from openavl.solver import AVLSolver

from tests.helpers import GEOMETRIES_DIR, REF_DIR

PLANE_AVL = GEOMETRIES_DIR / "plane.avl"
REF_BIN = REF_DIR / "amode_ref"
TOL = 1e-3

pytestmark = pytest.mark.core


def _make_synthetic_state():
    """Build a synthetic state matching the Fortran amode reference case."""
    if not PLANE_AVL.is_file():
        pytest.skip("plane.avl not found")

    solver = AVLSolver(PLANE_AVL)
    s = solver.state
    ir = 0

    s.vinf[:] = (0.8, -0.1, 0.2)
    s.wrot[:] = (0.01, -0.02, 0.03)
    s.cftot[:] = (0.4, -0.1, 0.2)
    s.cmtot[:] = (0.05, -0.02, 0.04)

    for iu in range(6):
        s.cftot_u[0, iu] = 0.01 * (iu + 1)
        s.cftot_u[1, iu] = -0.02 * (iu + 1)
        s.cftot_u[2, iu] = 0.03 * (iu + 1)
        s.cmtot_u[0, iu] = 0.005 * (iu + 1)
        s.cmtot_u[1, iu] = -0.006 * (iu + 1)
        s.cmtot_u[2, iu] = 0.007 * (iu + 1)

    s.cftot_d[:, 0] = (0.11, -0.12, 0.13)
    s.cmtot_d[:, 0] = (0.021, -0.022, 0.023)

    s.amass[:] = (
        (0.1, 0.01, -0.02),
        (0.01, 0.2, 0.03),
        (-0.02, 0.03, 0.15),
    )
    s.ainer[:] = (
        (0.02, 0.004, -0.003),
        (0.004, 0.03, 0.002),
        (-0.003, 0.002, 0.025),
    )

    s.parval[C.IPGEE, ir] = 9.81
    s.parval[C.IPRHO, ir] = 1.225
    s.parval[C.IPVEE, ir] = 30.0
    s.parval[C.IPPHI, ir] = 5.0
    s.parval[C.IPTHE, ir] = -2.0
    s.parval[C.IPPSI, ir] = 1.0
    s.parval[C.IPXCG, ir] = 0.1
    s.parval[C.IPYCG, ir] = -0.2
    s.parval[C.IPZCG, ir] = 0.3
    s.parval[C.IPMASS, ir] = 120.0
    s.parval[C.IPIXX, ir] = 12.0
    s.parval[C.IPIYY, ir] = 15.0
    s.parval[C.IPIZZ, ir] = 20.0
    s.parval[C.IPIXY, ir] = 0.5
    s.parval[C.IPIYZ, ir] = -0.4
    s.parval[C.IPIZX, ir] = 0.3
    s.parval[C.IPCLU, ir] = 0.02
    s.parval[C.IPCMU, ir] = -0.01
    s.parval[C.IPCLA, ir] = 0.03
    s.parval[C.IPCMA, ir] = -0.015

    for iv in range(s.nvtot):
        s.icon[iv, ir] = iv + 1

    return s


def _parse_ref_sections(text: str) -> dict[str, list[float]]:
    sections: dict[str, list[float]] = {}
    current = None
    num_re = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eEdD][-+]?\d+)?")
    for line in text.splitlines():
        if line.startswith("BEGIN "):
            current = line[6:].strip()
            sections[current] = []
        elif current:
            sections[current].extend(float(v.replace("d", "e").replace("D", "e")) for v in num_re.findall(line))
    return sections


def test_sysmat_construction():
    """12x12 system matrix matches Fortran reference when available."""
    state = _make_synthetic_state()
    asys, bsys, rsys = build_sysmat(state)

    assert asys.shape == (C.JETOT, C.JETOT)
    assert bsys.shape[0] == C.JETOT
    assert rsys.shape == (C.JETOT,)
    assert np.any(asys)

    ref_path = REF_BIN if REF_BIN.is_file() else REF_BIN.parent / "amode_ref.exe"
    if not ref_path.is_file():
        pytest.skip("Fortran amode reference binary not found")

    try:
        proc = subprocess.run([str(ref_path)], capture_output=True, text=True, check=False)
    except OSError as exc:
        pytest.skip(f"Cannot execute reference binary: {exc}")
    if proc.returncode != 0:
        pytest.skip(proc.stderr or "amode_ref failed")

    sections = _parse_ref_sections(proc.stdout)
    nums = sections.get("SYSMAT", [])
    nsys = int(nums[0])
    row_len = nsys + state.ncontrol
    actual = [nsys]
    for i in range(nsys):
        row = nums[1 + i * (row_len + 1) : 1 + (i + 1) * (row_len + 1)]
        actual.extend(row[: nsys + state.ncontrol])

    expected = [C.JETOT]
    for i in range(C.JETOT):
        expected.extend(asys[i, :].tolist())
        expected.extend(bsys[i, : state.ncontrol].tolist())

    assert actual[0] == expected[0]
    assert actual[1:] == pytest.approx(expected[1:], abs=TOL)


def test_runchk_detects_duplicate_constraints():
    """RUNCHK rejects run cases with redundant constraints."""
    state = _make_synthetic_state()
    assert runchk(state, 0)
    state.icon[1, 0] = state.icon[0, 0]
    assert not runchk(state, 0)


@pytest.mark.skipif(not PLANE_AVL.is_file(), reason="plane.avl not found")
def test_eigenvalues_physical():
    """Trimmed plane case yields finite eigenvalues with negative real parts."""
    solver = AVLSolver(PLANE_AVL)
    solver.set_parameter("mass", 1.0)
    solver.set_parameter("velocity", 1.0)
    solver.set_parameter("density", 1.0)
    solver.set_parameter("gravity", 1.0)
    solver.setup_trim(mode=1)
    solver.execute_run(max_iter=20)

    result = solver.eigenvalues(use_approx=True)
    assert len(result.eigenvalues) >= 4
    stable = 0
    for value in result.eigenvalues:
        assert np.isfinite(value.real)
        assert np.isfinite(value.imag)
        if value.real <= 1e-3:
            stable += 1
    assert stable >= len(result.eigenvalues) // 2


@pytest.mark.skipif(not PLANE_AVL.is_file(), reason="plane.avl not found")
def test_mode_identification():
    """Eigenanalysis identifies classical dynamic mode names."""
    solver = AVLSolver(PLANE_AVL)
    solver.set_parameter("mass", 1.0)
    solver.set_parameter("velocity", 1.0)
    solver.set_parameter("density", 1.0)
    solver.set_parameter("gravity", 1.0)
    solver.setup_trim(mode=1)
    solver.execute_run(max_iter=20)

    result = solver.eigenvalues(use_approx=True)
    names = {mode.name.split()[0] for mode in result.modes}
    assert "short" in names or "phugoid" in names or "longitudinal" in names
    assert "Dutch" in names or "roll" in names or "spiral" in names or "lateral" in names


def test_compute_eigenmode_metrics():
    """Derived eigenmode scalars match closed-form expectations."""
    metrics = compute_eigenmode_metrics(-0.5 + 1.2j, vee=1.0, bref=1.0, unitl=1.0)

    assert metrics.sigma == pytest.approx(-0.5)
    assert metrics.omega == pytest.approx(1.2)
    assert metrics.frequency_hz == pytest.approx(1.2 / (2.0 * math.pi))
    assert metrics.damping_ratio == pytest.approx(0.5 / math.hypot(0.5, 1.2))
    assert metrics.time_constant == pytest.approx(2.0)
    assert metrics.period_s == pytest.approx(2.0 * math.pi / 1.2)
    assert metrics.time_to_half_s == pytest.approx(math.log(2.0) * 2.0)

    subsidence = compute_eigenmode_metrics(-0.4 + 0.0j, vee=10.0, bref=2.0, unitl=1.0)
    assert subsidence.frequency_hz == pytest.approx(0.0)
    assert subsidence.period_s == math.inf
    assert subsidence.time_to_half_s == pytest.approx(math.log(2.0) / 0.4)


def test_appmat_shape():
    """Approximation matrix builder returns 12-state matrices."""
    state = _make_synthetic_state()
    asys, bsys, _ = build_appmat(state)
    assert asys.shape == (C.JETOT, C.JETOT)
    assert bsys.shape[0] == C.JETOT


def test_identify_modes_returns_flight_mode_objects():
    """Mode identification wraps eigenpairs in FlightMode dataclasses."""
    state = _make_synthetic_state()
    result = solve_eigenvalues(state, use_approx=True)
    assert len(result.modes) == len(result.eigenvalues)
    for mode in result.modes:
        assert mode.name
        assert mode.eigenvector.shape == (C.JETOT,)
        if mode.frequency_hz > 1e-12:
            assert mode.period_s == pytest.approx(1.0 / mode.frequency_hz)
        if mode.time_constant > 0 and math.isfinite(mode.time_constant):
            assert mode.time_to_half_s == pytest.approx(math.log(2.0) * mode.time_constant)


@pytest.mark.skipif(not PLANE_AVL.is_file(), reason="plane.avl not found")
def test_get_system_matrix_matches_eigenvalues():
    """Modal getters return the same A matrix as eigenvalues()."""
    solver = AVLSolver(PLANE_AVL)
    solver.set_parameter("mass", 1.0)
    solver.set_parameter("velocity", 1.0)
    solver.set_parameter("density", 1.0)
    solver.set_parameter("gravity", 1.0)
    solver.setup_trim(mode=1)
    solver.execute_run(max_iter=20)

    result = solver.eigenvalues(use_approx=True)
    asys, bsys, rsys = solver.get_system_matrices(use_approx=True)

    assert np.allclose(asys, result.state_matrix)
    assert bsys.shape[1] == solver.state.ncontrol
    assert rsys.shape == (C.JETOT,)


def test_body_axis_signs_flip_selected_rows_and_columns():
    """Body-axis display applies AVL SYSSHO sign pattern to A, B, and R."""
    from openavl.analysis.amode import apply_body_axis_signs

    asys = np.arange(C.JETOT * C.JETOT, dtype=np.float64).reshape(C.JETOT, C.JETOT)
    bsys = np.ones((C.JETOT, 2), dtype=np.float64)
    rsys = np.arange(C.JETOT, dtype=np.float64)

    signed_a, signed_b, signed_r = apply_body_axis_signs(asys, bsys, rsys, ncontrol=2)

    usgn = np.ones(C.JETOT, dtype=np.float64)
    for idx in (C.JEU, C.JEW, C.JEP, C.JER, C.JEX, C.JEZ):
        usgn[idx] = -1.0

    expected_a = asys * usgn[:, np.newaxis] * usgn[np.newaxis, :]
    expected_b = bsys * usgn[:, np.newaxis]
    expected_r = rsys * usgn

    assert np.allclose(signed_a, expected_a)
    assert np.allclose(signed_b, expected_b)
    assert np.allclose(signed_r, expected_r)
    assert not np.allclose(signed_a, asys)
