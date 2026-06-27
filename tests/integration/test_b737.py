"""End-to-end integration test for b737.avl."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

import openavl.constants as C
from openavl.solver import AVLSolver

from tests.helpers import GEOMETRIES_DIR, REF_DIR

B737_AVL = GEOMETRIES_DIR / "b737.avl"
REF_BIN = REF_DIR / "b737_exec_ref"
TOL_FORCE = 2e-4
TOL_STRIP = 5e-2


def apply_b737_constraints(state) -> None:
    """Match b737_exec_ref.f / JS pipeline constraint wiring."""
    ir = 0
    state.conval[C.ICCL, ir] = 0.6
    state.conval[C.ICBETA, ir] = 0.0
    state.conval[C.ICROTX, ir] = 0.0
    state.conval[C.ICROTY, ir] = 0.0
    state.conval[C.ICROTZ, ir] = 0.0

    state.icon[C.IVALFA, ir] = C.ICCL
    state.icon[C.IVBETA, ir] = C.ICBETA
    state.icon[C.IVROTX, ir] = C.ICROTX
    state.icon[C.IVROTY, ir] = C.ICROTY
    state.icon[C.IVROTZ, ir] = C.ICROTZ

    for n in range(state.ncontrol):
        iv = C.IVTOT + n
        ic = C.ICTOT + n
        state.conval[ic, ir] = 0.0
        state.icon[iv, ir] = ic


def build_b737_solver() -> AVLSolver:
    """Build b737 solver with the reference run-case parameters."""
    if not B737_AVL.is_file():
        pytest.skip(f"b737.avl not found: {B737_AVL}")

    solver = AVLSolver(
        B737_AVL,
        alpha=2.0,
        beta=0.0,
        cl=0.6,
        vel=16.34,
        rho=1.225,
        gravity=9.81,
        bank=0.0,
        cd0=0.0,
        xcg=60.0,
        ycg=0.0,
        zcg=0.0,
    )
    apply_b737_constraints(solver.state)
    return solver


def _parse_ref_output(stdout: str) -> dict:
    out: dict = {"strips": {}}
    num_re = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eEdD][-+]?\d+)?")
    for line in stdout.splitlines():
        if not line.strip():
            continue
        parts = line.strip().split()
        if parts[0] == "FORCE":
            nums = [float(v.replace("d", "e").replace("D", "e")) for v in num_re.findall(line)]
            out["force"] = {
                "CLTOT": nums[0],
                "CDTOT": nums[1],
                "CYTOT": nums[2],
                "CMTOT": nums[3:6],
                "CFTOT": nums[6:9],
            }
        elif parts[0] == "CDVTOT":
            out["CDVTOT"] = float(num_re.findall(line)[0].replace("d", "e").replace("D", "e"))
        elif parts[0] == "NVOR":
            out["NVOR"] = int(float(num_re.findall(line)[0]))
        elif parts[0] == "NSTRIP":
            out["NSTRIP"] = int(float(num_re.findall(line)[0]))
        elif parts[0] == "STRIP":
            nums = [float(v.replace("d", "e").replace("D", "e")) for v in num_re.findall(line)]
            j = int(nums[0])
            out["strips"][j] = {
                "y": nums[1],
                "z": nums[2],
                "cnc": nums[3],
                "cla": nums[4],
                "clt": nums[5],
                "dwwake": nums[6],
                "off": int(nums[7]) if len(nums) > 7 else 0,
            }
    return out


pytestmark = pytest.mark.integration


@pytest.mark.reference
def test_b737_exec_matches_fortran_ref():
    if not B737_AVL.is_file():
        pytest.skip(f"b737.avl not found: {B737_AVL}")
    if not (REF_BIN.is_file() or (REF_BIN.parent / "b737_exec_ref.exe").is_file()):
        pytest.skip("Fortran reference binary not found: b737_exec_ref")

    ref_path = REF_BIN if REF_BIN.is_file() else REF_BIN.parent / "b737_exec_ref.exe"
    try:
        proc = subprocess.run([str(ref_path), str(B737_AVL)], capture_output=True, text=True, check=False)
    except OSError as exc:
        pytest.skip(f"Cannot execute reference binary: {exc}")
    if proc.returncode != 0:
        pytest.skip(proc.stderr or "b737_exec_ref failed")
    ref = _parse_ref_output(proc.stdout)

    solver = build_b737_solver()
    solver.execute_run(max_iter=20)
    results = solver.get_results()
    state = solver.state

    assert state.nvor == ref["NVOR"]
    assert state.nstrip == ref["NSTRIP"]
    assert results["CL"] == pytest.approx(ref["force"]["CLTOT"], abs=TOL_FORCE)
    assert results["CD"] == pytest.approx(ref["force"]["CDTOT"], abs=TOL_FORCE)
    assert results["CY"] == pytest.approx(ref["force"]["CYTOT"], abs=TOL_FORCE)
    assert [results["Cl"], results["Cm"], results["Cn"]] == pytest.approx(ref["force"]["CMTOT"], abs=TOL_FORCE)
    assert [results["Cx"], results["Cy"], results["Cz"]] == pytest.approx(ref["force"]["CFTOT"], abs=TOL_FORCE)
    assert results["CDV"] == pytest.approx(ref["CDVTOT"], abs=TOL_FORCE)

    for j in range(state.nstrip):
        strip_ref = ref["strips"].get(j + 1)
        assert strip_ref is not None, f"missing ref strip {j + 1}"
        assert state.rle[1, j] == pytest.approx(strip_ref["y"], abs=TOL_STRIP)
        assert state.rle[2, j] == pytest.approx(strip_ref["z"], abs=TOL_STRIP)

        if strip_ref["off"]:
            assert state.cnc[j] == pytest.approx(0.0, abs=TOL_STRIP)
            assert state.cla_lstrp[j] == pytest.approx(0.0, abs=TOL_STRIP)
            assert state.clt_lstrp[j] == pytest.approx(0.0, abs=TOL_STRIP)
            assert state.dwwake[j] == pytest.approx(0.0, abs=TOL_STRIP)
        else:
            assert state.cnc[j] == pytest.approx(strip_ref["cnc"], abs=TOL_STRIP)
            assert state.cla_lstrp[j] == pytest.approx(strip_ref["cla"], abs=TOL_STRIP)
            assert state.clt_lstrp[j] == pytest.approx(strip_ref["clt"], abs=TOL_STRIP)
            assert state.dwwake[j] == pytest.approx(strip_ref["dwwake"], abs=TOL_STRIP)
