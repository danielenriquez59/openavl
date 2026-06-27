"""End-to-end integration test for plane.avl."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

from openavl.solver import AVLSolver

from tests.helpers import GEOMETRIES_DIR, REF_DIR

PLANE_AVL = GEOMETRIES_DIR / "plane.avl"
REF_BIN = REF_DIR / "plane_exec_ref"
TOL = 2e-4


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
            }
        elif parts[0] == "NVOR":
            out["NVOR"] = int(float(num_re.findall(line)[0]))
        elif parts[0] == "NSTRIP":
            out["NSTRIP"] = int(float(num_re.findall(line)[0]))
    return out


pytestmark = pytest.mark.integration


@pytest.mark.reference
def test_plane_exec_matches_fortran_ref():
    if not PLANE_AVL.is_file():
        pytest.skip(f"plane.avl not found: {PLANE_AVL}")
    if not (REF_BIN.is_file() or (REF_BIN.parent / "plane_exec_ref.exe").is_file()):
        pytest.skip("Fortran reference binary not found: plane_exec_ref")

    ref_path = REF_BIN if REF_BIN.is_file() else REF_BIN.parent / "plane_exec_ref.exe"
    try:
        proc = subprocess.run([str(ref_path), str(PLANE_AVL)], capture_output=True, text=True, check=False)
    except OSError as exc:
        pytest.skip(f"Cannot execute reference binary: {exc}")
    if proc.returncode != 0:
        pytest.skip(proc.stderr or "plane_exec_ref failed")
    ref = _parse_ref_output(proc.stdout)

    solver = AVLSolver(
        PLANE_AVL,
        alpha=-0.1455,
        beta=0.0,
        cl=0.390510,
        vel=64.5396,
        rho=0.0005846,
        gravity=32.18,
        cd0=0.00835,
        xcg=0.02463,
        ycg=0.0,
        zcg=0.2239,
    )
    solver.execute_run(max_iter=20)
    results = solver.get_results()

    assert solver.state.nvor == ref["NVOR"]
    assert solver.state.nstrip == ref["NSTRIP"]
    assert results["CL"] == pytest.approx(ref["force"]["CLTOT"], abs=TOL)
    assert results["CD"] == pytest.approx(ref["force"]["CDTOT"], abs=TOL)
    assert results["CY"] == pytest.approx(ref["force"]["CYTOT"], abs=TOL)
    assert [results["Cl"], results["Cm"], results["Cn"]] == pytest.approx(ref["force"]["CMTOT"], abs=TOL)
