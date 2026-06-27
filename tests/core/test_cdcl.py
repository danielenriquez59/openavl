"""Tests for openavl.cdcl."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from openavl.cdcl import cdcl
from tests.helpers import REF_DIR

pytestmark = pytest.mark.core

CASES = [
    {"cdclpol": [-1.0, 0.08, 0.0, 0.02, 1.0, 0.09], "cl": -1.5},
    {"cdclpol": [-1.0, 0.08, 0.0, 0.02, 1.0, 0.09], "cl": -0.5},
    {"cdclpol": [-1.0, 0.08, 0.0, 0.02, 1.0, 0.09], "cl": 0.2},
    {"cdclpol": [-1.0, 0.08, 0.0, 0.02, 1.0, 0.09], "cl": 1.3},
    {"cdclpol": [-0.5, 0.06, 0.2, 0.015, 1.4, 0.11], "cl": 0.7},
]


def _run_fortran_ref(cases: list[dict]) -> list[tuple[float, float]]:
    from tests.helpers import run_ref_binary

    ref_path = REF_DIR / "cdcl_ref"
    if not (ref_path.is_file() or (REF_DIR / "cdcl_ref.exe").is_file()):
        pytest.skip("Fortran reference binary not found: cdcl_ref")
    lines = [str(len(cases))]
    for c in cases:
        lines.append(" ".join(str(float(v)) for v in [*c["cdclpol"], c["cl"]]))
    values = run_ref_binary(ref_path if ref_path.is_file() else REF_DIR / "cdcl_ref.exe", "\n".join(lines))
    return [(values[i * 2], values[i * 2 + 1]) for i in range(len(cases))]


@pytest.mark.reference
def test_cdcl_matches_fortran_ref():
    ref = _run_fortran_ref(CASES)
    for c, (cd_ref, cd_cl_ref) in zip(CASES, ref, strict=True):
        cd, cd_cl = cdcl(np.array(c["cdclpol"], dtype=np.float64), c["cl"])
        assert cd == pytest.approx(cd_ref, abs=1e-5)
        assert cd_cl == pytest.approx(cd_cl_ref, abs=1e-5)
