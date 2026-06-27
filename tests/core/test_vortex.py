"""Tests for openavl.vortex."""

from __future__ import annotations

import numpy as np
import pytest

from openavl.vortex import vorvelc
from tests.helpers import load_json_fixture, run_ref_binary

pytestmark = pytest.mark.core


@pytest.mark.fixture
def test_vorvelc_matches_js_fixture(fixtures_dir):
    """VORVELC matches JS reference fixture."""
    data = load_json_fixture(fixtures_dir, "vorvelc_cases.json")
    actual: list[float] = []
    for case in data["cases"]:
        u, v, w = vorvelc(
            case["x"], case["y"], case["z"], case["lbound"],
            case["x1"], case["y1"], case["z1"],
            case["x2"], case["y2"], case["z2"],
            case["beta"], case["rcore"],
        )
        actual.extend([u, v, w])
    np.testing.assert_allclose(np.asarray(actual, dtype=np.float64), np.asarray(data["expected"], dtype=np.float64), rtol=0, atol=1e-5)


@pytest.mark.reference
def test_vorvelc_matches_fortran_ref(ref_binary, fixtures_dir):
    """VORVELC matches Fortran vorvelc_ref when available."""
    ref_path = ref_binary("vorvelc_ref")
    if ref_path is None:
        pytest.skip("Fortran reference binary not found")

    data = load_json_fixture(fixtures_dir, "vorvelc_cases.json")
    lines = [str(len(data["cases"]))]
    for case in data["cases"]:
        lines.append(
            " ".join(
                str(case[k])
                for k in (
                    "x", "y", "z", "lbound",
                    "x1", "y1", "z1", "x2", "y2", "z2", "beta", "rcore",
                )
            ).replace("True", "1").replace("False", "0")
        )
    for i, case in enumerate(data["cases"]):
        lines[i + 1] = " ".join(
            [
                str(case["x"]), str(case["y"]), str(case["z"]),
                "1" if case["lbound"] else "0",
                str(case["x1"]), str(case["y1"]), str(case["z1"]),
                str(case["x2"]), str(case["y2"]), str(case["z2"]),
                str(case["beta"]), str(case["rcore"]),
            ]
        )

    ref_vals = run_ref_binary(ref_path, "\n".join(lines))
    np.testing.assert_allclose(ref_vals, data["expected"], rtol=0, atol=1e-5)
