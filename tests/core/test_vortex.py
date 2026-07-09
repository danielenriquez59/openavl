"""Tests for openavl.vortex."""

from __future__ import annotations

import numpy as np
import pytest

from openavl.vortex import vorvelc
from tests.helpers import load_json_fixture

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
