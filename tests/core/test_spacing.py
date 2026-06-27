"""Tests for openavl.spacing."""

from __future__ import annotations

import numpy as np
import pytest

from openavl.spacing import akima, cspacer, nrmliz, spacer
from tests.helpers import load_json_fixture, run_ref_binary


def _build_spacing_outputs(data: dict) -> list[float]:
    """Build spacing routine outputs matching sgutil.test.mjs layout."""
    n = data["n"]
    x = np.array(data["x"], dtype=np.float64)
    y = np.array(data["y"], dtype=np.float64)
    yy, slp = akima(x, y, data["xx"])
    xn = nrmliz(x.copy())
    xsp = spacer(n, data["pspace"])
    xpt, xvr, xsr, xcp = cspacer(data["nvc"], data["cspace"], data["claf"])
    return [
        yy,
        slp,
        *xn.tolist(),
        *xsp[1:].tolist(),
        *xpt[1 : data["nvc"] + 2].tolist(),
        *xvr[1 : data["nvc"] + 1].tolist(),
        *xsr[1 : data["nvc"] + 1].tolist(),
        *xcp[1 : data["nvc"] + 1].tolist(),
    ]


pytestmark = pytest.mark.core


@pytest.mark.fixture
def test_spacing_matches_js_fixture(fixtures_dir):
    """SPACER/CSPACER/AKIMA/NRMLIZ match JS reference fixture."""
    data = load_json_fixture(fixtures_dir, "spacing_cases.json")
    actual = _build_spacing_outputs(data)
    np.testing.assert_allclose(actual, data["expected"], rtol=0, atol=1e-5)


@pytest.mark.reference
def test_spacing_matches_fortran_ref(ref_binary, fixtures_dir):
    """Spacing routines match Fortran sgutil_ref when available."""
    ref_path = ref_binary("sgutil_ref")
    if ref_path is None:
        pytest.skip("Fortran reference binary not found")

    data = load_json_fixture(fixtures_dir, "spacing_cases.json")
    ref = run_ref_binary(ref_path)
    actual = _build_spacing_outputs(data)
    np.testing.assert_allclose(actual, ref[: len(actual)], rtol=0, atol=1e-5)
