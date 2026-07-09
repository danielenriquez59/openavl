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


def _hand_cspacer_endpoints(nvc: int, cspace: float, claf: float = 1.0):
    """Hand-compute CSPACER panel-edge pins for regression checks."""
    xpt, _, _, _ = cspacer(nvc, cspace, claf)
    return xpt


@pytest.mark.parametrize(
    "nvc,cspace",
    [
        (10, 1.0),   # pure cosine
        (10, -1.0),  # pure sine
        (8, 0.5),    # blended uniform/cosine
        (6, 1.5),    # blended cosine/sine
    ],
)
def test_cspacer_pins_leading_and_trailing_edges(nvc, cspace):
    """CSPACER must pin XPT(1)=0 and XPT(NVC+1)=1 (sgutil.f), not the pad slot."""
    xpt = _hand_cspacer_endpoints(nvc, cspace)
    assert xpt[0] == pytest.approx(0.0)
    assert xpt[1] == pytest.approx(0.0)
    assert xpt[nvc + 1] == pytest.approx(1.0)
    # Without the LE pin, cosine spacing leaves a small positive xpt[1].
    if cspace == 1.0:
        dth1 = np.pi / (4 * nvc + 2)
        unpinned = 0.5 * (1.0 - np.cos(dth1))
        assert unpinned > 1e-4
        assert xpt[1] != pytest.approx(unpinned)
