"""Tests for openavl.geom.spacing."""

from __future__ import annotations

import numpy as np
import pytest

from openavl.geom.spacing import akima, cspacer, nrmliz, spacer
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
    """SPACER/CSPACER/AKIMA/NRMLIZ match reference fixture.

    SPACER/CSPACER/NRMLIZ values come from the JS port. The AKIMA (yy, slp)
    entries were regenerated from the literal sgutil.f transcription below,
    since the JS port shared the GEO-01 indexing bug.
    """
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


def _akima_f77(X, Y, N, XX):
    """Literal transcription of sgutil.f AKIMA, kept in Fortran 1-based indexing.

    Independent of the port in openavl.geom.spacing; index 0 is unused padding.
    """
    X = [None] + list(X)
    Y = [None] + list(Y)
    D = [None, 0.0, 0.0, 0.0, 0.0, 0.0]
    T = [None, 0.0, 0.0]
    if X[1] == X[N]:
        return Y[1], 0.0
    XORDR = 1.0
    if X[1] > X[N]:
        XORDR = -1.0
    IBOT = 1
    ITOP = N
    XXO = XX * XORDR
    while True:
        NSTEP = (ITOP - IBOT) // 2
        I = IBOT + NSTEP
        XO = X[I] * XORDR
        if XXO >= XO:
            IBOT = I
        if XXO < XO:
            ITOP = I
        if NSTEP == 0:
            break
    for J in range(1, 6):
        K = I + (J - 2)
        if (K - 1) >= 1 and K <= N:
            D[J] = (Y[K] - Y[K - 1]) / (X[K] - X[K - 1])
    if N == 2:
        D[2] = D[3]
    if (I + 2) > N:
        D[4] = 2.0 * D[3] - D[2]
    if (I + 3) > N:
        D[5] = 2.0 * D[4] - D[3]
    if (I - 1) < 1:
        D[2] = 2.0 * D[3] - D[4]
    if (I - 2) < 1:
        D[1] = 2.0 * D[2] - D[3]
    for J in (1, 2):
        A = abs(D[J + 3] - D[J + 2])
        B = abs(D[J + 1] - D[J])
        if (A + B) == 0.0:
            A = 1.0
            B = 1.0
        T[J] = (A * D[J + 1] + B * D[J + 2]) / (A + B)
    if XX == X[I + 1]:
        return Y[I + 1], T[2]
    XINT = X[I + 1] - X[I]
    XDIF = XX - X[I]
    P0 = Y[I]
    P1 = T[1]
    P2 = (3.0 * D[3] - 2.0 * T[1] - T[2]) / XINT
    P3 = (T[1] + T[2] - 2.0 * D[3]) / (XINT * XINT)
    YY = P0 + XDIF * (P1 + XDIF * (P2 + XDIF * P3))
    SLP = P1 + XDIF * (2.0 * P2 + XDIF * (3.0 * P3))
    return YY, SLP


@pytest.mark.parametrize("descending", [False, True])
def test_akima_reproduces_linear_data(descending):
    """Linear data: exact value and slope in every interval and at nodes (GEO-01)."""
    x = np.arange(6.0)
    y = 2.0 * x - 1.0
    if descending:
        x, y = x[::-1].copy(), y[::-1].copy()
    # first interval, interior, last interval, nodes, and mild extrapolation
    for xx in [0.25, 0.5, 1.5, 2.7, 3.3, 4.75, 0.0, 1.0, 3.0, 5.0, -0.5, 5.5]:
        yy, slp = akima(x, y, xx)
        assert yy == pytest.approx(2.0 * xx - 1.0, abs=1e-12)
        assert slp == pytest.approx(2.0, abs=1e-12)


def test_akima_old_bug_values_are_gone():
    """y=x on 0..5 previously gave 5.12 at 0.25 and 1.375 at 1.5."""
    x = np.arange(6.0)
    assert akima(x, x, 0.25) == pytest.approx((0.25, 1.0), abs=1e-12)
    assert akima(x, x, 1.5) == pytest.approx((1.5, 1.0), abs=1e-12)


def test_akima_reproduces_quadratic_data():
    """Quadratic data on a uniform grid: value and slope reproduced in the interior."""
    x = np.linspace(0.0, 1.0, 11)
    y = 3.0 * x**2 - 2.0 * x + 0.5
    for xx in np.linspace(0.12, 0.88, 23):
        yy, slp = akima(x, y, float(xx))
        assert yy == pytest.approx(3.0 * xx**2 - 2.0 * xx + 0.5, abs=1e-12)
        assert slp == pytest.approx(6.0 * xx - 2.0, abs=1e-11)


def test_akima_matches_fortran_transcription():
    """Port agrees with the literal sgutil.f transcription on irregular data."""
    cases = [
        ([0.0, 1.0, 2.5, 3.0, 4.5, 6.0], [0.0, 1.2, 0.8, 1.6, 1.1, 2.0]),
        ([0.0, 1.0], [1.0, 3.0]),
        ([0.0, 0.3, 1.0], [0.0, 0.5, 0.2]),
        ([0.0, 0.1, 0.15, 0.4, 0.7, 1.0, 1.2], [0.0, 0.3, 0.2, 0.2, 0.9, 0.4, 0.6]),
    ]
    rng = np.random.default_rng(1234)
    for _ in range(20):
        n = int(rng.integers(2, 9))
        cases.append((np.sort(rng.uniform(0.0, 1.0, n)).tolist(), rng.normal(size=n).tolist()))
    for xs, ys in cases:
        n = len(xs)
        span = xs[-1] - xs[0]
        probes = list(xs) + list(np.linspace(xs[0] - 0.1 * span, xs[-1] + 0.1 * span, 37))
        for xa, ya in ((xs, ys), (xs[::-1], ys[::-1])):
            for xx in probes:
                ref = _akima_f77(xa, ya, n, float(xx))
                got = akima(np.array(xa), np.array(ya), float(xx))
                np.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-12)


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
