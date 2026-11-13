"""LU factor reuse and gucalc unit-solution cache."""

from __future__ import annotations

import dataclasses
import importlib

import numpy as np
import pytest

from openavl.analysis.deriv import compute_stability_derivatives
from openavl.math.linalg import baksub, ludcmp
from openavl.core.solver import AVLSolver
from tests.helpers import GEOMETRIES_DIR

# openavl.core re-exports the setup() function under the submodule's name.
setup_mod = importlib.import_module("openavl.core.setup")

pytestmark = pytest.mark.core

TOL = 1e-12


def _matrix(n: int, order: str, pad: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Random well-conditioned matrix (optionally embedded in a larger buffer)."""
    rng = np.random.default_rng(n + pad)
    dense = rng.standard_normal((n, n)) + n * np.eye(n)
    buf = np.zeros((n + pad, n + pad), dtype=np.float64, order=order)
    buf[:n, :n] = dense
    return buf, dense


@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("pad", [0, 3])
@pytest.mark.parametrize("nrhs", [None, 1, 7])
def test_ludcmp_baksub_match_numpy_solve(order: str, pad: int, nrhs: int | None):
    """Factor once, then several back-substitutions all match numpy.linalg.solve."""
    n = 40
    a, dense = _matrix(n, order, pad)
    indx = np.zeros(n + pad, dtype=np.int32)
    ludcmp(a, n, indx, np.zeros(n + pad))
    factors = a.copy()

    rng = np.random.default_rng(7)
    for _ in range(3):
        shape = (n + pad,) if nrhs is None else (n + pad, nrhs)
        b = rng.standard_normal(shape)
        expected = np.linalg.solve(dense, b[:n])
        tail = b[n:].copy()
        baksub(a, n, indx, b)
        scale = np.abs(expected).max()
        assert np.abs(b[:n] - expected).max() <= TOL * scale
        np.testing.assert_array_equal(b[n:], tail)
        # Back-substitution must not disturb the stored factors.
        np.testing.assert_array_equal(a, factors)


def test_baksub_column_block_view():
    """A non-contiguous column block (as gucalc passes) is solved in place."""
    n = 30
    a, dense = _matrix(n, "F")
    indx = np.zeros(n, dtype=np.int32)
    ludcmp(a, n, indx, np.zeros(n))
    store = np.random.default_rng(3).standard_normal((n + 5, 6, 2))
    view = store[:n, :6, 1]
    expected = np.linalg.solve(dense, view)
    baksub(a, n, indx, view)
    assert np.abs(store[:n, :6, 1] - expected).max() <= TOL * np.abs(expected).max()


def test_ludcmp_fortran_factors_in_place_like_c_order():
    """F-ordered input is factored in place and yields the same factors as C order."""
    n = 25
    a_f, _ = _matrix(n, "F")
    a_c = np.ascontiguousarray(a_f)
    buf = a_f.__array_interface__["data"][0]
    ip_f = np.zeros(n, dtype=np.int32)
    ip_c = np.zeros(n, dtype=np.int32)
    ludcmp(a_f, n, ip_f, np.zeros(n))
    ludcmp(a_c, n, ip_c, np.zeros(n))
    assert a_f.__array_interface__["data"][0] == buf
    np.testing.assert_array_equal(a_f, a_c)
    np.testing.assert_array_equal(ip_f, ip_c)


def test_baksub_rejects_non_finite_rhs():
    """Non-finite RHS still raises, as with SciPy's default check_finite."""
    n = 5
    a, _ = _matrix(n, "F")
    indx = np.zeros(n, dtype=np.int32)
    ludcmp(a, n, indx, np.zeros(n))
    b = np.ones(n)
    b[2] = np.nan
    with pytest.raises(ValueError):
        baksub(a, n, indx, b)


# --- gucalc cache -----------------------------------------------------------

BD_AVL = GEOMETRIES_DIR / "bd.avl"
VANILLA_AVL = GEOMETRIES_DIR / "vanilla.avl"
RATES = {"pb/2v": 0.05, "qc/2v": 0.02, "rb/2v": -0.03}


def _bd_rates() -> AVLSolver:
    """Bubble Dancer (with fuselage body) trimmed with non-zero body rates."""
    if not BD_AVL.is_file():
        pytest.skip(f"bd.avl not found: {BD_AVL}")
    s = AVLSolver(BD_AVL, cl=0.7, cd0=0.017, rho=1.225, gravity=9.81, xcg=3.4, zcg=0.5, bank=0.0)
    s.set_constraint("beta", "cll", 0.0)
    s.set_constraint("elevator", "cm", 0.0)
    s.set_constraint("rudder", "cn", 0.0)
    for name, value in RATES.items():
        s.set_constraint(name, name, value)
    return s


def _vanilla_rates() -> AVLSolver:
    """Vanilla (no bodies) trimmed with non-zero body rates."""
    if not VANILLA_AVL.is_file():
        pytest.skip(f"vanilla.avl not found: {VANILLA_AVL}")
    s = AVLSolver(VANILLA_AVL, cl=1.16647, vel=30.8633, rho=0.2, gravity=10.0, xcg=0.65, cd0=0.0, beta=5.0)
    s.set_constraint("elevator", "cm", 0.0)
    s.set_constraint("rudder", "cn", 0.0)
    for name, value in RATES.items():
        s.set_constraint(name, name, value)
    return s


def _snapshot(s: AVLSolver) -> dict[str, np.ndarray]:
    st = s.state
    n, nc = st.nvor, st.ncontrol
    out = {
        "gam_u_0": st.gam_u_0[:n, :6].copy(),
        "gam_u_d": st.gam_u_d[:n, :6, :nc].copy(),
        "gam_u": st.gam_u[:n, :6].copy(),
        "gam_d": st.gam_d[:n, :nc].copy(),
        "gam": st.gam[:n].copy(),
    }
    r = s.get_results()
    for k in ("CL", "CD", "CY", "Cl", "Cm", "Cn"):
        out[k] = np.array(float(r[k]))
    d = compute_stability_derivatives(st)
    for f in dataclasses.fields(d):
        try:
            out[f"d_{f.name}"] = np.asarray(getattr(d, f.name), dtype=float)
        except (TypeError, ValueError):
            pass
    return out


def _assert_same(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> None:
    assert a.keys() == b.keys()
    for k in a:
        scale = max(float(np.abs(a[k]).max(initial=0.0)), 1e-300)
        diff = float(np.abs(a[k] - b[k]).max(initial=0.0))
        assert diff <= TOL * scale, f"{k}: max diff {diff:.3e} (scale {scale:.3e})"


def _fresh_at(reused: AVLSolver, build, change) -> AVLSolver:
    """New solver with the same settings, evaluated at the reused solver's variables."""
    fresh = build()
    change(fresh)
    src, dst = reused.state, fresh.state
    dst.alfa, dst.beta = src.alfa, src.beta
    dst.wrot[:] = src.wrot
    dst.delcon[: src.ncontrol] = src.delcon[: src.ncontrol]
    fresh.execute_run(max_iter=0)
    return fresh


def _run_change(build, change) -> AVLSolver:
    s = build()
    s.execute_run()
    assert s.state.gucalc_cache is not None
    change(s)
    s.execute_run()
    assert s.get_results()["converged"]
    return s


def test_gucalc_cache_reused_across_alpha_sweep():
    """Changing only the operating point reuses the cached unit solutions."""
    s = _bd_rates()
    s.execute_run()
    cache = s.state.gucalc_cache
    s.set_constraint("alpha", "alpha", 4.0)
    s.execute_run()
    assert s.state.gucalc_cache is cache
    fresh = _fresh_at(s, _bd_rates, lambda x: x.set_constraint("alpha", "alpha", 4.0))
    _assert_same(_snapshot(s), _snapshot(fresh))


def test_gucalc_cache_invalidated_by_xcg_no_body():
    """xcg changes xyzref without clearing any flag; reused == fresh (no bodies)."""
    change = lambda x: x.set_parameter("xcg", 0.5)  # noqa: E731
    s = _vanilla_rates()
    s.execute_run()
    before = s.state.gam_u_0[: s.state.nvor, 3:6].copy()
    s = _run_change(_vanilla_rates, change)
    assert np.abs(s.state.gam_u_0[: s.state.nvor, 3:6] - before).max() > 0.0
    _assert_same(_snapshot(s), _snapshot(_fresh_at(s, _vanilla_rates, change)))


def test_gucalc_cache_invalidated_by_xcg_with_body(monkeypatch):
    """bd.avl (body + rates): cached path equals the uncached path after an xcg change.

    A fresh solver is not the reference here: a reused solver does not rebuild
    the body source/doublet influence after a CG change (a known, separate
    issue), so reused and fresh solvers differ with or without this cache.
    """
    change = lambda x: x.set_parameter("xcg", 3.0)  # noqa: E731
    cached = _snapshot(_run_change(_bd_rates, change))
    monkeypatch.setattr(setup_mod, "_gucalc_cache_valid", lambda state: False)
    uncached = _snapshot(_run_change(_bd_rates, change))
    _assert_same(cached, uncached)


def test_gucalc_cache_invalidated_by_mach():
    """A Mach change refactors the AIC and rebuilds the unit solutions."""
    change = lambda x: x.set_parameter("mach", 0.3)  # noqa: E731
    s = _run_change(_bd_rates, change)
    _assert_same(_snapshot(s), _snapshot(_fresh_at(s, _bd_rates, change)))


def test_gucalc_cache_with_control_and_constraint_change():
    """Fixed deflections and new rate targets give the same answer as a fresh solver."""

    def change(x: AVLSolver) -> None:
        x.set_constraint("elevator", "elevator", 2.0)
        x.set_constraint("pb/2v", "pb/2v", -0.02)
        x.set_constraint("rb/2v", "rb/2v", 0.04)

    s = _run_change(_bd_rates, change)
    _assert_same(_snapshot(s), _snapshot(_fresh_at(s, _bd_rates, change)))
