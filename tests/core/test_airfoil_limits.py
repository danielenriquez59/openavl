"""Regression tests for airfoil x/c limits and NACA/camber tables (GEO-06, GEO-15).

Reference CL/CD/Cm values were generated with Fortran AVL 3.52 through the
``optvl`` 2.5.0 package (``OVLSolver(geo_file=...)``, ``set_parameter("CD0", 0)``,
``set_constraint("alpha", "alpha", 4.0)``, ``execute_run()``,
``get_total_forces()``), run on the exact files written by ``_write_case``
below. optvl is not imported here.
"""

from __future__ import annotations

import shutil
import warnings

import numpy as np
import pytest

from openavl import constants as C
from openavl.fileio.parser import parse_avl
from openavl.geom.airfoil import build_camber_slope, build_naca_slope
from openavl.solver import AVLSolver

from tests.helpers import GEOMETRIES_DIR

pytestmark = pytest.mark.core

ALPHA_DEG = 4.0
TOL = 1e-6

_HEADER = """Test {name}
0.0
0 0 0.0
6.0 1.0 6.0
0.25 0.0 0.0
SURFACE
Wing
{nchord} {cspace} 10 1.0
YDUPLICATE
0.0
"""

_SECTIONS = """SECTION
0.0 0.0 0.0 1.0 0.0
{airfoil}
SECTION
0.2 3.0 0.0 0.8 -1.0
{airfoil}
"""


def _coarse_sd7037_coords() -> str:
    """Every 8th point of sd7037.dat plus its x = 0 point (20 points < 50)."""
    rows = []
    for line in (GEOMETRIES_DIR / "sd7037.dat").read_text().splitlines()[1:]:
        parts = line.split()
        if len(parts) >= 2:
            rows.append(line)
    return "\n".join(r for i, r in enumerate(rows) if i % 8 == 0 or r.split()[0] == "0.000000")


# name -> (nchord, cspace, airfoil text placed after each SECTION line)
CASES = {
    "naca_lim": (12, "1.0", "NACA 0.0 0.8\n2412"),
    "naca_lim_le": (12, "1.0", "NACA 0.2 1.0\n4412"),
    "naca_full_explicit": (12, "1.0", "NACA 0.0 1.0\n2412"),
    "afil_lim": (12, "1.0", "AFILE 0.0 0.7\nsd7037.dat"),
    "afil_lim_both": (12, "1.0", "AFILE 0.15 0.95\nag35.dat"),
    "naca_p0": (12, "1.0", "NACA\n2012"),
    "naca_5digit": (12, "1.0", "NACA\n23012"),
    "naca_dense": (40, "0.0", "NACA\n4412"),
    "airf_coarse": (12, "1.0", "AIRFOIL\n" + _coarse_sd7037_coords()),
}

# Fortran AVL 3.52 (optvl 2.5.0), alpha = 4 deg, CD0 = 0: (CL, CD, Cm).
OPTVL_REF = {
    "naca_lim": (0.3262353457, 0.0057289550, -0.0559280839),
    "naca_lim_le": (0.6019536636, 0.0196171915, -0.0934755001),
    "naca_full_explicit": (0.3970743816, 0.0085021958, -0.0645668924),
    "afil_lim": (0.3057750124, 0.0050323003, -0.0658659861),
    "afil_lim_both": (0.5188005302, 0.0145495603, -0.0551922468),
    "naca_p0": (0.3691862567, 0.0073514497, -0.0333830154),
    "naca_5digit": (1.6195219453, 0.1435906229, -0.2357638494),
    "naca_dense": (0.5449870481, 0.0160567702, -0.1148973123),
    "airf_coarse": (0.4503915304, 0.0109545229, -0.0860153999),
}
# optvl LRANGE (SURF_L) for these files: 0 for naca_lim, naca_lim_le, afil_lim,
# afil_lim_both; 1 for the rest.


def _write_case(directory, name: str):
    nchord, cspace, airfoil = CASES[name]
    text = _HEADER.format(name=name, nchord=nchord, cspace=cspace) + _SECTIONS.format(airfoil=airfoil)
    path = directory / f"{name}.avl"
    path.write_text(text)
    for dat in ("sd7037.dat", "ag35.dat"):
        shutil.copy(GEOMETRIES_DIR / dat, directory / dat)
    return path


def _solve(path) -> AVLSolver:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        solver = AVLSolver(path, base_dir=path.parent)
    solver.set_parameter("cd0", 0.0)
    solver.set_parameter("mach", 0.0)
    solver.set_variable("alpha", ALPHA_DEG)
    solver.set_variable("beta", 0.0)
    s = solver.state
    s.icon[C.IVALFA, 0] = C.ICALFA
    s.conval[C.ICALFA, 0] = ALPHA_DEG
    s.icon[C.IVBETA, 0] = C.ICBETA
    s.conval[C.ICBETA, 0] = 0.0
    for ic, iv in ((C.ICROTX, C.IVROTX), (C.ICROTY, C.IVROTY), (C.ICROTZ, C.IVROTZ)):
        s.icon[iv, 0] = ic
        s.conval[ic, 0] = 0.0
    solver.execute_run(max_iter=20)
    return solver


@pytest.mark.parametrize("name", sorted(CASES))
def test_fixed_alpha_forces_match_fortran(tmp_path, name):
    """CL, CD, Cm at fixed alpha match Fortran AVL 3.52 for limit/NACA cases."""
    s = _solve(_write_case(tmp_path, name)).state
    cl, cd, cm = OPTVL_REF[name]
    assert float(s.cltot) == pytest.approx(cl, abs=TOL)
    assert float(s.cdtot) == pytest.approx(cd, abs=TOL)
    assert float(s.cmtot[1]) == pytest.approx(cm, abs=TOL)


@pytest.mark.parametrize(
    ("name", "expected"),
    [("naca_lim", False), ("afil_lim", False), ("naca_full_explicit", True), ("naca_p0", True)],
)
def test_lrange_follows_airfoil_limits(tmp_path, name, expected):
    """LRANGE is false only for a partial x/c range (optvl LRANGE agrees)."""
    s = _solve(_write_case(tmp_path, name)).state
    assert [bool(v) for v in s.lrange[: s.nsurf]] == [expected, expected]


def _parse_one(airfoil: str, extra: str = ""):
    text = _HEADER.format(name="p", nchord=4, cspace="1.0") + _SECTIONS.format(airfoil=airfoil + extra)
    return parse_avl(text).surfaces[0]


def test_naca_limits_and_designator_from_next_line():
    surf = _parse_one("NACA 0.0 0.8\n2412")
    sec = surf.sections[0]
    assert sec.naca == "2412"
    assert (sec.xfmin, sec.xfmax) == (0.0, 0.8)
    assert surf.lrange is False


def test_afile_limits_only_from_keyword_line():
    """Numbers after a quoted filename are ignored; limits come from the keyword."""
    sec = _parse_one('AFILE\n"sd7037.dat" 0.2 0.5').sections[0]
    assert sec.airfoil_file == "sd7037.dat"
    assert (sec.xfmin, sec.xfmax) == (0.0, 1.0)
    sec = _parse_one("AFILE 0.1 1.5\nsd7037.dat").sections[0]
    assert (sec.xfmin, sec.xfmax) == (0.1, 1.0)


@pytest.mark.parametrize(("line", "expected"), [("NACA 0.005 0.995", True), ("NACA 0.0 0.98", False)])
def test_lrange_threshold(line, expected):
    assert _parse_one(line + "\n2412").lrange is expected


def test_claf_does_not_clear_lrange(tmp_path):
    """CLAF is the lift-slope factor; it no longer disables CPOML (LRANGE)."""
    path = tmp_path / "claf.avl"
    path.write_text(
        _HEADER.format(name="claf", nchord=4, cspace="1.0")
        + _SECTIONS.format(airfoil="NACA\n2412\nCLAF\n0.8")
    )
    s = _solve(path).state
    assert bool(s.lrange[0]) is True


def test_bfile_keyword_limits_not_part_of_path():
    text = "T\n0\n0 0 0\n1 1 1\n0 0 0\nBODY\nFuse\n10 1.0\nBFILE 0.0 1.0\nfuseBD.dat\n"
    assert parse_avl(text).bodies[0].body_file == "fuseBD.dat"


def test_naca_table_matches_ainput_grid():
    """50 uniform points over [xfmin, xfmax], normalised to 0..1 (ainput.f NACA)."""
    cam = build_naca_slope("2412")
    assert cam.x.size == 50
    np.testing.assert_allclose(cam.x, np.arange(50) / 49.0, atol=1e-15)
    cam = build_naca_slope("2412", xfmin=0.0, xfmax=0.8)
    np.testing.assert_allclose(cam.x, np.arange(50) / 49.0, atol=1e-15)
    xf = 0.8 * np.arange(50) / 49.0
    np.testing.assert_allclose(cam.s, np.where(xf < 0.4, 0.02 * 2 * (0.4 - xf) / 0.16, 0.02 * 2 * (0.4 - xf) / 0.36))


def test_naca_zero_position_slope_matches_fortran():
    """m != 0 with p = 0 gives SLP = -2 m x for x > 0, as in ainput.f."""
    cam = build_naca_slope("2012")
    xf = np.arange(50) / 49.0
    np.testing.assert_allclose(cam.s, -0.04 * xf, atol=1e-15)


def test_naca_five_digit_read_like_avl_with_warning():
    with pytest.warns(UserWarning, match="4-digit"):
        cam = build_naca_slope("23012")
    xf = np.arange(50) / 49.0
    np.testing.assert_allclose(cam.s, -0.46 * xf, atol=1e-15)


def test_camber_table_always_50_points_and_normalised_with_limits():
    text = _coarse_sd7037_coords()
    coords = [[float(v) for v in line.split()[:2]] for line in text.splitlines()]
    assert len(coords) < 50
    cam = build_camber_slope(coords, xfmin=0.1, xfmax=0.9)
    assert cam.x.size == 50
    assert cam.x[0] == pytest.approx(0.0, abs=1e-15)
    assert cam.x[-1] == pytest.approx(1.0, abs=1e-15)
