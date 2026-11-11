"""Parser rules that follow ainput.f (GEO-08 to GEO-13).

Reference forces were generated with Fortran AVL 3.52 through the ``optvl``
2.5.0 package, run on the exact files written by ``_write_case`` below:
``OVLSolver(geo_file=...)``, ``set_parameter("CD0", 0)``,
``set_constraint("alpha", "alpha", 4.0)``, ``set_constraint("beta", "beta", BETA)``,
``execute_run()``, ``get_total_forces()``; strip counts from ``CASE_I.NSTRIP``
and components from ``get_avl_fort_arr("SURF_I", "LNCOMP")``. optvl is not
imported here.
"""

from __future__ import annotations

import warnings

import pytest

from openavl import constants as C
from openavl.core.reporting import reported_totals
from openavl.fileio.parser import parse_avl, parse_avl_file, prepare_model
from openavl.geometry import Aircraft
from openavl.solver import AVLSolver

from tests.helpers import GEOMETRIES_DIR

pytestmark = pytest.mark.core

ALPHA_DEG = 4.0
TOL = 1e-7


def _hdr(title: str, sym: str = "0 0 0.0", ref: str = "4.0 1.0 4.0") -> str:
    return f"{title}\n0.0\n{sym}\n{ref}\n0.25 0.0 0.0\n"


_COORDS = """1.0 0.0
0.75 0.03
0.5 0.05
0.25 0.05
0.1 0.035
0.0 0.0
0.1 -0.01
0.25 -0.01
0.5 -0.005
0.75 0.0
1.0 0.0
"""

_WING = """SURFACE
Wing
6 1.0 10 1.0
YDUPLICATE
0.0
{surf}SECTION
0 0 0 1 0
{a}
SECTION
0.1 2 0 0.6 0
{b}
"""

_COMPONENTS = """SURFACE
Wing
8 1.0 12 1.0
YDUPLICATE
0.0
SECTION
0 0 0 1 0
SECTION
0.1 3 0 0.6 0
SURFACE
Canard
6 1.0 9 -1.0
SECTION
1.05 0 0.0 0.5 2
SECTION
1.10 1.5 0.0 0.4 2
SURFACE
Tail
6 1.0 9 1.0
INDEX
2
SECTION
1.58 0.03 0.0 0.5 -1
SECTION
1.62 1.47 0.0 0.4 -1
"""

_HALF = """SURFACE
Wing
6 1.0 10 1.0
SECTION
0 0 0 1 0
SECTION
0.1 2 0 0.6 0
SURFACE
Fin
4 1.0 6 1.0
SECTION
1.0 0 0 0.5 0
SECTION
1.2 0 0.8 0.3 0
"""

_AF_DAT = "af\n" + _COORDS

# name -> (file text, beta)
CASES = {
    # GEO-09: CLAF (and DESIGN) right after an inline AIRFOIL block.
    "airf_claf": (_hdr("airf claf") + _WING.format(surf="", a="AIRFOIL\n" + _COORDS + "CLAF\n1.1", b="NACA\n2412"), 5.0),
    "airf_design_claf": (
        _hdr("airf design claf")
        + _WING.format(surf="", a="AIRFOIL\n" + _COORDS + "DESIGN\ntw 1.0\nCLAF\n1.1", b="NACA\n2412"),
        5.0,
    ),
    # GEO-10: SURFACE line with Nspan but no Sspace; SECTION line with 6 values.
    "surf3": (
        _hdr("surf3")
        + "SURFACE\nWing\n6 1.0 12\nYDUPLICATE\n0.0\n"
        + "SECTION\n0 0 0 1 0 5 -2.0\nSECTION\n0.1 1 0 0.8 0 3 -1.0\nSECTION\n0.2 2 0 0.6 0 0 0\n",
        5.0,
    ),
    "sect6": (
        _hdr("sect6")
        + "SURFACE\nWing\n6 1.0\nYDUPLICATE\n0.0\n"
        + "SECTION\n0 0 0 1 0 5 -2.0\nSECTION\n0.1 1 0 0.8 0 7\nSECTION\n0.2 2 0 0.6 0 4 1.0\n",
        5.0,
    ),
    "sect7trunc": (
        _hdr("sect7trunc")
        + "SURFACE\nWing\n6 1.0\nYDUPLICATE\n0.0\n"
        + "SECTION\n0 0 0 1 0 5 -2.0\nSECTION\n0.1 1 0 0.8 0 6.9995 -1.0\nSECTION\n0.2 2 0 0.6 0 4 1.0\n",
        5.0,
    ),
    # GEO-11a: CDCL with a CL tie and with a negative CD are kept and sorted as in ainput.f.
    "cdcl_tie": (
        _hdr("cdcl tie")
        + _WING.format(surf="", a="CDCL\n-0.6 0.02 0.2 0.008 1.2 0.025", b="CDCL\n0.5 0.03 -0.5 0.02 0.5 0.01"),
        5.0,
    ),
    "cdcl_negcd": (
        _hdr("cdcl negcd")
        + _WING.format(surf="", a="CDCL\n-0.6 0.02 0.2 0.008 1.2 0.025", b="CDCL\n-0.5 0.02 0.3 -0.004 1.2 0.03"),
        5.0,
    ),
    # GEO-11c/d: IZSYM clamped to -1; Sref/Cref/Bref <= 0 reset to 1.
    "izsym_m3": (_hdr("izsym -3", sym="0 -3 -0.5") + _WING.format(surf="", a="", b=""), 5.0),
    "refneg": (_hdr("ref <= 0", ref="-4.0 0.0 -1.0") + _WING.format(surf="", a="", b=""), 5.0),
    # GEO-11c: IYSYM 2 behaves as 1 (beta = 0 with y-symmetry).
    "iysym2": (_hdr("iysym 2", sym="2 0 0.0") + _HALF, 0.0),
    # GEO-11e: the last of NACA / AIRFOIL / AFILE wins.
    "afil_then_naca": (_hdr("afil naca") + _WING.format(surf="", a="AFILE\naf.dat\nNACA\n4412", b="AFILE\naf.dat\nNACA\n4412"), 5.0),
    "airf_then_naca": (_hdr("airf naca") + _WING.format(surf="", a="AIRFOIL\n" + _COORDS + "NACA\n4412", b="NACA\n0012"), 5.0),
    "naca_then_afil": (_hdr("naca afil") + _WING.format(surf="", a="NACA\n4412\nAFILE\naf.dat", b="NACA\n4412\nAFILE\naf.dat"), 5.0),
    "afil_lim_then_naca": (
        _hdr("afil lim naca") + _WING.format(surf="", a="AFILE 0.0 0.5\naf.dat\nNACA\n4412", b="NACA 0.0 0.5\n4412\nAFILE\naf.dat"),
        5.0,
    ),
    "naca_then_missing_afil": (_hdr("missing afil") + _WING.format(surf="", a="NACA\n4412\nAFILE\nnope.dat", b="NACA\n4412"), 5.0),
    # GEO-12: default component numbers count the YDUPLICATE image.
    "comp_mixed": (_hdr("components") + _COMPONENTS, 5.0),
}

# Fortran AVL 3.52 (optvl 2.5.0), alpha = 4 deg, CD0 = 0:
# (CL, CD, CY, Cl, Cm, Cn, NSTRIP, LNCOMP)
OPTVL_REF = {
    "airf_claf": (0.3781294343, 0.0115026123, -0.0000072822, 0.0003741845, -0.0251175420, -0.0000040014, 20, [1, 1]),
    "airf_design_claf": (0.3781294343, 0.0115026123, -0.0000072822, 0.0003741845, -0.0251175420, -0.0000040014, 20, [1, 1]),
    "surf3": (0.2240182233, 0.0040098530, 0.0000000000, -0.0002257923, -0.0073510428, -0.0000000000, 16, [1, 1]),
    "sect6": (0.0829539046, 0.0021857747, -0.0000000000, -0.0000461926, 0.0009366477, 0.0000000000, 10, [1, 1]),
    "sect7trunc": (0.2241336691, 0.0039937667, -0.0000000000, -0.0002266947, -0.0073953342, -0.0000000000, 24, [1, 1]),
    "cdcl_tie": (0.2240714173, 0.0113269066, -0.0006360583, -0.0000144078, 0.0023204305, -0.0000000512, 20, [1, 1]),
    "cdcl_negcd": (0.2240714173, 0.0061827982, -0.0001860071, -0.0000144091, 0.0023204305, -0.0000000324, 20, [1, 1]),
    "izsym_m3": (0.1890207662, 0.0041413540, -0.0000000000, -0.0000216899, 0.0042675964, -0.0000000000, 20, [1, 1]),
    "refneg": (0.8962856690, 0.0162269072, 0.0000000000, -0.0002305821, 0.0092817221, -0.0000000000, 20, [1, 1]),
    "iysym2": (0.2257865204, 0.0040877741, 0.0000000000, -0.0000000000, 0.0023381895, -0.0000000000, 16, [1, 2]),
    "afil_then_naca": (0.4670527019, 0.0174890280, -0.0000200671, 0.0007446971, -0.0615426009, -0.0000185390, 20, [1, 1]),
    "airf_then_naca": (0.3814272637, 0.0118655548, -0.0000095010, 0.0003594698, -0.0406302155, -0.0000029402, 20, [1, 1]),
    "naca_then_afil": (0.3623386847, 0.0105466719, -0.0000070005, 0.0003955102, -0.0309818012, -0.0000037671, 20, [1, 1]),
    "afil_lim_then_naca": (0.4300476398, 0.0148008318, -0.0000145187, 0.0005673098, -0.0515337213, -0.0000093914, 20, [1, 1]),
    "naca_then_missing_afil": (0.3097948458, 0.0082788725, -0.0000047473, 0.0003708159, -0.0185919548, -0.0000053383, 20, [1, 1]),
    "comp_mixed": (0.4441213168, 0.0071315642, 0.0000578414, -0.0107203270, -0.0430408636, -0.0003406023, 42, [1, 1, 3, 2]),
}
# optvl stops with "Redundant y-symmetry specifications" for YDUPLICATE 0.0 with
# IYSYM = 1 or -1 (surface and body); it ignores "AINC" ("Line ignored"), which
# the AVL source in avl/src/ainput.f accepts as ANGLE.


def _write_case(directory, name: str):
    text, _beta = CASES[name]
    path = directory / f"{name}.avl"
    path.write_text(text)
    (directory / "af.dat").write_text(_AF_DAT)
    return path


def _solve(path, beta: float) -> AVLSolver:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        solver = AVLSolver(path, base_dir=path.parent)
    solver.set_parameter("cd0", 0.0)
    solver.set_variable("alpha", ALPHA_DEG)
    solver.set_variable("beta", beta)
    s = solver.state
    s.icon[C.IVALFA, 0] = C.ICALFA
    s.conval[C.ICALFA, 0] = ALPHA_DEG
    s.icon[C.IVBETA, 0] = C.ICBETA
    s.conval[C.ICBETA, 0] = beta
    for ic, iv in ((C.ICROTX, C.IVROTX), (C.ICROTY, C.IVROTY), (C.ICROTZ, C.IVROTZ)):
        s.icon[iv, 0] = ic
        s.conval[ic, 0] = 0.0
    solver.execute_run(max_iter=20)
    return solver


@pytest.mark.parametrize("name", sorted(CASES))
def test_fixed_alpha_forces_match_fortran(tmp_path, name):
    s = _solve(_write_case(tmp_path, name), CASES[name][1]).state
    cl, cd, cy, crl, cm, cn, nstrip, lncomp = OPTVL_REF[name]
    mom = reported_totals(s)["CM"]
    assert float(s.cltot) == pytest.approx(cl, abs=TOL)
    assert float(s.cdtot) == pytest.approx(cd, abs=TOL)
    assert float(s.cytot) == pytest.approx(cy, abs=TOL)
    assert float(mom[0]) == pytest.approx(crl, abs=TOL)
    assert float(mom[1]) == pytest.approx(cm, abs=TOL)
    assert float(mom[2]) == pytest.approx(cn, abs=TOL)
    assert int(s.nstrip) == nstrip
    assert [int(v) for v in s.lncomp[: s.nsurf]] == lncomp


# --- GEO-08: redundant y-symmetry -------------------------------------------------


@pytest.mark.parametrize("sym", ["1 0 0.0", "-1 0 0.0", "3 0 0.0"])
def test_ydup_zero_with_iysym_is_rejected(sym):
    text = _hdr("redundant", sym=sym) + _WING.format(surf="", a="", b="")
    with pytest.raises(ValueError, match="Redundant y-symmetry"):
        parse_avl(text)


def test_body_ydup_zero_with_iysym_is_rejected():
    text = _hdr("redundant body", sym="1 0 0.0") + "BODY\nPod\n10 1.0\nTRANSLATE\n0 1.0 0\nYDUPLICATE\n0.0\n"
    with pytest.raises(ValueError, match="Redundant y-symmetry"):
        parse_avl(text)


def test_ydup_offset_with_iysym_is_accepted():
    """ainput.f only stops for YDUPLICATE exactly 0.0."""
    text = _hdr("offset", sym="1 0 0.0") + _WING.format(surf="", a="", b="").replace("YDUPLICATE\n0.0", "YDUPLICATE\n0.5")
    assert parse_avl(text).surfaces[0].yduplicate == 0.5


def test_api_symmetric_wing_with_iysym_is_rejected():
    aircraft = Aircraft(name="redundant", iysym=1)
    wing = aircraft.add_wing("Wing", n_chord=4, n_span=6, symmetric=True)
    wing.add_section(xyzle=[0.0, 0.0, 0.0], chord=1.0)
    wing.add_section(xyzle=[0.0, 2.0, 0.0], chord=0.6)
    with pytest.raises(ValueError, match="Redundant y-symmetry"):
        aircraft.to_avl_model()


def test_api_body_ydup_zero_with_iysym_is_rejected():
    aircraft = Aircraft(name="redundant body", iysym=-1)
    aircraft.add_body("Pod", n_body=10, translate=[0.0, 1.0, 0.0], yduplicate=0.0)
    with pytest.raises(ValueError, match="Redundant y-symmetry"):
        aircraft.to_avl_model()


# --- GEO-09: AIRFOIL block termination ---------------------------------------------


def _first_section(name: str):
    return parse_avl(CASES[name][0]).surfaces[0].sections[0]


def test_airfoil_block_stops_at_claf():
    sec = _first_section("airf_claf")
    assert len(sec.airfoil_coords) == 11
    assert sec.claf == pytest.approx(1.1)


def test_airfoil_block_skips_blank_lines_and_stops_at_one_number_line():
    """RDLINE skips blank lines; a line without two numbers ends the block."""
    blank = _COORDS.replace("0.0 0.0\n", "0.0 0.0\n\n", 1)
    text = _hdr("blank") + _WING.format(surf="", a="AIRFOIL\n" + blank + "CLAF\n1.1", b="")
    sec = parse_avl(text).surfaces[0].sections[0]
    assert len(sec.airfoil_coords) == 11
    assert sec.claf == pytest.approx(1.1)

    # optvl reads the same 6 points here and ignores the rest up to CLAF.
    one = _COORDS.replace("0.0 0.0\n", "0.0 0.0\n0.5\n", 1)
    text = _hdr("one") + _WING.format(surf="", a="AIRFOIL\n" + one + "CLAF\n1.1", b="")
    sec = parse_avl(text).surfaces[0].sections[0]
    assert len(sec.airfoil_coords) == 6
    assert sec.claf == pytest.approx(1.1)


def test_airfoil_block_followed_by_cdcl():
    text = _hdr("cdcl") + _WING.format(surf="", a="AIRFOIL\n" + _COORDS + "CDCL\n-0.5 0.02 0.3 0.008 1.2 0.03", b="")
    sec = parse_avl(text).surfaces[0].sections[0]
    assert len(sec.airfoil_coords) == 11
    assert sec.cdcl == pytest.approx([-0.5, 0.02, 0.3, 0.008, 1.2, 0.03])


# --- GEO-10: short SURFACE / SECTION lines -----------------------------------------


@pytest.mark.parametrize(
    ("line", "expected"),
    [("6 1.0", (0, 0.0)), ("6 1.0 12", (0, 0.0)), ("6 1.0 12 -1.5", (12, -1.5)), ("6 1.0 11.9995 -1.5", (12, -1.5))],
)
def test_surface_spacing_line(line, expected):
    text = _hdr("s") + f"SURFACE\nWing\n{line}\nSECTION\n0 0 0 1 0 4 1.0\nSECTION\n0 1 0 1 0 0 0\n"
    surf = parse_avl(text).surfaces[0]
    assert (surf.n_span, surf.s_space) == expected


@pytest.mark.parametrize(
    ("line", "expected"),
    [("0 0 0 1 0", (0, 0.0)), ("0 0 0 1 0 7", (0, 0.0)), ("0 0 0 1 0 7 -2.0", (7, -2.0)), ("0 0 0 1 0 6.9995 1", (7, 1.0))],
)
def test_section_spacing_line(line, expected):
    text = _hdr("s") + f"SURFACE\nWing\n6 1.0\nSECTION\n{line}\nSECTION\n0 1 0 1 0 0 0\n"
    sec = parse_avl(text).surfaces[0].sections[0]
    assert (sec.n_span, sec.s_space) == expected


# --- GEO-11 -----------------------------------------------------------------------


def test_cdcl_tie_sorted_like_ainput_and_warns():
    """Strict-comparison sort: (0.5,.03) is LMAX, the tied (0.5,.01) becomes LMID."""
    with pytest.warns(UserWarning, match="CDCL polar"):
        sec = parse_avl(CASES["cdcl_tie"][0]).surfaces[0].sections[1]
    assert sec.cdcl == pytest.approx([-0.5, 0.02, 0.5, 0.01, 0.5, 0.03])


def test_cdcl_negative_cd_kept_with_warning():
    with pytest.warns(UserWarning, match="CDCL polar"):
        sec = parse_avl(CASES["cdcl_negcd"][0]).surfaces[0].sections[1]
    assert sec.cdcl == pytest.approx([-0.5, 0.02, 0.3, -0.004, 1.2, 0.03])


def test_ainc_is_alias_for_angle():
    """ainput.f: KEYWD.EQ.'ANGL' .OR. KEYWD.EQ.'AINC' (optvl 2.5.0 ignores AINC)."""
    for kw in ("ANGLE", "AINC"):
        text = _hdr("a") + _WING.format(surf=f"{kw}\n2.5\n", a="", b="")
        assert parse_avl(text).surfaces[0].angle_deg == pytest.approx(2.5)


@pytest.mark.parametrize(("sym", "expected"), [("2 -3 0.0", (1, -1)), ("-5 7 0.0", (-1, 1)), ("0 0 0.0", (0, 0))])
def test_symmetry_flags_clamped(sym, expected):
    header = parse_avl(_hdr("s", sym=sym) + _HALF).header
    assert (header.iysym, header.izsym) == expected


def test_nonpositive_reference_values_reset_to_one():
    header = parse_avl(CASES["refneg"][0]).header
    assert (header.sref, header.cref, header.bref) == (1.0, 1.0, 1.0)


def test_last_airfoil_keyword_wins():
    surf = parse_avl(CASES["afil_then_naca"][0]).surfaces[0]
    assert (surf.sections[0].naca, surf.sections[0].airfoil_file) == ("4412", None)
    surf = parse_avl(CASES["naca_then_afil"][0]).surfaces[0]
    assert (surf.sections[0].naca, surf.sections[0].airfoil_file) == (None, "af.dat")
    sec = parse_avl(CASES["airf_then_naca"][0]).surfaces[0].sections[0]
    assert (sec.naca, sec.airfoil_coords) == ("4412", None)


def test_missing_afile_after_naca_uses_flat_plate(tmp_path):
    """ainput.f AFIL with no file: 'Using default zero-camber airfoil'."""
    model = parse_avl(CASES["naca_then_missing_afil"][0])
    with pytest.warns(UserWarning, match="flat plate"):
        prepare_model(model, base_dir=tmp_path)


# --- GEO-12: default component index ------------------------------------------------


def test_default_component_counts_ydup_images():
    model = prepare_model(parse_avl(CASES["comp_mixed"][0]))
    assert [s.component for s in model.surfaces] == [1, 3, 2]


# --- GEO-13: DESIGN -----------------------------------------------------------------


@pytest.mark.parametrize("name", ["testdes.avl", "supra.avl", "d81.avl"])
def test_design_keyword_warns_once_and_is_ignored(name):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = parse_avl_file(GEOMETRIES_DIR / name)
    design = [w for w in caught if "DESIGN" in str(w.message)]
    assert len(design) == 1
    assert issubclass(design[0].category, UserWarning)
    assert all(s.sections for s in model.surfaces)


def test_design_data_line_consumed():
    with pytest.warns(UserWarning, match="DESIGN"):
        sec = _first_section("airf_design_claf")
    assert len(sec.airfoil_coords) == 11
    assert sec.claf == pytest.approx(1.1)
