"""Tests for openavl.parser."""

import warnings
from pathlib import Path

import pytest

from openavl.parser import parse_avl, parse_avl_file, prepare_model

from tests.helpers import GEOMETRIES_DIR

PLANE_AVL = GEOMETRIES_DIR / "plane.avl"

pytestmark = pytest.mark.core


def test_parse_inline_comments_and_afil():
    text = "\n".join(
        [
            "Test Aircraft",
            "0.11 ! mach inline comment",
            "0 0 0",
            "10 2 12",
            "0 0 0",
            "SURFACE",
            "Wing",
            "3 1 0 1",
            "SECTION",
            "0 0 0 1.2 2.0 7 1  # section span count",
            'AFIL "airfoils/wing.dat" ! file comment',
            "SECTION",
            "1 2 0 0.8 1.0 1 1",
        ]
    )
    model = parse_avl(text)
    assert model.header.mach == pytest.approx(0.11)
    assert len(model.surfaces) == 1
    assert len(model.surfaces[0].sections) == 2
    assert model.surfaces[0].sections[0].n_span == 7
    assert model.surfaces[0].sections[0].airfoil_file == "airfoils/wing.dat"
    assert model.airfoil_files == ["airfoils/wing.dat"]


def test_parse_afile_next_line_path_under_airfoils_dir():
    """Next-line AFILE paths under ``airfoils/`` must not be read as AIRF."""
    text = "\n".join(
        [
            "Test Aircraft",
            "0.0",
            "0 0 0",
            "10 2 12",
            "0 0 0",
            "SURFACE",
            "Wing",
            "3 1 0 1",
            "SECTION",
            "0 0 0 1.2 2.0 7 1",
            "AFILE",
            r"airfoils\GOE316.dat",
            "SECTION",
            "1 2 0 0.8 1.0 1 1",
            "AFILE",
            "airfoils/GOE316.dat",
        ]
    )
    model = parse_avl(text)
    assert model.surfaces[0].sections[0].airfoil_file == r"airfoils\GOE316.dat"
    assert model.surfaces[0].sections[1].airfoil_file == "airfoils/GOE316.dat"
    assert model.airfoil_files == [r"airfoils\GOE316.dat", "airfoils/GOE316.dat"]


def test_parse_plane_avl():
    assert PLANE_AVL.is_file(), f"missing test geometry: {PLANE_AVL}"
    model = parse_avl_file(PLANE_AVL)
    assert model.header.title.strip() == "Plane Vanilla"
    assert model.header.sref == pytest.approx(12.0)
    assert model.header.cref == pytest.approx(1.0)
    assert model.header.bref == pytest.approx(15.0)
    assert len(model.surfaces) == 3
    assert model.surfaces[0].name.strip() == "WING"
    assert model.surfaces[0].yduplicate == pytest.approx(0.0)
    assert model.surfaces[0].angle_deg == pytest.approx(4.0)

    prepare_model(model)
    assert set(model.control_map) == {"aileron", "elevator", "rudder"}
    assert len(model.control_map) == 3


def test_parse_cdcl_orders_points_by_lift_coefficient():
    text = "\n".join(
        [
            "CDCL Test",
            "0.0",
            "0 0 0",
            "1 1 1",
            "0 0 0",
            "SURFACE",
            "Wing",
            "2 1 1 1",
            "CDCL",
            "1.2 0.08 -0.8 0.06 0.1 0.02",
            "SECTION",
            "0 0 0 1 0",
            "SECTION",
            "1 1 0 1 0",
        ]
    )

    model = parse_avl(text)

    assert model.surfaces[0].cdcl == pytest.approx([-0.8, 0.06, 0.1, 0.02, 1.2, 0.08])


@pytest.mark.parametrize("avl_path", sorted(GEOMETRIES_DIR.glob("*.avl")))
def test_parse_all_run_geometries(avl_path: Path):
    model = parse_avl_file(avl_path)
    assert model.header.title
    prepare_model(model, base_dir=GEOMETRIES_DIR)


def test_prepare_model_warns_when_airfoil_file_missing(tmp_path: Path):
    """Missing AFIL paths should warn and fall back to flat-plate camber."""
    text = "\n".join(
        [
            "Missing Airfoil",
            "0.0",
            "0 0 0",
            "1 1 1",
            "0 0 0",
            "SURFACE",
            "Wing",
            "2 1 1 1",
            "SECTION",
            "0 0 0 1 0",
            "AFIL missing.dat",
            "SECTION",
            "1 1 0 1 0",
        ]
    )
    model = parse_avl(text)

    with pytest.warns(UserWarning, match="Airfoil file not found: .*missing\\.dat"):
        prepare_model(model, base_dir=tmp_path)

    assert model.surfaces[0].sections[0].airfoil_coords is None
    assert model.surfaces[0].sections[0].airfoil_camber is not None


_HEADER_LINES = ["Regression", "0.0", "0 0 0", "1 1 1", "0 0 0"]


@pytest.mark.parametrize(
    ("name", "expected"),
    [("allegro.avl", 0.020), ("bd.avl", 0.017), ("hers-gp10.avl", 0.0)],
)
def test_parse_cdp_header_line(name: str, expected: float):
    """GEO-03: the optional 6th header line is the baseline profile drag."""
    model = parse_avl_file(GEOMETRIES_DIR / name)
    assert model.header.cdp == pytest.approx(expected)


def test_absent_cdp_line_does_not_consume_keyword():
    """GEO-03: with no CDp line the next line is a keyword and must be kept."""
    text = "\n".join(
        _HEADER_LINES + ["SURFACE", "Wing", "2 1 1 1", "SECTION", "0 0 0 1 0", "SECTION", "1 1 0 1 0"]
    )
    model = parse_avl(text)
    assert model.header.cdp == 0.0
    assert len(model.surfaces) == 1
    assert len(model.surfaces[0].sections) == 2


def test_cdp_is_default_cd0_and_option_overrides():
    """GEO-03: CDp seeds the cd0 run parameter unless cd0= is given."""
    from openavl import constants as C
    from openavl.core.state import AVLState

    model = parse_avl_file(GEOMETRIES_DIR / "allegro.avl")
    prepare_model(model, base_dir=GEOMETRIES_DIR)
    assert AVLState.from_model(model).parval[C.IPCD0, 0] == pytest.approx(0.020)
    assert AVLState.from_model(model, cd0=0.005).parval[C.IPCD0, 0] == pytest.approx(0.005)


def test_parse_noalbe():
    """GEO-04: NOALBE clears lvalbe on the three hers-gp10 ground-plane surfaces."""
    model = parse_avl_file(GEOMETRIES_DIR / "hers-gp10.avl")
    assert [s.lvalbe for s in model.surfaces] == [True, False, False, False]

    text = "\n".join(
        _HEADER_LINES + ["SURFACE", "Wing", "2 1 1 1", "NOALBE", "SECTION", "0 0 0 1 0", "SECTION", "1 1 0 1 0"]
    )
    surf = parse_avl(text).surfaces[0]
    assert surf.lvalbe is False
    assert len(surf.sections) == 2


def test_control_keyword_with_trailing_comment_reads_next_line():
    """GEO-07: a comment after CONTROL must not become the control name."""
    text = "\n".join(
        _HEADER_LINES
        + [
            "SURFACE",
            "Wing",
            "2 1 1 1",
            "SECTION",
            "0 0 0 1 0",
            "CONTROL    ! name gain xhinge",
            "flap  1.0  0.7  0 1 0  -1",
            "SECTION",
            "1 1 0 1 0",
            "CONTROL",
            "flap",
        ]
    )
    model = parse_avl(text)
    ctrl = model.surfaces[0].sections[0].controls[0]
    assert ctrl.name == "flap"
    assert ctrl.gain == pytest.approx(1.0)
    assert ctrl.xhinge == pytest.approx(0.7)
    assert ctrl.vhinge == pytest.approx([0.0, 1.0, 0.0])
    assert ctrl.sgn_dup == pytest.approx(-1.0)
    # Fortran short-line defaults: gain 1, xhinge 0, vhinge 0, sgn_dup 1.
    short = model.surfaces[0].sections[1].controls[0]
    assert (short.name, short.gain, short.xhinge, short.vhinge, short.sgn_dup) == (
        "flap",
        1.0,
        0.0,
        [0.0, 0.0, 0.0],
        1.0,
    )


def test_control_partial_hinge_vector_defaults_to_zero():
    """GEO-07: fewer than three hinge-vector components means all zero (ainput.f)."""
    text = "\n".join(
        _HEADER_LINES
        + ["SURFACE", "Wing", "2 1 1 1", "SECTION", "0 0 0 1 0", "CONTROL", "elev 2.0 0.6 0 1"]
    )
    ctrl = parse_avl(text).surfaces[0].sections[0].controls[0]
    assert ctrl.gain == pytest.approx(2.0)
    assert ctrl.xhinge == pytest.approx(0.6)
    assert ctrl.vhinge == [0.0, 0.0, 0.0]


def test_quoted_afile_name_drops_trailing_numbers():
    """GEO-17: ``"ag38.dat" 0 1`` names ag38.dat and resolves the file."""
    model = parse_avl_file(GEOMETRIES_DIR / "al.avl")
    sec = model.surfaces[0].sections[3]
    assert sec.airfoil_file == "ag38.dat"
    assert "ag38.dat" in model.airfoil_files

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        prepare_model(model, base_dir=GEOMETRIES_DIR)
    assert sec.airfoil_coords


def test_prepare_model_warns_when_airfoil_file_has_no_base_dir():
    """Relative AFIL paths without base_dir should warn."""
    text = "\n".join(
        [
            "Missing Base Dir",
            "0.0",
            "0 0 0",
            "1 1 1",
            "0 0 0",
            "SURFACE",
            "Wing",
            "2 1 1 1",
            "SECTION",
            "0 0 0 1 0",
            "AFIL airfoils/wing.dat",
            "SECTION",
            "1 1 0 1 0",
        ]
    )
    model = parse_avl(text)

    with pytest.warns(UserWarning, match="relative path with no base_dir"):
        prepare_model(model)


def test_comma_separated_header_section_and_scale():
    """Commas separate values, as in Fortran list-directed input / GETFLT."""
    text = "\n".join(
        [
            "Comma Plane",
            "0.0",
            "0, 0, 0.0",
            "10.0, 1.0, 10.0",
            "0.25, 0.0, 0.0",
            "SURFACE",
            "Wing",
            "8, 1.0, 12, 1.0",
            "SCALE",
            "2.0, 2.0, 2.0",
            "SECTION",
            "0.0, 0.0, 0.0, 2.0, 0.0",
            "SECTION",
            "0.0, 5.0, 0.0, 1.0, 0.0",
        ]
    )
    model = parse_avl(text)
    h = model.header
    assert (h.sref, h.cref, h.bref) == (10.0, 1.0, 10.0)
    assert h.xref == pytest.approx(0.25)
    surf = model.surfaces[0]
    assert surf.n_chord == 8
    assert surf.scale == [2.0, 2.0, 2.0]
    assert len(surf.sections) == 2
    assert surf.sections[0].chord == pytest.approx(2.0)
    assert surf.sections[1].yle == pytest.approx(5.0)


def test_fortran_d_exponent_numbers():
    from openavl.fileio.parser import parse_numbers

    assert parse_numbers("0.0 0.0 0.0 1.0D0 0.0") == [0.0, 0.0, 0.0, 1.0, 0.0]
    assert parse_numbers("2.5d-3, 1D2") == [pytest.approx(2.5e-3), 100.0]
    text = "\n".join(_HEADER_LINES + ["SURFACE", "Wing", "2 1 1 1", "SECTION", "0 0 0 1.5D0 0"])
    assert parse_avl(text).surfaces[0].sections[0].chord == pytest.approx(1.5)


def test_d_and_comma_rules_leave_names_and_paths_alone():
    """Control names and file paths that look D-ish or contain commas stay text."""
    from openavl.fileio.parser import parse_numbers

    assert parse_numbers("d1 1d 1.0dx airfoil1d0.dat") == []
    text = "\n".join(
        _HEADER_LINES
        + [
            "SURFACE",
            "Wing",
            "2 1 1 1",
            "SECTION",
            "0 0 0 1 0",
            "AFILE airfoils/foil,d2.dat",
            "CONTROL",
            "d1elev 1.0D0 0.7 0 1 0 -1",
        ]
    )
    sec = parse_avl(text).surfaces[0].sections[0]
    assert sec.airfoil_file == "airfoils/foil,d2.dat"
    ctrl = sec.controls[0]
    assert (ctrl.name, ctrl.gain, ctrl.xhinge, ctrl.sgn_dup) == ("d1elev", 1.0, 0.7, -1.0)
