"""Roundtrip tests: Geometry API -> AVLModel vs parsed .avl files."""

from __future__ import annotations

import pytest

from openavl.geometry import Aircraft
from openavl.parser import parse_avl_file, prepare_model

from tests.helpers import GEOMETRIES_DIR

PLANE_AVL = GEOMETRIES_DIR / "plane.avl"

pytestmark = pytest.mark.core


def build_plane_aircraft() -> Aircraft:
    """Build a programmatic equivalent of tests/data/avl/geometries/plane.avl."""
    aircraft = Aircraft(
        name="Plane Vanilla",
        mach=0.0,
        sref=12.0,
        cref=1.0,
        bref=15.0,
    )

    wing = aircraft.add_wing(
        "WING",
        n_chord=1,
        c_space=1.0,
        n_span=16,
        s_space=-2.0,
        symmetric=True,
        angle=4.0,
    )
    root = wing.add_section(xyzle=[-0.25, 0.0, 0.0], chord=1.0, n_span=8, s_space=1.0)
    root.add_control("aileron", gain=1.0, xhinge=0.0, vhinge=[0.0, 0.0, 0.0], sgn_dup=-1.0)
    tip = wing.add_section(xyzle=[-0.175, 7.5, 0.5], chord=0.7, s_space=0.0)
    tip.add_control("aileron", gain=1.0, xhinge=0.0, vhinge=[0.0, 0.0, 0.0], sgn_dup=-1.0)

    stab = aircraft.add_wing(
        "STAB",
        n_chord=1,
        c_space=1.0,
        n_span=7,
        s_space=-2.0,
        symmetric=True,
        translate=[6.0, 0.0, 0.5],
    )
    stab_root = stab.add_section(xyzle=[-0.1, 0.0, 0.0], chord=0.4, n_span=7, s_space=-1.25)
    stab_root.add_control("elevator", gain=1.0, xhinge=0.0, vhinge=[0.0, 0.0, 0.0], sgn_dup=1.0)
    stab_tip = stab.add_section(xyzle=[-0.075, 2.0, 0.0], chord=0.3, s_space=0.0)
    stab_tip.add_control("elevator", gain=1.0, xhinge=0.0, vhinge=[0.0, 0.0, 0.0], sgn_dup=1.0)

    fin = aircraft.add_wing(
        "FIN",
        n_chord=1,
        c_space=1.0,
        n_span=10,
        s_space=1.0,
        translate=[6.0, 0.0, 0.5],
    )
    fin_root = fin.add_section(xyzle=[-0.1, 0.0, 0.0], chord=0.4, n_span=7, s_space=-1.25)
    fin_root.add_control("rudder", gain=1.0, xhinge=0.0, vhinge=[0.0, 0.0, 0.0], sgn_dup=1.0)
    fin_tip = fin.add_section(xyzle=[-0.075, 0.0, 1.0], chord=0.3, s_space=0.0)
    fin_tip.add_control("rudder", gain=1.0, xhinge=0.0, vhinge=[0.0, 0.0, 0.0], sgn_dup=1.0)

    return aircraft


def _assert_headers_match(a, b) -> None:
    assert a.title.strip() == b.title.strip()
    assert a.mach == pytest.approx(b.mach)
    assert a.iysym == b.iysym
    assert a.izsym == b.izsym
    assert a.zsym == pytest.approx(b.zsym)
    assert a.sref == pytest.approx(b.sref)
    assert a.cref == pytest.approx(b.cref)
    assert a.bref == pytest.approx(b.bref)
    assert a.xref == pytest.approx(b.xref)
    assert a.yref == pytest.approx(b.yref)
    assert a.zref == pytest.approx(b.zref)


def _assert_surfaces_match(api_model, file_model) -> None:
    assert len(api_model.surfaces) == len(file_model.surfaces)
    for api_surf, file_surf in zip(api_model.surfaces, file_model.surfaces):
        assert api_surf.name.strip() == file_surf.name.strip()
        assert api_surf.n_chord == file_surf.n_chord
        assert api_surf.c_space == pytest.approx(file_surf.c_space)
        assert api_surf.n_span == file_surf.n_span
        assert api_surf.s_space == pytest.approx(file_surf.s_space)
        assert api_surf.yduplicate == file_surf.yduplicate
        assert api_surf.angle_deg == pytest.approx(file_surf.angle_deg)
        assert api_surf.translate == pytest.approx(file_surf.translate)
        assert api_surf.scale == pytest.approx(file_surf.scale)
        assert len(api_surf.sections) == len(file_surf.sections)

        for api_sec, file_sec in zip(api_surf.sections, file_surf.sections):
            assert api_sec.xle == pytest.approx(file_sec.xle)
            assert api_sec.yle == pytest.approx(file_sec.yle)
            assert api_sec.zle == pytest.approx(file_sec.zle)
            assert api_sec.chord == pytest.approx(file_sec.chord)
            assert api_sec.ainc == pytest.approx(file_sec.ainc)
            assert api_sec.n_span == file_sec.n_span
            assert api_sec.s_space == pytest.approx(file_sec.s_space)
            assert len(api_sec.controls) == len(file_sec.controls)
            for api_ctrl, file_ctrl in zip(api_sec.controls, file_sec.controls):
                assert api_ctrl.name == file_ctrl.name
                assert api_ctrl.gain == pytest.approx(file_ctrl.gain)
                assert api_ctrl.xhinge == pytest.approx(file_ctrl.xhinge)
                assert api_ctrl.vhinge == pytest.approx(file_ctrl.vhinge)
                assert api_ctrl.sgn_dup == pytest.approx(file_ctrl.sgn_dup)


def test_plane_convert_roundtrip():
    assert PLANE_AVL.is_file(), f"missing test geometry: {PLANE_AVL}"
    file_model = prepare_model(parse_avl_file(PLANE_AVL), base_dir=PLANE_AVL.parent)
    api_model = build_plane_aircraft().to_avl_model()

    _assert_headers_match(api_model.header, file_model.header)
    _assert_surfaces_match(api_model, file_model)
    assert api_model.control_map == file_model.control_map
