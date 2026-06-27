"""Tests for openavl.parser."""

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
