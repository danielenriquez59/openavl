"""Unit tests for the Geometry API builder classes."""

from __future__ import annotations

import pytest

from openavl.geometry import Aircraft, AirfoilType

pytestmark = pytest.mark.core


def test_aircraft_builder_pattern():
    aircraft = Aircraft(name="My Plane", sref=9.0, cref=0.9, bref=10.0)
    wing = aircraft.add_wing("Wing", n_chord=8, n_span=20, s_space=-1.0, symmetric=True)
    root = wing.add_section(xyzle=[0, 0, 0], chord=1.0)
    root.set_airfoil_naca("2412").add_control("aileron", gain=1.0, xhinge=0.75, sgn_dup=-1.0)
    tip = wing.add_section(xyzle=[0, 5, 0], chord=0.6)
    tip.set_airfoil_naca("0012")

    assert aircraft.name == "My Plane"
    assert len(aircraft.wings) == 1
    assert wing.yduplicate == pytest.approx(0.0)
    assert len(wing.sections) == 2
    assert root.airfoil is not None
    assert root.airfoil.af_type == AirfoilType.NACA
    assert root.airfoil.naca == "2412"
    assert len(root.controls) == 1
    assert root.controls[0].name == "aileron"
    assert root.controls[0].sgn_dup == pytest.approx(-1.0)
    assert root.xyzle == pytest.approx([0.0, 0.0, 0.0])
    assert tip.xyzle == pytest.approx([0.0, 5.0, 0.0])


def test_symmetric_sugar():
    wing = Aircraft(sref=1, cref=1, bref=1).add_wing("W", symmetric=True)
    assert wing.yduplicate == pytest.approx(0.0)

    wing2 = Aircraft(sref=1, cref=1, bref=1).add_wing("W", yduplicate=1.5)
    assert wing2.yduplicate == pytest.approx(1.5)
    assert wing2.yduplicate != 0.0


def test_section_airfoil_setters_chain():
    wing = Aircraft(sref=1, cref=1, bref=1).add_wing("W")
    sec = wing.add_section(chord=1.0)
    wing.add_section(chord=0.5)

    sec.set_airfoil_file("airfoils/wing.dat")
    assert sec.airfoil.af_type == AirfoilType.FILE
    assert sec.airfoil.file_path == "airfoils/wing.dat"

    sec.set_airfoil_coords([[0.0, 0.0], [1.0, 0.01]])
    assert sec.airfoil.af_type == AirfoilType.COORDS
    assert len(sec.airfoil.coords) == 2


def test_body_builder():
    aircraft = Aircraft(sref=1, cref=1, bref=1)
    body = aircraft.add_body("Fuselage", n_body=12, b_space=-1.5, body_file="body.dat")
    assert len(aircraft.bodies) == 1
    assert body.name == "Fuselage"
    assert body.n_body == 12
    assert body.body_file == "body.dat"


def test_validation_rejects_bad_refs():
    aircraft = Aircraft(sref=0, cref=1, bref=1)
    aircraft.add_wing("W")
    aircraft.wings[0].add_section(chord=1.0)
    aircraft.wings[0].add_section(chord=1.0)
    with pytest.raises(ValueError, match="sref"):
        aircraft.to_avl_model()


def test_validation_requires_two_sections():
    aircraft = Aircraft(sref=1, cref=1, bref=1)
    aircraft.add_wing("W").add_section(chord=1.0)
    with pytest.raises(ValueError, match="at least 2 sections"):
        aircraft.to_avl_model()


def test_validation_rejects_zero_chord():
    aircraft = Aircraft(sref=1, cref=1, bref=1)
    wing = aircraft.add_wing("W")
    wing.add_section(chord=1.0)
    wing.add_section(chord=0.0)
    with pytest.raises(ValueError, match="chord"):
        aircraft.to_avl_model()
