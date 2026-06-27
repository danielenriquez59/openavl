"""Tests for CdclPolar and geometry API CDCL integration."""

from __future__ import annotations

import pytest

from openavl.geometry import Aircraft, CdclPolar

pytestmark = pytest.mark.core


def test_cdcl_polar_named_constructor():
    polar = CdclPolar(
        cl_neg=-0.5,
        cd_neg=0.05,
        cl_min=0.3,
        cd_min=0.008,
        cl_pos=1.5,
        cd_pos=0.05,
    )
    assert polar.as_list() == pytest.approx([-0.5, 0.05, 0.3, 0.008, 1.5, 0.05])
    assert polar.is_active is True


def test_cdcl_polar_from_points_sorts_by_cl():
    polar = CdclPolar.from_points(
        (1.5, 0.05),
        (-0.5, 0.05),
        (0.3, 0.008),
    )
    assert polar.cl_neg == pytest.approx(-0.5)
    assert polar.cl_min == pytest.approx(0.3)
    assert polar.cl_pos == pytest.approx(1.5)


def test_cdcl_polar_from_list():
    polar = CdclPolar.from_list([-0.4, 0.06, 0.3, 0.010, 1.2, 0.06])
    assert polar.cd_min == pytest.approx(0.010)
    assert polar.cl_pos == pytest.approx(1.2)


def test_cdcl_polar_rejects_unordered_cl():
    with pytest.raises(ValueError, match="cl_neg"):
        CdclPolar(cl_neg=0.4, cd_neg=0.05, cl_min=0.3, cd_min=0.008, cl_pos=1.5, cd_pos=0.05)

    with pytest.raises(ValueError, match="cl_min"):
        CdclPolar(cl_neg=-0.5, cd_neg=0.05, cl_min=1.5, cd_min=0.008, cl_pos=1.5, cd_pos=0.05)


def test_cdcl_polar_rejects_negative_cd():
    with pytest.raises(ValueError, match="cd_min"):
        CdclPolar(cl_neg=-0.5, cd_neg=0.05, cl_min=0.3, cd_min=-0.001, cl_pos=1.5, cd_pos=0.05)


def test_cdcl_polar_from_list_requires_six_values():
    with pytest.raises(ValueError, match="6 values"):
        CdclPolar.from_list([0.0, 0.01, 0.2])


def test_cdcl_polar_zero_cd_min_is_inactive():
    polar = CdclPolar(cl_neg=-0.5, cd_neg=0.05, cl_min=0.3, cd_min=0.0, cl_pos=1.5, cd_pos=0.05)
    assert polar.is_active is False


def test_section_set_cdcl_polar_chains():
    aircraft = Aircraft(sref=1, cref=1, bref=1)
    wing = aircraft.add_wing("W")
    polar = CdclPolar.from_list([-0.5, 0.05, 0.3, 0.008, 1.5, 0.05])
    root = wing.add_section(chord=1.0).set_cdcl_polar(polar)
    wing.add_section(chord=0.8)

    assert root.cdcl is polar
    model = aircraft.to_avl_model()
    assert model.surfaces[0].sections[0].cdcl == pytest.approx(polar.as_list())


def test_wing_cdcl_polar_inherited_by_sections():
    aircraft = Aircraft(sref=1, cref=1, bref=1)
    wing = aircraft.add_wing("W").set_cdcl_polar(
        CdclPolar.from_list([-0.5, 0.05, 0.3, 0.008, 1.5, 0.05])
    )
    wing.add_section(chord=1.0)
    wing.add_section(chord=0.8)

    model = aircraft.to_avl_model()
    for sec in model.surfaces[0].sections:
        assert sec.cdcl == pytest.approx([-0.5, 0.05, 0.3, 0.008, 1.5, 0.05])


def test_section_cdcl_polar_overrides_wing_default():
    wing_polar = CdclPolar.from_list([-0.5, 0.05, 0.3, 0.008, 1.5, 0.05])
    sec_polar = CdclPolar.from_list([-0.4, 0.06, 0.3, 0.010, 1.2, 0.06])
    aircraft = Aircraft(sref=1, cref=1, bref=1)
    wing = aircraft.add_wing("W").set_cdcl_polar(wing_polar)
    wing.add_section(chord=1.0).set_cdcl_polar(sec_polar)
    wing.add_section(chord=0.8)

    model = aircraft.to_avl_model()
    assert model.surfaces[0].sections[0].cdcl == pytest.approx(sec_polar.as_list())
    assert model.surfaces[0].sections[1].cdcl == pytest.approx(wing_polar.as_list())
