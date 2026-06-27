"""Tests for AVL-compatible reported coefficient mapping."""

from __future__ import annotations

import math

import pytest

from openavl.core.reporting import nasa_dir, reported_totals
from openavl.core.state import AVLState


def _state_with_totals(cr, cm, cn, *, alfa_deg=0.0, lnasa_sa=True):
    state = AVLState()
    state.lnasa_sa = lnasa_sa
    state.alfa = math.radians(alfa_deg)
    state.cmtot[:] = (cr, cm, cn)
    state.cftot[:] = (1.0, 2.0, 3.0)
    return state


def test_nasa_dir_default():
    state = AVLState()
    state.lnasa_sa = True
    assert nasa_dir(state) == -1.0


def test_reported_body_axis_moments_apply_dir():
    state = _state_with_totals(0.04, -0.07, -0.008)
    reported = reported_totals(state)
    assert reported["CM"] == pytest.approx([-0.04, -0.07, 0.008])
    assert reported["CF"] == pytest.approx([-1.0, 2.0, -3.0])


def test_reported_stability_axis_moments_with_sideslip():
    state = _state_with_totals(0.04, -0.07, -0.008, alfa_deg=8.0)
    reported = reported_totals(state)
    ca = math.cos(state.alfa)
    sa = math.sin(state.alfa)
    expected_cl_sa = -(0.04 * ca + (-0.008) * sa)
    expected_cn_sa = -((-0.008) * ca - 0.04 * sa)
    assert reported["CM_sa"][0] == pytest.approx(expected_cl_sa)
    assert reported["CM_sa"][1] == pytest.approx(-0.07)
    assert reported["CM_sa"][2] == pytest.approx(expected_cn_sa)
