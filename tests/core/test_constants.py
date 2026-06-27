"""Tests for openavl.constants."""

import pytest

from openavl.constants import (
    ICALFA,
    ICCL,
    IVALFA,
    IVBETA,
    NDMAX,
    NUMAX,
)


pytestmark = pytest.mark.core


def test_numax_ndmax():
    assert NUMAX == 6
    assert NDMAX == 30


def test_zero_based_variable_indices():
    assert IVALFA == 0
    assert IVBETA == 1


def test_zero_based_constraint_indices():
    assert ICALFA == 0
    assert ICCL == 5
