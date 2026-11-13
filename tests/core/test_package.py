"""The public package works without its optional extras."""

from __future__ import annotations

import subprocess
import sys

import pytest

pytestmark = pytest.mark.core


def test_star_import_works_without_jax():
    """``from openavl import *`` must not need the optional JAX extra."""
    code = "import sys; sys.modules['jax'] = None; from openavl import *; AVLSolver"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
