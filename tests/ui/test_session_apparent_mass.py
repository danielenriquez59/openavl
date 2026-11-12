"""The GUI's mass loading includes apparent mass, like the script API.

``web.session._create_solver`` (used after a mass upload or an AVL edit)
skipped ``appget``, so apparent mass and inertia were zero and the GUI's
eigenmodes differed from ``AVLSolver(..., mass_file=...)`` by up to ~18% on
Supra.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from openavl.fileio.mass import parse_mass_file
from openavl.fileio.parser import parse_avl_file, prepare_model
from openavl.core.solver import AVLSolver
from openavl.web import session as web

from tests.helpers import GEOMETRIES_DIR

SUPRA_AVL = GEOMETRIES_DIR / "supra.avl"
SUPRA_MASS = GEOMETRIES_DIR.parent / "mass" / "supra.mass"

pytestmark = pytest.mark.ui


def _gui_solver():
    model = prepare_model(parse_avl_file(SUPRA_AVL), base_dir=SUPRA_AVL.parent)
    return web._create_solver(model, mass_props=parse_mass_file(SUPRA_MASS), **web._DEFAULT_FLIGHT_PARAMS)


def _api_solver():
    return AVLSolver(SUPRA_AVL, mass_file=SUPRA_MASS, **web._DEFAULT_FLIGHT_PARAMS)


def _eigenvalues(solver):
    web._apply_default_trim(solver)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        solver.execute_run()
    modes = solver.eigenvalues().modes
    return np.array(sorted((complex(m.eigenvalue) for m in modes), key=lambda z: (round(z.real, 8), z.imag)))


def test_gui_mass_path_sets_apparent_mass_like_the_api():
    gui, api = _gui_solver(), _api_solver()
    assert np.abs(api.state.amass).max() > 0.0, "fixture should have non-zero apparent mass"
    np.testing.assert_allclose(gui.state.amass, api.state.amass, rtol=1e-12, atol=0)
    np.testing.assert_allclose(gui.state.ainer, api.state.ainer, rtol=1e-12, atol=0)


def test_gui_eigenvalues_match_the_api():
    np.testing.assert_allclose(_eigenvalues(_gui_solver()), _eigenvalues(_api_solver()), rtol=1e-9, atol=1e-12)
