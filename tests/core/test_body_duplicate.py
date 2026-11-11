"""Regression tests for body YDUPLICATE (Fortran BDUPL, GEO-05).

Reference values
----------------
The total forces below come from Fortran AVL 3.52, run through the ``optvl``
2.5.0 Python package on the byte-identical ``TWIN_AVL`` text (with
``fuseBD.dat`` alongside), at alpha = 4 deg, beta = 5 deg, qc/2V = 0.05,
CD0 = 0::

    from optvl import OVLSolver
    o = OVLSolver(geo_file="twin.avl", debug=False)
    o.set_parameter("CD0", 0.0)
    o.set_constraint("alpha", "alpha", 4.0)
    o.set_constraint("beta", "beta", 5.0)
    o.set_constraint("pitch rate", "pitch rate", 0.05)
    o.execute_run()
    o.get_total_forces()

OpenAVL agreed with the unrounded values to <= 3e-9. Without the image body
it was off by up to 1e-2 (Cm), so 1e-7 is a tight but safe tolerance.
The wing is a flat plate so airfoil-model changes do not move these numbers.
"""

from __future__ import annotations

import shutil

import numpy as np
import pytest

from openavl.solver import AVLSolver
from tests.helpers import GEOMETRIES_DIR

FUSE_DAT = GEOMETRIES_DIR / "fuseBD.dat"

TWIN_AVL = """Twin pod test
0.0
0  0  0.0
200.0  5.0  40.0
5.0  0.0  0.0
#
SURFACE
Wing
8  1.0  16  1.0
YDUPLICATE
0.0
ANGLE
2.0
SECTION
0.0  0.0  0.0  5.0  0.0
SECTION
1.0  20.0 1.0  4.0  0.0
#
BODY
Pod
16  1.0
TRANSLATE
-8.0  {ty}  -1.0
{ydup}
BFIL
fuseBD.dat
"""

REF_TWIN = {
    "CL": 0.32495712596710014,
    "CD": 0.02488999434969914,
    "CY": -0.001599123170943786,
    "Cl": -0.004580152394014514,
    "Cm": 0.2107615654582134,
    "Cn": -0.002840810970381918,
}
TOL = 1e-7


def _write_geometry(tmp_path, ty: float = 6.0, ydup: str = "YDUPLICATE\n0.0"):
    if not FUSE_DAT.is_file():
        pytest.skip(f"fuseBD.dat not found: {FUSE_DAT}")
    shutil.copy(FUSE_DAT, tmp_path / "fuseBD.dat")
    path = tmp_path / "twin.avl"
    path.write_text(TWIN_AVL.format(ty=ty, ydup=ydup))
    return path


def test_body_image_nodes_and_sizing(tmp_path):
    single = AVLSolver(_write_geometry(tmp_path, ydup="")).state
    twin = AVLSolver(_write_geometry(tmp_path)).state

    assert single.nbody == 1
    assert twin.nbody == 2
    assert twin.nlnode == 2 * single.nlnode == 32
    assert twin.nlmax >= twin.nlnode
    assert twin.lfrst.size == 2 and twin.cdbdy.size == 2
    assert list(twin.nl[:2]) == [16, 16]
    assert list(twin.lfrst[:2]) == [0, 16]

    n = int(twin.nl[0])
    orig = twin.rl[:, 0:n]
    image = twin.rl[:, 16:16 + n]
    np.testing.assert_array_equal(orig, single.rl[:, 0:n])
    np.testing.assert_array_equal(image[0], orig[0])
    np.testing.assert_array_equal(image[1], -orig[1] + 2.0 * 0.0)
    np.testing.assert_array_equal(image[2], orig[2])
    np.testing.assert_array_equal(twin.radl[16:16 + n], twin.radl[0:n])
    assert np.all(orig[1] == 6.0)


def test_body_image_about_offset_plane(tmp_path):
    state = AVLSolver(_write_geometry(tmp_path, ty=0.0, ydup="YDUPLICATE\n-3.0")).state
    assert state.nbody == 2
    np.testing.assert_array_equal(state.rl[1, 16:32], -state.rl[1, 0:16] - 6.0)


def test_body_duplicate_on_symmetry_plane_rejected(tmp_path):
    with pytest.raises(ValueError, match="symmetry plane"):
        AVLSolver(_write_geometry(tmp_path, ty=0.0))


def test_twin_body_forces_match_avl(tmp_path):
    solver = AVLSolver(_write_geometry(tmp_path), cd0=0.0)
    solver.set_constraint("alpha", "alpha", 4.0)
    solver.set_constraint("beta", "beta", 5.0)
    solver.set_constraint("qc/2v", "qc/2v", 0.05)
    solver.execute_run()
    results = solver.get_results()
    for key, ref in REF_TWIN.items():
        assert results[key] == pytest.approx(ref, abs=TOL), key
