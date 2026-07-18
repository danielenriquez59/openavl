"""Regression test for JaxAVLComp moment partials (A3).

``jac.CM.wrot``/``jac.CM.delcon`` from ``jax.jacrev(run_analysis)`` are indexed
``[CM output, input]``. Diagonal-dominant cases (e.g. pure pitch trim) don't
expose an axis swap because they barely exercise the off-diagonal terms, so
this test uses a flow condition with simultaneous roll/pitch/yaw rates and
control deflections and checks all nine ``CMx``/``CMy``/``CMz`` vs.
``pb2v``/``qc2v``/``rb2v`` partials plus the control-surface partials against
finite differences.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("openmdao")

from tests.jax_backend.require_jax import require_jax

require_jax()

import openmdao.api as om

from openavl.jax.openmdao import JaxAVLComp

from tests.helpers import GEOMETRIES_DIR

PLANE_AVL = GEOMETRIES_DIR / "plane.avl"

FD_TOL = 2e-4

pytestmark = pytest.mark.integration


@pytest.mark.reference
def test_jax_avl_comp_moment_partials_asymmetric_case():
    """CM partials on an asymmetric flow condition match finite differences."""
    if not PLANE_AVL.is_file():
        pytest.skip(f"{PLANE_AVL.name} not found")

    prob = om.Problem()
    prob.model.add_subsystem("avl", JaxAVLComp(geo_file=str(PLANE_AVL)))
    prob.setup()
    prob.set_val("avl.alpha", np.deg2rad(4.0))
    prob.set_val("avl.beta", np.deg2rad(3.0))
    prob.set_val("avl.pb2v", 0.02)
    prob.set_val("avl.qc2v", 0.01)
    prob.set_val("avl.rb2v", -0.015)
    prob.set_val("avl.mach", 0.0)
    prob.set_val("avl.delcon_0", np.deg2rad(2.0))
    prob.set_val("avl.delcon_1", np.deg2rad(-1.5))
    prob.set_val("avl.delcon_2", np.deg2rad(1.0))
    prob.run_model()

    data = prob.check_partials(method="fd", step=1e-6, out_stream=None)
    comp_data = data["avl"]

    checked = 0
    for (of, wrt), meta in comp_data.items():
        if of not in ("CMx", "CMy", "CMz"):
            continue
        j_fwd = np.asarray(meta["J_fwd"])
        j_fd = np.asarray(meta["J_fd"])
        np.testing.assert_allclose(
            j_fwd, j_fd, atol=FD_TOL, rtol=FD_TOL, err_msg=f"d({of})/d({wrt})"
        )
        checked += 1

    assert checked > 0
