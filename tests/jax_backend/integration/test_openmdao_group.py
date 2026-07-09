"""OpenMDAO integration tests for OpenAVLGroup."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("openmdao")

from tests.jax_backend.require_jax import require_jax

require_jax()

import openmdao.api as om

from openavl.jax.openmdao import JaxAVLComp
from openavl.jax.openmdao_group import OpenAVLGroup

from tests.helpers import GEOMETRIES_DIR

PLANE_AVL = GEOMETRIES_DIR / "plane.avl"

COEFF_TOL = 1e-8
FD_TOL = 1e-4

pytestmark = pytest.mark.integration


def _scalar(prob: om.Problem, name: str) -> float:
    return float(np.asarray(prob.get_val(name)).item())


@pytest.mark.reference
def test_openavl_group_matches_jax_avl_comp() -> None:
    """OpenAVLGroup at baseline geometry matches JaxAVLComp force coefficients."""
    if not PLANE_AVL.is_file():
        pytest.skip(f"{PLANE_AVL.name} not found")

    prob_old = om.Problem()
    prob_old.model.add_subsystem("avl", JaxAVLComp(geo_file=str(PLANE_AVL)))
    prob_old.setup()
    prob_old.set_val("avl.alpha", np.deg2rad(5.0))
    prob_old.set_val("avl.beta", 0.0)
    prob_old.set_val("avl.mach", 0.0)
    prob_old.run_model()

    prob_new = om.Problem()
    prob_new.model = OpenAVLGroup(geo_file=str(PLANE_AVL))
    prob_new.setup()
    prob_new.set_val("alpha", 5.0)
    prob_new.set_val("beta", 0.0)
    prob_new.set_val("mach", 0.0)
    prob_new.run_model()

    assert _scalar(prob_new, "CL") == pytest.approx(_scalar(prob_old, "avl.CL"), abs=COEFF_TOL)
    assert _scalar(prob_new, "CD") == pytest.approx(_scalar(prob_old, "avl.CD"), abs=COEFF_TOL)
    assert _scalar(prob_new, "CY") == pytest.approx(_scalar(prob_old, "avl.CY"), abs=COEFF_TOL)

@pytest.mark.parametrize("avl_path", [PLANE_AVL])
@pytest.mark.reference
def test_check_totals_geometry(avl_path: Path) -> None:
    """OpenMDAO FD totals agree with JAX jacvec for geometry design variables."""
    if not avl_path.is_file():
        pytest.skip(f"{avl_path.name} not found")

    prob = om.Problem()
    prob.model = OpenAVLGroup(geo_file=str(avl_path))
    prob.setup()

    prob.set_val("alpha", 5.0)
    prob.set_val("beta", 0.0)
    prob.set_val("mach", 0.0)
    prob.run_model()

    wrt = ["WING:aincs"]
    of = ["CL", "CD", "Cm"]
    check_data = prob.check_totals(of=of, wrt=wrt, method="fd", step=1e-6, out_stream=None)

    for key, data in check_data.items():
        j_analytic = data.get("J_fwd")
        if j_analytic is None:
            j_analytic = data.get("J_rev")
        j_fd = data.get("J_fd")
        if j_analytic is None or j_fd is None:
            continue
        np.testing.assert_allclose(
            np.asarray(j_analytic),
            np.asarray(j_fd),
            atol=FD_TOL,
            rtol=FD_TOL,
            err_msg=str(key),
        )
