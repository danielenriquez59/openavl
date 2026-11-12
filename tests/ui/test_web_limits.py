"""Web-server guards: lattice size limit, trim convergence flag, strict JSON."""

from __future__ import annotations

import json
import warnings

import pytest

from openavl.web.app import _dumps
from openavl.web.session import (
    WEB_MAX_VORTICES,
    SessionState,
    load_example,
    load_uploaded_model,
    rebuild_solver_from_pending,
    run_solve,
)

pytestmark = pytest.mark.ui

FIN_ONLY_AVL = (
    "Fin\n0.0\n0 0 0\n1.0 1.0 1.0\n0 0 0\nSURFACE\nFin\n4 1.0 6 1.0\n"
    "SECTION\n0 0 0 1 0\nSECTION\n0 0 1 1 0\n"
)


def test_oversized_upload_rejected():
    """An uploaded lattice above the web limit (but below C.NVMAX) is refused."""
    session = SessionState(session_id="test")
    # 41 x 40 = 1640 vortices
    session.pending_avl_text = FIN_ONLY_AVL.replace("4 1.0 6 1.0", "41 1.0 40 1.0")
    with pytest.raises(ValueError, match=rf"1640 vortices.*limit of {WEB_MAX_VORTICES}"):
        rebuild_solver_from_pending(session)


def test_unconverged_trim_flagged_in_results():
    """A trim the Newton loop cannot meet reaches the client with converged=False."""
    session = SessionState(session_id="test")
    load_example(session, "supra")
    session.solver.replace_constraints([("alpha", "cl", 25.0), ("elevator", "cm", 0.0)])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = run_solve(session)
    results = next(m for m in out["messages"] if m["type"] == "results")
    assert results["converged"] is False


def test_nan_payloads_serialize_as_strict_json():
    """Fin-only model yields NaN xnp/sm; every message must still be valid JSON."""
    session = SessionState(session_id="test")
    session.pending_avl_text = FIN_ONLY_AVL
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = load_uploaded_model(session)
    for message in out["messages"]:
        decoded = json.loads(_dumps(message))
        json.dumps(decoded, allow_nan=False)
    results = json.loads(_dumps(next(m for m in out["messages"] if m["type"] == "results")))
    assert results["xnp"] is None
