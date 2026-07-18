"""Tests for web GUI run-case application and solve correlation."""

from __future__ import annotations

import asyncio
import json

import pytest

from openavl import constants as C
from openavl.solver import AVLSolver
from openavl.web.app import _apply_run_case, _send_solve_result, _solve_identity
from openavl.web.session import SessionState

from tests.helpers import GEOMETRIES_DIR

SUPRA_AVL = GEOMETRIES_DIR / "supra.avl"

pytestmark = pytest.mark.ui


@pytest.mark.skipif(not SUPRA_AVL.is_file(), reason="supra.avl not found")
def test_apply_run_case_replaces_inputs_and_constraints():
    """Applying a GUI case removes constraints left by the previous case."""
    solver = AVLSolver(SUPRA_AVL)
    solver.set_constraint("elevator", "cm", 0.0)
    session = SessionState(session_id="case-test", solver=solver)

    _apply_run_case(
        session,
        {
            "inputs": {"alpha": 3.5, "velocity": 22.0},
            "constraints": [{"variable": "alpha", "constraint": "cl", "value": 0.65}],
        },
    )

    elevator = solver.model.control_map["elevator"]
    assert solver.state.alfa / solver.state.dtr == pytest.approx(3.5)
    assert solver.state.parval[C.IPVEE, 0] == pytest.approx(22.0)
    assert solver.state.icon[C.IVALFA, 0] == C.ICCL
    assert solver.state.icon[C.IVTOT + elevator, 0] == C.ICTOT + elevator


def test_solve_identity_preserves_client_ids():
    """Client case and solve IDs remain stable through dispatch."""
    assert _solve_identity({"case_id": "case-2", "solve_id": "solve-7"}) == (
        "case-2",
        "solve-7",
    )


def test_send_solve_result_correlates_batch_and_completion():
    """Every solve payload and its completion marker carry matching IDs."""

    class FakeWebSocket:
        def __init__(self) -> None:
            self.messages: list[dict[str, object]] = []

        async def send_text(self, text: str) -> None:
            self.messages.append(json.loads(text))

    websocket = FakeWebSocket()
    asyncio.run(
        _send_solve_result(
            websocket,
            {"messages": [{"type": "results", "CL": 0.7}, {"type": "surface_forces"}]},
            case_id="case-2",
            solve_id="solve-7",
        )
    )

    assert [message["type"] for message in websocket.messages] == [
        "results",
        "surface_forces",
        "solve_complete",
    ]
    assert all(message["case_id"] == "case-2" for message in websocket.messages)
    assert all(message["solve_id"] == "solve-7" for message in websocket.messages)
