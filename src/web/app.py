"""FastAPI application for the OpenAVL web GUI."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from contextlib import asynccontextmanager
from functools import partial
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

from openavl.web import session as session_mod
from openavl.web.session import (
    EXAMPLES,
    SessionState,
    add_airfoil_dependency,
    add_body_dependency,
    create_session,
    load_uploaded_model,
    mark_disconnected,
    rebuild_solver_from_pending,
    run_solve,
    touch_session,
    upload_airfoil,
    upload_body,
)

logger = logging.getLogger(__name__)

_STATIC_DIR = Path(__file__).resolve().parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start background session cleanup on app startup."""
    cleanup_task = asyncio.create_task(session_mod.cleanup_sessions())
    yield
    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass


app = FastAPI(title="OpenAVL Web GUI", lifespan=lifespan)


@app.get("/api/examples")
async def list_examples() -> list[dict[str, str]]:
    """Return built-in aircraft examples available to the frontend."""
    return [{"name": name, "label": str(spec["label"])} for name, spec in EXAMPLES.items()]


async def _send_json(websocket: WebSocket, payload: dict[str, Any]) -> None:
    """Serialize and send one JSON WebSocket message."""
    await websocket.send_text(json.dumps(payload))


async def _run_in_executor(func, *args, **kwargs) -> Any:
    """Run blocking solver work off the event loop."""
    loop = asyncio.get_running_loop()
    bound = partial(func, *args, **kwargs)
    return await loop.run_in_executor(None, bound)


def _solve_identity(data: dict[str, Any]) -> tuple[str | None, str]:
    """Return the optional case ID and a stable ID for one solve response batch."""
    case_id = data.get("case_id")
    return (str(case_id) if case_id is not None else None, str(data.get("solve_id") or uuid.uuid4()))


async def _send_solve_result(
    websocket: WebSocket,
    result: dict[str, Any],
    *,
    case_id: str | None,
    solve_id: str,
) -> None:
    """Send one correlated solve response batch followed by its completion marker."""
    for message in result["messages"]:
        correlated = {**message, "case_id": case_id, "solve_id": solve_id}
        await _send_json(websocket, correlated)
    await _send_json(
        websocket,
        {"type": "solve_complete", "case_id": case_id, "solve_id": solve_id},
    )


def _apply_run_case(session: SessionState, data: dict[str, Any]) -> None:
    """Apply a complete GUI run case to the active solver without executing it."""
    if session.solver is None:
        raise RuntimeError("No model loaded.")

    raw_inputs = data.get("inputs", {})
    raw_constraints = data.get("constraints", [])
    if not isinstance(raw_inputs, dict) or not isinstance(raw_constraints, list):
        raise ValueError("Run case inputs and constraints must be an object and an array.")

    constraints: list[tuple[str, str, float]] = []
    for row in raw_constraints:
        if not isinstance(row, dict):
            raise ValueError("Each run case constraint must be an object.")
        constraints.append(
            (
                str(row.get("variable", "")),
                str(row.get("constraint", "")),
                float(row.get("value", 0.0)),
            )
        )
    session.solver.replace_constraints(constraints)
    for key, raw_value in raw_inputs.items():
        try:
            session.solver.set_parameter(str(key), float(raw_value))
        except KeyError:
            session.solver.set_variable(str(key), float(raw_value))


async def _handle_message(session: SessionState, websocket: WebSocket, data: dict[str, Any]) -> None:
    """Dispatch one client WebSocket message."""
    touch_session(session)
    msg_type = data.get("type")

    try:
        if msg_type == "load_example":
            name = str(data.get("name", ""))
            case_id, solve_id = _solve_identity(data)
            await _send_json(
                websocket,
                {"type": "solve_started", "case_id": case_id, "solve_id": solve_id},
            )
            result = await _run_in_executor(session_mod.load_example, session, name)
            await _send_solve_result(
                websocket,
                result,
                case_id=case_id,
                solve_id=solve_id,
            )
            return

        if msg_type == "upload_avl":
            session.pending_avl_text = str(data.get("text", ""))
            session.airfoil_base_dir = None
            result = await _run_in_executor(rebuild_solver_from_pending, session)
            await _send_json(websocket, result)
            return

        if msg_type == "load_avl":
            session.pending_avl_text = str(data.get("text", ""))
            session.airfoil_base_dir = None
            case_id, solve_id = _solve_identity(data)
            await _send_json(
                websocket,
                {"type": "solve_started", "case_id": case_id, "solve_id": solve_id},
            )
            result = await _run_in_executor(load_uploaded_model, session)
            await _send_solve_result(
                websocket,
                result,
                case_id=case_id,
                solve_id=solve_id,
            )
            return

        if msg_type == "upload_airfoil":
            path = str(data.get("path", ""))
            text = str(data.get("text", ""))
            result = await _run_in_executor(upload_airfoil, session, path, text)
            await _send_json(websocket, result)
            return

        if msg_type == "add_airfoil_dependency":
            path = str(data.get("path", ""))
            result = await _run_in_executor(add_airfoil_dependency, session, path)
            await _send_json(websocket, result)
            return

        if msg_type == "upload_body":
            path = str(data.get("path", ""))
            text = str(data.get("text", ""))
            result = await _run_in_executor(upload_body, session, path, text)
            await _send_json(websocket, result)
            return

        if msg_type == "add_body_dependency":
            path = str(data.get("path", ""))
            result = await _run_in_executor(add_body_dependency, session, path)
            await _send_json(websocket, result)
            return

        if msg_type == "upload_mass":
            session.pending_mass_text = str(data.get("text", ""))
            if session.pending_avl_text:
                result = await _run_in_executor(rebuild_solver_from_pending, session)
                await _send_json(websocket, result)
            return

        if msg_type == "set_flight_param":
            if session.solver is None:
                raise RuntimeError("No model loaded.")
            key = str(data.get("key", ""))
            value = float(data.get("value", 0.0))
            session.solver.set_parameter(key, value)
            return

        if msg_type == "set_constraint":
            if session.solver is None:
                raise RuntimeError("No model loaded.")
            variable = str(data.get("variable", ""))
            constraint = str(data.get("constraint", ""))
            value = float(data.get("value", 0.0))
            session.solver.set_constraint(variable, constraint, value)
            return

        if msg_type == "apply_run_case":
            _apply_run_case(session, data)
            return

        if msg_type == "solve":
            if session.solver is None:
                raise RuntimeError("No model loaded.")
            case_id, solve_id = _solve_identity(data)
            await _send_json(
                websocket,
                {"type": "solve_started", "case_id": case_id, "solve_id": solve_id},
            )
            async with session.lock:
                result = await _run_in_executor(run_solve, session)
            await _send_solve_result(
                websocket,
                result,
                case_id=case_id,
                solve_id=solve_id,
            )
            return

        await _send_json(websocket, {"type": "error", "message": f"Unknown message type: {msg_type}"})

    except Exception as exc:
        logger.exception("WebSocket handler error")
        error: dict[str, Any] = {"type": "error", "message": str(exc)}
        if data.get("solve_id") is not None:
            error["solve_id"] = str(data["solve_id"])
            case_id = data.get("case_id")
            error["case_id"] = str(case_id) if case_id is not None else None
        await _send_json(websocket, error)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """Stateful WebSocket endpoint with per-connection solver isolation."""
    await websocket.accept()
    session = create_session()
    touch_session(session)

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                await _send_json(websocket, {"type": "error", "message": "Invalid JSON message."})
                continue
            if not isinstance(data, dict):
                await _send_json(websocket, {"type": "error", "message": "Message must be a JSON object."})
                continue
            await _handle_message(session, websocket, data)
    except WebSocketDisconnect:
        mark_disconnected(session)
    except Exception:
        mark_disconnected(session)
        raise


_STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/", StaticFiles(directory=_STATIC_DIR, html=True), name="static")
