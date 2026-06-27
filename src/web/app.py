"""FastAPI application for the OpenAVL web GUI."""

from __future__ import annotations

import asyncio
import json
import logging
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
    create_session,
    load_uploaded_model,
    mark_disconnected,
    rebuild_solver_from_pending,
    run_solve,
    touch_session,
    upload_airfoil,
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


async def _send_messages(websocket: WebSocket, messages: list[dict[str, Any]]) -> None:
    """Send a list of protocol messages to the client."""
    for message in messages:
        await _send_json(websocket, message)


async def _run_in_executor(func, *args, **kwargs) -> Any:
    """Run blocking solver work off the event loop."""
    loop = asyncio.get_running_loop()
    bound = partial(func, *args, **kwargs)
    return await loop.run_in_executor(None, bound)


async def _handle_message(session: SessionState, websocket: WebSocket, data: dict[str, Any]) -> None:
    """Dispatch one client WebSocket message."""
    touch_session(session)
    msg_type = data.get("type")

    try:
        if msg_type == "load_example":
            name = str(data.get("name", ""))
            await _send_json(websocket, {"type": "solve_started"})
            result = await _run_in_executor(session_mod.load_example, session, name)
            await _send_messages(websocket, result["messages"])
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
            await _send_json(websocket, {"type": "solve_started"})
            result = await _run_in_executor(load_uploaded_model, session)
            await _send_messages(websocket, result["messages"])
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

        if msg_type == "solve":
            if session.solver is None:
                raise RuntimeError("No model loaded.")
            await _send_json(websocket, {"type": "solve_started"})
            async with session.lock:
                result = await _run_in_executor(run_solve, session)
            await _send_messages(websocket, result["messages"])
            return

        await _send_json(websocket, {"type": "error", "message": f"Unknown message type: {msg_type}"})

    except Exception as exc:
        logger.exception("WebSocket handler error")
        await _send_json(websocket, {"type": "error", "message": str(exc)})


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
