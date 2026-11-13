"""The GUI WebSocket accepts only its own (loopback) page, not other websites."""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from openavl.web.app import app

pytestmark = pytest.mark.ui


@pytest.mark.parametrize(
    "origin",
    ["http://evil.example", "http://127.0.0.1.evil.example:8080", "null"],
)
def test_websocket_rejects_other_origins(origin):
    with TestClient(app) as client, pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/ws", headers={"origin": origin}):
            pass


@pytest.mark.parametrize(
    "origin",
    [None, "http://127.0.0.1:8080", "http://localhost:8080", "http://[::1]:8080"],
)
def test_websocket_accepts_the_gui_page_and_scripts(origin):
    headers = {} if origin is None else {"origin": origin}
    with TestClient(app) as client, client.websocket_connect("/ws", headers=headers):
        pass
