"""Launch the OpenAVL web GUI in the default browser."""

from __future__ import annotations

import os
import threading
import time
import webbrowser


def main() -> None:
    """Start uvicorn and open the web GUI in the system browser.

    Honors ``HOST`` and ``PORT`` environment variables (Render sets ``PORT``).
    Defaults to ``127.0.0.1:8000`` for local use; binds to ``0.0.0.0`` when
    ``PORT`` is set and ``HOST`` is unset.
    """
    import uvicorn

    port = int(os.environ.get("PORT", "8000"))
    host = os.environ.get("HOST")
    if host is None:
        host = "0.0.0.0" if "PORT" in os.environ else "127.0.0.1"

    if host in ("127.0.0.1", "localhost"):
        url = f"http://{host}:{port}"

        def _open_browser() -> None:
            time.sleep(1.0)
            webbrowser.open(url)

        threading.Thread(target=_open_browser, daemon=True).start()

    uvicorn.run("openavl.web.app:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
