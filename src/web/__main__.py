"""Launch the OpenAVL web GUI in the default browser."""

from __future__ import annotations

import threading
import time
import webbrowser


def main() -> None:
    """Start uvicorn and open the web GUI in the system browser."""
    import uvicorn

    host = "127.0.0.1"
    port = 8000
    url = f"http://{host}:{port}"

    def _open_browser() -> None:
        time.sleep(1.0)
        webbrowser.open(url)

    threading.Thread(target=_open_browser, daemon=True).start()
    uvicorn.run("openavl.web.app:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
