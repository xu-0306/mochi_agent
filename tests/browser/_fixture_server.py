from __future__ import annotations

import argparse
import json
import signal
import threading
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

_NORMAL_COLOUR = "#2563eb"
_MISMATCH_COLOUR = "#b91c1c"


class _FixtureRequestHandler(BaseHTTPRequestHandler):
    server_version = "MochiBrowserFixture/1.0"

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        parsed = urlparse(self.path)
        if parsed.path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
            return
        if parsed.path not in {"/", "/fixture"}:
            self.send_error(404, "fixture route not found")
            return

        mode = parse_qs(parsed.query).get("mode", ["normal"])[0]
        if mode not in {"normal", "mismatch"}:
            self.send_error(400, "mode must be normal or mismatch")
            return

        colour = _MISMATCH_COLOUR if mode == "mismatch" else _NORMAL_COLOUR
        body = (
            "<!doctype html><html><head><meta charset=\"utf-8\">"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
            f"<title>Browser harness fixture: {mode}</title>"
            "<style>html,body{box-sizing:border-box;height:100%;margin:0;}"
            f"html,body{{background:{colour};}}"
            "</style></head>"
            f"<body data-fixture-state=\"{mode}\"></body></html>"
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, _format: str, *_args: object) -> None:
        """Keep the fixture deterministic and leave reporting to its caller."""


@dataclass
class BrowserFixtureServer:
    host: str = "127.0.0.1"
    port: int = 0
    _server: ThreadingHTTPServer | None = field(default=None, init=False, repr=False)
    _thread: threading.Thread | None = field(default=None, init=False, repr=False)

    @property
    def url(self) -> str:
        if self._server is None:
            raise RuntimeError("browser fixture server is not running")
        return f"http://{self.host}:{self._server.server_port}"

    def start(self) -> BrowserFixtureServer:
        if self._server is not None:
            return self
        self._server = ThreadingHTTPServer((self.host, self.port), _FixtureRequestHandler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        if self._server is None:
            return
        self._server.shutdown()
        self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._thread = None
        self._server = None

    def __enter__(self) -> BrowserFixtureServer:
        return self.start()

    def __exit__(self, _exc_type: object, _exc: object, _traceback: object) -> None:
        self.stop()


def start_fixture_server(*, host: str = "127.0.0.1", port: int = 0) -> BrowserFixtureServer:
    return BrowserFixtureServer(host=host, port=port).start()


def _write_ready_file(path: Path, server: BrowserFixtureServer) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"url": server.url, "schema_version": "1.0"}
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _main() -> int:
    parser = argparse.ArgumentParser(description="Serve the deterministic Mochi browser fixture.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--ready-file", type=Path, required=True)
    args = parser.parse_args()

    stop_requested = threading.Event()

    def _request_stop(_signum: int, _frame: object) -> None:
        stop_requested.set()

    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)

    with start_fixture_server(host=args.host, port=args.port) as server:
        _write_ready_file(args.ready_file, server)
        while not stop_requested.wait(0.2):
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
