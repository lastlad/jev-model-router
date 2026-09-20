"""A stand-in for the Jev endpoint with keyword-driven answers, for tests that run the real proxy.

Levels: "prove" -> 3, "thanks" -> 0, otherwise 1. "wrong" -> quality complaint. Run it as a module
(`python -m tests.integration.fake_jev PORT`) or via the `fake_jev` fixture.
"""

from __future__ import annotations

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer


def answers(state: dict) -> dict:
    text = json.dumps(state).lower()
    level = 3 if "prove" in text else 0 if "thanks" in text else 1
    probs = {str(i): (0.85 if i == level else 0.05) for i in range(4)}
    turns = state["history"]["turn_count"]
    return {
        "required_tier": {
            "type": "score",
            "score": float(level),
            "confidence": 0.85,
            "legend": {str(i): "" for i in range(4)},
            "probabilities": probs,
        },
        "task_type": {
            "type": "choice",
            "choice": "code",
            "confidence": 0.9,
            "probabilities": {"code": 0.9, "other": 0.1},
        },
        "continues_task": {"type": "noul", "noul": 0.9 if turns > 1 else 0.1},
        "needs_history": {"type": "noul", "noul": 0.8 if turns > 1 else 0.1},
        "quality_complaint": {"type": "noul", "noul": 0.95 if "wrong" in text else 0.02},
        "expected_output": {
            "type": "score",
            "score": 1.0,
            "confidence": 0.8,
            "legend": {"0": "", "1": "", "2": ""},
            "probabilities": {"0": 0.1, "1": 0.8, "2": 0.1},
        },
    }


class Handler(BaseHTTPRequestHandler):
    requests: list[dict] = []

    def do_POST(self) -> None:  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        Handler.requests.append(body["state"])
        out = json.dumps(
            {"model": "jev-fake", "usage": {"input_tokens": 100, "output_tokens": 0}, "answers": answers(body["state"])}
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *a: object) -> None:
        pass


def serve(port: int) -> HTTPServer:
    server = HTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


if __name__ == "__main__":
    HTTPServer(("127.0.0.1", int(sys.argv[1]) if len(sys.argv) > 1 else 4010), Handler).serve_forever()
