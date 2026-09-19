import json
from http.server import BaseHTTPRequestHandler, HTTPServer


def answers(state: dict) -> dict:
    text = json.dumps(state).lower()
    level = 3 if "prove" in text else 0 if "thanks" in text else 1
    probs = {str(i): (0.85 if i == level else 0.05) for i in range(4)}
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
        "continues_task": {"type": "noul", "noul": 0.9 if state["history"]["turn_count"] > 1 else 0.1},
        "needs_history": {"type": "noul", "noul": 0.8 if state["history"]["turn_count"] > 1 else 0.1},
        "quality_complaint": {"type": "noul", "noul": 0.95 if "wrong" in text else 0.02},
        "expected_output": {
            "type": "score",
            "score": 1.0,
            "confidence": 0.8,
            "legend": {"0": "", "1": "", "2": ""},
            "probabilities": {"0": 0.1, "1": 0.8, "2": 0.1},
        },
    }


class H(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        with open("fake_jev_requests.jsonl", "a") as f:
            f.write(json.dumps(body["state"]) + "\n")
        out = json.dumps(
            {
                "model": "jev-1.13.0",
                "usage": {"input_tokens": 100, "output_tokens": 0},
                "answers": answers(body["state"]),
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *a):
        pass


HTTPServer(("127.0.0.1", 4010), H).serve_forever()
