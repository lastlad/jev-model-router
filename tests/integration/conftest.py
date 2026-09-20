"""Boot a fake Jev endpoint and a real LiteLLM proxy (mock deployments, no provider keys) once per session."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest

from . import fake_jev

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
MASTER_KEY = "sk-smoke"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="session")
def jev_port() -> Iterator[int]:
    port = _free_port()
    server = fake_jev.serve(port)
    yield port
    server.shutdown()


@pytest.fixture(scope="session")
def decisions_log(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("proxy") / "decisions.jsonl"


@pytest.fixture(scope="session")
def proxy(jev_port: int, decisions_log: Path) -> Iterator[str]:
    port = _free_port()
    env = {
        **os.environ,
        "TYPESAFE_API_KEY": "fake",
        "TYPESAFE_BASE_URL": f"http://127.0.0.1:{jev_port}",
        "JEV_ROUTER_CONFIG": str(HERE / "routers"),
        "JEV_ROUTER_LOG_FILE": str(decisions_log),
        "PYTHONPATH": str(ROOT),
    }
    log = decisions_log.parent / "litellm.log"
    with open(log, "w") as out:
        proc = subprocess.Popen(
            [str(Path(sys.executable).parent / "litellm"), "--config", str(HERE / "proxy.yaml"), "--port", str(port)],
            env=env,
            stdout=out,
            stderr=subprocess.STDOUT,
            cwd=ROOT,
        )
    base = f"http://127.0.0.1:{port}"
    try:
        for _ in range(120):
            if proc.poll() is not None:
                raise RuntimeError(f"litellm exited early:\n{log.read_text()[-4000:]}")
            try:
                if httpx.get(f"{base}/health/liveliness", timeout=1).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            time.sleep(0.5)
        else:
            raise RuntimeError(f"litellm did not come up:\n{log.read_text()[-4000:]}")
        yield base
    finally:
        proc.terminate()
        try:
            proc.wait(10)
        except subprocess.TimeoutExpired:
            proc.kill()


@pytest.fixture
def client(proxy: str) -> Iterator[httpx.Client]:
    with httpx.Client(base_url=proxy, headers={"Authorization": f"Bearer {MASTER_KEY}"}, timeout=60) as c:
        yield c


@pytest.fixture
def decisions(decisions_log: Path):
    """Return a callable that yields decision events logged since the fixture was created."""
    start = decisions_log.stat().st_size if decisions_log.exists() else 0

    def read(event: str = "decision", at_least: int = 1, timeout_s: float = 3.0) -> list[dict]:
        """Events of `event` kind logged since the fixture was created, polling briefly for async ones."""
        deadline = time.time() + timeout_s
        while True:
            rows: list[dict] = []
            if decisions_log.exists():
                with open(decisions_log) as f:
                    f.seek(start)
                    rows = [json.loads(line) for line in f.read().splitlines() if line.strip()]
            found = [r for r in rows if r.get("event") == event]
            if len(found) >= at_least or time.time() > deadline:
                return found
            time.sleep(0.1)

    return read
