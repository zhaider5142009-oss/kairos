"""The GPU bridge must be an upgrade that is invisible when it is absent.

`engine.py` imports `kairos.gpu_dml` at module scope, so every claim this file
tests is a claim about the daemon's survival:

* importing the bridge pulls in neither torch nor onnxruntime — a box with no
  DirectML wheel must still start the engine;
* a forecast with no server listening falls through to the WSL bridge rather
  than raising into `PredictionEngine`, which would take the prediction with it;
* one refusal opens a circuit so a whole watchlist costs one refused socket,
  not one per instrument — and the circuit expires, because a sidecar that
  starts five minutes later must be used;
* `health()` of a dead server is `None`, not an exception, so the desk can tell
  "not running" from "running and slow".

The GPU path itself is deliberately not exercised here: it needs the RX 560, an
812 MB exported graph and roughly 2 GB of the 8 GB this box has, and a test that
only passes on one operator's hardware is a test the repository cannot run.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.fixture
def gpu():
    from kairos import gpu_dml
    return gpu_dml


@pytest.fixture(autouse=True)
def _fresh_circuit(gpu):
    """The breaker is module state shared with the live daemon; without this the
    tests decide each other's outcome by file order."""
    gpu._CLIENT["down_until"] = 0.0
    yield
    gpu._CLIENT["down_until"] = 0.0


def test_importing_the_bridge_pulls_in_no_torch():
    """`engine.py` imports this at module scope. A heavy top-level import would
    make the missing onnxruntime-directml wheel fatal for every user."""
    code = ("import sys; import kairos.gpu_dml;"
            "bad=[m for m in ('torch','onnxruntime') if m in sys.modules];"
            "print(bad)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, cwd=str(Path(__file__).resolve().parents[1]))
    assert out.returncode == 0, out.stderr[-400:]
    assert out.stdout.strip() == "[]", (
        f"the bridge imports {out.stdout.strip()} at module scope, so a box "
        "without the GPU wheel cannot start the engine")


def test_a_missing_server_falls_back_instead_of_raising(gpu, monkeypatch):
    """The engine calls this for every watchlist name on a live loop. A refused
    socket must become the old answer, not an exception at the call site."""
    seen = {}

    def fake_wsl(series, horizon=5):
        seen["args"] = (len(series), horizon)
        return "from-wsl"

    import kairos.wsl_bridge as W
    monkeypatch.setattr(W.WSL, "chronos_forecast", fake_wsl)
    gpu._CLIENT["down_until"] = 0.0
    out = gpu.chronos_forecast([1.0, 2.0, 3.0], horizon=4, port=1)
    assert out == "from-wsl"
    assert seen["args"] == (3, 4)


def test_one_refusal_closes_the_circuit_for_the_rest_of_the_batch(gpu, monkeypatch):
    """Without the breaker, 10 names on the watchlist pay 10 refused sockets per
    cycle. The breaker is the difference between an absent sidecar costing
    milliseconds and costing the loop its slot."""
    tries = []

    import urllib.request as U

    def counted(*a, **k):
        tries.append(1)
        raise OSError("connection refused")

    import kairos.wsl_bridge as W
    monkeypatch.setattr(W.WSL, "chronos_forecast", lambda s, h=5: "from-wsl")
    monkeypatch.setattr(U, "urlopen", counted)
    gpu._CLIENT["down_until"] = 0.0

    for _ in range(5):
        assert gpu.chronos_forecast([1.0, 2.0], port=1) == "from-wsl"
    assert len(tries) == 1, (
        f"the circuit never closed: {len(tries)} sockets attempted for a sidecar "
        "already known to be down")

    # and it must reopen, or a sidecar started after this cycle is never used
    gpu._CLIENT["down_until"] = 0.0
    assert gpu.chronos_forecast([1.0, 2.0], port=1) == "from-wsl"
    assert len(tries) == 2


def test_health_of_a_dead_server_is_none_not_a_trace(gpu):
    assert gpu.GPU.health(port=1) is None


def test_the_server_reply_shape_is_what_the_client_presents(gpu, monkeypatch):
    """The reply is assembled by `GpuServer._run_batch` and read back by
    `chronos_forecast` two processes apart — nothing type-checks the JSON in
    between, so a renamed key is a silent `point: 0.0`, i.e. a forecast that the
    next bar is flat. Drive both ends with the same dict."""
    import kairos.wsl_bridge as W

    reply = {"ok": True, "point": 105.0, "q10": 100.0, "q50": 105.0, "q90": 110.0,
             "confidence": 0.71, "backend": "gpu-directml", "batch": 4,
             "seconds": 0.12, "path": [101.0, 102.0, 103.0, 104.0, 105.0]}

    class _Resp:
        def read(self):
            return json.dumps(reply).encode()

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    import urllib.request as U
    monkeypatch.setattr(U, "urlopen", lambda *a, **k: _Resp())
    monkeypatch.setattr(W.WSL, "chronos_forecast",
                        lambda s, h=5: pytest.fail("the GPU answer was not taken"))
    gpu._CLIENT["down_until"] = 0.0

    f = gpu.chronos_forecast([100.0] * 40, horizon=5, port=1)
    assert f.model == "Chronos-Bolt"
    assert f.direction == 1
    assert f.expected_return == pytest.approx(0.05)
    assert f.confidence == pytest.approx(0.71)
    assert f.extras["gpu"] is True
    assert "gpu-directml" in f.detail and "batch=4" in f.detail


def test_the_desk_names_the_gpu_sidecar_state():
    """`health` is how the operator learns whether a Chronos number came off the
    card or off the fallback. Reporting neither is how a sidecar that died three
    days ago keeps looking like a slow model."""
    from kairos import command as C

    down = C._l_health({"data_online": True, "market_data": {"psx": "ok"},
                        "llm": {"provider": "x", "models": 2}, "gpu": None,
                        "universe_size": 500, "uptime_s": 600})
    assert "GPU forecaster not running" in down
    up = C._l_health({"data_online": True, "market_data": {"psx": "ok"},
                      "llm": {"provider": "x", "models": 2},
                      "gpu": {"provider": "DmlExecutionProvider"},
                      "universe_size": 500, "uptime_s": 600})
    assert "DmlExecutionProvider" in up and "not running" not in up


def test_a_gpu_answer_that_is_not_ok_falls_back(gpu, monkeypatch):
    """`ok: false` comes back as HTTP 500, but a proxy or a half-built server can
    return it with 200 — reading `point` off it would predict a price of zero."""
    import urllib.request as U
    import kairos.wsl_bridge as W

    class _Resp:
        def read(self):
            return json.dumps({"ok": False, "error": "DML out of memory"}).encode()

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(U, "urlopen", lambda *a, **k: _Resp())
    monkeypatch.setattr(W.WSL, "chronos_forecast", lambda s, h=5: "from-wsl")
    gpu._CLIENT["down_until"] = 0.0
    assert gpu.chronos_forecast([100.0, 101.0], port=1) == "from-wsl"
