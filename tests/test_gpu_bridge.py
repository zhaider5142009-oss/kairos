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
  "not running" from "running and slow";
* health reports the *union* of the two routes a foundation model can reach a
  prediction by. It used to consult the WSL bridge alone, so a Chronos answering
  off the GPU in 0.6 s was displayed to the operator as "WSL offline" — a live
  capability shown as an absent one;
* the server answers on its port before its weights are loaded, because that port
  is how a supervisor knows a process is on duty;
* one prediction says which route answered *it*. `Chronos-Bolt` is the same model
  name on both routes and the GPU client serves both, so the route is stamped by
  the producer, read by the report, and pinned in the template that prints it.

The GPU path itself is deliberately not exercised here: it needs the RX 560, an
812 MB exported graph and roughly 2 GB of the 8 GB this box has, and a test that
only passes on one operator's hardware is a test the repository cannot run.
"""

from __future__ import annotations

import json
import re
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
    assert f.extras["via"] == "gpu", "an answer off the card that cannot say so"
    assert "gpu-directml" in f.detail and "batch=4" in f.detail


def test_the_desk_names_the_gpu_sidecar_state():
    """`health` is how the operator learns whether a Chronos number came off the
    card or off the fallback. Reporting neither is how a sidecar that died three
    days ago keeps looking like a slow model."""
    from kairos import command as C

    down = C._l_health({"data_online": True, "market_data": {"psx": "ok"},
                        "llm": {"provider": "x", "models": 2},
                        "gpu": {"running": False, "error": "nothing answering"},
                        "foundation": {"label": "no foundation model answering"},
                        "universe_size": 500, "uptime_s": 600})
    assert "GPU forecaster down" in down
    assert "no foundation model answering" in down
    up = C._l_health({"data_online": True, "market_data": {"psx": "ok"},
                      "llm": {"provider": "x", "models": 2},
                      "gpu": {"running": True, "backend": "onnxruntime-directml"},
                      "foundation": {"label": "chronos-bolt-base on directml"},
                      "universe_size": 500, "uptime_s": 600})
    assert "onnxruntime-directml" in up and "GPU forecaster down" not in up


# ------------------------------------------------------- the union the desk reads
def _health_probes(monkeypatch, gpu_health, wsl_ready):
    """`system_health` touches the data vendors, the LLM router and the WSL shell.

    A test of one line of it has to hold the other three still, or it spends
    thirty seconds shelling out to a distro that is not running.
    """
    from kairos.data import SourceHealth

    monkeypatch.setattr("kairos.llm.provider_status", lambda: {})
    monkeypatch.setattr("kairos.llm.dead_seats", lambda: [])
    monkeypatch.setattr("kairos.llm.council_models", lambda: [])
    src = SourceHealth(yahoo=True, tradingview=True, psx_local=False)
    monkeypatch.setattr("kairos.engine.ENGINE.md",
                        type("M", (), {"health": lambda self: src})(), raising=False)
    monkeypatch.setattr("kairos.health.WSL", type("W", (), {
        "wsl_available": staticmethod(lambda: False),
        "models_ready": staticmethod(lambda: dict(wsl_ready))})())
    monkeypatch.setattr("kairos.gpu_dml.GPU", type("G", (), {
        "health": staticmethod(lambda: gpu_health)})())
    monkeypatch.setattr("kairos.health._public_url", lambda: "")
    monkeypatch.setattr("kairos.health.universe", lambda: [])


_UP = {"ok": True, "backend": "onnxruntime-directml",
       "provider": "DmlExecutionProvider", "model": "amazon/chronos-bolt-base",
       "batches": 11}
_OFF = {"chronos": False, "timesfm": False, "qlib": False}


def test_a_gpu_chronos_is_not_reported_as_no_foundation_model(monkeypatch):
    """The defect this closes: health consulted the WSL bridge alone, so on a box
    where the bridge is stopped and the sidecar answers in 0.6 s the desk printed
    "WSL offline" and the operator had no way to know a foundation model was in
    every prediction. A capability displayed as absent is retired by whoever
    reads the display."""
    from kairos.health import system_health

    _health_probes(monkeypatch, _UP, _OFF)
    h = system_health()
    assert h["wsl"]["models_ready"] is False      # the tension the union resolves
    assert h["gpu"]["running"] is True
    f = h["foundation"]
    assert f["chronos"] is True and f["any"] is True and f["via"] == "gpu"
    assert "directml" in f["label"].lower()


def test_the_bridge_alone_still_counts(monkeypatch):
    """Union, not preference: the WSL route is not demoted by the GPU existing."""
    from kairos.health import system_health

    _health_probes(monkeypatch, None, {"chronos": True, "timesfm": False,
                                       "qlib": False})
    h = system_health()
    assert h["gpu"]["running"] is False and h["gpu"]["error"]
    assert h["foundation"]["via"] == "wsl" and h["foundation"]["chronos"] is True


def test_a_sidecar_answering_not_ok_is_not_running(monkeypatch):
    """`ok: false` with 200 is a half-built server, not a slow one. Counting it as
    running would put "chronos-bolt-base on None" on the desk for a process that
    returns errors."""
    from kairos.health import system_health

    _health_probes(monkeypatch, {"ok": False, "error": "DML out of memory"}, _OFF)
    h = system_health()
    assert h["gpu"]["running"] is False
    assert "out of memory" in h["gpu"]["error"]
    assert h["foundation"]["via"] == "none"


def test_no_foundation_model_is_said_out_loud(monkeypatch):
    """Both routes dark is the state that must be visible, because a prediction
    made without one is a different system from the one the pack describes."""
    from kairos import command as C
    from kairos.health import system_health

    _health_probes(monkeypatch, None, _OFF)
    h = system_health()
    f = h["foundation"]
    assert f["any"] is False and f["via"] == "none" and f["chronos"] is False
    assert f["label"] == "no foundation model answering"
    line = C._l_health({"data_online": True, "market_data": {"psx": "ok"},
                        "llm": {"provider": "x", "models": 2},
                        "gpu": h["gpu"], "foundation": f,
                        "universe_size": 500, "uptime_s": 600})
    assert "no foundation model answering" in line
    # the sentence that used to promise a fallback without checking for one
    assert "Chronos on the WSL bridge" not in line


def test_the_desk_reads_the_union_not_one_route():
    """A JS template that reads the bridge flag alone re-creates the defect and no
    Python test can see it, so the template itself is pinned here."""
    html = (Path(__file__).resolve().parents[1] / "kairos" / "api" /
            "ui.html").read_text(encoding="utf-8")
    assert "h.wsl.models_ready" not in html, (
        "the foundation row reads a single route again; a GPU-served Chronos "
        "will be displayed as no foundation model")
    assert "h.foundation.any" in html and "h.gpu.running" in html


# ----------------------------------------------------- what a keeper can rely on
def test_the_port_answers_before_the_weights_are_on():
    """`serve()` used to load the graph and only then bind the port, so the port
    stayed shut for the ~40 s the weights take to reach the card. A supervisor
    decides whether a process is on duty by asking that port, and an unanswered
    port reads as "nobody is bringing this up" — which starts a second 812 MB load
    on one 4 GB card. Answering must not require a loaded model."""
    from kairos.gpu_dml import GpuServer, _health_body

    code, body = _health_body(GpuServer(None))
    assert code == 200, "a loading server refused to say it was loading"
    assert body["ok"] is False and body["loading"] is True
    assert "backend" not in body, "health named a card the graph is not on yet"


def test_health_names_the_card_only_once_the_graph_is_on_it():
    from kairos import gpu_dml as G

    class _Loaded:
        batch = 16

        def health(self):
            return {"ok": True, "backend": "onnxruntime-directml", "model": "m"}

    srv = G.GpuServer(None)
    srv.f = _Loaded()
    code, body = G._health_body(srv)
    assert code == 200 and body["ok"] is True
    assert body["backend"] == "onnxruntime-directml"
    assert body["batches"] == 0, "the server's own counters are not folded in"


def test_the_batch_thread_waits_for_the_weights_then_resumes():
    """The batching thread reads `self.f.batch`, and it now starts against a server
    that has no forecaster yet. A queued request must wait for the graph, not be
    answered by a thread that crashes on it — and must not be lost."""
    import threading
    import time

    from kairos import gpu_dml as G

    srv = G.GpuServer(None)
    srv.run_forever()
    seen: list[int] = []
    srv._run_batch = lambda b: seen.append(len(b))
    box = {"done": threading.Event(), "series": [100.0, 101.0], "horizon": 2,
           "reply": None}
    srv._q.put(box)
    time.sleep(0.4)
    assert seen == [], "a batch ran without a forecaster"
    assert srv._q.qsize() == 1, "the request was taken off the queue and dropped"

    srv.f = type("F", (), {"batch": 4})()
    deadline = time.time() + 3.0
    while not seen and time.time() < deadline:
        time.sleep(0.05)
    assert seen == [1], "the thread never resumed once the graph was assigned"


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


# --------------------------------------------------- which route answered this call
def _fc(via=None, model="Chronos-Bolt"):
    from kairos.models_stat import ModelForecast

    extras = {"q10": 100.0, "q50": 105.0, "q90": 110.0}
    if via is not None:
        extras["via"] = via
    return ModelForecast(model, 1, 0.05, 5, 0.6, "quantile10=100.0 (test)",
                         extras=extras)


def _routes(monkeypatch, chronos, timesfm):
    """`foundation_forecasts` reads the two callables off the module globals, so a
    test decides what each route did without a GPU, a distro or a bar of history."""
    import kairos.engine as E

    monkeypatch.setattr(E, "GPU", type("G", (), {
        "chronos_forecast": staticmethod(chronos)}), raising=True)
    monkeypatch.setattr(E, "WSL", type("W", (), {
        "timesfm_forecast": timesfm}), raising=True)
    return E.foundation_forecasts([100.0] * 40, 5)


def test_the_prediction_names_the_route_that_answered(monkeypatch):
    """`health` says which route is up; an operator reading one prediction needs to
    know which route answered *it*, because the exported graph and the eager bridge
    are two implementations of one model name and only one of them was repaired."""
    out, report = _routes(monkeypatch, lambda s, h=5: _fc("gpu"),
                          lambda s, h=5: _fc("wsl", "TimesFM-2.5"))
    assert len(out) == 2, "a forecast the route named was not added to the battery"
    assert [(r["model"], r["state"], r["via"]) for r in report] == [
        ("Chronos", "forecast", "gpu"), ("TimesFM", "forecast", "wsl")]


def test_the_route_follows_the_producer_not_the_caller(monkeypatch):
    """The GPU client answers for Chronos on both routes: it calls the bridge itself
    when the sidecar refuses. Crediting the function the engine happened to call
    would file every one of those fall-through answers under the card."""
    _, report = _routes(monkeypatch, lambda s, h=5: _fc("wsl"),
                        lambda s, h=5: None)
    assert report[0]["via"] == "wsl", "the report named the caller, not the answer"


def test_a_forecast_that_stamped_no_route_does_not_claim_one(monkeypatch):
    """`unknown` on the desk is the honest reading of a producer that forgot the
    stamp; guessing from context is how a bridge answer spends an afternoon
    advertised as GPU work."""
    _, report = _routes(monkeypatch, lambda s, h=5: _fc(None),
                        lambda s, h=5: None)
    assert report[0]["via"] == "unknown"
    assert report[0]["state"] == "forecast", "the answer itself was still wanted"


def test_a_route_that_did_not_answer_says_it_answered_nothing(monkeypatch):
    """`via: none` beside `absent` is what stops the desk printing a route for a
    model that contributed nothing to this prediction."""
    out, report = _routes(monkeypatch, lambda s, h=5: None,
                          lambda s, h=5: None)
    assert out == []
    assert all(r["state"] == "absent" and r["via"] == "none" for r in report)
    assert len(report) == 2

    def _boom(series, horizon):
        raise ValueError("bridge returned a non-numeric point")

    _, raised = _routes(monkeypatch, _boom, lambda s, h=5: None)
    assert raised[0]["state"] == "raised" and raised[0]["via"] == "none"
    assert "ValueError" in raised[0]["why"]


def test_both_producers_stamp_their_own_route(monkeypatch):
    """The report is only as true as the stamps. Nothing type-checks these dicts,
    so each producer is pinned where it builds the forecast."""
    import kairos.wsl_bridge as W

    monkeypatch.setattr(W.WSL, "_call", lambda payload: {
        "point": 105.0, "q10": 100.0, "q50": 105.0, "q90": 110.0,
        "confidence": 0.6})
    W.WSL._cache.clear()
    assert W.WSL.chronos_forecast([100.0] * 40, horizon=5).extras["via"] == "wsl"
    assert W.WSL.timesfm_forecast([100.0] * 40, horizon=5).extras["via"] == "wsl"
    W.WSL._cache.clear()


def test_the_desk_reads_only_keys_the_foundation_report_sends(monkeypatch):
    """A renamed key in an HTML template string is an `undefined` the conditional
    swallows: the row simply stops appearing, in a file no Python test executes."""
    html = (Path(__file__).resolve().parents[1] / "kairos" / "api" /
            "ui.html").read_text(encoding="utf-8")
    block = re.search(r"if\(r\.foundation_models&&r\.foundation_models\.length\)\{"
                      r"(.*?)\n  \}\n", html, re.S)
    assert block, ("the prediction panel no longer renders the per-prediction "
                   "foundation report — the engine has been publishing it to nobody")
    read = set(re.findall(r"\b(?:r|fm)\.([A-Za-z_][A-Za-z_0-9]*)", block.group(1)))
    assert read, "the block reads no fields, so the contract below proves nothing"

    _, report = _routes(monkeypatch, lambda s, h=5: _fc("gpu"),
                        lambda s, h=5: _fc("wsl", "TimesFM-2.5"))
    sent = {"foundation_models"} | set(report[0])
    assert read <= sent, (f"the desk asks for {sorted(read - sent)}, which the "
                          "report does not send")
