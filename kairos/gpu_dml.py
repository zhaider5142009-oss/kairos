"""GPU foundation-model bridge: Chronos-Bolt forecasts computed on the local AMD GPU.

torch on this box is CPU-only (no CUDA, no ROCm, no torch-directml), so the only way to get
KAIROS' foundation-model forecast onto the RX 560 is ONNX Runtime with the DirectML provider.
The graph runs the whole expensive half — the input MLP, the T5 encoder, the 12-layer decoder
and the quantile head. Only instance norm and patching stay in eager torch on the CPU, which
also sidesteps two exporter bugs: torch's legacy tracer mis-traces Patch.unfold and dynamo
dies on T5's head view.

Measured on this box (i5-6400 / 8 GB / RX 560, chronos-bolt-base fp32, batch of 16 rows of
real closes, 2026-10-23, reproduce with `python tools/gpu_verify.py`):
    graph on DirectML   0.589 s per batch
    eager torch on CPU  2.865 s per batch        (so 4.9x, and the card is not the bottleneck
                                                   the CPU was — it is the half that ran twice)
    CPU-side prep       0.01 s per batch
    mean relative error 1.19e-07 vs the eager torch path (max 1.34e-06) — float32 noise, so the
    forecast the desk shows is the forecast the CPU would have produced, not an approximation
    of it.

Server (resident, so weights load once and batches can be filled):
    python -m kairos.gpu_dml serve [--port 8611] [--model base] [--batch 16]

Engine use: `GPU.chronos_forecast()` has the same contract as `WSL.chronos_forecast()` and falls
back to the WSL bridge whenever the GPU server is not reachable, so nothing changes when it is down.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np

MODEL_DIR = Path(__file__).resolve().parent.parent / "gpu_models"
DEFAULT_PORT = 8611
CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache"


# --------------------------------------------------------------------------- split model
def _preprocess(model, ctx: np.ndarray):
    """The only work that stays on the CPU: instance norm and patching.

    This mirrors the first half of ChronosBoltModelForForecasting.encode() statement for
    statement but stops before the input MLP. encode() as written also runs the 12-layer T5
    encoder and returns its output, which is exactly the work the GPU graph is here to do —
    calling it as-is meant the CPU ran the encoder and the GPU ran it again. Measured on this
    box (`tools/gpu_verify.py`, batch of 16): that duplicated encoder pass costs 1.99 s per
    batch, and what is left here costs 0.01 s. The input MLP goes into the graph with the
    encoder because it consumes the same patched tensor and is on the way to every layer below.
    """
    import torch

    t = torch.from_numpy(np.ascontiguousarray(ctx, dtype=np.float32))
    cfg = model.chronos_config
    with torch.no_grad():
        mask = torch.isnan(t).logical_not().to(t.dtype)
        if t.shape[-1] > cfg.context_length:
            t = t[..., -cfg.context_length:]
            mask = mask[..., -cfg.context_length:]

        context, loc_scale = model.instance_norm(t)
        context = context.to(model.dtype)
        mask = mask.to(model.dtype)

        patched_context = model.patch(context)
        patched_mask = torch.nan_to_num(model.patch(mask), nan=0.0)
        patched_context = torch.where(patched_mask > 0.0, patched_context, 0.0)
        patched_context = torch.cat([patched_context, patched_mask], dim=-1)
        attention_mask = patched_mask.sum(dim=-1) > 0
        if cfg.use_reg_token:
            attention_mask = torch.cat(
                [attention_mask.to(model.dtype),
                 torch.ones_like(attention_mask).to(model.dtype)[..., :1]], dim=-1)
    return patched_context, attention_mask, loc_scale


class _Tail:
    """input MLP + encoder + decoder + quantile head — everything that costs compute.

    Output stays in instance-norm space; model.forward() unscales afterwards, so the caller must
    apply instance_norm.inverse() to get forecasts in real units.
    """

    def __init__(self, model, batch: int) -> None:
        import torch

        self.model = model
        self.batch = int(batch)

        class Inner(torch.nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m
                self.reg_embed = m.shared.weight[m.config.reg_token_id].detach().clone()

            def forward(self, patched, attention_mask):
                ie = self.m.input_patch_embedding(patched)
                if self.m.chronos_config.use_reg_token:
                    ie = torch.cat([ie, self.reg_embed.expand(ie.shape[0], 1, -1).to(ie.dtype)],
                                   dim=-2)
                hidden = self.m.encoder(attention_mask=attention_mask, inputs_embeds=ie)[0]
                seq = self.m.decode(ie, attention_mask, hidden)
                nq = self.m.num_quantiles
                pl = self.m.chronos_config.prediction_length
                return self.m.output_patch_embedding(seq).view(ie.shape[0], nq, pl)

        self.module = Inner(model)

    def export(self, path: Path) -> Path:
        """Trace at the model's full context length, then fix the [REG]-token float indices."""
        import torch
        import onnx
        from onnx import TensorProto, numpy_helper

        cfg = self.model.chronos_config
        rng = np.random.RandomState(0)
        walk = 100 + np.cumsum(rng.randn(cfg.context_length // 5).astype(np.float32))
        ctx = np.concatenate([np.full(cfg.context_length - walk.size, np.nan, np.float32), walk])
        patched, am, _ls = _preprocess(self.model, np.stack([ctx] * self.batch, 0))
        torch.onnx.export(self.module, (patched, am), str(path),
                          input_names=["patched_context", "attention_mask"],
                          output_names=["quantile_preds"], opset_version=17,
                          do_constant_folding=True, dynamo=False)

        m = onnx.load(str(path))
        index_ops = {"Gather": (1,), "GatherElements": (1,), "GatherND": (1,),
                     "Reshape": (1,), "Expand": (1,), "Slice": (1, 2, 3, 4),
                     "Squeeze": (1,), "Unsqueeze": (1,), "Split": (1,), "Pad": (1,), "Tile": (1,)}
        consts = {}
        for n in m.graph.node:
            if n.op_type == "Constant":
                for a in n.attribute:
                    if a.name == "value" and a.t.data_type == TensorProto.FLOAT:
                        for out in n.output:
                            consts[out] = n
        fixed = 0
        for node in m.graph.node:
            for pos in index_ops.get(node.op_type, ()):
                if pos >= len(node.input) or node.input[pos] not in consts:
                    continue
                src = consts[node.input[pos]]
                arr = None
                for a in src.attribute:
                    if a.name == "value":
                        arr = numpy_helper.to_array(a.t)
                if arr is None or arr.size and np.any(arr != arr.astype(np.int64)):
                    continue
                new = numpy_helper.from_array(arr.astype(np.int64))
                for i, a in enumerate(src.attribute):
                    if a.name == "value":
                        src.attribute[i].t.CopyFrom(new)
                del consts[node.input[pos]]
                fixed += 1
        if fixed:
            out = path.with_name(path.stem + "_fixed.onnx")
            onnx.save(m, str(out))
            return out
        return path


class GpuForecaster:
    """Holds the DirectML session and answers batches of [context_length] series."""

    def __init__(self, model: str = "base", batch: int = 16) -> None:
        import torch
        from chronos import ChronosBoltPipeline

        self.repo = "amazon/chronos-bolt-%s" % model
        self.batch = int(batch)
        self._lock = threading.Lock()
        self.timing = {"preprocess": 0.0, "dml": 0.0, "batches": 0}
        MODEL_DIR.mkdir(parents=True, exist_ok=True)

        self.pipeline = ChronosBoltPipeline.from_pretrained(
            self.repo, torch_dtype=torch.float32, local_files_only=True)
        self.model = self.pipeline.model
        self.model.eval()
        self.ctx_len = int(self.model.chronos_config.context_length)
        self.pred_len = int(self.model.chronos_config.prediction_length)
        self.quantiles = list(self.model.chronos_config.quantiles)

        import onnxruntime as ort

        raw = MODEL_DIR / ("chronos_graph_%s%d.onnx" % (model, self.batch))
        self.onnx_path = self._build_onnx(raw)
        so = ort.SessionOptions()
        so.log_severity_level = 3
        self.session = ort.InferenceSession(
            str(self.onnx_path), sess_options=so,
            providers=["DmlExecutionProvider", "CPUExecutionProvider"])
        self.provider = self.session.get_providers()[0]

    def _build_onnx(self, raw: Path) -> Path:
        fixed = raw.with_name(raw.stem + "_fixed.onnx")
        if fixed.exists():
            return fixed
        if raw.exists():
            return raw
        out = _Tail(self.model, self.batch).export(raw)
        if out != raw:
            raw.unlink(missing_ok=True)
        return out

    # ------------------------------------------------------------------ forecast
    def forecast(self, contexts: list[np.ndarray]) -> np.ndarray:
        """contexts: raw price series (any length <= context_length). -> [n, nq, 64] real units."""
        pad = self._pad_contexts(contexts)
        with self._lock:
            t0 = time.time()
            patched, attention_mask, loc_scale = _preprocess(self.model, pad)
            t1 = time.time()
            q = self.session.run(
                None, {"patched_context": patched.numpy(),
                       "attention_mask": attention_mask.numpy()})[0]
            t2 = time.time()
            out = self._unscale(np.asarray(q, dtype=np.float32), loc_scale)[:len(contexts)]
        self.timing["preprocess"] += t1 - t0
        self.timing["dml"] += t2 - t1
        self.timing["batches"] += 1
        return out

    def _pad_contexts(self, contexts: list[np.ndarray]) -> np.ndarray:
        """Left-pad with NaN to the model's context length — same convention as predict_quantiles,
        and required because the exported graph has a static token count."""
        rows = []
        for c in contexts:
            arr = np.asarray(c, dtype=np.float32).ravel()
            arr = arr[-self.ctx_len:]
            if arr.size < self.ctx_len:
                arr = np.concatenate([np.full(self.ctx_len - arr.size, np.nan, np.float32), arr])
            rows.append(arr)
        while len(rows) < self.batch:
            rows.append(rows[0])
        return np.stack(rows[:max(self.batch, len(rows))], 0)

    def _unscale(self, q: np.ndarray, loc_scale) -> np.ndarray:
        import torch

        n = q.shape[0]
        t = torch.from_numpy(np.ascontiguousarray(q, dtype=np.float32)).view(n, -1)
        out = self.model.instance_norm.inverse(t, loc_scale)
        return out.view(n, self.model.num_quantiles, self.pred_len).numpy()

    def health(self) -> dict[str, Any]:
        return {"ok": True, "backend": "onnxruntime-directml", "provider": self.provider,
                "model": self.repo, "batch": self.batch, "onnx": str(self.onnx_path),
                "prediction_length": self.pred_len, "quantiles": self.quantiles,
                "timing": {k: (round(v, 2) if isinstance(v, float) else v)
                           for k, v in self.timing.items()}}


# --------------------------------------------------------------------------- server
class GpuServer:
    """One GPU at a time, but requests are pooled into a full batch so the card stays fed."""

    def __init__(self, forecaster: "GpuForecaster | None" = None,
                 batch_wait: float = 0.12) -> None:
        self.f = forecaster
        self.batch_wait = batch_wait
        self._q: queue.Queue = queue.Queue()
        self.stats = {"batches": 0, "series": 0, "gpu_seconds": 0.0}

    def submit(self, series: list[float], horizon: int) -> dict[str, Any]:
        box: dict[str, Any] = {"done": threading.Event(), "series": series,
                               "horizon": horizon, "reply": None}
        self._q.put(box)
        box["done"].wait(timeout=300)
        return box["reply"]

    def run_forever(self) -> None:
        t = threading.Thread(target=self._loop, daemon=True)
        t.start()

    def _loop(self) -> None:
        while self.f is None:
            time.sleep(0.2)
        while True:
            batch = [self._q.get()]
            deadline = time.time() + self.batch_wait
            while len(batch) < self.f.batch:
                left = deadline - time.time()
                if left <= 0:
                    break
                try:
                    batch.append(self._q.get(timeout=left))
                except queue.Empty:
                    break
            self._run_batch(batch)

    def _run_batch(self, batch: list[dict[str, Any]]) -> None:
        t0 = time.time()
        try:
            out = self.f.forecast([b["series"] for b in batch])
        except Exception as exc:  # never leave a caller hanging on a GPU fault
            for b in batch:
                b["reply"] = {"ok": False, "error": "%s: %s" % (type(exc).__name__, str(exc)[:200])}
                b["done"].set()
            return
        el = time.time() - t0
        self.stats["batches"] += 1
        self.stats["series"] += len(batch)
        self.stats["gpu_seconds"] += el
        for j, b in enumerate(batch):
            h = min(int(b["horizon"]), self.f.pred_len)
            idx = {q: self.f.quantiles.index(q) for q in (0.1, 0.5, 0.9)}
            q10_row, med_row, q90_row = out[j][idx[0.1]], out[j][idx[0.5]], out[j][idx[0.9]]
            path = [float(x) for x in med_row[:h]]
            last = float(b["series"][-1]) or 1.0
            point = path[-1] if path else last
            q10, q90 = float(q10_row[h - 1]), float(q90_row[h - 1])
            q50 = float(med_row[h - 1])
            sigma = max(abs(q90 - q10) / 2.0, 1e-9)
            conf = max(0.2, min(0.9, 0.5 + 0.25 * abs(q50 - last) / sigma))
            b["reply"] = {"ok": True, "model": "chronos", "point": point, "path": path,
                          "q10": q10, "q50": q50, "q90": q90, "sigma": sigma,
                          "confidence": conf, "backend": "gpu-directml",
                          "batch": len(batch), "seconds": round(el, 3)}
            b["done"].set()


def _health_body(srv: "GpuServer") -> tuple[int, dict[str, Any]]:
    """What `GET /health` answers, split out so the contract a keeper depends on is
    testable without a GPU.

    An unbound port used to be the only answer a watcher got while the weights
    loaded, which is indistinguishable from "nobody is bringing this up" — and the
    weights take ~40 s. Answering on the port from the first instant, with `ok`
    false until the graph is on the card, makes the port mean *a keeper is on duty*
    and the body mean *it can forecast yet*.
    """
    f = srv.f
    if f is None:
        return 200, {"ok": False, "loading": True,
                     "error": "forecaster still loading its graph"}
    return 200, dict(f.health(), **srv.stats)


def serve(port: int = DEFAULT_PORT, model: str = "base", batch: int = 16) -> None:
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class H(BaseHTTPRequestHandler):
        def _send(self, code: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            if self.path.startswith("/health"):
                code, payload = _health_body(srv)
                self._send(code, payload)
            else:
                self._send(404, {"ok": False, "error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            if not self.path.startswith("/forecast"):
                self._send(404, {"ok": False, "error": "not found"})
                return
            if srv.f is None:
                self._send(503, {"ok": False, "error": "forecaster loading"})
                return
            try:
                req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                series = [float(x) for x in req["series"]]
                reply = srv.submit(series, int(req.get("horizon", 5)))
            except Exception as exc:
                self._send(400, {"ok": False, "error": "%s: %s"
                                          % (type(exc).__name__, str(exc)[:200])})
                return
            self._send(200 if reply.get("ok") else 500, reply)

        def log_message(self, *a) -> None:
            pass

    # Bind first, load second: see _health_body. The socket is the keeper's only
    # signal that a process is on duty, and it used to appear 40 s after the
    # process did.
    srv = GpuServer(None)
    httpd = ThreadingHTTPServer(("127.0.0.1", port), H)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    print("kairos gpu forecaster: listening on :%d, loading %s" % (port, model),
          flush=True)
    f = GpuForecaster(model=model, batch=batch)
    srv.f = f
    srv.run_forever()
    print("kairos gpu forecaster: %s on %s (batch %d) at http://127.0.0.1:%d"
          % (f.repo, f.provider, f.batch, port), flush=True)
    threading.Event().wait()


# --------------------------------------------------------------------------- client
_CLIENT = {"down_until": 0.0}


def chronos_forecast(close_series: list[float], horizon: int = 5,
                     port: int = DEFAULT_PORT):
    """Drop-in for WSL.chronos_forecast: GPU when the server answers, WSL bridge when it doesn't."""
    from .models_stat import ModelForecast, _direction

    now = time.time()
    if now < _CLIENT["down_until"]:
        return _wsl_fallback(close_series, horizon)
    try:
        req = json.dumps({"series": [float(x) for x in close_series],
                          "horizon": int(horizon)}).encode()
        with urllib.request.urlopen(urllib.request.Request(
                "http://127.0.0.1:%d/forecast" % port, data=req,
                headers={"Content-Type": "application/json"}), timeout=120) as r:
            out = json.loads(r.read())
        if not out.get("ok"):
            raise RuntimeError(str(out.get("error"))[:120])
    except Exception:
        _CLIENT["down_until"] = now + 60.0
        return _wsl_fallback(close_series, horizon)

    point = float(out.get("point") or 0.0)
    last = float(close_series[-1]) or 1.0
    exp_ret = point / last - 1
    return ModelForecast("Chronos-Bolt", _direction(exp_ret), exp_ret, horizon,
                         float(out.get("confidence", 0.5)),
                         "quantile10=%s median=%s quantile90=%s (%s, batch=%s in %ss)"
                         % (out.get("q10"), out.get("q50"), out.get("q90"),
                            out.get("backend"), out.get("batch"), out.get("seconds")),
                         extras={"q10": out.get("q10"), "q50": out.get("q50"),
                                 "q90": out.get("q90"), "gpu": True, "via": "gpu"})


def _wsl_fallback(close_series: list[float], horizon: int = 5):
    from .wsl_bridge import WSL
    return WSL.chronos_forecast(close_series, horizon)


class _GpuClient:
    """Namespace used by engine.py: same call shape as the WSL bridge."""
    chronos_forecast = staticmethod(chronos_forecast)

    @staticmethod
    def health(port: int = DEFAULT_PORT) -> dict[str, Any] | None:
        try:
            with urllib.request.urlopen("http://127.0.0.1:%d/health" % port, timeout=3) as r:
                return json.loads(r.read())
        except Exception:
            return None


GPU = _GpuClient()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(prog="kairos.gpu_dml")
    ap.add_argument("cmd", nargs="?", default="serve")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--model", default=os.environ.get("KAIROS_GPU_MODEL", "base"))
    ap.add_argument("--batch", type=int, default=int(os.environ.get("KAIROS_GPU_BATCH", "16")))
    a = ap.parse_args()
    if a.cmd == "serve":
        serve(a.port, a.model, a.batch)
    elif a.cmd == "health":
        print(json.dumps(GPU.health(a.port) or {"ok": False, "error": "gpu server not running"}))
    else:
        ap.print_help()
