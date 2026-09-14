"""
Hardware acceleration layer for Jarvis Reborn.

Routes ONNX model inference to the machine's GPU via the DirectML execution
provider (works on any DX12 GPU — including AMD Polaris such as the RX 560).
Every run has an automatic CPU fallback with per-provider statistics, so the
system degrades gracefully instead of crashing when a GPU op fails.

Everything in this module is optional at import time: if onnxruntime is not
present, all functions report "unavailable" rather than raising.

Quick check:  python -m jarvis_predict.compute
"""

from __future__ import annotations

import logging
import time
from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import Any, Sequence

import numpy as np

LOGGER = logging.getLogger("jarvis.compute")

_ORT = None
try:
    import onnxruntime as _ORT  # type: ignore
except Exception as _e:  # pragma: no cover
    LOGGER.debug("onnxruntime not importable: %s", _e)


# ---------------------------------------------------------------------------
# Capability probing
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def provider_state() -> dict[str, Any]:
    """Snapshots which execution providers this runtime/system offers."""
    provs: list[str] = []
    ver = "not-importable"
    if _ORT is not None:
        try:
            provs = list(_ORT.get_available_providers())
            ver = _ORT.__version__
        except Exception as e:  # pragma: no cover
            LOGGER.warning("ORT provider probe failed: %s", e)
    dml_ok = "DmlExecutionProvider" in provs
    return {
        "onnxruntime": ver,
        "providers": provs,
        "dml": dml_ok,
        "backend": "DirectML" if dml_ok else "CPU",
    }


def _dxgi_adapters() -> list[tuple[str, int]]:
    """Enumerate DXGI adapters → [(description, dedicated_vram_bytes)].
    Uses WMI/CIM via PowerShell as the most reliable cross-Win10/11 path.
    Cached after the first call so the subprocess cost is paid only once."""
    import subprocess
    try:
        ps = (
            'Get-CimInstance Win32_VideoController '
            '| Select-Object Name, AdapterRAM '
            '| ConvertTo-Json -Compress'
        )
        r = subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps],
            capture_output=True, text=True, timeout=5
        )
        import json
        data = json.loads(r.stdout.strip() or "[]")
        if isinstance(data, dict):
            data = [data]
        return [(d.get("Name", "?"), int(d.get("AdapterRAM") or 0)) for d in data]
    except Exception as e:
        LOGGER.debug("WMI GPU probe failed: %s", e)
        return []


def gpu_info_simple() -> dict[str, Any]:
    """GPU description + dedicated VRAM (bytes). Best-effort + cached."""
    @lru_cache(maxsize=1)
    def _probe() -> dict[str, Any]:
        addrs = _dxgi_adapters()
        if not addrs:
            return {"found": False}
        addrs.sort(key=lambda t: t[1], reverse=True)
        name, vram = addrs[0]
        return {
            "found": True, "name": name,
            "dedicated_vram_bytes": vram,
            "dedicated_vram_gb": round(vram / (1024 ** 3), 2),
        }
    return _probe()


def detect(run_benchmark: bool = False) -> dict[str, Any]:
    """Full capability report. Optionally micro-benchmarks the DML path."""
    p = provider_state()
    rep: dict[str, Any] = {
        "onnxruntime": p["onnxruntime"],
        "providers": p["providers"],
        "dml": p["dml"],
        "backend": p["backend"],
        "gpu": gpu_info_simple(),
        "message": (
            "GPU acceleration ENABLED (DirectML)"
            if p["dml"] else
            "GPU acceleration UNAVAILABLE - CPU only"
        ),
    }
    if run_benchmark:
        rep["benchmark"] = _benchmark()
    return rep


def _matmul_graph() -> None:
    """Return a tiny (A×B)->C ONNX graph; requires the `onnx` package."""
    try:
        import onnx
        from onnx import helper, TensorProto
    except Exception:
        return None

    A = helper.make_tensor_value_info("A", TensorProto.FLOAT, [16, 16])
    B = helper.make_tensor_value_info("B", TensorProto.FLOAT, [16, 16])
    C = helper.make_tensor_value_info("C", TensorProto.FLOAT, [16, 16])
    node = helper.make_node("MatMul", ["A", "B"], ["C"])
    graph = helper.make_graph([node], "matmul16x16", [A, B], [C])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    onnx.checker.check_model(model)
    return model.SerializeToString()


def _benchmark() -> dict:
    """Micro-benchmark of a 16x16 matmul on each provider."""
    out: dict = {}
    a = np.random.RandomState(0).rand(16, 16).astype(np.float32)

    def _np_ms(n=2000):
        t0 = time.perf_counter()
        for _ in range(n):
            a @ a
        return (time.perf_counter() - t0) / n * 1000.0

    out["numpy_cpu_ms"] = _np_ms()
    if _ORT is None:
        out["note"] = "onnxruntime absent; no GPU benchmark"
        return out
    try:
        import onnx
    except Exception:
        out["dml_ms"] = None
        out["note"] = "onnx package absent; cannot build benchmark graph"
        return out
    model_bytes = _matmul_graph()
    if model_bytes is None:
        out["dml_ms"] = None
        out["note"] = "benchmark graph build failed"
        return out
    feeds = {"A": a, "B": a}
    for prov, label in (["DmlExecutionProvider", "dml"], ["CPUExecutionProvider", "cpu_ort"]):
        try:
            sess = _ORT.InferenceSession(model_bytes, providers=[prov])
            t0 = time.perf_counter()
            for _ in range(200):
                sess.run(None, feeds)
            out[label + "_ms"] = (time.perf_counter() - t0) / 200 * 1000.0
        except Exception as e:
            out[label + "_ms"] = None
            out[label + "_error"] = str(e)[:80]
    return out


# ---------------------------------------------------------------------------
# Session pool (DML-first with auto CPU fallback)
# ---------------------------------------------------------------------------

class ONNXSessionPool:
    """Caches reopened ONNX sessions, preferring DirectML and falling back
    to CPU on any backend failure."""

    def __init__(self, *, force_cpu: bool = False) -> None:
        if _ORT is None:
            self._providers: list[str] = []
        elif force_cpu:
            self._providers = ["CPUExecutionProvider"]
        else:
            try:
                self._providers = (
                    ["DmlExecutionProvider", "CPUExecutionProvider"]
                    if provider_state().get("dml") else ["CPUExecutionProvider"]
                )
            except Exception:
                self._providers = ["CPUExecutionProvider"]
        self._sessions: dict[str, Any] = {}
        self._lock = Lock()
        self.stats = {
            "gpu_inferences": 0, "cpu_inferences": 0,
            "gpu_fallbacks": 0, "gpu_errors": 0,
        }
        self._gpu_disabled = bool(force_cpu)

    @property
    def backend(self) -> str:
        if self._gpu_disabled:
            return "CPU"
        return "DirectML" if "DmlExecutionProvider" in self._providers else "CPU"

    def session_for(self, key: str, model: str | Path) -> Any:
        """Return a cached InferenceSession for `model`, reopening on demand."""
        if not _ORT:
            raise RuntimeError("onnxruntime is not installed")
        with self._lock:
            if key in self._sessions:
                return self._sessions[key]
            LOGGER.info("[compute] opening ONNX session '%s' backend=%s",
                        key, self.backend)
            self._sessions[key] = _ORT.InferenceSession(
                str(model), providers=self._providers)
            return self._sessions[key]

    def run(
        self,
        key: str,
        model: str | Path,
        inputs: dict[str, np.ndarray],
        output_names: Sequence[str] | None = None,
    ) -> Any:
        """Run inference. Prefers DML; on DML failure retries on CPU exactly
        once and records the event for later reporting."""
        if not _ORT:
            raise RuntimeError("onnxruntime is not installed")
        out_names = list(output_names) if output_names else None

        def _execute(providers: list[str]) -> Any:
            sess = _ORT.InferenceSession(str(model), providers=providers)
            return sess.run(out_names, inputs)

        if not self._gpu_disabled and "DmlExecutionProvider" in self._providers:
            try:
                res = _execute(self._providers)
                self.stats["gpu_inferences"] += 1
                with self._lock:
                    self._sessions.setdefault(key, None)
                return res
            except Exception as e:
                self.stats["gpu_fallbacks"] += 1
                self.stats["gpu_errors"] += 1
                LOGGER.warning(
                    "[compute] DML inference failed for '%s': %s - retrying on CPU",
                    key, e)
                self._gpu_disabled = True
                self._providers = ["CPUExecutionProvider"]
                self._sessions.pop(key, None)
        res = _execute(["CPUExecutionProvider"])
        self.stats["cpu_inferences"] += 1
        return res


_SESSION_POOL: ONNXSessionPool | None = None


def get_sessions(force_cpu: bool = False) -> ONNXSessionPool:
    """Process-wide session pool (lazily built, durable)."""
    global _SESSION_POOL
    if _SESSION_POOL is None or (force_cpu and _SESSION_POOL.backend != "CPU"):
        _SESSION_POOL = ONNXSessionPool(force_cpu=force_cpu)
    return _SESSION_POOL


def run_onnx(
    key: str,
    model: str | Path,
    inputs: dict[str, np.ndarray],
    output_names: Sequence[str] | None = None,
) -> Any:
    """Convenience: run one ONNX inference through the shared pool."""
    return get_sessions().run(key, model, inputs, output_names)


def force_cpu() -> None:
    """Disable the GPU path process-wide (call at startup when told to)."""
    get_sessions(force_cpu=True)


__all__ = [
    "provider_state", "gpu_info_simple", "detect",
    "ONNXSessionPool", "get_sessions", "run_onnx", "force_cpu",
]


def main() -> None:  # CLI: python -m jarvis_predict.compute
    import json
    print(json.dumps(detect(run_benchmark=True), indent=2, default=str))


if __name__ == "__main__":
    main()