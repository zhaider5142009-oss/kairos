"""WSL foundation-model bridge: Chronos, TimesFM, qlib.

Heavy pretrained models run inside a WSL distro (Linux CUDA toolchains are far
easier there), invoked as subprocesses speaking JSON over stdin/stdout. The
distro and venv location are configurable (`wsl:` section in config.yaml);
health is probed before every batch; if WSL or a model is missing, the engine
silently falls back to the native battery — no fabrication, no crashes.

Protocol (stdin -> stdout):
    {"model": "chronos"|"timesfm"|"qlib", "series": [floats], "horizon": int}
 -> {"ok": true, "point": float, "path": [floats], "sigma": float, ...}
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from .config import CFG

BRIDGE_DIR = Path(__file__).resolve().parent / "wsl_scripts"
BRIDGE_SCRIPT = "kairos_models.py"


def _distro() -> str:
    return str(CFG.section("wsl").get("distro") or "Ubuntu")


def _venv() -> str:
    return str(CFG.section("wsl").get("venv_path") or "~/.kairos-venv")


class WslBridge:
    def __init__(self) -> None:
        self._cache: dict[str, Any] = {}
        self._last_probe: tuple[bool, float] = (False, 0.0)

    # ------------------------------------------------------------------ probe
    def wsl_available(self, force: bool = False) -> bool:
        ok, ts = self._last_probe
        if not force and time.time() - ts < 300 and ts > 0:
            return ok
        if shutil.which("wsl") is None:
            self._last_probe = (False, time.time())
            return False
        try:
            r = subprocess.run(
                ["wsl", "-d", _distro(), "-e", "python3", "-c", "print('KAIROS_OK')"],
                capture_output=True, text=True, timeout=30, check=False)
            ok = "KAIROS_OK" in (r.stdout or "")
        except Exception:
            ok = False
        self._last_probe = (ok, time.time())
        return ok

    def models_ready(self) -> dict[str, bool]:
        """Which foundation models are importable inside WSL right now."""
        if not self.wsl_available():
            return {"chronos": False, "timesfm": False, "qlib": False}
        try:
            r = subprocess.run(
                ["wsl", "-d", _distro(), "-e", "python3", "-c",
                 "import importlib.util as u;"
                 "print(json := __import__('json').dumps({"
                 "'chronos': bool(u.find_spec('chronos')),"
                 "'timesfm': bool(u.find_spec('timesfm')),"
                 "'qlib': bool(u.find_spec('qlib'))}))"],
                capture_output=True, text=True, timeout=60, check=False)
            return json.loads(r.stdout.strip().splitlines()[-1])
        except Exception:
            return {"chronos": False, "timesfm": False, "qlib": False}

    # ------------------------------------------------------------------ setup
    def setup(self) -> str:
        """Install the model stack inside WSL (heavy: ~4GB, one-time)."""
        if not self.wsl_available(force=True):
            return (f"WSL distro '{_distro()}' not reachable. Start it with "
                    f"`wsl -d {_distro()}` (or set wsl.distro in config.yaml) "
                    "then re-run setup.")
        venv = _venv()
        cmds = (
            f"python3 -m venv {venv} --upgrade-deps && "
            f"{venv}/bin/pip install --quiet "
            "chronos-forecasting 'timesfm[torch]' pyqlib pandas numpy torch --index-url https://pypi.org/simple"
        )
        try:
            r = subprocess.run(["wsl", "-d", _distro(), "-e", "bash", "-lc", cmds],
                               capture_output=True, text=True, timeout=1800,
                               check=False)
            out = (r.stdout or "")[-2000:] + (r.stderr or "")[-2000:]
            # pip exits non-zero when a wheel fails. Reporting that as a
            # finished install sends the user to predict with no models.
            head = "SETUP COMPLETE" if r.returncode == 0 else f"SETUP FAILED (exit {r.returncode})"
            return head + "\n" + out
        except Exception as exc:
            return f"SETUP FAILED: {exc}"

    # ---------------------------------------------------------------- predict
    def _call(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        if not self.wsl_available():
            return None
        script_win = BRIDGE_DIR / BRIDGE_SCRIPT
        if not script_win.exists():
            return None
        try:
            r = subprocess.run(
                ["wsl", "-d", _distro(), "--cd", str(BRIDGE_DIR), "-e", "bash",
                 "-lc", f"{_venv()}/bin/python {BRIDGE_SCRIPT}"],
                input=json.dumps(payload), capture_output=True, text=True,
                timeout=240, check=False)
            if r.returncode != 0:
                return None
            line = (r.stdout or "").strip().splitlines()
            if not line:
                return None
            data = json.loads(line[-1])
            return data if data.get("ok") else None
        except Exception:
            return None

    def forecast(self, model: str, series: list[float], horizon: int = 5,
                 use_cache: bool = True) -> dict[str, Any] | None:
        key = f"{model}:{hash(tuple(round(x, 6) for x in series)):x}:{horizon}"
        if use_cache and key in self._cache:
            return self._cache[key]
        out = self._call({"model": model, "series": series, "horizon": horizon})
        if out is not None and use_cache:
            self._cache[key] = out
        return out

    # ------------------------------------------------- native ModelForecast
    def chronos_forecast(self, close_series: list[float], horizon: int = 5):
        from .models_stat import ModelForecast, _direction
        out = self.forecast("chronos", close_series, horizon)
        if out is None:
            return None
        point = float(out.get("point") or 0.0)
        last = float(close_series[-1]) or 1.0
        exp_ret = point / last - 1
        return ModelForecast("Chronos-Bolt", _direction(exp_ret), exp_ret, horizon,
                              float(out.get("confidence", 0.5)),
                              f"quantile10={out.get('q10')} median={out.get('q50')} "
                              f"quantile90={out.get('q90')} ({out.get('backend', 'wsl')})",
                              extras={"q10": out.get("q10"), "q50": out.get("q50"),
                                      "q90": out.get("q90")})

    def timesfm_forecast(self, close_series: list[float], horizon: int = 5):
        from .models_stat import ModelForecast, _direction
        out = self.forecast("timesfm", close_series, horizon)
        if out is None:
            return None
        point = float(out.get("point") or 0.0)
        last = float(close_series[-1]) or 1.0
        exp_ret = point / last - 1
        return ModelForecast("TimesFM-2.5", _direction(exp_ret), exp_ret, horizon,
                              float(out.get("confidence", 0.55)),
                              f"experimental quantiles "
                              f"({out.get('backend', 'wsl')})",
                              extras={"q10": out.get("q10"), "q90": out.get("q90")})


WSL = WslBridge()
