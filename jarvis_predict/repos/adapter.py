"""Repo integration fabric for Jarvis Reborn.

Discovers the quant/trading repos installed on this machine (LEAN, FinRL,
qlib, FinGPT, FinRobot, RD-Agent, TradeCast, TimesFM, Chronos) and exposes
them as safe, on-demand adapters. Every adapter is exception-proof and is
invoked through a worker pool so the hot trading path is never blocked.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import threading
import time
from typing import Any, Callable, Dict, List, Optional

_log = logging.getLogger("jarvis.repos")

_HOME = os.path.expanduser("~")

DEFAULT_MODULES = {
    "lean": "QuantConnect",
    "finrl": "finrl",
    "qlib": "qlib",
    "fingpt": "fingpt",
    "finrobot": "finrobot",
    "rdagent": "rdagent",
    "tradecast": "tradecast",
    "timesfm": "timesfm",
    "chronos": "chronos",
}

def _command(args: List[str], timeout: int = 25) -> dict:
    try:
        cp = subprocess.run(
            args, capture_output=True, text=True, timeout=timeout,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        return {"exit": cp.returncode, "out": (cp.stdout or "")[-4000:], "err": (cp.stderr or "")[-4000:]}
    except FileNotFoundError:
        return {"exit": -1, "out": "", "err": "executable not found"}
    except Exception as e:  # noqa: BLE001
        return {"exit": -1, "out": "", "err": str(e)}


class RepoAdapter:
    """Base for a wrapped external repo. Always fails soft."""

    name = "base"
    kind = "repo"

    def __init__(self, ctx: "RepoFabric", cfg: dict):
        self.ctx = ctx
        self.cfg = cfg
        self._lock = threading.Lock()
        self._last = 0.0
        self._cache: Dict[str, Any] = {}

    @property
    def repo_dir(self) -> Optional[str]:
        p = self.cfg.get("path") or self.ctx.paths.get(self.name)
        return p if p and os.path.isdir(p) else None

    @property
    def module_name(self) -> str:
        return self.cfg.get("module") or DEFAULT_MODULES.get(self.name, self.name)

    def available(self) -> bool:
        return self.repo_dir is not None

    def _find_python(self) -> Optional[str]:
        return shutil.which("python") or shutil.which("py")

    def probe(self) -> dict:
        d = self.repo_dir
        if not d:
            return {"available": False, "reason": "repo not found on this machine"}
        py = self._find_python()
        if py is None:
            return {"available": True, "reason": "no python found (external tool only)"}
        check = (
            "import importlib.util,sys;"
            "print('ok' if importlib.util.find_spec(sys.argv[1]) else 'missing')"
        )
        r = _command([py, "-c", check, self.module_name], timeout=40)
        return {
            "available": True,
            "python_ok": r.get("exit") == 0 and (r.get("out") or "").strip() == "ok",
            "detail": (r.get("out") or r.get("err") or "").strip()[-120:],
        }

    def run(self, fn: Callable[[], dict], ttl: Optional[float] = None) -> dict:
        with self._lock:
            now = time.monotonic()
            if ttl and self._last and (now - self._last) < ttl:
                return dict(self._cache)
            try:
                out = fn()
                self._cache = out
                self._last = now
                out["source"] = self.name
                return out
            except Exception as e:  # noqa: BLE001
                _log.exception("repo adapter %s failed", self.name)
                return {"source": self.name, "ok": False, "error": str(e)}

    def status(self) -> dict:
        found = self.repo_dir
        return {
            "name": self.name,
            "kind": self.kind,
            "available": found is not None,
            "path": found,
        }

    def summary(self) -> dict:
        return {"name": self.name, "kind": self.kind}

    def load_later(self) -> dict:
        """Heavy/optional load reported separately."""
        return {"ok": False, "reason": "not implemented"}