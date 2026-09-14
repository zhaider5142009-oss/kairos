"""QuantConnect LEAN local adapter for Jarvis Reborn.

Wraps the local LEAN engine to:
  - List available LEAN algorithms / projects
  - Launch backtests as background jobs
  - Fetch backtest results / statistics
  - Convert jarvis signals to LEAN entry/exit events

This module runs on demand and is fully non-blocking.
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import os
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .adapter import _command

_log = logging.getLogger("jarvis.repos.lean")

_LEAN_REPO = os.path.expanduser(r"~\LEAN")
_LAUNCHER_DIRS = [
    os.path.join(_LEAN_REPO, "Launcher", "bin", "Release"),
    os.path.join(_LEAN_REPO, "bin", "Release"),
    os.path.join(_LEAN_REPO, "Lean", "bin", "Release"),
]


@dataclass
class BacktestJob:
    job_id: str
    algorithm: str
    start: Optional[str]
    end: Optional[str]
    status: str = "pending"
    result: Optional[dict] = None
    started_at: float = 0.0
    finished_at: float = 0.0
    error: Optional[str] = None


class LeanEngine:
    """Thin wrapper around local QuantConnect LEAN binary / CLI."""

    def __init__(self, repo_root: str = _LEAN_REPO):
        self.repo_root = repo_root
        self.launcher = self._find_launcher()
        self.cli = shutil.which("lean")
        self._jobs: Dict[str, BacktestJob] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ discovery

    def _find_launcher(self) -> Optional[str]:
        for d in _LAUNCHER_DIRS:
            exe = os.path.join(d, "QuantConnect.Lean.Launcher.exe")
            if os.path.isfile(exe):
                return exe
        return None

    def available(self) -> bool:
        return self.launcher is not None or self.cli is not None

    def version(self) -> str:
        if self.cli:
            r = _command([self.cli, "--version"], timeout=30)
            return (r.get("out") or "").strip() or "unknown"
        return "launcher" if self.launcher else "absent"

    # ------------------------------------------------------------------ projects

    def list_projects(self, max_depth: int = 3) -> List[dict]:
        """Scan the repo for algo scripts / project directories."""
        out: List[dict] = []
        if not os.path.isdir(self.repo_root):
            return out
        for root, dirs, files in os.walk(self.repo_root):
            depth = root.replace(self.repo_root, "").count(os.sep)
            if depth > max_depth:
                dirs.clear()
                continue
            algo_files = [f for f in files if f.endswith((".py", ".cs", ".dll"))]
            if algo_files:
                out.append({
                    "name": os.path.basename(root),
                    "path": root,
                    "scripts": algo_files,
                })
        return out[:50]

    # ------------------------------------------------------------------ backtests

    def launch_backtest(self, algorithm: str,
                        start: Optional[str] = None,
                        end: Optional[str] = None,
                        params: Optional[dict] = None) -> BacktestJob:
        """Fire-and-forget a LEAN backtest. Returns immediately."""
        job_id = f"lean-{int(time.time()*1000)}"
        job = BacktestJob(job_id=job_id, algorithm=algorithm, start=start, end=end, started_at=time.time())
        with self._lock:
            self._jobs[job_id] = job
        t = threading.Thread(target=self._run, args=(job,), daemon=True, name=f"lean-{job_id[:16]}")
        t.start()
        return job

    def _run(self, job: BacktestJob) -> None:
        job.status = "running"
        try:
            if self.cli:
                self._run_via_cli(job)
            elif self.launcher:
                self._run_via_launcher(job)
            else:
                job.error = "lean not available"
                job.status = "failed"
        except Exception as e:  # noqa: BLE001
            _log.exception("lean backtest %s failed", job.job_id)
            job.error = str(e)
            job.status = "failed"
        finally:
            job.finished_at = time.time()

    def _run_via_cli(self, job: BacktestJob) -> None:
        args = [
            self.cli, "backtest",
            job.algorithm,
            "--output", job.job_id,
        ]
        if job.start:
            args += ["--start", job.start]
        if job.end:
            args += ["--end", job.end]
        r = _command(args, timeout=3600)
        if r.get("exit") != 0:
            job.error = (r.get("err") or "lean backtest failed")[-1500:]
            job.status = "failed"
        else:
            out_dir = os.path.join(os.getcwd(), job.job_id, "results")
            job.result = self._parse_output(out_dir)
            job.status = "done"

    def _run_via_launcher(self, job: BacktestJob) -> None:
        """Fallback: launch QuantConnect.Lean.Launcher.exe in a temp directory."""
        work = os.path.join(os.environ.get("TEMP", os.getcwd()), job.job_id)
        os.makedirs(work, exist_ok=True)
        algo_cfg = self._write_config(job, work)
        cp = subprocess.run(
            [self.launcher, "--config", algo_cfg],
            capture_output=True, text=True, timeout=3600, cwd=work,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        if cp.returncode != 0:
            job.error = (cp.stderr or "launcher failed")[-1500:]
            job.status = "failed"
        else:
            result_path = os.path.join(work, "results")
            job.result = self._parse_output(result_path)
            job.status = "done"

    def _write_config(self, job: BacktestJob, work: str) -> str:
        cfg = {
            "algorithm-type-name": job.algorithm,
            "data-folder": os.path.join(self.repo_root, "Data"),
            "algorithm-mode": "backtest",
        }
        if job.start:
            cfg["start-date"] = job.start
        if job.end:
            cfg["end-date"] = job.end
        p = os.path.join(work, "config.json")
        with open(p, "w") as f:
            json.dump(cfg, f, indent=2)
        return p

    def _parse_output(self, out_dir: str) -> Optional[dict]:
        # Scan for JSON results produced by LEAN
        if not os.path.isdir(out_dir):
            return None
        for root, _, files in os.walk(out_dir):
            for fname in files:
                if fname.endswith(".json"):
                    try:
                        with open(os.path.join(root, fname)) as f:
                            data = json.load(f)
                        if isinstance(data, dict):
                            stats = data.get("Results", {})
                            return {
                                "total-return": stats.get("TotalPerformance", {}).get("CompoundAnnualReturnValue"),
                                "sharpe": stats.get("TotalPerformance", {}).get("SharpeRatio"),
                                "max-drawdown": stats.get("TotalPerformance", {}).get("MaximumDrawdown"),
                                "num-trades": stats.get("TotalPerformance", {}).get("TotalTrades"),
                                "raw": data,
                            }
                    except Exception:  # noqa: BLE001
                        continue
        return None

    # ------------------------------------------------------------------ status

    def job_status(self, job_id: str) -> Optional[dict]:
        job = self._jobs.get(job_id)
        if job is None:
            return None
        return {
            "job_id": job.job_id,
            "algorithm": job.algorithm,
            "status": job.status,
            "result": job.result,
            "error": job.error,
            "elapsed_s": (job.finished_at or time.time()) - job.started_at,
        }

    def all_jobs(self) -> List[dict]:
        with self._lock:
            return [self.job_status(jid) for jid in self._jobs]

    def summary(self) -> dict:
        return {
            "available": self.available(),
            "version": self.version(),
            "launcher": self.launcher or "absent",
            "cli": "yes" if self.cli else "no",
            "projects": len(self.list_projects()),
            "backtests": len(self._jobs),
        }


_engine: Optional[LeanEngine] = None


def get_lean_engine() -> LeanEngine:
    global _engine
    if _engine is None:
        _engine = LeanEngine()
    return _engine