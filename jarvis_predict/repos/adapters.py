"""Concrete repo adapters: LEAN, FinRL, qlib, FinGPT, FinRobot, RD-Agent, TradeCast, TimesFM, Chronos."""

from __future__ import annotations

import fnmatch
import os
from typing import Any, Dict, List, Optional

from .adapter import RepoAdapter, _command

DEFAULT_PATHS = {
    "lean": r"C:\Users\Haider\LEAN",
    "finrl": r"C:\Users\Haider\FinRL",
    "qlib": r"C:\Users\Haider\qlib",
    "fingpt": r"C:\Users\Haider\FinGPT",
    "finrobot": r"C:\Users\Haider\FinRobot",
    "rdagent": r"C:\Users\Haider\RD-Agent",
    "tradecast": r"C:\Users\Haider\TradeCast",
    "timesfm": r"C:\Users\Haider\timesfm",
    "chronos": r"C:\Users\Haider\chronos-forecasting",
}


class LeanAdapter(RepoAdapter):
    """LEAN local engine (QuantConnect). Reports lean CLI/engine availability."""

    name = "lean"
    kind = "backtest"

    def probe(self) -> dict:
        base = super().probe()
        cli = _command(["lean", "--version"], timeout=30)
        base["cli"] = "ok" if cli.get("exit") == 0 else (cli.get("err") or "lean cli not installed")
        # Look for compiled LEAN launcher in the standard locations.
        launcher = None
        for cand in (
            os.path.join(self.repo_dir or "", "Launcher", "bin", "Release", "QuantConnect.Lean.Launcher.exe"),
            os.path.join(self.repo_dir or "", "bin", "Release", "QuantConnect.Lean.Launcher.exe"),
        ):
            if cand and os.path.isfile(cand):
                launcher = cand
                break
        base["launcher"] = launcher or "not built"
        return base

    def summary(self) -> dict:
        d = self.repo_dir
        return {
            "name": self.name,
            "kind": "research engine",
            "home": d or "absent",
            "capabilities": ["backtest", "research", "data-feed", "optimization"],
        }


class FinRLAdapter(RepoAdapter):
    name = "finrl"
    kind = "ai"

    def probe(self) -> dict:
        base = super().probe()
        if not base.get("python_ok"):
            return base
        py = self._find_python()
        r = _command([py, "-c",
                      "from finrl.meta.preprocessor.preprocessors import data_split; print('finrl-meta ok')"],
                     timeout=60)
        base["flows"] = "ok" if r.get("exit") == 0 else (r.get("err") or "finrl-meta import failed")
        return base

    def summary(self) -> dict:
        return {"name": self.name, "kind": "reinforcement learning", "home": self.repo_dir or "absent",
                "capabilities": ["ppo", "a2c", "ddpg", "sac", "td3", "elegantrl"]}


class QlibAdapter(RepoAdapter):
    name = "qlib"
    kind = "research"

    def probe(self) -> dict:
        base = super().probe()
        if not base.get("python_ok"):
            return base
        py = self._find_python()
        r = _command([py, "-c", "import qlib; print(getattr(qlib, '__version__', '?'))"], timeout=60)
        base["version"] = (r.get("out") or "").strip()
        return base

    def summary(self) -> dict:
        return {"name": self.name, "kind": "quant research platform",
                "home": self.repo_dir or "absent",
                "capabilities": ["alpha158", "alpha360", "lgbm", "xgboost", "transformer", "backtest"]}


class FinGPTAdapter(RepoAdapter):
    name = "fingpt"
    kind = "nlp"

    def probe(self) -> dict:
        base = super().probe()
        if not base.get("python_ok"):
            return base
        py = self._find_python()
        r = _command([py, "-c", "import fingpt; print('ok')"], timeout=60)
        base["package"] = "ok" if r.get("exit") == 0 else (r.get("err") or "fingpt pkg import failed")
        return base

    def summary(self) -> dict:
        return {"name": self.name, "kind": "financial LLM", "home": self.repo_dir or "absent",
                "capabilities": ["sentiment", "forecast-llm", "ner", "qa"]}


class FinRobotAdapter(RepoAdapter):
    name = "finrobot"
    kind = "agent"

    def probe(self) -> dict:
        base = super().probe()
        # FinRobot is an agent framework; check dir + (optionally) package
        sub = os.path.join(self.repo_dir or "", "finrobot")
        base["modules"] = os.path.isdir(sub)
        return base

    def summary(self) -> dict:
        return {"name": self.name, "kind": "financial analyst agent",
                "home": self.repo_dir or "absent",
                "capabilities": ["stock-analysis", "ai-analyst", "report-gen"]}


class RdAgentAdapter(RepoAdapter):
    name = "rdagent"
    kind = "agent"

    def probe(self) -> dict:
        base = super().probe()
        sub = os.path.join(self.repo_dir or "", "rdagent")
        base["modules"] = os.path.isdir(sub)
        return base

    def summary(self) -> dict:
        return {"name": self.name, "kind": "quant R&D agent", "home": self.repo_dir or "absent",
                "capabilities": ["factor-mining", "strategy-evolution", "backtest"]}


class TradeCastAdapter(RepoAdapter):
    """TradeCast terminal on this machine; signals licensing/status files."""

    name = "tradecast"
    kind = "terminal"

    def probe(self) -> dict:
        base = super().probe()
        d = self.repo_dir or ""
        base["settings"] = os.path.isfile(os.path.join(d, "UserSetting6.properties"))
        return base

    def summary(self) -> dict:
        return {"name": self.name, "kind": "charting terminal", "home": self.repo_dir or "absent",
                "capabilities": ["charts", "watchlists"]}


class TimesFMAdapter(RepoAdapter):
    name = "timesfm"
    kind = "forecast"


class ChronosAdapter(RepoAdapter):
    name = "chronos"
    kind = "forecast"


ADAPTER_TYPES = {
    LeanAdapter.name: LeanAdapter,
    FinRLAdapter.name: FinRLAdapter,
    QlibAdapter.name: QlibAdapter,
    FinGPTAdapter.name: FinGPTAdapter,
    FinRobotAdapter.name: FinRobotAdapter,
    RdAgentAdapter.name: RdAgentAdapter,
    TradeCastAdapter.name: TradeCastAdapter,
    TimesFMAdapter.name: TimesFMAdapter,
    ChronosAdapter.name: ChronosAdapter,
}