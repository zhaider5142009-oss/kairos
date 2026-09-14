"""
Repo scanner — indexes the machine's AI/trading repos for Jarvis.

Jarvis can draw capability hints from every relevant repo on the PC:
whether TimesFM / qlib / Chronos / FinGPT / FinRobot / DeepSeek / Kimi-K3
clones exist, where they live, and whether their packages are importable
right now. The swarm's ResearcherAgent and the meta-orchestrator use this
to (1) prefer local artifacts, (2) describe what the system already has,
and (3) log provenance. It is read-only and never executes repo code.
"""

from __future__ import annotations

import importlib.util
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

LOGGER = logging.getLogger("jarvis.repos")


@dataclass(frozen=True)
class RepoInfo:
    name: str
    kind: str                 # model | framework | agent | data | other
    path: Path
    imported: bool            # is its top-level package importable right now?
    hint: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "kind": self.kind,
            "path": str(self.path),
            "imported": self.imported,
            "hint": self.hint,
        }


@dataclass
class RepoScan:
    repos: list[RepoInfo] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"repos": [r.to_dict() for r in self.repos]}

    def names(self) -> list[str]:
        return [r.name for r in self.repos]


def _importable(pkg: str) -> bool:
    try:
        return importlib.util.find_spec(pkg) is not None
    except Exception:
        return False


_KNOWN: list[tuple[str, str, str, str]] = [
    ("timesfm",     "model",     "timesfm",            "google-research timesfm clone"),
    ("qlib",        "framework", "qlib",               "microsoft qlib ML zoo"),
    ("chronos",     "model",     "chronos",            "amazon chronos forecasting"),
    ("FinGPT",      "model",     "FinGPT",             "AI4Finance llm forecaster"),
    ("FinRobot",    "agent",     "FinRobot",           "AI4Finance multi-agent analyst"),
    ("DeepSeek-V3", "model",     "DeepSeek-V3",        "deepseek V3 weights/dev kit"),
    ("Kimi-K3",     "model",     "Kimi-K3",            "moonshot kimi K3 dev kit"),
    ("RD-Agent",    "agent",     "RD-Agent",           "microsoft research agent"),
    ("TradeCast",   "data",      "TradeCast",          "JS Global Capital terminal data"),
    ("Uncensored-Local-AI-Multiplatform", "model", "Uncensored-Local-AI-Multiplatform",
     "local LLM multiplatform stack"),
]


def scan(*, root: str | None = None) -> RepoScan:
    """Scan the known repo locations under a user root (default: home dir)."""
    base = Path(root or Path.home())
    scan_out: RepoScan = RepoScan()
    seen = set()
    for name, kind, folder, hint in _KNOWN:
        path = base / folder
        if not path.is_dir():
            alt = base / name
            if alt.is_dir():
                path = alt
            else:
                continue
        if name in seen:
            continue
        seen.add(name)
        scan_out.repos.append(RepoInfo(
            name=name, kind=kind, path=path,
            imported=_importable(folder.lower().replace("-", "_")) or
                    _importable(name.lower().replace("-", "_")),
            hint=hint,
        ))
    # Also sweep for any git repo under the user root (depth 2) that looks
    # ML/trading flavoured, so Jarvis can notice new tools on the machine.
    extra = _sweep(base, depth=2, cap=40)
    for r in extra:
        if r.name not in seen:
            seen.add(r.name)
            scan_out.repos.append(r)
    LOGGER.info("Repo scan found %d relevant repos", len(scan_out.repos))
    return scan_out


def _sweep(base: Path, depth: int, cap: int) -> list[RepoInfo]:
    out: list[RepoInfo] = []
    ml_hint = ("model", "trade", "forecast", "agent", "time", "neural", "llm", "quant")
    try:
        entries = list(sorted(base.iterdir())) if base.is_dir() else []
    except OSError:
        return []                     # Access denied (e.g. Windows junctions)
    for d in entries:
        if len(out) >= cap:
            break
        if d.name.startswith("."):
            continue
        try:
            is_dir = d.is_dir()
            is_git = (d / ".git").is_dir()
        except OSError:
            continue
        if not is_dir:
            continue
        if is_git:
            name = d.name
            if any(t in name.lower() for t in ml_hint):
                out.append(RepoInfo(
                    name=name, kind="other", path=d,
                    imported=False, hint="swept git repo",
                ))
        elif depth > 1:
            out.extend(_sweep(d, depth - 1, cap - len(out)))
    return out


__all__ = ["RepoInfo", "RepoScan", "scan", "_importable"]