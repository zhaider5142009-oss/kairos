"""
Agent swarm foundation — shared contracts for every Jarvis agent.

An *agent* is a small, single-purpose analytical worker that casts a
signed vote on a decision and explains itself. Agents never block the
event loop: they are pure functions over a frozen decision snapshot plus
optionally raw price arrays. The swarm runs them concurrently and merges
their votes through a consensus director.

Contracts here:
    AgentVote     — one agent's opinion (signed vote + confidence + notes)
    AgentReport   — agent execution telemetry (for the UI / agents view)
    SwarmContext  — the package of data every agent may consume
    BaseAgent     — the interface every specialized agent implements
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class AgentVote:
    """A single agent's signed opinion."""

    agent: str
    role: str
    direction: int                 # -1 / 0 / +1
    confidence: float              # 0..1
    weight: float = 1.0            # base influence (scaled by the learner)
    note: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent": self.agent,
            "role": self.role,
            "direction": self.direction,
            "confidence": round(self.confidence, 4),
            "weight": round(self.weight, 4),
            "note": self.note,
            "extra": {k: str(v)[:80] for k, v in self.extra.items()},
        }


@dataclass
class AgentReport:
    """Execution telemetry for one agent run (streamed to the UI)."""

    agent: str
    role: str
    status: str = "returned"       # returned | skipped | error
    elapsed_ms: float = 0.0
    note: str = ""
    started_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent": self.agent,
            "role": self.role,
            "status": self.status,
            "elapsed_ms": round(self.elapsed_ms, 1),
            "note": self.note,
            "started_at": self.started_at,
        }


@dataclass
class SwarmContext:
    """Everything a swarm of agents needs for one symbol/decision.

    `snapshot` is the runtime's cached per-symbol decision (models,
    fusion, ta, quantum, patterns, news, forces, calendar). `arrays`
    carries the raw price series so TA/pattern agents can recompute
    lightweight features without re-running the 12-model fleet.
    """

    symbol: str
    snapshot: dict[str, Any] = field(default_factory=dict)
    arrays: dict[str, Any] = field(default_factory=dict)   # close/high/low/open
    action_type: str = "forecast"                          # forecast | explain | research
    query: str = ""                                        # natural-language ask (optional)
    mode: str = "auto"                                     # auto | deep | quick

    def get(self, key: str, default: Any = None) -> Any:
        return self.snapshot.get(key, default)


class BaseAgent(Protocol):
    """Every agent exposes the same surface.

    Implementations receive a SwarmContext and either return an AgentVote
    (when they have an opinion) or None (when they abstain). Timings and
    errors are handled by the swarm harness, so agents stay tiny.
    """

    name: str
    role: str
    description: str

    def vote(self, ctx: SwarmContext) -> AgentVote | None:
        """Cast a signed vote. Return None to abstain."""
        ...


def abstain() -> AgentVote | None:
    return None


def now_ms() -> float:
    return time.perf_counter() * 1000.0


__all__ = [
    "AgentVote",
    "AgentReport",
    "SwarmContext",
    "BaseAgent",
    "abstain",
]