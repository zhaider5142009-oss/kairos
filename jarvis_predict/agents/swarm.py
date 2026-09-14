"""
Agent swarm — the coordinated prediction/decision cortex of Jarvis.

The swarm turns the raw feature snapshot (one per symbol, per cycle) into
a *coordinated* decision:

    1. ComputeRouter evaluates the task → complexity tier, agent count,
       and token budget (token-efficient: tiny questions cost nothing).
    2. The swarm engages the appropriate subset of specialised agents in
       parallel (thread pool) — abstaining agents simply return None.
    3. The ConsensusDirector fuses votes into a SwarmDecision, honouring
       risk vetoes (CalendarGuard blackouts, RiskShield guards).
    4. For deep/hive tiers, an LLM reasoner (OpenRouter) reviews the
       reduced vote bundle — small prompt, bounded tokens.
    5. The ModelLearner reweights agents/models based on realised
       outcomes, so the entire swarm gets sharper over time.

Everything is cached by `symbol` for a short TTL, and the swarm emits
telemetry (AgentReport stream) the UI shows in the Agents panel.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from .base import AgentReport, AgentVote, BaseAgent, SwarmContext
from .capabilities import default_agents
from .learner import ModelLearner
from .router import ComputeProfile, ComputeRouter

LOGGER = logging.getLogger("jarvis.swarm")


@dataclass
class SwarmDecision:
    symbol: str
    direction: int
    confidence: float
    votes: list[AgentVote] = field(default_factory=list)
    profile: ComputeProfile | None = None
    meta_reasoning: str = ""
    meta_source: str = "consensus"
    elapsed_ms: float = 0.0
    summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "direction": self.direction,
            "confidence": round(self.confidence, 4),
            "votes": [v.to_dict() for v in self.votes],
            "profile": self.profile.to_dict() if self.profile else None,
            "meta_reasoning": self.meta_reasoning,
            "meta_source": self.meta_source,
            "elapsed_ms": round(self.elapsed_ms, 1),
            "summary": self.summary,
        }


class _Veto(RuntimeError):
    """Raised by the consensus director when a veto fires."""


def _signed_weighted(vote: AgentVote, mult: float) -> float:
    return float(vote.direction) * vote.confidence * vote.weight * mult


class ConsensusDirector:
    """Fuses agent votes into one decision, honouring vetoes + the learner."""

    def __init__(self, learner: ModelLearner | None = None,
                 min_confidence: float = 0.20) -> None:
        self.learner = learner
        self.min_confidence = float(min_confidence)

    def fuse(self, votes: list[AgentVote]) -> tuple[int, float, str]:
        # Veto check: any high-weight negative vote with conf ≥ 0.9 is a veto.
        for v in votes:
            if v.role == "risk" and v.direction == -1 and v.confidence >= 0.90:
                raise _Veto(v.note or "risk veto")

        bullish = bearish = 0.0
        parts: list[str] = []
        for v in votes:
            mult = self.learner.multiplier(f"agent:{v.agent}") if self.learner else 1.0
            score = _signed_weighted(v, mult)
            if v.direction > 0:
                bullish += score
            elif v.direction < 0:
                bearish += abs(score)
            parts.append(f"{v.agent}({v.note})")
        if bullish == bearish:
            return 0, 0.0, "balanced agents — no edge"
        total = bullish + bearish
        direction = 1 if bullish > bearish else -1
        confidence = min(1.0, abs(bullish - bearish) / max(total, 1e-9))
        if confidence < self.min_confidence:
            return 0, 0.0, "below confidence floor"
        summary = " / ".join(parts[:6])
        return direction, confidence, summary


class AgentSwarm:
    """Coordinated multi-agent analysis engine."""

    def __init__(
        self,
        *,
        agents: list[BaseAgent] | None = None,
        router: ComputeRouter | None = None,
        learner: ModelLearner | None = None,
        min_confidence: float = 0.20,
        cache_ttl: float = 4.0,
        max_workers: int = 8,
        meta_reasoner: Callable[[dict], Any] | None = None,
        on_report: Callable[[AgentReport], None] | None = None,
    ) -> None:
        self.agents = list(agents) if agents is not None else default_agents()
        self.router = router or ComputeRouter()
        self.learner = learner or ModelLearner()
        self.director = ConsensusDirector(self.learner, min_confidence)
        self.cache_ttl = float(cache_ttl)
        self.meta_reasoner = meta_reasoner          # optional OpenRouter wrapper
        self.on_report = on_report or (lambda r: None)
        self._max_workers = int(max_workers)
        self._cache: dict[str, tuple[float, SwarmDecision]] = {}
        self._lock = threading.RLock()
        self._pool = None

    # ------------------------------------------------------------------ public

    async def predict(
        self,
        ctx: SwarmContext,
        *,
        profile: ComputeProfile | None = None,
    ) -> SwarmDecision:
        """Run the swarm for one symbol/task. Returns a SwarmDecision."""
        cache_key = f"{ctx.symbol}:{ctx.action_type}"
        with self._lock:
            hit = self._cache.get(cache_key)
            if hit and (time.time() - hit[0]) < self.cache_ttl:
                return hit[1]

        t0 = time.perf_counter()
        profile = profile or self.router.evaluate(
            task_type=ctx.action_type, query=ctx.query,
            symbol=ctx.symbol, snapshot=ctx.snapshot, mode=ctx.mode,
        )
        votes = await self._run_agents(ctx, profile.agent_count)
        try:
            direction, confidence, summary = self.director.fuse(votes)
        except _Veto as e:
            direction, confidence, summary = 0, 0.0, str(e)

        meta_text, meta_src = "", "consensus"
        if profile.tier in ("deep", "hive") and self.meta_reasoner is not None:
            try:
                bundle = {
                    "symbol": ctx.symbol,
                    "votes": [v.to_dict() for v in votes[:16]],
                    "direction": direction,
                    "confidence": confidence,
                    "tier": profile.tier,
                    "token_budget": profile.max_output_tokens,
                }
                decision = await asyncio.wait_for(
                    asyncio.to_thread(self.meta_reasoner, bundle),
                    timeout=max(6.0, profile.max_output_tokens / 400.0),
                )
                d, c, r, src = self._normalise_meta(decision)
                if d in (-1, 0, 1):
                    direction, confidence, meta_text, meta_src = d, c, r, src
            except Exception as e:
                LOGGER.debug("meta reasoner failed: %s", e)

        elapsed = (time.perf_counter() - t0) * 1000.0
        decision = SwarmDecision(
            symbol=ctx.symbol,
            direction=direction,
            confidence=confidence,
            votes=votes,
            profile=profile,
            meta_reasoning=meta_text,
            meta_source=meta_src,
            elapsed_ms=elapsed,
            summary=summary,
        )
        with self._lock:
            self._cache[cache_key] = (time.time(), decision)
        # Occasionally prune cache.
        with self._lock:
            if len(self._cache) > 64:
                now = time.time()
                self._cache = {k: v for k, v in self._cache.items()
                               if now - v[0] < self.cache_ttl * 4}
        return decision

    def predict_sync(self, ctx: SwarmContext, *, profile: ComputeProfile | None = None) -> SwarmDecision:
        """Synchronous convenience for chat handlers / non-async callers."""
        return asyncio.run(self.predict(ctx, profile=profile))

    # ------------------------------------------------------------------ voting

    async def _run_agents(self, ctx: SwarmContext, count: int) -> list[AgentVote]:
        """Run up to `count` agents concurrently. Returns non-None votes."""
        agents = self._select(count)
        if not agents:
            return []
        loop = asyncio.get_running_loop()
        if self._pool is None:
            self._pool = await asyncio.to_thread(
                lambda: __import__("concurrent.futures").futures
                        .ThreadPoolExecutor(max_workers=self._max_workers))
        tasks = [
            loop.run_in_executor(self._pool, self._run_one, a, ctx)
            for a in agents
        ]
        reports = await asyncio.gather(*tasks, return_exceptions=True)
        votes: list[AgentVote] = []
        for a, r in zip(agents, reports):
            if isinstance(r, AgentVote):
                votes.append(r)
            elif isinstance(r, BaseException):
                if not isinstance(r, ValueError):
                    LOGGER.debug("agent %s error: %s", a.name, r)
        return votes

    def _run_one(self, agent: BaseAgent, ctx: SwarmContext) -> AgentVote | None:
        t0 = time.perf_counter()
        try:
            vote = agent.vote(ctx)
            self.on_report(AgentReport(
                agent=agent.name, role=getattr(agent, "role", "?"),
                status="returned", elapsed_ms=(time.perf_counter() - t0) * 1000.0,
                note=vote.note if vote else "abstained",
                started_at=time.time(),
            ))
            return vote
        except Exception as e:  # agents must never crash the swarm
            self.on_report(AgentReport(
                agent=agent.name, role=getattr(agent, "role", "?"),
                status="error", elapsed_ms=(time.perf_counter() - t0) * 1000.0,
                note=f"{type(e).__name__}: {e}", started_at=time.time(),
            ))
            return None

    def _select(self, count: int) -> list[BaseAgent]:
        if count <= 0:
            return []
        if len(self.agents) <= count:
            return list(self.agents)
        # Deterministic round-robin priority: risk champions first, then
        # ensemble, then the fastest feature readers.
        priority = sorted(
            self.agents,
            key=lambda a: (0 if getattr(a, "role", "") == "risk" else
                           1 if getattr(a, "role", "") == "ensemble" else
                           2 if getattr(a, "role", "") == "technical" else 3),
        )
        return priority[:count]

    @staticmethod
    def _normalise_meta(decision: Any) -> tuple[int, float, str, str]:
        """Accept dict or object from the OpenRouter meta-reasoner."""
        if hasattr(decision, "direction"):
            return (int(decision.direction),
                    float(getattr(decision, "confidence", 0.0) or 0.0),
                    str(getattr(decision, "reasoning", "") or ""),
                    str(getattr(decision, "source", "openrouter") or "openrouter"))
        if isinstance(decision, dict):
            return (int(decision.get("direction", 0)),
                    float(decision.get("confidence", 0.0) or 0.0),
                    str(decision.get("reasoning", "") or ""),
                    str(decision.get("source") or "openrouter"))
        return 0, 0.0, "", "consensus"

    # ------------------------------------------------------------------ util

    def status_table(self) -> list[dict]:
        return [
            {
                "name": a.name,
                "role": getattr(a, "role", "?"),
                "description": getattr(a, "description", ""),
            }
            for a in self.agents
        ]

    def clear_cache(self) -> None:
        with self._lock:
            self._cache.clear()


__all__ = ["AgentSwarm", "SwarmDecision", "ConsensusDirector"]