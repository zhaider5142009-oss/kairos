"""
Compute router — the token/compute intelligence of Jarvis.

Every task (forecast, research, chat, mt5 op) is scored for complexity
on a 0..1 scale. The score decides:

    tier        — which compute class to use (deterministic / local / llm)
    agents      — how many specialised agents the swarm should engage
    token_budget— max input+output tokens for this task
    model_chain — which LLM chain to prefer (cheap → expensive)

The design goal is token efficiency: a "status" question costs ~0 tokens,
a routine forecast uses the local fleet only, and only genuinely deep
asks (research, multi-symbol regime analysis) escalate to the LLM tier
with a larger budget. The router is pure — no I/O, instant, testable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ComputeProfile:
    """Immutable routing decision for one task."""

    complexity: float          # 0..1
    tier: str                  # "trivial" | "local" | "standard" | "deep" | "hive"
    agent_count: int           # how many agents the swarm should engage
    max_input_tokens: int
    max_output_tokens: int
    model_chain: tuple[str, ...]
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "complexity": round(self.complexity, 2),
            "tier": self.tier,
            "agent_count": self.agent_count,
            "max_input_tokens": self.max_input_tokens,
            "max_output_tokens": self.max_output_tokens,
            "model_chain": list(self.model_chain),
            "reason": self.reason,
        }


# LLM chains, cheapest-first. All available on the user's OpenRouter keys.
_CHAIN_CHEAP = ("deepseek/deepseek-chat",)
_CHAIN_MID = (
    "deepseek/deepseek-chat",
    "openai/gpt-4o-mini",
)
_CHAIN_DEEP = (
    "deepseek/deepseek-chat",
    "openai/gpt-4o",
    "meta-llama/llama-3.1-70b-instruct",
)
_CHAIN_HIVE = (
    "deepseek/deepseek-chat",
    "openai/gpt-4o",
    "meta-llama/llama-3.1-70b-instruct",
    "openai/gpt-4o-mini",
)

_HIGH_IMPACT = re.compile(
    r"\b(fomc|nff?p|npf|cpi|ecb|ppb|fed|blackout|rate cut|hike|breaking|war|crash|"
    r"research|swarm|deep|explain|why|report|analyse|analyze|compare|multi|regime|"
    r"correlation|backtest|strategy)\b",
    re.I,
)

_SYMBOL_HINT = re.compile(r"\b[A-Z]{2,6}\s?\d{0,5}\b")

# Task-type base complexity, tuned empirically.
_TASK_BASE = {
    "status": 0.03,
    "symbols": 0.05,
    "models": 0.10,
    "news": 0.18,
    "calendar": 0.15,
    "forces": 0.20,
    "forecast": 0.35,
    "explain": 0.55,
    "timeframes": 0.50,
    "patterns": 0.30,
    "quantum": 0.30,
    "research": 0.80,
}


class ComputeRouter:
    """Pure scorer + tier/schedule decision maker."""

    LIGHT_AGENTS = 4
    STANDARD_AGENTS = 9
    DEEP_AGENTS = 16

    def evaluate(
        self,
        *,
        task_type: str = "forecast",
        query: str = "",
        symbol: str | None = None,
        snapshot: dict | None = None,
        mode: str = "auto",
    ) -> ComputeProfile:
        """Score a task and produce the routing decision.

        mode: "auto" (score-derived), "quick" (force cheaper), "deep"
        (force the full swarm + LLM chain), "local" (never touch LLM).
        """
        base = float(_TASK_BASE.get(task_type, 0.35))

        complexity = base

        # Esoteric / high-impact vocabulary nudges it up.
        q = query or ""
        if _HIGH_IMPACT.search(q):
            complexity += 0.15
        if len(re.findall(r"[.,;:?!]", q)) >= 2:
            complexity += 0.05
        if symbol and snapshot:
            model_agreement = float((snapshot.get("model_consensus") or {}).get("agreement", 1.0))
            # Disagreement among models makes the task harder to settle.
            complexity += (1.0 - model_agreement) * 0.25
            news_abs = abs(float((snapshot.get("news") or {}).get("sentiment", 0.0)))
            if news_abs > 0.5:
                complexity += 0.05
            if (snapshot.get("calendar") or {}).get("in_blackout"):
                complexity += 0.10
            forces_hint = int((snapshot.get("forces") or {}).get("hint", 0))
            if forces_hint:
                complexity += 0.05
        complexity = max(0.0, min(1.0, complexity))

        if mode == "quick":
            complexity = min(complexity, 0.30)
        elif mode == "local":
            complexity = min(complexity, 0.45)
        elif mode == "deep":
            complexity = max(complexity, 0.75)

        if complexity < 0.06:
            tier, agents, max_in, max_out = "trivial", 0, 0, 0
        elif complexity < 0.20:
            tier, agents, max_in, max_out = "local", self.LIGHT_AGENTS, 2000, 200
        elif complexity < 0.45:
            tier, agents, max_in, max_out = "standard", self.STANDARD_AGENTS, 6000, 600
        elif complexity < 0.72:
            tier, agents, max_in, max_out = "deep", self.DEEP_AGENTS, 12000, 1200
        else:
            tier, agents, max_in, max_out = "hive", self.DEEP_AGENTS + 4, 24000, 2400

        if tier in ("trivial", "local"):
            chain: tuple[str, ...] = ()
        elif tier == "standard":
            chain = _CHAIN_CHEAP
        elif tier == "deep":
            chain = _CHAIN_MID
        else:
            chain = _CHAIN_DEEP if mode == "deep" else _CHAIN_HIVE

        reason = self._reason(task_type, complexity, tier, agents)
        return ComputeProfile(
            complexity=complexity,
            tier=tier,
            agent_count=agents,
            max_input_tokens=max_in,
            max_output_tokens=max_out,
            model_chain=chain,
            reason=reason,
        )

    @staticmethod
    def _reason(task_type: str, complexity: float, tier: str, agents: int) -> str:
        if task_type == "status":
            return "trivial read — deterministic, no LLM"
        if complexity < 0.45:
            return f"{task_type} is routine — fleet + {agents} agents, local/cheap compute only"
        if complexity < 0.72:
            return f"deep {task_type} — {agents} agents, LLM oversight with bounded budget"
        return f"hive-grade {task_type} — {agents} coordinated agents, extended LLM budget"


# ---- static convenience -------------------------------------------------------

def token_budget(profile: ComputeProfile) -> int:
    """Total tokens this task should be allowed to consume."""
    return profile.max_input_tokens + profile.max_output_tokens


__all__ = ["ComputeRouter", "ComputeProfile", "token_budget"]