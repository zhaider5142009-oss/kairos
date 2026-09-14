"""
Agent subsystem — multi-agent coordination with token-aware routing.

Public surface:
    AgentSwarm, SwarmDecision, SwarmContext,
    ComputeRouter, ComputeProfile,
    ModelLearner,
    default_agents, roles,
    scan (repo scanner), RepoScan
"""

from __future__ import annotations

from .base import AgentReport, AgentVote, BaseAgent, SwarmContext
from .capabilities import default_agents, roles
from .learner import ModelLearner
from .repos import RepoScan, scan
from .router import ComputeProfile, ComputeRouter, token_budget
from .swarm import AgentSwarm, ConsensusDirector, SwarmDecision

__all__ = [
    "AgentReport",
    "AgentVote",
    "BaseAgent",
    "SwarmContext",
    "AgentSwarm",
    "SwarmDecision",
    "ConsensusDirector",
    "ComputeRouter",
    "ComputeProfile",
    "token_budget",
    "ModelLearner",
    "default_agents",
    "roles",
    "scan",
    "RepoScan",
]