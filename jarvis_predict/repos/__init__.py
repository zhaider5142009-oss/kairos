"""Jarvis Reborn — local quant/trading repo integration layer.

Usage::

    from jarvis_predict.repos import RepoFabric, create_fabric
    fabric = create_fabric()          # discovers all repos at init
    probes = fabric.probe_all()       # background probe in parallel
    info   = fabric.get("qlib").probe()
"""

from .adapter import RepoAdapter
from .fabric import RepoFabric, create_fabric
from .adapters import (
    ADAPTER_TYPES,
    DEFAULT_PATHS,
    LeanAdapter,
    FinRLAdapter,
    QlibAdapter,
    FinGPTAdapter,
    FinRobotAdapter,
    RdAgentAdapter,
    TradeCastAdapter,
    TimesFMAdapter,
    ChronosAdapter,
)

__all__ = [
    "RepoAdapter",
    "RepoFabric",
    "create_fabric",
    "ADAPTER_TYPES",
    "DEFAULT_PATHS",
    "LeanAdapter",
    "FinRLAdapter",
    "QlibAdapter",
    "FinGPTAdapter",
    "FinRobotAdapter",
    "RdAgentAdapter",
    "TradeAdapter",
    "TimesFMAdapter",
    "ChronosAdapter",
]