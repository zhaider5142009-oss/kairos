"""
Connectors package — the JARVIS integration layer.

120 connectors across 12 categories, built from ``catalog.py`` by
:class:`ConnectorRegistry` (see ``base.py`` for the model).

Usage:
    from jarvis_predict.connectors import ConnectorRegistry
    registry = ConnectorRegistry.from_config(config)
    registry.start()
    registry.snapshot()
"""

from .base import (
    STATUS_OK, STATUS_DEGRADED, STATUS_UNCONFIGURED, STATUS_OFFLINE,
    STATUS_MOCK, STATUS_UNKNOWN, STATUS_LABEL,
    CATEGORIES, CATEGORY_NAMES,
    Connector, ConnectorHealth,
)
from .registry import ConnectorRegistry
from .catalog import CATALOG, catalog
from .impl import build_connector

__all__ = [
    "ConnectorRegistry",
    "Connector", "ConnectorHealth",
    "STATUS_OK", "STATUS_DEGRADED", "STATUS_UNCONFIGURED", "STATUS_OFFLINE",
    "STATUS_MOCK", "STATUS_UNKNOWN", "STATUS_LABEL",
    "CATEGORIES", "CATEGORY_NAMES",
    "CATALOG", "catalog", "build_connector",
]