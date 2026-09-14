"""
Listener subsystem — voice / MT5 events / HTTP webhook.

All listeners share one Dispatcher. The dispatcher:
- Holds the TradeExecutor reference (so listeners can confirm/cancel).
- Holds a reference to the runtime for status queries.
- Notifies all subscribed callbacks when an event is published.
"""

from .dispatcher import Dispatcher, Event, EventType
from .voice import VoiceListener
from .mt5_events import MT5EventListener
from .http_api import HttpListener

__all__ = [
    "Dispatcher",
    "Event",
    "EventType",
    "VoiceListener",
    "MT5EventListener",
    "HttpListener",
]
