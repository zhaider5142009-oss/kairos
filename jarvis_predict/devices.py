"""Cross-device connectivity for Jarvis Reborn.

Lets the same Jarvis instance be reachable from a phone, tablet or another
PC on the LAN. A lightweight registry tracks who's connected, heartbeats keep
entries fresh, and an audit trail is kept for security.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import socket
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

_log = logging.getLogger("jarvis.devices")

_MAX_DEVICES = 64
_HEARTBEAT_TTL = 60.0


@dataclass
class Device:
    device_id: str
    name: str
    role: str = "peer"            # peer | hub | mobile | desktop | lite
    host: str = ""
    port: int = 0
    user_agent: str = ""
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    capabilities: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "device_id": self.device_id,
            "name": self.name,
            "role": self.role,
            "host": self.host,
            "port": self.port,
            "user_agent": self.user_agent,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "online": (time.time() - self.last_seen) < _HEARTBEAT_TTL,
            "capabilities": self.capabilities,
        }


class DeviceRegistry:
    """Thread-safe registry of devices that have connected to this Jarvis."""

    def __init__(self, state_path: Optional[str] = None):
        self.state_path = state_path
        self._devices: Dict[str, Device] = {}
        self._lock = threading.Lock()
        self._load()

    # --------------------------------------------------------------- persistence

    def _load(self) -> None:
        if not self.state_path or not os.path.isfile(self.state_path):
            return
        try:
            with open(self.state_path) as f:
                raw = json.load(f)
            for item in raw.get("devices", []):
                d = Device(**{k: item[k] for k in
                              ("device_id", "name", "role", "host", "port",
                               "user_agent", "capabilities") if k in item})
                d.first_seen = float(item.get("first_seen", time.time()))
                d.last_seen = float(item.get("last_seen", time.time()))
                self._devices[d.device_id] = d
        except Exception:
            _log.exception("device registry load failed")

    def _save(self) -> None:
        if not self.state_path:
            return
        try:
            os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
            with open(self.state_path, "w") as f:
                json.dump({"devices": [d.to_dict() for d in self._devices.values()]},
                          f, indent=2)
        except Exception:
            _log.exception("device registry save failed")

    # --------------------------------------------------------------- operations

    def register(self, *, device_id: str, name: str, role: str = "peer",
                 host: str = "", port: int = 0, user_agent: str = "",
                 capabilities: Optional[List[str]] = None) -> dict:
        with self._lock:
            now = time.time()
            d = self._devices.get(device_id)
            if d is None:
                if len(self._devices) >= _MAX_DEVICES:
                    oldest = min(self._devices.values(),
                                 key=lambda x: x.last_seen)
                    del self._devices[oldest.device_id]
                d = Device(device_id=device_id, name=name, role=role, host=host,
                           port=port, user_agent=user_agent,
                           capabilities=capabilities or [])
                self._devices[device_id] = d
            d.name = name or d.name
            d.role = role or d.role
            if host:
                d.host = host
            if port:
                d.port = port
            if user_agent:
                d.user_agent = user_agent
            if capabilities:
                d.capabilities = capabilities
            d.last_seen = now
            self._save()
            return d.to_dict()

    def heartbeat(self, device_id: str) -> bool:
        with self._lock:
            d = self._devices.get(device_id)
            if d is None:
                return False
            d.last_seen = time.time()
            return True

    def list_devices(self) -> List[dict]:
        with self._lock:
            # prune stale
            now = time.time()
            self._devices = {
                did: d for did, d in self._devices.items()
                if (now - d.last_seen) < 24 * 3600  # keep day-old entries
            }
            return [d.to_dict() for d in self._devices.values()]

    def remove(self, device_id: str) -> bool:
        with self._lock:
            ok = self._devices.pop(device_id, None) is not None
            self._save()
            return ok

    def count_online(self) -> int:
        return sum(1 for d in self.list_devices() if d["online"])


def default_state_path() -> str:
    p = os.environ.get("JARVIS_STATE_DIR", "")
    if p:
        return os.path.join(p, "devices.json")
    return os.path.join(os.path.expanduser("~"), ".jarvis", "devices.json")


def local_device_info() -> dict:
    """Describe this machine as a device (for announcing to other hubs)."""
    hostname = socket.gethostname()
    try:
        host = socket.gethostbyname(hostname)
    except Exception:
        host = "127.0.0.1"
    return {
        "name": hostname,
        "role": "hub",
        "host": host,
        "user_agent": f"jarvis-reborn-{platform.system()}-{platform.release()}",
        "capabilities": ["status", "signals", "positions", "strategy", "repos", "chat"],
    }