"""
Connector framework — the JARVIS integration layer.

A :class:`Connector` wraps one external system (exchange, data vendor, broker,
LLM provider, notifier, …). Every connector is:

    * **data-driven** — defined once in :mod:`catalog` and built by
      :class:`ConnectorRegistry` from ``config.yaml``
    * **lazy** — nothing heavy is imported or started at boot
    * **exception-safe** — ``check()`` never raises; failures degrade to a
      status, not a crash
    * **cached** — probes are re-run at most every ``ttl`` seconds

Health statuses (``ConnectorHealth.status``):

    - ``ok``            healthy, live
    - ``degraded``      reachable but partial / slow / limited
    - ``unconfigured``  needs an API key or account to activate
    - ``offline``       network / auth / probe failure
    - ``mock``          simulated on purpose (no real integration exists)
    - ``unknown``       not probed yet (first sweep pending)

Categories (``CATEGORIES``) cover market, FX, TradFi, on-chain, macro,
news/sentiment, AI/LLM, quant research, risk analytics, execution, alerts and
web intelligence — the full JARVIS integration surface.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

LOGGER = logging.getLogger("jarvis.connectors")

# ----- health statuses -------------------------------------------------------

STATUS_OK = "ok"
STATUS_DEGRADED = "degraded"
STATUS_UNCONFIGURED = "unconfigured"
STATUS_OFFLINE = "offline"
STATUS_MOCK = "mock"
STATUS_UNKNOWN = "unknown"

STATUS_LABEL = {
    STATUS_OK: "Live",
    STATUS_DEGRADED: "Degraded",
    STATUS_UNCONFIGURED: "Needs key",
    STATUS_OFFLINE: "Offline",
    STATUS_MOCK: "Simulated",
    STATUS_UNKNOWN: "Checking…",
}


# ----- categories ------------------------------------------------------------

CATEGORIES: list[tuple[str, str]] = [
    ("market_crypto", "Crypto Exchanges & Market Data"),
    ("market_fx", "FX & Rates"),
    ("market_tradfi", "Stocks / Bonds / Indices"),
    ("onchain", "On-chain & Crypto Meta"),
    ("macro", "Macro & Government Data"),
    ("news", "News & Sentiment"),
    ("ai_llm", "AI / LLM Providers"),
    ("quant_research", "Quant & Research"),
    ("risk", "Risk & Portfolio Analytics"),
    ("execution", "Execution & Brokers"),
    ("notify", "Notifications & Alerts"),
    ("web_intel", "Web Intelligence"),
]
CATEGORY_NAMES: dict[str, str] = dict(CATEGORIES)


# ----- health -----------------------------------------------------------------

@dataclass
class ConnectorHealth:
    """Result of a connector probe."""
    status: str = STATUS_UNKNOWN
    latency_ms: float = 0.0
    detail: str = ""
    last_checked: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "status_label": STATUS_LABEL.get(self.status, self.status),
            "latency_ms": round(self.latency_ms, 1),
            "detail": self.detail,
            "last_checked": self.last_checked,
            **({} if not self.extra else self.extra),
        }


# ----- base -------------------------------------------------------------------

class Connector:
    """Base connector. Subclass ``_probe`` (or use the catalog helpers)."""

    key: str = ""
    name: str = ""
    category: str = ""          # one of CATEGORY_NAMES
    description: str = ""
    vendor: str = ""
    # Human-friendly guidance shown in the UI when not 'ok'.
    config_hint: str = ""

    def __init__(self) -> None:
        self._health = ConnectorHealth()
        self._last_checked_at = 0.0
        self._ttl_ok = 60.0        # re-probe healthy connectors every 60s
        self._ttl_bad = 300.0      # slow re-probe for unhealthy ones

    # -- probing -------------------------------------------------------------

    def check(self, force: bool = False) -> ConnectorHealth:
        """Return cached health, probing if stale. Never raises."""
        now = time.time()
        ttl = self._ttl_ok if self._health.status == STATUS_OK else self._ttl_bad
        if force or now - self._last_checked_at >= ttl:
            try:
                self._health = self._probe()
            except Exception as e:  # safety net — never raise to callers
                LOGGER.debug("Connector %s probe raised: %s", self.key, e)
                self._health = ConnectorHealth(
                    status=STATUS_OFFLINE, detail=f"probe error: {type(e).__name__}",
                    last_checked=time.time(),
                )
            self._health.last_checked = time.time()
            self._last_checked_at = now
        return self._health

    def _probe(self) -> ConnectorHealth:
        """Subclasses implement. Raise to signal failure."""
        raise NotImplementedError

    def cached_health(self) -> ConnectorHealth:
        """Last known health without probing — instant, never raises."""
        return self._health

    # -- introspection --------------------------------------------------------

    def snapshot(self, probe: bool = True) -> dict[str, Any]:
        """JSON-safe dict used by the REST endpoint + UI.

        ``probe=False`` returns the cached health instantly (no network);
        ideal for request paths while the background sweep keeps it warm.
        """
        h = self.check() if probe else self._health
        return {
            "key": self.key,
            "name": self.name,
            "category": self.category,
            "category_label": CATEGORY_NAMES.get(self.category, self.category),
            "vendor": self.vendor,
            "description": self.description,
            "config_hint": self.config_hint,
            **h.as_dict(),
        }

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Connector {self.key} status={self._health.status}>"


# ----- small shared helpers ---------------------------------------------------

def _http_get(
    url: str,
    *,
    timeout: float = 4.0,
    headers: dict[str, str] | None = None,
    data: bytes | None = None,
    method: str | None = None,
) -> tuple[int, float, str]:
    """Timed HTTP GET/POST via the standard library. Raises on failure."""
    import urllib.request

    req = urllib.request.Request(url, data=data, headers=headers or {}, method=method)
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read(512).decode("utf-8", errors="replace")
        return resp.status, (time.perf_counter() - t0) * 1000.0, body


def _http_ok(health_url: str, *, ok_detail: str = "reachable", timeout: float = 4.0,
             headers: dict[str, str] | None = None) -> ConnectorHealth:
    """Probe an endpoint; return ok on 2xx, offline otherwise."""
    import urllib.error

    try:
        status, ms, _ = _http_get(health_url, timeout=timeout, headers=headers)
    except urllib.error.HTTPError as e:  # a 4xx/5xx that still reached us
        return ConnectorHealth(status=STATUS_OFFLINE, latency_ms=0.0,
                               detail=f"HTTP {e.code}")
    except Exception as e:
        return ConnectorHealth(status=STATUS_OFFLINE, detail=f"{type(e).__name__}: {e}")
    if 200 <= status < 300:
        return ConnectorHealth(status=STATUS_OK, latency_ms=ms, detail=ok_detail)
    return ConnectorHealth(status=STATUS_DEGRADED, latency_ms=ms, detail=f"HTTP {status}")


def _local_probe(port: int, *, host: str = "127.0.0.1", detail: str = "local service") -> ConnectorHealth:
    """Check a TCP port is accepting connections (local daemons, brokers)."""
    import socket
    t0 = time.perf_counter()
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(1.5)
    try:
        s.connect((host, port))
    except Exception as e:
        return ConnectorHealth(status=STATUS_OFFLINE, detail=f"{type(e).__name__}: {e}")
    finally:
        s.close()
    return ConnectorHealth(status=STATUS_OK, latency_ms=(time.perf_counter() - t0) * 1000.0,
                           detail=detail)


__all__ = [
    "STATUS_OK", "STATUS_DEGRADED", "STATUS_UNCONFIGURED", "STATUS_OFFLINE",
    "STATUS_MOCK", "STATUS_UNKNOWN", "STATUS_LABEL",
    "CATEGORIES", "CATEGORY_NAMES",
    "Connector", "ConnectorHealth",
    "_http_get", "_http_ok", "_local_probe",
]