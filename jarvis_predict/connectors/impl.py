"""
Concrete connector implementations — the small set of probe behaviours behind
the ~120-connector catalog.

Kinds:
    * ``http``       plain public GET     → ok on 2xx
    * ``key``        needs an API key     → unconfigured without it
    * ``mock``       simulated on purpose → always 'ok (simulated)'
    * ``local``      local TCP daemon     → port reachable?
    * ``openrouter`` live OpenRouter keys (reads main config)
    * ``metatrader5`` the local MT5 terminal (reads main config)

Special small subclasses cover header/URL quirks (Anthropic, Gemini, Tiingo,
Telegram, Discord webhooks, …).
"""

from __future__ import annotations

import logging
import time
from typing import Any

from .base import (
    STATUS_OK, STATUS_DEGRADED, STATUS_UNCONFIGURED, STATUS_OFFLINE, STATUS_MOCK,
    Connector, ConnectorHealth, _http_get, _http_ok, _local_probe,
)

LOGGER = logging.getLogger("jarvis.connectors.impl")


# ---------------------------------------------------------------------------
# generic kinds
# ---------------------------------------------------------------------------

class HttpConnector(Connector):
    """Public endpoint smoke test — ok on 2xx, degraded on 4xx/5xx, offline otherwise."""

    def __init__(self, *, url: str, ok_detail: str = "reachable",
                 headers: dict[str, str] | None = None, timeout: float = 4.0) -> None:
        super().__init__()
        self.url = url
        self.ok_detail = ok_detail
        self.headers = headers or {}
        self._timeout = timeout

    def _probe(self) -> ConnectorHealth:
        return _http_ok(self.url, ok_detail=self.ok_detail,
                        headers=self.headers, timeout=self._timeout)


def _mk_http(key: str, spec: dict[str, Any]) -> Connector:
    c = HttpConnector(url=spec["url"], ok_detail=spec.get("ok_detail", "reachable"),
                      headers=spec.get("headers"), timeout=spec.get("timeout", 4.0))
    return _assign(c, key, spec)


class KeyConnector(HttpConnector):
    """API-key connector. Without a key it never touches the network."""

    def __init__(self, *, api_key: str | None = None, key_env: str = "",
                 auth_header: str = "Bearer", **kw: Any) -> None:
        super().__init__(**kw)
        self.api_key = api_key or ""
        self.key_env = key_env or ""
        self.auth_header = auth_header

    def _probe(self) -> ConnectorHealth:
        if not self.api_key:
            return ConnectorHealth(
                status=STATUS_UNCONFIGURED,
                detail=f"Add an API key (env {self.key_env or self.key} or config.connectors.{self.key}.api_key).",
            )
        headers = dict(self.headers or {})
        headers["Authorization"] = f"{self.auth_header} {self.api_key}"
        h = _http_ok(self.url, ok_detail=self.ok_detail, headers=headers,
                     timeout=self._timeout)
        if h.status == STATUS_DEGRADED:
            h.detail = f"auth or endpoint issue — check the key for env {self.key_env or self.key}"
        return h


def _mk_key(key: str, spec: dict[str, Any]) -> Connector:
    api_key = spec.get("api_key")  # resolved by the registry from config/env
    c = KeyConnector(api_key=api_key, key_env=spec.get("key_env", ""),
                     url=spec["url"], ok_detail=spec.get("ok_detail", "reachable"),
                     headers=spec.get("headers"), timeout=spec.get("timeout", 4.0))
    return _assign(c, key, spec)


class SimulatedConnector(Connector):
    """Honest stand-in for closed/broker systems. Always reports simulated."""

    def __init__(self, *, note: str = "") -> None:
        super().__init__()
        self._note = note or "simulated on local data"

    def _probe(self) -> ConnectorHealth:
        return ConnectorHealth(status=STATUS_MOCK, latency_ms=0.4,
                               detail=self._note)


def _mk_mock(key: str, spec: dict[str, Any]) -> Connector:
    c = SimulatedConnector(note=spec.get("ok_detail", "simulated on local data"))
    return _assign(c, key, spec)


class LocalTcpConnector(Connector):
    """Local daemon (ollama, bitcoin core, …) — reachable on a TCP port?"""

    def __init__(self, *, port: int, host: str = "127.0.0.1", note: str) -> None:
        super().__init__()
        self.port = port
        self.host = host
        self.note = note

    def _probe(self) -> ConnectorHealth:
        h = _local_probe(self.port, host=self.host, detail=self.note)
        if h.status == STATUS_OK:
            h.detail = f"{self.note} on {self.host}:{self.port}"
        return h


def _mk_local(key: str, spec: dict[str, Any]) -> Connector:
    c = LocalTcpConnector(port=int(spec["port"]), host=spec.get("host", "127.0.0.1"),
                          note=spec.get("ok_detail", "local service"))
    return _assign(c, key, spec)


# ---------------------------------------------------------------------------
# specialised connectors
# ---------------------------------------------------------------------------

class OpenRouterConnector(Connector):
    """The main LLM layer — probes the live keys from config.openrouter."""

    def __init__(self, *, api_keys: list[str] | None = None) -> None:
        super().__init__()
        self.api_keys = api_keys or []
        self._ttl_ok = 120.0

    def _probe(self) -> ConnectorHealth:
        if not self.api_keys:
            return ConnectorHealth(
                status=STATUS_UNCONFIGURED,
                detail="Add keys under openrouter.api_keys in config.yaml or the OPENROUTER_API_KEYS env var.",
            )
        key = self.api_keys[0]
        try:
            status, ms, _ = _http_get(
                "https://openrouter.ai/api/v1/models",
                headers={"Authorization": f"Bearer {key}"}, timeout=6.0)
        except Exception as e:
            return ConnectorHealth(status=STATUS_OFFLINE, detail=f"{type(e).__name__}: {e}")
        if status == 200:
            return ConnectorHealth(status=STATUS_OK, latency_ms=ms,
                                   detail=f"{len(self.api_keys)} key(s) live")
        return ConnectorHealth(status=STATUS_DEGRADED, latency_ms=ms,
                               detail=f"HTTP {status}")


class MetaTraderConnector(Connector):
    """The local MetaTrader 5 terminal (execution + market data)."""

    def __init__(self, *, terminal_path: str = "C:/Program Files/MetaTrader 5/terminal64.exe") -> None:
        super().__init__()
        self.terminal_path = terminal_path
        self._ttl_ok = 120.0

    def _probe(self) -> ConnectorHealth:
        try:
            import MetaTrader5 as mt5
        except Exception as e:
            return ConnectorHealth(status=STATUS_OFFLINE,
                                   detail=f"MetaTrader5 package: {type(e).__name__}")
        try:
            ok = mt5.initialize(path=self.terminal_path, timeout=15000)
            if ok:
                ti = mt5.terminal_info()
                acc = mt5.account_info()
                bits = []
                if ti is not None:
                    bits.append(f"build {ti.build} · connected={ti.connected}")
                if acc is not None:
                    bits.append(f"acct {acc.login} · balance {acc.balance:.2f} {acc.currency}")
                mt5.shutdown()
                return ConnectorHealth(status=STATUS_OK,
                                       detail=" · ".join(bits) or "terminal live")
            return ConnectorHealth(status=STATUS_DEGRADED,
                                   detail=f"terminal running but no account connected ({mt5.last_error()})")
        except Exception as e:
            return ConnectorHealth(status=STATUS_OFFLINE, detail=f"{type(e).__name__}: {e}")


class QuantConnectConnector(Connector):
    """QuantConnect — cloud research/Live platform + local LEAN engine.

    Probes the public cloud endpoint for licensing reachability and the
    local LEAN engine (default 127.0.0.1:14380) for live algo compute.
    """

    def __init__(self, *, lean_port: int = 14380,
                 lean_host: str = "127.0.0.1") -> None:
        super().__init__()
        self.lean_port = int(lean_port)
        self.lean_host = lean_host
        self._ttl_ok = 120.0

    def _probe(self) -> ConnectorHealth:
        bits = []
        cloud = None
        try:
            status, ms, _ = _http_get(
                "https://www.quantconnect.com/", timeout=6.0,
                headers={"User-Agent": "JarvisReborn/2.0"})
            cloud = (status, ms)
        except Exception as e:
            cloud = None
        if cloud and 200 <= cloud[0] < 300:
            bits.append(f"cloud {cloud[1]:.0f}ms")
        lean = _local_probe(self.lean_port, host=self.lean_host,
                            detail="LEAN engine (local)")
        if lean.status == STATUS_OK:
            bits.append(f"LEAN engine local:{self.lean_port}")
        if not bits:
            return ConnectorHealth(status=STATUS_OFFLINE,
                                   detail="QuantConnect cloud + local LEAN engine unreachable")
        if cloud is None or not (200 <= cloud[0] < 300):
            return ConnectorHealth(status=STATUS_DEGRADED,
                                   detail=" · ".join(bits) + " — cloud offline")
        if lean.status != STATUS_OK:
            return ConnectorHealth(status=STATUS_DEGRADED,
                                   detail=" · ".join(bits) + " — start the LEAN engine (lean live deploy)")
        return ConnectorHealth(status=STATUS_OK, latency_ms=cloud[1],
                               detail=" · ".join(bits) + " — backtest & live algo stack ready")


class AnthropicConnector(KeyConnector):
    """Anthropic uses the x-api-key header, not Authorization."""

    def _probe(self) -> ConnectorHealth:
        if not self.api_key:
            return ConnectorHealth(status=STATUS_UNCONFIGURED,
                                   detail=f"Add an API key (env {self.key_env}).")
        headers = dict(self.headers or {})
        headers["x-api-key"] = self.api_key
        headers["anthropic-version"] = "2023-06-01"
        return _http_ok(self.url, ok_detail=self.ok_detail, headers=headers,
                        timeout=self._timeout)


class GeminiConnector(KeyConnector):
    """Gemini passes the key as a query parameter."""

    def _probe(self) -> ConnectorHealth:
        if not self.api_key:
            return ConnectorHealth(status=STATUS_UNCONFIGURED,
                                   detail=f"Add an API key (env {self.key_env}).")
        return _http_ok(self.url.format(key=self.api_key), ok_detail=self.ok_detail,
                        timeout=self._timeout)


class TiingoConnector(KeyConnector):
    """Tiingo uses an 'Authorization: Token <key>' header."""

    def _probe(self) -> ConnectorHealth:
        if not self.api_key:
            return ConnectorHealth(status=STATUS_UNCONFIGURED,
                                   detail=f"Add an API key (env {self.key_env}).")
        headers = dict(self.headers or {})
        headers["Authorization"] = f"Token {self.api_key}"
        h = _http_ok(self.url, ok_detail=self.ok_detail, headers=headers,
                     timeout=self._timeout)
        if h.status == STATUS_DEGRADED:
            h.detail = f"auth or endpoint issue — check env {self.key_env}"
        return h


class TelegramConnector(Connector):
    """Telegram bot — needs a bot token; probes getMe."""

    def __init__(self, *, token: str | None = None) -> None:
        super().__init__()
        self.token = token or ""

    def _probe(self) -> ConnectorHealth:
        if not self.token:
            return ConnectorHealth(status=STATUS_UNCONFIGURED,
                                   detail="Create a bot with @BotFather and add connectors.telegram_bot.token (or env TELEGRAM_BOT_TOKEN).")
        try:
            status, ms, _ = _http_get(
                f"https://api.telegram.org/bot{self.token}/getMe",
                timeout=5.0)
        except Exception as e:
            return ConnectorHealth(status=STATUS_OFFLINE, detail=f"{type(e).__name__}: {e}")
        if status == 200:
            return ConnectorHealth(status=STATUS_OK, latency_ms=ms, detail="bot token valid")
        return ConnectorHealth(status=STATUS_DEGRADED, latency_ms=ms, detail=f"HTTP {status}")


class DiscordConnector(Connector):
    """Discord webhook."""

    def __init__(self, *, url: str = "") -> None:
        super().__init__()
        self.url = url

    def _probe(self) -> ConnectorHealth:
        if not self.url:
            return ConnectorHealth(status=STATUS_UNCONFIGURED,
                                   detail="Add a Discord webhook URL under connectors.discord_webhook.url.")
        return _http_ok(self.url, ok_detail="webhook reachable", timeout=5.0)


class SlackConnector(Connector):
    """Slack incoming webhook."""

    def __init__(self, *, url: str = "") -> None:
        super().__init__()
        self.url = url

    def _probe(self) -> ConnectorHealth:
        if not self.url:
            return ConnectorHealth(status=STATUS_UNCONFIGURED,
                                   detail="Add a Slack webhook URL under connectors.slack_webhook.url.")
        return _http_ok(self.url, ok_detail="webhook reachable", timeout=5.0)


class SmtpConnector(Connector):
    """Email alerts via SMTP — configurable host/port."""

    def __init__(self, *, host: str = "", port: int = 587) -> None:
        super().__init__()
        self.host = host
        self.port = port

    def _probe(self) -> ConnectorHealth:
        if not self.host:
            return ConnectorHealth(status=STATUS_UNCONFIGURED,
                                   detail="Add connectors.email_smtp.host / .port to enable email alerts.")
        return _local_probe(self.port, host=self.host, detail=f"SMTP {self.host}:{self.port}")


class GenericWebhookConnector(Connector):
    """User-defined HTTP(S) webhook endpoint."""

    def __init__(self, *, url: str = "") -> None:
        super().__init__()
        self.url = url

    def _probe(self) -> ConnectorHealth:
        if not self.url:
            return ConnectorHealth(status=STATUS_UNCONFIGURED,
                                   detail="Add connectors.generic_webhook.url to enable.")
        return _http_ok(self.url, ok_detail="endpoint reachable", timeout=5.0)


# ---------------------------------------------------------------------------
# builders
# ---------------------------------------------------------------------------

def build_connector(key: str, spec: dict[str, Any]) -> Connector:
    """Instantiate one catalog entry. Never raises."""
    kind = spec.get("kind", "http")
    try:
        if kind == "http":
            return _mk_http(key, spec)
        if kind == "key":
            return _mk_key(key, spec)
        if kind == "mock":
            return _mk_mock(key, spec)
        if kind == "local":
            return _mk_local(key, spec)
        if kind == "openrouter":
            c = OpenRouterConnector(api_keys=spec.get("api_keys") or [])
            return _assign(c, key, spec)
        if kind == "metatrader5":
            c = MetaTraderConnector(terminal_path=spec.get("terminal_path",
                                  "C:/Program Files/MetaTrader 5/terminal64.exe"))
            return _assign(c, key, spec)
        if kind == "qc":
            c = QuantConnectConnector(lean_port=spec.get("port", 14380),
                                      lean_host=spec.get("host", "127.0.0.1"))
            return _assign(c, key, spec)
        if kind == "anthropic":
            c = AnthropicConnector(api_key=spec.get("api_key"),
                                   key_env=spec.get("key_env", ""),
                                   url=spec["url"],
                                   ok_detail=spec.get("ok_detail", "reachable"),
                                   headers=spec.get("headers"),
                                   timeout=spec.get("timeout", 4.0))
            return _assign(c, key, spec)
        if kind == "gemini":
            c = GeminiConnector(api_key=spec.get("api_key"),
                                key_env=spec.get("key_env", ""),
                                url=spec["url"],
                                ok_detail=spec.get("ok_detail", "reachable"),
                                timeout=spec.get("timeout", 4.0))
            return _assign(c, key, spec)
        if kind == "tiingo":
            c = TiingoConnector(api_key=spec.get("api_key"),
                                key_env=spec.get("key_env", ""),
                                url=spec["url"],
                                ok_detail=spec.get("ok_detail", "reachable"),
                                timeout=spec.get("timeout", 4.0))
            return _assign(c, key, spec)
        if kind == "telegram":
            c = TelegramConnector(token=spec.get("api_key"))
            return _assign(c, key, spec)
        if kind == "discord":
            c = DiscordConnector(url=spec.get("webhook_url", ""))
            return _assign(c, key, spec)
        if kind == "slack":
            c = SlackConnector(url=spec.get("webhook_url", ""))
            return _assign(c, key, spec)
        if kind == "smtp":
            c = SmtpConnector(host=spec.get("host", ""), port=spec.get("port", 587))
            return _assign(c, key, spec)
        if kind == "webhook":
            c = GenericWebhookConnector(url=spec.get("webhook_url", ""))
            return _assign(c, key, spec)
    except Exception as e:  # catalog bug — never crash the registry
        LOGGER.exception("Failed to build connector %s: %s", key, e)
    fallback = SimulatedConnector(note="build failed — simulated")
    return _assign(fallback, key, spec)


def _assign(c: Connector, key: str, spec: dict[str, Any]) -> Connector:
    c.key = key
    c.name = spec.get("name", key)
    c.category = spec.get("category", "")
    c.vendor = spec.get("vendor", c.name)
    c.description = spec.get("description", "")
    c.config_hint = spec.get("config_hint", spec.get("description", ""))
    return c


__all__ = [
    "HttpConnector", "KeyConnector", "SimulatedConnector", "LocalTcpConnector",
    "OpenRouterConnector", "MetaTraderConnector", "QuantConnectConnector",
    "AnthropicConnector", "GeminiConnector", "TiingoConnector",
    "TelegramConnector", "DiscordConnector", "SlackConnector",
    "SmtpConnector", "GenericWebhookConnector",
    "build_connector",
]