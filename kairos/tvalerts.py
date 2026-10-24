"""TradingView alert ingestion: a claim to be recorded, never an instruction.

KAIROS already reads TradingView on the way *out* — the PSX scanner in
`data.py` supplies live quotes and the cross-section. This is the way in: a
TradingView alert can POST the same event the operator's own chart fired, so an
external opinion lands in a journal that can be compared with what the engine
called, instead of living in a notification on a phone.

Three rules decide the shape of this module:

* **It cannot trade.** The webhook appends a line and answers. It has no path to
  the order bridge, no arming flag, and the alert is not fed into a prediction.
  A chart alert is somebody else's signal; the desk's own gates still decide
  whether anything is placed, and today nothing is.
* **It fails closed.** Disabled by default; a blank secret refuses every call
  rather than accepting them (an empty shared secret is no secret, and a route
  that treats "unset" as "open" is how webhooks become public write endpoints);
  a symbol the universe cannot price is rejected rather than journalled as
  something the system could act on.
* **It records what arrived, not what we wish had arrived.** The line carries
  the receipt time, the fields as parsed and a digest of the canonical event, so
  a later reader can tell two identical alerts apart and can tell a replay from
  an original.

The master API token is deliberately *not* required here — TradingView would
have to hold it, and that token also unlocks the desk's real-book reads. The
webhook's own secret buys append-only access to one file and nothing else.
"""

from __future__ import annotations

import hashlib
import hmac
import math
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import CFG
from .store import dumps_strict
from .universe import universe

MAX_BODY = 4096
_SIDES = {"buy": 1, "long": 1, "sell": -1, "short": -1}
_lock = threading.Lock()


class AlertError(ValueError):
    """A payload that cannot be recorded as a signal, with the reason why."""


@dataclass(frozen=True)
class Alert:
    symbol: str
    side: int
    price: float
    timeframe: str
    note: str
    received_at: str
    digest: str


def settings() -> dict[str, Any]:
    """The webhook's config, defaults included, resolved on every call.

    Read per request rather than cached at import: an operator who disables the
    webhook should not have to restart the desk for the route to close.
    """
    cfg = CFG.section("data").get("tradingview_webhook") or {}
    journal = cfg.get("journal") or (CFG.data_dir / "tv_alerts.jsonl")
    return {
        "enabled": bool(cfg.get("enabled", False)),
        "secret": str(cfg.get("secret") or ""),
        "max_bytes": int(cfg.get("max_bytes") or MAX_BODY),
        "journal": Path(journal),
    }


def secret_matches(given: str, expected: str) -> bool:
    """Constant-time, and false for every caller when no secret is set."""
    if not expected:
        return False
    return hmac.compare_digest(str(given).encode("utf-8"), str(expected).encode("utf-8"))


def _instrument_keys() -> set[str]:
    return {i.key.strip().upper() for i in universe()}


def parse(payload: Any, now: datetime | None = None) -> Alert:
    """Validate one alert payload; raise `AlertError` with what was wrong."""
    if not isinstance(payload, dict):
        raise AlertError("payload must be a JSON object")

    raw = payload.get("symbol")
    if not isinstance(raw, str) or not raw.strip():
        raise AlertError("symbol is required")
    key = raw.strip().upper()
    if ":" in key:                      # "PSX:WHEAT" -> "WHEAT"
        key = key.rsplit(":", 1)[-1].strip()
    if key not in _instrument_keys():
        raise AlertError(f"symbol {raw!r} is not in the universe")

    side_raw = str(payload.get("side") or payload.get("action") or "").strip().lower()
    if side_raw not in _SIDES:
        raise AlertError("side must be buy/long or sell/short")

    price = payload.get("price")
    if isinstance(price, bool) or not isinstance(price, (int, float)):
        raise AlertError("price is required and must be a number")
    price = float(price)
    if not math.isfinite(price) or price <= 0:
        raise AlertError("price must be finite and positive")

    timeframe = str(payload.get("timeframe") or payload.get("interval") or "unknown")
    timeframe = timeframe.strip()[:16] or "unknown"
    note = str(payload.get("note") or payload.get("message") or "")[:280]

    ts = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    received_at = ts.isoformat(timespec="seconds")
    canonical = dumps_strict({"symbol": key, "side": _SIDES[side_raw],
                              "price": price, "timeframe": timeframe,
                              "note": note, "received_at": received_at})
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
    return Alert(key, _SIDES[side_raw], price, timeframe, note, received_at, digest)


def record(alert: Alert, path: Path) -> Path:
    """Append the alert to its own journal, with the crash guard the audit trail has."""
    from .execution import AuditJournal      # lazy: keeps this module import-light

    journal = AuditJournal(Path(path))
    with _lock:
        ok = journal.log("TV_ALERT", {
            "symbol": alert.symbol,
            "side": alert.side,
            "price": alert.price,
            "timeframe": alert.timeframe,
            "note": alert.note,
            "received_at": alert.received_at,
            "digest": alert.digest,
        })
    if not ok:
        raise RuntimeError(journal.last_error or "alert not recorded")
    return journal.path
