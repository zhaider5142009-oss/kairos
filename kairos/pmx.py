"""PMX (Pakistan Mercantile Exchange) venue scaffold — account pending.

The operator's PMX commodity-futures account is opening soon. This module
reserves the exact integration shape KAIROS already uses for PSX cash:

  * venue string  "PMX_PMX"  in OrderIntent / RiskGate / audit journal
  * broker-ready tickets (human-confirmed) via PmxTicket — same compliance
    fields as JS Global tickets, separate out_dir so nothing mixes
  * a config section (`pmx:`) ready for the broker name / account number,
    and (if the broker later exposes a web terminal like JS Global did) a
    read-only sync hook identical to kairos.jsgsync

No live PMX execution exists yet, and none should be invented: every order
stays a ticket until the account is live and a supervised flow is proven.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from .config import CFG
from .execution import JOURNAL, OrderIntent

VENUE = "PMX_PMX"


def _conf() -> dict[str, Any]:
    return CFG.section("pmx") or {}


def ready() -> dict[str, Any]:
    c = _conf()
    return {"venue": VENUE, "configured": bool(c.get("broker")),
            "account_bound": bool(c.get("account")),
            "execution": "TICKETS_ONLY",
            "status": ("scaffold ready — awaiting live PMX account"
                       if not c.get("account") else
                       "account bound — broker sync not yet implemented; "
                       "tickets remain human-confirmed")}


class PmxTicket:
    """Broker-ready PMX futures ticket, written to data/pmx_orders/."""

    def __init__(self) -> None:
        self.out_dir = CFG.data_dir / "pmx_orders"
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def build_ticket(self, intent: OrderIntent, contract: str | None = None,
                     expiry: str | None = None, margin_pct: float | None = None) -> dict[str, Any]:
        c = _conf()
        ticket = {
            "ticket_id": f"PMX-{intent.client_order_id}",
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "broker": c.get("broker") or "REPLACE_ME",
            "account_number": c.get("account") or "REPLACE_ME",
            "market": "PMX", "exchange": "PMEX",
            "contract": contract or intent.symbol,
            "expiry": expiry, "side": intent.side,
            "order_type": intent.order_type,
            "quantity": int(intent.quantity),
            "price": (intent.limit_price if intent.order_type == "LIMIT"
                      else "MARKET"),
            "margin_pct": margin_pct,
            "stop_loss": intent.stop_loss, "take_profit": intent.take_profit,
            "strategy_comment": intent.comment,
            "status": "AWAITING_HUMAN_CONFIRMATION",
        }
        path = self.out_dir / f"{ticket['ticket_id']}.json"
        path.write_text(json.dumps(ticket, indent=2), encoding="utf-8")
        JOURNAL.log("PMX_TICKET", {"ticket_id": ticket["ticket_id"],
                                   "contract": ticket["contract"],
                                   "side": intent.side, "path": str(path)})
        return ticket
