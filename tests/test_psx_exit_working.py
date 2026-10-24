"""A JS Global LIMIT sell rests at the broker; it does not close the book.

The autopilot's PSX path places a day LIMIT and calls the answer `PLACED`. Until
now that answer took the same branch as a paper fill: `_book_close` popped the
managed entry — stop, target, entry date — and added the estimated loss to
`consec_losses`, while the venue still held every share. The position then had
nobody managing its stop, no maturity date, and no memory that an exit was
already offered, so the next cycle could offer the same quantity again.

These tests run the real cycle against a stubbed portal. `PLACED` must leave the
entry, its stop and the loss counter untouched; the resting quantity must stop
the next cycle from doubling the order; and reconciliation — not the order
acknowledgement — must be what forgets the position, once the venue reports
itself flat.

The last test is the other half of the split: a venue that reports the sale
completed still closes the book, because that loss was actually taken.
"""

from __future__ import annotations

import kairos.autopilot as AP
from test_autopilot_cycle import _fresh_paper
from test_risk_veto_channel import _engine, _vote


class _Journal:
    """Records what the cycle writes down, instead of a file to parse back."""

    def __init__(self) -> None:
        self.rows: list[tuple[str, dict]] = []

    def log(self, event: str, payload: dict | None = None) -> None:
        self.rows.append((event, payload or {}))

    def events(self) -> list[str]:
        return [e for e, _ in self.rows]

    def count(self, event: str) -> int:
        return self.events().count(event)


class _Portal:
    """The funded PSX book as the portal reports it, plus the broker's order desk.

    `book()` is what the autopilot reads at the top of a cycle; `place()` is the
    broker accepting a LIMIT. Accepting one moves the shares from free to
    resting, which is exactly the fact the next cycle has to respect.
    """

    def __init__(self, price: float, qty: float = 100.0) -> None:
        self.holdings = [{"symbol": "OGDC", "qty": qty,
                          "avg": round(price + 5, 4),
                          "mtm": price, "pending_sell": 0.0}]
        self.orders: list[dict] = []

    def book(self) -> dict:
        return {"cash": 500_000.0, "equity": 1_000_000.0,
                "holdings": [dict(h) for h in self.holdings]}

    def place(self, symbol, side, qty, price, order_type, note="") -> dict:
        self.orders.append({"symbol": symbol, "side": side, "qty": qty,
                            "type": order_type, "note": note})
        for h in self.holdings:
            if h["symbol"] == symbol:
                h["pending_sell"] += qty
        return {"ok": True, "order_no": "90001", "account": "TEST-ACCOUNT"}

    def settled(self) -> None:
        """The broker filled the resting order: the venue is flat."""
        self.holdings = []


def _prediction(monkeypatch) -> dict:
    res = _engine(monkeypatch, [_vote(1, 0.9, "Bull")])._predict_uncached(
        "OGDC", 5, use_wsl=False, include_news=False)
    assert res["ok"] is True, res.get("error")
    return res


def _autopilot(monkeypatch, tmp_path, portal: _Portal, journal: _Journal,
               res: dict, mode: str = "live") -> AP.Autopilot:
    """One autopilot over one real engine prediction, with a three-line entry.

    Only the venue, the broker, the journal and the prediction are stubbed: the
    cycle, the sizing, the exit rules and the state file are the production ones.
    """
    ap = AP.Autopilot()
    ap.state_path = tmp_path / "autopilot.json"
    ap.st = ap._defaults()
    price = float(res["price"])
    ap.st.update({
        "mode": mode,
        "symbols": ["OGDC"],
        # This file is about an exit already in flight. Park the entry bar where
        # no score can clear it, so a later cycle cannot open a fresh position
        # and the only order any of these cycles can send is the exit.
        "entry_threshold": 2.0,
        "last_rotation": AP._now(),
        "held": {"OGDC": {"started": AP._now(), "side": "BUY", "qty": 100.0,
                          "entry": round(price + 5, 4),
                          "stop": round(price + 1, 4),
                          "stop0": round(price + 1, 4),
                          "target": round(price * 1.2, 4)}},
    })
    monkeypatch.setattr(ap, "_psx_book", portal.book)
    monkeypatch.setattr(ap.mt5, "ensure", lambda: False)
    monkeypatch.setattr(ap.gate, "check", lambda intent, acct, pos: (True, ""))
    monkeypatch.setattr(AP, "PaperBroker", _fresh_paper(tmp_path))
    monkeypatch.setattr(AP, "JOURNAL", journal)

    from kairos import jsgtrade
    monkeypatch.setattr(jsgtrade, "enabled", lambda: True)
    monkeypatch.setattr(jsgtrade, "place_now", portal.place)

    real_section = AP.CFG.section
    monkeypatch.setattr(AP.CFG, "section",
                        lambda name: ({"auto_place": True}
                                      if name == "autopilot" else real_section(name)))

    from kairos.engine import ENGINE
    monkeypatch.setattr(ENGINE, "predict", lambda key, horizon=None, **kw: res)
    monkeypatch.setattr(ENGINE, "refresh_news", lambda watch: {})
    return ap


# ------------------------------------------------------------------ the order
def test_a_resting_sell_leaves_the_position_and_its_stop_intact(monkeypatch,
                                                               tmp_path):
    res = _prediction(monkeypatch)
    journal, portal = _Journal(), _Portal(res["price"])
    ap = _autopilot(monkeypatch, tmp_path, portal, journal, res)

    out = ap.cycle()

    assert out["trades"] == 1
    assert [(o["side"], o["qty"]) for o in portal.orders] == [("SELL", 100.0)]
    held = ap.st["held"]["OGDC"]
    # the venue still holds the shares: the stop, the target and the entry date
    # are the only things that keep managing them
    assert held["stop"] == round(float(res["price"]) + 1, 4)
    assert held["target"] == round(float(res["price"]) * 1.2, 4)
    assert held["started"]
    assert held["exit_working"]["qty"] == 100.0
    assert "order 90001" in held["exit_working"]["order"]
    # the loss was not taken, so it cannot count toward the entry cooldown
    assert ap.st["consec_losses"] == 0
    assert journal.count("AUTOPILOT_BOOK_CLOSE") == 0
    assert journal.count("AUTOPILOT_EXIT_WORKING") == 1
    row = ap.st["signals"][0]
    assert row["action"] == "EXIT-WORKING"
    assert row["detail"] == "STOP exit working at the broker"
    assert ap.st["actions"][0]["action"] == "EXIT-WORKING"


def test_the_next_cycle_does_not_offer_the_same_shares_twice(monkeypatch, tmp_path):
    """The portal now reports 100 shares committed to a resting sell. Offering
    them again is either an oversell refusal or, at a venue that does not net,
    a second exit on a position that has only one."""
    res = _prediction(monkeypatch)
    journal, portal = _Journal(), _Portal(res["price"])
    ap = _autopilot(monkeypatch, tmp_path, portal, journal, res)

    ap.cycle()
    carved = ap.st["held"]["OGDC"]["exit_working"]["ts"]
    second = ap.cycle()

    assert len(portal.orders) == 1, "the resting sell was offered twice"
    assert journal.count("AUTOPILOT_PLACED") == 1
    assert second["trades"] == 0
    assert ap.st["held"]["OGDC"]["exit_working"]["ts"] == carved
    assert ap.st["signals"][0]["action"] == "hold"


def test_the_entry_is_forgotten_only_when_the_venue_is_flat(monkeypatch, tmp_path):
    """The resting order may sit unfilled for the rest of the day. Reconciliation
    is what ends the trade, and it ends it when the shares are gone."""
    res = _prediction(monkeypatch)
    journal, portal = _Journal(), _Portal(res["price"])
    ap = _autopilot(monkeypatch, tmp_path, portal, journal, res)

    ap.cycle()
    assert "OGDC" in ap.st["held"]
    portal.settled()
    ap.cycle()

    assert "OGDC" not in ap.st["held"]
    assert journal.count("AUTOPILOT_RECONCILE") == 1
    assert len(portal.orders) == 1
    assert ap.st["signals"][0]["action"] == "hold"


# ------------------------------------------------------------- the other half
def test_a_paper_fill_still_closes_the_book(monkeypatch, tmp_path):
    """A venue that reports the sale completed has closed the position, and the
    loss it took belongs in the cooldown count — the split above must not have
    made every close conditional on a broker."""
    res = _prediction(monkeypatch)
    journal, portal = _Journal(), _Portal(res["price"])
    price = float(res["price"])
    prepare = _fresh_paper(tmp_path)

    class _Preloaded(prepare):
        def __init__(self) -> None:
            super().__init__()
            self.positions["OGDC"] = {"qty": 100.0,
                                      "entry": round(price + 5, 4),
                                      "last": price, "commission": 0.0}

    ap = _autopilot(monkeypatch, tmp_path, portal, journal, res, mode="paper")
    monkeypatch.setattr(AP, "PaperBroker", _Preloaded)

    out = ap.cycle()

    assert out["trades"] == 1
    assert "OGDC" not in ap.st["held"]
    assert ap.st["consec_losses"] == 1
    assert journal.count("AUTOPILOT_BOOK_CLOSE") == 1
    assert ap.st["signals"][0]["action"] == "CLOSED"
    assert portal.orders == [], "paper mode reached the broker"
