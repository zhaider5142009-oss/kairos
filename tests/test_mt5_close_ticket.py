"""An MT5 exit closes one ticket. It used to flatten the whole symbol.

The old close path sent a fresh opposite DEAL — which on a hedging account opens
a second position instead of closing the first — and then walked the terminal's
position list closing every ticket whose symbol matched. Two things followed:
the exit paid two spreads instead of one, and any position the autopilot did not
manage (a manual trade, another strategy's ticket) was liquidated with it.

The managed position's ticket is already on the venue row the cycle reads, so
the exit now goes straight to `close_position(ticket)`. These tests run the real
cycle against a stubbed terminal: the operator's other ticket must survive, a
close with no ticket must send nothing at all, a terminal error must leave the
position managed, and an entry must still be an entry.
"""

from __future__ import annotations

from datetime import datetime

import kairos.autopilot as AP
from kairos.execution import Fill
from test_autopilot_cycle import _AGREE, _fresh_paper
from test_psx_exit_working import _Journal
from test_risk_veto_channel import _engine, _vote


class _Terminal:
    """The MT5 bridge as the autopilot uses it: positions, an account, two doors."""

    def __init__(self, rows: list[dict]) -> None:
        self.rows = rows
        self.connected = True
        self.last_error = None
        self.closed: list[int] = []
        self.deals: list = []

    def ensure(self) -> bool:
        return True

    def positions(self) -> list[dict]:
        return [dict(r) for r in self.rows]

    def account(self) -> dict:
        return {"venue": "MT5", "login": "90001", "equity": 100_000.0,
                "balance": 100_000.0, "day_pnl": 0.0,
                "open_positions": len(self.rows), "leverage": 1.0}

    def resolve_symbol(self, key: str) -> str:
        return "XAUUSD" if key == "GOLD" else key

    def tick(self, symbol: str) -> dict:
        return {"bid": 100.0, "ask": 100.1, "spread": 0.1}

    def spec(self, symbol: str | None) -> dict | None:
        # The lot is stated, in the broker's own three numbers. A terminal that
        # would not answer is the other test: nothing gets sized from a guess.
        return {"contract_size": 100.0, "volume_min": 0.01, "volume_step": 0.01}

    def modify_sl(self, ticket: int, sl=None, tp=None) -> bool:
        return True

    def close_position(self, ticket: int) -> bool:
        self.closed.append(int(ticket))
        self.rows = [r for r in self.rows if r.get("ticket") != ticket]
        return True

    def place_order(self, intent) -> Fill:
        self.deals.append(intent)
        self.rows.append({"symbol": "XAUUSD", "side": intent.side,
                          "qty": intent.quantity, "entry": 100.0,
                          "last": 100.0, "pnl": 0.0, "ticket": 4321})
        return Fill(intent.client_order_id, "99999", "XAUUSD", intent.side,
                    intent.quantity, 100.0, 0.0, 0.0,
                    datetime.now().isoformat(timespec="seconds"), "MT5")


def _position(price: float, ticket: int | None, qty: float = 0.05) -> dict:
    row = {"symbol": "XAUUSD", "side": "BUY", "qty": qty,
           "entry": round(price + 5, 4), "last": price,
           "pnl": round(-5 * qty, 2)}
    if ticket is not None:
        row["ticket"] = ticket
    return row


def _prediction(monkeypatch, battery=()) -> dict:
    res = _engine(monkeypatch, [_vote(1, 0.9, "Bull")], battery)._predict_uncached(
        "GOLD", 5, use_wsl=False, include_news=False)
    assert res["ok"] is True, res.get("error")
    return res


def _autopilot(monkeypatch, tmp_path, term: _Terminal, journal: _Journal,
               res: dict, managed: bool = True) -> AP.Autopilot:
    ap = AP.Autopilot()
    ap.state_path = tmp_path / "autopilot.json"
    ap.st = ap._defaults()
    price = float(res["price"])
    ap.st.update({
        "mode": "live",
        "symbols": ["GOLD"],
        # Park the entry bar out of reach in the exit tests, so the only order
        # they can send is the exit under test.
        "entry_threshold": 2.0 if managed else 0.18,
        "last_rotation": AP._now(),
        "held": {"GOLD": {"started": AP._now(), "side": "BUY", "qty": 0.05,
                          "entry": round(price + 5, 4),
                          "stop": round(price + 1, 4),
                          "stop0": round(price + 1, 4),
                          "target": round(price * 1.2, 4)}} if managed else {},
    })
    monkeypatch.setattr(ap, "mt5", term)
    monkeypatch.setattr(ap, "_psx_book", lambda: None)
    monkeypatch.setattr(ap.gate, "check", lambda intent, acct, pos: (True, ""))
    monkeypatch.setattr(AP, "PaperBroker", _fresh_paper(tmp_path))
    monkeypatch.setattr(AP, "JOURNAL", journal)
    from kairos.engine import ENGINE
    monkeypatch.setattr(ENGINE, "predict", lambda key, horizon=None, **kw: res)
    monkeypatch.setattr(ENGINE, "refresh_news", lambda watch: {})
    return ap


# ------------------------------------------------------------------- the exit
def test_an_exit_closes_only_the_ticket_it_manages(monkeypatch, tmp_path):
    res = _prediction(monkeypatch)
    price = float(res["price"])
    # Two tickets in one symbol: the operator's own trade, and the one the
    # autopilot manages. The venue row the cycle reads names the managed one.
    term = _Terminal([_position(price, 99999), _position(price, 111111)])
    journal = _Journal()
    ap = _autopilot(monkeypatch, tmp_path, term, journal, res)

    ap.cycle()

    assert term.closed == [111111], "the exit closed a ticket it did not manage"
    assert term.deals == [], "the exit sent a fresh opposite deal"
    assert [r["ticket"] for r in term.positions()] == [99999]
    assert "GOLD" not in ap.st["held"]
    assert ap.st["consec_losses"] == 1
    assert journal.count("AUTOPILOT_BOOK_CLOSE") == 1
    assert ap.st["signals"][0]["action"] == "LIVE-FILLED"


def test_a_close_without_a_ticket_sends_nothing(monkeypatch, tmp_path):
    """No ticket means no safe close: an opposite deal on a hedging account is a
    new position, and a symbol sweep is somebody else's trade. Refuse loudly and
    keep managing the position so the next cycle can try again."""
    res = _prediction(monkeypatch)
    term = _Terminal([_position(float(res["price"]), None)])
    journal = _Journal()
    ap = _autopilot(monkeypatch, tmp_path, term, journal, res)

    ap.cycle()

    assert term.closed == [] and term.deals == []
    assert "GOLD" in ap.st["held"]
    assert ap.st["consec_losses"] == 0
    assert journal.count("AUTOPILOT_BOOK_CLOSE") == 0
    assert journal.count("AUTOPILOT_CLOSE_UNVERIFIED") == 1
    assert ap.st["signals"][0]["action"] == "CLOSE-UNVERIFIED"


def test_a_terminal_error_leaves_the_position_managed(monkeypatch, tmp_path):
    """A close that never reached the broker must not be booked as a close."""
    res = _prediction(monkeypatch)
    term = _Terminal([_position(float(res["price"]), 111111)])
    journal = _Journal()

    def _gone(ticket):
        raise RuntimeError("terminal gone")

    term.close_position = _gone
    ap = _autopilot(monkeypatch, tmp_path, term, journal, res)

    ap.cycle()

    assert "GOLD" in ap.st["held"]
    assert ap.st["consec_losses"] == 0
    assert journal.count("AUTOPILOT_BOOK_CLOSE") == 0
    assert journal.count("AUTOPILOT_CLOSE_FAIL") == 1
    assert ap.st["signals"][0]["action"] == "LIVE-REJECT"
    assert any(a["action"] == "CLOSE-FAILED" for a in ap.st["actions"])


# ------------------------------------------------------------------ the entry
def test_an_entry_still_goes_out_as_a_deal(monkeypatch, tmp_path):
    """The branch was split in two: the entry half must not have been closed
    off with the exit half."""
    res = _prediction(monkeypatch, battery=_AGREE)
    term = _Terminal([])
    journal = _Journal()
    ap = _autopilot(monkeypatch, tmp_path, term, journal, res, managed=False)

    out = ap.cycle()

    assert out["trades"] == 1, ap.st["signals"][0]
    assert len(term.deals) == 1
    assert term.deals[0].side == "BUY"
    assert term.closed == [], "an entry closed a position"
    assert ap.st["signals"][0]["action"] == "LIVE-FILLED"
