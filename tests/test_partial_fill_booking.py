"""A venue that struck part of an order traded part of it, not all of it.

Two answers used to be read as one. `Mt5Bridge.place_order` accepted
`TRADE_RETCODE_DONE_PARTIAL` — whose entire message is that less was traded than
was asked for — and built the fill at the volume from the request. The autopilot
then booked its managed row at the size it had *asked* for, and a close that came
back partial was booked as a whole close: the P&L of quantity nobody traded went
into the ledger, and the remainder the venue still held lost the stop, the target
and the maturity date it was being managed by.

These run the real bridge and the real cycle against stubbed venues. No terminal,
no exchange, no network.
"""

from __future__ import annotations

import kairos.autopilot as AP
import kairos.binance as B
import kairos.execution as E
from kairos.execution import Fill, Mt5Bridge, OrderIntent
from kairos.universe import Instrument
from test_autopilot_cycle import _AGREE, _fresh_paper
from test_doctrine_bound import _res
from test_mt5_close_ticket import _Terminal, _prediction
from test_psx_exit_working import _Journal


class _Writing(_Journal):
    """A trail that accepts every write: `place_order` refuses to send an order it
    cannot record, and this suite is about the size of the answer, not the journal."""

    def log(self, event: str, payload: dict | None = None) -> bool:
        super().log(event, payload)
        return True

GOLD = Instrument(key="GOLD", name="Gold", asset_class="commodity",
                  sector="materials", currency="USD", market="GLOBAL")


def _gfill(qty: float, status: str, price: float) -> Fill:
    """A Binance spot answer for the pair KAIROS trades GOLD under."""
    return Fill(client_order_id="c1", broker_order_id="8", symbol="GOLDUSDT",
                side="SELL", quantity=qty, price=price, commission=0.0,
                slippage=0.0, ts="2026-10-24T00:00:00", venue="BINANCE",
                status=status)


# ------------------------------------------------------- the bridge answers
class _TradeResult:
    def __init__(self, retcode: int, order: int, volume: float, price: float):
        self.retcode = retcode
        self.order = order
        self.volume = volume
        self.price = price
        self.comment = ""


class _Mt5:
    """The module slice `place_order` touches, with a retcode and volume knob."""

    TRADE_ACTION_DEAL = 1
    ORDER_TYPE_BUY = 0
    ORDER_TYPE_SELL = 1
    ORDER_TIME_GTC = 0
    ORDER_FILLING_IOC = 1
    TRADE_RETCODE_DONE = 10009
    TRADE_RETCODE_DONE_PARTIAL = 10010

    def __init__(self, result: _TradeResult) -> None:
        self.result = result
        self.sent: list[dict] = []

    def order_send(self, request: dict):
        self.sent.append(request)
        return self.result


def _bridge(result: _TradeResult) -> Mt5Bridge:
    b = Mt5Bridge()
    b.connected = True
    b._mt5 = _Mt5(result)
    b.resolve_symbol = lambda key: "XAUUSD"                    # type: ignore[method-assign]
    b.tick = lambda symbol: {"bid": 100.0, "ask": 100.1,
                             "spread": 0.1}                   # type: ignore[method-assign]
    return b


def _intent(qty: float) -> OrderIntent:
    return OrderIntent(symbol="GOLD", side="BUY", quantity=qty, venue="MT5")


def test_a_partial_mt5_deal_reports_the_volume_that_was_struck(monkeypatch):
    """DONE_PARTIAL is an accepted order, and it says 0.02 of 0.05 went out."""
    monkeypatch.setattr(E, "JOURNAL", _Writing())
    b = _bridge(_TradeResult(_Mt5.TRADE_RETCODE_DONE_PARTIAL, 555, 0.02, 100.0))

    fill = b.place_order(_intent(0.05))

    assert fill is not None
    assert fill.quantity == 0.02, "the fill named the order, not the trade"


def test_a_whole_mt5_deal_still_reports_the_whole_order(monkeypatch):
    """Control: the venue's volume is trusted, not doubted into a smaller number."""
    monkeypatch.setattr(E, "JOURNAL", _Writing())
    b = _bridge(_TradeResult(_Mt5.TRADE_RETCODE_DONE, 556, 0.05, 100.0))

    fill = b.place_order(_intent(0.05))

    assert fill is not None and fill.quantity == 0.05


def test_a_bridge_that_names_no_volume_does_not_become_a_zero_trade(monkeypatch):
    """Some builds leave `volume` at 0 on a done deal. Falling back to the request
    keeps those working; reporting the 0 would turn a real trade into one for
    nothing at all."""
    monkeypatch.setattr(E, "JOURNAL", _Writing())
    b = _bridge(_TradeResult(_Mt5.TRADE_RETCODE_DONE, 557, 0.0, 100.0))

    fill = b.place_order(_intent(0.05))

    assert fill is not None and fill.quantity == 0.05


# ---------------------------------------------------------- the order path
class _Client:
    """A Binance book that answers with one prepared fill and nothing else."""

    def __init__(self, fill: Fill, price: float, rows: list[dict]) -> None:
        self.fill = fill
        self.price = price
        self.rows = rows
        self.asked: list[float] = []

    def account(self) -> dict:
        return {"venue": "BINANCE", "equity": 100_000.0, "balance": 100_000.0,
                "day_pnl": 0.0, "open_positions": len(self.rows), "leverage": 1.0}

    def positions(self) -> list[dict]:
        return [dict(r) for r in self.rows]

    def tick(self, pair: str) -> float:
        # the spot client answers with the price it last saw, not a book
        return self.price

    def outcome(self, coid: str) -> str:
        return "PLACED"

    def place_order(self, intent) -> Fill:
        self.asked.append(float(intent.quantity))
        rows = self.fill.quantity
        if rows > 0:
            self.rows.append({"symbol": "GOLDUSDT", "side": intent.side,
                              "qty": rows, "entry": self.price,
                              "last": self.price, "pnl": 0.0})
        return self.fill


def _execute_close(monkeypatch, tmp_path, fill: Fill):
    """One real close order through the real order path."""
    ap = AP.Autopilot()
    ap.state_path = tmp_path / "autopilot.json"
    ap.st = ap._defaults()
    ap._bn_armed = True
    journal = _Journal()
    client = _Client(fill, 100.0, [])
    monkeypatch.setattr(AP, "JOURNAL", journal)
    monkeypatch.setattr(B, "BINANCE", client)
    monkeypatch.setattr(B, "pair_for", lambda key: "GOLDUSDT")
    monkeypatch.setattr(ap.gate, "check", lambda intent, acct, pos: (True, ""))
    paper = _fresh_paper(tmp_path)()
    act, _d, struck = ap._execute(GOLD, "SELL", 0.5, 100.0, _res(), paper,
                                  live=True, close=True)
    return act, struck, journal


def test_a_close_that_struck_part_is_not_a_close(monkeypatch, tmp_path):
    """0.2 of 0.5: the venue answered, and its answer is that most of the position
    is still open."""
    act, struck, journal = _execute_close(monkeypatch, tmp_path,
                                          _gfill(0.2, "PARTIALLY_FILLED", 100.0))

    assert act == "LIVE-PARTIAL"
    assert struck == 0.2
    partial = [p for e, p in journal.rows if e == "AUTOPILOT_PARTIAL"]
    assert len(partial) == 1
    assert partial[0]["asked"] == 0.5 and partial[0]["struck"] == 0.2
    assert partial[0]["close"] is True


def test_a_whole_close_is_still_a_close(monkeypatch, tmp_path):
    """Control: nothing here distrusts a venue that filled the order."""
    act, struck, journal = _execute_close(monkeypatch, tmp_path,
                                          _gfill(0.5, "FILLED", 100.0))

    assert act == "LIVE-FILLED" and struck == 0.5
    assert journal.count("AUTOPILOT_PARTIAL") == 0


# --------------------------------------------------------------- the cycle
def _position_row(price: float, qty: float = 0.5) -> dict:
    return {"symbol": "GOLDUSDT", "side": "BUY", "qty": qty,
            "entry": round(price + 5, 4), "last": price,
            "pnl": round(-5 * qty, 2)}


def _ap_binance(monkeypatch, tmp_path, client, res, managed: bool):
    """The real loop, with GOLD resting on an armed Binance book."""
    ap = AP.Autopilot()
    ap.state_path = tmp_path / "autopilot.json"
    ap.st = ap._defaults()
    price = float(res["price"])
    ap.st.update({
        "mode": "live", "symbols": ["GOLD"],
        # park the entry bar out of reach unless this cycle is here to enter
        "entry_threshold": 2.0 if managed else 0.18,
        "last_rotation": AP._now(),
        "held": {"GOLD": {"started": AP._now(), "side": "BUY", "qty": 0.5,
                          "entry": round(price + 5, 4),
                          "stop": round(price + 1, 4),
                          "stop0": round(price + 1, 4),
                          "target": round(price * 1.2, 4)}} if managed else {},
    })
    journal = _Journal()
    monkeypatch.setattr(ap, "mt5", _Terminal([]))
    monkeypatch.setattr(ap, "_psx_book", lambda: None)
    monkeypatch.setattr(ap.gate, "check", lambda intent, acct, pos: (True, ""))
    monkeypatch.setattr(AP, "PaperBroker", _fresh_paper(tmp_path))
    monkeypatch.setattr(AP, "JOURNAL", journal)
    monkeypatch.setattr(AP, "_binance_live_ready", lambda: True)
    monkeypatch.setattr(B, "pair_for",
                        lambda key: "GOLDUSDT" if key == "GOLD" else None)
    monkeypatch.setattr(B, "BINANCE", client)
    from kairos.engine import ENGINE
    monkeypatch.setattr(ENGINE, "predict", lambda key, horizon=None, **kw: res)
    monkeypatch.setattr(ENGINE, "refresh_news", lambda watch: {})
    return ap, journal


def test_a_partial_close_books_only_the_part_that_traded_and_keeps_the_rest(
        monkeypatch, tmp_path):
    """The stop is breached and 0.2 of the 0.5 closes. The ledger may count two
    tenths, and the three tenths the venue still holds must keep its stop: booking
    the whole close realised a loss nobody took *and* dropped the row every exit
    is read from."""
    res = _prediction(monkeypatch, battery=_AGREE)
    price = float(res["price"])
    rows = [_position_row(price)]
    client = _Client(_gfill(0.2, "PARTIALLY_FILLED", price), price, rows)
    ap, journal = _ap_binance(monkeypatch, tmp_path, client, res, managed=True)

    ap.cycle()

    assert ap.st["signals"][0]["action"] == "LIVE-PARTIAL", ap.st["signals"][0]
    hb = ap.st["held"].get("GOLD")
    assert hb is not None, "the remainder lost the row its exits are read from"
    assert float(hb["qty"]) == 0.3
    booked = [p for e, p in journal.rows if e == "AUTOPILOT_BOOK_CLOSE"]
    assert len(booked) == 1
    # -5 a unit on the two tenths that closed, not on the half the book holds
    assert abs(booked[0]["pnl_est"] - (-5 * 0.2)) < 0.01
    assert float(booked[0]["qty"]) == 0.2


def test_a_whole_close_books_the_position_and_ends_the_row(monkeypatch, tmp_path):
    """Control: the exit that fills is still a completed exit."""
    res = _prediction(monkeypatch, battery=_AGREE)
    price = float(res["price"])
    rows = [_position_row(price)]
    client = _Client(_gfill(0.5, "FILLED", price), price, rows)

    def _flat(intent) -> Fill:
        client.asked.append(float(intent.quantity))
        rows.clear()
        return client.fill

    client.place_order = _flat                                   # type: ignore[method-assign]
    ap, journal = _ap_binance(monkeypatch, tmp_path, client, res, managed=True)

    ap.cycle()

    assert ap.st["signals"][0]["action"] == "LIVE-FILLED", ap.st["signals"][0]
    assert "GOLD" not in ap.st["held"]
    booked = [p for e, p in journal.rows if e == "AUTOPILOT_BOOK_CLOSE"]
    assert len(booked) == 1 and float(booked[0]["qty"]) == 0.5


def test_a_partial_entry_is_booked_at_the_size_that_was_struck(monkeypatch,
                                                               tmp_path):
    """The managed row is both the risk row and the size the desk reports. Sizing
    it at the order rather than the fill puts a number on the screen that no venue
    ever traded."""
    res = _prediction(monkeypatch, battery=_AGREE)
    price = float(res["price"])
    client = _Client(_gfill(0.01, "PARTIALLY_FILLED", price), price, [])
    ap, journal = _ap_binance(monkeypatch, tmp_path, client, res, managed=False)

    ap.cycle()

    assert ap.st["signals"][0]["action"] == "LIVE-FILLED", ap.st["signals"][0]
    assert client.asked and client.asked[0] > 0.01, "no partial to find"
    hb = ap.st["held"].get("GOLD")
    assert hb is not None
    assert float(hb["qty"]) == 0.01, "the book is sized at the order, not the trade"
    assert journal.count("AUTOPILOT_PARTIAL") == 1
