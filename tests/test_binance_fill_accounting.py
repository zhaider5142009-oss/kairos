"""An order the venue answered with nothing executed is not a fill.

The Binance branch of `_execute` booked any truthy `Fill` as `LIVE-FILLED`, and
`place_order` returns a Fill for every POST that got an answer. A MARKET order
into no liquidity comes back `EXPIRED` with zero executed quantity, so that
acknowledgement closed a position the venue still held, or opened a trade that
never existed - and the book, the ledger and the stop all followed the answer
rather than the tape.

The venue's own accounting of executed quantity and average price is covered in
`test_binance_venue.py`; these run the real order path against a stubbed client.
"""

from __future__ import annotations

from types import SimpleNamespace

import kairos.autopilot as AP
import kairos.binance as B
from kairos.autopilot import Autopilot
from kairos.execution import Fill, PaperBroker
from kairos.universe import Instrument
from test_doctrine_bound import _res
from test_psx_exit_working import _Journal

BTC = Instrument(key="BTC", name="Bitcoin", asset_class="crypto",
                 sector="digital assets", currency="USD", market="CRYPTO")


def _fill(qty: float, status: str, price: float = 0.0) -> Fill:
    return Fill(client_order_id="c1", broker_order_id="8", symbol="BTCUSDT",
                side="SELL", quantity=qty, price=price, commission=0.0,
                slippage=0.0, ts="2026-10-24T00:00:00", venue="BINANCE",
                status=status)


def _binance_ap(monkeypatch, tmp_path, fill: Fill) -> tuple[Autopilot, _Journal]:
    """The order path runs; nothing reaches the network."""
    ap = Autopilot()
    ap.state_path = tmp_path / "autopilot.json"
    ap.st = ap._defaults()
    ap._bn_armed = True
    journal = _Journal()
    client = SimpleNamespace(
        account=lambda: {"venue": "BINANCE", "equity": 10_000.0,
                         "balance": 10_000.0, "day_pnl": 0.0,
                         "open_positions": 0, "leverage": 1.0},
        positions=lambda: [],
        tick=lambda pair: {"bid": 100.0, "ask": 100.1, "spread": 0.1},
        place_order=lambda intent: fill,
        outcome=lambda coid: "PLACED")
    monkeypatch.setattr(AP, "JOURNAL", journal)
    monkeypatch.setattr(B, "BINANCE", client)
    monkeypatch.setattr(ap.gate, "check", lambda intent, acct, pos: (True, ""))
    return ap, journal


def _exit(ap: Autopilot) -> str:
    act, _detail = ap._execute(BTC, "SELL", 0.5, 100.0, _res(),
                               PaperBroker(capital=1_000_000.0),
                               live=True, close=True)
    return act


def test_an_order_that_struck_nothing_does_not_read_as_a_close(monkeypatch,
                                                               tmp_path):
    ap, journal = _binance_ap(monkeypatch, tmp_path, _fill(0.0, "EXPIRED"))

    assert _exit(ap) == "LIVE-EMPTY"
    assert journal.count("AUTOPILOT_UNFILLED") == 1


def test_a_resting_answer_is_not_a_close_either(monkeypatch, tmp_path):
    """An order that is accepted but has not struck anything has not exited the
    position, whatever the acknowledgement is called."""
    ap, journal = _binance_ap(monkeypatch, tmp_path, _fill(0.0, "NEW"))

    assert _exit(ap) == "LIVE-EMPTY"
    assert journal.count("AUTOPILOT_UNFILLED") == 1


def test_a_fill_that_struck_something_is_still_a_close(monkeypatch, tmp_path):
    """Control: the guard is about nothing executed, not about distrust."""
    ap, journal = _binance_ap(monkeypatch, tmp_path, _fill(0.5, "FILLED", 104.0))

    assert _exit(ap) == "LIVE-FILLED"
    assert journal.count("AUTOPILOT_UNFILLED") == 0
