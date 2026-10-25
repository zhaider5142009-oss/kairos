"""One notional, three units: the size is the venue's question, not the loop's.

The cycle sizes an entry from the notional its risk layer cleared and then hands
the number to `_execute`, which routes it. Those two decisions used to be made in
the wrong order. The size was computed as `max(0.01, notional / price / 100)` —
MT5's CFD convention, where one lot is a hundred units and the broker's floor is a
hundredth of a lot — and applied to *every* live non-PSX name, including the ones
`_execute` sends to Binance spot, where an order is whole units of the asset
itself. A crypto entry therefore went out a hundred times too small; and once the
approved notional fell under the lot floor, `0.01` of whatever the name costs
became larger than the risk layer had allowed — a floor from one venue overrode a
cap from the other. Paper sized the same name in units, so the published paper
record was not the strategy being traded.

These run the real cycle against stubbed venues and hold both of them to the same
invariant: what reaches the venue, priced in its own unit, is the notional the
cluster cap cleared — no more, and not a hundred times less.
"""

from __future__ import annotations

import pytest

import kairos.autopilot as AP
import kairos.binance as B
from test_autopilot_cycle import _AGREE
from test_binance_fill_accounting import BTC
from test_mt5_close_ticket import _Terminal, _autopilot, _prediction
from test_partial_fill_booking import GOLD, _Client, _ap_binance, _gfill
from test_psx_exit_working import _Journal


def _cleared(monkeypatch, ap) -> list[float]:
    """Record the notional the cluster cap cleared, so the size can be judged
    against the number the risk layer actually approved."""
    real = ap._cluster_guard
    seen: list[float] = []

    def _guard(inst, side, notional, eq, pos_list, currency=""):
        out = real(inst, side, notional, eq, pos_list, currency)
        seen.append(float(out["allowed_notional"]))
        return out

    monkeypatch.setattr(ap, "_cluster_guard", _guard)
    return seen


def _capped(monkeypatch, ap, cap: float) -> None:
    """Force the cleared notional down to `cap`, so the test sizes the trade the
    risk layer approved rather than whatever a default book happens to allow."""
    real = ap._cluster_guard

    def _guard(inst, side, notional, eq, pos_list, currency=""):
        out = real(inst, side, notional, eq, pos_list, currency)
        return {**out, "allowed_notional": min(float(out["allowed_notional"]), cap)}

    monkeypatch.setattr(ap, "_cluster_guard", _guard)


class _Exact(_Client):
    """A spot book that trades exactly what it is asked for."""

    def place_order(self, intent):
        qty = float(intent.quantity)
        self.asked.append(qty)
        self.fill = _gfill(qty, "FILLED", self.price)
        self.rows.append({"symbol": "GOLDUSDT", "side": intent.side, "qty": qty,
                          "entry": self.price, "last": self.price, "pnl": 0.0})
        return self.fill


# ---------------------------------------------------------------- the spot size
def test_a_spot_entry_is_sized_in_units_of_the_asset(monkeypatch, tmp_path):
    res = _prediction(monkeypatch, battery=_AGREE)
    price = float(res["price"])
    client = _Exact(_gfill(0.0, "FILLED", price), price, [])
    ap, _journal = _ap_binance(monkeypatch, tmp_path, client, res, managed=False)
    cleared = _cleared(monkeypatch, ap)

    out = ap.cycle()

    assert out["trades"] == 1, ap.st["signals"][0]
    assert cleared and cleared[0] > 10
    asked = client.asked[0]
    # `round(qty, 4)`: the intent is sent at the venue's quantity precision
    assert asked == pytest.approx(cleared[0] / price, rel=1e-4)
    # the size the lot division used to send, for the record: same notional,
    # named in another venue's unit
    assert asked != round(max(0.01, cleared[0] / price / 100), 2)


def test_a_small_notional_is_not_traded_at_another_venue_s_lot_floor(monkeypatch,
                                                                     tmp_path):
    """A 0.01-lot floor belongs to MT5. Borrowed for spot it decides the size by a
    rule the venue never set: here it turned a 0.33-unit order into 0.01 of one,
    and on a name that costs more than the cap per unit it sends *more* than the
    risk layer approved — the floor overrides the cap either way."""
    res = _prediction(monkeypatch, battery=_AGREE)
    price = float(res["price"])
    cap = 40.0
    client = _Exact(_gfill(0.0, "FILLED", price), price, [])
    ap, _journal = _ap_binance(monkeypatch, tmp_path, client, res, managed=False)
    _capped(monkeypatch, ap, cap)

    ap.cycle()

    assert client.asked, ap.st["signals"][0]
    asked = client.asked[0]
    assert asked == pytest.approx(cap / price, rel=1e-4)
    assert asked * price <= cap * 1.001, "the size overrode the risk cap"


# ----------------------------------------------------------------- the CFD size
def test_a_cfd_entry_is_still_sized_in_lots(monkeypatch, tmp_path):
    """Control: MT5 was right, and the fix is venue-aware sizing, not unit
    sizing. A CFD there is dealt in lots of a hundred units; a spot-sized order
    on the same name would be a hundred trades."""
    res = _prediction(monkeypatch, battery=_AGREE)
    price = float(res["price"])
    monkeypatch.setattr(AP, "_binance_live_ready", lambda: False)
    term = _Terminal([])
    ap = _autopilot(monkeypatch, tmp_path, term, _Journal(), res, managed=False)
    cleared = _cleared(monkeypatch, ap)

    ap.cycle()

    assert len(term.deals) == 1, ap.st["signals"][0]
    lots = float(term.deals[0].quantity)
    # the largest whole lot that fits, stated as an invariant rather than a
    # formula: this size fits the cap, the next step up does not
    assert lots * price * 100 <= cleared[0] + 1e-9
    assert (lots + 0.01) * price * 100 > cleared[0]


# -------------------------------------------------------------------- the routing
def test_the_book_is_chosen_before_the_size_is(monkeypatch):
    """One answer, in one place: the venue decides the unit, so `PAPER` and an
    unarmed crypto book must not silently borrow MT5's convention."""
    ap = AP.Autopilot()
    monkeypatch.setattr(B, "pair_for",
                        lambda key: "BTCUSDT" if key == BTC.key else None)
    ap._bn_armed = True
    assert ap._venue_for(BTC, True, False) == "BINANCE"
    assert ap._venue_for(GOLD, True, False) == "MT5", "a name with no pair left"
    ap._bn_armed = False
    assert ap._venue_for(BTC, True, False) == "MT5", "an unarmed book was named"
    assert ap._venue_for(BTC, False, False) == "PAPER"
    assert ap._venue_for(BTC, True, True) == "JS_GLOBAL"
