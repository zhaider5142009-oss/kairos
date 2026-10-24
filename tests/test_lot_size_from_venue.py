"""A lot is the broker's number, and a size that cannot fit the cap is a veto.

Two assumptions used to be baked into every CFD order. The first: a lot is a
hundred units and the smallest order is 0.01 of them. Those are per-symbol
settings the terminal owns — a 1,000-barrel oil CFD and a 100-ounce gold CFD are
not the same hundred dollars, and sizing from the assumption mis-sizes every
order on the symbol in whichever direction the broker chose. The second: the
smallest tradeable position was treated as a size to round *up* to. A risk cap is
a ceiling, so rounding up to the floor placed an order the risk layer had never
approved — and when the venue's minimum cost more than the whole approved
notional, the name is simply untradeable at that account size, which is a veto,
not a trade.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import kairos.autopilot as AP
from kairos.autopilot import _size_for
from kairos.execution import Mt5Bridge
from test_autopilot_cycle import _AGREE
from test_mt5_close_ticket import _Terminal, _autopilot, _prediction
from test_psx_exit_working import _Journal
from test_spot_size_convention import _capped, _cleared

GOLD_SPEC = {"contract_size": 100.0, "volume_min": 0.01, "volume_step": 0.01}


# ------------------------------------------------------------------ the helper
def test_a_cfd_size_is_the_largest_whole_step_that_fits_the_cap():
    lots = _size_for("MT5", 18_000.0, 120.0, GOLD_SPEC)
    assert lots == 1.5
    assert (lots + 0.01) * 120.0 * 100.0 > 18_000.0


def test_the_contract_size_comes_from_the_terminal_not_from_the_code():
    """Same notional, same price, a broker whose lot is a thousand units: the
    size changes by ten, because it was read rather than assumed."""
    wide = {**GOLD_SPEC, "contract_size": 1000.0}
    assert _size_for("MT5", 18_000.0, 120.0, wide) == pytest.approx(0.15)
    assert _size_for("MT5", 18_000.0, 120.0, None) == 1.5      # assumed fallback


def test_a_step_of_five_hundredths_is_not_rounded_to_a_hundredth():
    """A broker that deals in 0.05 lots gets 0.05 multiples; the size is still the
    largest one under the cap, not the nearest."""
    step = {**GOLD_SPEC, "volume_step": 0.05}
    lots = _size_for("MT5", 18_000.0, 120.0, step)
    assert lots == 1.50
    assert abs(lots / 0.05 - round(lots / 0.05)) < 1e-9


def test_a_size_under_the_venue_s_minimum_is_no_trade():
    """0.01 lots of a name quoted at 2,650 costs 2,650, and the risk layer
    approved 40. Rounding up to the floor would place an order sixty-six times the
    approved size, so the answer is no order at all. The same rule makes a share
    price above the cap untradeable on PSX; a spot unit has no such floor."""
    assert _size_for("MT5", 40.0, 2_650.0, GOLD_SPEC) == 0.0
    assert _size_for("JS_GLOBAL", 1_000.0, 233.4, None) == 4.0     # whole shares
    assert _size_for("JS_GLOBAL", 233.0, 233.4, None) == 0.0
    assert _size_for("BINANCE", 40.0, 100.0, None) == 0.4          # units


def test_an_unreadable_number_sizes_nothing():
    for bad in [(0.0, 100.0), (100.0, 0.0), (-500.0, 100.0), (float("nan"), 100.0),
                (100.0, float("inf"))]:
        assert _size_for("MT5", bad[0], bad[1], GOLD_SPEC) == 0.0


# ------------------------------------------------------------------ the bridge
def _bridge(info) -> Mt5Bridge:
    b = Mt5Bridge()
    b.connected = True
    b._mt5 = SimpleNamespace(symbol_info=lambda symbol: info)
    return b


def test_the_bridge_reads_the_three_settings_that_decide_a_size():
    info = SimpleNamespace(trade_contract_size=1000.0, volume_min=0.1,
                           volume_step=0.1)
    assert _bridge(info).spec("XAUUSD") == {"contract_size": 1000.0,
                                            "volume_min": 0.1,
                                            "volume_step": 0.1}


def test_a_half_filled_spec_is_no_spec(monkeypatch):
    """A zero step or contract size is not a trading rule, and reading one as a
    number divides the order by zero or floors it to nothing."""
    for bad in [SimpleNamespace(trade_contract_size=0.0, volume_min=0.01,
                                volume_step=0.01),
                SimpleNamespace(trade_contract_size=100.0, volume_min=0.01,
                               volume_step=0.0),
                None]:
        assert _bridge(bad).spec("XAUUSD") is None
    down = _bridge(SimpleNamespace(trade_contract_size=100.0, volume_min=0.01,
                                   volume_step=0.01))
    down.connected = False
    assert down.spec("XAUUSD") is None


# -------------------------------------------------------------------- the loop
def test_a_name_the_venue_cannot_trade_is_a_veto_not_an_order(monkeypatch, tmp_path):
    """One lot step at this quote costs more than the whole approved notional.
    The old code rounded *up* to it and sent the order; the risk layer's cap is a
    ceiling, so the name is untradeable here and the desk says so."""
    res = _prediction(monkeypatch, battery=_AGREE)
    price = float(res["price"])
    cap = 50.0
    assert cap < GOLD_SPEC["volume_min"] * GOLD_SPEC["contract_size"] * price
    monkeypatch.setattr(AP, "_binance_live_ready", lambda: False)
    term = _Terminal([])
    term.spec = lambda symbol: dict(GOLD_SPEC)            # type: ignore[method-assign]
    journal = _Journal()
    ap = _autopilot(monkeypatch, tmp_path, term, journal, res, managed=False)
    _capped(monkeypatch, ap, cap)

    out = ap.cycle()

    assert out["trades"] == 0, ap.st["signals"][0]
    assert term.deals == [], "the venue minimum was traded anyway"
    assert ap.st["signals"][0]["action"] == "MIN-LOT"
    assert journal.count("AUTOPILOT_MIN_LOT") == 1
    assert "GOLD" not in ap.st["held"]


def test_a_name_the_venue_can_trade_still_goes_out(monkeypatch, tmp_path):
    """Control: the veto is about the cap, not about distrust of the venue."""
    res = _prediction(monkeypatch, battery=_AGREE)
    price = float(res["price"])
    monkeypatch.setattr(AP, "_binance_live_ready", lambda: False)
    term = _Terminal([])
    term.spec = lambda symbol: dict(GOLD_SPEC)            # type: ignore[method-assign]
    ap = _autopilot(monkeypatch, tmp_path, term, _Journal(), res, managed=False)
    cleared = _cleared(monkeypatch, ap)

    out = ap.cycle()

    assert out["trades"] == 1, ap.st["signals"][0]
    assert len(term.deals) == 1
    lots = float(term.deals[0].quantity)
    assert lots * price * 100.0 <= cleared[0] + 1e-9
