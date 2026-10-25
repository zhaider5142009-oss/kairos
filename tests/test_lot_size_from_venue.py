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

A spot pair has the same two settings under other names — a quantity ladder, a
minimum order and a minimum order value — and the sizing layer has to know them
before it chooses a size, or the desk computes a trade the exchange declines to
accept and spends a signed request to find out.

The third assumption was the friendliest-looking one: read the settings when the
terminal gives them, fall back to a hundred-unit lot when it does not. The MT5
bridge sends the size it is handed as the order's volume, so there is no second
check to catch the guess — a silent terminal and a 1,000-unit lot together put
ten times the approved notional on the market. No answer from the broker is no
order.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import kairos.autopilot as AP
from kairos.autopilot import _size_for
from kairos.execution import Mt5Bridge
from test_autopilot_cycle import _AGREE
from test_mt5_close_ticket import _Terminal, _autopilot, _prediction
from test_partial_fill_booking import _ap_binance, _gfill
from test_psx_exit_working import _Journal
from test_spot_size_convention import _Exact, _capped, _cleared

GOLD_SPEC = {"contract_size": 100.0, "volume_min": 0.01, "volume_step": 0.01}
# a pair in BTCUSDT's own shape: a hundred-thousandth of a coin, five dollars
# at a time at least
BTC_RULES = {"step_size": 0.00001, "min_qty": 0.00001, "min_notional": 5.0}
THOUSANDTHS = {"step_size": 0.001, "min_qty": 0.001, "min_notional": 5.0}
WHOLE_UNITS = {"step_size": 1.0, "min_qty": 1.0, "min_notional": 5.0}


# ------------------------------------------------------------------ the helper
def test_a_cfd_size_is_the_largest_whole_step_that_fits_the_cap():
    lots = _size_for("MT5", 18_000.0, 120.0, GOLD_SPEC)
    assert lots == 1.5
    assert (lots + 0.01) * 120.0 * 100.0 > 18_000.0


def test_the_contract_size_comes_from_the_terminal_not_from_the_code():
    """Same notional, same price, a broker whose lot is a thousand units: the
    size changes by ten, because it was read rather than assumed. A terminal
    that gives no answer gives no size either — the ten-times error is exactly
    what the read was there to prevent, so falling back to a hundred units on
    silence would keep it."""
    wide = {**GOLD_SPEC, "contract_size": 1000.0}
    assert _size_for("MT5", 18_000.0, 120.0, wide) == pytest.approx(0.15)
    assert _size_for("MT5", 18_000.0, 120.0, None) == 0.0


def test_a_spec_with_one_number_missing_is_no_spec():
    """The three settings are one answer, not three. A dict that names a lot and
    a minimum but no step would be read as "step of zero" and the order floored
    onto a ladder that does not exist; a partial read is a terminal that has not
    answered."""
    for partial in [{}, {"contract_size": 100.0},
                    {"contract_size": 100.0, "volume_min": 0.01},
                    {"volume_step": 0.01, "volume_min": 0.01},
                    {"contract_size": 100.0, "volume_min": 0.01,
                     "volume_step": 0.0}]:
        assert _size_for("MT5", 18_000.0, 120.0, partial) == 0.0


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


def test_a_terminal_that_will_not_name_the_lot_trades_nothing(monkeypatch, tmp_path):
    """The cap here is generous — the name is tradeable. What is missing is the
    broker's answer to "how big is a lot", and the number this path sends is the
    size itself, as the order's volume. Guess a hundred units on a 1,000-unit lot
    and the trade is ten times what the risk layer approved, with nothing further
    down the line to notice. So an unreadable lot is a veto, and it is reported as
    the terminal's silence rather than as a cap the name failed."""
    res = _prediction(monkeypatch, battery=_AGREE)
    monkeypatch.setattr(AP, "_binance_live_ready", lambda: False)
    term = _Terminal([])
    term.spec = lambda symbol: None                        # type: ignore[method-assign]
    journal = _Journal()
    ap = _autopilot(monkeypatch, tmp_path, term, journal, res, managed=False)
    cleared = _cleared(monkeypatch, ap)

    out = ap.cycle()

    assert out["trades"] == 0, ap.st["signals"][0]
    assert term.deals == [], "an order was placed on an assumed contract size"
    assert cleared and cleared[0] > 100.0, "the fixture no longer clears a lot"
    signal = ap.st["signals"][0]
    assert signal["action"] == "MIN-LOT"
    assert "would not state" in signal["detail"], signal["detail"]
    assert journal.count("AUTOPILOT_MIN_LOT") == 1
    assert "GOLD" not in ap.st["held"]


# ---------------------------------------------------------------- the spot ladder
def test_a_spot_size_is_the_largest_ladder_step_that_fits_the_cap():
    """notional/price gives 10.0455 units on a pair that deals in thousandths:
    the tradeable size is 10.045, and the next tick does not fit the cap."""
    qty = _size_for("BINANCE", 1004.55, 100.0, THOUSANDTHS)
    assert qty == 10.045
    assert (qty + 0.001) * 100.0 > 1004.55


def test_a_size_below_the_pair_s_minimum_quantity_is_no_order():
    """Half of the smallest quantity the pair deals in is not a rounded-up
    minimum — it is no order at all."""
    step_only = {"step_size": 0.001, "min_qty": 0.001}
    assert _size_for("BINANCE", 0.05, 100.0, step_only) == 0.0
    assert _size_for("BINANCE", 0.15, 100.0, step_only) == pytest.approx(0.001)


def test_a_size_below_the_pair_s_minimum_value_is_no_order():
    """BTCUSDT refuses anything worth under 5 USDT however many coins it is, so a
    4-dollar cap buys nothing on that pair. The quantity is legal; the order is
    not — which is the second way a venue's settings veto a size."""
    assert _size_for("BINANCE", 4.0, 100.0, THOUSANDTHS) == 0.0
    assert _size_for("BINANCE", 6.0, 100.0, THOUSANDTHS) == pytest.approx(0.06)


def test_a_pair_with_no_published_rules_sizes_in_plain_units():
    """Fail-open on the settings read: not knowing the ladder is no reason to
    refuse a trade the exchange may well accept."""
    assert _size_for("BINANCE", 40.0, 100.0, {}) == 0.4
    assert _size_for("BINANCE", 40.0, 100.0, {"step_size": 0.0}) == 0.4


class _Ladder(_Exact):
    """A spot book that publishes one pair's rules and counts the reads."""

    def __init__(self, fill, price: float, rows: list[dict], rules: dict) -> None:
        super().__init__(fill, price, rows)
        self.rules = rules
        self.reads = 0

    def filters(self, pair: str) -> dict:
        self.reads += 1
        return dict(self.rules)


def _capped_and_recorded(monkeypatch, ap, cap: float) -> list[float]:
    """Hold the cleared notional to `cap` and report what the trade was sized on."""
    real = ap._cluster_guard
    seen: list[float] = []

    def _guard(inst, side, notional, eq, pos_list, currency=""):
        out = real(inst, side, notional, eq, pos_list, currency)
        allowed = min(float(out["allowed_notional"]), cap)
        seen.append(allowed)
        return {**out, "allowed_notional": allowed}

    monkeypatch.setattr(ap, "_cluster_guard", _guard)
    return seen


# ----------------------------------------------------------------------- the loop
def test_a_size_the_exchange_would_decline_is_a_veto_not_a_request(monkeypatch,
                                                                    tmp_path):
    """One unit is this pair's smallest quantity and the cap cannot buy one. The
    bridge would truncate it to zero and refuse; the desk gets there first, so
    the loop records why nothing happened without spending a signed call."""
    res = _prediction(monkeypatch, battery=_AGREE)
    price = float(res["price"])
    cap = round(price * 0.6, 2)
    assert 10 < cap < price, "the fixture no longer straddles the whole-unit ladder"
    client = _Ladder(_gfill(0.0, "FILLED", price), price, [], WHOLE_UNITS)
    ap, journal = _ap_binance(monkeypatch, tmp_path, client, res, managed=False)
    _capped(monkeypatch, ap, cap)

    out = ap.cycle()

    assert out["trades"] == 0, ap.st["signals"][0]
    assert client.asked == [], "an order the exchange had already declined was sent"
    assert ap.st["signals"][0]["action"] == "MIN-LOT"
    assert journal.count("AUTOPILOT_MIN_LOT") == 1
    assert client.reads == 1, "the pair's settings were read once per entry attempt"
    assert "GOLD" not in ap.st["held"]


def test_a_size_on_the_pair_s_own_ladder_reaches_the_venue_whole(monkeypatch,
                                                                 tmp_path):
    """A hundred-thousandth ladder is the exchange's, so the size chosen for it
    has five decimals. The number carried to the venue used to be rounded to four
    first — which rounds 12.34567 up to 12.3457, a size the exchange accepts and
    the risk layer never approved."""
    res = _prediction(monkeypatch, battery=_AGREE)
    price = float(res["price"])
    # between two ladder steps and nearer the lower one, so the size below it is
    # decided by the ladder rather than by float noise
    cap = price * 12.3456745
    client = _Ladder(_gfill(0.0, "FILLED", price), price, [], BTC_RULES)
    ap, _journal = _ap_binance(monkeypatch, tmp_path, client, res, managed=False)
    cleared = _capped_and_recorded(monkeypatch, ap, cap)

    out = ap.cycle()

    assert out["trades"] == 1, ap.st["signals"][0]
    assert cleared and cleared[0] == pytest.approx(cap, rel=1e-6)
    asked = client.asked[0]
    assert asked == pytest.approx(12.34567, abs=1e-9), "the ladder was damaged on the way"
    assert asked * price <= cap + 1e-9, "the size spent past the approved cap"
    assert (asked + 0.00001) * price > cap, "the size left a step on the table"
