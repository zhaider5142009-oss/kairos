"""Controls that read as limits but were not wired to anything.

Four defects these guard:

1. `rules.max_daily_loss_pct` and `risk.daily_loss_limit_pct` were one
   instruction with two names, two units and two enforcement points: the
   doctrine armed the autopilot's entry breaker, the risk block armed the
   pre-trade gate. Tightening the knob an operator meant moved one door.
   `daily_loss_fraction` now resolves one brake — the tighter reading — and
   every refusal names the key that produced it.
2. The posture computed risk numbers with bare `float()` calls, so an account
   feed that returned `"n/a"` raised out of the middle of the cycle that was
   meant to decide whether any order could go out at all. An unreadable figure
   is now zero where zero already means "no usable figure" (equity, peak), and
   its own blocker where zero would be a lie (the day's P&L, which the loss
   brake reads as a flat day, and a held position's price, which the exposure
   cap would read as free).
3. `features.compute_features` used `float(c.iloc[-1]) or 1.0` as the price
   denominator for every normalized feature. A halted or delisted close is not
   a missing divisor — it is a hole — and substituting one rupee scaled MACD,
   ATR%, momentum and each candle ratio by it, which is a *value*, which is
   what a model consumes without asking. The same non-positive print also
   reached `np.log` inside the Hurst, entropy and regression helpers, where it
   becomes -inf: enough to make `np.histogram` raise and `np.polyfit` fit a
   collapse, which is an exception out of the instrument loop.
4. `Autopilot._live_pos` mirrored the venue's position list into instance state
   and nothing read it back: two copies of a live book, one of them free to go
   stale.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from kairos import autopilot as AP
from kairos import execution as E
from kairos import features as F
from kairos import rules as R
from kairos.autopilot import Autopilot
from kairos.config import Config


# ------------------------------------------------------------------ one brake
def test_the_tighter_reading_governs_and_says_so():
    frac, src = R.daily_loss_fraction({"daily_loss_limit_pct": 0.10},
                                      {"max_daily_loss_pct": 0.5})
    assert frac == pytest.approx(0.005) and src == "rules.max_daily_loss_pct"
    frac, src = R.daily_loss_fraction({"daily_loss_limit_pct": 0.01},
                                      {"max_daily_loss_pct": 5.0})
    assert frac == pytest.approx(0.01) and src == "risk.daily_loss_limit_pct"


def test_the_two_units_mean_the_same_thing():
    """3% as a percent and 3% as a fraction are one limit, not a hundred."""
    frac, _ = R.daily_loss_fraction({"daily_loss_limit_pct": 0.03},
                                    {"max_daily_loss_pct": 3.0})
    assert frac == pytest.approx(0.03)


def test_missing_keys_fall_back_together():
    """Neither block names the brake; both defaults must describe the same one
    or the doctrine and the gate disagree on day one of a fresh install."""
    frac, src = R.daily_loss_fraction({}, {})
    assert frac == pytest.approx(R.DEFAULT_RULES["max_daily_loss_pct"] / 100.0)
    assert src.endswith("daily_loss_limit_pct")


@pytest.mark.parametrize("rules_cfg, risk_cfg", [
    ({"max_daily_loss_pct": "three percent"}, {}),
    ({}, {"daily_loss_limit_pct": None}),
    ({"max_daily_loss_pct": 0.0}, {}),
    ({}, {"daily_loss_limit_pct": -0.02}),
    ({"max_daily_loss_pct": float("nan")}, {}),
    ({}, {"daily_loss_limit_pct": float("inf")}),
])
def test_an_unreadable_brake_is_a_brake_at_zero_not_an_absent_one(rules_cfg, risk_cfg):
    """Zero means "the next order is refused", which is what an operator who
    typed nonsense deserves; `None` or a default would mean "no limit"."""
    frac, src = R.daily_loss_fraction(risk_cfg, rules_cfg)
    assert frac == 0.0 and "not a" in src


def test_the_gate_and_the_posture_read_one_brake_from_one_config(tmp_path, monkeypatch):
    """Two enforcement points must answer to the same number, or the pack's
    "one daily loss brake" sentence is a claim about a config file nobody
    checked."""
    f = tmp_path / "config.yaml"
    f.write_text("app:\n  data_dir: data\n"
                 "risk:\n  daily_loss_limit_pct: 0.02\n"
                 "rules:\n  max_daily_loss_pct: 5.0\n", encoding="utf-8")
    cfg = Config(f)
    monkeypatch.setattr(E, "CFG", cfg)
    monkeypatch.setattr(R, "CFG", cfg)
    monkeypatch.setattr(AP, "CFG", cfg)
    gate_limit = E.RiskGate()._limits()["daily_loss_limit"]

    a = Autopilot()
    a.state_path = tmp_path / "autopilot.json"
    a._save()
    posture = a._risk_posture({"equity": 100_000.0, "day_pnl": -100.0}, [])
    assert posture["daily_loss_limit_pct"] == pytest.approx(gate_limit * 100, abs=0.01)


# ------------------------------------------------------------------- posture
class _Cfg:
    """Only the `risk:` block is overridden; everything else is the test config."""

    def __init__(self, risk):
        self.risk = risk

    def section(self, name):
        return self.risk if name == "risk" else {}

    def __getattr__(self, item):
        from kairos.config import CFG as real
        return getattr(real, item)


@pytest.fixture
def ap(tmp_path, monkeypatch):
    monkeypatch.setattr(AP, "CFG", _Cfg({"daily_loss_limit_pct": 0.02}))
    monkeypatch.setattr(R, "rules", lambda: {
        "max_daily_loss_pct": 5.0, "max_total_exposure_pct": 60.0,
        "max_consecutive_losses": 4})
    a = Autopilot()
    a.state_path = tmp_path / "autopilot.json"
    a._save()
    return a


def test_the_posture_publishes_which_brake_is_live(ap):
    p = ap._risk_posture({"equity": 100_000.0, "day_pnl": -100.0}, [])
    assert p["daily_loss_limit_pct"] == pytest.approx(2.0)
    assert p["daily_loss_limit_from"] == "risk.daily_loss_limit_pct"
    assert p["block_entries"] is False


def test_the_doctrine_knob_reaches_the_posture_too(ap, monkeypatch):
    monkeypatch.setattr(R, "rules", lambda: {
        "max_daily_loss_pct": 0.5, "max_total_exposure_pct": 60.0,
        "max_consecutive_losses": 4})
    p = ap._risk_posture({"equity": 100_000.0, "day_pnl": -1_000.0}, [])
    assert p["daily_loss_limit_pct"] == pytest.approx(0.5)
    assert p["daily_loss_limit_from"] == "rules.max_daily_loss_pct"
    assert p["block_entries"] is True
    assert any("rules.max_daily_loss_pct" in b for b in p["blockers"])


def test_a_hole_in_the_account_feed_degrades_the_posture_instead_of_raising(ap):
    """The posture runs before any order is planned, so an exception here ends
    the cycle — the same cycle that manages every held stop. A reading that is
    not a number is a blocker that says which one, not a crash and not a pass."""
    for acct in ({"equity": "n/a", "day_pnl": None},
                 {"equity": float("nan"), "day_pnl": float("nan")},
                 {"equity": 100_000.0, "day_pnl": "n/a"}):
        p = ap._risk_posture(acct, [])
        assert p["block_entries"] is True, acct


def test_an_unusable_day_reading_blinds_the_brake_and_says_so(ap):
    """`float("n/a")` used to raise; a silent zero would have read "flat day"
    and let the next order through. Zero is the one value the loss brake cannot
    afford to invent, so an absent day is its own answer — at both doors."""
    p = ap._risk_posture({"equity": 100_000.0, "day_pnl": "n/a"}, [])
    assert p["day_pnl"] is None
    assert p["block_entries"] is True
    assert any("day P&L unreadable" in b for b in p["blockers"])
    assert not E.RiskGate().check(E.OrderIntent(symbol="OGDC", side="BUY",
                                                quantity=10),
                                  {"equity": 100_000.0, "day_pnl": "n/a"}, [])[0]


def test_a_held_position_with_no_readable_price_is_not_free(ap):
    """Summing zero for a row the venue could not value understates gross
    exposure against the cap that is the only brake on how much is at risk."""
    p = ap._risk_posture({"equity": 100_000.0, "day_pnl": 0.0},
                         [{"symbol": "OGDC", "qty": 100, "last": "n/a"}])
    assert p["exposure_pct"] == 0.0
    assert p["block_entries"] is True
    assert any("understated" in b for b in p["blockers"])
    # a ticket with no quantity genuinely risks nothing, and must not pause the
    # whole tower forever
    clear = ap._risk_posture({"equity": 100_000.0, "day_pnl": 0.0},
                             [{"symbol": "OGDC", "qty": 0, "last": 0.0}])
    assert clear["block_entries"] is False


def test_the_venue_position_list_is_not_mirrored_into_instance_state(ap):
    """A second copy of a live book, written every cycle and read by nobody, is
    how a manager ends up steering from data the venue already replaced."""
    assert not hasattr(ap, "_live_pos")


# ------------------------------------------------------------------- features
def _bars(n: int = 80, last_close: float | None = None) -> pd.DataFrame:
    rng = np.random.default_rng(20261022)
    close = 60 + np.cumsum(rng.normal(0.1, 0.6, n))
    high = close + rng.uniform(0.1, 1.2, n)
    low = close - rng.uniform(0.1, 1.2, n)
    open_ = close + rng.normal(0, 0.4, n)
    vol = rng.uniform(1e5, 9e5, n)
    if last_close is not None:
        close[-1] = last_close
        high[-1] = max(high[-1], last_close)
        low[-1] = min(low[-1], last_close)
    return pd.DataFrame({"open": open_, "high": high, "low": low,
                         "close": close, "volume": vol})


@pytest.mark.parametrize("feature", ["macd_norm", "macd_hist_norm", "momentum10",
                                     "atr14_pct"])
def test_a_zero_close_is_a_hole_not_a_one_rupee_price(feature):
    """`or 1.0` on the denominator turned a halted print into a price of one.

    The scale is not merely wrong, it is enormous: ATR% becomes the whole ATR in
    rupees, MACD a hundred times its normalized value, and the model battery
    reads a number rather than an absence.
    """
    good = F.compute_features(_bars())
    assert math.isfinite(good[feature]), "the same series must produce a value"
    halted = F.compute_features(_bars(last_close=0.0))
    assert math.isnan(halted[feature]), f"{feature} must be a hole, not a scaled guess"


def test_a_normal_close_still_produces_the_normalized_values():
    f = F.compute_features(_bars())
    for feature in ("macd_norm", "momentum10", "atr14_pct", "body_ratio"):
        assert feature in f and math.isfinite(f[feature])
