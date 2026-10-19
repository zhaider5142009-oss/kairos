"""Live arming without MetaTrader: the PSX venue must stand on its own, and
every risk number that moves real rupees must come from the REAL JS Global
book — never from the paper account or a USD MT5 figure.

All offline: the portal is stubbed, so nothing here can reach the broker.
"""

import pytest

from kairos import autopilot as AP
from kairos import jsgsync, jsgtrade
from kairos.autopilot import Autopilot


@pytest.fixture
def ap(tmp_path, monkeypatch):
    monkeypatch.setattr(AP, "CFG", _RealCfg())
    a = Autopilot()
    a.state_path = tmp_path / "autopilot.json"
    a._save()
    return a


class _RealCfg:
    """The real config object; only the autopilot section is overridden."""

    def __init__(self):
        from kairos.config import CFG as real
        self._real = real

    def section(self, name):
        if name == "autopilot":
            return {"auto_place": True}
        return self._real.section(name)

    def __getattr__(self, item):
        return getattr(self._real, item)


def _book(cash=6500.52, holdings=None):
    h = holdings if holdings is not None else [
        {"symbol": "OGDC", "qty": 3.0, "avg": 316.92, "mtm": 299.47,
         "value": 898.41, "pending_sell": 0.0, "account": "52366"},
        {"symbol": "PPL", "qty": 15.0, "avg": 227.39, "mtm": 218.85,
         "value": 3282.75, "pending_sell": 0.0, "account": "52366"},
    ]
    value = round(sum(x["value"] for x in h), 2)
    return {"ok": True, "venue": "JS_GLOBAL", "accounts": ["MAIMOONA FARRUKH"],
            "cash": cash, "holdings_value": value,
            "equity": round(cash + value, 2),
            "equity_basis": "free cash + mark-to-market holdings",
            "holdings": h, "ts": "2026-10-19 18:00:00"}


class _Inst:
    def __init__(self, key):
        self.key = key


def _never_places(monkeypatch):
    def boom(*a, **kw):
        raise AssertionError("reached the broker in an offline test")
    monkeypatch.setattr(jsgtrade, "place_now", boom)
    monkeypatch.setattr(jsgtrade, "enabled", lambda: True)


# ------------------------------------------------------------------ arming
def test_live_arms_with_no_mt5_when_the_psx_keys_are_on(ap, monkeypatch):
    """The blocker this whole change removes: PSX-only arming."""
    monkeypatch.setattr(AP, "_psx_live_ready", lambda: True)
    monkeypatch.setattr(AP.KILL, "engaged", lambda: False)
    ap.mt5.connected = False
    monkeypatch.setattr(ap, "_mt5_connect", lambda: False)
    out = ap.start(mode="live", confirm_live=True)
    try:
        assert out["ok"] is True
        assert "JS Global PSX" in out["note"]
    finally:
        ap.stop()


def test_live_still_refuses_with_no_venue(ap, monkeypatch):
    monkeypatch.setattr(AP, "_psx_live_ready", lambda: False)
    monkeypatch.setattr(AP.KILL, "engaged", lambda: False)
    ap.mt5.connected = False
    monkeypatch.setattr(ap, "_mt5_connect", lambda: False)
    out = ap.start(mode="live", confirm_live=True)
    assert out["ok"] is False
    assert "no live venue" in out["error"]


def test_live_needs_the_confirm_and_a_clear_kill_switch(ap, monkeypatch):
    monkeypatch.setattr(AP, "_psx_live_ready", lambda: True)
    assert ap.start(mode="live", confirm_live=False)["ok"] is False
    monkeypatch.setattr(AP.KILL, "engaged", lambda: True)
    out = ap.start(mode="live", confirm_live=True)
    assert out["ok"] is False
    assert "kill switch" in out["error"]


# ------------------------------------------------------------- real account
def test_posture_runs_on_the_real_book_not_the_paper_book(ap, monkeypatch):
    monkeypatch.setattr(jsgsync, "real_book", lambda *a, **kw: _book())
    book = ap._psx_book()
    acct = ap._live_account(False, book)
    assert acct["equity"] == 10681.68          # the actual funded account
    assert acct["balance"] == 6500.52          # free cash, not notional
    assert acct["venue"] == "JS_GLOBAL"
    # the paper book is a million-rupee fiction — it must never leak in
    from kairos.execution import PaperBroker
    assert PaperBroker().account()["equity"] != acct["equity"]


def test_no_venue_at_all_means_no_account_snapshot(ap):
    assert ap._live_account(False, None) is None


def test_both_venues_sum_without_mixing_a_name(ap, monkeypatch):
    monkeypatch.setattr(ap.mt5, "account", lambda: {
        "venue": "MT5", "equity": 1000.0, "balance": 900.0, "day_pnl": 10.0,
        "open_positions": 1})
    acct = ap._live_account(True, _book())
    assert acct["equity"] == 11681.68
    assert acct["venue"] == "MT5+JS_GLOBAL"


def test_day_pnl_is_measured_from_the_first_snapshot_of_the_day(ap):
    book = _book()
    assert ap._psx_day_pnl(book) == 0.0                 # baseline taken
    assert ap.st["psx_day_start"]["equity"] == book["equity"]
    book["equity"] = round(book["equity"] - 900.0, 2)   # OGDC slid intraday
    assert ap._psx_day_pnl(book) == -900.0
    ap.st["psx_day_start"] = {"date": "2000-01-01", "equity": 50000.0}
    assert ap._psx_day_pnl(book) == 0.0                 # new day, new baseline


def test_day_pnl_baseline_survives_a_restart(ap):
    ap._psx_day_pnl(_book())
    ap._save()
    again = Autopilot()
    again.state_path = ap.state_path
    assert again._load()["psx_day_start"] == ap.st["psx_day_start"]


def test_breaker_opens_on_a_real_daily_loss(ap, monkeypatch):
    """The circuit breaker must fire off real PKR, not paper equity."""
    from kairos import rules as R
    monkeypatch.setattr(R, "rules", lambda: {
        "max_daily_loss_pct": 2.0, "max_total_exposure_pct": 95.0,
        "max_consecutive_losses": 4})
    book = _book()
    book["cash"] = round(book["cash"] - 5000.0, 2)
    book["equity"] = round(book["equity"] - 5000.0, 2)
    from datetime import datetime
    today = datetime.now().strftime("%Y-%m-%d")
    ap.st["psx_day_start"] = {"date": today, "equity": 10681.68}
    ap.st["equity_peak"] = 10681.68
    acct = ap._psx_part(book)
    assert acct["day_pnl"] == -5000.0
    posture = ap._risk_posture(acct, ap._venue_positions(False, book))
    assert posture["block_entries"] is True
    assert any("daily loss" in b for b in posture["blockers"])


# ------------------------------------------------------------- positions
def test_holdings_carry_mtm_so_stops_apply(ap):
    rows = ap._venue_positions(False, _book())
    assert {r["symbol"] for r in rows} == {"OGDC", "PPL"}
    ogdc = [r for r in rows if r["symbol"] == "OGDC"][0]
    assert ogdc["last"] == 299.47 and ogdc["entry"] == 316.92
    assert ogdc["pnl"] == pytest.approx(-52.35, abs=0.01)


def test_reconcile_keeps_mt5_memory_when_only_the_terminal_is_down(ap):
    ap.st["held"] = {"GOLD": {"side": "BUY"}, "PPL": {"side": "BUY"}}
    ap._reconcile({}, live=True, mt5_up=False, psx_up=False)
    assert set(ap.st["held"]) == {"GOLD", "PPL"}   # nothing is authoritative


def test_reconcile_drops_a_position_the_real_book_no_longer_reports(ap):
    held = {"PPL": {"qty": 15}}
    ap.st["held"] = {"PPL": {"side": "BUY"}, "OGDC": {"side": "BUY"}}
    ap._reconcile(held, live=True, mt5_up=False, psx_up=True)
    assert set(ap.st["held"]) == {"PPL"}           # sold venue-side -> forgotten


# ---------------------------------------------------------- venue failure
def test_portal_down_is_reported_once_and_never_read_as_zero(ap, monkeypatch):
    calls = []
    monkeypatch.setattr(AP, "JOURNAL", type("J", (), {
        "log": staticmethod(lambda tag, payload: calls.append((tag, payload)))}))

    def down(*a, **kw):
        raise jsgsync.JsGlobalError("portal login rejected")
    monkeypatch.setattr(jsgsync, "real_book", down)
    assert ap._psx_book() is None
    assert ap._psx_book() is None
    assert [c[0] for c in calls] == ["AUTOPILOT_VENUE_LOSS"]
    assert ap._live_account(False, ap._psx_book()) is None


# ------------------------------------------------------- order-side guards
def test_buy_larger_than_free_cash_never_reaches_the_broker(ap, monkeypatch):
    _never_places(monkeypatch)
    ap._psx_live = _book(cash=500.0)
    assert ap._psx_auto_place(_Inst("OGDC"), "BUY", 100, 300.0,
                              _intent(), False) is None


def test_short_sale_is_blocked_on_a_cash_account(ap, monkeypatch):
    _never_places(monkeypatch)
    ap._psx_live = _book()
    assert ap._psx_auto_place(_Inst("SYSIT"), "SELL", 50, 200.0,
                              _intent(), False) is None


def test_closing_more_than_is_sellable_is_blocked(ap, monkeypatch):
    _never_places(monkeypatch)
    book = _book()
    book["holdings"][1]["pending_sell"] = 10.0    # 15 held, 10 already resting
    ap._psx_live = book
    assert ap._psx_auto_place(_Inst("PPL"), "SELL", 15, 219.0,
                              _intent(), True) is None


def test_sellable_close_does_reach_the_bridge(ap, monkeypatch):
    sent = []
    monkeypatch.setattr(jsgtrade, "enabled", lambda: True)
    monkeypatch.setattr(jsgtrade, "place_now", lambda *a, **kw:
                        sent.append(a) or {"ok": True, "order_no": 1,
                                           "account": "52366"})
    ap._psx_live = _book()
    out = ap._psx_auto_place(_Inst("PPL"), "SELL", 15, 219.0, _intent(), True)
    assert out == ("PLACED", "order 1 at 52366")
    assert sent[0][1] == "SELL"


def test_ticket_only_when_the_portal_is_unreachable(ap, monkeypatch):
    _never_places(monkeypatch)
    ap._psx_down = True
    ap._psx_live = _book()
    assert ap._psx_auto_place(_Inst("OGDC"), "BUY", 3, 300.0,
                              _intent(), False) is None


def test_disarmed_config_still_returns_none(ap, monkeypatch):
    from kairos.config import CFG as real
    monkeypatch.setattr(AP, "CFG", real)     # the real file: auto_place off
    monkeypatch.setattr(jsgtrade, "enabled", lambda: False)
    assert ap._psx_auto_place(_Inst("OGDC"), "BUY", 3, 300.0,
                              _intent(), False) is None


def _intent():
    from kairos.execution import OrderIntent
    return OrderIntent(symbol="TEST", side="BUY", quantity=1, venue="JS_GLOBAL")
