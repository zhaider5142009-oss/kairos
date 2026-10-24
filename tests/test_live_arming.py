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
         "value": 898.41, "pending_sell": 0.0, "account": "99999"},
        {"symbol": "PPL", "qty": 15.0, "avg": 227.39, "mtm": 218.85,
         "value": 3282.75, "pending_sell": 0.0, "account": "99999"},
    ]
    value = round(sum(x["value"] for x in h), 2)
    return {"ok": True, "venue": "JS_GLOBAL", "accounts": ["MAIMOONA FARRUKH"],
            "cash": cash, "holdings_value": value,
            "equity": round(cash + value, 2),
            "equity_basis": "free cash + mark-to-market holdings",
            "holdings": h, "ts": "2026-10-19 18:00:00"}


class _Inst:
    def __init__(self, key, psx_ticker=None):
        self.key = key
        self.psx_ticker = psx_ticker


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


def _zero_book():
    """What the portal hands back when it answers 200 but reads nothing.

    `ledger_balance` turns a mis-keyed `GetAccountBalance` into 0.0 rather than
    an error, so a fully degraded account can still arrive as a healthy-looking
    zero that raises nothing on the way out.
    """
    return {"ok": True, "venue": "JS_GLOBAL", "accounts": ["MAIMOONA FARRUKH"],
            "cash": 0.0, "holdings_value": 0.0, "equity": 0.0,
            "equity_basis": "free cash + mark-to-market holdings",
            "holdings": [], "ts": "2026-10-20 09:00:00"}


def _logged(ap, monkeypatch):
    calls: list = []

    class _J:
        @staticmethod
        def log(tag, payload):
            calls.append((tag, payload))

    monkeypatch.setattr(AP, "JOURNAL", _J())
    return calls


def _loss_rules(monkeypatch):
    from kairos import rules as R
    monkeypatch.setattr(R, "rules", lambda: {
        "max_daily_loss_pct": 3.0, "max_total_exposure_pct": 95.0,
        "max_consecutive_losses": 4})


# ------------------------------------------------------- degraded live reads
def test_a_zero_book_is_a_venue_loss_not_an_empty_account(ap, monkeypatch):
    """The docstring of `_psx_book` promises this and the code did not do it.

    A raising outage already became None. A zeroing outage did not, and every
    `if eq > 0` guard downstream reads zero equity as "nothing to check".
    """
    calls = _logged(ap, monkeypatch)
    monkeypatch.setattr(jsgsync, "real_book", lambda *a, **kw: _zero_book())
    assert ap._psx_book() is None
    assert [c[0] for c in calls] == ["AUTOPILOT_VENUE_LOSS"]
    assert ap._psx_down is True          # and so the auto-place path stands down


def test_a_degraded_read_never_stamps_the_day_baseline(ap):
    """A zero baseline is never restored: it pins day P&L at zero all day."""
    assert ap._psx_day_pnl(_zero_book()) == 0.0
    assert ap.st["psx_day_start"] is None            # nothing latched from it
    base = _book()
    assert ap._psx_day_pnl(base) == 0.0              # first usable read stamps
    assert ap.st["psx_day_start"]["equity"] == base["equity"]


def test_a_morning_outage_cannot_inert_the_daily_breaker_all_day(ap, monkeypatch):
    """The money path this guards, end to end.

    Degraded read at the open, healthy book the rest of the day, real -5%
    loss. Before the fix the outage stamped a zero baseline, day P&L measured
    against zero forever after, and neither the autopilot breaker nor the
    pre-trade gate could see a loss the account was actually carrying.
    """
    _loss_rules(monkeypatch)
    monkeypatch.setattr(jsgsync, "real_book", lambda *a, **kw: _zero_book())
    assert ap._psx_book() is None

    base = _book()
    ap.st["equity_peak"] = base["equity"]
    down = _book()
    down["cash"] = round(down["cash"] - 0.05 * base["equity"], 2)
    down["equity"] = round(base["equity"] * 0.95, 2)
    acct = ap._psx_part(down)                        # day baseline set here
    assert acct["equity"] > 0
    down2 = _book()
    down2["cash"] = round(down["cash"] - 1200.0, 2)
    down2["equity"] = round(down["equity"] - 1200.0, 2)
    acct = ap._psx_part(down2)
    assert acct["day_pnl"] == pytest.approx(-1200.0)
    posture = ap._risk_posture(acct, ap._venue_positions(False, down2))
    assert posture["block_entries"] is True
    assert any("daily loss" in b for b in posture["blockers"])


def test_a_baseline_latched_at_zero_by_an_older_build_self_heals(ap):
    """State already written before this fix must not stay poison forever."""
    from datetime import datetime
    ap.st["psx_day_start"] = {"date": datetime.now().strftime("%Y-%m-%d"),
                              "equity": 0.0}
    book = _book()
    assert ap._psx_day_pnl(book) == 0.0              # re-stamped, not measured off 0
    assert ap.st["psx_day_start"]["equity"] == book["equity"]


def test_a_cycle_with_no_equity_figure_reports_a_blocker(ap, monkeypatch):
    """Unreadable risk must not report as a clean risk picture.

    Each breaker compares against a positive amount, so with eq at zero the
    posture used to come back with no blockers — the gate still refused the
    order, but the desk and /api/autopilot showed a cycle that looked safe.
    """
    _loss_rules(monkeypatch)
    posture = ap._risk_posture({"venue": "JS_GLOBAL", "equity": 0.0,
                                "day_pnl": 0.0, "balance": 0.0}, [])
    assert posture["block_entries"] is True
    assert any("equity" in b for b in posture["blockers"])
    assert ap._risk_posture(None, [])["block_entries"] is True


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


# ------------------------------------------- an outage is not a flat account
def _bridge(fake_terminal):
    """A bridge wired to a fake terminal: no MetaTrader5, no network."""
    from kairos.execution import Mt5Bridge
    b = Mt5Bridge()
    b.connected = True
    b._mt5 = fake_terminal
    return b


class _NoTerminal:
    @staticmethod
    def positions_get():
        return None                    # MetaTrader's "the request failed"


class _FlatTerminal:
    @staticmethod
    def positions_get():
        return ()                      # the terminal's "you hold nothing"


def test_a_failed_position_retrieve_is_not_an_empty_account():
    """The bridge collapsed two different answers into one.

    `positions_get()` returns None when the request failed and an empty tuple
    when the account really is flat. Turning the first into the second made a
    dead terminal look like a flat book: the max-positions and exposure
    breakers both passed on an empty list, and reconciliation deleted the
    managed live entries it was supposed to be watching.
    """
    assert _bridge(_NoTerminal()).positions() is None
    assert _bridge(_FlatTerminal()).positions() == []


def test_the_account_snapshot_reports_unknown_positions_as_unknown():
    from types import SimpleNamespace

    class _Term:
        @staticmethod
        def account_info():
            return SimpleNamespace(login=1234, name="TEST", server="srv",
                                   currency="USD", equity=1000.0, balance=900.0,
                                   margin_free=800.0, leverage=100, profit=0.0)

        @staticmethod
        def positions_get():
            return None

    acct = _bridge(_Term()).account()
    assert acct["open_positions"] is None
    assert acct["equity"] == 1000.0


def test_venue_positions_are_unknown_when_a_venue_refuses(ap, monkeypatch):
    monkeypatch.setattr(ap.mt5, "positions", lambda: None)
    assert ap._venue_positions(True, _book()) is None
    # an answered-empty list is still the fact it always was
    monkeypatch.setattr(ap.mt5, "positions", lambda: [])
    assert {r["symbol"] for r in ap._venue_positions(True, _book())} == {"OGDC", "PPL"}


def test_a_fused_account_never_invents_a_position_count(ap, monkeypatch):
    """MT5 unreadable + PSX readable must not sum to a plausible flat number."""
    monkeypatch.setattr(ap.mt5, "account", lambda: {
        "venue": "MT5", "equity": 1000.0, "balance": 900.0, "day_pnl": 10.0,
        "open_positions": None})
    acct = ap._live_account(True, _book())
    assert acct["open_positions"] is None
    assert acct["equity"] == 11681.68


def test_reconcile_keeps_managed_entries_when_the_venue_was_not_read(ap):
    """Forgetting a live trade because a read failed is worse than keeping it."""
    ap.st["held"] = {"GOLD": {"side": "BUY"}, "PPL": {"side": "BUY"}}
    ap._reconcile({}, live=True, mt5_up=True, psx_up=True, pos_known=False)
    assert set(ap.st["held"]) == {"GOLD", "PPL"}


def test_an_unreadable_position_list_blocks_instead_of_showing_flat(ap, monkeypatch):
    _loss_rules(monkeypatch)
    acct = ap._psx_part(_book())
    posture = ap._risk_posture(acct, None)
    assert posture["block_entries"] is True
    assert any("exposure" in b.lower() for b in posture["blockers"])
    assert posture["exposure_pct"] is None
    assert posture["n_positions"] is None
    # a genuinely flat account still reports zero, not unknown
    clear = ap._risk_posture(acct, [])
    assert clear["exposure_pct"] == 0.0
    assert clear["n_positions"] == 0
    assert not any("exposure" in b.lower() for b in clear["blockers"])


class _Portal:
    """Stub for `jsgsync._get`, keyed by endpoint path."""

    def __init__(self, **answers):
        self.answers = answers
        self.calls: list[str] = []

    def __call__(self, path, params=None):
        self.calls.append(path)
        v = self.answers[path]
        if isinstance(v, Exception):
            raise v
        return v


def _portal(monkeypatch, **answers):
    p = _Portal(**answers)
    monkeypatch.setattr(jsgsync, "_get", p)
    monkeypatch.setattr(jsgsync, "accounts", lambda: [
        {"accountNo": "99999", "accountName": "TEST HOLDER"}])
    monkeypatch.setattr(jsgsync, "ledger_balance", lambda no: 6500.52)
    jsgsync.invalidate_book()
    return p


def test_a_failing_holdings_view_raises_instead_of_reporting_flat(monkeypatch):
    """`real_book` promised a transport failure raises; the views disagreed.

    Both endpoints swallowed any exception into None and returned an empty
    list, so a portal that answered nothing looked like an account holding
    nothing — and because cash was still readable, the book arrived as a
    plausible funded, flat one that every circuit breaker accepted.
    """
    _portal(monkeypatch, **{"/Home/GetOpenPosition": [],
                            "/Home/GetCollaterals": jsgsync.JsGlobalError("502")})
    with pytest.raises(jsgsync.JsGlobalError):
        jsgsync.real_book()


def test_a_successful_empty_answer_is_still_a_fact(monkeypatch):
    """Null on the wire after a 200 is the account saying it holds nothing."""
    _portal(monkeypatch, **{"/Home/GetOpenPosition": None,
                            "/Home/GetCollaterals": []})
    assert jsgsync.open_positions("99999", "TEST HOLDER") == []


def test_a_partial_portal_read_stands_the_venue_down(ap, monkeypatch):
    """The same money path the desk walks, end to end and offline."""
    calls = _logged(ap, monkeypatch)
    _portal(monkeypatch, **{"/Home/GetOpenPosition": jsgsync.JsGlobalError("timeout"),
                            "/Home/GetCollaterals": []})
    assert ap._psx_book() is None
    assert ap._psx_down is True
    assert [c[0] for c in calls] == ["AUTOPILOT_VENUE_LOSS"]


def test_sync_prunes_nothing_when_a_view_fails(ap, monkeypatch):
    """A failed read must not release positions the live book is watching."""
    from kairos import livebook
    livebook.add("OGDC", 3, 316.92)
    _portal(monkeypatch, **{"/Home/GetOpenPosition": [],
                            "/Home/GetCollaterals": jsgsync.JsGlobalError("503")})
    monkeypatch.setattr(jsgsync, "pending_orders", lambda acc: [])
    with pytest.raises(jsgsync.JsGlobalError):
        jsgsync.sync()
    assert "OGDC" in livebook.positions()
    livebook.remove("OGDC")


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
                                           "account": "99999"})
    ap._psx_live = _book()
    out = ap._psx_auto_place(_Inst("PPL"), "SELL", 15, 219.0, _intent(), True)
    assert out == ("PLACED", "order 1 at 99999")
    assert sent[0][1] == "SELL"


def _capture(monkeypatch):
    sent = []
    monkeypatch.setattr(jsgtrade, "enabled", lambda: True)
    monkeypatch.setattr(jsgtrade, "place_now", lambda *a, **kw:
                        sent.append(a) or {"ok": True, "order_no": 2,
                                           "account": "99999"})
    return sent


def test_a_respelled_name_reaches_the_broker_as_the_exchange_ticker(ap, monkeypatch):
    """KAIROS calls it BANK; the broker's script is BOP."""
    from kairos.universe import resolve

    sent = _capture(monkeypatch)
    ap._psx_live = _book(cash=100000.0)
    out = ap._psx_auto_place(resolve("BANK"), "BUY", 5, 27.3, _intent(), False)
    assert out[0] == "PLACED"
    assert sent[0][0] == "BOP"


def test_a_holding_under_the_exchange_ticker_is_sellable(ap, monkeypatch):
    """The live book carries the broker's spelling, not the engine's key."""
    from kairos.universe import resolve

    sent = _capture(monkeypatch)
    ap._psx_live = _book(holdings=[
        {"symbol": "BOP", "qty": 20.0, "avg": 25.0, "mtm": 27.3,
         "value": 546.0, "pending_sell": 0.0, "account": "99999"}])
    out = ap._psx_auto_place(resolve("BANK"), "SELL", 15, 27.3, _intent(), True)
    assert out[0] == "PLACED"
    assert sent[0][0] == "BOP"


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


def test_a_restart_never_reports_live(tmp_path):
    """The desk prints LIVE straight from these fields, so a file that still
    claims an arming must not survive into a process that has no arming."""
    import json

    p = tmp_path / "autopilot.json"
    p.write_text(json.dumps({"running": True, "mode": "live",
                             "armed_live": True, "cycles": 99}), encoding="utf-8")
    a = Autopilot()
    a.state_path = p
    st = a._load()
    assert st["running"] is False
    assert st["mode"] == "paper"
    assert st["armed_live"] is False
    assert st["cycles"] == 99          # history survives, the arming does not
