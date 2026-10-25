"""A position is managed by the scan loop, so the loop must scan it.

Every exit the desk owns — hard stop, target, trailing ratchet, signal flip,
maturity — is evaluated inside `for sym in st["symbols"]`. But the watchlist is
an operator-editable list: `configure(symbols=...)` replaces it wholesale and a
PSX rotation rebuilds it from that day's screener. A held name that is dropped
from it therefore keeps a stop, a target and an entry date that nothing reads,
while the venue still holds the trade. `_reconcile` deliberately never forgets a
position the venue still reports, so nothing else removes or manages it either:
the position sits there until a human closes it. That is a fail-silent guard,
not a missing feature — the desk shows a healthy book with a stop that binds
nothing.

The scan list now follows `held` as well as the watchlist, and a held key the
universe cannot resolve is recorded once instead of skipped in silence.
"""

from __future__ import annotations

import kairos.autopilot as AP
import kairos.data as D
from kairos.execution import PaperBroker
from test_autopilot_cycle import _AGREE, _BULL
from test_risk_veto_channel import _weight_and_notes

RATE = 300.0


def _ap(**st) -> AP.Autopilot:
    ap = AP.Autopilot()
    ap.st = ap._defaults()
    ap.st.update(st)
    return ap


# ------------------------------------------------------------------ scan list
def test_the_scan_list_follows_a_held_name_the_watchlist_dropped():
    ap = _ap(symbols=["OGDC"], held={"GOLD": {"side": "BUY", "qty": 1}})
    assert ap._scan_list() == ["OGDC", "GOLD"]


def test_the_scan_list_does_not_scan_a_name_twice():
    ap = _ap(symbols=["GOLD", "OGDC"], held={"GOLD": {"qty": 1}})
    assert ap._scan_list() == ["GOLD", "OGDC"]


def test_a_held_key_with_no_universe_row_is_named_once():
    ap = _ap(held={"NOSUCHKEY": {"qty": 1}})
    ap._unmanaged("NOSUCHKEY")
    ap._unmanaged("NOSUCHKEY")
    hits = [a for a in ap.st["actions"] if a["action"] == "UNMANAGED"]
    assert len(hits) == 1 and hits[0]["symbol"] == "NOSUCHKEY"


def test_a_watchlist_name_with_no_universe_row_is_not_an_alarm():
    ap = _ap(symbols=["NOSUCHKEY"])
    ap._unmanaged("NOSUCHKEY")
    assert not ap.st["actions"]


# --------------------------------------------------------------------- cycle
def _desk(monkeypatch, tmp_path):
    """One autopilot over one paper ledger that survives between cycles."""
    res = _weight_and_notes(monkeypatch, _BULL, _AGREE)[2]

    def _book():
        # `PaperBroker` binds `CFG.data_dir` in its constructor, so construction
        # is the only window in which the ledger can be sent to the test's own
        # directory; the path it captures then follows every later save.
        where = AP.CFG.data_dir
        AP.CFG.data_dir = tmp_path
        try:
            return PaperBroker()
        finally:
            AP.CFG.data_dir = where

    ap = AP.Autopilot()
    ap.state_path = tmp_path / "autopilot.json"
    ap.st = ap._defaults()
    ap.st.update({"symbols": ["GOLD"], "last_rotation": AP._now()})
    from kairos.engine import ENGINE
    monkeypatch.setattr(ENGINE, "predict", lambda key, horizon=None, **kw: res)
    monkeypatch.setattr(ENGINE, "refresh_news", lambda watch: {})
    monkeypatch.setattr(AP, "PaperBroker", _book)
    monkeypatch.setattr(D, "usd_pkr", lambda: RATE)
    return ap, res


def test_a_held_name_with_no_watchlist_row_still_gets_its_stop(monkeypatch,
                                                              tmp_path):
    ap, res = _desk(monkeypatch, tmp_path)
    first = ap.cycle()
    assert first["trades"] == 1, (
        "no entry fired, so the second cycle had nothing to strand: "
        f"{ap.st['signals'][0]['action']}")
    assert "GOLD" in ap.st["held"]

    # The operator narrows the watchlist away from a live position, and the
    # price falls through the stop it was given.
    ap.st["symbols"] = []
    ap.st["entry_threshold"] = 0.99
    res["price"] = ap.st["held"]["GOLD"]["stop"] * 0.90

    out = ap.cycle()
    assert out["trades"] == 1, "the stop did not fire for an unwatched holding"
    assert "GOLD" not in ap.st["held"]
    row = next(r for r in ap.st["signals"] if r["symbol"] == "GOLD")
    assert row["action"] == "CLOSED"
    assert row["detail"] == "STOP exit"
