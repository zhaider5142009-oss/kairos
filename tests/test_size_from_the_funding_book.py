"""An order is a fraction of the book that has to fund it — in that book's currency.

The desk and the circuit breakers read one account snapshot, and with two venues
live that snapshot adds a USD CFD account to a PKR portal book. As an upper bound
on total exposure the sum is tolerable: it overstates a rupee-heavy book against a
dollar one, and every breaker that reads it errs toward refusing. It was also the
number the loop sized from, and there the direction reverses. Fifteen percent of
a million-one rupees divided by a dollar price is not a smaller CFD order than
fifteen percent of 100,000 dollars — it is about eleven times bigger, in a currency
the account does not hold. The same sum, read the other way, spends a dollar
balance as if it were rupees.

So the snapshot stays a bound, and every size now comes from the venue's own
account. A venue that the routing chose but that never reported one is a veto, not
a zero: zero would look like a small book, and the point of the journal is to tell
those two apart.
"""

from __future__ import annotations

import kairos.autopilot as AP
from test_autopilot_cycle import _AGREE
from test_live_arming import _book
from test_lot_size_from_venue import GOLD_SPEC
from test_mt5_close_ticket import _Terminal, _autopilot, _prediction
from test_psx_exit_working import _Journal
from test_spot_size_convention import _cleared

MT5_EQUITY = 100_000.0          # the terminal's own USD account
PSX_EQUITY = 1_000_000.0        # the portal's own PKR book
BN_EQUITY = 250.0               # a third book, small enough to notice if ignored
MERGED = MT5_EQUITY + PSX_EQUITY + BN_EQUITY


def _both_venues(monkeypatch, tmp_path, res, journal=None,
                 mt5_eq: float = MT5_EQUITY, psx_eq: float = PSX_EQUITY):
    """A live cycle with a dollar CFD account and a rupee portal book, both up."""
    term = _Terminal([])
    read = term.account
    term.account = lambda: {**read(), "equity": mt5_eq,
                            "balance": mt5_eq}                # type: ignore[misc]
    term.spec = lambda symbol: dict(GOLD_SPEC)    # type: ignore[method-assign]
    ap = _autopilot(monkeypatch, tmp_path, term, journal or _Journal(), res,
                    managed=False)
    monkeypatch.setattr(AP, "_binance_live_ready", lambda: False)
    # `_autopilot` parks the PSX venue off; this fixture puts it back, funded.
    monkeypatch.setattr(ap, "_psx_book", lambda: _book(cash=psx_eq, holdings=[]))
    # Two gates upstream of the size have their own tests. Here they would only
    # decide by whether the fixture's name has local history, and the question is
    # which book funds the order.
    monkeypatch.setattr(ap, "_psx_liquid", lambda inst: None)
    from kairos import fugu as _F
    monkeypatch.setattr(_F, "conduct", lambda key, res=None, posture=None: {
        "action": "EXECUTE", "edge_r": 1.0, "conviction": 0.9, "p_hit": 0.7,
        "reasons": []})
    return term, ap


# ------------------------------------------------------------------ the accounts
def test_each_venue_reads_its_own_equity():
    """The snapshot the desk shows is the sum; the number an order is sized on is
    not. Reading one as the other is the whole bug."""
    ap = AP.Autopilot()
    ap._venue_accts = {"MT5": {"equity": MT5_EQUITY},
                       "JS_GLOBAL": {"equity": PSX_EQUITY},
                       "BINANCE": {"equity": BN_EQUITY}}
    merged = ap._merge_account(ap._venue_accts)
    assert merged["equity"] == MERGED
    assert ap._venue_equity("MT5", merged) == MT5_EQUITY
    assert ap._venue_equity("JS_GLOBAL", merged) == PSX_EQUITY
    assert ap._venue_equity("BINANCE", merged) == BN_EQUITY
    # a venue the routing named but that contributed no account has no equity —
    # and no equity is not the same fact as a small one
    assert ap._venue_equity("PMEX", merged) is None


def test_a_single_book_needs_no_split(monkeypatch, tmp_path):
    """Control: paper and one live venue have one book, so the snapshot already
    is the venue's own equity. Nothing tightens here."""
    ap = AP.Autopilot()
    assert ap._venue_equity("PAPER", {"equity": 1_000_000.0}) == 1_000_000.0


def test_a_book_that_answered_with_no_number_is_no_equity_too():
    """An account read that came back empty is the other half of the same veto.
    `equity: None` through the figure helper is 0.0, and 0.0 sizes an order at
    nothing while the desk still shows the other venue's money — a silent venue
    and an empty book look identical unless the accessor says otherwise."""
    ap = AP.Autopilot()
    ap._venue_accts = {"MT5": {"equity": None}, "JS_GLOBAL": {"equity": PSX_EQUITY}}
    merged = ap._merge_account(ap._venue_accts)
    assert ap._venue_equity("MT5", merged) is None
    assert ap._venue_equity("JS_GLOBAL", merged) == PSX_EQUITY


# ---------------------------------------------------------------------- the loop
def test_the_cfd_is_sized_on_the_dollar_account_not_the_sum(monkeypatch, tmp_path):
    """Same signal, same price, two live books. The order has to fit 15% of the
    CFD account, because that is the account the terminal would have to fund it
    from; 15% of the rupee book added on top is a size the dollar account never
    approved."""
    res = _prediction(monkeypatch, battery=_AGREE)
    price = float(res["price"])
    term, ap = _both_venues(monkeypatch, tmp_path, res)
    cleared = _cleared(monkeypatch, ap)

    out = ap.cycle()

    assert out["trades"] == 1, ap.st["signals"][0]
    assert cleared and cleared[0] <= 0.15 * MT5_EQUITY + 1e-9, \
        "the CFD was sized on the other venue's rupees"
    lots = float(term.deals[0].quantity)
    assert lots > 0
    assert lots * price * GOLD_SPEC["contract_size"] <= 0.15 * MT5_EQUITY + 1e-9


def test_the_desk_shows_which_book_each_order_was_sized_against(monkeypatch, tmp_path):
    """The operator-facing summary used to print one merged equity and call it
    rupees. Once sizes come from each venue's own book, the same disclosure has to
    show those numbers, or the desk explains an order the total cannot fund."""
    res = _prediction(monkeypatch, battery=_AGREE)
    _term, ap = _both_venues(monkeypatch, tmp_path, res)

    ap.cycle()

    shown = ap.status()["live_equity"]
    assert shown["MT5"] == MT5_EQUITY
    assert shown["JS_GLOBAL"] == PSX_EQUITY
    assert ap.status()["live_account"]["equity"] == MT5_EQUITY + PSX_EQUITY


def test_the_psx_ticket_is_sized_on_the_rupee_book(monkeypatch, tmp_path):
    """The same sum read from the other venue: whole shares paid for in PKR, out
    of a small rupee book next to a large dollar one. A dollar balance added as if
    it were rupees buys eleven times the position the portal account can carry —
    and the position is in a scrip the portal, not the terminal, would settle."""
    res = _prediction(monkeypatch, battery=_AGREE)
    res = {**res, "symbol": "OGDC", "name": "OGDC", "price": 299.47}
    price = float(res["price"])
    rupees = 100_000.0
    _term, ap = _both_venues(monkeypatch, tmp_path, res,
                             mt5_eq=1_000_000.0, psx_eq=rupees)
    ap.st["symbols"] = ["OGDC"]
    seen: list[tuple[str, float]] = []

    def _capture(inst, side, qty, *a, **kw):
        seen.append((inst.key, qty))
        return "TICKET", "staged", None

    monkeypatch.setattr(ap, "_execute", _capture)

    out = ap.cycle()

    assert out["trades"] == 1, str(ap.st["signals"][0])
    assert seen, str(ap.st["signals"][0])
    assert seen[0][0] == "OGDC"
    qty = seen[0][1]
    assert qty == int(qty), "the portal book was not sized in whole shares"
    assert qty >= 1, "the rupee book cannot even buy one share"
    assert qty * price <= 0.15 * rupees + 1e-9, \
        "the PSX ticket was sized on the CFD account's dollars"

def test_a_venue_chosen_for_an_order_that_reported_nothing_trades_nothing(
        monkeypatch, tmp_path):
    """The terminal answered for positions and stayed silent about the account.
    The merged snapshot still holds a plausible number — the PSX book — and the
    order is routed to MT5, where it would be spent in dollars. Nothing is sent,
    and the veto is recorded as a missing account rather than as a signal that
    failed to earn a size."""
    res = _prediction(monkeypatch, battery=_AGREE)
    journal = _Journal()
    term, ap = _both_venues(monkeypatch, tmp_path, res, journal)
    term.account = lambda: None                    # type: ignore[method-assign]

    out = ap.cycle()

    assert out["trades"] == 0, ap.st["signals"][0]
    assert term.deals == [], "an order went to a venue with no account reading"
    signal = ap.st["signals"][0]
    assert signal["action"] == "NO-EQUITY"
    assert "MT5" in signal["detail"], signal["detail"]
    assert journal.count("AUTOPILOT_NO_EQUITY") == 1
    # the trail has to tell a silent venue from an empty book: one needs a
    # connection, the other needs funding
    assert journal.rows[-1][1]["read"] is False
    assert "GOLD" not in ap.st["held"]


def test_an_empty_book_is_named_as_an_empty_book(monkeypatch, tmp_path):
    """The terminal answered for the account and the account is empty. The desk
    cannot get that from a merged snapshot that still shows the portal's rupees,
    and it is not the same fix as a link that never answered."""
    res = _prediction(monkeypatch, battery=_AGREE)
    journal = _Journal()
    term, ap = _both_venues(monkeypatch, tmp_path, res, journal, mt5_eq=0.0)

    out = ap.cycle()

    assert out["trades"] == 0, str(ap.st["signals"][0])
    assert term.deals == []
    assert "no usable equity" in ap.st["signals"][0]["detail"]
    assert journal.count("AUTOPILOT_NO_EQUITY") == 1
    assert journal.rows[-1][1]["read"] is True
