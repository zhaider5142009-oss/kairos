"""Binance answers an order it cannot express, so the numbers must be expressible.

Every spot pair publishes the ladder its size and price move on. On BTCUSDT the
quantity step is 0.00001 BTC, the price tick is 0.01 USDT, the smallest order is
0.00001 BTC and the smallest value 5 USDT — all read live from
`/api/v3/exchangeInfo`, and none of them guessed. An order off that ladder is not
filled at the nearest valid size: it is refused, with a `FILTER FAILURE` the
operator then has to interpret while the entry they meant to take is gone.

The sizing layer above this venue already does the right thing about ceilings
(`_size_for` divides a notional by a price and gets whatever decimals that
produces). This is the other half: the venue's own units, applied at the door,
truncated *down* because the approved notional is a ceiling and rounding up buys
more than was signed off.

Alongside it, what a refusal *is*. A 4xx is the exchange reading the order and
declining it: nothing is live, and the desk has to be able to say that, rather
than leaving a declined order filed in the same "we do not know" bucket as one
whose reply never arrived.
"""

from __future__ import annotations

import urllib.error
import urllib.request

import pytest

from kairos import binance as B
from kairos.config import CFG
from kairos.execution import OrderIntent
from test_binance_venue import CONF, PERMIT, _arm


def _rules(step: str = "0.00001000", min_qty: str = "0.00001000",
           tick: str = "0.01000000", min_notional: str = "5.00000000",
           status: str = "TRADING") -> dict:
    """One pair's `exchangeInfo` answer, in the shape the venue really sends.

    `MARKET_LOT_SIZE` is included with its reported zero step: it is a real
    filter on a real pair, and reading it as the ladder would truncate every
    order to zero.
    """
    return {"symbols": [{
        "symbol": "BTCUSDT", "status": status,
        "filters": [
            {"filterType": "PRICE_FILTER", "minPrice": "0.01000000",
             "maxPrice": "1000000.00000000", "tickSize": tick},
            {"filterType": "LOT_SIZE", "minQty": min_qty, "maxQty": "9000.00000000",
             "stepSize": step},
            {"filterType": "MARKET_LOT_SIZE", "minQty": "0.00000000",
             "maxQty": "135.48244404", "stepSize": "0.00000000"},
            {"filterType": "NOTIONAL", "minNotional": min_notional,
             "applyMinToMarket": True, "avgPriceMins": 5},
            {"filterType": "MAX_NUM_ORDERS", "maxNumOrders": 200},
        ]}]}


class _Venue:
    """The pair's rules, then one order — with every request remembered."""

    def __init__(self, rules=None, fail: Exception | None = None) -> None:
        self.rules = _rules() if rules is None else rules
        self.fail = fail
        self.sent: list[tuple[str, str, bool]] = []
        self.signed: dict = {}

    def __call__(self, path, params=None, signed=False, method="GET"):
        self.sent.append((path, method, signed))
        if path.startswith("/api/v3/exchangeInfo"):
            return self.rules if isinstance(self.rules, dict) else dict(self.rules)
        if path.startswith("/api/v3/order"):
            self.signed.update(params or {})
            if self.fail is not None:
                raise self.fail
            return {"orderId": 1234, "status": "FILLED", "symbol": "BTCUSDT",
                    "executedQty": self.signed.get("quantity", "0"), "fills": []}
        return {}

    @property
    def posts(self) -> list[tuple[str, str, bool]]:
        return [c for c in self.sent if c[1] == "POST"]

    @property
    def reads(self) -> int:
        return len([c for c in self.sent if c[0].startswith("/api/v3/exchangeInfo")])


@pytest.fixture()
def desk(tmp_path, monkeypatch):
    """An armed client whose transport is answered here, journal on a temp disk."""
    monkeypatch.setattr(CFG, "data_dir", tmp_path)
    monkeypatch.setattr(B, "_conf", lambda: dict(CONF))
    monkeypatch.setattr(B.JOURNAL, "path", tmp_path / "audit.jsonl")
    monkeypatch.setattr(B.JOURNAL, "last_error", None)
    client = B.BinanceClient()
    client._perms = dict(PERMIT)          # armed without a network read
    client._perms_at = 9e12
    events: list[tuple[str, dict]] = []
    real_log = B.JOURNAL.log

    def spy(event: str, payload: dict) -> bool:
        events.append((event, payload))
        return real_log(event, payload)

    monkeypatch.setattr(B.JOURNAL, "log", spy)
    client.events = events                # type: ignore[attr-defined]
    return client


# ------------------------------------------------------------------ the ladder
def test_a_quantity_off_the_ladder_is_put_on_it(desk, monkeypatch):
    """150.53162753 of BTC is not an order on a 0.00001 ladder.

    The venue does not round it to the nearest valid size — it refuses it. What
    goes out is the size the pair can actually express, and the audit row shows
    that number rather than the one that was asked for.
    """
    _arm(monkeypatch, desk)
    venue = _Venue()
    monkeypatch.setattr(desk, "_send", venue)

    desk.place_order(OrderIntent(symbol="BTC", side="BUY", quantity=150.53162753))

    assert venue.posts, "the order never left"
    assert venue.signed["quantity"] == "150.53162"
    assert desk.events[0][1]["qty"] == 150.53162      # what the journal promises


def test_the_ladder_is_truncated_never_rounded_up(desk, monkeypatch):
    """0.000015 BTC sits exactly between two steps, and rounding buys the upper
    one — more than the risk layer approved. The only safe direction is down."""
    _arm(monkeypatch, desk)
    venue = _Venue()
    monkeypatch.setattr(desk, "_send", venue)

    desk.place_order(OrderIntent(symbol="BTC", side="BUY", quantity=0.000015))

    assert venue.signed["quantity"] == "0.00001"


def test_a_step_of_ten_units_is_not_the_same_as_no_decimals(desk, monkeypatch):
    """Some pairs trade in whole tens. Truncating the *decimal places* of a step
    is a different operation from truncating to a *multiple* of it, and only the
    second one is what the venue means."""
    _arm(monkeypatch, desk)
    venue = _Venue(rules=_rules(step="10.00000000", min_qty="10.00000000"))
    monkeypatch.setattr(desk, "_send", venue)

    desk.place_order(OrderIntent(symbol="BTC", side="BUY", quantity=153.9))

    assert venue.signed["quantity"] == "150"


def test_the_market_lot_size_of_zero_is_not_used_as_a_step(desk, monkeypatch):
    """A pair can report a market-order step of zero. Read as a ladder it divides
    by nothing or floors every order to nothing; the answer is to ignore it."""
    _arm(monkeypatch, desk)
    venue = _Venue()
    monkeypatch.setattr(desk, "_send", venue)

    desk.place_order(OrderIntent(symbol="BTC", side="BUY", quantity=0.5))

    assert venue.signed["quantity"] == "0.5"


# ---------------------------------------------------------------- the minimums
def test_a_size_under_one_step_is_refused_not_sent_as_zero(desk, monkeypatch):
    """0.0004 on a 0.001 ladder truncates to nothing. An order for quantity "0"
    is not a small order, and the venue's own step is the number worth naming."""
    _arm(monkeypatch, desk)
    venue = _Venue(rules=_rules(step="0.00100000", min_qty="0.00100000"))
    monkeypatch.setattr(desk, "_send", venue)
    intent = OrderIntent(symbol="BTC", side="BUY", quantity=0.0004)

    assert desk.place_order(intent) is None
    assert venue.posts == []
    assert desk.outcome(intent.client_order_id) == "REFUSED"
    assert "step of 0.001" in desk.events[-1][1]["reason"]


def test_a_size_under_the_pair_s_minimum_is_named_not_eaten(desk, monkeypatch):
    """0.00005 BTC is a size this pair can express and an amount it will not take:
    a step of 0.00001, a minimum order of 0.001. The exchange would refuse it with
    a code the operator has to look up; here the journal says which number the
    pair requires and the order never goes out."""
    _arm(monkeypatch, desk)
    venue = _Venue(rules=_rules(step="0.00001000", min_qty="0.00100000"))
    monkeypatch.setattr(desk, "_send", venue)
    intent = OrderIntent(symbol="BTC", side="BUY", quantity=0.00005)

    assert desk.place_order(intent) is None
    assert venue.posts == []
    assert desk.outcome(intent.client_order_id) == "REFUSED"
    assert "minimum order" in desk.events[-1][1]["reason"]


def test_an_order_worth_less_than_the_pair_takes_is_refused(desk, monkeypatch):
    """minNotional is 5 USDT on this pair; one hundredth of a bitcoin at a hundred
    is one."""
    _arm(monkeypatch, desk)
    venue = _Venue()
    monkeypatch.setattr(desk, "_send", venue)
    intent = OrderIntent(symbol="BTC", side="SELL", quantity=0.01,
                         order_type="LIMIT", limit_price=100.0)

    assert desk.place_order(intent) is None
    assert venue.posts == []
    assert "minimum" in desk.events[-1][1]["reason"] and "5" in desk.events[-1][1]["reason"]


# -------------------------------------------------------------------- the tick
def test_a_buy_limit_is_never_placed_above_the_price_meant(desk, monkeypatch):
    """104.997 on a 0.01 tick: rounding up would state a willingness to pay more
    than the desk decided, so a buy goes down to the grid."""
    _arm(monkeypatch, desk)
    venue = _Venue()
    monkeypatch.setattr(desk, "_send", venue)

    desk.place_order(OrderIntent(symbol="BTC", side="BUY", quantity=0.5,
                                 order_type="LIMIT", limit_price=104.997))

    assert venue.signed["price"] == "104.99"


def test_a_sell_limit_is_never_placed_below_the_price_meant(desk, monkeypatch):
    """The same number on the other side: a sell limit under the intended price
    accepts less than was decided, so it goes up. Both directions land on the
    grid, neither crosses what the operator meant."""
    _arm(monkeypatch, desk)
    venue = _Venue()
    monkeypatch.setattr(desk, "_send", venue)

    desk.place_order(OrderIntent(symbol="BTC", side="SELL", quantity=0.5,
                                 order_type="LIMIT", limit_price=104.997))

    assert venue.signed["price"] == "105"


# ------------------------------------------------------------------ the reading
def test_the_ladder_is_read_once_per_pair_not_once_per_order(desk, monkeypatch):
    """The filters are a property of the listing, not of the market, and the
    request that answers them is large. An order path that asked before every
    POST would pay for it in latency on the one path that is measured in
    milliseconds."""
    _arm(monkeypatch, desk)
    venue = _Venue()
    monkeypatch.setattr(desk, "_send", venue)

    desk.place_order(OrderIntent(symbol="BTC", side="BUY", quantity=0.5))
    desk.place_order(OrderIntent(symbol="BTC", side="BUY", quantity=0.5))

    assert venue.reads == 1
    assert len(venue.posts) == 2


def test_rules_that_cannot_be_read_do_not_stop_an_exit(desk, monkeypatch):
    """An exchange that answers nothing about its own metadata is still an
    exchange that can fill a close. Failing closed here would strand a live
    position over a settings read, so the numbers pass through as they are."""
    _arm(monkeypatch, desk)
    venue = _Venue(rules={})                       # the reply held no symbol
    monkeypatch.setattr(desk, "_send", venue)

    fill = desk.place_order(OrderIntent(symbol="BTC", side="SELL", quantity=0.5))

    assert fill is not None
    assert venue.signed["quantity"] == "0.5"
    assert desk.filters("BTCUSDT") is None


def test_a_pair_that_is_not_trading_has_no_rules_to_obey(desk, monkeypatch):
    """The venue keeps the filter list of a halted symbol but marks the symbol. A
    status other than TRADING is not a market to size against."""
    _arm(monkeypatch, desk)
    venue = _Venue(rules=_rules(status="BREAK"))
    monkeypatch.setattr(desk, "_send", venue)

    assert desk.filters("BTCUSDT") is None
    assert venue.reads == 1


# ---------------------------------------------------------------- how a no sounds
def test_a_declined_order_is_a_refusal_not_an_unknown(desk, monkeypatch):
    """HTTP 400 is the exchange reading this order and not taking it. Filed as
    "unconfirmed" it leaves an operator waiting for a fill that will never come,
    and a risk layer holding a position that was never traded."""
    _arm(monkeypatch, desk)
    declined = B.BinanceError("/api/v3/order HTTP 400: "
                              '{"code":-1013,"msg":"Filter failure: LOT_SIZE"}',
                              refused=True)
    venue = _Venue(fail=declined)
    monkeypatch.setattr(desk, "_send", venue)
    intent = OrderIntent(symbol="BTC", side="BUY", quantity=0.5)

    assert desk.place_order(intent) is None
    assert desk.outcome(intent.client_order_id) == "REFUSED"
    kinds = [e[0] for e in desk.events]
    assert "BINANCE_UNCONFIRMED" not in kinds
    assert desk.events[-1][1]["reason"].startswith("venue declined the order")


def test_a_reply_that_never_came_is_still_an_unknown(desk, monkeypatch):
    """The other half, and the reason the two must not share a bucket: with no
    status code there is no way to know whether the engine matched the order, and
    saying "refused" would be a claim about a live position."""
    _arm(monkeypatch, desk)
    down = B.BinanceError("/api/v3/order unreachable (URLError: name not resolved)")
    monkeypatch.setattr(desk, "_send", _Venue(fail=down))
    intent = OrderIntent(symbol="BTC", side="BUY", quantity=0.5)

    assert desk.place_order(intent) is None
    assert desk.outcome(intent.client_order_id) == "UNCONFIRMED"
    assert desk.events[-1][0] == "BINANCE_UNCONFIRMED"


def test_a_server_side_failure_leaves_the_order_unconfirmed(desk, monkeypatch):
    """A 5xx is the venue's own machinery failing, possibly after it matched the
    order. Only a 4xx is an answer about *this* order, and `_send` is what reads
    the status code that tells them apart."""
    _arm(monkeypatch, desk)
    gate = B.BinanceError("/api/v3/order HTTP 500: internal error", refused=False)
    monkeypatch.setattr(desk, "_send", _Venue(fail=gate))
    intent = OrderIntent(symbol="BTC", side="BUY", quantity=0.5)

    assert desk.place_order(intent) is None
    assert desk.outcome(intent.client_order_id) == "UNCONFIRMED"


def test_the_status_code_decides_whether_the_venue_answered(desk, monkeypatch):
    """`_send` is the only place that sees the HTTP status, so it is the only
    place allowed to decide whether the venue answered. Both codes go through the
    real transport rather than through a flag set by hand.
    """
    for code, answered in ((400, True), (403, True), (500, False)):
        exc = urllib.error.HTTPError("https://binance.invalid/api/v3/order",
                                     code, "Bad", {}, None)
        exc.read = lambda: b'{"code":-1013,"msg":"Filter failure"}'

        def raise_it(req, timeout=None, _exc=exc):
            raise _exc

        monkeypatch.setattr(urllib.request, "urlopen", raise_it)
        with pytest.raises(B.BinanceError) as got:
            desk._send("/api/v3/order", {"symbol": "BTCUSDT"},
                       signed=True, method="POST")
        assert got.value.refused is answered, code
