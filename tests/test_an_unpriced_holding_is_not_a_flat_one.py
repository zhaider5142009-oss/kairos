"""A holding nobody can price has no P&L — and that is not zero.

`livebook.review()` catches a failed prediction and returns a row with a status
and an error but no `last`, `pnl` or `pnl_pct`. Every reader then filled the gap
with `.get(k, 0)`: the CLI table printed `0.00` beside a green `+0`, the chat
totalled the book over the rows it could price and called the sum the book's,
and the desk painted the missing P&L red. All three invented a flat position —
the same fail-open the desk already refuses elsewhere, printed on the screen an
operator reads before deciding whether the book is fine.
"""

from __future__ import annotations

from io import StringIO
from types import SimpleNamespace

from rich.console import Console

import kairos.cli as CLI
import kairos.command as CMD
import kairos.engine as ENGINE_MOD
import kairos.livebook as LB


def _book(monkeypatch, predict) -> None:
    monkeypatch.setattr(LB, "positions", lambda: {
        "OGDC": {"qty": 100, "entry": 120.0, "stop": None}})
    monkeypatch.setattr(ENGINE_MOD.ENGINE, "predict", predict)
    monkeypatch.setattr(LB, "_cache", None)


def _unpriced(**extra) -> dict:
    row = {"symbol": "OGDC", "qty": 100, "entry": 120.0, "stop": None,
           "status": "NO DATA", "error": "no history for OGDC",
           "last": None, "pnl": None, "pnl_pct": None}
    row.update(extra)
    return row


def test_a_failed_prediction_leaves_no_price_and_no_pnl(monkeypatch):
    def boom(_sym):
        raise ValueError("no history for OGDC")
    _book(monkeypatch, boom)
    row = LB.review()[0]
    assert row["status"] == "NO DATA"
    assert row["last"] is None and row["pnl"] is None and row["pnl_pct"] is None
    assert row["qty"] == 100 and row["entry"] == 120.0
    assert "no history" in row["error"]


def test_the_cli_table_says_unreadable_instead_of_zero(monkeypatch):
    monkeypatch.setattr(LB, "review", lambda force=False: [_unpriced()])
    buf = StringIO()
    monkeypatch.setattr(CLI, "_console",
                        lambda: Console(file=buf, width=140, color_system=None))
    args = SimpleNamespace(add=None, remove=None, sync=False, ticket=None)
    assert CLI.cmd_livebook(args) == 0
    out = buf.getvalue()
    assert "unreadable" in out
    assert "NO DATA" in out
    # the invented flat position: no P&L, no green zero
    assert "+0" not in out


def test_the_cli_table_still_prints_a_priced_holding(monkeypatch):
    priced = {"symbol": "OGDC", "qty": 100, "entry": 120.0, "stop": 118.0,
              "status": "HOLD — in profit", "last": 124.5, "pnl": 450.0,
              "pnl_pct": 3.75, "direction": "UP"}
    monkeypatch.setattr(LB, "review", lambda force=False: [priced])
    buf = StringIO()
    monkeypatch.setattr(CLI, "_console",
                        lambda: Console(file=buf, width=140, color_system=None))
    args = SimpleNamespace(add=None, remove=None, sync=False, ticket=None)
    assert CLI.cmd_livebook(args) == 0
    out = buf.getvalue()
    assert "124.50" in out and "+450" in out
    assert "118.00" in out
    assert "unreadable" not in out


def test_the_summary_total_says_which_rows_it_covers():
    priced = {"symbol": "HBL", "qty": 50, "entry": 100.0, "last": 109.0,
              "pnl": 450.0, "pnl_pct": 9.0, "status": "HOLD — in profit"}
    text = CMD._l_livebook({"rows": [priced, _unpriced()], "count": 2})
    assert "unrealised +450 on 1 of 2 priced" in text
    assert "OGDC NO DATA (price unreadable)" in text


def test_a_wholly_unpriced_book_says_unknown_not_zero():
    text = CMD._l_livebook({"rows": [_unpriced()], "count": 1})
    assert "unrealised unknown (0 of 1 priced)" in text
    assert "unrealised +0" not in text
