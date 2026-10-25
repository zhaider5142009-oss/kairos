"""The status command must survive the book it exists to explain.

`PaperBroker.account()` returns `equity: None` when a holding cannot be marked
— a dollar name with no usable cross into a rupee book — and the desk already
prints "unreadable" for that state. `status` did not: it formatted the field
with `:,.0f` after `.get("equity", 0)`, which hands back the stored None rather
than the default, so the command raised TypeError at exactly the moment an
operator runs it to find out why the book looks wrong. A clean zero here would
be the other wrong answer: the paper book is not flat, it is unmeasurable.
"""

from __future__ import annotations

from io import StringIO
from types import SimpleNamespace

from rich.console import Console

import kairos.autopilot as AP
import kairos.cli as CLI


def _st(**acct) -> dict:
    return {"mode": "paper", "running": True, "cycles": 7,
            "last_cycle": "2026-10-25T12:00:00", "interval_s": 600,
            "entry_threshold": 0.18, "kill_switch": False,
            "signals": [], "actions": [], "paper_positions": [],
            "risk_posture": {}, "paper_account": acct}


def _run(monkeypatch, st: dict) -> str:
    monkeypatch.setattr(AP.AUTOPILOT, "status", lambda: st)
    buf = StringIO()
    monkeypatch.setattr(CLI, "_console",
                        lambda: Console(file=buf, width=140, color_system=None))
    args = SimpleNamespace(what="status", symbols=None, interval=None,
                           threshold=None, max_trades=None, mode="paper",
                           yes_live=False)
    assert CLI.cmd_autopilot(args) == 0
    return buf.getvalue()


def test_status_prints_unreadable_instead_of_raising(monkeypatch):
    out = _run(monkeypatch, _st(equity=None, day_pnl=None))
    assert "paper equity: unreadable" in out
    assert "day P&L: unreadable" in out


def test_status_still_prints_the_numbers_when_it_has_them(monkeypatch):
    out = _run(monkeypatch, _st(equity=993_737.93, day_pnl=-3_501.69))
    assert "paper equity: 993,738" in out
    assert "day P&L: -3,502" in out
