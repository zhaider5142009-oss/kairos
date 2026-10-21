"""The pre-trade gate must obey the config file, not the config file's memory.

Three defects these guard:

1. `RiskGate.__init__` snapshotted the `risk:` block when the object was built.
   The autopilot builds one gate for the life of the process, so a loss limit
   tightened while the tower ran kept the old, looser number — the operator
   believed they were protected from the next order, and were not until a
   restart. Limits are now read on every check.
2. `_seen_coids` was an unbounded set in a process that runs for days.
3. The equity guards were keyed on `> 0`, so an account reading that was
   negative — a margin call — satisfied "we have an equity figure" and then
   opted out of every check that needed one.
"""

from __future__ import annotations

import os
import time

import pytest
import yaml

from kairos import execution as E
from kairos.config import Config

BASE = """\
app:
  data_dir: data
risk:
  max_positions: 5
  daily_loss_limit_pct: 0.10
  max_order_value_pct: 0.10
  max_order_qty: 100000
"""


class _Kill:
    @staticmethod
    def engaged():
        return False


@pytest.fixture
def gate(tmp_path, monkeypatch):
    f = tmp_path / "config.yaml"
    f.write_text(BASE, encoding="utf-8")
    monkeypatch.setattr(E, "CFG", Config(f))
    monkeypatch.setattr(E, "KILL", _Kill())
    return E.RiskGate()


def _bump(path, seconds: int = 10):
    """Force a distinct mtime; two writes can land on the same clock tick."""
    st = path.stat()
    os.utime(path, (st.st_atime, st.st_mtime + seconds))


def _intent(**kw):
    kw.setdefault("symbol", "OGDC")
    kw.setdefault("side", "BUY")
    kw.setdefault("quantity", 100)
    return E.OrderIntent(**kw)


def _check(g, acct=None, pos=None):
    return g.check(_intent(), acct if acct is not None else {"equity": 100_000.0,
                                                             "day_pnl": -4_000.0},
                   pos if pos is not None else [])


def test_tightened_loss_limit_binds_without_a_restart(gate):
    ok, _ = _check(gate)
    assert ok, "-4k on 100k equity is inside a 10% limit"

    raw = yaml.safe_load(E.CFG.path.read_text(encoding="utf-8"))
    raw["risk"]["daily_loss_limit_pct"] = 0.01
    E.CFG.path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    _bump(E.CFG.path)
    E.CFG._next_check = time.monotonic() - 1

    ok, why = _check(gate)
    assert not ok and "daily loss limit" in why


def test_relaxed_limit_releases_without_a_restart(gate):
    raw = yaml.safe_load(E.CFG.path.read_text(encoding="utf-8"))
    raw["risk"]["daily_loss_limit_pct"] = 0.001
    E.CFG.path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    _bump(E.CFG.path)
    E.CFG._next_check = time.monotonic() - 1
    assert not _check(gate)[0]

    raw["risk"]["daily_loss_limit_pct"] = 0.50
    E.CFG.path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    _bump(E.CFG.path, 20)
    E.CFG._next_check = time.monotonic() - 1
    assert _check(gate)[0]


def test_missing_config_keys_fall_back_to_the_safe_default(gate, tmp_path):
    raw = yaml.safe_load(E.CFG.path.read_text(encoding="utf-8"))
    del raw["risk"]["max_order_qty"]
    E.CFG.path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    _bump(E.CFG.path)
    E.CFG._next_check = time.monotonic() - 1
    ok, _ = _check(gate)
    assert ok                                       # default of 100000 still applies


def test_fail_closed_on_a_missing_account_snapshot(gate):
    ok, why = gate.check(_intent(), None, [])
    assert not ok and "fail-closed" in why
    ok, why = gate.check(_intent(), {"equity": 100_000.0}, None)
    assert not ok and "fail-closed" in why


def test_a_negative_equity_fails_closed_instead_of_blinding_the_loss_check(gate):
    """A margin-call account is not a tradeable one.

    The daily-loss comparison is guarded by `equity > 0`, so a negative reading
    slipped past the zero/absent test and then skipped the loss check entirely —
    the one account state that most deserves a hard stop. An unreadable equity
    used to raise out of the middle of the gate instead of answering "no".
    """
    for acct in ({"equity": -250.0, "day_pnl": -90_000.0},
                 {"equity": 0.0, "day_pnl": 0.0},
                 {"equity": "n/a"},
                 {"equity": None}):
        ok, why = gate.check(_intent(), acct, [])
        assert not ok, f"{acct} must not be allowed to trade"


def test_an_unusable_number_stops_at_the_gate(gate):
    """Bounds are comparisons, and NaN loses every comparison there is.

    `quantity <= 0 or quantity > max` reads False for a NaN, so a hole in the
    feed cleared both bounds; the limit-price check multiplies the figure and
    then compares it — also False. This is the gate every venue's order passes
    through, paper included, and a NaN fill lands in the same book the
    published track record is read from.
    """
    for kw in ({"quantity": float("nan")},
               {"quantity": float("inf")},
               {"order_type": "LIMIT", "limit_price": float("nan")},
               {"order_type": "LIMIT", "limit_price": float("inf")}):
        ok, why = gate.check(_intent(**kw), {"equity": 100_000.0}, [])
        assert not ok, f"{kw} must not be allowed to trade"
    assert gate.check(_intent(), {"equity": 100_000.0}, [])[0]


def test_same_client_order_id_is_rejected_once(gate):
    intent = _intent()
    assert gate.check(intent, {"equity": 100_000.0}, [])[0]
    ok, why = gate.check(intent, {"equity": 100_000.0}, [])
    assert not ok and "duplicate" in why


def test_idempotency_window_is_bounded(gate, monkeypatch):
    monkeypatch.setattr(E, "MAX_SEEN_CIDS", 50)
    acct, pos = {"equity": 100_000.0}, []
    for _ in range(400):
        gate.check(_intent(), acct, pos)
    assert len(gate._seen_coids) == 50
    # the oldest ids are gone; a long-running process cannot grow this forever
    assert gate.check(_intent(), acct, pos)[0]


def test_kill_switch_still_blocks_before_any_config_read(gate, monkeypatch):
    class _Engaged:
        @staticmethod
        def engaged():
            return True

    monkeypatch.setattr(E, "KILL", _Engaged())
    ok, why = gate.check(_intent(), {"equity": 100_000.0}, [])
    assert not ok and "kill switch" in why
