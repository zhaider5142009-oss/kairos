"""Attaching to *an* MT5 terminal is not the same as attaching to *your* account.

The bridge was written when one broker terminal existed on this machine. With a
JS Global/MT5 account and a PMEX futures account both reachable, the old
semantics — "some `terminal64.exe` is open, so connect and read
`account_info()`" — means KAIROS can size positions, judge free margin and
enforce the daily loss break against the equity of an account the operator never
meant to trade. `terminal_path` was accepted by `attach()` and then dropped on
the floor, so the one setting that could have pinned the install did nothing
either.

These tests stub the `MetaTrader5` module and a fake running-process list; no
terminal, broker or network is involved.
"""

from __future__ import annotations

import sys
import types

import pytest

from kairos.execution import Mt5Bridge


class _Account:
    def __init__(self, login: int, server: str = "SOME-SERVER"):
        self.login = login
        self.server = server
        self.equity = 1000.0
        self.balance = 1000.0
        self.currency = "USD"
        self.margin_free = 900.0
        self.leverage = 100
        self.profit = 0.0


class _StubMt5:
    """The slice of the MetaTrader5 module that attach() actually touches."""

    def __init__(self, account: _Account | None, ok: bool = True):
        self.account = account
        self.ok = ok
        self.kwargs_seen: dict = {}
        self.shutdowns = 0

    def initialize(self, **kwargs):
        self.kwargs_seen = kwargs
        return self.ok

    def account_info(self):
        return self.account

    def last_error(self):
        return "stubbed failure"

    def shutdown(self):
        self.shutdowns += 1


@pytest.fixture
def bridge(monkeypatch):
    """A bridge with the throttle lifted and no process discovery."""
    monkeypatch.setattr(Mt5Bridge, "_last_attach_try", 0.0)
    b = Mt5Bridge()
    yield b
    monkeypatch.setattr(Mt5Bridge, "_last_attach_try", 0.0)


def _stub(monkeypatch, account, ok=True) -> _StubMt5:
    stub = _StubMt5(account, ok)
    monkeypatch.setitem(sys.modules, "MetaTrader5", stub)  # type: ignore[arg-type]
    return stub


def _running(monkeypatch, path: str):
    monkeypatch.setattr(Mt5Bridge, "_running_terminal_path",
                        staticmethod(lambda cached_only=False: path))


def test_attach_refuses_a_terminal_signed_into_another_account(bridge, monkeypatch):
    """The money-path case: wrong account, no connection, connection closed."""
    _running(monkeypatch, r"C:\MT5\terminal64.exe")
    stub = _stub(monkeypatch, _Account(login=5055885470, server="MetaQuotes-Demo"))
    attached = bridge.attach(login=90001, password="x", server="PMEX-Live")
    assert attached is False, "a foreign account must never look like ours"
    assert bridge.connected is False
    assert stub.shutdowns == 1, "the IPC link is closed, the app is left alone"
    assert "90001" in bridge.last_error and "5055885470" in bridge.last_error
    assert "MetaQuotes-Demo" in bridge.last_error, (
        "the operator needs to know which account is actually open, not just that "
        "something was wrong")


def test_attach_succeeds_when_the_signed_in_account_is_the_configured_one(bridge, monkeypatch):
    _running(monkeypatch, r"C:\MT5\terminal64.exe")
    stub = _stub(monkeypatch, _Account(login=90001, server="PMEX-Live"))
    assert bridge.attach(login=90001, password="x", server="PMEX-Live") is True
    assert bridge.connected is True
    assert bridge.attached_account == {"login": 90001, "server": "PMEX-Live"}
    assert stub.shutdowns == 0


def test_a_configured_path_that_is_not_the_running_install_refuses(bridge, monkeypatch):
    """`initialize(path=…)` launches a terminal when the path is not already open.

    That is the one behaviour this bridge promises it does not have, so the path
    is forwarded only after the running process is confirmed to be that install.
    """
    _running(monkeypatch, r"C:\Program Files\MetaTrader 5\terminal64.exe")
    stub = _stub(monkeypatch, _Account(login=90001))
    other = r"D:\PMEX\terminal64.exe"
    assert bridge.attach(login=90001, password="x", server="S",
                         terminal_path=other) is False
    assert stub.kwargs_seen == {}, "nothing was forwarded, so nothing was launched"
    assert "second terminal" in bridge.last_error


def test_the_matching_install_path_is_forwarded(bridge, monkeypatch):
    _running(monkeypatch, r"C:\Program Files\MetaTrader 5\terminal64.exe|"
                         r"D:\PMEX\terminal64.exe")
    stub = _stub(monkeypatch, _Account(login=90001))
    assert bridge.attach(login=90001, password="x", server="S",
                         terminal_path=r"D:\PMEX\terminal64.exe") is True
    assert stub.kwargs_seen.get("path") == r"D:\PMEX\terminal64.exe"
    assert stub.kwargs_seen.get("login") == 90001


def test_no_credentials_in_config_still_attaches_to_the_open_terminal(bridge, monkeypatch):
    """Documented fallback: with nothing to compare against, attach to what is
    running — but say which account that turned out to be."""
    _running(monkeypatch, r"C:\MT5\terminal64.exe")
    _stub(monkeypatch, _Account(login=12345, server="ANY"))
    assert bridge.attach() is True
    assert bridge.attached_account["login"] == 12345


def test_the_instruction_names_the_configured_venue_not_a_lucky_guess(bridge, monkeypatch):
    """The old text told a PMEX operator to sign in to JS Markets — wrong advice,
    delivered at exactly the moment the operator is looking for the right advice."""
    from kairos.config import CFG

    monkeypatch.setattr(CFG, "section",
                        lambda name: ({"venue": "PMEX", "server": "PMEX-Live"}
                                      if name == "mt5" else {}))
    _running(monkeypatch, "")
    _stub(monkeypatch, _Account(login=1))
    assert bridge.attach(login=1, password="x", server="PMEX-Live") is False
    assert "PMEX" in bridge.last_error and "JS Markets" not in bridge.last_error
    bridge._mt5 = None
    assert "JS Markets" not in bridge._hint()


# --------------------------------------------------------------- respawn arm --
# A loop that trades 24/7 cannot depend on a GUI app somebody opened once, so the
# operator may arm `mt5.spawn`. Three things must hold: it is off unless armed,
# it never puts a credential on a command line (every local process can read one,
# crash dumps included), and it is bounded in time.

def _mt5_cfg(monkeypatch, **keys):
    from kairos.config import CFG
    section = {"venue": "PMEX", "login": 90001, "password": "x", "server": "S"}
    section.update(keys)
    monkeypatch.setattr(CFG, "section",
                        lambda name: section if name == "mt5" else {})
    return section


def test_an_unarmed_bridge_does_not_start_anything(bridge, monkeypatch):
    launches: list = []
    import subprocess
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: launches.append((a, k)))
    _mt5_cfg(monkeypatch)
    _running(monkeypatch, "")
    stub = _stub(monkeypatch, _Account(login=90001))
    assert bridge.attach(login=90001, password="x", server="S") is False
    assert launches == [], "spawning the terminal is the operator's word, not ours"
    assert "spawn" in bridge.last_error.lower() or "not running" in bridge.last_error.lower()
    assert stub.kwargs_seen == {}


def test_a_spawn_of_a_path_that_is_not_a_file_refuses(bridge, monkeypatch):
    import subprocess
    launches: list = []
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: launches.append(a))
    _mt5_cfg(monkeypatch, spawn=True,
             terminal_path=r"C:\definitely\not\here\terminal64.exe")
    _running(monkeypatch, "")
    _stub(monkeypatch, _Account(login=90001))
    assert bridge.attach(login=90001, password="x", server="S",
                         terminal_path=r"C:\definitely\not\here\terminal64.exe") is False
    assert launches == []
    assert "not a file" in bridge.last_error


def test_an_armed_spawn_starts_the_configured_terminal_and_then_attaches(bridge, monkeypatch,
                                                                         tmp_path):
    """The argv is the whole test: one element, the path, and no secret in it."""
    import subprocess
    exe = tmp_path / "terminal64.exe"
    exe.write_bytes(b"not a real terminal")
    argv: list = []

    def fake_popen(args, *a, **k):
        argv.append(list(args))
        _running(monkeypatch, str(exe))     # the terminal "appears"
        return types.SimpleNamespace()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    _mt5_cfg(monkeypatch, spawn=True, terminal_path=str(exe))
    _running(monkeypatch, "")
    stub = _stub(monkeypatch, _Account(login=90001, server="S"))
    assert bridge.attach(login=90001, password="x", server="S",
                         terminal_path=str(exe)) is True
    assert argv == [[str(exe)]], (
        f"the terminal was launched with {argv} — credentials must never travel on "
        "a command line, where any local process can read them")
    assert stub.kwargs_seen.get("path") == str(exe)


def test_a_spawn_that_never_produces_a_terminal_gives_up_on_a_deadline(bridge, monkeypatch,
                                                                       tmp_path):
    import subprocess
    exe = tmp_path / "terminal64.exe"
    exe.write_bytes(b"not a real terminal")
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: types.SimpleNamespace())
    _mt5_cfg(monkeypatch, spawn=True, terminal_path=str(exe))
    _running(monkeypatch, "")
    _stub(monkeypatch, _Account(login=90001))
    monkeypatch.setattr(Mt5Bridge, "_SPAWN_WAIT_S", 0.0)
    assert bridge.attach(login=90001, password="x", server="S",
                         terminal_path=str(exe)) is False
    assert "no terminal64 appeared" in bridge.last_error

