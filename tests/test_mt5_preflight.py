"""Why a live MT5 link cannot exist yet is outside the IPC link.

`attach()` can only say what `initialize()` and `account_info()` answer. Three
failures that actually block a PMEX link never get that far: `mt5.server` is
blank, the only installed terminal belongs to another broker, and the network
already refused the configured account. All three are recorded by MetaTrader 5
itself, in its own UTF-16 logs, and reading them is what turns "not connected"
into an instruction an operator can act on.

No terminal, broker, network or real account number is involved: every fixture
below is a throwaway data directory under `tmp_path`.
"""

from __future__ import annotations

import json
import sys

import pytest

from kairos.config import CFG
from kairos.execution import Mt5Bridge

HASH = "0123456789ABCDEF0123456789ABCDEF"
INSTALL = r"C:\Program Files\MetaTrader 5"


def _terminal(root, lines, *, login_line_encoding="utf-16", when="20261023",
              install=INSTALL, name=HASH):
    term = root / name
    (term / "logs").mkdir(parents=True)
    (term / "origin.txt").write_bytes(install.encode("utf-16"))
    body = "\r\n".join(lines) + "\r\n"
    (term / "logs" / f"{when}.log").write_bytes(body.encode(login_line_encoding))
    return term


def _refusal(account: str = "90024745", server: str = "MetaQuotes-Demo",
             why: str = "Invalid account") -> str:
    return (f"QR\t0\t16:43:09.102\tNetwork\t'{account}': authorization on "
            f"{server} failed ({why})")


def _success(account: str = "90024745", server: str = "PMEX-Live") -> str:
    return (f"QR\t0\t10:01:02.000\tNetwork\t'{account}': authorized on {server} "
            f"through Access Point EU 2 (ping: 92.00 ms, build 6201)")


@pytest.fixture
def cfg(monkeypatch):
    """The PMEX configuration that is actually on this machine: no server."""
    def set_cfg(**kw):
        section = {"venue": "PMEX", "server": "", "login": 90024745, "password": "x"}
        section.update(kw)
        monkeypatch.setattr(CFG, "section",
                            lambda name: section if name == "mt5" else {})
        return section
    return set_cfg


def test_a_demo_only_terminal_does_not_look_like_a_pmex_link(tmp_path, cfg):
    cfg()
    _terminal(tmp_path, [_refusal()])
    pf = Mt5Bridge.preflight(root=str(tmp_path))
    text = " ".join(pf["blockers"])
    assert "mt5.server is blank" in text, (
        "an empty trade server is the reason no sign-in can be aimed at PMEX")
    assert "no terminal here has ever reached a PMEX network" in text
    assert "MetaQuotes-Demo" in text
    assert "Invalid account" in text, "the network's own words, not ours"
    assert pf["networks"] == ["MetaQuotes-Demo"]


def test_the_preflight_never_publishes_the_account_number(tmp_path, cfg):
    cfg()
    _terminal(tmp_path, [_refusal()])
    pf = Mt5Bridge.preflight(root=str(tmp_path))
    blob = json.dumps(pf)
    assert "90024745" not in blob
    assert pf["login_tail"] == "…745"
    assert pf["last_attempt"]["account"] == "…745"
    assert pf["last_attempt_for_this_account"]["account"] == "…745"


def test_another_accounts_refusal_is_evidence_but_not_ours(tmp_path, cfg):
    """The demo account this machine actually holds must not be read as our
    account being refused: a status panel that conflates the two sends the
    operator to fix a login that was never tried."""
    cfg()
    _terminal(tmp_path, [_refusal(account="5055885470")])
    pf = Mt5Bridge.preflight(root=str(tmp_path))
    assert pf["last_attempt"]["account"] == "…470", "the newest answer on this machine"
    assert pf["last_attempt_for_this_account"] is None
    assert not any("refused" in b for b in pf["blockers"])
    assert "MetaQuotes-Demo" in " ".join(pf["blockers"])


def test_a_network_that_already_signed_the_account_in_is_not_called_a_blocker(
        tmp_path, cfg):
    cfg(server="PMEX-Live")
    _terminal(tmp_path, [_success()])
    pf = Mt5Bridge.preflight(root=str(tmp_path))
    assert pf["last_attempt"]["ok"] is True
    assert not any("refused" in b or "blank" in b for b in pf["blockers"])


def test_a_utf8_log_reads_the_same_as_a_utf16_one(tmp_path, cfg):
    cfg(server="PMEX-Live")
    _terminal(tmp_path, [_success()], login_line_encoding="utf-8")
    rows = Mt5Bridge.auth_history(root=str(tmp_path))
    assert rows and rows[0]["server"] == "PMEX-Live" and rows[0]["ok"] is True


def test_the_history_is_filtered_by_the_configured_account(tmp_path, cfg):
    cfg()
    _terminal(tmp_path, [_refusal(account="11111777"),
                         _refusal(account="90024745")], login_line_encoding="utf-8")
    mine = Mt5Bridge.auth_history(root=str(tmp_path), login=90024745)
    assert [r["account"] for r in mine] == ["…745"]
    assert all(r["account"] != "…777" for r in mine)


def test_a_successful_sign_in_is_quoted_as_the_state_not_a_failure(tmp_path):
    _terminal(tmp_path, [_success()])
    clue = Mt5Bridge.signin_clue(90024745, root=str(tmp_path))
    assert "signed …745 in on PMEX-Live" in clue
    assert "refused" not in clue


def test_the_operator_is_told_the_terminal_that_rejected_the_account(tmp_path, cfg):
    cfg()
    _terminal(tmp_path, [_refusal()])
    pf = Mt5Bridge.preflight(root=str(tmp_path))
    assert pf["terminals"] == [{"id": HASH, "install": INSTALL}], (
        "the install a data directory belongs to is read from its own origin.txt")


def test_the_hint_names_the_install_gap_instead_of_advice_that_cannot_be_followed(
        monkeypatch, cfg):
    cfg()
    monkeypatch.setattr(Mt5Bridge, "_scan_auth", staticmethod(lambda root=None: [
        {"terminal": HASH, "install": INSTALL, "login": 90024745,
         "server": "MetaQuotes-Demo", "ok": False, "reason": "Invalid account",
         "when": "20261023"}]))
    hint = Mt5Bridge()._hint()
    assert "PMEX" in hint
    assert "No terminal on this machine has ever reached a PMEX network" in hint
    assert "MetaQuotes-Demo" in hint


def test_attach_says_which_network_refused_the_account(monkeypatch, cfg):
    """The failure message is the whole deliverable here: an operator reading it
    must learn the broker that answered, not merely that nothing is connected."""
    import types

    cfg()
    from kairos.execution import Mt5Bridge as B
    monkeypatch.setattr(B, "_last_attach_try", 0.0)
    monkeypatch.setattr(B, "_running_terminal_path",
                        staticmethod(lambda cached_only=False: ""))
    monkeypatch.setattr(B, "_scan_auth", staticmethod(lambda root=None: [
        {"terminal": HASH, "install": INSTALL, "login": 90024745,
         "server": "MetaQuotes-Demo", "ok": False, "reason": "Invalid account",
         "when": "20261023"}]))
    monkeypatch.setitem(sys.modules, "MetaTrader5",
                        types.SimpleNamespace(initialize=lambda **k: False,
                                              last_error=lambda: (-6, "no link"),
                                              account_info=lambda: None,
                                              shutdown=lambda: None))

    br = B()
    assert br.attach(login=90024745, password="x", server="") is False
    err = br.last_error
    assert "MetaQuotes-Demo" in err and "Invalid account" in err
    assert "…745" in err, "the account is quoted masked, never in full"
    assert "90024745" not in err


def test_a_machine_with_no_terminal_data_is_not_accused_of_using_a_demo(
        tmp_path, cfg):
    cfg()
    pf = Mt5Bridge.preflight(root=str(tmp_path))
    assert pf["terminals"] == [] and pf["networks"] == []
    assert not any("has ever reached" in b for b in pf["blockers"]), (
        "with no evidence at all the preflight says nothing it cannot prove")
