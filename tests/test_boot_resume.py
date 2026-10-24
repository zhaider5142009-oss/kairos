"""The one call that starts the trading loop, and the reason it must give back.

`_auto_resume` runs on every boot of the desk. Before it was written, the boot
handler called `AUTOPILOT.start("paper")`, threw the return value away, and
caught every exception into `pass` — so a refused resume, a crash, and an
operator who had deliberately stopped the loop all produced the same desk:
"not running", no reason. The reason only exists at this moment, and used to be
discarded at the only moment it was ever available.

These are the repository's first tests that enter the app's real startup path.
Every one of them holds `start()` by a stub first: the hook is on the module
singleton, and a thread that outlives a fixture has no broker stub left to guard
it — which is why `tests/test_news.py` calls the desk endpoint as a function
rather than over HTTP.
"""

from __future__ import annotations

import inspect
import pathlib

import pytest
from fastapi.testclient import TestClient

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _boot(monkeypatch, start):
    """Run the app's startup path with the loop's door held by `start`."""
    from kairos.api import server as SV
    from kairos.autopilot import AUTOPILOT

    monkeypatch.setattr(AUTOPILOT, "start", start)
    with TestClient(SV.app):
        pass
    return AUTOPILOT


@pytest.fixture
def ap_error():
    """The singleton's error field, pinned to empty and restored after."""
    from kairos.autopilot import AUTOPILOT

    saved = AUTOPILOT.st.get("last_error")
    AUTOPILOT.st["last_error"] = None
    try:
        yield AUTOPILOT
    finally:
        AUTOPILOT.st["last_error"] = saved


def test_boot_asks_for_the_paper_loop_and_never_for_the_live_one(monkeypatch, ap_error):
    """The boot resume is paper-only by doctrine: a restart may never arm real
    money, so the mode and the confirmation flag are both pinned here."""
    calls: list[tuple[str, bool]] = []

    def start(mode: str = "paper", confirm_live: bool = False) -> dict:
        calls.append((mode, bool(confirm_live)))
        return {"ok": True, "mode": mode}

    _boot(monkeypatch, start)
    assert calls == [("paper", False)], f"boot asked for {calls}"


def test_a_refused_resume_lands_in_the_field_the_desk_already_prints(monkeypatch, ap_error):
    """`start()` returns a refusal instead of raising, so a caller that ignores
    the return value cannot fail visibly — and the desk reads exactly this
    field, so the reason reaches a person without a new channel."""
    def start(mode: str = "paper", confirm_live: bool = False) -> dict:
        return {"ok": False, "error": "state file unreadable"}

    ap = _boot(monkeypatch, start)
    assert ap.st["last_error"] == "boot: state file unreadable"
    # ...and it reaches the payload the Pro Desk tab renders.
    assert ap.status()["last_error"] == "boot: state file unreadable"


def test_an_exception_at_boot_is_reported_not_swallowed(monkeypatch, ap_error):
    """The handler this replaced caught everything into `pass`. A crash at boot
    is now named with its type, because the type is the part that says which
    subsystem refused to come up."""
    def start(mode: str = "paper", confirm_live: bool = False) -> dict:
        raise RuntimeError("state file unreadable")

    ap = _boot(monkeypatch, start)
    assert ap.st["last_error"] == "boot: raised RuntimeError: state file unreadable"


@pytest.mark.parametrize("error", ["autopilot already running",
                                   "autopilot already running on this box"])
def test_a_loop_that_is_already_up_is_not_reported_as_a_failure(monkeypatch, ap_error, error):
    """A restart onto a loop that is still running is the normal case, not a
    fault. `start()` refuses with 'already running', and writing that into
    `last_error` would teach an operator to ignore the field."""
    ap_error.st["last_error"] = "cycle: an older fault"

    def start(mode: str = "paper", confirm_live: bool = False) -> dict:
        return {"ok": False, "error": error}

    ap = _boot(monkeypatch, start)
    assert ap.st["last_error"] == "cycle: an older fault"


def test_a_resume_that_worked_writes_no_boot_note(monkeypatch, ap_error):
    """A successful boot says nothing: the note is only for what did not happen.
    An error from the previous life of the process stays exactly where it was,
    rather than being overwritten by a success."""
    ap_error.st["last_error"] = "cycle: an older fault"

    def start(mode: str = "paper", confirm_live: bool = False) -> dict:
        return {"ok": True, "mode": mode}

    ap = _boot(monkeypatch, start)
    assert ap.st["last_error"] == "cycle: an older fault"


def test_an_operator_who_turned_auto_start_off_is_not_overridden(monkeypatch, ap_error):
    """`auto_start_paper: false` means a human stopped the loop on purpose. Boot
    must not argue with that, and must not start a trading thread to prove it."""
    from kairos.api import server as SV

    calls: list[str] = []

    def start(mode: str = "paper", confirm_live: bool = False) -> dict:
        calls.append(mode)
        return {"ok": True}

    monkeypatch.setattr(SV.CFG, "section",
                        lambda name: {"auto_start_paper": False} if name == "autopilot" else {})
    _boot(monkeypatch, start)
    assert calls == [], "boot overrode the operator's own switch"


def test_the_resume_is_a_lifespan_and_the_app_still_answers(monkeypatch, ap_error):
    """`@app.on_event("startup")` is deprecated, and the way a deprecated hook
    fails is not a warning in a log — it is a trading loop that never starts on
    a system advertised as resuming itself. So the empty `on_startup` list is
    the assertion, and the health call proves the refactor still serves."""
    from kairos.api import server as SV
    from kairos.autopilot import AUTOPILOT

    calls: list[str] = []

    def start(mode: str = "paper", confirm_live: bool = False) -> dict:
        calls.append(mode)
        return {"ok": True}

    monkeypatch.setattr(SV.CFG, "section", lambda name: {})
    monkeypatch.setattr(AUTOPILOT, "start", start)
    with TestClient(SV.app) as client:
        assert calls == ["paper"]
        assert client.get("/api/health").status_code == 200
    assert SV.app.router.on_startup == [], "the resume is back on the deprecated hook"
    assert SV.app.router.lifespan_context is SV._lifespan


def test_the_error_field_is_still_read_by_the_page_that_shows_it():
    """The fix is only a fix if something renders it. Both halves of the wire
    are checked here: the field leaves `status()`, and the desk row prints it."""
    from kairos.autopilot import AUTOPILOT

    assert '"last_error"' in inspect.getsource(AUTOPILOT.status)
    html = (ROOT / "kairos" / "api" / "ui.html").read_text(encoding="utf-8")
    assert "d.last_error" in html, "the desk stopped rendering the boot note"


def test_the_handler_this_replaced_could_not_have_failed_visibly(monkeypatch, ap_error):
    """Falsification: the removed body, rebuilt verbatim, beside the new one.

    Given the same refusal, the old shape records nothing — not because the
    refusal is quiet, but because a returned value that is never read has no
    route to a human. If this test ever fails on the first assertion, the old
    handler has grown a memory and the comparison no longer proves anything."""
    def refusal(mode: str = "paper", confirm_live: bool = False) -> dict:
        return {"ok": False, "error": "state file unreadable"}

    def old_handler():  # the removed startup body, verbatim
        try:
            refusal("paper")
        except Exception:
            pass

    old_handler()
    assert ap_error.st["last_error"] is None, (
        "the rebuilt old handler now leaves a trace; this test is comparing "
        "nothing against anything")

    _boot(monkeypatch, refusal)
    assert ap_error.st["last_error"] == "boot: state file unreadable"
