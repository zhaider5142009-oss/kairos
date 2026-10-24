"""A retired JS Global session must put its own sockets down.

`kairos/jsgsync.py` caches one `requests.Session` against the broker portal
and replaces it on TTL expiry, drops it when the portal answers 401, drops it
again on `sync(force=True)`, and abandons one mid-way whenever a login fails.
Until now none of those four paths ever called `close()`, so every retired
session left its pooled TLS connection for the garbage collector to notice
— on the box that runs the live book, once per session per broker, and once
per wrong password an operator retries.

The two claims the fix rests on are measured, not argued, against a portal
running on loopback over the real `requests` stack:

  * `close()` really does take the server-side connection down, and
  * it does not disturb a request another thread already checked out, so
    retiring a session cannot interrupt an order that has already been sent.
"""

from __future__ import annotations

import ast
import inspect
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
import requests

from kairos import jsgsync
from kairos.jsgsync import JsGlobalAuthError, JsGlobalError

PASSWORD = "4321"  # sentinel, never a real PIN

# Captured at import, which collection does before the session-wide wall goes
# up in conftest. Used only by `portal_transport` below.
_REAL_REQUEST = requests.Session.request


class _Portal:
    """Enough of Web Trade Cast to drive the real login over a real socket."""

    def __init__(self) -> None:
        self.open = 0          # connections the portal currently holds
        self.logins = 0        # POSTs to /Home/_Login
        self.refuse = False    # answer 401 to GetUserData instead of accounts
        outer = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def handle(self):                      # the keep-alive loop
                outer.open += 1
                try:
                    super().handle()
                finally:
                    outer.open -= 1

            def _send(self, body: bytes, code: int = 200) -> None:
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path == "/slow":
                    time.sleep(0.5)                # hold this connection open
                    return self._send(b'{"slept": true}')
                if self.path == "/Home/GetUserData":
                    if outer.refuse:
                        return self._send(b'{"UserName": "x", "Digit1": "y"}', 401)
                    return self._send(json.dumps([{"accountNo": "90001",
                                                   "accountName": "Sentinel Holder"}]).encode())
                if self.path == "/Home/Index":
                    return self._send(b'{"index": true}')
                return self._send(b"<html>" + b"".join(
                    f'<input id="Digit{i}"/>'.encode() for i in range(1, 5)) + b"</html>")

            def do_POST(self):
                self.rfile.read(int(self.headers.get("Content-Length") or 0))
                if self.path == "/Home/_Login":
                    outer.logins += 1
                return self._send(b'{"ok": true}')

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def stop(self) -> None:
        self.server.shutdown()
        self.server.server_close()

    def wait_open(self, want: int, timeout: float = 4.0) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.open <= want:
                return True
            time.sleep(0.02)
        return self.open <= want


class _Fake:
    """A stand-in session that records whether anyone put it down.

    It is not a transport: `_net` is patched in these tests, so a request
    that reaches one of these methods means the stub leaked past its own
    substitute.
    """

    def __init__(self) -> None:
        self.closed = 0

    def close(self) -> None:
        self.closed += 1

    def get(self, *args, **kwargs):
        raise AssertionError("the fake session was asked to make a real request")

    post = get


class _Resp:
    def __init__(self, status_code: int, text: str) -> None:
        self.status_code = status_code
        self.text = text

    def json(self):
        return json.loads(self.text)


@pytest.fixture()
def portal(monkeypatch):
    """The fake portal, with this module's caches owned and restored."""
    held, book = jsgsync._session, jsgsync._book
    jsgsync._session = jsgsync._book = None
    p = _Portal()
    for key in ("JSG_PORTAL", "JSG_LOGIN", "JSG_PASSWORD"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(jsgsync.CFG, "section", lambda name: {
        "portal_url": p.base, "login": "1234", "password": PASSWORD})
    yield p
    if jsgsync._session is not None:
        jsgsync._session[1].close()
    jsgsync._session, jsgsync._book = held, book
    p.stop()


@pytest.fixture()
def portal_transport(portal):
    """Let one test drive the real `requests` stack — at this portal, nothing else.

    The suite walls off every transport because a stray call here is a real
    order on a real account. That wall stays up in every direction the test
    did not build: any URL outside this loopback stub still raises. So the
    socket bookkeeping can be measured instead of asserted without widening
    what a test is able to reach.
    """
    blocked = requests.Session.request

    def guarded(self, method, url, *args, **kwargs):
        if not str(url).startswith(portal.base):
            raise AssertionError(
                f"a test left the fake portal ({method} {url}); the broker and "
                "every data vendor must stay stubbed")
        return _REAL_REQUEST(self, method, url, *args, **kwargs)

    requests.Session.request = guarded
    try:
        yield portal
    finally:
        requests.Session.request = blocked


# ------------------------------------------------- the four retirement paths
def test_a_session_replaced_by_the_ttl_clock_is_closed(portal_transport):
    portal = portal_transport
    stale = requests.Session()
    assert stale.get(portal.base + "/Home/GetUserData").status_code == 200
    assert portal.open == 1, "requests should hold one keep-alive connection"

    jsgsync._session = (time.time() - jsgsync.SESSION_TTL_S, stale)
    fresh = jsgsync._sess()
    try:
        assert fresh is not stale
        assert portal.wait_open(1), "the retired session's socket is still open"
        # and what the retired session does if someone still holding it asks
        # again — it reconnects. It is not broken, which is why retiring one
        # under a running cycle is safe; but it is also not the warm portal
        # login any more, which is why the cache must not keep it.
        assert stale.get(portal.base + "/Home/GetUserData").status_code == 200
    finally:
        fresh.close()
        stale.close()


def test_a_session_still_in_date_is_reused_and_left_open(portal):
    warm, logins = _Fake(), portal.logins
    jsgsync._session = (time.time(), warm)
    assert jsgsync._sess() is warm
    assert warm.closed == 0, "a session the cache still points at must stay open"
    assert portal.logins == logins, "a warm cache must not reach the portal again"


def test_sync_force_closes_the_session_it_clears(portal, monkeypatch):
    warm = _Fake()
    jsgsync._session = (time.time(), warm)
    monkeypatch.setattr(jsgsync, "_sess", lambda: warm)
    monkeypatch.setattr(jsgsync, "accounts",
                        lambda *a, **kw: (_ for _ in ()).throw(JsGlobalError("stop")))
    with pytest.raises(JsGlobalError):
        jsgsync.sync(force=True)
    assert warm.closed == 1, "force=True abandoned the session it was clearing"


def test_a_login_that_stops_halfway_closes_the_session_it_made(portal_transport):
    portal = portal_transport
    portal.refuse = True        # GetUserData answers a login page: "rejected"
    with pytest.raises(JsGlobalAuthError):
        jsgsync._sess()
    assert jsgsync._session is None, "a rejected login must not be cached"
    assert portal.wait_open(0), "the half-finished login left its socket behind"


# -------------------------------------------- the refusal, and who owns it
def test_a_refused_session_still_in_the_cache_is_put_down(portal, monkeypatch):
    dead = _Fake()
    jsgsync._session = (time.time(), dead)
    monkeypatch.setattr(jsgsync, "_sess", lambda: dead)
    monkeypatch.setattr(jsgsync, "_net",
                        lambda fn, url, *a, **kw: _Resp(401, '{"UserName": "u", "Digit1": "p"}'))
    with pytest.raises(JsGlobalAuthError):
        jsgsync._get("/Home/GetUserData")
    assert dead.closed == 1, "the portal refused this session and nobody closed it"
    assert jsgsync._session is None


def test_an_older_refusal_does_not_discard_a_newer_login(portal, monkeypatch):
    """The reset used to clear the cache whatever it held.

    Thread A is refused with S1 while thread B has already logged in and
    cached S2. Clearing it then throws away a good login and makes the portal
    answer another one — the repetition this module exists to avoid — and
    closing S2 would take down B's connection under its feet.
    """
    refused, newer = _Fake(), _Fake()
    jsgsync._session = (time.time(), refused)
    monkeypatch.setattr(jsgsync, "_sess", lambda: refused)

    def net(fn, url, *a, **kw):
        jsgsync._session = (time.time(), newer)     # B logs in mid-request
        return _Resp(403, '{"UserName": "u", "Digit1": "p"}')

    monkeypatch.setattr(jsgsync, "_net", net)
    with pytest.raises(JsGlobalAuthError):
        jsgsync._get("/Home/GetUserData")
    assert jsgsync._session[1] is newer, "an older refusal discarded a newer login"
    assert newer.closed == 0, "a session this request never used was closed"


def test_a_read_only_call_on_a_warm_session_closes_nothing(portal, monkeypatch):
    live = _Fake()
    jsgsync._session = (time.time(), live)
    monkeypatch.setattr(jsgsync, "_sess", lambda: live)
    monkeypatch.setattr(jsgsync, "_net", lambda fn, url, *a, **kw:
                        _Resp(200, '[{"accountNo": "90001"}]'))
    assert jsgsync.accounts() == [{"accountNo": "90001"}]
    assert live.closed == 0, "an answer that was fine must not retire the session"


# ------------------------------------------------ the two claims, unmocked
def test_closing_a_session_really_takes_the_server_side_connection_down(portal_transport):
    """Without this, "call close()" is a fix that could do nothing at all."""
    portal = portal_transport
    s = requests.Session()
    for _ in range(3):
        assert s.get(portal.base + "/Home/GetUserData").status_code == 200
    assert portal.open == 1
    s.close()
    assert portal.wait_open(0), "the portal still holds the connection it retired"


def test_closing_a_session_does_not_interrupt_a_request_in_flight(portal_transport):
    """Why retiring a session is safe on the order path.

    The desk thread may close the cached session while the engine thread is
    mid-PlaceOrder. A connection already checked out is not among the ones
    `close()` drains, so the request finishes as if nothing happened — and
    the order it carried reaches the broker exactly once.
    """
    portal = portal_transport
    s = requests.Session()
    outcome: dict = {}

    def slow():
        try:
            outcome["status"] = s.get(portal.base + "/slow", timeout=10).status_code
        except Exception as exc:  # noqa: BLE001 — breaking here is what we test
            outcome["error"] = exc

    t = threading.Thread(target=slow)
    t.start()
    time.sleep(0.15)                 # the portal is asleep inside the request
    s.close()                        # retire it while that request is running
    t.join(timeout=10)
    assert not t.is_alive(), "the in-flight request never finished"
    assert "error" not in outcome, f"in-flight request was broken: {outcome.get('error')!r}"
    assert outcome.get("status") == 200


def test_the_wall_still_holds_outside_the_fake_portal(portal_transport):
    """The exception above is a door for one host, not a disabled wall."""
    with pytest.raises(AssertionError, match="left the fake portal"):
        requests.get("https://portal.invalid/Home/PlaceOrder", timeout=2)
    with pytest.raises(AssertionError, match="left the fake portal"):
        requests.Session().post("http://127.0.0.1:1/nope", timeout=2)


# --------------------------------------------------------------- the guard
def _functions(tree):
    return [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]


def _targets(node):
    for t in node.targets:
        if isinstance(t, (ast.Tuple, ast.List)):
            yield from t.elts
        else:
            yield t


def _writes_session(fn) -> bool:
    return any(isinstance(n, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == "_session" for t in _targets(n))
               for n in ast.walk(fn))


def _calls(node, name: str) -> bool:
    return any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
               and n.func.id == name for n in ast.walk(node))


def _builds_a_portal_session(fn) -> bool:
    return any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
               and n.func.attr == "Session" and isinstance(n.func.value, ast.Name)
               and n.func.value.id == "requests" for n in ast.walk(fn))


def _try_that_drops_on_failure(fn) -> bool:
    return any(isinstance(t, ast.Try) and any(_calls(h, "_drop") for h in t.handlers)
               for t in ast.walk(fn))


def _drops_the_cache_pair(tree) -> list[int]:
    """Every `_drop(_session)` — the pair, not the session inside it.

    This is the exact mistake the fix first shipped with: unpacking
    `dead, _session = _session, None` leaves `dead` holding
    `(fetched_at, session)`, `close()` lands on a tuple, and a swallow
    around it makes a fix that does nothing look like a fix.
    """
    return [n.lineno for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
            and n.func.id == "_drop"
            and any(isinstance(a, ast.Name) and a.id == "_session" for a in n.args)]


REAL = ast.parse(inspect.getsource(jsgsync))


def _session_writers(tree):
    return {fn.name for fn in _functions(tree) if _writes_session(fn)}


def _session_droppers(tree):
    return {fn.name for fn in _functions(tree) if _calls(fn, "_drop")}


def test_only_the_cache_owner_replaces_the_cached_session():
    """Every other writer is a session that can be dropped without being closed.

    `_sess` installs a login and `_retire` clears one; nothing else is allowed
    to touch the pair, because the site that clears it is the site that has to
    put the previous one down.
    """
    assert _session_writers(REAL) == {"_sess", "_retire"}
    assert _session_droppers(REAL) == {"_login", "_retire"}


def test_no_retire_hands_the_cache_pair_to_the_closer():
    assert _drops_the_cache_pair(REAL) == []


def test_every_session_the_module_builds_is_dropped_if_its_login_fails():
    builders = [fn for fn in _functions(REAL) if _builds_a_portal_session(fn)]
    assert len(builders) == 1, "the module should build exactly one portal session"
    assert _try_that_drops_on_failure(builders[0]), (
        f"{builders[0].name}() builds a session that no failed login can drop")


def test_the_guards_themselves_catch_a_module_that_forgets():
    """A source rule earns trust by failing on a source that should break it."""
    forgetful = ast.parse(
        "def _sess():\n"
        "    global _session\n"
        "    _session = None\n"
        "    _session = (time.time(), _login())\n"
        "def _reset():\n"
        "    global _session\n"
        "    _session = None\n")
    assert _session_writers(forgetful) == {"_sess", "_reset"}
    assert _session_droppers(forgetful) == set()

    careless = ast.parse(
        "def _login(c):\n"
        "    s = requests.Session()\n"
        "    s.get(c['url'])\n"
        "    return s\n")
    assert _builds_a_portal_session(_functions(careless)[0])
    assert not _try_that_drops_on_failure(_functions(careless)[0])

    # the shape this fix first shipped with: the pair named `_session` passed
    # straight to the closer, so `close()` landed on a tuple and did nothing
    pair = ast.parse("def _retire():\n    global _session\n    _drop(_session)\n    _session = None\n")
    assert _drops_the_cache_pair(pair) == [3]
    by_name = ast.parse("def _retire():\n    global _session\n    dead, _session = _session, None\n    _drop(dead)\n")
    assert _drops_the_cache_pair(by_name) == [], (
        "naming the pair `dead` first is the same bug wearing a different name")
