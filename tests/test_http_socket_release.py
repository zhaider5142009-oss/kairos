"""A refused HTTP request is an open file, and this engine used to drop it on the floor.

Measured on the running desk, in `data/serve.log`: seven
`ResourceWarning: unclosed <ssl.SSLSocket [closed] fd=988..1964>` hits, with the
allocation traceback landing on `data.py`'s news fetch through `_http_get`. The
reason is a detail of the standard library that no test had ever accounted for:
`urlopen` raises `HTTPError` *as the response* — the body and the socket it
arrived on stay attached to the exception — so the `with` block that closes
every success never sees a failure. A handler that records the error and moves
on leaves the descriptor until the garbage collector gets round to it, and
`_http_get`'s retry loop did it three times per call, from several threads.

Two things came out of reading those handlers. The first is the socket. The
second is that `_http_get` re-raised every failure as a plain `ConnectionError`
and dropped the status code — and the news service is built to pause GDELT for
half an hour when it answers 429. Measured from this machine, every GDELT
request *is* a 429, so the penalty box engaged exactly once in the test suite
(where the injected fake transport raised the raw `HTTPError` the engine never
produces) and never in production.
"""

from __future__ import annotations

import ast
import pathlib
import time
import urllib.error
import urllib.request

import pytest

from kairos import binance as B
from kairos import data as D
from kairos import news as N
from kairos.netio import HttpStatus, close_error

URL = "https://vendor.invalid/api"
CLOSE = "close_error"


class _Socket:
    """Stands in for the `http.client.HTTPResponse` an HTTPError carries."""

    def __init__(self, body: bytes = b'{"error": "no"}') -> None:
        self.closed = False
        self._body = body

    def read(self, n: int = -1) -> bytes:
        return self._body

    def close(self) -> None:
        self.closed = True

    # what `tempfile._TemporaryFileWrapper` asks of the object it wraps
    def flush(self) -> None: ...
    def writable(self) -> bool: return False
    def readable(self) -> bool: return True
    def seekable(self) -> bool: return False


class _Resp:
    """A success, which the `with` block already handles."""

    def __init__(self, body: bytes = b'[{"d": [1.0]}]') -> None:
        self.closed = False
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> bool:
        self.closed = True
        return False


def _failing(code: int, reason: str = "refused"):
    """A fake `urlopen` that raises `code`, plus every socket it handed out."""
    sockets: list[_Socket] = []
    calls: list[str] = []

    def _urlopen(req, *a, **kw):
        calls.append(getattr(req, "full_url", str(req)))
        sock = _Socket()
        sockets.append(sock)
        raise urllib.error.HTTPError(URL, code, reason, {}, sock)

    return _urlopen, calls, sockets


# --------------------------------------------------------------------------
# 1. the socket
# --------------------------------------------------------------------------
def test_a_refused_get_puts_its_socket_down(monkeypatch):
    urlopen, calls, sockets = _failing(429, "Too Many Requests")
    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    with pytest.raises(ConnectionError):
        D._http_get(URL, retries=0)
    assert len(calls) == 1
    assert sockets[0].closed, "the error response is the socket's only owner"


def test_every_attempt_of_a_retried_refusal_is_closed(monkeypatch):
    urlopen, calls, sockets = _failing(503, "Service Unavailable")
    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    with pytest.raises(ConnectionError):
        D._http_get(URL, retries=1)
    assert len(calls) == 2
    assert all(s.closed for s in sockets), "two attempts, two descriptors"


def test_the_post_door_closes_too(monkeypatch):
    urlopen, calls, sockets = _failing(403, "Forbidden")
    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    with pytest.raises(urllib.error.HTTPError):
        D._http_post_json(URL, {"q": 1})
    assert calls and sockets[0].closed


def test_a_venue_that_reads_the_body_before_raising_still_closes(monkeypatch, tmp_path):
    """Binance's handler consumes the error body to report why — then owes the socket."""
    urlopen, calls, sockets = _failing(429, "Too Many Requests")
    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(B, "_conf", lambda: {"api_key": "", "secret": "",
                                             "base_url": "https://binance.invalid"})
    c = B.BinanceClient()
    with pytest.raises(B.BinanceError):
        c._send("/api/v3/ticker/price")
    assert calls and sockets[0].closed
    assert c.error and "429" in c.error


def test_close_error_ignores_anything_that_is_not_a_response():
    for exc in (ValueError("x"), ConnectionError("y"), HttpStatus(429, URL), OSError("z")):
        close_error(exc)                      # must not raise, must not touch anything


# --------------------------------------------------------------------------
# 2. the status code, which is the answer
# --------------------------------------------------------------------------
@pytest.mark.parametrize("code", [404, 429, 500])
def test_the_status_the_provider_answered_survives_the_transport(monkeypatch, code):
    urlopen, _, _ = _failing(code)
    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    with pytest.raises(HttpStatus) as got:
        D._http_get(URL, retries=0)
    assert got.value.code == code


def test_a_provider_answer_is_still_caught_as_a_connection_failure(monkeypatch):
    """Every existing caller catches `ConnectionError`; the new type must not escape them."""
    urlopen, _, _ = _failing(500, "Boom")
    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    caught = None
    try:
        D._http_get(URL, retries=0)
    except ConnectionError as exc:
        caught = exc
    assert isinstance(caught, ConnectionError)


def test_a_dead_wire_is_still_reported_as_a_dead_wire(monkeypatch):
    def _urlopen(req, *a, **kw):
        raise urllib.error.URLError("name or service not known")

    monkeypatch.setattr(urllib.request, "urlopen", _urlopen)
    with pytest.raises(ConnectionError) as got:
        D._http_get(URL, retries=0)
    assert not isinstance(got.value, HttpStatus)
    assert "GET failed" in str(got.value)


def test_a_404_is_not_asked_again(monkeypatch):
    """Retrying an answer that will not change is a slower failure and more load."""
    urlopen, calls, sockets = _failing(404, "Not Found")
    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    start = time.time()
    with pytest.raises(HttpStatus):
        D._http_get(URL, retries=2)
    assert len(calls) == 1
    assert time.time() - start < 0.5, "the retry sleeps are not spent on a 404"
    assert sockets[0].closed


@pytest.mark.parametrize("code", [408, 429, 500, 503])
def test_a_come_back_later_is_asked_again(monkeypatch, code):
    urlopen, calls, sockets = _failing(code)
    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    with pytest.raises(HttpStatus):
        D._http_get(URL, retries=1)
    assert len(calls) == 2
    assert all(s.closed for s in sockets)


def test_a_success_is_untouched(monkeypatch):
    resp = _Resp(b'{"chart": 1}')
    monkeypatch.setattr(urllib.request, "urlopen", lambda req, *a, **kw: resp)
    assert D._http_get(URL, retries=0) == b'{"chart": 1}'
    assert resp.closed


# --------------------------------------------------------------------------
# 3. the backoff that was wired to a transport the engine never uses
# --------------------------------------------------------------------------
def test_a_rate_limited_gdelt_pause_survives_the_real_transport(monkeypatch):
    """The tone service and `_http_get`, not a fake transport built to pass."""
    urlopen, calls, sockets = _failing(429, "Too Many Requests")
    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    g = N.GdeltTone()                        # no `get=`: production transport
    assert g.tone("Engro Corporation") is None
    assert g.blocked == 1, "a 429 that is not counted is a rate limit nobody saw"
    assert g._down_until > time.time()
    g.tone("Engro Corporation")
    assert len(calls) == 1, "the penalty box must stop the next request, not log it"
    assert all(s.closed for s in sockets)


def test_an_unrelated_gdelt_answer_does_not_engage_the_pause(monkeypatch):
    urlopen, calls, sockets = _failing(404, "Not Found")
    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    g = N.GdeltTone()
    assert g.tone("Nothing") is None
    assert g.blocked == 0 and g._down_until == 0.0
    g.tone("Nothing")
    assert len(calls) == 2                   # a missing page is not a rate limit
    assert all(s.closed for s in sockets)


# --------------------------------------------------------------------------
# 4. the shape of the whole codebase, so the next handler cannot forget
# --------------------------------------------------------------------------
_PKG = pathlib.Path(D.__file__).parent
# Anything that can bind an HTTPError: the error itself, or a class above it.
_CATCHES_ERROR_RESPONSE = {"HTTPError", "URLError", "OSError", "Exception", "BaseException"}


def _handler_names(h: ast.ExceptHandler) -> set[str]:
    """The exception type names a handler lists, flattened out of any tuple."""
    if h.type is None:
        return {*_CATCHES_ERROR_RESPONSE}       # bare `except:` binds everything
    node = h.type
    if isinstance(node, ast.Tuple):
        out: set[str] = set()
        for el in node.elts:
            out |= _handler_names(ast.ExceptHandler(type=el, name=None, body=[]))
        return out
    if isinstance(node, ast.Attribute):
        return {node.attr}
    if isinstance(node, ast.Name):
        return {node.id}
    return set()


def _calls_urlopen(body: list[ast.stmt]) -> bool:
    for stmt in body:
        for sub in ast.walk(stmt):
            if isinstance(sub, ast.Call):
                f = sub.func
                if isinstance(f, ast.Attribute) and f.attr == "urlopen":
                    return True
    return False


def _closes_directly(node: ast.AST) -> bool:
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call):
            f = sub.func
            if isinstance(f, ast.Name) and f.id == CLOSE:
                return True
            if isinstance(f, ast.Attribute) and f.attr == CLOSE:
                return True
    return False


def _closers(tree: ast.Module) -> set[str]:
    """Module-level functions in this file that put a response down.

    A handler that hands its exception to one builder is not a handler that
    forgets to close — but the delegation only counts because the function
    called really does close, so this is computed from the file rather than
    written down as an allowlist. Resolved to a fixpoint so a builder that
    delegates in turn still counts, and a handler that calls a helper closing
    nothing stays an offender.
    """
    defs = [n for n in tree.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    known = {d.name for d in defs if _closes_directly(d)}
    grew = True
    while grew:
        grew = False
        for d in defs:
            if d.name in known:
                continue
            if any(isinstance(s, ast.Call) and isinstance(s.func, ast.Name)
                   and s.func.id in known for s in ast.walk(d)):
                known.add(d.name)
                grew = True
    return known


def _puts_it_down(h: ast.ExceptHandler, closers: set[str] = frozenset()) -> bool:
    return _closes_directly(h) or any(
        isinstance(s, ast.Call) and isinstance(s.func, ast.Name)
        and s.func.id in closers for s in ast.walk(h))


def _error_response_handlers(tree: ast.Module):
    """Every handler in a module that can be handed an open HTTP response."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Try):
            continue
        earlier_http_error = False
        for h in node.handlers:
            names = _handler_names(h)
            capable = bool(names & _CATCHES_ERROR_RESPONSE)
            # a handler after one that already caught HTTPError sees none
            if capable and not earlier_http_error:
                yield node, h
            if names == {"HTTPError"}:
                earlier_http_error = True


def test_every_handler_of_a_network_call_in_the_engine_puts_its_socket_down():
    offenders = []
    for py in sorted(_PKG.glob("*.py")):
        tree = ast.parse(py.read_text(encoding="utf-8"))
        closers = _closers(tree)
        for node, h in _error_response_handlers(tree):
            if _calls_urlopen(node.body) and not _puts_it_down(h, closers):
                offenders.append(f"{py.name}:{h.lineno}")
    # Handlers that catch an HTTPError by name anywhere else are the same debt:
    # the object they bind is an open response whatever produced it.
    for py in sorted(_PKG.glob("*.py")):
        tree = ast.parse(py.read_text(encoding="utf-8"))
        closers = _closers(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Try):
                continue
            for h in node.handlers:
                if "HTTPError" in _handler_names(h) and not _calls_urlopen(node.body) \
                        and not _puts_it_down(h, closers):
                    offenders.append(f"{py.name}:{h.lineno}")
    assert not offenders, f"open responses left to the GC at {offenders}"


def test_a_handler_that_delegates_to_a_helper_that_closes_nothing_still_offends():
    """The delegation rule must be earned by the helper, not by its name."""
    src = ('import urllib.request\n'
           'def _looks_like_a_builder(e):\n'
           '    return str(e)\n'
           'def f(u):\n'
           '    try:\n'
           '        with urllib.request.urlopen(u) as r:\n'
           '            return r.read()\n'
           '    except Exception as e:\n'
           '        return _looks_like_a_builder(e)\n')
    tree = ast.parse(src)
    found = [(t, h) for t, h in _error_response_handlers(tree)
             if _calls_urlopen(t.body) and not _puts_it_down(h, _closers(tree))]
    assert len(found) == 1
    # and the same handler passes once the helper it calls really puts it down
    src2 = src.replace('    return str(e)\n', '    close_error(e)\n    return str(e)\n')
    tree2 = ast.parse(src2)
    assert all(_puts_it_down(h, _closers(tree2))
               for t, h in _error_response_handlers(tree2) if _calls_urlopen(t.body))


def test_the_guard_itself_catches_a_handler_that_forgets():
    """A scan that finds nothing on any input is not a guard."""
    src = ('import urllib.request\n'
           'def f(u):\n'
           '    try:\n'
           '        with urllib.request.urlopen(u) as r:\n'
           '            return r.read()\n'
           '    except Exception:\n'
           '        return None\n')
    found = [(t, h) for t, h in _error_response_handlers(ast.parse(src))
             if not _puts_it_down(h)]
    assert len(found) == 1
    assert _calls_urlopen(found[0][0].body)
