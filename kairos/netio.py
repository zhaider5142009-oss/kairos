"""The stdlib HTTP error contract: a failed request is an open file.

`urllib.request.urlopen` raises `HTTPError` for a non-2xx answer, and the
exception *is* the response — the body and the connection it arrived on are
still attached to it. A `with` block never puts that down, because the raise
happens before the response is ever handed to the block. So a handler that
records the error and moves on leaves a live TLS socket waiting for the
garbage collector. Measured on the running desk: `ResourceWarning: unclosed
<ssl.SSLSocket [closed] fd=1964>`, with the allocation traceback landing on
`data.py`'s news fetch and its retry loop — three sockets per call on a box
that is meant to hold none. `close_error()` is the one line every such
handler owes.

`HttpStatus` is the other half of the same contract. A fetch that failed
because a provider *answered* is not a fetch that failed because the wire was
dead, and the difference is the status code. A backoff built to pause half an
hour on a 429 cannot act on a message that only says "GET failed".
`status_of()` is how a handler that caught a broad exception reads that code
without assuming which exception class it caught.
"""

from __future__ import annotations

import urllib.error

# The two 4xx that mean "ask again". Any other 4xx will answer the same way on
# the next attempt, so retrying it buys a slower failure and more load on a
# provider that has already said no.
ASK_AGAIN = frozenset({408, 429})


class HttpStatus(ConnectionError):
    """A provider answered with a status the caller could not use.

    Subclasses `ConnectionError` because every caller of this codebase's
    transport already treats an unanswered fetch that way; the point of the
    subclass is to carry `code` on top of it, not to change who catches it.
    """

    def __init__(self, code: int, url: str, reason: str = "") -> None:
        super().__init__(f"HTTP {code} {reason} from {url}".strip())
        self.code = int(code)
        self.url = url
        self.reason = reason


class PayloadTooLarge(ConnectionError):
    """The host answered, but with more bytes than the caller will hold.

    Distinct from a dead wire and from a provider's refusal: the host is fine
    and will answer the same way again, so the request must not be retried —
    retrying it downloads the same oversized body twice.
    """

    def __init__(self, url: str, limit: int) -> None:
        super().__init__(f"body from {url} exceeds {limit} bytes")
        self.url = url
        self.limit = int(limit)


def close_error(exc: BaseException) -> None:
    """Put down the response an HTTP error is still holding. Anywhere else, does nothing."""
    if not isinstance(exc, urllib.error.HTTPError):
        return
    try:
        exc.close()
    except Exception:  # noqa: BLE001 — the peer may have closed it already; that is not a new failure
        pass


def status_of(exc: BaseException) -> int:
    """The status a failed request carries, or 0 when no provider answered with one.

    The code is the only part of a failed fetch that says what to do next — 401
    rotate a key, 402 fund the account, 429 wait, 500 retry — and a handler that
    keeps only `str(exc)` throws it away. A message cannot stand in for it: the
    digits in a provider's prose ("you requested 4296 tokens") are
    indistinguishable from a status code by substring, so acting on them means
    obeying a number that never described the request.

    0 is not a status. It is the answer to "did anything answer?", which is a
    different question from any code and must stay distinguishable from one.
    """
    if isinstance(exc, (urllib.error.HTTPError, HttpStatus)):
        try:
            return int(exc.code)
        except (TypeError, ValueError):  # a transport that printed no usable code
            return 0
    return 0


def retryable(exc: BaseException) -> bool:
    """Is this failure worth another request?"""
    if isinstance(exc, PayloadTooLarge):
        return False
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code >= 500 or exc.code in ASK_AGAIN
    return True
