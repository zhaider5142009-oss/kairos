"""Local PSX client circuit breaker — no network.

A dead localhost service used to cost a full socket timeout per PSX symbol, so
a 20-symbol scan spent ~100 s dialling a port nobody answers on.
"""

from __future__ import annotations

import time

import pytest

from kairos import data as D


class _Boom(Exception):
    pass


@pytest.fixture
def dead_client(monkeypatch):
    c = D.LocalPsxClient()
    c.enabled = True
    c._down_until = 0.0

    def fake_get(url, timeout=6, retries=1):
        monkeypatch.setitem(_COUNT, "n", _COUNT["n"] + 1)
        monkeypatch.setitem(_COUNT, "timeout", timeout)
        raise _Boom("connection refused")

    _COUNT = {"n": 0, "timeout": None}
    monkeypatch.setattr(D, "_http_get", fake_get)
    c._count = _COUNT
    return c


def test_dead_host_is_only_dialled_once(dead_client):
    for t in ("OGDC", "PPL", "HUBC", "ENGRO"):
        assert dead_client.history(t) is None
    assert dead_client._count["n"] == 1
    assert dead_client.alive() is False
    assert dead_client._count["n"] == 1


def test_breaker_opens_then_retries(dead_client, monkeypatch):
    assert dead_client.history("OGDC") is None
    assert time.time() < dead_client._down_until
    # when the window expires the symbol is fetchable again
    dead_client._down_until = time.time() - 1
    assert dead_client.history("OGDC") is None
    assert dead_client._count["n"] == 2


def test_probe_timeout_is_short(dead_client):
    dead_client.history("OGDC")
    assert dead_client._count["timeout"] <= 3


def test_success_keeps_the_source_open(monkeypatch):
    c = D.LocalPsxClient()
    c.enabled = True
    c._down_until = 0.0
    monkeypatch.setattr(D, "_http_get",
                        lambda url, timeout=6, retries=1: '{"data": []}')
    assert c.history("OGDC") is None          # empty payload, not a dead host
    assert c._down_until == 0.0
