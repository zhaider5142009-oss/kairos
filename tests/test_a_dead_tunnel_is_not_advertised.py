"""A hostname nobody is checking is not a public address.

`/api/health` reported `public_url` straight out of `data/tunnel_url.txt`, the
file `tools/keepalive.ps1` writes when a quick tunnel comes up. Nothing in that
read asked whether the keeper was still alive: the keeper died overnight, the
file kept the hostname from the last successful registration, and the desk and
the UI went on offering a link that no longer resolved. The tunnel was "down"
for eighteen hours while every surface said the opposite, which is the exact
failure mode the tunnel keeper exists to prevent.

The keeper now re-stamps the file on every successful probe, and health only
advertises a URL that carries a fresh stamp. Empty is the honest answer; the UI
already renders it as "tunnel down".
"""

from __future__ import annotations

import os
import time

import kairos.health as H

URL = "https://quiet-river-example.trycloudflare.com"


def _stamp(tmp_path, monkeypatch, text=URL, age_s=0.0) -> None:
    monkeypatch.setattr(H.CFG, "data_dir", tmp_path)
    p = tmp_path / "tunnel_url.txt"
    p.write_text(text, encoding="utf-8")
    if age_s:
        then = time.time() - age_s
        os.utime(p, (then, then))


def test_a_fresh_stamp_is_the_public_address(tmp_path, monkeypatch):
    _stamp(tmp_path, monkeypatch, age_s=30)
    assert H._public_url() == URL


def test_a_stale_stamp_is_no_address_at_all(tmp_path, monkeypatch):
    """The keeper stopped hours ago: the file is evidence, not a promise."""
    _stamp(tmp_path, monkeypatch, age_s=H._TUNNEL_TTL_S + 600)
    assert H._public_url() == ""


def test_no_file_is_no_address(tmp_path, monkeypatch):
    monkeypatch.setattr(H.CFG, "data_dir", tmp_path)
    assert H._public_url() == ""


def test_an_empty_file_is_no_address(tmp_path, monkeypatch):
    """Written at keeper start and again before a heal: blank means not up."""
    _stamp(tmp_path, monkeypatch, text="", age_s=1)
    assert H._public_url() == ""
