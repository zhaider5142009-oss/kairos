"""Config writer: one scalar replaced without losing the file's comments."""

from __future__ import annotations

import pytest
import yaml

from kairos.config import Config

SAMPLE = """\
# operator notes: keep this file private
app:
  data_dir: data

js_global:
  portal_url: "https://broker.example"
  login: "99999"
  allow_orders: false
"""


@pytest.fixture
def cfg(tmp_path):
    f = tmp_path / "config.yaml"
    f.write_text(SAMPLE, encoding="utf-8")
    return Config(f)


def test_existing_key_is_replaced_in_place(cfg):
    assert cfg.store_secret("js_global", "allow_orders", True)
    text = cfg.path.read_text(encoding="utf-8")
    raw = yaml.safe_load(text)
    assert raw["js_global"]["allow_orders"] is True
    assert raw["js_global"]["login"] == "99999"        # untouched


def test_comments_and_other_blocks_survive(cfg):
    cfg.store_secret("js_global", "trading_pin", "1234")
    text = cfg.path.read_text(encoding="utf-8")
    assert "# operator notes" in text
    assert "app:" in text and "data_dir: data" in text


def test_new_key_lands_inside_the_block(cfg):
    cfg.store_secret("js_global", "max_order_notional", 3000)
    raw = yaml.safe_load(cfg.path.read_text(encoding="utf-8"))
    assert raw["js_global"]["max_order_notional"] == 3000
    assert "app" in raw                                # nothing spilled upward


def test_leading_zeros_are_kept(cfg):
    """A PIN like 0123 must round-trip as a string, not collapse to int 123."""
    cfg.store_secret("js_global", "trading_pin", "0123")
    raw = yaml.safe_load(cfg.path.read_text(encoding="utf-8"))
    assert raw["js_global"]["trading_pin"] == "0123"


def test_clearing_a_pin_leaves_it_falsy(cfg):
    cfg.store_secret("js_global", "trading_pin", "1234")
    cfg.store_secret("js_global", "trading_pin", "")
    assert not cfg._raw["js_global"]["trading_pin"]


def test_missing_section_is_appended(cfg):
    cfg.store_secret("autopilot", "auto_place", True)
    raw = yaml.safe_load(cfg.path.read_text(encoding="utf-8"))
    assert raw["autopilot"]["auto_place"] is True


def test_reload_is_immediate(cfg):
    cfg.store_secret("js_global", "allow_orders", True)
    assert cfg.section("js_global")["allow_orders"] is True


def test_no_config_file_means_no_write(tmp_path):
    c = Config(tmp_path / "absent.yaml")
    assert c.store_secret("js_global", "trading_pin", "1234") is False


def test_forget_pin_writes_an_empty_value(monkeypatch):
    """The desk's disarm button must store nothing, not a quoted placeholder.

    A route that wrote the two-character string '""' looked successful while
    leaving a truthy PIN on file — i.e. unattended execution stayed armed.
    """
    from kairos.api import server

    seen = []
    monkeypatch.setattr(server.CFG, "store_secret",
                        lambda s, k, v: seen.append((s, k, v)) or True)
    import kairos.execution as E

    class _NoJournal:
        @staticmethod
        def log(*a, **k):
            pass

    monkeypatch.setattr(E, "JOURNAL", _NoJournal)
    server.jsg_pin_clear()
    assert seen == [("js_global", "trading_pin", "")]
