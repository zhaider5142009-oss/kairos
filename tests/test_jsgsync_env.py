"""Environment overrides let a headless deploy drive the JS Global bridge.

No network: _conf() is pure config resolution.
"""

from __future__ import annotations

import pytest

from kairos import jsgsync
from kairos.jsgsync import JsGlobalAuthError

BASE = {"portal_url": "https://cfg.example", "login": "cfglogin",
        "password": "cfgpw"}


@pytest.fixture(autouse=True)
def fake_config(monkeypatch):
    monkeypatch.setattr(jsgsync.CFG, "section", lambda name: dict(BASE))
    for k in ("JSG_PORTAL", "JSG_LOGIN", "JSG_PASSWORD", "JSG_PIN",
              "JSG_ALLOW_ORDERS", "JSG_ORDER_ACCOUNT"):
        monkeypatch.delenv(k, raising=False)


def test_defaults_are_fail_closed():
    c = jsgsync._conf()
    assert c["url"] == "https://cfg.example" and c["login"] == "cfglogin"
    assert c["allow_orders"] is False and c["trading_pin"] == ""
    assert c["market"] == "REG" and c["max_order_notional"] == 250_000.0


def test_env_secrets_override_config(monkeypatch):
    monkeypatch.setenv("JSG_PORTAL", "https://env.example/")
    monkeypatch.setenv("JSG_LOGIN", "envlogin")
    monkeypatch.setenv("JSG_PASSWORD", "envpw")
    monkeypatch.setenv("JSG_PIN", "9999")
    monkeypatch.setenv("JSG_ALLOW_ORDERS", "true")
    monkeypatch.setenv("JSG_ORDER_ACCOUNT", "90001")
    c = jsgsync._conf()
    assert c["url"] == "https://env.example"          # trailing slash stripped
    assert c["login"] == "envlogin" and c["password"] == "envpw"
    assert c["trading_pin"] == "9999" and c["allow_orders"] is True
    assert c["order_account"] == "90001"


@pytest.mark.parametrize("raw,want", [("1", True), ("on", True), ("yes", True),
                                      ("0", False), ("off", False), ("", False)])
def test_allow_orders_accepts_env_strings(monkeypatch, raw, want):
    monkeypatch.setenv("JSG_ALLOW_ORDERS", raw)
    assert jsgsync._conf()["allow_orders"] is want


def test_missing_credentials_fail_closed(monkeypatch):
    monkeypatch.setattr(jsgsync.CFG, "section", lambda name: {})
    with pytest.raises(JsGlobalAuthError):
        jsgsync._conf()
