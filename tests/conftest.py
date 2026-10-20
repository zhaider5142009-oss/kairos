"""Test isolation for a system that trades real money.

Without this file the suite imports the operator's own config, so a test run
writes into the production journals, can read the live broker credentials, and
— because `Autopilot.start()` spawns a daemon thread the test never joins —
leaves a loop running after pytest removes the stub that was guarding it.

Three walls, all before any `kairos` import:

1. a generated config in a temp directory, so every module-level journal,
   cache and state file lands there and no live key exists to be read;
2. no network transport for the session — a broker call is a test failure,
   not a side effect;
3. every `Autopilot` built by a test is stopped and joined at teardown, so no
   thread outlives the fixtures that made it safe.
"""

from __future__ import annotations

import os
import pathlib
import tempfile
import threading

import pytest

# ---------------------------------------------------------------------------
# 1. a config of our own — must be set before kairos is imported anywhere
# ---------------------------------------------------------------------------
_TMP = pathlib.Path(tempfile.mkdtemp(prefix="kairos-test-"))
(_TMP / "config.yaml").write_text(
    "app:\n"
    "  data_dir: data\n"
    "  cache_dir: data/cache\n"
    "  reports_dir: reports\n"
    "api_server:\n"
    "  enabled: false\n"
    "  token: test-token\n"
    "autopilot:\n"
    "  auto_place: false\n"
    "js_global:\n"
    "  portal_url: https://portal.invalid\n"
    "  login: \"\"\n"
    "  password: \"\"\n"
    "  allow_orders: false\n"
    "  trading_pin: \"\"\n"
    "mt5:\n"
    "  login: \"\"\n"
    "  password: \"\"\n"
    "  server: \"\"\n",
    encoding="utf-8",
)
os.environ["KAIROS_CONFIG"] = str(_TMP / "config.yaml")
os.environ["KAIROS_TOKEN"] = "test-token"
os.environ["KAIROS_AUTH"] = "0"
for _var in ("JSG_PIN", "JSG_LOGIN", "JSG_PASSWORD", "JSG_ALLOW_ORDERS",
             "OPENROUTER_API_KEY", "MOONSHOT_API_KEY", "KIMI_API_KEY"):
    os.environ.pop(_var, None)


# ---------------------------------------------------------------------------
# 2. no transport, session-wide
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session", autouse=True)
def _no_network():
    import requests

    def _blocked(self, method, url, *args, **kwargs):
        raise AssertionError(
            f"a test reached for the network ({method} {url}); the broker and "
            "every data vendor must stay stubbed")

    real = requests.Session.request
    requests.Session.request = _blocked
    try:
        yield
    finally:
        requests.Session.request = real


@pytest.fixture(scope="session", autouse=True)
def _stop_the_module_singleton():
    from kairos.autopilot import AUTOPILOT

    try:
        yield
    finally:
        AUTOPILOT._stop.set()
        if AUTOPILOT._thread:
            AUTOPILOT._thread.join(timeout=3)


# ---------------------------------------------------------------------------
# 3. no autopilot survives its own test
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _no_orphan_loop():
    from kairos import autopilot as AP

    built: list = []
    real_init = AP.Autopilot.__init__

    def spying(self, *a, **kw):
        real_init(self, *a, **kw)
        built.append(self)

    AP.Autopilot.__init__ = spying
    try:
        yield
    finally:
        AP.Autopilot.__init__ = real_init
        for inst in built:
            inst._stop.set()
            if inst._thread:
                inst._thread.join(timeout=3)
        stragglers = [t.name for t in threading.enumerate()
                      if t.name == "kairos-autopilot" and t.is_alive()]
        assert not stragglers, (
            f"{len(stragglers)} autopilot thread(s) outlived the test that "
            "armed them — a loop running after teardown has no broker stub")
