"""Offline smoke tests — no network, no heavy/optional deps required.

These run in CI to guarantee the package imports, the universe and strategy
registries are populated, and config loads with sensible defaults. They are
deliberately cheap: they never hit Yahoo, OpenRouter, MT5 or WSL.
"""

from __future__ import annotations

import json

import pytest

import kairos
from kairos import strategies, universe
from kairos.config import CFG


@pytest.fixture
def isolated_registry(monkeypatch, tmp_path):
    """Keep registry mutations out of the live data dir and undo them after,
    so no test can leak a fake ticker into the next one."""
    snapshot = (list(universe.ALL), dict(universe._BY_KEY),
                dict(universe._BY_YAHOO), dict(universe._BY_TV))
    path = tmp_path / "reg.json"
    monkeypatch.setattr(universe, "_registered_path", lambda: path)
    yield path
    universe.ALL[:] = snapshot[0]
    for name, data in (("_BY_KEY", snapshot[1]), ("_BY_YAHOO", snapshot[2]),
                       ("_BY_TV", snapshot[3])):
        getattr(universe, name).clear()
        getattr(universe, name).update(data)


def test_version_is_string():
    assert isinstance(kairos.__version__, str)
    assert kairos.__version__.count(".") >= 1


def test_universe_is_populated():
    # the curated seed is 100; scan-driven PSX rotation can add more
    assert len(universe.ALL) >= 100
    assert len(universe.CRYPTO) == 22


def test_register_psx_is_idempotent_and_collision_safe(isolated_registry):
    n = len(universe.ALL)
    assert universe.register_psx("ZZTEST", name="Test Mill") is not None
    assert len(universe.ALL) == n + 1
    assert universe.register_psx("ZZTEST") is not None   # already PSX: reuse
    assert len(universe.ALL) == n + 1
    assert universe.resolve("ZZTEST").yahoo == "ZZTEST.KA"
    assert universe.register_psx("BTC") is None          # crypto key collision
    assert len(universe.ALL) == n + 1


def test_registered_psx_survives_a_restart(isolated_registry):
    """A rotated PSX watchlist must still resolve after the daemon restarts,
    or the desk cannot analyse or exit the paper book's PSX lines."""
    path = isolated_registry
    seed = len(universe.ALL)
    universe.register_psx("ZZPERSIST", name="Persist Mill", sector="Test")
    assert len(universe.ALL) == seed + 1
    assert path.exists() and "ZZPERSIST" in path.read_text(encoding="utf-8")

    # emulate the fresh process: forget it in memory, then restore from disk
    universe.ALL.pop()
    universe._BY_KEY.pop("ZZPERSIST", None)
    universe._BY_YAHOO.pop("ZZPERSIST.KA", None)
    universe._BY_TV.pop("PSX:ZZPERSIST", None)
    assert universe.resolve("ZZPERSIST") is None

    universe._restore_registered()
    assert universe.resolve("ZZPERSIST").market == "PSX"


def test_registered_cache_is_bounded(isolated_registry):
    for i in range(60):
        universe.register_psx(f"ZZB{i}", name=f"Bound {i}")
    assert len(json.loads(isolated_registry.read_text(encoding="utf-8"))) == 40


def test_universe_has_expected_crypto():
    keys = {i.key.upper() for i in universe.CRYPTO}
    for expected in {"BTC", "ETH", "SOL", "BNB", "XRP", "DOGE"}:
        assert expected in keys


def test_resolve_is_case_insensitive():
    assert universe.resolve("BTC").key == "BTC"
    assert universe.resolve("btc").key == "BTC"
    assert universe.resolve("nope-not-a-ticker") is None


def test_strategy_registry_populated():
    assert len(strategies.STRATEGIES) >= 100


def test_config_has_dirs():
    # config must load even without config.yaml present
    assert CFG.data_dir is not None
    assert CFG.root is not None
