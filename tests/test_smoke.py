"""Offline smoke tests — no network, no heavy/optional deps required.

These run in CI to guarantee the package imports, the universe and strategy
registries are populated, and config loads with sensible defaults. They are
deliberately cheap: they never hit Yahoo, OpenRouter, MT5 or WSL.
"""

from __future__ import annotations

import kairos
from kairos import strategies, universe
from kairos.config import CFG


def test_version_is_string():
    assert isinstance(kairos.__version__, str)
    assert kairos.__version__.count(".") >= 1


def test_universe_is_populated():
    assert len(universe.ALL) == 100
    assert len(universe.CRYPTO) == 22


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
