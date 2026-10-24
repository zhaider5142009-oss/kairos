"""The LEAN seam must export bars truthfully and never imply a run.

Docker is installed on this desk while its engine is not running, so "docker"
found on PATH is exactly the kind of evidence that used to be mistaken for a
capability. These tests pin the distinction: an export is a data artifact with
a digest, `status()` names each missing piece, and nothing here returns a
backtest result it did not compute.
"""

from __future__ import annotations

import hashlib
import json

import pandas as pd
import pytest

from kairos import lean as LEAN


def bars(*rows):
    return pd.DataFrame([{"date": d, "open": o, "high": h, "low": low,
                          "close": c, "volume": v}
                         for d, o, h, low, c, v in rows])


GOOD = bars(("2026-10-20", 10.0, 11.0, 9.5, 10.5, 1000),
            ("2026-10-21", 10.5, 12.0, 10.4, 11.8, 1200),
            ("2026-10-22", 11.8, 12.2, 11.0, 11.2, 900))


# ------------------------------------------------------------------ status
def test_nothing_installed_is_not_ready_and_says_what_is_missing():
    st = LEAN.status(which=lambda name: None, probe=lambda d: pytest.fail(
        "the probe must not run without a docker client"))
    assert st["ready"] is False
    assert "the lean CLI" in st["missing"]
    assert "the docker client" in st["missing"]


def test_a_docker_binary_with_a_silent_engine_is_not_ready():
    st = LEAN.status(which=lambda name: f"C:/x/{name}.exe",
                     probe=lambda d: (None, False))
    assert st["ready"] is False
    assert "a running docker engine" in st["missing"]
    assert "the docker client" not in st["missing"]


def test_an_engine_without_the_image_is_named_as_such():
    st = LEAN.status(which=lambda name: f"C:/x/{name}.exe",
                     probe=lambda d: ("27.1.1", False))
    assert st["ready"] is False
    assert any(LEAN.LEAN_IMAGE in gap for gap in st["missing"])


def test_engine_and_image_together_are_ready():
    st = LEAN.status(which=lambda name: f"C:/x/{name}.exe",
                     probe=lambda d: ("27.1.1", True))
    assert st["ready"] is True
    assert st["missing"] == []


# ------------------------------------------------------------------ export
def test_the_csv_carries_the_header_and_every_bar(tmp_path):
    out = LEAN.export_bars("GOLD", data_dir=tmp_path, bars=GOOD)
    text = (tmp_path / "GOLD.csv").read_text(encoding="utf-8")
    lines = text.strip().splitlines()
    assert lines[0] == "date,open,high,low,close,volume"
    assert len(lines) == 1 + len(GOOD)
    assert lines[1].startswith("2026-10-20,10,11,9.5,10.5,1000")
    assert out["rows"] == 3


def test_an_unsorted_frame_is_written_in_date_order(tmp_path):
    shuffled = GOOD.iloc[[2, 0, 1]]
    LEAN.export_bars("GOLD", data_dir=tmp_path, bars=shuffled)
    text = (tmp_path / "GOLD.csv").read_text(encoding="utf-8")
    positions = [text.index(day) for day in ("2026-10-20", "2026-10-21",
                                             "2026-10-22")]
    assert positions == sorted(positions)


def test_two_rows_for_one_date_are_refused_by_name(tmp_path):
    doubled = pd.concat([GOOD, GOOD.iloc[[1]]], ignore_index=True)
    with pytest.raises(LEAN.LeanError, match="repeat 2026-10-21"):
        LEAN.export_bars("GOLD", data_dir=tmp_path, bars=doubled)
    assert not (tmp_path / "GOLD.csv").exists()


def test_a_non_finite_close_is_refused_with_its_date(tmp_path):
    broken = GOOD.copy()
    broken.loc[1, "close"] = float("nan")
    with pytest.raises(LEAN.LeanError, match="non-finite close on 2026-10-21"):
        LEAN.export_bars("GOLD", data_dir=tmp_path, bars=broken)


def test_a_missing_column_is_named(tmp_path):
    thin = GOOD.drop(columns=["volume"])
    with pytest.raises(LEAN.LeanError, match="volume"):
        LEAN.export_bars("GOLD", data_dir=tmp_path, bars=thin)


def test_no_bars_is_refused_rather_than_exported_as_empty(tmp_path):
    with pytest.raises(LEAN.LeanError, match="nothing to export"):
        LEAN.export_bars("GOLD", data_dir=tmp_path, bars=GOOD.iloc[0:0])


def test_an_unknown_instrument_is_refused(tmp_path):
    with pytest.raises(LEAN.LeanError, match="unknown instrument"):
        LEAN.export_bars("NOT_A_TICKER", data_dir=tmp_path, bars=GOOD)


# ---------------------------------------------------------------- manifest
def test_the_manifest_digest_is_the_sha256_of_the_written_file(tmp_path):
    out = LEAN.export_bars("GOLD", data_dir=tmp_path, bars=GOOD)
    raw = (tmp_path / "GOLD.csv").read_bytes()
    manifest = json.loads((tmp_path / "GOLD.manifest.json").read_text("utf-8"))
    assert manifest["digest_sha256"] == hashlib.sha256(raw).hexdigest()
    assert manifest["digest_sha256"] == out["digest"]
    assert manifest["rows"] == 3
    assert manifest["first_date"] == "2026-10-20"
    assert manifest["last_date"] == "2026-10-22"


def test_the_manifest_refuses_to_be_read_as_a_backtest(tmp_path):
    LEAN.export_bars("GOLD", data_dir=tmp_path, bars=GOOD)
    manifest = json.loads((tmp_path / "GOLD.manifest.json").read_text("utf-8"))
    assert "not a backtest" in manifest["note"]
    assert "digest_sha256" in manifest


def test_the_same_bars_export_to_the_same_digest(tmp_path):
    first = LEAN.export_bars("GOLD", data_dir=tmp_path, bars=GOOD)
    again = LEAN.export_bars("GOLD", data_dir=tmp_path, bars=GOOD.copy())
    assert first["digest"] == again["digest"]


def test_volume_is_written_as_delivered_including_zero(tmp_path):
    quiet = bars(("2026-10-20", 1.0, 1.0, 1.0, 1.0, 0))
    LEAN.export_bars("EURUSD", data_dir=tmp_path, bars=quiet)
    assert (tmp_path / "EURUSD.csv").read_text("utf-8").strip().endswith(",0")


# --------------------------------------------------------------------- CLI
def test_the_cli_command_prints_the_gap_and_exports(monkeypatch, tmp_path, capsys):
    from kairos import cli
    monkeypatch.setattr(LEAN, "status", lambda: {
        "ready": False, "cli": None, "docker": "C:/x/docker.exe",
        "engine": None, "image": False,
        "missing": ["a running docker engine"], "data_dir": str(tmp_path)})
    seen = {}

    def fake_export(symbol, data_dir=None, days=None, bars=None):
        seen["symbol"] = symbol
        seen["dir"] = data_dir
        return {"symbol": symbol.upper(), "rows": 3, "first": "2026-10-20",
                "last": "2026-10-22", "csv": str(data_dir / "GOLD.csv"),
                "manifest": "m", "digest": "d"}

    monkeypatch.setattr(LEAN, "export_bars", fake_export)
    code = cli.main(["lean", "--export", "GOLD", "--data-dir", str(tmp_path)])
    assert code == 0
    assert seen["symbol"] == "GOLD"
    assert str(seen["dir"]) == str(tmp_path)
    out = capsys.readouterr().out
    assert "cannot run" in out
    assert "3 bars" in out
