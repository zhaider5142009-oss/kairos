"""Config edits bind while the process is running.

The tower stays up for days. A file that is only read at import means every
arming key, threshold and risk limit is frozen until a restart — and nothing in
the operator's experience says so. `Config.section()` now re-reads on mtime
change, so the file is the live source of truth.
"""

from __future__ import annotations

import os
import time

import pytest

from kairos.config import Config

BASE = """\
app:
  data_dir: data
autopilot:
  auto_place: false
  interval_s: 300
"""


@pytest.fixture
def cfg(tmp_path):
    f = tmp_path / "config.yaml"
    f.write_text(BASE, encoding="utf-8")
    c = Config(f)
    c._next_check = time.monotonic() - 1        # the first read is always allowed
    return c


def _touch(path, seconds: int = 10):
    st = path.stat()
    os.utime(path, (st.st_atime, st.st_mtime + seconds))


def test_edited_value_is_seen_without_a_restart(cfg):
    assert cfg.section("autopilot")["auto_place"] is False
    cfg.path.write_text(BASE.replace("auto_place: false", "auto_place: true"),
                        encoding="utf-8")
    _touch(cfg.path)
    cfg._next_check = time.monotonic() - 1
    assert cfg.section("autopilot")["auto_place"] is True


def test_unedited_file_is_not_reparsed_every_call(cfg):
    calls = []
    real = cfg._stat

    def counting():
        calls.append(1)
        return real()

    cfg._stat = counting
    cfg._next_check = time.monotonic() + 60      # window still open
    for _ in range(500):
        cfg.section("autopilot")
    assert calls == []                           # the throttle holds


def test_a_half_written_file_keeps_the_last_good_config(cfg):
    cfg.path.write_text("{{{\n  not yaml: [", encoding="utf-8")
    _touch(cfg.path, 20)
    cfg._next_check = time.monotonic() - 1
    assert cfg.section("autopilot")["auto_place"] is False
    # and the broken file is not re-read on every subsequent call
    cfg._next_check = time.monotonic() - 1
    assert cfg.section("autopilot")["auto_place"] is False


def test_deleted_config_file_does_not_crash_a_running_loop(cfg):
    cfg.path.unlink()
    cfg._next_check = time.monotonic() - 1
    assert cfg.section("autopilot")["auto_place"] is False


def test_store_secret_leaves_the_in_process_view_coherent(cfg):
    assert cfg.store_secret("autopilot", "auto_place", True)
    assert cfg.section("autopilot")["auto_place"] is True
    # the write refreshed the stamp, so the next read is not a stale re-parse
    cfg._next_check = time.monotonic() - 1
    assert cfg.section("autopilot")["auto_place"] is True


def test_app_paths_do_not_move_under_a_running_process(cfg):
    """data_dir is bound into objects at import, so it is deliberately NOT
    hot-reloaded — a kill switch that changed address mid-run would be worse
    than one that needs a restart."""
    cfg.path.write_text(BASE.replace("data_dir: data", "data_dir: elsewhere"),
                        encoding="utf-8")
    _touch(cfg.path, 30)
    cfg._next_check = time.monotonic() - 1
    cfg.section("app")
    assert cfg.data_dir.name == "data"
