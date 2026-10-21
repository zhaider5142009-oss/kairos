"""The two files that *are* the track record must survive an interrupted write.

Both the prediction journal and the paper book are rewritten whole rather than
appended to. They used to be overwritten in place, so a crash mid-write left a
truncated file — and every loader below them read an unreadable file as "no
state yet". For the paper book that means the next cycle opens at full capital
with an empty position list and zero realized P&L: a reset that looks exactly
like a healthy account, and whose equity is the number live sizing divides by.
"""

import json

import pytest

import kairos.execution as E
from kairos.config import CFG
from kairos.store import atomic_write


@pytest.fixture()
def data(monkeypatch, tmp_path):
    """Give this test its own data directory and its own kill switch.

    PaperBroker resolves its path from CFG at construction, so patching the
    config is enough for the book. The kill switch is a module-level object
    built at import, so its path has to be patched directly — otherwise the
    test would engage the operator's real one.
    """
    monkeypatch.setattr(CFG, "data_dir", tmp_path)
    monkeypatch.setattr(E.KILL, "path", tmp_path / "KILL_SWITCH")
    return tmp_path


def _events(monkeypatch) -> list:
    seen: list[tuple[str, dict]] = []
    monkeypatch.setattr(E.JOURNAL, "log",
                        lambda event, payload=None: seen.append((event, payload or {})))
    return seen


# ---------------------------------------------------------------- atomic_write
def test_a_failed_write_leaves_the_previous_file_intact(tmp_path, monkeypatch):
    p = tmp_path / "journal.jsonl"
    atomic_write(p, '{"a": 1}\n')
    monkeypatch.setattr(type(p), "write_text",
                        lambda self, *a, **k: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(OSError):
        atomic_write(p, "GONE\n")
    assert p.read_text(encoding="utf-8") == '{"a": 1}\n'
    assert not [f for f in tmp_path.iterdir() if f.name.endswith(".tmp")]


def test_no_temp_file_survives_a_good_write(tmp_path):
    p = tmp_path / "book.json"
    atomic_write(p, "{}")
    assert p.read_text(encoding="utf-8") == "{}"
    assert [f.name for f in tmp_path.iterdir()] == ["book.json"]


def test_two_writers_of_one_file_never_leave_a_half_written_file(tmp_path):
    """The bar store now fills several symbols at once, so the same file can be
    written from two threads. A shared temp name let one writer truncate the
    other's payload mid-write, and Windows then refused the rename outright —
    a lost bar file that no caller would have noticed. Each writer gets its own
    temp, the replace retries through the momentary lock, and no writer is
    allowed to fail quietly."""
    import threading

    p = tmp_path / "GOLD.csv"
    long_a, short_b = "a" * 200_000, "b" * 20
    for _ in range(6):
        errors: list[BaseException] = []

        def write(payload: str) -> None:
            try:
                atomic_write(p, payload)
            except BaseException as exc:          # noqa: BLE001 — the point of the test
                errors.append(exc)

        threads = [threading.Thread(target=write, args=(v,))
                   for v in (long_a, short_b, long_a, short_b)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors, f"a writer failed: {errors[0]!r}"
        body = p.read_text(encoding="utf-8")
        assert len(body) in (len(long_a), len(short_b)), "a mixed payload landed"
        assert set(body) <= {"a"} or set(body) <= {"b"}
    assert not [f for f in tmp_path.iterdir() if f.name.endswith(".tmp")]


# -------------------------------------------------------------------- the book
def test_a_healthy_book_round_trips(data):
    first = E.PaperBroker(capital=1_000_000.0)
    first.cash, first.realized_pnl = 912_345.5, -4_200.25
    first.positions = {"GOLD": {"qty": 3.0, "entry": 4100.0}}
    first._save()

    second = E.PaperBroker(capital=1_000_000.0)
    assert second.cash == 912_345.5
    assert second.realized_pnl == -4_200.25
    assert second.positions["GOLD"]["qty"] == 3.0


def test_a_missing_book_is_a_clean_first_run(data):
    broker = E.PaperBroker(capital=250_000.0)
    assert broker.cash == 250_000.0
    assert not E.KILL.engaged(), "no file is not damage, it is only a first run"


def test_a_torn_book_is_not_read_as_a_fresh_account(data, monkeypatch):
    (data / "paper_account.json").write_text(
        '{"capital": 1000000.0, "cash": 912345.5', encoding="utf-8")
    events = _events(monkeypatch)

    E.PaperBroker(capital=1_000_000.0)

    assert [e[0] for e in events] == ["PAPER_STATE_DAMAGED", "KILL_SWITCH_ENGAGED"]
    assert E.KILL.engaged(), "never size an order against a book it cannot read"


def test_the_damaged_book_is_kept_not_deleted(data, monkeypatch):
    original = '{"capital": 1000000.0, "cash": 912345.5, "positions": {"GOLD": 3}'
    (data / "paper_account.json").write_text(original, encoding="utf-8")
    _events(monkeypatch)

    E.PaperBroker(capital=1_000_000.0)

    damaged = [f for f in data.iterdir() if f.name.endswith(".damaged")]
    assert len(damaged) == 1
    assert damaged[0].read_text(encoding="utf-8") == original


def test_a_recovered_operator_can_release_and_reload(data, monkeypatch):
    (data / "paper_account.json").write_text("{ not json", encoding="utf-8")
    _events(monkeypatch)
    E.PaperBroker(capital=1_000_000.0)
    assert E.KILL.engaged()

    E.KILL.release()
    good = {"capital": 1_000_000.0, "cash": 800_000.0, "positions": {},
            "fills": [], "realized_pnl": 12.5, "day_pnl": 0.0, "day_stamp": ""}
    atomic_write(data / "paper_account.json", json.dumps(good))
    broker = E.PaperBroker(capital=1_000_000.0)
    assert broker.cash == 800_000.0


# ---------------------------------------------------------------- daily roll
def test_the_day_roll_is_persisted_not_only_held_in_memory(data):
    """The book rolls its daily figure when the date changes, but used to keep
    the roll in memory and write the file only on a fill. On a quiet stretch —
    which is most of this account — the ledger on disk kept the previous day's
    stamp and the previous day's loss, and the track record published those.

    A daily loss limit is a brake; a stale stamp means nobody can tell from the
    file which day the number belongs to.
    """
    from datetime import datetime

    stale = {"capital": 1_000_000.0, "cash": 990_000.0, "positions": {},
             "fills": [], "realized_pnl": -10_000.0,
             "day_pnl": -9_033.0, "day_stamp": "2000-01-01"}
    atomic_write(data / "paper_account.json", json.dumps(stale))

    broker = E.PaperBroker(capital=1_000_000.0)
    assert broker.account()["day_pnl"] == 0.0

    on_disk = json.loads((data / "paper_account.json").read_text(encoding="utf-8"))
    assert on_disk["day_stamp"] == datetime.now().strftime("%Y-%m-%d")
    assert on_disk["day_pnl"] == 0.0
    assert on_disk["realized_pnl"] == -10_000.0     # the roll must not lose history


def test_reading_the_book_does_not_rewrite_it_every_time(data):
    from datetime import datetime

    today = datetime.now().strftime("%Y-%m-%d")
    atomic_write(data / "paper_account.json", json.dumps(
        {"capital": 1_000_000.0, "cash": 1_000_000.0, "positions": {},
         "fills": [], "realized_pnl": 0.0, "day_pnl": 0.0, "day_stamp": today}))
    broker = E.PaperBroker(capital=1_000_000.0)
    path = data / "paper_account.json"
    before = path.stat().st_mtime_ns
    for _ in range(50):
        broker.account()
    assert path.stat().st_mtime_ns == before        # the desk polls this constantly
