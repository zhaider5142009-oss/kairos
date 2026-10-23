"""The file that holds a human's real order intent must not be allowed to lie.

`data/jsg_pending.json` is the staging queue between "KAIROS wants to place
this" and "the broker has it". Five defects sat on it:

1. `_load()` answered `[]` for a queue it could not read. Every write here
   rebuilds the whole file from that read, so one truncated write was enough:
   the next stage or drop replaced the queue with a list holding none of the
   tickets that survived in it, and the desk printed "no staged orders" the
   whole time.
2. `_save()` used a plain `write_text`, which truncates the destination before
   rewriting it — the crash that produced the unreadable file in the first
   place. It now goes through `store.atomic_write`.
3. Nothing locked the read-modify-write, so the desk and an armed autopilot
   could each rebuild the file from a list that no longer described it.
4. `ack()` removed its ticket by rewriting the list read *before* the broker
   POST, so anything staged while the broker was busy was put back by the
   overwrite — and if that overwrite died, the "may or may not have reached
   the broker" record was never written because the write came first.
5. TTL expiry deleted tickets with no journalled event at all, so the ack that
   later said "no pending order" described a stage that had simply stopped
   existing.

The last two tests are the same class one level over: `Autopilot._age_days`
read an unreadable timestamp as zero days old, which froze watchlist rotation
and made a held entry look permanently not-due-yet.
"""

from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timedelta

import pytest

from kairos import jsgsync, jsgtrade
from kairos.autopilot import Autopilot
from kairos.execution import AuditJournal
from kairos.jsgsync import JsGlobalError

CAPS = [{"symbol": "OGDC", "market": "REG", "upperCap": 340.0, "lowerLock": 290.0}]


class FakeResponse:
    def __init__(self, text: str = "12345", status_code: int = 200,
                 payload=None) -> None:
        self.text, self.status_code = text, status_code
        self._payload = payload

    def json(self):
        if self._payload is not None:
            return self._payload
        return json.loads(self.text)


class FakeSession:
    def __init__(self, caps=None, post_text: str = "12345") -> None:
        self.posts: list[tuple[str, dict]] = []
        self.caps = caps or []
        self.post_text = post_text

    def get(self, url: str, params=None, timeout=None) -> FakeResponse:
        if url.endswith("/Home/GetUpperLowerCap"):
            return FakeResponse("", 200, self.caps)
        return FakeResponse(json.dumps([{"accountNo": "90001",
                                         "accountName": "TEST ACCOUNT"}]))

    def post(self, url: str, data=None, timeout=None) -> FakeResponse:
        self.posts.append((url, dict(data or {})))
        return FakeResponse(self.post_text)


class Gate:
    def engaged(self) -> bool:
        return False


def _conf(**over):
    base = dict(url="https://portal.invalid", login="tester",
                password="hunter2-test", portal_url="https://portal.invalid",
                allow_orders=True, trading_pin="1234", order_account="90001",
                market="REG", max_order_notional=250_000.0)
    base.update(over)
    return base


@pytest.fixture()
def q(tmp_path, monkeypatch):
    """A temp queue and journal; the broker is a fake and never called."""
    jrn = AuditJournal(path=tmp_path / "audit.jsonl")
    monkeypatch.setattr(jsgsync, "_conf", lambda: _conf())
    monkeypatch.setattr(jsgtrade, "_conf", lambda: _conf())
    monkeypatch.setattr(jsgtrade, "_sess", lambda: FakeSession(CAPS))
    monkeypatch.setattr(jsgtrade, "PENDING_PATH", tmp_path / "pending.json")
    monkeypatch.setattr(jsgtrade, "JOURNAL", jrn)
    monkeypatch.setattr(jsgtrade, "KILL", Gate())
    monkeypatch.setattr(jsgtrade, "_caps", None)
    monkeypatch.setattr(jsgtrade, "_caps_retry_at", 0.0)
    monkeypatch.setattr(jsgtrade, "_last_queue_error", None)
    return tmp_path, jrn


def events(jrn: AuditJournal) -> list[dict]:
    return jrn.tail(200)


def corrupt(path):
    """What a crash mid-write used to leave behind: half of a JSON array."""
    raw = json.dumps([{"id": "deadbeef", "t": time.time(), "symbol": "ENGRO",
                       "side": "BUY", "qty": 40, "price": 120.0}])
    path.write_text(raw[: len(raw) // 2], encoding="utf-8")
    return raw[: len(raw) // 2]


# ------------------------------------------------- a queue that cannot be read
def test_a_damaged_queue_is_reported_instead_of_showing_empty(q):
    """The empty list used to be indistinguishable from a queue with nothing in it."""
    tmp_path, _ = q
    jsgtrade.stage("OGDC", "BUY", 5, 300.0)
    assert jsgtrade.queue_health()["ok"] and len(jsgtrade.pending()) == 1
    damaged = corrupt(tmp_path / "pending.json")
    assert jsgtrade.pending() == []
    health = jsgtrade.queue_health()
    assert health["ok"] is False and "JSON" in health["error"]
    # the bytes are left where they are so a human can still salvage the ticket
    assert (tmp_path / "pending.json").read_text(encoding="utf-8") == damaged


def test_staging_refuses_instead_of_rebuilding_from_an_empty_read(q):
    """The defect: [] + the new ticket, written over a file holding real tickets."""
    tmp_path, jrn = q
    corrupt(tmp_path / "pending.json")
    with pytest.raises(JsGlobalError, match="written over it"):
        jsgtrade.stage("OGDC", "BUY", 5, 300.0)
    assert "JSG_STAGED" not in [e["event"] for e in events(jrn)]
    assert (tmp_path / "pending.json").read_text(encoding="utf-8").startswith("[{")


def test_a_drop_cannot_be_asked_of_an_unreadable_queue(q):
    tmp_path, _ = q
    corrupt(tmp_path / "pending.json")
    with pytest.raises(JsGlobalError, match="written over it"):
        jsgtrade.drop("deadbeef")


def test_the_state_endpoint_carries_the_queues_own_health(q):
    tmp_path, _ = q
    assert jsgtrade.state()["queue"]["ok"] is True
    corrupt(tmp_path / "pending.json")
    st = jsgtrade.state()
    assert st["queue"]["ok"] is False and st["queue"]["path"].endswith("pending.json")
    assert st["pending"] == 0     # and the number is still a number, not a lie


# ------------------------------------------------------------------- writing
def test_the_queue_is_only_ever_replaced_by_a_rename(q, monkeypatch):
    """A whole-file write is what leaves a half-file behind after a crash."""
    tmp_path, _ = q
    seen: list[str] = []
    monkeypatch.setattr(jsgtrade, "atomic_write", lambda p, t: seen.append(t))
    jsgtrade.stage("OGDC", "BUY", 5, 300.0)
    assert len(seen) == 1 and json.loads(seen[0])[0]["symbol"] == "OGDC"
    assert not (tmp_path / "pending.json").exists()


def test_a_write_that_does_not_land_is_not_a_staged_order(q, monkeypatch):
    tmp_path, jrn = q

    def full(path, text):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(jsgtrade, "atomic_write", full)
    with pytest.raises(JsGlobalError, match="could not be written"):
        jsgtrade.stage("OGDC", "BUY", 5, 300.0)
    assert "JSG_STAGED" not in [e["event"] for e in events(jrn)]


# ------------------------------------------------------------------- expiry
def test_an_expired_ticket_is_written_down_as_it_leaves(q):
    """It was a real order request; the trail has to say when it stopped being one."""
    tmp_path, jrn = q
    row = jsgtrade.stage("OGDC", "BUY", 5, 300.0)
    rows = jsgtrade._load()
    rows[0]["t"] = time.time() - jsgtrade.TTL_S - 1
    jsgtrade._save(rows)
    assert jsgtrade.pending() == []
    gone = [e for e in events(jrn) if e["event"] == "JSG_TICKET_EXPIRED"]
    assert len(gone) == 1
    assert gone[0]["id"] == row["id"] and gone[0]["symbol"] == "OGDC"
    assert gone[0]["age_s"] > jsgtrade.TTL_S
    assert json.loads((tmp_path / "pending.json").read_text(encoding="utf-8")) == []


def test_a_ticket_with_no_usable_time_expires_loudly(q):
    """A row that cannot be aged cannot be kept either — but it must not vanish quietly."""
    tmp_path, jrn = q
    jsgtrade._save([{"id": "a1b2c3d4", "symbol": "OGDC", "side": "BUY",
                     "qty": 5, "price": 300.0, "t": None, "ts": "not a date"}])
    assert jsgtrade.pending() == []
    gone = [e for e in events(jrn) if e["event"] == "JSG_TICKET_EXPIRED"]
    assert len(gone) == 1 and gone[0]["age_s"] is None


def test_a_ticket_aged_by_its_iso_stamp_is_not_expired(q):
    """The `ts` the desk prints is the fallback when `t` is missing."""
    tmp_path, _ = q
    recent = (datetime.now() - timedelta(minutes=3)).isoformat(timespec="seconds")
    jsgtrade._save([{"id": "ffff0000", "symbol": "OGDC", "side": "BUY",
                     "qty": 5, "price": 300.0, "ts": recent}])
    assert [r["id"] for r in jsgtrade.pending()] == ["ffff0000"]


# ------------------------------------------------------------- concurrent use
def test_tickets_staged_at_the_same_time_all_survive(q):
    """Two unguarded rebuilds of the same file lose one of them."""
    tmp_path, _ = q
    gate = threading.Barrier(6)
    ids: list[str] = []
    lock = threading.Lock()

    def worker(i: int) -> None:
        gate.wait()
        row = jsgtrade.stage("OGDC", "BUY", 5 + i, 300.0)
        with lock:
            ids.append(row["id"])

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(ids) == sorted(r["id"] for r in jsgtrade._load())


def test_an_order_staged_during_the_broker_post_survives_the_ack(q, monkeypatch):
    """ack() used to rewrite the queue from the list it read before the POST."""
    tmp_path, _ = q
    real_post = jsgtrade._post_order
    row = jsgtrade.stage("OGDC", "BUY", 5, 300.0)

    def post_that_stages(c, r, pin):
        jsgtrade.stage("ENGRO", "BUY", 7, 300.0)   # arrives while the broker is busy
        return real_post(c, r, pin)

    monkeypatch.setattr(jsgtrade, "_post_order", post_that_stages)
    out = jsgtrade.ack(row["id"], pin="1234")
    assert out["ok"]
    left = jsgtrade.pending()
    assert len(left) == 1 and left[0]["symbol"] == "ENGRO"


def test_a_lost_post_is_journalled_before_the_queue_write(q, monkeypatch):
    """The record of an order with an unknown outcome must not depend on a disk write."""
    tmp_path, jrn = q
    row = jsgtrade.stage("OGDC", "BUY", 5, 300.0)

    def die(c, r, pin):
        raise JsGlobalError("portal did not answer")

    def full(path, text):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(jsgtrade, "_post_order", die)
    monkeypatch.setattr(jsgtrade, "atomic_write", full)
    with pytest.raises(JsGlobalError, match="do not ack it again") as exc:
        jsgtrade.ack(row["id"], pin="1234")
    assert "may or may not have reached" in str(exc.value)
    assert "JSG_UNCONFIRMED" in [e["event"] for e in events(jrn)]


def test_an_order_that_went_out_reports_a_queue_it_could_not_clear(q, monkeypatch):
    tmp_path, jrn = q
    real_write = jsgtrade.atomic_write
    writes = {"n": 0}

    def flaky(path, text):
        writes["n"] += 1
        if writes["n"] > 1:
            raise OSError(28, "No space left on device")
        real_write(path, text)

    monkeypatch.setattr(jsgtrade, "atomic_write", flaky)
    row = jsgtrade.stage("OGDC", "BUY", 5, 300.0)
    out = jsgtrade.ack(row["id"], pin="1234")
    assert out["ok"] and "queue_warning" in out
    assert "JSG_QUEUE_STUCK" in [e["event"] for e in events(jrn)]


# --------------------------------------------------- the same class, one over
class _Cfg:
    """The live config, with the autopilot section pinned for the test."""

    def __init__(self, data):
        self._data = data
        from kairos.config import CFG as real
        self._real = real

    def section(self, name):
        return self._data.get(name, self._real.section(name))

    def __getattr__(self, item):
        return getattr(self._real, item)


@pytest.fixture()
def ap(tmp_path, monkeypatch):
    from kairos import autopilot as AP
    monkeypatch.setattr(AP, "CFG", _Cfg({"autopilot": {"psx_auto_watch": True,
                                                       "auto_place": False}}))
    monkeypatch.setattr(AP, "JOURNAL", AuditJournal(path=tmp_path / "audit.jsonl"))
    a = Autopilot()
    a.state_path = tmp_path / "autopilot.json"
    a._save()
    a.st["horizon_days"] = 3.0
    a.st["exit_threshold"] = 0.9
    return a, AP


def test_an_unreadable_age_has_no_number_invented(ap):
    a, _ = ap
    assert a._age_days("2026-10-20T10:00:00") > 0
    for bad in ("", "not a date", None, 12345, "2026-13-45"):
        assert a._age_days(bad) is None, bad


def test_a_held_entry_with_no_readable_start_keeps_its_other_exits(ap):
    """Zero days old meant the maturity exit was off, forever, quietly."""
    a, AP = ap
    hb = {"entry": 100.0, "started": "garbage"}
    hp = {"side": "BUY", "entry": 100.0, "qty": 10}

    class _I:
        key = "OGDC"

    assert a._manage_exit(_I(), hp, hb, 100.0, 0.0) is None
    assert hb["age_warned"] is True
    assert [x["action"] for x in a.st["actions"]][0] == "AGE UNREADABLE"
    assert "AUTOPILOT_AGE_UNREADABLE" in [e["event"] for e in AP.JOURNAL.tail(50)]
    # the warning is for the row, not for every cycle that reads it
    before = len(a.st["actions"])
    a._manage_exit(_I(), hp, hb, 100.0, 0.0)
    assert len(a.st["actions"]) == before
    # and a readable start date still matures on schedule
    hb2 = {"entry": 100.0,
           "started": (datetime.now() - timedelta(days=10)).isoformat()}
    assert a._manage_exit(_I(), hp, hb2, 100.0, 0.0) == "MATURE"


def test_an_unreadable_rotation_stamp_does_not_freeze_the_watchlist(ap, monkeypatch):
    """Zero days old also meant "rotated recently" — which is never again."""
    a, AP = ap
    calls = {"n": 0}

    class _Scan:
        def scan(self, top=15):
            calls["n"] += 1
            return {"ok": False}

    import kairos.engine as EN
    monkeypatch.setattr(EN, "ENGINE", _Scan())
    a.st["last_rotation"] = "not a date"
    a._maybe_rotate_psx()
    assert calls["n"] == 1
    a.st["last_rotation"] = datetime.now().isoformat()
    a._maybe_rotate_psx()
    assert calls["n"] == 1          # a real recent stamp still throttles it
