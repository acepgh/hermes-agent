"""Tests that per-thread read connections are reclaimed when their thread dies.

Under WAL, recall/browse queries run on a per-thread read-only connection
(``SessionDB._get_read_conn``) cached in ``threading.local``.  Those
connections were also registered in a strong ``set`` so that short-lived
reader threads' connections could not be GC'd without ``close()`` — a
GC'd-but-unclosed connection never decrements
``sqlite_safe_read._live_connections``, which would leave the byte-probe
guard armed for the life of the process.

Nothing removed entries from that set except ``SessionDB.close()``.  In a
process that never closes its SessionDB — a gateway runs for weeks — every
reader thread that ever ran pinned one connection and roughly two fds, until
the process hit EMFILE and stopped serving.  A live gateway was found holding
47 ``state.db`` + 45 ``state.db-wal`` descriptors against a 256-fd limit.

The registry is now keyed by a weak reference to the owning thread, and
``_reap_dead_read_conns()`` releases connections whose thread has exited.
Read connections also opt out of ``check_same_thread`` so the reaper can
actually close them: ``close()`` is what releases the fd *and* decrements the
byte-probe tracker, and it raises ``ProgrammingError`` from any other thread
under the default.
"""

import gc
import os
import threading

import pytest

from hermes_state import SessionDB


def _open_fds() -> int:
    """Real open descriptors for this process (portable enough for macOS/Linux)."""
    try:
        return len(os.listdir("/dev/fd"))
    except OSError:  # pragma: no cover — non-POSIX
        pytest.skip("/dev/fd unavailable")


@pytest.fixture()
def db(tmp_path):
    d = SessionDB(tmp_path / "state.db")
    if not d._wal_active:
        d.close()
        pytest.skip("read-path split is WAL-only")
    yield d
    d.close()


def _read_from_new_thread(db):
    t = threading.Thread(target=lambda: db.list_sessions_rich(limit=1))
    t.start()
    t.join()


class TestDeadThreadReaping:
    def test_registry_does_not_grow_with_sequential_threads(self, db):
        """Threads that come and go must not accumulate connections."""
        for _ in range(40):
            _read_from_new_thread(db)
        gc.collect()
        db._reap_dead_read_conns()
        # At most the connection belonging to a thread still being torn down.
        assert len(db._read_conns) <= 1

    def test_fds_bounded_by_concurrency_not_total_threads(self, db):
        """The regression: fd usage tracked threads-ever-created, unbounded.

        Bursts of 8 concurrent readers, repeated — the shape a gateway
        produces.  Descriptor count must plateau at roughly peak concurrency
        rather than climbing with the 160 threads that ran in total.
        """
        _read_from_new_thread(db)
        gc.collect()
        db._reap_dead_read_conns()
        baseline = _open_fds()

        for _ in range(20):
            ts = [
                threading.Thread(target=lambda: db.list_sessions_rich(limit=1))
                for _ in range(8)
            ]
            for t in ts:
                t.start()
            for t in ts:
                t.join()
            del ts
        gc.collect()
        db._reap_dead_read_conns()

        # Pre-fix this grew by ~2 fds for each of the 160 threads.  Allow
        # generous headroom for the 8-wide burst still settling.
        assert _open_fds() - baseline < 40
        assert len(db._read_conns) <= 8

    def test_live_thread_connection_is_not_reaped(self, db):
        """The reaper must never close a connection still in use."""
        opened = threading.Event()
        release = threading.Event()
        result = {}

        def worker():
            db.list_sessions_rich(limit=1)  # opens this thread's connection
            opened.set()
            release.wait(timeout=10)
            # Must still work: the reaper ran while this thread was alive.
            result["rows"] = db.list_sessions_rich(limit=1)

        t = threading.Thread(target=worker)
        t.start()
        assert opened.wait(timeout=10)
        db._reap_dead_read_conns()  # sweep while the owner is alive
        assert len(db._read_conns) == 1
        release.set()
        t.join(timeout=10)
        assert "rows" in result  # no ProgrammingError / closed-connection error

    def test_close_drains_connections_from_dead_threads(self, db, tmp_path):
        """close() must release fds *and* untrack, even cross-thread.

        The old close() called conn.close() bare inside `except Exception:
        pass`; for a dead thread's connection that raises ProgrammingError,
        so it silently skipped both the fd release and the untrack.
        """
        from hermes_cli.sqlite_safe_read import has_live_connection

        for _ in range(5):
            _read_from_new_thread(db)
        db.close()
        gc.collect()
        assert len(db._read_conns) == 0
        assert not has_live_connection(tmp_path / "state.db")
