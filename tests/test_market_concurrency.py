from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime
from threading import Event

import polars as pl
import pytest

from vnpy.config.runtime_config import ParquetConfig
from vnpy.datafeed.data_market_module import (
    clear_market_data, get_market_store, load_market_snapshot,
    register_market_module, unregister_market_module,
)
from vnpy.datafeed.data_market_store import MarketDataRequest, MarketDataStore
from vnpy.datafeed.data_parquet_feed import ParquetDataFeed
from vnpy.event.engine import ModuleEngine
from vnpy.event.event import EngineEvent, EventType
from vnpy.event.module_node import ModuleStoppedError


@pytest.fixture(autouse=True)
def synthetic_source(monkeypatch):
    # All I/O is controlled by Events or a synthetic frame in these tests.
    monkeypatch.setattr(ParquetDataFeed, '_resolve_day_bar_dir', staticmethod(lambda root: root))


def setting():
    return ParquetConfig(root='.', start='2026-01-01', end='2026-01-02')


def frame(close=10.0):
    return pl.DataFrame({'datetime': [datetime(2026, 1, 1)],
                         'vt_symbol': ['SHSE.600000'], 'close': [close]})


def post_load(engine):
    reply = Future()
    assert engine.post_event('market', EngineEvent(EventType.MARKET_LOAD_REQUEST, {
        'request': MarketDataRequest.from_config(setting()), 'reply': reply,
    }))
    return reply


def block_loader(monkeypatch):
    entered, release = Event(), Event()
    def load(self, **kwargs):
        entered.set()
        assert release.wait(5), 'test did not release loader'
        return frame(20.0)
    monkeypatch.setattr(ParquetDataFeed, 'load_frame', load)
    return entered, release


def test_refresh_does_not_block_readers_and_keeps_old_version(monkeypatch):
    store = MarketDataStore()
    request = MarketDataRequest.from_config(setting())
    monkeypatch.setattr(ParquetDataFeed, 'load_frame', lambda self, **kw: frame())
    original = store.load(request).snapshot
    entered, release = block_loader(monkeypatch)
    with ThreadPoolExecutor(2) as pool:
        refreshing = pool.submit(store.load, request, reload=True)
        try:
            assert entered.wait(2)
            reading = pool.submit(store.snapshot)
            old = reading.result(timeout=1)
            assert old.snapshot_id == original.snapshot_id
            old.frame.replace_column(2, pl.Series('close', [99.0]))
            assert store.snapshot().frame['close'][0] == 10.0
        finally:
            release.set()
        new = refreshing.result(timeout=2).snapshot
    assert new.frame['close'][0] == 20.0
    assert original.frame['close'][0] == 10.0
    assert new.snapshot_id != original.snapshot_id


def test_failed_refresh_keeps_published_snapshot(monkeypatch):
    store = MarketDataStore()
    request = MarketDataRequest.from_config(setting())
    monkeypatch.setattr(ParquetDataFeed, 'load_frame', lambda self, **kw: frame())
    original = store.load(request).snapshot
    def fail(self, **kwargs):
        raise OSError('source unavailable')
    monkeypatch.setattr(ParquetDataFeed, 'load_frame', fail)
    with pytest.raises(OSError, match='source unavailable'):
        store.load(request, reload=True)
    assert store.snapshot().snapshot_id == original.snapshot_id


def test_concurrent_same_request_reads_source_once(monkeypatch):
    store = MarketDataStore()
    request = MarketDataRequest.from_config(setting())
    calls = []
    def load(self, **kwargs):
        calls.append(1)
        return frame()
    monkeypatch.setattr(ParquetDataFeed, 'load_frame', load)
    with ThreadPoolExecutor(8) as pool:
        results = list(pool.map(lambda _: store.load(request), range(16)))
    assert len(calls) == 1
    assert sum(not r.reused for r in results) == 1
    assert len({r.snapshot.snapshot_id for r in results}) == 1


def test_clear_serializes_with_load_without_resurrecting_snapshot(monkeypatch):
    store = MarketDataStore()
    request = MarketDataRequest.from_config(setting())
    entered, release = block_loader(monkeypatch)
    clearing_started = Event()
    def clear():
        clearing_started.set()
        store.clear()
    with ThreadPoolExecutor(2) as pool:
        loading = pool.submit(store.load, request)
        try:
            assert entered.wait(2)
            clearing = pool.submit(clear)
            assert clearing_started.wait(2)
            assert not clearing.done()
        finally:
            release.set()
        loaded = loading.result(timeout=2)
        clearing.result(timeout=2)
    with pytest.raises(RuntimeError, match='not been loaded'):
        store.snapshot()
    assert loaded.snapshot.frame['close'][0] == 20.0


def test_generic_stop_fails_queued_request_and_supports_clean_restart(monkeypatch):
    engine = ModuleEngine()
    register_market_module(engine)
    node = engine.get_module('market')
    entered, release = block_loader(monkeypatch)
    first = post_load(engine)
    try:
        assert entered.wait(2)
        second = post_load(engine)
        with ThreadPoolExecutor(1) as pool:
            stopping = pool.submit(engine.stop_module, 'market')
            try:
                with pytest.raises(ModuleStoppedError):
                    second.result(timeout=1)
                assert not engine.post_event('market', EngineEvent(EventType.TIMER, {}))
                assert not engine.start_module('market')
            finally:
                release.set()
            assert stopping.result(timeout=2)
        assert first.result(timeout=1).snapshot.frame['close'][0] == 20.0
        assert node._queue.unfinished_tasks == 0
        assert node.queue_size() == 0
        assert load_market_snapshot(engine, setting(), timeout=1).reused
    finally:
        release.set()
        unregister_market_module(engine)


def test_unregister_fails_pending_requests_and_clears_store(monkeypatch):
    engine = ModuleEngine()
    store = register_market_module(engine)
    entered, release = block_loader(monkeypatch)
    first = post_load(engine)
    try:
        assert entered.wait(2)
        second = post_load(engine)
        with ThreadPoolExecutor(1) as pool:
            closing = pool.submit(unregister_market_module, engine)
            try:
                with pytest.raises(ModuleStoppedError):
                    second.result(timeout=1)
            finally:
                release.set()
            closing.result(timeout=2)
        assert first.result(timeout=1).snapshot.frame.height == 1
        assert not engine.module_exists('market')
        with pytest.raises(RuntimeError, match='not been loaded'):
            store.snapshot()
    finally:
        release.set()
        unregister_market_module(engine)


def test_timed_out_queued_clear_is_not_executed(monkeypatch):
    engine = ModuleEngine()
    register_market_module(engine)
    entered, release = block_loader(monkeypatch)
    first = post_load(engine)
    try:
        assert entered.wait(2)
        with pytest.raises(TimeoutError, match='cancelled before execution'):
            clear_market_data(engine, timeout=0.02)
        release.set()
        first.result(timeout=2)
        # A later request acts as a barrier: the cancelled clear has been dequeued.
        assert load_market_snapshot(engine, setting(), timeout=1).reused
        assert engine.get_module('market').last_error is None
        assert get_market_store(engine).snapshot().frame['close'][0] == 20.0
    finally:
        release.set()
        unregister_market_module(engine)


def test_timed_out_running_load_can_complete_without_future_state_error(monkeypatch):
    engine = ModuleEngine()
    entered, release = block_loader(monkeypatch)
    with ThreadPoolExecutor(1) as pool:
        loading = pool.submit(load_market_snapshot, engine, setting(), timeout=0.1)
        try:
            assert entered.wait(2)
            with pytest.raises(TimeoutError, match='already running'):
                loading.result(timeout=1)
            release.set()
            assert load_market_snapshot(engine, setting(), timeout=2).reused
            assert engine.get_module('market').last_error is None
        finally:
            release.set()
            unregister_market_module(engine)


def test_unregister_timeout_does_not_clear_live_context(monkeypatch):
    engine = ModuleEngine()
    store = register_market_module(engine)
    node = engine.get_module('market')
    entered, release = block_loader(monkeypatch)
    first = post_load(engine)
    original_stop = node.stop
    monkeypatch.setattr(node, 'stop', lambda **kw: original_stop(timeout=0.02, **kw))
    try:
        assert entered.wait(2)
        with pytest.raises(RuntimeError, match='still stopping'):
            unregister_market_module(engine)
        assert engine.get_module('market') is node
        assert get_market_store(engine) is store
        assert not engine.start_module('market')
        release.set()
        first.result(timeout=2)
        node._thread.join(timeout=2)
        unregister_market_module(engine)
        assert not engine.module_exists('market')
    finally:
        release.set()
        unregister_market_module(engine)


@pytest.mark.parametrize('timeout', [None, True, 0, -1, float('inf'), float('nan')])
def test_invalid_timeout_does_not_register_module(timeout):
    engine = ModuleEngine()
    with pytest.raises(ValueError, match='positive finite'):
        load_market_snapshot(engine, setting(), timeout=timeout)
    assert not engine.module_exists('market')


def test_worker_cannot_wait_for_its_own_queue():
    engine = ModuleEngine()
    answer = Future()
    def entry(ctx, event):
        try:
            load_market_snapshot(engine, setting(), timeout=1)
        except Exception as exc:
            answer.set_exception(exc)
    assert engine.register_module('market', entry)
    assert engine.start_module('market')
    try:
        assert engine.post_event('market', EngineEvent(EventType.TIMER, {}))
        with pytest.raises(RuntimeError, match='own queue'):
            answer.result(timeout=1)
    finally:
        assert engine.unregister_module('market')


def test_business_timeout_is_not_rewritten(monkeypatch):
    engine = ModuleEngine()
    def load(self, **kwargs):
        raise TimeoutError('source timeout')
    monkeypatch.setattr(ParquetDataFeed, 'load_frame', load)
    try:
        with pytest.raises(TimeoutError, match='^source timeout$'):
            load_market_snapshot(engine, setting(), timeout=1)
    finally:
        unregister_market_module(engine)


def test_stop_during_worker_final_cleanup_leaves_no_wakeup_in_queue(monkeypatch):
    engine = ModuleEngine()
    assert engine.register_module('probe', lambda ctx, event: None)
    node = engine.get_module('probe')
    cleaning, release = Event(), Event()
    original_reject = node._reject_pending
    def reject(events):
        if node.is_worker_thread():
            cleaning.set()
            assert release.wait(5)
        original_reject(events)
    monkeypatch.setattr(node, '_reject_pending', reject)
    assert engine.start_module('probe')
    try:
        assert engine.post_event('probe', EngineEvent(EventType.STOP, {}))
        assert cleaning.wait(2)
        # A stop call now enqueues a wakeup after the worker's final queue drain.
        assert not node.stop(timeout=0.02)
        release.set()
        assert node.stop(timeout=2)
        assert node.queue_size() == 0
        assert node._queue.unfinished_tasks == 0
        assert engine.start_module('probe')
    finally:
        release.set()
        assert engine.unregister_module('probe')


def test_stop_rejects_requests_even_before_worker_starts():
    engine = ModuleEngine()
    assert engine.register_module('market', lambda ctx, event: None)
    reply = post_load(engine)
    assert engine.stop_module('market')
    with pytest.raises(ModuleStoppedError):
        reply.result(timeout=1)
    node = engine.get_module('market')
    assert node._queue.unfinished_tasks == 0
    assert engine.unregister_module('market')
