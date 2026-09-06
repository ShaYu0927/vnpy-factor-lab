"""Whole-market daily Alpha101 calculation on a dedicated ModuleEngine queue."""
from __future__ import annotations

from datetime import date
from hashlib import sha256
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import polars as pl

from vnpy.alpha.alphas.alpha101 import Alpha101
from vnpy.alpha.logger import logger
from vnpy.datafeed.daily_store import DailyMarketStore, atomic_parquet
from vnpy.event.base_module import BaseModule, make_module_entry
from vnpy.event.event import EngineEvent, EventType


class DailyAlphaBatchModule(BaseModule):
    """READY is an explicit end-of-day barrier, not inferred from arriving bars.

    Request: trade_date, expected_symbols, excluded_symbols={symbol: reason}.
    Successful payloads contain a result path, never the full factor matrix.
    Retry the same event_id to retry delivery without recalculating in-process.
    Only the latest request is cached. A new request creates a new result file.
    """

    def handle(self, event: EngineEvent) -> None:
        if event.event_type != EventType.MARKET_DATA_READY:
            return
        logger.info("[batch/receive] module=%s request=%s date=%s source=%s",
                    self.name, event.event_id, event.get("trade_date"), event.source)
        targets = self.get_config("batch_targets", [])
        if not isinstance(targets, (list, tuple)) or not targets or any(not isinstance(t, str) or not t for t in targets):
            raise ValueError("batch_targets must be a nonempty list of module names")
        targets = tuple(dict.fromkeys(targets))
        cached = self.get_state("last_batch")
        if cached is None or cached["event_id"] != event.event_id:
            try:
                payload = self.calculate(event)
            except Exception as exc:
                logger.exception("Daily Alpha101 batch failed: event=%s", event.event_id)
                self.set_state("last_error", str(exc))
                for target in targets:
                    self.post(target, EventType.FACTOR_BATCH_FAILED,
                              {"request_id": event.event_id, "trade_date": event.get("trade_date"), "error": str(exc)})
                return
            cached = {"event_id": event.event_id, "payload": payload, "delivered": set()}
            self.set_state("last_batch", cached)
            self.set_state("last_error", None)
        else:
            logger.info("[batch/reuse] request=%s reuse saved result; delivered=%s",
                        event.event_id, sorted(cached["delivered"]))
        for target in targets:
            if target not in cached["delivered"]:
                if self.post(target, EventType.FACTOR_BATCH_READY, dict(cached["payload"])):
                    cached["delivered"].add(target)
                    logger.info("[batch/publish] request=%s target=%s event=FACTOR_BATCH_READY queued=True",
                                event.event_id, target)
                else:
                    logger.error("Batch delivery rejected: target=%s request=%s; retry same request", target, event.event_id)

    def calculate(self, event: EngineEvent) -> dict:
        started = perf_counter()
        day = date.fromisoformat(event.get("trade_date"))
        expected = event.get("expected_symbols")
        excluded = event.get("excluded_symbols", {})
        if not isinstance(expected, (list, tuple)) or not expected or any(not isinstance(s, str) or not s for s in expected):
            raise ValueError("expected_symbols must specify the complete daily universe")
        if len(expected) != len(set(expected)):
            raise ValueError("expected_symbols must be unique")
        if not isinstance(excluded, dict) or any(s not in expected or not isinstance(reason, str) or not reason.strip() for s, reason in excluded.items()):
            raise ValueError("excluded_symbols must map expected symbols to explicit reasons")
        active = set(expected) - set(excluded)
        if not active:
            raise ValueError("daily universe contains no active symbols")
        logger.info("[batch/universe] request=%s expected=%d active=%d excluded=%d",
                    event.event_id, len(expected), len(active), len(excluded))
        if excluded:
            logger.info("[batch/exclusions] request=%s reasons=%s", event.event_id, excluded)
        frame = DailyMarketStore(self.get_config("daily_store", "data/daily")).load_window(
            day.isoformat(), int(self.get_config("history_dates", 320)))
        current = frame.filter(pl.col("datetime") == day)
        actual = set(current["vt_symbol"].to_list())
        if missing := active - actual:
            raise ValueError(f"daily data incomplete: missing={sorted(missing)}")
        active_rows = current.filter(pl.col("vt_symbol").is_in(sorted(active)))
        for field in ("open", "high", "low", "close", "volume"):
            values = active_rows[field].cast(pl.Float64, strict=False)
            if values.null_count() or not values.is_finite().all():
                raise ValueError(f"daily input has missing/non-finite {field}")
        logger.info("[batch/validate] request=%s passed active=%d available_today=%d OHLCV=finite",
                    event.event_id, len(active), len(actual))
        # Today's membership is explicit. Preserve historical cross sections unchanged.
        frame = frame.filter((pl.col("datetime") != day) | pl.col("vt_symbol").is_in(sorted(active)))
        frame = frame.with_columns(pl.col("datetime").cast(pl.Datetime("us")))
        logger.info("Daily Alpha101 start: date=%s rows=%d active=%d excluded=%d",
                    day, frame.height, len(active), len(excluded))
        panel = frame.rename({"datetime": "date", "vt_symbol": "symbol"}).to_pandas()
        calculator = Alpha101(panel)
        del panel, frame
        result = calculator.compute_all(self.get_config("alpha101_factors"), at=day)
        counts = {name: int(value) for name, value in result.notna().sum().items()}
        logger.info("[batch/calculate] request=%s complete rows=%d factors=%d valid_cells=%d/%d all_missing_factors=%s",
                    event.event_id, len(result), len(counts), sum(counts.values()), result.size,
                    [name for name, count in counts.items() if not count])
        output = pl.from_pandas(result.reset_index()).rename({"date": "datetime", "symbol": "vt_symbol"})
        closes = current.filter(pl.col("vt_symbol").is_in(sorted(active))).select("vt_symbol", "close")
        output = output.with_columns(pl.col("datetime").cast(pl.Datetime("us"))).join(closes, on="vt_symbol", how="left")
        batch_id = sha256(event.event_id.encode()).hexdigest()
        path = Path(self.get_config("result_store", "data/alpha101")).resolve() / day.isoformat() / f"{batch_id}-{uuid4().hex}.parquet"
        atomic_parquet(output, path)
        logger.info("[batch/save] request=%s bytes=%d path=%s", event.event_id, path.stat().st_size, path)
        payload = {"batch_id": batch_id, "request_id": event.event_id, "trade_date": day.isoformat(),
                   "result_path": str(path), "symbol_count": output.height, "factor_count": len(counts),
                   "valid_counts": counts, "excluded_symbols": excluded, "elapsed_seconds": perf_counter() - started}
        logger.info("Daily Alpha101 complete: date=%s shape=%s elapsed=%.3fs path=%s",
                    day, output.shape, payload["elapsed_seconds"], path)
        return payload


daily_alpha_batch_entry = make_module_entry(DailyAlphaBatchModule)
