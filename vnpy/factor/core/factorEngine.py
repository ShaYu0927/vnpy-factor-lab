from __future__ import annotations

import os
import time
import traceback
from abc import ABC, abstractmethod
from concurrent.futures import Executor, Future, ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

class ExecutionMode(str, Enum):
    """
    Factor execution mode.

    SYNC:
        Best for debugging and small datasets.

    THREAD:
        Best for I/O-heavy tasks, pandas/numpy/polars operations that may release GIL,
        or real-time pipelines where calculation should not block the callback thread.

    PROCESS:
        Best for pure Python CPU-heavy factor calculation. Factor objects and input data
        must be picklable.
    """

    SYNC = "sync"
    THREAD = "thread"
    PROCESS = "process"
    
@dataclass(slots=True)
class FactorContext:
    """
    Runtime context passed to every factor.
    trade_date:
        Current calculation date or timestamp.

    params:
        Extra runtime parameters. Keep it read-only by convention.
    """

    trade_date: Optional[str] = None
    params: Mapping[str, Any] = field(default_factory=dict)
    
@dataclass(slots=True)
class FactorValue:
    """
    Single factor output.
    """
    symbol: str
    factor_name: str
    value: Any
    trade_date: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)
    
@dataclass(slots=True)
class FactorError:
    """
    Calculation error record.
    """

    symbol: str
    factor_name: str
    error: str
    traceback_text: str
    
    
@dataclass(slots=True)
class FactorBatchResult:
    """
    Batch result returned by FactorEngine.
    """

    values: List[FactorValue] = field(default_factory=list)
    errors: List[FactorError] = field(default_factory=list)
    elapsed_ms: float = 0.0

    @property
    def ok(self) -> bool:
        return not self.errors
    
class Factor(ABC):
    """
    Base class of all factors.

    Important rule:
        Factor should be stateless or read-only after construction.
        Do not create threads inside a Factor.
        Do not mutate shared global state inside calculate().
    """

    name: str = "base_factor"
    min_bars: int = 1

    @abstractmethod
    def calculate(self, symbol: str, data: Any, context: FactorContext) -> Any:
        """
        Calculate factor value for one symbol.

        Args:
            symbol: Security symbol, for example "sh.600000".
            data: Historical bars or feature table of this symbol.
            context: Runtime context.

        Returns:
            Raw factor value. The engine will wrap it as FactorValue.
        """

        raise NotImplementedError

    def validate(self, symbol: str, data: Any, context: FactorContext) -> bool:
        """
        Optional validation hook.

        Override this method if the factor needs to check data length,
        columns, frequency, suspension status, etc.
        """
        return data is not None
    

@dataclass(frozen=True, slots=True)
class FactorTask:
    """
    Internal calculation task.
    """
    symbol: str
    factor: Factor
    data: Any
    context: FactorContext
    
@dataclass(slots=True)
class FactorEngineConfig:
    """
    Engine configuration.
    """
    mode: ExecutionMode = ExecutionMode.SYNC
    max_workers: Optional[int] = None
    fail_fast: bool = False
    skip_invalid_data: bool = True
    chunk_size: int = 256

    def resolved_workers(self) -> int:
        if self.max_workers and self.max_workers > 0:
            return self.max_workers

        cpu_count = os.cpu_count() or 1

        if self.mode == ExecutionMode.PROCESS:
            return max(1, cpu_count - 1)

        if self.mode == ExecutionMode.THREAD:
            return min(32, cpu_count * 4)

        return 1
    
class FactorEngine:
    """
    Factor calculation scheduler.

    Responsibilities:
        1. Build factor calculation tasks.
        2. Choose sync/thread/process execution mode.
        3. Collect values and errors.
        4. Keep factor implementation clean and independent from concurrency logic.
    """

    def __init__(self, factors: Sequence[Factor], mode: ExecutionMode = ExecutionMode.SYNC, max_workers: Optional[int] = None, fail_fast: bool = False, skip_invalid_data: bool = True, chunk_size: int = 256,) -> None:
        if not factors:
            raise ValueError("factors must not be empty")

        self._factors: Tuple[Factor, ...] = tuple(factors)
        self._config = FactorEngineConfig(
            mode=mode,
            max_workers=max_workers,
            fail_fast=fail_fast,
            skip_invalid_data=skip_invalid_data,
            chunk_size=chunk_size,
        )

        self._check_factor_names()

    @property
    def factors(self) -> Tuple[Factor, ...]:
        return self._factors

    @property
    def config(self) -> FactorEngineConfig:
        return self._config

    def calculate_one(self, symbol: str, data: Any, context: Optional[FactorContext] = None,) -> FactorBatchResult:
        """
        Calculate all factors for one symbol.
        """

        return self.calculate_many({symbol: data}, context=context)
    
    def calculate_many(self, symbol_data_map: Mapping[str, Any], context: Optional[FactorContext] = None, ) -> FactorBatchResult:
        """
        Calculate all configured factors for many symbols.

        Args:
            symbol_data_map:
                Mapping from symbol to bars/table/features.

            context:
                Runtime context, for example trade_date.
        """
        if not symbol_data_map:
            return FactorBatchResult()

        start = time.perf_counter()
        ctx = context or FactorContext()
        tasks = self._build_tasks(symbol_data_map, ctx)

        if self._config.mode == ExecutionMode.SYNC:
            result = self._run_sync(tasks)
        else:
            result = self._run_parallel(tasks)

        result.elapsed_ms = (time.perf_counter() - start) * 1000.0
        return result
    
    
    def calculate_factor_cross_section(self, factor_name: str, symbol_data_map: Mapping[str, Any], context: Optional[FactorContext] = None,) -> FactorBatchResult:
        """
        Calculate one specific factor across many symbols.

        Useful for cross-sectional factor ranking.
        """

        factor = self._get_factor(factor_name)
        engine = FactorEngine(
            factors=[factor],
            mode=self._config.mode,
            max_workers=self._config.max_workers,
            fail_fast=self._config.fail_fast,
            skip_invalid_data=self._config.skip_invalid_data,
            chunk_size=self._config.chunk_size,
        )
        return engine.calculate_many(symbol_data_map, context=context)
    
    def _check_factor_names(self) -> None:
        seen = set()
        for factor in self._factors:
            if not factor.name:
                raise ValueError("factor.name must not be empty")
            if factor.name in seen:
                raise ValueError(f"duplicate factor name: {factor.name}")
            seen.add(factor.name)
            
    def _get_factor(self, factor_name: str) -> Factor:
        for factor in self._factors:
            if factor.name == factor_name:
                return factor
        raise KeyError(f"factor not found: {factor_name}")
    

    def _build_tasks(
        self,
        symbol_data_map: Mapping[str, Any],
        context: FactorContext,
    ) -> List[FactorTask]:
        tasks: List[FactorTask] = []

        for symbol, data in symbol_data_map.items():
            for factor in self._factors:
                if self._config.skip_invalid_data and not factor.validate(symbol, data, context):
                    continue

                tasks.append(
                    FactorTask(
                        symbol=symbol,
                        factor=factor,
                        data=data,
                        context=context,
                    )
                )

        return tasks
    
    
    
    def _run_sync(self, tasks: Sequence[FactorTask]) -> FactorBatchResult:
        values: List[FactorValue] = []
        errors: List[FactorError] = []

        for task in tasks:
            value, error = _execute_factor_task(task)
            if error is not None:
                errors.append(error)
                if self._config.fail_fast:
                    break
            elif value is not None:
                values.append(value)

        return FactorBatchResult(values=values, errors=errors)

    
            
    def _run_parallel(self, tasks: Sequence[FactorTask]) -> FactorBatchResult:
        values: List[FactorValue] = []
        errors: List[FactorError] = []

        if not tasks:
            return FactorBatchResult()

        executor_cls = self._select_executor()
        max_workers = self._config.resolved_workers()

        with executor_cls(max_workers=max_workers) as executor:
            futures = self._submit_tasks(executor, tasks)

            for future in as_completed(futures):
                value, error = future.result()

                if error is not None:
                    errors.append(error)
                    if self._config.fail_fast:
                        self._cancel_remaining(futures)
                        break
                elif value is not None:
                    values.append(value)

        return FactorBatchResult(values=values, errors=errors)

    def _submit_tasks(
        self,
        executor: Executor,
        tasks: Sequence[FactorTask],
    ) -> List[Future]:
        return [executor.submit(_execute_factor_task, task) for task in tasks]

    def _cancel_remaining(self, futures: Iterable[Future]) -> None:
        for future in futures:
            if not future.done():
                future.cancel()
                    
    def _select_executor(self):
        if self._config.mode == ExecutionMode.THREAD:
            return ThreadPoolExecutor

        if self._config.mode == ExecutionMode.PROCESS:
            return ProcessPoolExecutor

        raise ValueError(f"unsupported execution mode: {self._config.mode}")
    
    

def _execute_factor_task(task: FactorTask) -> Tuple[Optional[FactorValue], Optional[FactorError]]:
        """
        Top-level worker function.

        It must stay at module top level so ProcessPoolExecutor can pickle it.
        """

        factor = task.factor
        symbol = task.symbol

        try:
            raw_value = factor.calculate(symbol=symbol, data=task.data, context=task.context)
            return (
                FactorValue(
                    symbol=symbol,
                    factor_name=factor.name,
                    value=raw_value,
                    trade_date=task.context.trade_date,
                ),
                None,
            )
        except Exception as exc:  # pylint: disable=broad-except
            return (
                None,
                FactorError(
                    symbol=symbol,
                    factor_name=factor.name,
                    error=str(exc),
                    traceback_text=traceback.format_exc(),
                ),
            )
    
