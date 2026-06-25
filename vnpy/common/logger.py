from __future__ import annotations

import atexit
import logging
import os
import queue
import sys
import threading
from datetime import datetime
from logging.handlers import QueueHandler, QueueListener, RotatingFileHandler
from pathlib import Path
from typing import Optional


class DateSizeRotatingFileHandler(RotatingFileHandler):
    """
    File handler that supports:
    1. Daily log filename, for example: quant_20260516.log
    2. File size rotation, for example: quant_20260516.log.1
    """

    def __init__(self, log_dir: str, app_name: str, max_bytes: int = 10 * 1024 * 1024, backup_count: int = 10, encoding: str = "utf-8",):
        self.log_dir = Path(log_dir)
        self.app_name = app_name
        self.current_date = self._today()

        filename = self._build_filename()

        super().__init__(
            filename=filename,
            mode="a",
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding=encoding,
        )

    def _today(self) -> str:
        return datetime.now().strftime("%Y%m%d")

    def _build_filename(self) -> str:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        return str(self.log_dir / f"{self.app_name}_{self.current_date}.log")

    def shouldRollover(self, record: logging.LogRecord) -> int:
        """
        Rollover when date changes or file size exceeds maxBytes.
        """

        if self._today() != self.current_date:
            return 1

        return super().shouldRollover(record)

    def doRollover(self) -> None:
        """
        If date changes, switch to a new dated log file.
        Otherwise, use normal size-based rollover.
        """

        new_date = self._today()

        if new_date != self.current_date:
            if self.stream:
                self.stream.close()
                self.stream = None

            self.current_date = new_date
            self.baseFilename = os.path.abspath(self._build_filename())

            if not self.delay:
                self.stream = self._open()

            return

        super().doRollover()

_APP_LOGGER_NAME = "quant"
_LOGGER_INITIALIZED = False
_LOGGER_LOCK = threading.RLock()
_QUEUE_LISTENER: Optional[QueueListener] = None


def init_global_logger(app_name: str = "quant", log_dir: str = "logs", level: int = logging.INFO, max_bytes: int = 10 * 1024 * 1024, backup_count: int = 10, enable_console: bool = True, enable_file: bool = True,) -> logging.Logger:
    """
    Initialize global logger.

    This function should be called once in main.py.
    Other modules should only call get_logger().
    """

    global _APP_LOGGER_NAME
    global _LOGGER_INITIALIZED
    global _QUEUE_LISTENER

    with _LOGGER_LOCK:
        if _LOGGER_INITIALIZED:
            return logging.getLogger(_APP_LOGGER_NAME)

        _APP_LOGGER_NAME = app_name

        logger = logging.getLogger(app_name)
        logger.setLevel(level)
        logger.propagate = False
        logger.handlers.clear()

        log_queue: queue.Queue = queue.Queue(-1)

        queue_handler = QueueHandler(log_queue)
        queue_handler.setLevel(level)
        logger.addHandler(queue_handler)

        formatter = logging.Formatter(
            fmt=(
                "%(asctime)s.%(msecs)03d | "
                "%(levelname)-8s | "
                "%(threadName)s | "
                "%(name)s | "
                "%(filename)s:%(lineno)d | "
                "%(message)s"
            ),
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        handlers: list[logging.Handler] = []

        if enable_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(level)
            console_handler.setFormatter(formatter)
            handlers.append(console_handler)

        if enable_file:
            file_handler = DateSizeRotatingFileHandler(
                log_dir=log_dir,
                app_name=app_name,
                max_bytes=max_bytes,
                backup_count=backup_count,
            )
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            handlers.append(file_handler)

        _QUEUE_LISTENER = QueueListener(log_queue, *handlers, respect_handler_level=True,)
        _QUEUE_LISTENER.start()

        _LOGGER_INITIALIZED = True

        atexit.register(shutdown_global_logger)

        return logger

def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get logger by module name.

    Example:
        logger = get_logger("risk.engine")
        logger.info("risk engine started")
    """

    if not _LOGGER_INITIALIZED:
        init_global_logger()

    if not name:
        return logging.getLogger(_APP_LOGGER_NAME)

    if name.startswith(_APP_LOGGER_NAME):
        return logging.getLogger(name)

    return logging.getLogger(f"{_APP_LOGGER_NAME}.{name}")

def shutdown_global_logger() -> None:
    """
    Stop queue listener and flush remaining logs.
    """

    global _QUEUE_LISTENER
    global _LOGGER_INITIALIZED

    with _LOGGER_LOCK:
        if _QUEUE_LISTENER:
            _QUEUE_LISTENER.stop()
            _QUEUE_LISTENER = None

        _LOGGER_INITIALIZED = False

def debug(message: str, *args, **kwargs) -> None:
    get_logger().debug(message, *args, **kwargs)

def info(message: str, *args, **kwargs) -> None:
    get_logger().info(message, *args, **kwargs)

def warning(message: str, *args, **kwargs) -> None:
    get_logger().warning(message, *args, **kwargs)

def error(message: str, *args, **kwargs) -> None:
    get_logger().error(message, *args, **kwargs)

def exception(message: str, *args, **kwargs) -> None:
    get_logger().exception(message, *args, **kwargs)