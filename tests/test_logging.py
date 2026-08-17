"""Unit tests for structured logging and rotating file handler."""

from __future__ import annotations

import json
import logging
from io import StringIO
from pathlib import Path

from src.monitoring.logging_setup import JsonFormatter, setup_logging


def test_json_formatter():
    fmt = JsonFormatter()
    record = logging.LogRecord(
        name="test_logger",
        level=logging.INFO,
        pathname="test.py",
        lineno=10,
        msg="test message",
        args=(),
        exc_info=None,
    )
    output = fmt.format(record)
    parsed = json.loads(output)
    assert parsed["level"] == "INFO"
    assert parsed["logger"] == "test_logger"
    assert parsed["msg"] == "test message"
    assert "ts" in parsed


def test_rotating_file_logging(tmp_path: Path):
    log_file = tmp_path / "logs" / "bot.log"
    stream = StringIO()

    setup_logging(
        level="DEBUG",
        stream=stream,
        log_file=log_file,
        max_bytes=1024,
        backup_count=2,
    )

    logger = logging.getLogger("test_rot")
    # Write enough logs to trigger rotation
    for i in range(50):
        logger.info("Message %03d: A relatively long log line to fill up 1024 bytes buffer", i)

    assert log_file.exists()
    assert log_file.stat().st_size > 0
    # Backup files should exist after rotating
    rotated = list(tmp_path.glob("logs/bot.log*"))
    assert len(rotated) >= 2
