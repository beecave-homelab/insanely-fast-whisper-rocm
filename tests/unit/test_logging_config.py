"""Tests for the logging configuration."""

import logging
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from rich.logging import RichHandler

from config import logging_config
from config.settings import Settings


def test_setup_logging_console_only(caplog):
    """Test setting up logging with console handler only."""
    # Clear any existing handlers
    logging.root.handlers = []

    # Setup logging with console handler only
    logging_config.setup_logging(log_level="INFO")

    # Check that root logger has a handler
    assert len(logging.root.handlers) > 0

    # Check that at least one handler is a RichHandler
    assert any(isinstance(h, RichHandler) for h in logging.root.handlers)

    # Test logging
    test_message = "Test log message"
    logging.info(test_message)

    # Check that the log message was captured
    assert test_message in caplog.text


def test_setup_logging_with_file(tmp_path):
    """Test setting up logging with file handler."""
    # Create a test log file
    log_file = tmp_path / "test.log"

    # Setup logging with file handler
    logging_config.setup_logging(log_level="DEBUG", log_file=log_file)

    # Test logging
    test_message = "Test file log message"
    logging.debug(test_message)

    # Ensure the log file was created and contains our message
    assert log_file.exists()
    assert test_message in log_file.read_text()


def test_log_levels(caplog):
    """Test that log levels are respected."""
    # Setup logging with INFO level
    logging_config.setup_logging(log_level="INFO")

    # Messages at different levels
    debug_msg = "This is a debug message"
    info_msg = "This is an info message"

    logging.debug(debug_msg)
    logging.info(info_msg)

    # Debug message should not appear in logs
    assert debug_msg not in caplog.text
    # Info message should appear
    assert info_msg in caplog.text


def test_logger_hierarchy():
    """Test that logger hierarchy is respected."""
    # Setup logging
    logging_config.setup_logging(log_level="INFO")

    # Get a child logger
    child_logger = logging.getLogger("tests.unit.test_logging")

    # Check that child logger inherits parent's level
    assert child_logger.getEffectiveLevel() == logging.INFO


def test_noisy_loggers_are_suppressed():
    """Test that noisy loggers have appropriate log levels set."""
    # Setup logging
    logging_config.setup_logging(log_level="INFO")

    # Check that noisy loggers are set to WARNING or higher
    assert logging.getLogger("transformers").level >= logging.WARNING
    assert logging.getLogger("torch").level >= logging.WARNING
    assert logging.getLogger("huggingface_hub").level >= logging.WARNING
    assert logging.getLogger("httpcore").level >= logging.WARNING
    assert logging.getLogger("httpx").level >= logging.WARNING
    assert logging.getLogger("urllib3").level >= logging.WARNING


def test_log_file_creation(tmp_path, monkeypatch):
    """Test that log file is created in the correct location."""
    # Create a test settings object with a custom logs directory
    logs_dir = tmp_path / "custom_logs"
    monkeypatch.setattr("config.logging_config.settings.logs_dir", logs_dir)

    # Import here to ensure the module uses the patched settings
    import importlib
    import config.logging_config

    importlib.reload(config.logging_config)

    # The module should have created the logs directory
    assert logs_dir.exists()
    assert logs_dir.is_dir()

    # The log file should have been created
    log_file = logs_dir / "app.log"
    assert log_file.exists()

    # Clean up
    importlib.reload(config.logging_config)
