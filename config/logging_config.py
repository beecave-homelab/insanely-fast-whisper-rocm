"""Logging configuration for the application.

This module provides logging configuration using Python's logging module with Rich
for enhanced console output. It sets up both console and file logging with
appropriate formatters and log levels.
"""

import logging
import logging.config
from pathlib import Path
from typing import Any, Dict, Optional, Union

# Rich is used for enhanced console output in the logging configuration
# The import is used in the string configuration below
# pylint: disable=import-error,unused-import
# type: ignore[import]
from rich.logging import RichHandler  # noqa: F401  # noqa: F811

from config.settings import settings


def setup_logging(
    log_level: Union[str, int] = logging.INFO,
    log_path: Optional[Union[str, Path]] = None,
) -> None:
    # pylint: disable=too-many-locals,too-many-statements
    """Configure logging with Rich handler for pretty console output.

    This function sets up logging with the following features:
    - Rich console output with colors and formatting
    - Optional file logging
    - Proper log level handling
    - Suppression of noisy loggers

    Args:
        log_level: The log level to use (can be string or int level).
        log_path: Optional path to the log file. If None, file logging is disabled.

    Note:
        The RichHandler is used in the logging configuration dictionary below,
        which is why we import it above even though it appears to be unused.
    """
    if isinstance(log_level, str):
        log_level = logging.getLevelName(log_level.upper())

    log_handlers: Dict[str, Dict[str, Any]] = {
        "console": {
            "level": log_level,
            "class": "rich.logging.RichHandler",
            "show_time": True,
            "show_path": False,
            "markup": True,
            "rich_tracebacks": True,
            "tracebacks_show_locals": False,
        }
    }

    if log_path:
        log_handlers["file"] = {
            "level": log_level,
            "class": "logging.FileHandler",
            "filename": str(log_path),
            "encoding": "utf-8",
            "formatter": "file",
        }

    log_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "console": {
                "format": "%(message)s",
                "datefmt": "[%X]",
            },
            "file": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
        },
        "handlers": log_handlers,
        "root": {
            "level": log_level,
            "handlers": ["console"] + (["file"] if log_path else []),
        },
        "loggers": {
            "__main__": {
                "level": log_level,
                "handlers": ["console"] + (["file"] if log_path else []),
                "propagate": False,
            },
            # Suppress noisy loggers
            "filelock": {"level": "WARNING"},
            "httpcore": {"level": "WARNING"},
            "httpx": {"level": "WARNING"},
            "urllib3": {"level": "WARNING"},
        },
    }

    # Apply the configuration
    logging.config.dictConfig(log_config)

    # Set log level for all loggers to the specified level
    logging.basicConfig(level=log_level, handlers=[])

    # Set log level for specific loggers
    for logger_name in ["uvicorn", "uvicorn.error", "uvicorn.access"]:
        logging.getLogger(logger_name).setLevel(log_level)
        logging.getLogger(logger_name).propagate = False

    # Set log level for transformers and other libraries
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)


# Initialize logging when module is imported
app_log_path = settings.logs_dir / "app.log" if settings.logs_dir else None
setup_logging(log_level=settings.log_level, log_path=app_log_path)

# Create logger for this module
logger = logging.getLogger(__name__)
logger.info("Logging configured successfully")

# Log important configuration
logger.info("Application starting with the following configuration:")
for field, value in settings.dict().items():
    if not field.startswith("_"):
        logger.debug("  %s: %s", field, value)
