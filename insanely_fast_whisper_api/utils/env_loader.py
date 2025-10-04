"""Handles initial environment setup, .env file loading, and debug flag detection.

This module is responsible for:
- Locating project root and user-specific .env files.
- Performing an initial load of .env files to determine if debug output is enabled.
- Providing utility functions and constants for conditional debug printing.
- Exposing paths and existence flags for .env files to be used by constants.py
  for the main .env loading sequence.
"""

import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Use proper Python logging for environment loading
logger = logging.getLogger(__name__)

# Determine Project Root based on this file's location
# Assumes this file is in insanely_fast_whisper_api/utils/
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROJECT_ROOT_ENV_FILE = PROJECT_ROOT / ".env"

USER_CONFIG_DIR = Path.home() / ".config" / "insanely-fast-whisper-api"
USER_ENV_FILE = USER_CONFIG_DIR / ".env"

_cli_debug_mode = "--debug" in sys.argv

# Temporarily load .env files to check LOG_LEVEL for initial debug print decision.
# This is a pre-load specifically for determining SHOW_DEBUG_PRINTS.
# The main loading in constants.py will handle the final override logic.

_project_root_env_exists_temp = PROJECT_ROOT_ENV_FILE.exists()
if _project_root_env_exists_temp:
    load_dotenv(PROJECT_ROOT_ENV_FILE, override=True)

if not USER_CONFIG_DIR.exists():
    USER_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
_user_env_exists_temp = USER_ENV_FILE.exists()
if _user_env_exists_temp:
    load_dotenv(
        USER_ENV_FILE, override=True
    )  # User .env can override project for LOG_LEVEL check

_env_log_level_temp = os.getenv("LOG_LEVEL", "").upper()
_env_debug_mode_temp = _env_log_level_temp == "DEBUG"

SHOW_DEBUG_PRINTS = _cli_debug_mode or _env_debug_mode_temp

# Configure logging early if debug mode is detected
if SHOW_DEBUG_PRINTS:
    logging.basicConfig(
        level=logging.INFO,  # Keep root logger at INFO
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        force=True,
    )
    # Only enable DEBUG for our application's loggers
    logging.getLogger("insanely_fast_whisper_api").setLevel(logging.DEBUG)
    # Suppress noisy third-party loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("torio").setLevel(logging.WARNING)
    # MIOpen warnings are printed directly to stderr, not through Python logging,
    # so we can't suppress them here. They're harmless workspace allocation attempts.


def debug_print(message: str) -> None:
    """Log environment loading messages at DEBUG level if enabled.

    Args:
        message: The message to log.
    """
    if SHOW_DEBUG_PRINTS:
        logger.debug(message)


# Perform the actual loading of the project root .env file here
if _project_root_env_exists_temp:
    debug_print(f"Loading project .env: {PROJECT_ROOT_ENV_FILE}")
    load_dotenv(
        PROJECT_ROOT_ENV_FILE, override=True
    )  # This is the main load for project root .env
else:
    debug_print(f"No project .env found at: {PROJECT_ROOT_ENV_FILE}")

# Expose pre-checked existence and paths for constants.py to use for the main
# load. PROJECT_ROOT_ENV_EXISTS is still useful if constants.py wants to know
# if it was loaded.
PROJECT_ROOT_ENV_EXISTS = _project_root_env_exists_temp
USER_ENV_EXISTS = _user_env_exists_temp
