"""Handles initial environment setup, .env file loading, and debug flag detection.

This module is responsible for:
- Locating project root and user-specific .env files.
- Performing an initial load of .env files to determine if debug output is enabled.
- Providing utility functions and constants for conditional debug printing.
- Exposing paths and existence flags for .env files to be used by constants.py
  for the main .env loading sequence.
"""

import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

# Determine Project Root based on this file's location
# Assumes this file is in insanely_fast_whisper_rocm/utils/
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROJECT_ROOT_ENV_FILE = PROJECT_ROOT / ".env"

USER_CONFIG_DIR = Path.home() / ".config" / "insanely-fast-whisper-rocm"
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


def debug_print(message: str):
    """Prints a message if SHOW_DEBUG_PRINTS is True, formatted like a log entry."""
    if SHOW_DEBUG_PRINTS:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]
        print(f"{timestamp} - ENV_LOADER_DEBUG - INFO - {message}")


# Perform the actual loading of the project root .env file here
debug_print(f"env_loader.py: Checking for project .env at: {PROJECT_ROOT_ENV_FILE}")
if _project_root_env_exists_temp:
    debug_print("env_loader.py: Found project .env file. Loading now...")
    load_dotenv(
        PROJECT_ROOT_ENV_FILE, override=True
    )  # This is the main load for project root .env
else:
    debug_print("env_loader.py: Project .env file NOT found.")

# Expose pre-checked existence and paths for constants.py to use for the main
# load. PROJECT_ROOT_ENV_EXISTS is still useful if constants.py wants to know
# if it was loaded.
PROJECT_ROOT_ENV_EXISTS = _project_root_env_exists_temp
USER_ENV_EXISTS = _user_env_exists_temp
