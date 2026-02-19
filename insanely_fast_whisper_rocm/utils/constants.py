"""Constants and configuration management for the Insanely Fast Whisper API.

This module serves as the **single source of truth** for all application
configuration, implementing a centralized configuration pattern that ensures
consistency across all modules.

Configuration Management:

This module handles:
- Loading configuration from `.env` files using `load_dotenv()`
- Defining all environment variables with appropriate defaults
- Type conversion for boolean, integer, and float environment variables
- Hugging Face token loading via a single env var: HF_TOKEN
- Configuration file discovery in standard locations

Usage Guidelines:

For Application Modules:
All application modules should import constants from this module rather than
accessing environment variables directly:

    # Correct - Use centralized constants
    from insanely_fast_whisper_rocm.utils.constants import DEFAULT_MODEL, HF_TOKEN

    # Incorrect - Direct environment access bypasses centralized config
    model = os.getenv("WHISPER_MODEL", "some-default")

For Adding New Configuration:
When adding new environment variables:

1. Define the constant in this module with appropriate type conversion
2. Add proper documentation explaining the purpose and default value
3. Use descriptive constant names that clearly indicate their purpose
4. Group related constants together in logical sections
5. Provide sensible defaults that work for most use cases

Configuration File Locations:

The module automatically loads `.env` files. The loading order and precedence is:
1. Project root `.env` file: Loaded first. This file is typically used for
   development-specific configurations or base defaults for the project.
   Variables from this file will override any pre-existing shell environment
   variables.
2. ~/.config/insanely-fast-whisper-rocm/.env: Loaded second. This file is for
   user-specific configurations. If the same variable exists in both this file
   and the project root `.env` (or in the shell environment), the value from
   this user-specific file will take precedence (override).

This allows users to override project defaults and shell settings without
modifying the project's version-controlled `.env` file.

Type Conversion Patterns

Boolean Variables:
    FEATURE_ENABLED = os.getenv("FEATURE_ENABLED", "false").lower() == "true"

Integer Variables:
    NUMERIC_SETTING = int(os.getenv("NUMERIC_SETTING", "10"))

Float Variables:
    DECIMAL_SETTING = float(os.getenv("DECIMAL_SETTING", "1.5"))

String Variables:
    STRING_SETTING = os.getenv("STRING_SETTING", "default_value")

Architecture Benefits:

- Single Source of Truth: All configuration in one location
- Consistent Defaults: No conflicting defaults across modules
- Proper .env Support: All modules benefit from centralized file loading
- Type Safety: Centralized type conversion prevents runtime errors
- Easy Testing: Simplified mocking and configuration validation
- Maintainability: Configuration changes only need to be made once
"""

from __future__ import annotations

import importlib.util
import logging
import os
import sys
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as pkg_version
from typing import Literal

from dotenv import load_dotenv

from insanely_fast_whisper_rocm.utils.env_loader import (
    PROJECT_ROOT,
    USER_CONFIG_DIR,
    USER_ENV_EXISTS,
    USER_ENV_FILE,
    debug_print,
)

# --- Determine Project Root ---
# Use the single source of truth from env_loader.py to avoid drift. This value
# points to the repository root (one level above the package directory), which is
# required for features like locating the top-level '.env' and '.env.example'.
# PROJECT_ROOT, USER_CONFIG_DIR, USER_ENV_FILE are imported from env_loader.

# Re-export selected symbols for external consumers that import from this module.
__all__ = [
    "PROJECT_ROOT",
]

# --- Logging Setup ---
logger = logging.getLogger(__name__)

# --- Load user-specific .env (overrides project .env) ---
if USER_ENV_EXISTS:
    debug_print(f"Loading user .env: {USER_ENV_FILE}")
    load_dotenv(USER_ENV_FILE, override=True)

# --- Log final environment state ---
debug_print(
    f"Config loaded from .env: model={os.getenv('WHISPER_MODEL')}, "
    f"batch_size={os.getenv('WHISPER_BATCH_SIZE')}, "
    f"log_level={os.getenv('LOG_LEVEL', 'INFO')} "
    f"(CLI flags will override when specified)"
)

# --- Test/CI awareness and optional FS-check skipping ---
# Detect pytest and allow an explicit opt-in env flag to skip filesystem checks
# in code paths that tests monkeypatch. Production runs will keep checks.
IS_TEST_ENV = ("PYTEST_CURRENT_TEST" in os.environ) or ("pytest" in sys.modules)
SKIP_FS_CHECKS = os.getenv("IFW_SKIP_FS_CHECKS", "0") == "1" or IS_TEST_ENV

# Model configuration
DEFAULT_MODEL = os.getenv("WHISPER_MODEL", "distil-whisper/distil-large-v3")
DEFAULT_DEVICE = os.getenv(
    "WHISPER_DEVICE", "0"
)  # Use "0" for CUDA, "mps" for Apple Silicon
DEFAULT_BATCH_SIZE = int(os.getenv("WHISPER_BATCH_SIZE", "4"))

_TIMESTAMP_TYPE_ENV = os.getenv("WHISPER_TIMESTAMP_TYPE", "chunk")
if _TIMESTAMP_TYPE_ENV not in ("chunk", "word"):
    # Fallback to "chunk" if the environment variable has an invalid value
    _TIMESTAMP_TYPE_ENV = "chunk"
DEFAULT_TIMESTAMP_TYPE: Literal["chunk", "word"] = _TIMESTAMP_TYPE_ENV

DEFAULT_LANGUAGE = os.getenv("WHISPER_LANGUAGE", "None")  # None means auto-detect
DEFAULT_DTYPE = os.getenv("WHISPER_DTYPE", "float16")  # Data type for model inference
DEFAULT_BETTER_TRANSFORMER = (
    os.getenv("WHISPER_BETTER_TRANSFORMER", "false").lower() == "true"
)  # Use BetterTransformer
DEFAULT_CHUNK_LENGTH = int(
    os.getenv("WHISPER_CHUNK_LENGTH", "30")
)  # Audio chunk length in seconds

# Processing limits and timeouts
MAX_BATCH_SIZE = 32  # Maximum allowed batch size
MIN_BATCH_SIZE = 1  # Minimum allowed batch size
COMMAND_TIMEOUT_SECONDS = 3600  # Maximum time allowed for processing (1 hour)
MAX_AUDIO_SIZE_MB = 100  # Maximum allowed audio file size in MB
MAX_CONCURRENT_REQUESTS = 10  # Maximum number of concurrent processing requests

# Progress UI granularity
# Number of chunks to submit per pipeline call for user-visible progress updates.
# Larger values reduce progress update frequency but may improve throughput.
DEFAULT_PROGRESS_GROUP_SIZE = int(os.getenv("PROGRESS_GROUP_SIZE", "4"))

# Diarization configuration
DEFAULT_DIARIZATION_MODEL = os.getenv(
    "WHISPER_DIARIZATION_MODEL", "pyannote/speaker-diarization"
)
HF_TOKEN = os.getenv("HF_TOKEN")
MIN_SPEAKERS = 1  # Minimum number of speakers for diarization
MAX_SPEAKERS = 10  # Maximum number of speakers for diarization

# File handling
UPLOAD_DIR = os.getenv("WHISPER_UPLOAD_DIR", "temp_uploads")
TEMP_FILE_TTL_SECONDS = 3600  # Time-to-live for temporary files (1 hour)
DEFAULT_TRANSCRIPTS_DIR = os.getenv(
    "WHISPER_TRANSCRIPTS_DIR", "transcripts"
)  # Default directory for saving transcripts
# Application runtime timezone, also used for filename timestamps.
#
# Honor, in order of precedence:
# 1) APP_TIMEZONE (new, explicit)
# 2) FILENAME_TIMEZONE (backwards-compat alias used by older tests/configs)
# 3) TZ (common environment variable)
#
# Default remains 'UTC' to preserve existing test expectations.
APP_TIMEZONE = os.getenv(
    "APP_TIMEZONE",
    os.getenv("FILENAME_TIMEZONE", os.getenv("TZ", "UTC")),
)
SAVE_TRANSCRIPTIONS = (
    os.getenv("SAVE_TRANSCRIPTIONS", "true").lower() == "true"
)  # Whether to save transcriptions to disk

# Audio chunking configuration
AUDIO_CHUNK_DURATION = float(
    os.getenv("AUDIO_CHUNK_DURATION", "600")
)  # 10 minutes in seconds
AUDIO_CHUNK_OVERLAP = float(os.getenv("AUDIO_CHUNK_OVERLAP", "1.0"))  # 1 second overlap
AUDIO_CHUNK_MIN_DURATION = float(
    os.getenv("AUDIO_CHUNK_MIN_DURATION", "5.0")
)  # Minimum 5 seconds


# Subtitle readability configuration
# These constants control SRT/VTT formatting for better readability
USE_READABLE_SUBTITLES = (
    os.getenv("USE_READABLE_SUBTITLES", "true").lower() == "true"
)  # Master switch for the new segmentation pipeline

MAX_LINE_CHARS = int(os.getenv("MAX_LINE_CHARS", "42"))  # Max characters per line
MAX_LINES_PER_BLOCK = int(
    os.getenv("MAX_LINES_PER_BLOCK", "2")
)  # Max lines per subtitle block
MAX_BLOCK_CHARS = int(
    os.getenv("MAX_BLOCK_CHARS", str(MAX_LINE_CHARS * MAX_LINES_PER_BLOCK))
)  # Hard limit for block characters
MAX_BLOCK_CHARS_SOFT = int(
    os.getenv("MAX_BLOCK_CHARS_SOFT", "90")
)  # Soft limit for block characters
MIN_CPS = float(os.getenv("MIN_CPS", "8.0"))  # Minimum characters per second
MAX_CPS = float(os.getenv("MAX_CPS", "22.0"))  # Maximum characters per second
MIN_SEGMENT_DURATION_SEC = float(
    os.getenv("MIN_SEGMENT_DURATION_SEC", "0.9")
)  # Min segment duration
MAX_SEGMENT_DURATION_SEC = float(
    os.getenv("MAX_SEGMENT_DURATION_SEC", "4.0")
)  # Max segment duration
MIN_WORD_DURATION_SEC = float(
    os.getenv("MIN_WORD_DURATION_SEC", "0.04")
)  # Minimum word duration for sanitization
DISPLAY_BUFFER_SEC = float(
    os.getenv("DISPLAY_BUFFER_SEC", "0.2")
)  # Buffer for display timing

# Words and phrases for clause splitting and merging heuristics
SOFT_BOUNDARY_WORDS = os.getenv(
    "SOFT_BOUNDARY_WORDS", "and,but,or,so,for,nor,yet"
).split(",")
INTERJECTION_WHITELIST = os.getenv("INTERJECTION_WHITELIST", "um,uh,ah,er,like").split(
    ","
)


# --- Timestamp Stabilization defaults ---
DEFAULT_STABILIZE = os.getenv("STABILIZE_DEFAULT", "false").lower() == "true"
DEFAULT_DEMUCS = os.getenv("DEMUCS_DEFAULT", "false").lower() == "true"
DEFAULT_VAD = os.getenv("VAD_DEFAULT", "false").lower() == "true"
DEFAULT_VAD_THRESHOLD = float(os.getenv("VAD_THRESHOLD_DEFAULT", "0.35"))


# ROCm/HIP Configuration (for AMD GPUs)
HSA_OVERRIDE_GFX_VERSION = os.getenv(
    "HSA_OVERRIDE_GFX_VERSION"
)  # Override for AMD GPU support

# PyTorch ROCm memory allocation
# PyTorch 2.9.0+ renamed PYTORCH_HIP_ALLOC_CONF to PYTORCH_ALLOC_CONF
# We detect the version and set the appropriate environment variable
_pytorch_alloc_conf = os.getenv("PYTORCH_ALLOC_CONF") or os.getenv(
    "PYTORCH_HIP_ALLOC_CONF"
)
if _pytorch_alloc_conf is None:
    _pytorch_alloc_conf = "garbage_collection_threshold:0.7,max_split_size_mb:128"
    logger.info(
        "Applying default PyTorch allocator configuration to reduce "
        "fragmentation: %s. "
        "You can customize this via the PYTORCH_ALLOC_CONF or PYTORCH_HIP_ALLOC_CONF "
        "environment variable or your .env file.",
        _pytorch_alloc_conf,
    )
else:
    logger.info("Using custom PyTorch allocator configuration: %s", _pytorch_alloc_conf)

# Detect PyTorch version to determine which environment variable to set
try:
    _torch_spec = importlib.util.find_spec("torch")
except (AttributeError, ImportError, ModuleNotFoundError, ValueError):
    # Torch module cannot be located or is in a broken/partially-initialized state
    _torch_spec = None

if _torch_spec is not None:
    try:
        torch_version = pkg_version("torch")
        major, minor, *_ = map(int, torch_version.split(".")[:2])
        _use_new_alloc_conf = (major, minor) >= (2, 9)

        if _use_new_alloc_conf:
            os.environ["PYTORCH_ALLOC_CONF"] = _pytorch_alloc_conf
            logger.info("PyTorch %s detected: using PYTORCH_ALLOC_CONF", torch_version)
        else:
            os.environ["PYTORCH_HIP_ALLOC_CONF"] = _pytorch_alloc_conf
            logger.info(
                "PyTorch %s detected: using PYTORCH_HIP_ALLOC_CONF", torch_version
            )
    except PackageNotFoundError:
        # torch package found but version metadata unavailable
        # Default to the newer variable for PyTorch 2.9.0+
        os.environ["PYTORCH_ALLOC_CONF"] = _pytorch_alloc_conf
        logger.debug(
            "PyTorch version metadata unavailable, defaulting to PYTORCH_ALLOC_CONF"
        )
        # Also set the old variable for backward compatibility
        os.environ["PYTORCH_HIP_ALLOC_CONF"] = _pytorch_alloc_conf
else:
    # torch not installed yet (e.g., during initial setup)
    # Default to the newer variable for PyTorch 2.9.0+
    os.environ["PYTORCH_ALLOC_CONF"] = _pytorch_alloc_conf
    logger.debug(
        "PyTorch not yet installed, defaulting to PYTORCH_ALLOC_CONF "
        "(will be auto-adjusted on first import)"
    )
    # Also set the old variable for backward compatibility during setup
    os.environ["PYTORCH_HIP_ALLOC_CONF"] = _pytorch_alloc_conf

# Export for backward compatibility
PYTORCH_HIP_ALLOC_CONF = _pytorch_alloc_conf

HIP_LAUNCH_BLOCKING = (
    os.getenv("HIP_LAUNCH_BLOCKING", "false").lower() == "true"
)  # Synchronous HIP kernel launches

# Torchaudio backend selection for environments that rely on the soundfile
# backend (e.g., stable-ts lambda path calling torchaudio.save). Set this to
# "1" (or any non-empty string) when libsndfile + the Python package
# "soundfile" are installed in the environment.
TORCHAUDIO_USE_SOUNDFILE = os.getenv("TORCHAUDIO_USE_SOUNDFILE")

# API configuration
API_TITLE = "Insanely Fast Whisper API"
API_DESCRIPTION = "A FastAPI wrapper around the insanely-fast-whisper tool."
API_HOST = os.getenv("API_HOST", "0.0.0.0")  # API server host
API_PORT = int(os.getenv("API_PORT", "8000"))  # API server port
DEV_API_PORT = int(os.getenv("DEV_API_PORT", "8889"))  # Development API port
DEFAULT_RESPONSE_FORMAT = "json"

# WebUI configuration
# Defaults mirror the Click defaults used by the WebUI CLI.
# These allow the WebUI to pick up host/port from environment when flags are not
# provided on the command line.
WEBUI_HOST = os.getenv("WEBUI_HOST", "0.0.0.0")  # WebUI server host
WEBUI_PORT = int(os.getenv("WEBUI_PORT", "7860"))  # WebUI server port
DEV_WEBUI_PORT = int(os.getenv("DEV_WEBUI_PORT", "7862"))  # Development WebUI port

# API version (tests expect a specific string). Prefer package metadata but
# fall back to the expected default for local/test runs.
try:
    API_VERSION = pkg_version("insanely-fast-whisper-rocm")
except PackageNotFoundError:
    API_VERSION = "2.1.5"

# Convenience aliases expected by legacy code/tests
# The tests reference FILENAME_TIMEZONE, CONFIG_DIR, and ENV_FILE.  Map these
# to the new centralized names so both new and old code paths work without
# duplication.
FILENAME_TIMEZONE = APP_TIMEZONE  # Backwards-compatible alias
CONFIG_DIR = USER_CONFIG_DIR
ENV_FILE = USER_ENV_FILE

# Logging configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")  # Logging level


# Response formats
RESPONSE_FORMAT_JSON = "json"
RESPONSE_FORMAT_TEXT = "text"
# New response formats to align with OpenAI Whisper API
RESPONSE_FORMAT_VERBOSE_JSON = "verbose_json"
RESPONSE_FORMAT_SRT = "srt"
RESPONSE_FORMAT_VTT = "vtt"

SUPPORTED_RESPONSE_FORMATS = {
    RESPONSE_FORMAT_JSON,
    RESPONSE_FORMAT_TEXT,
    RESPONSE_FORMAT_VERBOSE_JSON,
    RESPONSE_FORMAT_SRT,
    RESPONSE_FORMAT_VTT,
}

# Supported audio formats (lowercase extensions)
SUPPORTED_AUDIO_FORMATS: set[str] = {
    ".mp3",
    ".flac",
    ".wav",
    ".m4a",
}

# Supported video formats (lowercase extensions)
SUPPORTED_VIDEO_FORMATS: set[str] = {
    ".mp4",
    ".mkv",
    ".webm",
    ".mov",
}

# Combined upload formats for WebUI and CLI
SUPPORTED_UPLOAD_FORMATS: set[str] = SUPPORTED_AUDIO_FORMATS | SUPPORTED_VIDEO_FORMATS
