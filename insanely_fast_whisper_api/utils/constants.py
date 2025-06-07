"""Constants and configuration management for the Insanely Fast Whisper API.

This module serves as the **single source of truth** for all application configuration,
implementing a centralized configuration pattern that ensures consistency across all modules.

## Configuration Management

This module handles:
- Loading configuration from `.env` files using `load_dotenv()`
- Defining all environment variables with appropriate defaults
- Type conversion for boolean, integer, and float environment variables
- Token fallback logic (HF_TOKEN -> HUGGINGFACE_TOKEN)
- Configuration file discovery in standard locations

## Usage Guidelines

**For Application Modules:**
All application modules should import constants from this module rather than
accessing environment variables directly:

    # ✅ Correct - Use centralized constants
    from insanely_fast_whisper_api.utils.constants import DEFAULT_MODEL, HF_TOKEN

    # ❌ Incorrect - Direct environment access bypasses centralized config
    model = os.getenv("WHISPER_MODEL", "some-default")

**For Adding New Configuration:**
When adding new environment variables:

1. Define the constant in this module with appropriate type conversion
2. Add proper documentation explaining the purpose and default value
3. Use descriptive constant names that clearly indicate their purpose
4. Group related constants together in logical sections
5. Provide sensible defaults that work for most use cases

## Configuration File Locations

The module automatically loads `.env` files from these locations (in order):
1. `~/.config/insanely-fast-whisper-api/.env` (user-specific configuration)
2. Project root `.env` file (development configuration)

## Type Conversion Patterns

**Boolean Variables:**
    FEATURE_ENABLED = os.getenv("FEATURE_ENABLED", "false").lower() == "true"

**Integer Variables:**
    NUMERIC_SETTING = int(os.getenv("NUMERIC_SETTING", "10"))

**Float Variables:**
    DECIMAL_SETTING = float(os.getenv("DECIMAL_SETTING", "1.5"))

**String Variables:**
    STRING_SETTING = os.getenv("STRING_SETTING", "default_value")

## Architecture Benefits

- **Single Source of Truth**: All configuration in one location
- **Consistent Defaults**: No conflicting defaults across modules
- **Proper .env Support**: All modules benefit from centralized file loading
- **Type Safety**: Centralized type conversion prevents runtime errors
- **Easy Testing**: Simplified mocking and configuration validation
- **Maintainability**: Configuration changes only need to be made once
"""

import os
from pathlib import Path
from typing import Set, Literal

from dotenv import load_dotenv

# Configuration paths
CONFIG_DIR = Path.home() / ".config" / "insanely-fast-whisper-api"
ENV_FILE = CONFIG_DIR / ".env"

if not CONFIG_DIR.exists():
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

if ENV_FILE.exists():
    load_dotenv(ENV_FILE)

# Model configuration
DEFAULT_MODEL = os.getenv("WHISPER_MODEL", "distil-whisper/distil-large-v3")
DEFAULT_DEVICE = os.getenv(
    "WHISPER_DEVICE", "0"
)  # Use "0" for CUDA, "mps" for Apple Silicon
DEFAULT_BATCH_SIZE = int(os.getenv("WHISPER_BATCH_SIZE", "4"))

_timestamp_type_env = os.getenv("WHISPER_TIMESTAMP_TYPE", "chunk")
if _timestamp_type_env not in ("chunk", "word"):
    # Fallback to "chunk" if the environment variable has an invalid value
    _timestamp_type_env = "chunk"
DEFAULT_TIMESTAMP_TYPE: Literal["chunk", "word"] = _timestamp_type_env

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

# Diarization configuration
DEFAULT_DIARIZATION_MODEL = os.getenv(
    "WHISPER_DIARIZATION_MODEL", "pyannote/speaker-diarization"
)
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv(
    "HUGGINGFACE_TOKEN"
)  # Support both token names
MIN_SPEAKERS = 1  # Minimum number of speakers for diarization
MAX_SPEAKERS = 10  # Maximum number of speakers for diarization

# File handling
UPLOAD_DIR = os.getenv("WHISPER_UPLOAD_DIR", "temp_uploads")
TEMP_FILE_TTL_SECONDS = 3600  # Time-to-live for temporary files (1 hour)
DEFAULT_TRANSCRIPTS_DIR = os.getenv(
    "WHISPER_TRANSCRIPTS_DIR", "transcripts"
)  # Default directory for saving transcripts
FILENAME_TIMEZONE = os.getenv(
    "FILENAME_TIMEZONE", "UTC"
)  # Timezone for filename timestamps
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

# ROCm/HIP Configuration (for AMD GPUs)
HSA_OVERRIDE_GFX_VERSION = os.getenv(
    "HSA_OVERRIDE_GFX_VERSION"
)  # Override for AMD GPU support
PYTORCH_HIP_ALLOC_CONF = os.getenv(
    "PYTORCH_HIP_ALLOC_CONF"
)  # PyTorch ROCm memory allocation
HIP_LAUNCH_BLOCKING = (
    os.getenv("HIP_LAUNCH_BLOCKING", "false").lower() == "true"
)  # Synchronous HIP kernel launches

# API configuration
API_TITLE = "Insanely Fast Whisper API"
API_DESCRIPTION = "A FastAPI wrapper around the insanely-fast-whisper tool."
API_VERSION = "0.3.1"
API_HOST = os.getenv("API_HOST", "0.0.0.0")  # API server host
API_PORT = int(os.getenv("API_PORT", "8000"))  # API server port
DEFAULT_RESPONSE_FORMAT = "json"

# Logging configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")  # Logging level

# Response formats
RESPONSE_FORMAT_JSON = "json"
RESPONSE_FORMAT_TEXT = "text"
SUPPORTED_RESPONSE_FORMATS = {RESPONSE_FORMAT_JSON, RESPONSE_FORMAT_TEXT}

# Supported audio formats (lowercase extensions)
SUPPORTED_AUDIO_FORMATS: Set[str] = {
    ".mp3",
    ".flac",
    ".wav",
}
