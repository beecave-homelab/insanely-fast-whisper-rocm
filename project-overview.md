# Project Overview | Insanely Fast Whisper API (ROCm)

A comprehensive Whisper-based speech recognition toolkit designed specifically to provide **AMD GPU (ROCm v6.1) support** for high-performance Automatic Speech Recognition (ASR) and translation. This package extends the capabilities of the original [insanely-fast-whisper](https://github.com/Vaibhavs10/insanely-fast-whisper) by providing multiple interfaces, ROCm compatibility, and production-ready architecture. This overview is the **single source of truth** for developers working on this codebase.

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://www.python.org)
[![Version](https://img.shields.io/badge/Version-v0.5.0)](#version-summary)
[![API](https://img.shields.io/badge/API-FastAPI-green)](#api-server-details)
[![CLI](https://img.shields.io/badge/CLI-Click-yellow)](#cli-command-line-interface-details)
[![WebUI](https://img.shields.io/badge/WebUI-Gradio-orange)](#webui-gradio-interface-details)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE.txt)

---

## Table of Contents

- [Quickstart for Developers](#quickstart-for-developers)
- [Version Summary](#version-summary)
- [Project Features](#project-features)
- [Project Structure](#project-structure)
- [Architecture Highlights](#architecture-highlights)
- [Filename Conventions](#filename-conventions)
- [Configuration System](#configuration-system)
- [Application Interfaces](#application-interfaces)
  - [API Server Details](#api-server-details)
  - [WebUI (Gradio Interface) Details](#webui-gradio-interface-details)
  - [CLI (Command Line Interface) Details](#cli-command-line-interface-details)
- [Dependency Management](#dependency-management-with-pdm)
- [Error Handling](#error-handling)
- [Development Guidelines](#development-guidelines)
- [Deployment Options](#deployment-options)
- [Monitoring & Security](#monitoring--security)
- [Import Standardization](#import-standardization)
- [Recent Enhancements](#recent-enhancements)
- [Bug Fixes](#bug-fixes)

---

## Quickstart for Developers

```bash
# Clone and activate
git clone https://github.com/beecave-homelab/insanely-fast-whisper-rocm.git
cd insanely-fast-whisper-rocm
```

**Use Docker (recommended):**

```bash
cp .env.example .env
# Edit .env as needed

docker compose up --build -d
```

**Or run locally with PDM (for development):**

```bash
# Install PDM (if not already installed)
curl -sSL https://pdm-project.org/install-pdm.py | python3 -

# Install project dependencies (including dev and rocm groups)
pdm install -G dev -G rocm

# Activate the PDM-managed environment (optional, PDM handles it)
# pdm shell

# Choose your interface (run via PDM):
pdm run start-api          # API Server
pdm run start-webui        # WebUI Interface  
pdm run cli transcribe audio.mp3  # CLI
```

---

## Version Summary

### 🏷️ **Current Version: v0.5.0** *(June 2025)*

**Latest improvements**: Major architectural refactoring, migration to `pdm`, and a new modular CLI.

### 📊 **Release Overview**

| Version | Date | Type | Key Features |
|---------|------|------|--------------|
| **v0.5.0** | Jun 2025 | ✨ Minor | Major import refactor, `pdm` migration, modular CLI |
| **v0.4.1** | Jun 2025 | 🐛 Patch | WebUI download fixes, stability |
| **v0.4.0** | Jun 2025 | ✨ Minor | Versioning improvements, logging enhancements |
| **v0.3.1** | Jun 2025 | 🐛 Patch | ZIP fixes, audio validation, stability |
| **v0.3.0** | May 2025 | ✨ Minor | WebUI modularization, batch processing |
| **v0.2.1** | May 2025 | ♻️ Minor | Import standardization, core refinements |
| **v0.2.0** | May 2025 | 🔄 Major | Breaking: Hugging Face direct integration |
| **v0.1.2** | Mar 2025 | ✨ Minor | First WebUI introduction |
| **v0.1.1** | Jan 2025 | 🔧 Minor | Enhanced functionality, logging |
| **v0.1.0** | Jan 2025 | 🎉 Major | Initial Release: FastAPI wrapper |

### 📈 **Development Stats**

- **100+ commits** across 5 months of development
- **3 major architectural refactors**
- **1 breaking change** (v0.2.0)

---

> 📖 **For complete version history, changelog, and detailed release notes, see [VERSIONS.md](VERSIONS.md)**

---

## Project Features

### Primary Focus: ROCm Support

- **AMD GPU (ROCm v6.1) Support**: First-class AMD GPU acceleration for Whisper models

- **Extended Original Package**: Builds upon [insanely-fast-whisper](https://github.com/Vaibhavs10/insanely-fast-whisper) with additional interfaces and ROCm compatibility
- **Production-Ready Architecture**: Beyond CLI-only approach of original package

### Core Capabilities

- **Transcription**: Audio to text in source language

- **Translation**: Audio to English
- **Multiple Audio Formats**: Support for .wav, .flac, and .mp3 formats
- **Filename Standardization**: Predictable and configurable output naming

### Interface Options

- **FastAPI Server**: RESTful API with OpenAI-compatible v1 endpoints (`/v1/audio/transcriptions`, etc.)

- **Gradio WebUI**: Batch file upload, live progress tracking, ZIP downloads
- **CLI Interface**: Command-line tool for single-file processing
- **Model Management**: Automatic Hugging Face model downloading and caching
- **Docker Support**: Full containerization with development and production configurations

### Architecture

- **Modular Design**: Split core, audio, API, CLI, WebUI, and utils

- **Error Handling**: Layered, type-specific, with full trace logging
- **Direct Hugging Face Integration**: Native `transformers.pipeline` support
- **Configurable Processing**: Batch size, device, model selection
- **ROCm Integration**: Optimized PyTorch and ONNX runtime configurations for AMD GPUs

---

## Project Structure

```md
├── [insanely_fast_whisper_api/](./insanely_fast_whisper_api/)          # Main package
│   ├── [__init__.py](./insanely_fast_whisper_api/__init__.py)                     # Package initialization
│   ├── [__main__.py](./insanely_fast_whisper_api/__main__.py)                     # Module entry point
│   ├── [main.py](./insanely_fast_whisper_api/main.py)                         # FastAPI application entry
│   ├── [logging_config.yaml](./insanely_fast_whisper_api/logging_config.yaml)             # Logging configuration
│   ├── [api/](./insanely_fast_whisper_api/api/)                            # FastAPI application layer
│   │   ├── [__init__.py](./insanely_fast_whisper_api/api/__init__.py)
│   │   ├── [app.py](./insanely_fast_whisper_api/api/app.py)                      # FastAPI app setup
│   │   ├── [routes.py](./insanely_fast_whisper_api/api/routes.py)                   # API endpoints
│   │   ├── [models.py](./insanely_fast_whisper_api/api/models.py)                   # Pydantic data models
│   │   ├── [dependencies.py](./insanely_fast_whisper_api/api/dependencies.py)             # Dependency injection
│   │   ├── [middleware.py](./insanely_fast_whisper_api/api/middleware.py)               # Request/response middleware
│   │   └── [responses.py](./insanely_fast_whisper_api/api/responses.py)                # Response formatters
│   ├── [core/](./insanely_fast_whisper_api/core/)                           # Core ASR logic
│   │   ├── [__init__.py](./insanely_fast_whisper_api/core/__init__.py)
│   │   ├── [pipeline.py](./insanely_fast_whisper_api/core/pipeline.py)                 # ASR orchestration
│   │   ├── [asr_backend.py](./insanely_fast_whisper_api/core/asr_backend.py)              # Whisper model backend
│   │   ├── [storage.py](./insanely_fast_whisper_api/core/storage.py)                  # File lifecycle management
│   │   ├── [utils.py](./insanely_fast_whisper_api/core/utils.py)                    # Core utilities
│   │   └── [errors.py](./insanely_fast_whisper_api/core/errors.py)                   # Exception classes
│   ├── [audio/](./insanely_fast_whisper_api/audio/)                          # Audio processing
│   │   ├── [__init__.py](./insanely_fast_whisper_api/audio/__init__.py)
│   │   ├── [processing.py](./insanely_fast_whisper_api/audio/processing.py)               # Validation and preprocessing
│   │   └── [results.py](./insanely_fast_whisper_api/audio/results.py)                  # Output formatting
│   ├── [cli/](./insanely_fast_whisper_api/cli/)                            # CLI tools
│   │   ├── [__init__.py](./insanely_fast_whisper_api/cli/__init__.py)
│   │   ├── [cli.py](./insanely_fast_whisper_api/cli/cli.py)                      # CLI entry point
│   │   ├── [commands.py](./insanely_fast_whisper_api/cli/commands.py)                 # Subcommand logic
│   │   └── [facade.py](./insanely_fast_whisper_api/cli/facade.py)                   # High-level CLI wrapper
│   ├── [webui/](./insanely_fast_whisper_api/webui/)                          # Web UI (Gradio)
│   │   ├── [__init__.py](./insanely_fast_whisper_api/webui/__init__.py)
│   │   ├── [cli.py](./insanely_fast_whisper_api/webui/cli.py)                      # WebUI CLI entry point
│   │   ├── [ui.py](./insanely_fast_whisper_api/webui/ui.py)                       # Gradio interface
│   │   ├── [handlers.py](./insanely_fast_whisper_api/webui/handlers.py)                 # Upload + result management
│   │   ├── [formatters.py](./insanely_fast_whisper_api/webui/formatters.py)               # Export formats (TXT, JSON, SRT)
│   │   ├── [utils.py](./insanely_fast_whisper_api/webui/utils.py)                    # WebUI utilities
│   │   ├── [errors.py](./insanely_fast_whisper_api/webui/errors.py)                   # UI-specific exceptions
│   │   ├── [zip_creator.py](./insanely_fast_whisper_api/webui/zip_creator.py)              # ZIP archive builder
│   │   └── [merge_handler.py](./insanely_fast_whisper_api/webui/merge_handler.py)            # Transcription file merge handlers
│   └── [utils/](./insanely_fast_whisper_api/utils/)                          # General utilities
│       ├── [__init__.py](./insanely_fast_whisper_api/utils/__init__.py)
│       ├── [constants.py](./insanely_fast_whisper_api/utils/constants.py)                # Core environment variable definitions
│       ├── [env_loader.py](./insanely_fast_whisper_api/utils/env_loader.py)               # Hierarchical .env loading & debug print logic
│       ├── [download_hf_model.py](./insanely_fast_whisper_api/utils/download_hf_model.py)        # Model downloading & caching
│       ├── [file_utils.py](./insanely_fast_whisper_api/utils/file_utils.py)               # File operations
│       └── [filename_generator.py](./insanely_fast_whisper_api/utils/filename_generator.py)       # Unified filename logic
├── [scripts/](./scripts/)                            # Utility and maintenance scripts
│   └── [setup_config.py](./scripts/setup_config.py)               # Script to set up user-specific .env file
```

---

## Architecture Highlights

### Core Refactor (v0.2.0+)

- Direct integration with Hugging Face `pipeline`
- No subprocess dependency on `insanely-fast-whisper`
- Modular architecture: [`pipeline.py`](./insanely_fast_whisper_api/core/pipeline.py), [`asr_backend.py`](./insanely_fast_whisper_api/core/asr_backend.py), etc.
- *See [v0.2.0 changelog](VERSIONS.md#v020---may-2025) for complete architectural changes*

### WebUI Refactor (v0.3.0+)

- Full Gradio-based multi-file support
- Native ZIP creation and download buttons
- Real-time batch progress with `gr.Progress`
- Backward-compatible with single-file use
- *See [v0.3.0 changelog](VERSIONS.md#v030---may-2025) for WebUI modularization details*

### Multiple File Support (v0.3.1+)

- `gr.File(file_count="multiple")`
- Native batching + ZIP archive output
- Chunk-level processing progress
- *See [v0.3.1 changelog](VERSIONS.md#v031---june-2025) for latest enhancements*

---

## Filename Conventions

**Pattern:** `{audio_stem}_{task}_{timestamp}.{extension}`

| Part         | Meaning                              |
| ------------ | ------------------------------------ |
| `audio_stem` | Original filename without extension  |
| `task`       | `transcribe` or `translate`          |
| `timestamp`  | ISO 8601 format. Ends with 'Z' for UTC, or a UTC offset (e.g., `+0200`) for local/specific timezones. Format: `YYYYMMDDTHHMMSS[Z\|+HHMM\|-HHMM]` |
| `extension`  | `json`, `txt`, or `srt`              |

### Example

```txt
# Example with UTC (default)
interview_audio_translate_20250601T091234Z.txt

# Example with FILENAME_TIMEZONE=Europe/Amsterdam (assuming +02:00 offset)
interview_audio_translate_20250609T212928+0200.txt
```

Environment override for `TZ` (internally `APP_TIMEZONE`):

```bash
# TZ: Controls the timezone for timestamps in output filenames.
# This environment variable is read by the application and mapped to APP_TIMEZONE.
# Accepts:
#   "UTC" (default): Timestamps are in UTC, ending with 'Z'.
#   "local": Timestamps use the system's local timezone, ending with the local UTC offset (e.g., +0200).
#   IANA timezone string (e.g., "Europe/Amsterdam", "America/New_York"): Timestamps use the specified timezone,
#                          ending with its UTC offset.
TZ=Europe/Amsterdam
```

---

## Configuration System

### Configuration Files & Loading

The application uses a hierarchical approach for loading `.env` files, managed by [insanely_fast_whisper_api/utils/env_loader.py](./insanely_fast_whisper_api/utils/env_loader.py) and accessed via [insanely_fast_whisper_api/utils/constants.py](./insanely_fast_whisper_api/utils/constants.py).

1. **Project `.env`**: Located at the project root (e.g., `/path/to/project/.env`). This file can define project-specific defaults.
2. **User-specific `.env`**: Located at `~/.config/insanely-fast-whisper-api/.env`. This file is for user-specific overrides and sensitive information (like API keys).

**Loading Order & Override:**

- The project root `.env` is loaded first.
- The user-specific `~/.config/insanely-fast-whisper-api/.env` is loaded second and **will override** any variables previously set by the project `.env` or system environment variables.

**Key Configuration Files:**

- **[`.env.example`](./.env.example)**: A template file in the project root. Users should copy this to create their configuration files.
- **`~/.config/insanely-fast-whisper-api/.env`**: The primary user-specific configuration file. This is the recommended place for all user customizations.
- **Project `.env`** (Optional): Can be used for development-specific settings or non-sensitive project defaults.
- **[insanely_fast_whisper_api/utils/constants.py](./insanely_fast_whisper_api/utils/constants.py)**: Defines and provides centralized access to all configuration variables after they are loaded from the environment and `.env` files.
- **[insanely_fast_whisper_api/utils/env_loader.py](./insanely_fast_whisper_api/utils/env_loader.py)**: Contains the logic for loading `.env` files hierarchically and managing debug print statements based on `LOG_LEVEL` or CLI flags.
- **[logging_config.yaml](./insanely_fast_whisper_api/logging_config.yaml)**: Configures the application's logging behavior.

**User Configuration Setup Script:**

A utility script [scripts/setup_config.py](./scripts/setup_config.py) is provided to help users create their user-specific configuration file. It copies [`.env.example`](./.env.example) (located in the project root) to `~/.config/insanely-fast-whisper-api/.env`.

The script performs the following actions:

- Checks if [`.env.example`](./.env.example) exists in the project root.
- Creates the `~/.config/insanely-fast-whisper-api/` directory if it doesn't already exist.
- Copies the content of [`.env.example`](./.env.example) to `~/.config/insanely-fast-whisper-api/.env`.
- Prompts the user for confirmation if a configuration file already exists at the destination, to prevent accidental overwrites.
- Informs the user to edit the newly created or updated file to input their specific settings, such as `HUGGINGFACE_TOKEN` for gated models.

Refer to the [`.env.example`](./.env.example) file in the project root for a comprehensive list of all available configuration options and their descriptions (e.g., model settings, device selection, file handling parameters, timezone configuration).

Run it using PDM:

```bash
pdm run setup-config
```

Or directly:

```bash
python [scripts/setup_config.py](./scripts/setup_config.py)
```

**Important**: No direct `os.getenv()` calls should be made outside of [insanely_fast_whisper_api/utils/env_loader.py](./insanely_fast_whisper_api/utils/env_loader.py) or [insanely_fast_whisper_api/utils/constants.py](./insanely_fast_whisper_api/utils/constants.py) to ensure consistent configuration loading.

---

## Application Interfaces

This project provides multiple interfaces for interacting with the transcription and translation capabilities.

### API Server Details

The FastAPI server provides a robust and scalable way to integrate the speech recognition functionalities into other applications or services.

**Launch Options:**

You can start the API server with various options to customize its behavior:

```bash
# Launch with default settings (host: 0.0.0.0, port: 8888, workers: 1, log-level: info)
python -m [insanely_fast_whisper_api](./insanely_fast_whisper_api/__main__.py)

# See all available options and help
python -m [insanely_fast_whisper_api](./insanely_fast_whisper_api/__main__.py) --help

# Launch with a custom port
python -m [insanely_fast_whisper_api](./insanely_fast_whisper_api/__main__.py) --port 8001

# Launch with a custom host and port
python -m [insanely_fast_whisper_api](./insanely_fast_whisper_api/__main__.py) --host 127.0.0.1 --port 9000

# Launch with multiple workers (disables reload)
python -m [insanely_fast_whisper_api](./insanely_fast_whisper_api/__main__.py) --workers 4 --no-reload

# Launch with auto-reload enabled (for development)
python -m [insanely_fast_whisper_api](./insanely_fast_whisper_api/__main__.py) --reload

# Launch with a specific log level (e.g., debug)
python -m [insanely_fast_whisper_api](./insanely_fast_whisper_api/__main__.py) --log-level debug

# Launch in debug mode (enables debug logging for app and Uvicorn)
python -m [insanely_fast_whisper_api](./insanely_fast_whisper_api/__main__.py) --debug

# Launch with SSL (ensure dummy.key and dummy.crt exist or provide paths)
# python -m [insanely_fast_whisper_api](./insanely_fast_whisper_api/__main__.py) --ssl-keyfile dummy.key --ssl-certfile dummy.crt
```

**API Parameters:**

The API endpoints have distinct parameters. Core model settings (`model`, `device`, `batch_size`, etc.) are configured globally through environment variables and cannot be changed per-request.

- `/v1/audio/transcriptions`:
  - `file`: The audio file to transcribe (required).
  - `timestamp_type`: The granularity of the timestamps (`chunk` or `word`). If you provide `text` here, the response will be plain text instead of JSON. Defaults to `chunk`.
  - `language`: The language of the audio. If omitted, the model will auto-detect the language.
- `/v1/audio/translations`:
  - `file`: The audio file to translate (required).
  - `response_format`: The desired output format (`json` or `text`). Defaults to `json`.
  - `timestamp_type`: The granularity of the timestamps (`chunk` or `word`). Defaults to `chunk`.
  - `language`: The language of the audio. If omitted, the model will auto-detect the language.

### WebUI (Gradio Interface) Details

The Gradio WebUI offers an interactive, browser-based experience, particularly useful for batch processing multiple audio files.

**Launch Options:**

```bash
# Basic WebUI launch
python -m [insanely_fast_whisper_api.webui.cli](./insanely_fast_whisper_api/webui/cli.py)

# With debug logging (recommended for development or troubleshooting)
python -m [insanely_fast_whisper_api.webui.cli](./insanely_fast_whisper_api/webui/cli.py) --debug

# Custom host and port
python -m [insanely_fast_whisper_api.webui.cli](./insanely_fast_whisper_api/webui/cli.py) --port 7860 --host 0.0.0.0 --debug
```

### CLI (Command Line Interface) Details

The Command Line Interface is ideal for single-file processing, scripting, or quick tests.

**Command Examples and Options:**

```bash
# Transcribe audio file (basic)
python -m [insanely_fast_whisper_api.cli.cli](./insanely_fast_whisper_api/cli/cli.py) transcribe audio_file.mp3

# Transcribe with word-level timestamps (if supported by model/config)
# Note: Check if 'timestamp_type' or similar option is available via CLI help
# python -m [insanely_fast_whisper_api.cli.cli](./insanely_fast_whisper_api/cli/cli.py) transcribe audio_file.mp3 --timestamp-type word

# Transcribe without timestamps in the output (produces cleaner text if timestamps are not needed)
python -m [insanely_fast_whisper_api.cli.cli](./insanely_fast_whisper_api/cli/cli.py) transcribe audio_file.mp3 --no-timestamps

# Transcribe with debug logging enabled
python -m insanely_fast_whisper_api.cli.cli transcribe audio_file.mp3 --debug

# Translate audio to English
python -m insanely_fast_whisper_api.cli.cli translate audio_file.mp3
```

Consult `python -m insanely_fast_whisper_api.cli.cli --help` for a full list of commands and options.

---

---

## Dependency Management with PDM

This project uses [PDM (Python Development Master)](https://pdm-project.org/) for dependency management and package building, adhering to PEP 517, PEP 518, and PEP 621 standards. All project metadata, dependencies, and scripts are defined in the [`pyproject.toml`](./pyproject.toml) file.

### [`pyproject.toml`](./pyproject.toml) Structure

- **`[project]`**: Contains core project metadata such as name, version, authors, description, and classifiers.
  - **`dependencies`**: Lists core runtime dependencies required for the application to function.
  - **`optional-dependencies`**: Defines groups of dependencies that are not required for the core functionality but can be installed for specific purposes. Key groups include:
    - `dev`: Tools for development, such as linters (`black`, `isort`, `flake8`, `mypy`), testing frameworks (`pytest`, `pytest-cov`), and other utilities.
    - `rocm`: Dependencies specific to AMD ROCm GPU support, including the appropriate PyTorch build and ONNX runtime for ROCm.
    - `cpu`: Dependencies for CPU-only PyTorch execution.
    - `cuda`: Dependencies for NVIDIA CUDA GPU execution.
- **`[tool.pdm]`**: Configures PDM-specific settings.
  - **`scripts`**: Defines shortcuts for common commands (e.g., `lint`, `format`, `test`, `api`, `webui`, `cli`). These can be run using `pdm run <script_name>`.
  - **`dev-dependencies`**: PDM's way to specify development-only dependencies, often mirrored or managed via the `dev` group in `optional-dependencies` for broader compatibility.

### PDM Setup and Installation

1. **Install PDM**: If you don't have PDM, install it globally or per-user. A common method is:

    ```bash
    curl -sSL https://pdm-project.org/install-pdm.py | python3 -
    ```

    Follow the instructions to add PDM to your PATH.

2. **Install Project Dependencies**: Navigate to the project root directory and run:

    ```bash
    pdm install
    ```

    By default, this installs core dependencies. To include optional groups:

    ```bash
    # Install core + development tools
    pdm install -G dev

    # Install core + ROCm support
    pdm install -G rocm

    # Install core + development tools + ROCm support
    pdm install -G dev -G rocm
    ```

    PDM creates a `.venv` directory for the virtual environment and a `pdm.lock` file to ensure deterministic builds.

### Common PDM Commands

- **`pdm install`**: Install all dependencies as specified in `pdm.lock` (if it exists) or [`pyproject.toml`](./pyproject.toml).
  - `pdm install -G <group>`: Install dependencies from a specific optional group.
- **`pdm add <package>`**: Add a new dependency to [`pyproject.toml`](./pyproject.toml) and install it.
  - `pdm add -dG <group> <package>`: Add a package to a specific optional group.
- **`pdm remove <package>`**: Remove a dependency.
- **`pdm update`**: Update dependencies to their latest allowed versions according to [`pyproject.toml`](./pyproject.toml) and update `pdm.lock`.
- **`pdm run <script_name>`**: Execute a script defined in `[tool.pdm.scripts]` in [`pyproject.toml`](./pyproject.toml).
- **`pdm lock`**: Resolve dependencies and write to `pdm.lock` without installing.
- **`pdm shell`**: Activate the PDM-managed virtual environment in the current shell.

### Relationship with `requirements-*.txt` Files

While PDM manages dependencies through [`pyproject.toml`](./pyproject.toml), the `requirements-*.txt` files (e.g., `requirements.txt`, `requirements-rocm.txt`, `requirements-dev.txt`) are currently maintained primarily for Docker builds and specific environment setups where PDM might not be directly used for the build process itself, or for environments that predate full PDM integration.

Ideally, these `requirements.txt` files can be generated from `pdm.lock` using `pdm export` to ensure consistency:

```bash
# Export default dependencies
pdm export -o requirements.txt --without-hashes

# Export a specific group (e.g., rocm)
pdm export -G rocm -o requirements-rocm.txt --without-hashes

# Export development dependencies
pdm export -G dev -o requirements-dev.txt --without-hashes
```

This practice helps keep them synchronized with the PDM-managed dependencies.

> **PyTorch Note**: Due to PyTorch's specific index URL requirements for different compute platforms (CPU, CUDA, ROCm), its installation is carefully managed within PDM's dependency groups or via the `requirements-*.txt` files to ensure the correct version is fetched. PDM can handle custom source URLs if needed, which should be configured in [`pyproject.toml`](./pyproject.toml).

---

## API Endpoints

### `POST /v1/audio/transcriptions`

Transcribes audio to text.

- **`file`**: The audio file to transcribe (required).
- **`timestamp_type`**: Type of timestamp to generate (`chunk` or `word`). If set to `text`, the output is plain text instead of JSON. Default: `chunk`.
- **`language`**: Source language code (e.g., `en`). Auto-detects if not specified.

### `POST /v1/audio/translations`

Translates audio to English.

- **`file`**: The audio file to translate (required).
- **`response_format`**: Output format (`json` or `text`). Default: `json`.
- **`timestamp_type`**: Type of timestamp to generate (`chunk` or `word`). Default: `chunk`.
- **`language`**: Source language code (e.g., `en`). Auto-detects if not specified.

*Note: Key model parameters (model name, device, batch size, etc.) are configured globally via environment variables and are not modifiable per-request.*

---

## Error Handling

### Strategy

- Catch specific exceptions: `ValueError`, `HTTPException`, `RuntimeError`, etc.
- Use `from e` to preserve traceback
- Avoid broad `except Exception`

### Logging

- Full stack trace
- Log level: Critical / Error / Warning / Debug

### Layered Handling

- API → custom response formatting
- CLI → friendly messages
- WebUI → visual feedback

---

## Development Guidelines

### Code Style

- PEP8 + 88-char lines
- `black`, `isort`, `mypy`, `pylint`
- Type hints everywhere

### Testing

```bash
pytest tests/
```

### Code Quality Checks (Docker-based)

**Check Commands (No Changes):**

```bash
# Access the container
docker exec -it insanely-fast-whisper-rocm-api bash

# Check Black formatting (dry run)
black --check .

# Check isort import sorting (dry run)  
isort --check-only .

# Run mypy type checking
mypy insanely_fast_whisper_api/

# Run all checks together
black --check . && isort --check-only . && mypy insanely_fast_whisper_api/
```

**Auto-Fix Commands:**

```bash
# Access the container
docker exec -it insanely-fast-whisper-rocm-api bash

# Auto-format with Black
black .

# Auto-sort imports with isort
isort .

# Run auto-fixes together (Black + isort)
black . && isort .
```

---

## Deployment Options

### Local Development

**API Server (FastAPI):**

```bash
# Launch with default settings (host: 0.0.0.0, port: 8888, workers: 1, log-level: info)
python -m insanely_fast_whisper_api

# See all available options and help
python -m insanely_fast_whisper_api --help

# Launch with a custom port
python -m insanely_fast_whisper_api --port 8001

# Launch with a custom host and port
python -m insanely_fast_whisper_api --host 127.0.0.1 --port 9000

# Launch with multiple workers (disables reload)
python -m insanely_fast_whisper_api --workers 4 --no-reload

# Launch with auto-reload enabled (for development)
python -m insanely_fast_whisper_api --reload

# Launch with a specific log level (e.g., debug)
python -m insanely_fast_whisper_api --log-level debug

# Launch in debug mode (enables debug logging for app and Uvicorn)
python -m insanely_fast_whisper_api --debug

# Launch with SSL (ensure dummy.key and dummy.crt exist or provide paths)
# python -m insanely_fast_whisper_api --ssl-keyfile dummy.key --ssl-certfile dummy.crt
```

**WebUI (Gradio Interface):**

```bash
# Launch WebUI with debug logging
python -m insanely_fast_whisper_api.webui.cli --debug

# With custom host and port
python -m insanely_fast_whisper_api.webui.cli --port 7860 --host 0.0.0.0 --debug
```

**CLI (Command Line Interface):**

```bash
# Transcribe audio file
python -m insanely_fast_whisper_api.cli.cli transcribe audio_file.mp3

# Transcribe with options
python -m insanely_fast_whisper_api.cli.cli transcribe tests/conversion-test-file.mp3 --no-timestamps --debug

# Translate audio to English
python -m insanely_fast_whisper_api.cli.cli translate audio_file.mp3
```

### Docker Deployment

The project includes Docker configurations for both production and development environments, managed via Docker Compose.

- **Production (`docker-compose.yaml`)**: The [`docker-compose.yaml`](./docker-compose.yaml) file is optimized for production use. It builds a clean, minimal image and runs the application in a stable configuration. Use this for deployments or for running the application as a standalone service.

  ```bash
  # Build and run the production container
  docker compose up --build -d
  ```

- **Development (`docker-compose.dev.yaml`)**: The [`docker-compose.dev.yaml`](./docker-compose.dev.yaml) file is tailored for local development. It mounts the local source code into the container, enabling hot-reloading for immediate feedback on code changes. Use this file to work on the application without needing to rebuild the image for every change.

  ```bash
  # Build and run the development container
  docker compose -f docker-compose.dev.yaml up --build -d
  ```

**Access URLs:**

- WebUI: [http://localhost:7860](http://localhost:7860)
- API (when enabled): [http://localhost:8888/docs](http://localhost:8888/docs)

---

## Monitoring & Security

- Validates file uploads (type and size)
- Cleans up temp files after use
- Logs to stdout or file via YAML config
- Rate limiting and auth should be implemented in prod

---

## Import Standardization

All imports now follow **absolute import** conventions for improved IDE support and maintainability (v0.2.1+):

```python
# Good - Absolute imports
from insanely_fast_whisper_api.core.pipeline import WhisperPipeline
from insanely_fast_whisper_api.utils.constants import WHISPER_MODEL

# Deprecated - Relative imports (removed in v0.2.1)
# from .core.pipeline import WhisperPipeline
```

**Benefits:**

- Better IDE auto-completion and navigation
- Clearer dependency tracking
- Improved code maintainability
- Consistent import patterns across the codebase

*See [v0.2.1 changelog in VERSIONS.md](VERSIONS.md#v021---may-29-30-2025) for implementation details.*

---

## Recent Enhancements

*For complete feature details and changelog, see [VERSIONS.md](VERSIONS.md).*

### Multi-file WebUI Support (v0.3.1)

- Batch uploads with improved error handling
- ZIP downloads with TXT/SRT/JSON formats
- Real-time chunk-level progress tracking
- Fixed empty ZIP file bug

### Modular WebUI Architecture (v0.3.0)

- `ui.py`, `handlers.py`, `formatters.py`, `errors.py`
- Replaces monolithic `webui.py`
- CLI entrypoint for WebUI launches
- Configuration dataclasses

### Core Architecture Revolution (v0.2.0)

- Direct Hugging Face Transformers integration
- Dropped subprocess-based `insanely-fast-whisper` dependency
- Native `transformers.pipeline` support
- Performance optimizations with configurable batching

---

## Bug Fixes

*For complete bug fix details and changelog, see [VERSIONS.md](VERSIONS.md).*

### Fixed empty ZIP files (v0.3.1)

- **Issue**: WebUI ZIP downloads were missing transcription content
- **Root Cause**: `result_dict` was incorrectly accessed in `handlers.py`
- **Fix**: Corrected data structure access and improved error handling
- **Result**: Properly populated ZIP downloads with all transcription formats

### ✅ Audio Format Validation (v0.3.1)

- **Issue**: Deprecated audio extensions causing processing errors
- **Fix**: Updated supported format validation and removed legacy extensions
- **Result**: More reliable audio file processing

### ✅ Configuration Test Stability (v0.3.1)

- **Issue**: Inconsistent configuration test results
- **Fix**: Refactored centralized configuration tests for improved robustness
- **Result**: More reliable testing and validation

### ✅ WebUI Batch Download (v0.3.1)

- **Issue**: `TypeError` in Gradio `DownloadButton` when processing multiple files
- **Fix**: Ensured `value` parameter receives a file path instead of a function
- **Result**: Fixed `TypeError` in WebUI batch download functionality

---

## 📄 License

MIT License – see `LICENSE` file.

---

> 📌 For any changes, always update this file to reflect the current behavior of the codebase.
