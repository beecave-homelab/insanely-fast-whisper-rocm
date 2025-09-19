# Project Overview | Insanely Fast Whisper API (ROCm)

A comprehensive Whisper-based speech recognition toolkit designed specifically to provide **AMD GPU (ROCm v6.4.1) support** for high-performance Automatic Speech Recognition (ASR) and translation. This package extends the capabilities of the original [insanely-fast-whisper](https://github.com/Vaibhavs10/insanely-fast-whisper) by providing multiple interfaces, ROCm compatibility, and production-ready architecture.

> [!NOTE]
> This overview is the **single source of truth** for developers working on this codebase.

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://www.python.org)
[![Version](https://img.shields.io/badge/Version-v1.0.0-informational)](#version-summary)
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
- [Performance Benchmarking](#performance-benchmarking)
- [Dependency Management](#dependency-management-with-pdm)
- [Error Handling](#error-handling)
- [Development Guidelines](#development-guidelines)
- [Deployment Options](#deployment-options)
- [Monitoring & Security](#monitoring--security)
- [Import Standardization](#import-standardization)
- [Enhancement Highlights](#enhancement-highlights)
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
# Create your user configuration file (interactive)
pdm run setup-config  # or `python scripts/setup_config.py`

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

### 🏷️ **Current Version: v1.0.0** *(18-09-2025)*

**Latest improvements**: Fixed chunked transcription timestamp merging by offsetting segment/word timestamps per chunk start time. Introduced breaking API changes to `split_audio()` and `merge_chunk_results()`.

### 📊 **Release Overview**

| Version | Date | Type | Key Features |
|---------|------|------|--------------|
| **v1.0.0** | 18-09-2025 | 🔄 Major | Timestamp merge fix; Breaking API for `split_audio()` and `merge_chunk_results()` |
| **v0.10.0** | 23-07-2025 | ✨ Minor | M4A support & Stable-TS integration |
| v0.9.1 | 19-07-2025 | 🐛 Patch | Translation & model override fixes |
| **v0.8.0** | Jul 2025 | ✨ Minor | Entrypoint refactoring, CLI export formats, translation feature |
| **v0.7.0** | Jun 2025 | ✨ Minor | Major import refactor, `pdm` migration, modular CLI |
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
>
> **Note:** As of 2025-07-06, all release tags (v0.1.0 ... v0.9.0) have normalized commit mappings. The canonical mapping for each release is now found in VERSIONS.md under the 'Key Commits' section.

---

## Project Features

### Primary Focus: ROCm Support

- **AMD GPU (ROCm v6.4.1) Support**: First-class AMD GPU acceleration for Whisper models

- **Extended Original Package**: Builds upon [insanely-fast-whisper](https://github.com/Vaibhavs10/insanely-fast-whisper) with additional interfaces and ROCm compatibility
- **Production-Ready Architecture**: Beyond CLI-only approach of original package

### Core Capabilities

- **Transcription**: Audio to text in source language
- **Translation**: Audio to English
- **Native SDPA Acceleration**: Hugging Face `sdpa` attention implementation for faster processing on compatible hardware.
- **Word-level Timestamp Stabilization**: Optional integration with [`stable-ts`](https://github.com/jianfch/stable-ts). Enable via `--stabilize` (CLI) or corresponding API/WebUI options to obtain refined word-aligned segments.
- **Video & Audio Formats**: Support for standard audio files (.wav, .flac, .mp3, .m4a) **and** popular video containers (.mp4, .mkv, .webm, .mov) via automatic audio extraction with FFmpeg
- **Filename Standardization**: Predictable and configurable output naming

### Interface Options

- **FastAPI Server**: RESTful API with OpenAI-compatible v1 endpoints (`/v1/audio/transcriptions`, etc.)

- **Gradio WebUI**: Batch file upload, live progress tracking, ZIP downloads
- **CLI Interface**: Command-line tool for single-file processing
- **Model Management**: Automatic Hugging Face model downloading and caching
- **Docker Support**: Full containerization with development and production configurations
- **Direct Hugging Face Integration**: Native `transformers.pipeline` support
- **Configurable Processing**: Batch size, device, model selection
- **ROCm Integration**: Optimized PyTorch and ONNX runtime configurations for AMD GPUs
- **Native Attention Acceleration**: Uses `attn_implementation="sdpa"` for optimized performance without requiring `BetterTransformer`.

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
│   │   ├── [__main__.py](./insanely_fast_whisper_api/api/__main__.py)                  # API module entry
│   │   ├── [app.py](./insanely_fast_whisper_api/api/app.py)                      # FastAPI app setup
│   │   ├── [routes.py](./insanely_fast_whisper_api/api/routes.py)                   # API endpoints
│   │   ├── [models.py](./insanely_fast_whisper_api/api/models.py)                   # Pydantic data models
│   │   ├── [dependencies.py](./insanely_fast_whisper_api/api/dependencies.py)             # Dependency injection
│   │   ├── [middleware.py](./insanely_fast_whisper_api/api/middleware.py)               # Request/response middleware
│   │   └── [responses.py](./insanely_fast_whisper_api/api/responses.py)                # Response formatters
│   ├── [core/](./insanely_fast_whisper_api/core/)                           # Core ASR logic
│   │   ├── [__init__.py](./insanely_fast_whisper_api/core/__init__.py)
│   │   ├── [integrations/](./insanely_fast_whisper_api/core/integrations/)           # Integrations with other libs
│   │   │   ├── [__init__.py](./insanely_fast_whisper_api/core/integrations/__init__.py)
│   │   │   └── [stable_ts.py](./insanely_fast_whisper_api/core/integrations/stable_ts.py)      # stable-ts logic
│   │   ├── [pipeline.py](./insanely_fast_whisper_api/core/pipeline.py)                 # ASR orchestration
│   │   ├── [asr_backend.py](./insanely_fast_whisper_api/core/asr_backend.py)              # Whisper model backend
│   │   ├── [storage.py](./insanely_fast_whisper_api/core/storage.py)                  # File lifecycle management
│   │   ├── [utils.py](./insanely_fast_whisper_api/core/utils.py)                    # Core utilities
│   │   ├── [formatters.py](./insanely_fast_whisper_api/core/formatters.py)              # Output formatting logic
│   │   └── [errors.py](./insanely_fast_whisper_api/core/errors.py)                   # Exception classes
│   ├── [audio/](./insanely_fast_whisper_api/audio/)                          # Audio processing
│   │   ├── [__init__.py](./insanely_fast_whisper_api/audio/__init__.py)
│   │   ├── [conversion.py](./insanely_fast_whisper_api/audio/conversion.py)               # Audio conversion logic
│   │   ├── [processing.py](./insanely_fast_whisper_api/audio/processing.py)               # Validation and preprocessing
│   │   └── [results.py](./insanely_fast_whisper_api/audio/results.py)                  # Output formatting
│   ├── [cli/](./insanely_fast_whisper_api/cli/)                            # CLI tools
│   │   ├── [__init__.py](./insanely_fast_whisper_api/cli/__init__.py)
│   │   ├── [__main__.py](./insanely_fast_whisper_api/cli/__main__.py)                  # CLI module entry
│   │   ├── [cli.py](./insanely_fast_whisper_api/cli/cli.py)                      # CLI entry point
│   │   ├── [commands.py](./insanely_fast_whisper_api/cli/commands.py)                 # Subcommand logic
│   │   ├── [common_options.py](./insanely_fast_whisper_api/cli/common_options.py)         # Shared CLI options
│   │   └── [facade.py](./insanely_fast_whisper_api/cli/facade.py)                   # High-level CLI wrapper
│   ├── [webui/](./insanely_fast_whisper_api/webui/)                          # Web UI (Gradio)
│   │   ├── [__init__.py](./insanely_fast_whisper_api/webui/__init__.py)
│   │   ├── [__main__.py](./insanely_fast_whisper_api/webui/__main__.py)                  # WebUI module entry
│   │   ├── [app.py](./insanely_fast_whisper_api/webui/app.py)                      # Gradio App launcher
│   │   ├── [ui.py](./insanely_fast_whisper_api/webui/ui.py)                       # Gradio interface
│   │   ├── [handlers.py](./insanely_fast_whisper_api/webui/handlers.py)                 # Upload + result management
│   │   ├── [merge_handler.py](./insanely_fast_whisper_api/webui/merge_handler.py)            # Transcription file merge handlers
│   │   ├── [utils.py](./insanely_fast_whisper_api/webui/utils.py)                    # WebUI utilities
│   │   └── [errors.py](./insanely_fast_whisper_api/webui/errors.py)                   # UI-specific exceptions
│   └── [utils/](./insanely_fast_whisper_api/utils/)                          # General utilities
│       ├── [__init__.py](./insanely_fast_whisper_api/utils/__init__.py)
│       ├── [benchmark.py](./insanely_fast_whisper_api/utils/benchmark.py)                # Benchmarking utilities
│       ├── [constants.py](./insanely_fast_whisper_api/utils/constants.py)                # Core environment variable definitions
│       ├── [env_loader.py](./insanely_fast_whisper_api/utils/env_loader.py)               # Hierarchical .env loading & debug print logic
│       ├── [download_hf_model.py](./insanely_fast_whisper_api/utils/download_hf_model.py)        # Model downloading & caching
│       ├── [file_utils.py](./insanely_fast_whisper_api/utils/file_utils.py)               # File operations
│       ├── [filename_generator.py](./insanely_fast_whisper_api/utils/filename_generator.py)       # Unified filename logic
│       └── [format_time.py](./insanely_fast_whisper_api/utils/format_time.py)              # Time formatting utilities
├── [scripts/](./scripts/)                            # Utility and maintenance scripts
│   └── [setup_config.py](./scripts/setup_config.py)               # Script to set up user-specific .env file
```

---

## Architecture Highlights

### Native SDPA Acceleration vs. BetterTransformer

This project uses the native Scaled Dot Product Attention (SDPA) available in PyTorch 2.0+ and `transformers` as its primary method for accelerating the Whisper model's attention mechanism. This is achieved by setting `attn_implementation="sdpa"` when loading the model.

**Why SDPA?**

- **Modern Standard**: SDPA is the official, built-in acceleration method in PyTorch. It is the successor to older, patch-based approaches like Hugging Face's `BetterTransformer`.
- **Performance**: It provides significant speed improvements for attention-heavy models like Whisper, often matching or exceeding the performance of `BetterTransformer`.
- **Simplicity**: As a native feature, it doesn't require an extra dependency like `optimum` or manual model patching (`BetterTransformer.transform(model)`). Integration is cleaner and more robust.

The codebase automatically enables `sdpa` for any GPU-based device (`cuda`, `mps`) and disables it for CPU, ensuring optimal performance where available without manual configuration. `sdpa` is the current and recommended acceleration method.

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

### Enhancement Highlights

| Date     | ID                                     | Type | Description                                                                                             |
|----------|----------------------------------------|------|---------------------------------------------------------------------------------------------------------|
| Jun 2025 | [#5](https://github.com/issue/5)       | ✨    | **Native SDPA Acceleration**: Integrated `attn_implementation="sdpa"` for faster attention, replacing the need for BetterTransformer. |
| Jun 2025 | [#4](https://github.com/issue/4)       | ✨    | **Modular CLI**: Refactored the CLI into a modular structure with `click` commands and a `facade` for cleaner logic. |
| Jun 2025 | [#3](https://github.com/issue/3)       | 🔧    | **`pdm` Migration**: Replaced `requirements.txt` with `pdm` for robust dependency management. See [Dependency Management](#dependency-management-with-pdm). |
| Jun 2025 | [#2](https://github.com/issue/2)       | 🔧    | **Import Refactor**: Standardized all imports to be absolute, improving clarity and maintainability. See [Import Standardization](#import-standardization). |
| Jun 2025 | [#1](https://github.com/issue/1)       | 🐛    | **WebUI Stability**: Fixed issues with multi-file downloads and improved error handling in the Gradio interface. |

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

# Example with APP_TIMEZONE=Europe/Amsterdam (assuming +02:00 offset)
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

- **[setup script](./scripts/setup_config.py)**: A template file in the project root. Users should copy this to create their configuration files.
- **`~/.config/insanely-fast-whisper-api/.env`**: The primary user-specific configuration file. This is the recommended place for all user customizations.
- **Project `.env`** (Optional): Can be used for development-specific settings or non-sensitive project defaults.
- **[insanely_fast_whisper_api/utils/constants.py](./insanely_fast_whisper_api/utils/constants.py)**: Defines and provides centralized access to all configuration variables after they are loaded from the environment and `.env` files.
- **[insanely_fast_whisper_api/utils/env_loader.py](./insanely_fast_whisper_api/utils/env_loader.py)**: Contains the logic for loading `.env` files hierarchically and managing debug print statements based on `LOG_LEVEL` or CLI flags.
- **[logging_config.yaml](./insanely_fast_whisper_api/logging_config.yaml)**: Configures the application's logging behavior.

**User Configuration Setup Script:**

A utility script [`scripts/setup_config.py`](./scripts/setup_config.py) is provided to help users create their user-specific configuration file. It copies the project's `.env.example` file to `~/.config/insanely-fast-whisper-api/.env`.

The script performs the following actions:

- Checks if `.env.example` exists in the project root.
- Creates the `~/.config/insanely-fast-whisper-api/` directory if it doesn't already exist.
- Copies `.env.example` to `~/.config/insanely-fast-whisper-api/.env`.
- Prompts the user for confirmation if a configuration file already exists at the destination, to prevent accidental overwrites.
- Informs the user to edit the newly created file to input their specific settings, such as `HUGGINGFACE_TOKEN` for gated models.

Refer to the `.env.example` file in the project root for a comprehensive list of all available configuration options and their descriptions (e.g., model settings, device selection, file handling parameters, timezone configuration).

Run it using PDM:

```bash
pdm run setup-config
```

Or directly:

```bash
python scripts/setup_config.py
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
# Launch with default settings (http://0.0.0.0:8000, port: 8000, workers: 1, log-level: info)
python -m insanely_fast_whisper_api.api

# See all available options and help
python -m insanely_fast_whisper_api.api --help

# Launch with a custom port
python -m insanely_fast_whisper_api.api --port 8001

# Launch with a custom host and port
python -m insanely_fast_whisper_api.api --host 127.0.0.1 --port 9000

# Launch with multiple workers (disables reload)
python -m insanely_fast_whisper_api.api --workers 4 --no-reload

# Launch with auto-reload enabled (for development)
python -m insanely_fast_whisper_api.api --reload

# Launch with a specific log level (e.g., debug)
python -m insanely_fast_whisper_api.api --log-level debug

# Launch in debug mode (enables debug logging for app and Uvicorn)
python -m insanely_fast_whisper_api.api --debug

# Launch with SSL (ensure dummy.key and dummy.crt exist or provide paths)
# python -m insanely_fast_whisper_api.api --ssl-keyfile dummy.key --ssl-certfile dummy.crt
```

**API Parameters:**

The API endpoints have distinct parameters. Core model settings (`model`, `device`, `batch_size`, etc.) are configured globally through environment variables and cannot be changed per-request.

- `/v1/audio/transcriptions`:
  - `file`: The audio file to transcribe (required).
  - `timestamp_type`: The granularity of the timestamps (`chunk` or `word`). If you provide `text` here, the response will be plain text instead of JSON. Defaults to `chunk`.
  - `language`: The language of the audio. If omitted, the model will auto-detect the language.
  - `stabilize`: `bool` - Enable timestamp stabilization using `stable-ts`. Defaults to `False`.
  - `demucs`: `bool` - Enable Demucs noise reduction before transcription. Defaults to `False`.
  - `vad`: `bool` - Enable Silero VAD to filter out silent parts of the audio. Defaults to `False`.
  - `vad_threshold`: `float` - The threshold for VAD. Defaults to `0.35`.
- `/v1/audio/translations`:
  - `file`: The audio file to translate (required).
  - `response_format`: The desired output format (`json` or `text`). Defaults to `json`.
  - `stabilize`: `bool` - Enable timestamp stabilization using `stable-ts`. Defaults to `False`.
  - `demucs`: `bool` - Enable Demucs noise reduction before transcription. Defaults to `False`.
  - `vad`: `bool` - Enable Silero VAD to filter out silent parts of the audio. Defaults to `False`.
  - `vad_threshold`: `float` - The threshold for VAD. Defaults to `0.35`.
  - `timestamp_type`: The granularity of the timestamps (`chunk` or `word`). Defaults to `chunk`.
  - `language`: The language of the audio. If omitted, the model will auto-detect the language.

### WebUI (Gradio Interface) Details

The Gradio WebUI offers an interactive, browser-based experience—ideal for batch processing multiple audio/video files—and now **parity with the CLI for advanced audio-preprocessing features**:

- **Word-level timestamp stabilization** (`--stabilize`, powered by stable-ts)
- **Demucs noise reduction** (`--demucs`)
- **Voice Activity Detection** (`--vad`, with adjustable `--vad-threshold`)

**Launch Options:**

```bash
# Basic WebUI launch
python -m insanely_fast_whisper_api.webui

# With debug logging (recommended for development or troubleshooting)
python -m insanely_fast_whisper_api.webui --debug

# Custom host and port
python -m insanely_fast_whisper_api.webui --port 7860 --host 0.0.0.0 --debug
```

### CLI (Command Line Interface) Details

#### Internal Refactor (v0.9.2)

The CLI codebase was streamlined to eliminate hundreds of lines of duplicated `click` option declarations.

- **`cli/common_options.py`** now exposes an `audio_options` decorator that injects all shared flags (model, device, batch-size, language, export settings, etc.) into any command.
- **`cli/commands.py`** was rewritten so that `transcribe` and `translate` are *thin wrappers*:

  ```python
  @click.command(short_help="Transcribe an audio file")
  @audio_options
  def transcribe(audio_file: Path, **kwargs):
      _run_task(task="transcribe", audio_file=audio_file, **kwargs)
  ```

- The business logic lives in `_run_task`, and helpers such as `_handle_output_and_benchmarks` keep the file organized.

- New flag: `--stabilize` (timestamp post-processing via stable-ts).

Nothing changes for end-users — option names and behaviour remain the same — but the code is far easier to maintain and extend.

#### Performance Benchmarking

Use the `--benchmark` flag to measure processing speed and collect hardware stats.

> **Install dependencies**: Benchmarking relies on optional packages (`psutil`, `pyamdgpuinfo`, etc.). Install them with:
>
> ```bash
> pdm install -G bench
> ```

```bash
# Quick benchmark (JSON only; timestamps auto-disabled)
python -m insanely_fast_whisper_api.cli transcribe audio.mp3 --benchmark

# Benchmark and also export transcript as TXT
python -m insanely_fast_whisper_api.cli transcribe audio.mp3 --benchmark --export-format txt

# Pass arbitrary metadata
python -m insanely_fast_whisper_api.cli transcribe audio.mp3 --benchmark --benchmark-extra precision=fp16 tokenizer=fast
```

**Key behaviors**:

| Feature | Description |
|---------|-------------|
| Transcript export suppression | When `--benchmark` is set, transcript files are *not* saved unless you explicitly provide `--export-format`. |
| Auto `--no-timestamps` | Timestamps are disabled by default during benchmarking to measure raw model speed. Override by adding `--no-timestamps` on the CLI. |
| Output location | A JSON file is written to `benchmarks/` with name pattern `benchmark_<audio>_<task>_<timestamp>.json`. |
| Extra metadata | Use repeated `--benchmark-extra key=value` pairs to inject custom fields into the JSON. |
| Completion message | The benchmark path is printed **at the end** of the CLI output (📈 line) for quick copy-paste. |

The JSON includes runtime stats, total elapsed time, system info (OS, Python, Torch version), and GPU metrics (VRAM, temperature, power for AMD/CUDA when available).

---

The Command Line Interface is ideal for single-file processing, scripting, or quick tests. It supports multiple output formats and provides clear feedback on the transcription process.

#### Timestamp Stabilization (`--stabilize`)

Use `--stabilize` to refine timestamps with the [stable-ts](https://github.com/jianfch/stable-ts) library. When the flag is supplied, the CLI will post-process Whisper results via `stable_whisper.transcribe_any`, producing more reliable **word-level** timestamps while keeping chunk timings intact. The option works for both `transcribe` and `translate` commands and can be combined with any export or benchmarking settings.

#### Export Formats

The `--export-format` option controls the output file type. The following formats are available:

| Format | Description                                       | Output Directory      |
|--------|---------------------------------------------------|-----------------------|
| `json` | (Default) Standard JSON output with transcription | `transcripts/`        |
| `txt`  | Plain text                                        | `transcripts-txt/`    |
| `srt`  | SubRip subtitle format (requires timestamps)      | `transcripts-srt/`    |
| `all`  | Exports all three formats simultaneously          | (Respective above)    |

#### Command Examples and Options

```bash
# Transcribe and get a JSON file (default)
python -m insanely_fast_whisper_api.cli transcribe audio_file.mp3

# Transcribe and get a TXT file
python -m insanely_fast_whisper_api.cli transcribe audio_file.mp3 --export-format txt

# Transcribe and get all formats (JSON, SRT, TXT)
python -m insanely_fast_whisper_api.cli transcribe audio_file.mp3 --export-format all

# Translate and get an SRT file
python -m insanely_fast_whisper_api.cli translate audio_file.mp3 --export-format srt

# Transcribe with debug logging enabled
python -m insanely_fast_whisper_api.cli transcribe audio_file.mp3 --debug
```

Consult `python -m insanely_fast_whisper_api.cli --help` for a full list of commands and options.

---

#### Quiet Mode (`--quiet`)

Use `--quiet` to minimize console output. In quiet mode, the CLI shows only:

- The Rich progress bar during processing (when attached to a TTY).
- The final lines indicating where files were saved (for example, `💾 Saved TXT to: …`).

All intermediate INFO logs and auxiliary messages are suppressed. When
timestamp stabilization is enabled (via `--stabilize`) and optional post-
processing is active (Demucs/VAD), quiet mode also suppresses third-party
progress bars and HIP/MIOpen warnings emitted by underlying libraries.

Notes:

- On non-TTY output (pipes/redirects), the Rich progress bar is disabled by
  design; quiet mode will still emit the final saved-path line(s).
- `--debug` overrides quiet suppression and will emit detailed logs.

Example:

```bash
# Only show the progress bar and the final saved path(s)
python -m insanely_fast_whisper_api.cli transcribe audio.mp3 \
  --model openai/whisper-small \
  --batch-size 12 \
  --export-format txt \
  --quiet
```

---

## Dependency Management with PDM

This project uses [PDM (Python Development Master)](https://pdm-project.org/) for dependency management and package building, adhering to PEP 517, PEP 518, and PEP 621 standards. All project metadata, dependencies, and scripts are defined in the [`pyproject.toml`](./pyproject.toml) file.

### [`pyproject.toml`](./pyproject.toml) Structure

- **`[project]`**: Contains core project metadata such as name, version, authors, description, and classifiers.
  - **`dependencies`**: Lists core runtime dependencies required for the application to function.
  - **`optional-dependencies`**: Defines groups of dependencies that are not required for the core functionality but can be installed for specific purposes. Key groups include:
    - `dev`: Tools for development, such as linters (`black`, `isort`, `flake8`, `mypy`), testing frameworks (`pytest`, `pytest-cov`), and other utilities.
    - `rocm`: Dependencies specific to AMD ROCm GPU support, including the appropriate PyTorch build and ONNX runtime for ROCm.
    - `bench-torch-2_0_1`, `bench-torch-2_3_0`, `bench-torch-2_4_1`, `bench-torch-2_5_1`, `bench-torch-2_6_0`: Benchmarking groups for ROCm, each pinning a specific torch version (from the ROCm wheels source) and including `pyamdgpuinfo` for GPU metrics.
    - `cpu`: Dependencies for CPU-only PyTorch execution.
    - `cuda`: Dependencies for NVIDIA CUDA GPU execution.

#### Benchmarking with Multiple ROCm Torch Versions

To facilitate benchmarking across different ROCm-compatible PyTorch versions, the following optional dependency groups are available:

- `bench-torch-2-0-1`
- `bench-torch-2-3-0`
- `bench-torch-2-4-1`
- `bench-torch-2-5-1`
- `bench-torch-2-6-0`

Each group includes:

- The corresponding `torch` version from the ROCm wheels source
- `pyamdgpuinfo` for collecting GPU metrics during benchmarking

**Install a specific benchmarking group:**

```bash
pdm install -G bench-torch-2_3_0
```

**Example**: Benchmarking CLI with a specific torch version

```bash
pdm install -G bench-torch-2_4_1
python -m insanely_fast_whisper_api.cli transcribe audio.mp3 --benchmark
```

This allows reproducible benchmarking and easy switching between supported ROCm torch versions for performance comparison.

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

The Docker build now uses PDM to install all project dependencies directly from [`pyproject.toml`](./pyproject.toml) via `pdm install --prod`. The `requirements-*.txt` files (e.g., `requirements.txt`, `requirements-rocm.txt`, `requirements-dev.txt`) are maintained only for legacy or special environments where PDM is not available, but are no longer used in the Docker build.

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

Unit & API test suite:

```bash
pytest tests/
```

#### WebUI integration tests (Gradio)

These tests target the Gradio WebUI using `gradio_client`.

```bash
# Only run WebUI tests (marked `webui`)
pytest -m webui
```

Details:

- Requires `gradio-client>=0.7.0` (already part of the core deps).
- Session-scoped fixture `webui_server` (see `tests/conftest.py`) launches the WebUI once on port 7861 with the tiny Whisper model for speed.
- Tests auto-skip when the sample media files are absent.
- Custom marker `webui` is registered via `pytest.ini`:

```ini
[pytest]
markers =
    webui: integration tests that spin up the Gradio WebUI
```

Average runtime < 10 s on a laptop-class GPU.

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
# Launch with default settings (http://0.0.0.0:8000, port: 8000, workers: 1, log-level: info)
python -m insanely_fast_whisper_api.api

# See all available options and help
python -m insanely_fast_whisper_api.api --help

# Launch with a custom port
python -m insanely_fast_whisper_api.api --port 8001

# Launch with a custom host and port
python -m insanely_fast_whisper_api.api --host 127.0.0.1 --port 9000

# Launch with multiple workers (disables reload)
python -m insanely_fast_whisper_api.api --workers 4 --no-reload

# Launch with auto-reload enabled (for development)
python -m insanely_fast_whisper_api.api --reload

# Launch with a specific log level (e.g., debug)
python -m insanely_fast_whisper_api.api --log-level debug

# Launch in debug mode (enables debug logging for app and Uvicorn)
python -m insanely_fast_whisper_api.api --debug

# Launch with SSL (ensure dummy.key and dummy.crt exist or provide paths)
# python -m insanely_fast_whisper_api.api --ssl-keyfile dummy.key --ssl-certfile dummy.crt
```

**WebUI (Gradio Interface):**

The WebUI file uploader now accepts both audio **and** video files (`.wav`, `.flac`, `.mp3`, `.mp4`, `.mkv`, `.webm`, `.mov`). Video inputs are automatically converted to audio via FFmpeg before transcription—no extra flags required.

```bash
# Launch WebUI with debug logging
python -m insanely_fast_whisper_api.webui --debug

# With custom host and port
python -m insanely_fast_whisper_api.webui --port 7860 --host 0.0.0.0 --debug
```

**CLI (Command Line Interface):**

```bash
# Transcribe audio file
python -m insanely_fast_whisper_api.cli transcribe audio_file.mp3

# Transcribe with word-level stabilization
python -m insanely_fast_whisper_api.cli transcribe tests/conversion-test-file.mp3 --stabilize

# Transcribe with options
python -m insanely_fast_whisper_api.cli transcribe tests/conversion-test-file.mp3 --no-timestamps --debug

# Translate audio to English
python -m insanely_fast_whisper_api.cli translate audio_file.mp3
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
- API (when enabled): [http://localhost:8000/docs](http://localhost:8000/docs)

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

## Enhancement Timeline

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

### Native SDPA Acceleration (v0.7.0)

- Integrated `attn_implementation="sdpa"` for faster attention, replacing the need for BetterTransformer.

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

### ✅ Native SDPA Acceleration (v0.7.0)

- Integrated `attn_implementation="sdpa"` for faster attention, replacing the need for BetterTransformer.

---

## 📄 License

MIT License – see `LICENSE` file.

---

**Always update this file when code or configuration changes.**
