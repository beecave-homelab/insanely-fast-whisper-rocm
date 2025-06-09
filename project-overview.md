# Project Overview | Insanely Fast Whisper API (ROCm)

A comprehensive Whisper-based speech recognition toolkit designed specifically to provide **AMD GPU (ROCm v6.1) support** for high-performance Automatic Speech Recognition (ASR) and translation. This package extends the capabilities of the original [insanely-fast-whisper](https://github.com/Vaibhavs10/insanely-fast-whisper) by providing multiple interfaces, ROCm compatibility, and production-ready architecture. This overview is the **single source of truth** for developers working on this codebase.

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://www.python.org/)
[![Version](https://img.shields.io/badge/Version-v0.4.1-informational)](#version-summary)
[![API](https://img.shields.io/badge/API-FastAPI-green)](#api-endpoints)
[![CLI](https://img.shields.io/badge/CLI-Click-yellow)](#cli-tools-cli)
[![WebUI](https://img.shields.io/badge/WebUI-Gradio-orange)](#web-ui-webui)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

---

## Table of Contents

- [Quickstart for Developers](#quickstart-for-developers)
- [Version Summary](#version-summary)
- [Project Features](#project-features)
- [Project Structure](#project-structure)
- [Architecture Highlights](#architecture-highlights)
- [Filename Conventions](#filename-conventions)
- [Configuration System](#configuration-system)
- [Dependency Management](#dependency-management)
- [API Endpoints](#api-endpoints)
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
git clone https://git.beecave-homelab.com/lowie/insanely-fast-whisper-api.git
cd insanely-fast-whisper-api
```

**Use Docker (recommended):**

```bash
cp .env.example .env
# Edit .env as needed

docker compose up --build -d
```

**Or run locally with Python (for development):**

```bash
# Install and run
python3.10 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-rocm.txt
pip install -r requirements-onnxruntime-rocm.txt

# Choose your interface:
python -m insanely_fast_whisper_api              # API Server
python -m insanely_fast_whisper_api.webui.cli    # WebUI Interface  
python -m insanely_fast_whisper_api.cli.cli transcribe audio.mp3  # CLI
```

---

## Version Summary

### 🏷️ **Current Version: v0.4.1** *(June 2025)*

**Latest improvements**: Fixed Gradio DownloadButton TypeError and ZIP archive overwrite issues.

### 📊 **Release Overview**

| Version | Date | Type | Key Features |
|---------|------|------|--------------|
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

### Core Logic (`core/`)

- `pipeline.py`: ASR orchestration
- `asr_backend.py`: Whisper model backend
- `storage.py`: File lifecycle management
- `utils.py`: General utilities
- `errors.py`: Exception classes

### Audio Processing (`audio/`)

- `processing.py`: Validation and preprocessing
- `results.py`: Output formatting and handling

### API Layer (`api/`)

- `app.py`: FastAPI setup and startup
- `routes.py`: Endpoint definitions
- `models.py`: Pydantic schemas
- `middleware.py`: Request/response middleware
- `dependencies.py`: Dependency injection
- `responses.py`: Response formatters

### CLI Tools (`cli/`)

- `cli.py`: CLI entry point
- `commands.py`: Subcommand logic
- `facade.py`: High-level CLI wrapper

### Web UI (`webui/`)

- `cli.py`: WebUI CLI entry point
- `ui.py`: Gradio interface
- `handlers.py`: Upload + result management
- `formatters.py`: Export formats (TXT, JSON, SRT)
- `utils.py`: WebUI utilities
- `errors.py`: UI-specific exceptions
- `downloads/`: ZIP and file merging logic for downloads
  - `zip_creator.py`: Builder for creating ZIP archives
  - `merge_handler.py`: Handlers for merging transcription files

### Utilities (`utils/`)

- `constants.py`: Env var loading + default handling
- `download_hf_model.py`: Model downloading + caching
- `file_utils.py`: File operations
- `filename_generator.py`: Unified filename logic

---

## Architecture Highlights

### Core Refactor (v0.2.0+)

- Direct integration with Hugging Face `pipeline`
- No subprocess dependency on `insanely-fast-whisper`
- Modular architecture: `pipeline.py`, `asr_backend.py`, etc.
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
| `timestamp`  | ISO 8601 format (`YYYYMMDDTHHMMSSZ`) |
| `extension`  | `json`, `txt`, or `srt`              |

### Example

```txt
interview_audio_translate_20250601T091234Z.txt
```

Environment override:

```bash
FILENAME_TIMEZONE=Europe/Amsterdam  # or UTC (default)
```

---

## Configuration System

### Centralized via `utils/constants.py`

All environment variables are defined and loaded from here using `dotenv`.

### Benefits

- Consistent defaults and types
- Type-safe: all values converted properly (bool/int/float/str)
- `.env` supported in project root and user config

### Common Vars

- `WHISPER_MODEL`: Model name (e.g. `openai/whisper-large-v3`)
- `WHISPER_DEVICE`: e.g., `cpu`, `0`, etc.
- `WHISPER_BATCH_SIZE`: Parallel segment processing
- `SAVE_TRANSCRIPTIONS`: Save outputs to disk
- `FILENAME_TIMEZONE`: UTC or a timezone name

No `os.getenv()` calls outside `constants.py`.

---

## Dependency Management

### `pyproject.toml` with extras

- `[project.dependencies]`: Core runtime
- `[project.optional-dependencies]`: `dev`, `torch-cpu`, `torch-rocm`, etc.

### `requirements-*.txt` Files

- `requirements.txt`: Base runtime
- `requirements-dev.txt`: Linters, pytest, etc.
- `requirements-rocm.txt`: ROCm-only PyTorch
- `requirements-onnxruntime-rocm.txt`: Optional ROCm ONNX

> **PyTorch note:** Not in `pyproject.toml` due to index URL requirements

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
# Direct module execution
python -m insanely_fast_whisper_api

# With verbose logging
python -m insanely_fast_whisper_api -v

# Using uvicorn directly
uvicorn insanely_fast_whisper_api.main:app --host 0.0.0.0 --port 8888
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

### Docker Compose

```bash
docker compose up --build -d
```

**Default behavior**: Launches WebUI with debug logging on port 7860

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
