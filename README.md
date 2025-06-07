# Insanely Fast Whisper API (ROCm)

A comprehensive Whisper-based speech recognition toolkit designed specifically to provide **AMD GPU (ROCm) support** for high-performance audio transcription and translation. This package extends the capabilities of the original [insanely-fast-whisper](https://github.com/Vaibhavs10/insanely-fast-whisper) by providing multiple interfaces and ROCm compatibility.

## 🚀 What's Included

- **🔌 FastAPI Server**: RESTful API with OpenAI-compatible endpoints
- **🎛️ Gradio WebUI**: Web-based interface for batch file processing with live progress tracking
- **⚡ CLI Tools**: Command-line interface for single-file processing
- **📦 Model Management**: Automatic Hugging Face model downloading and caching
- **🏗️ Docker Support**: Full containerization with development and production configurations
- **🎯 ROCm Integration**: AMD GPU (ROCm v6.1) support for accelerated inference

## 🎯 Key Features

- **AMD GPU (ROCm) Support**: Primary focus on enabling Whisper models on AMD GPUs
- **Multiple Interfaces**: Choose between API, WebUI, or CLI based on your workflow
- **Batch Processing**: Handle multiple audio files simultaneously via WebUI
- **High Performance**: Optimized processing with configurable batch sizes and model parameters
- **Multiple Output Formats**: Support for JSON, TXT, and SRT subtitle formats
- **Standardized Filenames**: Consistent, timestamped output naming across all interfaces

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://www.python.org/)
[![Version](https://img.shields.io/badge/Version-v0.4.0-informational)](#version-summary)
[![API](https://img.shields.io/badge/API-FastAPI-green)](#api-endpoints)
[![CLI](https://img.shields.io/badge/CLI-Click-yellow)](#cli-tools-cli)
[![WebUI](https://img.shields.io/badge/WebUI-Gradio-orange)](#web-ui-webui)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

---

## 📚 Table of Contents

- [Features](#features)
- [Installation](#installation)
  - [Model Download](#model-download)
- [Configuration](#configuration)
- [Usage](#usage)
  - [API Server](#api-server)
  - [WebUI (Gradio Interface)](#webui-gradio-interface)
  - [CLI (Command Line Interface)](#cli-command-line-interface)
  - [Docker Usage](#docker-usage)
  - [Output Files and Filename Conventions](#output-files-and-filename-conventions)
  - [API Parameters](#api-parameters)
- [Project Structure](#project-structure)
- [Development](#development)
  - [Setting Up Development Environment](#setting-up-development-environment)
  - [Code Style](#code-style)
  - [Running Tests](#running-tests)
- [License](#license)
- [Contributing](#contributing)

## 🌟 Additional Features

- **Multi-format Audio Support**: Process various audio file formats seamlessly (.wav, .flac and .mp3)
- **OpenAI-Compatible API**: Drop-in replacement for OpenAI's audio endpoints (v1)
- **Environment-based Configuration**: Flexible configuration via `.env` files.
- **Real-time Progress**: Live progress tracking in WebUI for batch operations
- **ZIP Downloads**: Bundle multiple transcription formats for easy download
- **Robust Error Handling**: Comprehensive error management across all interfaces
- **Docker-first Deployment**: Production-ready containerization

## 🎯 Why This Package?

This package was created to address the lack of **AMD GPU (ROCm) support** in the original [insanely-fast-whisper](https://github.com/Vaibhavs10/insanely-fast-whisper) package. While the original focuses on NVIDIA CUDA and CPU inference, this package provides:

- **Native ROCm Support**: Optimized PyTorch and ONNX runtime configurations for AMD GPUs
- **Extended Interface Options**: Beyond the CLI-only approach of the original package
- **Production-Ready Architecture**: Modular design with proper error handling and logging
- **Batch Processing Capabilities**: Handle multiple files efficiently through the WebUI
- **Docker-first Deployment**: Easy setup and deployment with Docker Compose

Whether you're using AMD GPUs or need the additional interfaces (API, WebUI), this package provides a comprehensive solution for Whisper-based Automatic Speech Recognition (ASR).

## Installation

### Recommended: Docker Compose (Production & Development)

The recommended way to run the application is using Docker Compose:

1. Clone the repository:

    ```bash
    git clone https://github.com/beecave-homelab/insanely-fast-whisper-rocm-api.git
    cd insanely-fast-whisper-rocm-api
    ```

2. Set up configuration:

    ```bash
    cp .env.example .env
    # Edit .env as needed
    ```

3. Start the application:

    ```bash
    docker compose up --build -d
    ```

### Alternative: Python Virtual Environment (Development)

For local development, you can also install and run the application in a Python 3.10+ virtual environment:

1. Clone the repository:

    ```bash
    git clone https://github.com/beecave-homelab/insanely-fast-whisper-rocm-api.git
    cd insanely-fast-whisper-rocm-api
    ```

2. Create and activate virtual environment:

    ```bash
    python3.10 -m venv venv && source venv/bin/activate
    ```

3. Install dependencies:

    ```bash
    # Install base dependencies
    pip install -r requirements.txt
    
    # Install AMD GPU (ROCm) support
    pip install -r requirements-rocm.txt
    pip install -r requirements-onnxruntime-rocm.txt
    ```

    > **Important**: This application is specifically designed to provide **AMD GPU (ROCm) support** for Whisper models. The ROCm requirements files (`requirements-rocm.txt`, `requirements-onnxruntime-rocm.txt`) ensure proper PyTorch and ONNX runtime installation for AMD GPUs. While it also works on CPU and NVIDIA GPUs, ROCm support was the primary motivation for this package.

### Model Download

The application will automatically download the specified Whisper model on first use. You can also pre-download models using the included script (add these commands to the `docker-compose.yaml` file to run them on startup):

```bash
# Download the default model (specified in .env or WHISPER_MODEL env var)
python -m insanely_fast_whisper_api.download_hf_model

# Download a specific model
python -m insanely_fast_whisper_api.download_hf_model --model openai/whisper-large-v3

# Force re-download of the model
python -m insanely_fast_whisper_api.download_hf_model --force

# Use a custom cache directory
python -m insanely_fast_whisper_api.download_hf_model --cache_dir /path/to/cache
```

For private or gated models, set the `HUGGINGFACE_TOKEN` environment variable with your API token.

## Configuration

The API can be configured using environment variables in `~/.config/insanely-fast-whisper-api/.env`. A comprehensive example with all available options and their descriptions is provided in `.env.example`:

1. Copy the example configuration:

    ```bash
    mkdir -p ~/.config/insanely-fast-whisper-api
    cp .env.example ~/.config/insanely-fast-whisper-api/.env
    ```

2. Edit the configuration file:

    ```bash
    nano ~/.config/insanely-fast-whisper-api/.env
    ```

    Available configuration options include:

    ```bash
    # Model Configuration
    WHISPER_MODEL=openai/whisper-base    # Model to use for transcription
    WHISPER_DEVICE=cpu                   # Device for inference (cpu/cuda/mps)
    WHISPER_DTYPE=float32                # Model weight data type
    WHISPER_BATCH_SIZE=8                 # Batch size for processing
    WHISPER_CHUNK_LENGTH=30              # Audio chunk length in seconds
    WHISPER_BETTER_TRANSFORMER=false     # Whether to use BetterTransformer

    # File Handling
    WHISPER_UPLOAD_DIR=temp_uploads      # Directory for temporary files

    # Filename Configuration
    FILENAME_TIMEZONE=UTC                # Timezone for output filenames (UTC/local timezone)
    ```

    If no configuration file exists, the API will use these default values. The configuration file will be automatically created with default values on first run.

## Usage

The application provides three main interfaces: **API**, **WebUI**, and **CLI**.

### API Server

Start the API server using one of these methods (add these commands to the `docker-compose.yaml` file to run them on startup):

```bash
# Method 1: Direct module execution
python -m insanely_fast_whisper_api

# Method 2: With verbose logging
python -m insanely_fast_whisper_api -v

# Method 3: Using uvicorn directly
uvicorn insanely_fast_whisper_api.main:app --host 0.0.0.0 --port 8888
```

The API provides two main endpoints:

1. `/v1/audio/transcriptions` - Transcribe audio in its source language
2. `/v1/audio/translations` - Translate audio to English

Visit `http://localhost:8888/docs` for the interactive API documentation.

### WebUI (Gradio Interface)

Launch the web-based user interface for batch file processing (add these commands to the `docker-compose.yaml` file to run them on startup):

```bash
# Basic WebUI launch
python -m insanely_fast_whisper_api.webui.cli

# With debug logging (recommended)
python -m insanely_fast_whisper_api.webui.cli --debug

# Custom host and port
python -m insanely_fast_whisper_api.webui.cli --port 7860 --host 0.0.0.0 --debug
```

Access the WebUI at `http://localhost:7860` for:
- Multi-file batch processing
- Real-time progress tracking
- ZIP downloads with multiple formats (TXT, JSON, SRT)

### CLI (Command Line Interface)

Use the command line interface for single-file processing (add these commands to the `docker-compose.yaml` file to run them on startup):

```bash
# Transcribe audio file
python -m insanely_fast_whisper_api.cli.cli transcribe audio_file.mp3

# Transcribe with options
python -m insanely_fast_whisper_api.cli.cli transcribe audio_file.mp3 --no-timestamps --debug

# Translate audio to English
python -m insanely_fast_whisper_api.cli.cli translate audio_file.mp3
```

### Output Files and Filename Conventions

The API uses standardized filename conventions for all output files to ensure consistency across different interfaces and file types. All generated files follow the pattern:

**Format**: `{audio_stem}_{task}_{timestamp}.{extension}`

#### Examples

- **JSON Transcription**: `my_audio_transcribe_20250530T143022Z.json`
- **SRT Subtitle**: `interview_transcribe_20250530T091234Z.srt`
- **Text File**: `recording_translate_20250530T205316Z.txt`

#### File Locations

- **API**: Files are saved to the `transcripts/` directory when transcription saving is enabled
- **CLI**: Results are saved to the `transcripts/` directory by default
- **WebUI**: Files can be downloaded with standardized filenames and are temporarily stored for download

#### Timestamp Configuration

The timestamp format can be customized using the `FILENAME_TIMEZONE` environment variable:

```bash
# Use local timezone
FILENAME_TIMEZONE=Europe/Amsterdam

# Use UTC (default)
FILENAME_TIMEZONE=UTC
```

### API Parameters

The API endpoints have distinct parameters. Core model settings (`model`, `device`, `batch_size`, etc.) are configured globally through environment variables and cannot be changed per-request.

#### `/v1/audio/transcriptions`

- `file`: The audio file to transcribe (required).
- `timestamp_type`: The granularity of the timestamps (`chunk` or `word`). If you provide `text` here, the response will be plain text instead of JSON. Defaults to `chunk`.
- `language`: The language of the audio. If omitted, the model will auto-detect the language.

#### `/v1/audio/translations`

- `file`: The audio file to translate (required).
- `response_format`: The desired output format (`json` or `text`). Defaults to `json`.
- `timestamp_type`: The granularity of the timestamps (`chunk` or `word`). Defaults to `chunk`.
- `language`: The language of the audio. If omitted, the model will auto-detect the language.

## Project Structure

```md
├── insanely_fast_whisper_api/          # Main package
│   ├── __init__.py                     # Package initialization
│   ├── __main__.py                     # Module entry point
│   ├── main.py                         # FastAPI application entry
│   ├── logging_config.yaml             # Logging configuration
│   ├── api/                            # FastAPI application layer
│   │   ├── __init__.py
│   │   ├── app.py                      # FastAPI app setup
│   │   ├── routes.py                   # API endpoints
│   │   ├── models.py                   # Pydantic data models
│   │   ├── dependencies.py             # Dependency injection
│   │   ├── middleware.py               # Request/response middleware
│   │   └── responses.py                # Response formatters
│   ├── core/                           # Core ASR logic
│   │   ├── __init__.py
│   │   ├── pipeline.py                 # ASR orchestration
│   │   ├── asr_backend.py              # Whisper model backend
│   │   ├── storage.py                  # File lifecycle management
│   │   ├── utils.py                    # Core utilities
│   │   └── errors.py                   # Exception classes
│   ├── audio/                          # Audio processing
│   │   ├── __init__.py
│   │   ├── processing.py               # Validation and preprocessing
│   │   └── results.py                  # Output formatting
│   ├── cli/                            # CLI tools
│   │   ├── __init__.py
│   │   ├── cli.py                      # CLI entry point
│   │   ├── commands.py                 # Subcommand logic
│   │   └── facade.py                   # High-level CLI wrapper
│   ├── webui/                          # Web UI (Gradio)
│   │   ├── __init__.py
│   │   ├── cli.py                      # WebUI CLI entry point
│   │   ├── ui.py                       # Gradio interface
│   │   ├── handlers.py                 # Upload + result management
│   │   ├── formatters.py               # Export formats (TXT, JSON, SRT)
│   │   ├── utils.py                    # WebUI utilities
│   │   ├── errors.py                   # UI-specific exceptions
│   │   └── downloads/                  # ZIP/file merging for downloads
│   │       ├── zip_creator.py          # ZIP archive builder
│   │       └── merge_handler.py        # Transcription file merge handlers
│   └── utils/                          # General utilities
│       ├── __init__.py
│       ├── constants.py                # Environment variables & config
│       ├── download_hf_model.py        # Model downloading & caching
│       ├── file_utils.py               # File operations
│       └── filename_generator.py       # Unified filename logic
```

## Development

### Setting Up Development Environment

1. Create a virtual environment:

    ```bash
    python3.10 -m venv venv
    source venv/bin/activate
    ```

2. Install development dependencies:

    ```bash
    # Install base and AMD GPU dependencies
    pip install -r requirements.txt
    pip install -r requirements-rocm.txt
    pip install -r requirements-onnxruntime-rocm.txt
    
    # Install development tools
    pip install -r requirements-dev.txt
    ```

### Code Style

This project follows PEP 8 guidelines. Use the following tools to maintain code quality:

- Format code with Black:

```bash
black insanely_fast_whisper_api tests
```

- Check code style with Flake8:

```bash
flake8 insanely_fast_whisper_api tests
```

### Running Tests

Run the test suite:

```bash
pytest tests/
```

Run tests with coverage:

```bash
pytest --cov=insanely_fast_whisper_api tests/
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate and follow the code style guidelines outlined in the Development section.
