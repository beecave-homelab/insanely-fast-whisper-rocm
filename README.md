# Insanely Fast Whisper API (ROCm)

A comprehensive Whisper-based speech recognition toolkit designed specifically to provide **AMD GPU (ROCm) support** for high-performance (video to) audio transcription and translation. This package extends the capabilities of the original [insanely-fast-whisper](https://github.com/Vaibhavs10/insanely-fast-whisper) by providing multiple interfaces and ROCm compatibility.

## 🚀 What's Included

- **🔌 FastAPI Server**: RESTful API with OpenAI-compatible endpoints
- **🎛️ Gradio WebUI**: Web-based interface for batch file processing with live progress tracking
- **⚡ CLI Tools**: Command-line interface for single-file processing
- **📦 Model Management**: Automatic Hugging Face model downloading and caching
- **🏗️ Docker Support**: Full containerization with development and production configurations (now using PDM for dependency management in Docker builds)
- **🎯 ROCm Integration**: AMD GPU (ROCm v6.1) support for accelerated inference

## Key Features

- **AMD GPU (ROCm) Support**: Primary focus on enabling Whisper models on AMD GPUs
- **Multiple Interfaces**: Choose between API, WebUI, or CLI based on your workflow
- **Batch Processing**: Handle multiple audio **and video** files simultaneously via WebUI
- **High Performance**: Optimized processing with configurable batch sizes and model parameters
- **Multiple Output Formats**: Support for JSON, TXT, and SRT subtitle formats
- **Standardized Filenames**: Consistent, timestamped output naming across all interfaces

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org)
[![Version](https://img.shields.io/badge/Version-v0.9.0-informational)](#insanely-fast-whisper-api-rocm)
[![API](https://img.shields.io/badge/API-FastAPI-green)](#api-server)
[![CLI](https://img.shields.io/badge/CLI-Click-yellow)](#cli-command-line-interface)
[![WebUI](https://img.shields.io/badge/WebUI-Gradio-orange)](#webui-gradio-interface)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE.txt)

---

## 📚 Table of Contents

- [Key Features](#key-features)
- [Installation](#installation)
  - [Model Download](#model-download)
- [Configuration](#configuration)
- [Usage](#usage)
  - [API Server](#api-server)
  - [WebUI (Gradio Interface)](#webui-gradio-interface)
  - [CLI (Command Line Interface)](#cli-command-line-interface)
  - [Docker Usage](#recommended-docker-compose-production--development)
  - [Output Files and Filename Conventions](#output-files-and-filename-conventions)
  - [API Parameters](#api-parameters)
- [Development](#development)
- [License](#license)
- [Contributing](#contributing)

## 🌟 Additional Features

- **Modern Acceleration**: Uses native PyTorch 2.0 Scaled Dot Product Attention (`sdpa`) for optimized performance, which is the modern successor to `BetterTransformer`.
- **Video & Audio Support**: Process standard audio formats (.wav, .flac, .mp3) **and** video files (.mp4, .mkv, .webm, .mov) thanks to automatic audio extraction via FFmpeg
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
    git clone https://github.com/beecave-homelab/insanely-fast-whisper-rocm.git
    cd insanely-fast-whisper-rocm
    ```

2. Set up configuration (see [Configuration](#configuration)) for more details:

    ```bash
    # Create your user configuration file (interactive)
    # This generates `~/.config/insanely-fast-whisper-api/.env` with sensible defaults
    pdm run setup-config  # or `python scripts/setup_config.py` if you do not use PDM
    ```

3. Start the application:

    ```bash
    docker compose up --build -d
    ```

### Alternative: Local Development with PDM

For local development, PDM (Python Development Master) is used to manage dependencies and run scripts. Ensure you have Python 3.10+ installed.

1. Clone the repository:

    ```bash
    git clone https://github.com/beecave-homelab/insanely-fast-whisper-rocm.git
    cd insanely-fast-whisper-rocm
    ```

2. Install PDM (if you haven't already):

    ```bash
    curl -sSL https://pdm-project.org/install-pdm.py | python3 -
    ```

    Refer to the [official PDM documentation](https://pdm-project.org/latest/installation/) for other installation methods.

3. Install project dependencies using PDM:

> [!IMPORTANT]  
> **Benchmarking with Multiple ROCm Torch Versions:**
>
> To benchmark with a specific ROCm-compatible torch version, install the matching optional group (e.g., `bench-torch-2_3_0`):
>
> ```bash
> pdm install -G bench-torch-2_3_0
> ```
>
> This will install the specified torch version and benchmarking tools. See [`project-overview.md`](./project-overview.md#benchmarking-with-multiple-rocm-torch-versions) for details and the full list of available groups.
>
> This application is specifically designed to provide **AMD GPU (ROCm) support** for Whisper models. The `rocm` dependency group in [`pyproject.toml`](./pyproject.toml) ensures proper PyTorch and ONNX runtime installation for AMD GPUs. While it should technically also works on CPU and NVIDIA GPUs, ROCm support was the primary motivation for this package.

This command installs the project's core dependencies. To install optional groups for development or specific hardware support (like ROCm), use the `-G` flag:

```bash
# To install ROCm support
pdm install -G rocm

# To include development tools and ROCm support
pdm install -G dev -G rocm 
```

### Model Download

The application will automatically download the specified Whisper model on first use. You can also pre-download models using the included script (add these commands to the [`docker-compose.yaml`](./docker-compose.yaml) file to run them on startup):

```bash
# Download the default model (specified in .env or WHISPER_MODEL env var)
python -m insanely_fast_whisper_api.utils.download_hf_model

# Download a specific model
python -m insanely_fast_whisper_api.utils.download_hf_model --model openai/whisper-large-v3

# Force re-download of the model
python -m insanely_fast_whisper_api.utils.download_hf_model --force

# Use a custom cache directory
python -m insanely_fast_whisper_api.utils.download_hf_model --cache_dir /path/to/cache
```

For private or gated models, set the `HUGGINGFACE_TOKEN` environment variable with your API token.

## Configuration

The API can be configured using environment variables in `~/.config/insanely-fast-whisper-api/.env`. A template with all available options is generated automatically by the configuration setup script mentioned above.

For a detailed explanation of the configuration system, including hierarchical loading and key files, please see the [`Configuration System` section in `project-overview.md`](./project-overview.md#configuration-system).

### Initial User Configuration Setup

To create or update your user-specific configuration file (`~/.config/insanely-fast-whisper-api/.env`), you can use the provided setup script.

1. **Run the setup script:**

    This script helps you create the `~/.config/insanely-fast-whisper-api/.env` file.

    If you are using PDM (recommended for managing dependencies and scripts):

    ```bash
    pdm run setup-config
    ```

    Alternatively, you can run the script directly from the project root:

    ```bash
    python scripts/setup_config.py
    ```

2. **Edit your configuration file:**

    After running the script, open `~/.config/insanely-fast-whisper-api/.env` with your preferred text editor and customize the settings. Pay special attention to `HUGGINGFACE_TOKEN` if using gated models. Refer to [`.env.example`](./.env.example) in the project root for a full list of available options and their descriptions.

    If no configuration file exists, the API will use these default values. The configuration file will be automatically created with default values on first run.

## Usage

The application provides three main interfaces: **API**, **WebUI**, and **CLI**.

### API Server

The FastAPI server can be started with:

```bash
python -m insanely_fast_whisper_api.api
```

This launches the server (typically at `http://0.0.0.0:8000`). Interactive API documentation is available at `/docs`.

Key Endpoints:

- `/v1/audio/transcriptions`: Transcribe audio in its source language.
- `/v1/audio/translations`: Translate audio to English.

For detailed launch options and API parameters, see [`project-overview.md`](./project-overview.md#api-server-details).

### WebUI (Gradio Interface)

The Gradio WebUI provides a user-friendly interface for batch processing. Start it with:

```bash
python -m insanely_fast_whisper_api.webui
```

Access it at `http://localhost:7860` (default). Features include:

- Multi-file batch processing (audio & video)
- Real-time progress tracking
- ZIP downloads (TXT, JSON, SRT)

For detailed launch options, see [`project-overview.md`](./project-overview.md#webui-gradio-interface-details).

### CLI (Command Line Interface)

The CLI is suitable for single-file transcription and translation.

Basic usage:

```bash
# Transcribe and get a JSON file (default)
python -m insanely_fast_whisper_api.cli transcribe audio_file.mp3

# Transcribe and get a TXT file
python -m insanely_fast_whisper_api.cli transcribe audio_file.mp3 --export-format txt

# Transcribe and get all formats (JSON, SRT, TXT)
python -m insanely_fast_whisper_api.cli transcribe audio_file.mp3 --export-format all

# Translate and get an SRT file
python -m insanely_fast_whisper_api.cli translate audio_file.mp3 --export-format srt
```

For detailed commands and options, see [`project-overview.md`](./project-overview.md#cli-command-line-interface-details).

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

The timestamp format can be customized using the `APP_TIMEZONE` environment variable:

```bash
# Use local timezone
TZ=Europe/Amsterdam

# Use UTC (default)
TZ=UTC
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

## Development

See the [project-overview.md](./project-overview.md#development) for details on setting up the development environment, code style, and running tests.

## License

This project is licensed under the MIT [License](./LICENSE.txt) - see the LICENSE file for details.

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate and follow the code style guidelines outlined in the Development section.
