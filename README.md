# Insanely Fast Whisper (ROCm)

A comprehensive Whisper-based speech recognition toolkit designed specifically to provide **AMD GPU (ROCm) support** for high-performance (video to) audio transcription and translation. This package extends the capabilities of the original [insanely-fast-whisper](https://github.com/Vaibhavs10/insanely-fast-whisper) by providing multiple interfaces and ROCm compatibility.

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org)
[![Version](https://img.shields.io/badge/Version-v2.0.1-informational)](#insanely-fast-whisper-rocm)
[![ROCm Version](https://img.shields.io/badge/ROCm-v6.4-informational)](https://repo.radeon.com/rocm/manylinux/rocm-rel-6.4.1/)
[![API](https://img.shields.io/badge/API-FastAPI-green)](#api-server)
[![CLI](https://img.shields.io/badge/CLI-Click-yellow)](#cli-command-line-interface)
[![WebUI](https://img.shields.io/badge/WebUI-Gradio-orange)](#webui-gradio-interface)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE.txt)

## 🚀 What's Included

- **🔌 FastAPI Server**: RESTful API with OpenAI-compatible endpoints
- **🎛️ Gradio WebUI**: Web-based interface for batch file processing with live progress tracking
- **⚡ CLI Tools**: Command-line interface for single-file processing
- **📦 Model Management**: Automatic Hugging Face model downloading and caching
- **🏗️ Docker Support**: Full containerization with development and production configurations (now using PDM for dependency management in Docker builds)
- **🎯 ROCm Integration**: AMD GPU [(ROCm v6.4)](https://repo.radeon.com/rocm/manylinux/rocm-rel-6.4.1/) support for accelerated inference

## Table of Contents

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

## Key Features

- **AMD GPU (ROCm) Support**: Primary focus on enabling Whisper models on AMD GPUs
- **Multiple Interfaces**: Choose between API, WebUI, or CLI based on your workflow
- **Batch Processing**: Handle multiple audio **and video** files simultaneously via WebUI
- **High Performance**: Optimized processing with configurable batch sizes and model parameters
- **Multiple Output Formats**: Support for JSON, TXT, and SRT subtitle formats
- **Standardized Filenames**: Consistent, timestamped output naming across all interfaces
- **Readable Subtitles (SRT/VTT)**: Advanced segmentation pipeline that creates well-formed, readable subtitles by default, respecting line length, duration, and characters-per-second (CPS) constraints. This can be toggled with the `USE_READABLE_SUBTITLES` environment variable.
- **Word-level Timestamp Stabilization (CLI, API & WebUI)**: Optional `--stabilize` flag (powered by [stable-ts](https://github.com/jianfch/stable-ts)) greatly refines chunk timestamps, producing accurate word-aligned SRT/VTT output
- **Noise Reduction & Voice Activity Detection (CLI, API & WebUI)**: Optional `--demucs` and `--vad` flags provide Demucs-based denoising and intelligent speech-region detection (adjustable `--vad-threshold`) for cleaner, more accurate transcripts

---

## Additional Features

- **Modern Acceleration**: Uses native PyTorch 2.0 Scaled Dot Product Attention (`sdpa`) for optimized performance, which is the modern successor to `BetterTransformer`.
- **Video & Audio Support**: Process standard audio formats (.wav, .flac, .mp3) **and** video files (.mp4, .mkv, .webm, .mov) thanks to automatic audio extraction via FFmpeg
- **OpenAI-Compatible API**: Drop-in replacement for OpenAI's audio endpoints (v1)
- **Environment-based Configuration**: Flexible configuration via `.env` files.
- **Real-time Progress**: Live progress tracking in WebUI for batch operations
- **ZIP Downloads**: Bundle multiple transcription formats for easy download
- **Robust Error Handling**: Comprehensive error management across all interfaces
- **Docker-first Deployment**: Production-ready containerization

## Why This Package?

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
    # This generates `~/.config/insanely-fast-whisper-rocm/.env` with sensible defaults
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
python -m insanely_fast_whisper_rocm.utils.download_hf_model

# Download a specific model
python -m insanely_fast_whisper_rocm.utils.download_hf_model --model openai/whisper-large-v3

# Force re-download of the model
python -m insanely_fast_whisper_rocm.utils.download_hf_model --force

# Use a custom cache directory
python -m insanely_fast_whisper_rocm.utils.download_hf_model --cache_dir /path/to/cache
```

For private or gated models, set the `HF_TOKEN` environment variable with your API token.

## Configuration

The API can be configured using environment variables in `~/.config/insanely-fast-whisper-rocm/.env`. A template with all available options is generated automatically by the configuration setup script mentioned above.

Key configuration options include:

- `WHISPER_MODEL`: The Whisper model to use (e.g., `openai/whisper-large-v3`).
- `WHISPER_DEVICE`: The device to run on (`0` for CUDA, `mps` for Apple Silicon, `cpu`).
- `USE_READABLE_SUBTITLES`: `true` or `false`. Enables the new readable subtitle segmentation pipeline. Defaults to `true`.

For a detailed explanation of the configuration system, including hierarchical loading and key files, please see the [`Configuration System` section in `project-overview.md`](./project-overview.md#configuration-system).

### Initial User Configuration Setup

To create or update your user-specific configuration file (`~/.config/insanely-fast-whisper-rocm/.env`), you can use the provided setup script.

1. **Run the setup script:**

    This script helps you create the `~/.config/insanely-fast-whisper-rocm/.env` file.

    If you are using PDM (recommended for managing dependencies and scripts):

    ```bash
    pdm run setup-config
    ```

    Alternatively, you can run the script directly from the project root:

    ```bash
    python scripts/setup_config.py
    ```

2. **Edit your configuration file:**

    After running the script, open `~/.config/insanely-fast-whisper-rocm/.env` with your preferred text editor and customize the settings. Pay special attention to `HF_TOKEN` if using gated models. Refer to [`.env.example`](./.env.example) in the project root for a full list of available options and their descriptions.

    > [!IMPORTANT]
    > **ROCm / AMD GPU compatibility (check your `gfx` target):**
    >
    > Some AMD GPUs are not officially supported by a given ROCm release (for example, an RX 6600 is `gfx1032`, while many ROCm builds only ship kernels for `gfx1030`). If ROCm can’t find a matching code object for your card, GPU inference may fail.
    >
    > In that case, you can often work around this by **uncommenting** `HSA_OVERRIDE_GFX_VERSION` in your `.env` file and setting it to a supported target.
    >
    > To discover your GPU target:
    >
    > ```bash
    > rocm_agent_enumerator -name
    > rocminfo  # look for a GPU agent line like: Name: gfxXXXX
    > ```
    >
    > To choose a supported target for your ROCm version:
    >
    > - [ROCm compatibility matrix](https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html)
    >
    > GPU-to-`gfx` reference table:
    >
    > - [GPU hardware specifications (gfx targets)](https://rocm.docs.amd.com/en/latest/reference/gpu-arch-specs.html)
    >
    > Example: forcing `gfx1030` corresponds to `HSA_OVERRIDE_GFX_VERSION=10.3.0`.

    If no configuration file exists, the API will use these default values. The configuration file will be automatically created with default values on first run.

## Usage

The application provides three main interfaces: **API**, **WebUI**, and **CLI**.

### API Server

The FastAPI server can be started with:

```bash
python -m insanely_fast_whisper_rocm.api
```

This launches the server (typically at `http://0.0.0.0:8000`). Interactive API documentation is available at `/docs`.

Key Endpoints:

- `/v1/audio/transcriptions`: Transcribe audio in its source language.
- `/v1/audio/translations`: Translate audio to English.

For detailed launch options and API parameters, see [`project-overview.md`](./project-overview.md#api-server-details).

### WebUI (Gradio Interface)

The Gradio WebUI provides a user-friendly interface for batch processing. Start it with:

```bash
python -m insanely_fast_whisper_rocm.webui
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
# Transcribe with word-level timestamp stabilization
python -m insanely_fast_whisper_rocm.cli transcribe audio_file.mp3 --stabilize

# Transcribe and get a JSON file (default)
python -m insanely_fast_whisper_rocm.cli transcribe audio_file.mp3

# Transcribe and get a TXT file
python -m insanely_fast_whisper_rocm.cli transcribe audio_file.mp3 --export-format txt

# Transcribe and get all formats (JSON, SRT, TXT)
python -m insanely_fast_whisper_rocm.cli transcribe audio_file.mp3 --export-format all

# Translate and get an SRT file
python -m insanely_fast_whisper_rocm.cli translate audio_file.mp3 --export-format srt
```

For detailed commands and options, see [`project-overview.md`](./project-overview.md#cli-command-line-interface-details).

#### Quiet mode (`--quiet`)

Use `--quiet` to minimize console output. In quiet mode, only the Rich progress bar (when attached to a TTY) and the final saved-path line(s) are shown. Intermediate logs/messages are suppressed. This also hides third-party Demucs/VAD progress and HIP/MIOpen warnings when stabilization is enabled. See the CLI section in [`project-overview.md`](./project-overview.md#quiet-mode---quiet) for details.

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
- `stabilize`: `bool` - Enable timestamp stabilization using `stable-ts`. Defaults to `False`.
- `demucs`: `bool` - Enable Demucs noise reduction before transcription. Defaults to `False`.
- `vad`: `bool` - Enable Silero VAD to filter out silent parts of the audio. Defaults to `False`.
- `vad_threshold`: `float` - The threshold for VAD. Defaults to `0.35`.

#### `/v1/audio/translations`

- `file`: The audio file to translate (required).
- `response_format`: The desired output format (`json` or `text`). Defaults to `json`.
- `timestamp_type`: The granularity of the timestamps (`chunk` or `word`). Defaults to `chunk`.
- `language`: The language of the audio. If omitted, the model will auto-detect the language.
- `stabilize`: `bool` - Enable timestamp stabilization using `stable-ts`. Defaults to `False`.
- `demucs`: `bool` - Enable Demucs noise reduction before transcription. Defaults to `False`.
- `vad`: `bool` - Enable Silero VAD to filter out silent parts of the audio. Defaults to `False`.
- `vad_threshold`: `float` - The threshold for VAD. Defaults to `0.35`.

## Reviewer Quick Start (Lightweight Testing)

For code reviewers or contributors who need to run tests without a GPU or heavy ML libraries, a lightweight, CPU-only requirements file is provided.

1. **Install lightweight dependencies:**

    ```bash
    pip install -r requirements-reviewer.txt
    ```

2. **Run the CPU-safe test suite:**

    The following command runs tests that do not require `torch` or a GPU. It excludes tests for CUDA, the full ASR backend, and server integration tests.

    ```bash
    pytest -q -k "not (cuda or webui or api_integration or asr_backend_generation_config or asr_backend_timestamp or api)"
    ```

    This ensures that core logic, utilities, and the dummy pipeline can be validated quickly in any environment.

## Development

See the [project-overview.md](./project-overview.md#development) for details on setting up the development environment, code style, and running tests.

## License

This project is licensed under the MIT [License](./LICENSE.txt) - see the LICENSE file for details.

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate and follow the code style guidelines outlined in the Development section.
