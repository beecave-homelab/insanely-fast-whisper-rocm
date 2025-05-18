# Insanely Fast Whisper ROCm

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/docker-blue)](https://www.docker.com/)
[![ROCm](https://img.shields.io/badge/ROCm-6.1.2-orange)](https://rocm.docs.amd.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A high-performance, GPU-accelerated environment for generating transcripts from audio files using AMD hardware and the ROCm platform. The project provides both a command-line interface and a web-based user interface for transcribing audio files using OpenAI's Whisper model optimized for AMD GPUs.

## Features

- 🚀 Fast transcription using Distil-Whisper models
- 🎯 Optimized for AMD GPUs with ROCm
- 🖥️ Web-based interface using Gradio
- 🐳 Containerized with Docker and Docker Compose
- 📁 Automatic file monitoring and processing
- 🔄 Multiple output formats (txt, srt, vtt)
- 🧪 Comprehensive test suite
- 🛠️ Developer-friendly with pre-commit hooks and code formatting

## Table of Contents

- [Features](#features)
- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Docker Setup](#docker-setup)
  - [Development Setup](#development-setup)
- [Usage](#usage)
  - [Gradio Web UI](#gradio-web-ui)
  - [Automatic File Processing](#automatic-file-processing)
  - [Environment Variables](#environment-variables)
- [Development](#development)
  - [Project Structure](#project-structure)
  - [Testing](#testing)
  - [Code Quality](#code-quality)
  - [Pre-commit Hooks](#pre-commit-hooks)
- [License](#license)
- [Contributing](#contributing)

## Installation

### Prerequisites

- **Docker** (version 20.10 or newer)
- **Docker Compose**
- **AMD GPU** with ROCm 6.1.2 support
- **Python 3.10+** (for development)

### Docker Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/beecave-homelab/insanely-fast-whisper-rocm.git
   cd insanely-fast-whisper-rocm
   ```

2. Copy the example environment file and customize it:

   ```bash
   cp .env.example .env
   # Edit .env as needed
   ```

3. Build and start the containers:

   ```bash
   docker-compose up -d --build
   ```

4. Access the web interface at `http://localhost:7862`

### Development Setup

1. Create and activate a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

2. Install dependencies:

   ```bash
   pip install -e .
   pip install -r requirements-dev.txt
   ```

3. Install pre-commit hooks:

   ```bash
   pre-commit install
   ```

## Usage

### Gradio Web UI

The web interface provides an easy way to transcribe audio files:

1. Access the web interface at `http://localhost:7862`
2. Upload audio files using the file uploader
3. View and download transcripts in various formats

### Automatic File Processing

1. Place audio files in the `uploads/` directory
2. The service will automatically process them
3. Find transcripts in the `transcripts/` directory

### Environment Variables

Key environment variables (set in `.env`):

```env
# Core settings
MODEL=distil-whisper/distil-large-v3
BATCH_SIZE=6
LOG_LEVEL=INFO

# File paths
UPLOADS=uploads
TRANSCRIPTS=transcripts
LOGS=logs

# Output conversion
CONVERT_OUTPUT_FORMATS=txt,srt
CONVERT_CHECK_INTERVAL=120
PROCESSED_TXT_DIR=transcripts-txt
PROCESSED_SRT_DIR=transcripts-srt

# Web interface
WEB_HOST=0.0.0.0
WEB_PORT=7860
```

## Development

### Project Structure

```text
.
├── config/               # Application configuration
│   ├── __init__.py
│   ├── settings.py       # Pydantic settings
│   └── logging_config.py # Logging configuration
├── src/                  # Source code
│   ├── core/             # Core functionality
│   ├── models/           # Data models
│   ├── services/         # Business logic
│   ├── utils/            # Utility functions
│   └── web/              # Web interface
├── tests/                # Tests
│   ├── unit/             # Unit tests
│   └── integration/      # Integration tests
├── .github/              # GitHub workflows
├── scripts/              # Utility scripts
└── docker/               # Docker-related files
```

### Testing

Run the test suite:

```bash
# Run all tests with coverage
make test

# Run tests without coverage
make test-fast

# Run a specific test file
pytest tests/unit/test_settings.py -v
```

### Code Quality

Format and check code quality:

```bash
# Format code
make format

# Run all linters
make lint

# Run specific linters
make lint-flake8
make lint-mypy
make lint-black
make lint-isort
```

### Pre-commit Hooks

Pre-commit hooks are configured to run automatically on each commit. They include:

- Black code formatting
- isort import sorting
- Flake8 linting
- MyPy type checking

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

Please make sure to update tests as appropriate.
