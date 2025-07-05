# 📋 Version History & Changelog

**Insanely Fast Whisper API** - Complete version history and feature evolution tracking.

[![Version](https://img.shields.io/badge/Version-v0.6.0-informational)](#release-timeline)

---

## 📑 Table of Contents

- [`v0.6.0` (Current) - *05-07-2025*](#v060-current---05-07-2025)
- [`v0.5.0` - *June 2025*](#v050---june-2025)
- [`v0.4.1` - *June 2025*](#v041---june-2025)
- [`v0.4.0` - *June 2025*](#v040---june-2025)
- [`v0.3.1` - *June 4, 2025*](#v031---june-4-2025)
- [`v0.3.0` - *May 27-31, 2025*](#v030---may-27-31-2025)
- [`v0.2.1` - *May 29-30, 2025*](#v021---may-29-30-2025)
- [`v0.2.0` - *May 20-21, 2025*](#v020---may-20-21-2025)
- [`v0.1.2` - *March 8, 2025*](#v012---march-8-2025)
- [`v0.1.1` - *January 19, 2025*](#v011---january-19-2025)
- [`v0.1.0` - *January 18, 2025*](#v010---january-18-2025)

---

## 🔄 Semantic Versioning (SemVer)

This project follows [Semantic Versioning](https://semver.org/) format: `MAJOR.MINOR.PATCH`

- **MAJOR**: Breaking changes or architectural overhauls
- **MINOR**: New features and enhancements (backward compatible)
- **PATCH**: Bug fixes and small improvements

---

## Release Timeline

### `v0.6.0` (Current) - *05-07-2025*

#### ✨ Minor Release: Translation CLI, SDPA attention, ASR refactors

#### ✨ **New Features in v0.6.0**

- **Added**: Translation functionality to CLI tool
- **Enhanced**: Replaced BetterTransformer with native PyTorch SDPA attention

#### 🔧 **Improvements in v0.6.0**

- **Refactored**: ASR pipeline and backend for improved model loading and processing
- **Refactored**: Removed BetterTransformer configuration
- **Improved**: Language processing logic

#### 📝 **Commits**: `496f49a`, `bbd78e4`, `e16511b`, `3e78fe4`, `ec08c5e`

---

### `v0.5.0` - *June 2025*

#### ✨ Feature Release: Major Restructure & ROCm Support

This release marks a significant architectural overhaul:

- The project was migrated to a fully modular structure.
- Dependency management was switched to `pdm`.
- A new modular CLI and a user configuration script were added.

#### ✨ New Features in v0.5.0

- **Modular CLI**: Created a new CLI module with distinct commands for transcription and other utilities. (Commit `10d529d`)
- **User Configuration**: Added a setup script (`setup_config.py`) to simplify user-specific `.env` configuration. (Commit `d13d17a`)

#### ♻️ Refactoring & Improvements in v0.5.0

- **Project Restructure**: Major refactoring of the entire codebase into a modular structure (`api`, `core`, `cli`, `webui`, `utils`). (Commits `6ad709c`, `056e0e2`, `517abca`, `914724c`)
- **Dependency Management**: Migrated to `pdm` and refined `pyproject.toml` with optional dependency groups (`rocm`, `dev`). (Commits `d999f8d`, `8af2858`)

- **Audio Processing**: Enhanced audio utilities and improved error handling. (Commit `5d7b306`)
- **Docker Configuration**: Updated `Dockerfile` and `docker-compose.yaml` to align with the new project structure and remove `pipx` dependency. (Commit `6ad709c`)

#### 📝 Commits: `d999f8d`, `8af2858`, `b4d7791`, `112b627`, `f237624`

---

### `v0.4.1` - *June 2025*

#### 🐛 WebUI Download Fixes & Stability

#### 🐛 Bug Fixes in v0.4.1

- **Fixed**: `TypeError` in Gradio `DownloadButton` when updating visibility/value.
  - **Issue**: Returning `gr.DownloadButton()` instances instead of `gr.update()` dictionaries caused `TypeError: expected str, bytes or os.PathLike object, not function`.
  - **Root Cause**: Incorrect usage of Gradio update mechanisms for `gr.DownloadButton`.
  - **Solution**: Changed assignments to use `gr.update(value=filepath, visible=True)` for showing and `gr.update(visible=False)` for hiding download buttons. (Related to commit `76252e4`)
- **Fixed**: ZIP archive overwrites for different download types in WebUI.
  - **Issue**: Requesting different ZIP formats (e.g., "All", "TXT only") for the same batch of files resulted in the last requested ZIP overwriting previous ones.
  - **Root Cause**: `BatchZipBuilder.create` used the same base filename derived from `batch_id` for all ZIP types.
  - **Solution**: Appended unique suffixes (e.g., `_all_formats`, `_txt_only`) to the `batch_id` when constructing filenames for `BatchZipBuilder.create`, ensuring distinct archive names.

#### 🔧 Improvements in v0.4.1

- **Docs**: Added documentation for Gradio `DownloadButton` `TypeError` fix and ZIP archive overwrite fix (`c3eba0c`).
- **Chore**: Updated Dockerfile labels and added source URL (`a2f2ac2`).
- **Chore**: Enhanced support for long audio files (`e1ea7c3`).
- **Chore**: Removed an unused test file (`b447757`).

#### 📝 Commits in v0.4.1

`c3eba0c`, `76252e4`, `a2f2ac2`, `e1ea7c3`, `b447757`

---

### `v0.4.0` - *June 2025*

#### ✨ Enhanced Versioning & Logging

#### ✨ New Features in v0.4.0

- **Enhanced**: Improved versioning system
- **Enhanced**: Standardized logging format across the application

#### 📝 Commits in v0.4.0

`8874ad1`, `687bb2f`, `65157c8`, `0a3659f`

---

### `v0.3.1` - *June 4, 2025*

#### 🐛 Stability & Multi-file Processing

#### ✨ New Features in v0.3.1

- Native Gradio multi-file processing features
- Enhanced transcription processing with improved error handling
- Improved configuration test robustness and clarity

#### 🐛 Bug Fixes in v0.3.1

- **Fixed**: Empty ZIP file downloads in WebUI batch processing
  - **Issue**: WebUI ZIP downloads were missing transcription content
  - **Root Cause**: `result_dict` was incorrectly accessed in `handlers.py`
  - **Solution**: Corrected data structure access and improved error handling
- **Fixed**: Audio format validation issues
  - **Issue**: Deprecated audio extensions causing processing errors
  - **Solution**: Updated supported format validation and removed legacy extensions
- **Fixed**: Configuration test inconsistencies
  - **Issue**: Inconsistent configuration test results
  - **Solution**: Refactored centralized configuration tests for improved robustness

#### 🔧 Improvements in v0.3.1

- Updated supported audio format validation
- Enhanced error messages for better debugging
- Improved ZIP handling functionality

#### 📝 Commits in v0.3.1

`35f3584`, `a61d52f`, `9615350`, `138e14e`, `eb53217`, `0174de8`, `bc9b9d4`

---

### `v0.3.0` - *May 27-31, 2025*

#### ✨ WebUI Modularization & Advanced Features

#### 🏗️ Major Changes in v0.3.0

- **Complete WebUI refactor** into modular components:
  - `ui.py`: Gradio interface components
  - `handlers.py`: Upload and result management
  - `formatters.py`: Export formats (TXT, JSON, SRT)
  - `errors.py`: UI-specific error handling

#### ✨ New Features in v0.3.0

- CLI entrypoint for WebUI (`insanely-fast-whisper-webui`)
- Batch file processing with real-time progress tracking
- ZIP archive downloads for batch results
- Automatic Hugging Face model downloading and caching
- Timezone-aware filename generation
- Configuration dataclasses for better type safety
- Enhanced audio chunking with configurable overlap and duration

#### 🔧 Improvements in v0.3.0

- Centralized configuration tests and enhanced `.env` file support
- Docker Compose configurations for Hugging Face cache
- Dependency management enhancements in `pyproject.toml`
- Filename conventions and centralized configuration documentation

#### 📝 Key Commits in v0.3.0

`d6c1d74`, `73ef337`, `a162bcd`, `2a46385`, `f643be6`, `b366b4d`

---

### `v0.2.1` - *May 29-30, 2025*

#### ♻️ Import Standardization & Core Refinements

#### 🏗️ Major Changes in v0.2.1

- **Refactored all imports** to absolute paths for improved IDE support:

  ```python
  # ✅ New absolute imports
  from insanely_fast_whisper_api.core.pipeline import WhisperPipeline
  
  # ❌ Deprecated relative imports
  # from .core.pipeline import WhisperPipeline
  ```

#### ✨ New Features in v0.2.1

- Comprehensive error handling with custom exception classes
- Storage backend abstraction for ASR results
- Core utility functions and pipeline base classes
- Enhanced ASR pipeline with improved parameters and backend support

#### 🔧 Improvements in v0.2.1

- Better code organization and maintainability
- Improved dependency tracking
- Enhanced IDE auto-completion and navigation
- Consistent import patterns across the codebase

#### 📝 Commits in v0.2.1

`2d3fef9`, `5429378`, `36ddcf5`, `0142a23`, `94e69c9`, `8a7fbe5`

---

### `v0.2.0` - *May 20-21, 2025*

#### 🔄 Architectural Revolution ⚠️ BREAKING CHANGES

#### 💥 Breaking Changes in v0.2.0

- **Migrated from subprocess-based `insanely-fast-whisper`** to direct Hugging Face Transformers integration
- Removed external CLI tool dependencies
- Changed core pipeline architecture

#### ✨ New Features in v0.2.0

- Native `transformers.pipeline` support for Whisper models
- Progress callbacks for chunk-level processing
- Configurable batch sizes and chunk processing for different hardware
- Enhanced performance optimization
- Docker support and containerization improvements

#### 🔧 Improvements in v0.2.0

- Simplified dependencies by removing external tools
- Improved error handling and logging throughout ASR pipeline
- Better performance with direct model integration
- More reliable and faster transcription processing

#### 📝 Key Commits in v0.2.0

 `b8514a9`, `53051d2`

---

### `v0.1.2` - *March 8, 2025*

#### 🎨 WebUI Introduction

#### ✨ New Features in v0.1.2

- **First Gradio-based web interface** (`webui.py`)
- Interactive audio file upload and transcription
- Real-time transcription results display

#### 🔧 Improvements in v0.1.2

- Enhanced ASR pipeline with task parameter support
- Added custom exceptions for better error handling
- Improved CLI options and environment variable support
- Updated requirements with additional dependencies

#### 📝 Key Commits in v0.1.2

`7d1191d`, `272f3f8`, `08a1604`

---

### `v0.1.1` - *January 19, 2025*

#### ✨ New Features in v0.1.1

- Comprehensive logging configuration with timezone support
- PyYAML support for configuration files
- Environment variable support for enhanced configuration
- Enhanced ASR pipeline with custom exceptions

#### 🔧 Improvements in v0.1.1

- Enhanced FastAPI application with detailed documentation
- Improved device string conversion and transcription commands
- Better error handling and user feedback
- Enhanced CLI options and functionality

#### 🐛 Bug Fixes in v0.1.1

- Fixed requirements-rocm.txt by removing unused torchaudio
- Improved logging configuration for Uvicorn server

#### 📝 Key Commits in v0.1.1

`9524596`, `ea3abe4`, `455e33b`, `d950fb3`, `1bdd079`, `323d763`, `434bf39`

---

### `v0.1.0` - *January 18, 2025*

#### 🎉 Initial Release

#### ✨ Initial Features in v0.1.0

- FastAPI wrapper for Whisper-based ASR pipeline
- Modular project structure with organized components
- OpenAI-compatible API endpoints:
  - `POST /v1/audio/transcriptions`
  - `POST /v1/audio/translations`
- Basic CLI functionality
- Docker support with `Dockerfile` and `docker-compose.yaml`
- Comprehensive testing framework
- ROCm and CUDA GPU support
- Environment-based configuration system

#### 🏗️ Project Structure in v0.1.0

- Organized codebase with clear separation of concerns
- Comprehensive testing setup
- Documentation and README
- License and contribution guidelines

#### 📝 Key Commits in v0.1.0

`3dd155e`, `7fb6672`, `306a702` (Initial commit)

---
