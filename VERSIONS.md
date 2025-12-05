# 📋 Version History & Changelog

> **Note:** Key Commits sections were updated on 2025-07-06 to reflect normalized tags and true release commits.

**Insanely Fast Whisper API** - Complete version history and feature evolution tracking.

[![Version](https://img.shields.io/badge/Version-v1.0.2-informational)](#release-timeline)

---

## 📑 Table of Contents

- [`v1.0.2` (Current) - *05-12-2025*](#v102-current---05-12-2025)
- [`v1.0.1` - *04-12-2025*](#v101---04-12-2025)
- [`v1.0.0` - *18-09-2025*](#v100---18-09-2025)
- [`v0.10.1` - *16-09-2025*](#v0101---16-09-2025)
- [`v0.10.0` - *23-07-2025*](#v0100---23-07-2025)
- [`v0.9.1` - *19-07-2025*](#v091---19-07-2025)
- [`v0.9.0` - *06-07-2025*](#v090---06-07-2025)
- [`v0.8.0` - *06-07-2025*](#v080---06-07-2025)
- [`v0.7.0` - *06-07-2025*](#v070---06-07-2025)
- [`v0.6.0` - *05-07-2025*](#v060---05-07-2025)
- [`v0.5.0` - *05-06-2025*](#v050---07-06-2025)
- [`v0.4.1` - *06-06-2025*](#v041---06-06-2025)
- [`v0.4.0` - *06-06-2025*](#v040---06-06-2025)
- [`v0.3.1` - *04-06-2025*](#v031---04-06-2025)
- [`v0.3.0` - *27-05-2025*](#v030---27-05-2025)
- [`v0.2.1` - *29-05-2025*](#v021---29-05-2025)
- [`v0.2.0` - *20-05-2025*](#v020---20-05-2025)
- [`v0.1.2` - *08-03-2025*](#v012---08-03-2025)
- [`v0.1.1` - *19-01-2025*](#v011---19-01-2025)
- [`v0.1.0` - *18-01-2025*](#v010---18-01-2025)

---

## 🔄 Semantic Versioning (SemVer)

This project follows [Semantic Versioning](https://semver.org/) format: `MAJOR.MINOR.PATCH`

- **MAJOR**: Breaking changes or architectural overhauls
- **MINOR**: New features and enhancements (backward compatible)
- **PATCH**: Bug fixes and small improvements

---

## Release Timeline

### `v1.0.2` (Current) - *05-12-2025*

#### 🐛 Patch Release: API Routes Fix & Code Cleanup

This release focuses on code correctness and maintainability through TDD-driven fixes.

#### 🐛 Bug Fixes in v1.0.2

- **Fixed**: API routes passed invalid parameters (`stabilize`, `demucs`, `vad`, `vad_threshold`) to `WhisperPipeline.process()`.
  - **Issue**: These parameters are not part of the `process()` method signature.
  - **Root Cause**: Parameters were incorrectly forwarded; stabilization is handled separately by `stabilize_timestamps()` post-processing.
  - **Solution**: Removed invalid parameters from `asr_pipeline.process()` calls in both `create_transcription` and `create_translation` routes.

- **Fixed**: Dead code in `SrtFormatter.format()` (lines 427-550) was unreachable.
  - **Issue**: ~124 lines of fallback formatting code appeared after a `return` statement.
  - **Solution**: Removed the unreachable code block, reducing complexity and improving maintainability.

#### 🧪 Tests in v1.0.2

- **Added**: New `tests/api/test_routes_process_params.py` with 3 TDD tests verifying that stabilization parameters are NOT passed to `process()`.
- **Updated**: Existing `tests/api/test_api.py` tests to assert correct behavior (params NOT in `process()` call).

#### 📝 Key Commits in v1.0.2

`9563a34`, `2eafe64`, `8ca2c8f`, `e13005e`

---

### `v1.0.1` - *04-12-2025*

#### 🐛 Patch Release: WebUI ZIP Fixes, Benchmarking & Tooling

This release delivers a set of safe, backward-compatible improvements focused on WebUI stability, subtitle quality diagnostics, and local contributor tooling.

#### 🐛 Bug Fixes in v1.0.1

- **Fixed**: Duplicate `batch_summary.json` entries in WebUI batch ZIP archives.
  - **Issue**: Multi-file downloads sometimes produced ZIPs with duplicate `batch_summary.json`, triggering warnings and wasting space.
  - **Root Cause**: `BatchZipBuilder.add_summary()` was called manually before `build()`, which also adds a summary when `include_summary=True`.
  - **Solution**: Removed explicit `add_summary()` calls in WebUI handlers and now rely on `build()` to add the summary exactly once.

#### 🔧 Improvements in v1.0.1

- **Improved**: Benchmark result model now uses Pydantic v2 `ConfigDict` configuration, removing deprecation warnings while keeping models frozen.
- **Improved**: Segmentation and SRT quality utilities refined and covered by updated tests for better subtitle readability and diagnostics.
- **Improved**: Local tooling with `scripts/local-ci.sh` and CI-friendly `pdm` scripts for linting, formatting, tests, and coverage.

#### 📝 Key Commits in v1.0.1

`4220ebd`, `80fc087`, `4e02cd1`, `9565c66`, `c837998`, `2ded632`, `cb1f571`, `28b4405`

---

### `v1.0.0` - *18-09-2025*

#### 💥 Breaking Changes in v1.0.0

- `split_audio()` return type changed from `list[str]` to `list[tuple[str, float]]`.
  - Each tuple contains `(chunk_path, chunk_start_time_seconds)`.
- `merge_chunk_results()` parameter changed from `list[dict]` to `list[tuple[dict, float]]`.
  - Each tuple contains `(chunk_result_dict, chunk_start_time_seconds)`.

These functions are part of the public API (re-exported from `insanely_fast_whisper_rocm/__init__.py`). Any external code calling them must be updated.

#### 🐛 Bug Fixes in v1.0.0

- Fixed incorrect, overlapping timestamps in merged transcripts when audio is processed in chunks.
  - Root cause: chunk-level timestamps from Whisper are relative to each chunk (starting at 0.0), and previous merge logic simply concatenated segments.
  - Solution: offset segment and word timestamps by the chunk's start time before merging, producing a continuous, monotonic timeline.

#### 🛠 Migration Notes

- If you previously used `split_audio(path, ...) -> list[str]`, update your code to handle start times:

  ```python
  from insanely_fast_whisper_rocm.audio.processing import split_audio

  for chunk_path, start_time in split_audio(path, chunk_duration=30.0):
      ...
  ```

- If you previously used `merge_chunk_results(results: list[dict])`, now pass start times along with each result:

  ```python
  from insanely_fast_whisper_rocm.audio.results import merge_chunk_results

  merged = merge_chunk_results([(result_dict, start_time_seconds) for ...])
  ```

- Overlap handling: the pipeline currently uses `chunk_overlap=0.0`. If you enable overlaps externally, the merger offsets timestamps but does not de-duplicate overlapped text; consider adding your own de-duplication if needed.

#### 📝 Key Commits in v1.0.0

`<populate-on-release>`

---

### `v0.10.1` - *16-09-2025*

#### 🐛 **Bug Fix & Refactor Release: Docker Tags & Codebase Cleanup**

This patch addresses an issue with Docker image tags in `docker-compose` files and includes several documentation and code quality improvements from a recent codebase review.

#### 🐛 **Bug Fixes in v0.10.1**

- **Fixed**: Docker image tags in `docker-compose.dev.yaml` and `docker-compose.yaml` were not being updated consistently.
  - **Issue**: The `image:` tag was hardcoded, preventing it from being updated automatically during version bumps.
  - **Root Cause**: Manual versioning in docker-compose files.
  - **Solution**: Updated image tags to ensure consistency with the project version.

#### 🔧 **Improvements in v0.10.1**

- **Docs**: Added `AGENTS.md` to define strict coding standards for the project.
- **Docs**: Completed a full review and update of `project-overview.md` to ensure it is factually correct and aligned with the current codebase.

#### 📝 **Key Commits in v0.10.1**

`fc61fb2`, `b68d199`, `3064745`

---

### `v0.10.0` - *23-07-2025*

#### ✨ **Feature Release: M4A Audio Support & Stable-TS Integration**

This release introduces support for `.m4a` audio files and integrates `stable-ts` for more accurate word-level timestamp generation.

#### ✨ **New Features in v0.10.0**

- **Added**: Support for M4A audio files via `pydub`. The application can now process `.m4a` files seamlessly alongside other supported formats.
- **Added**: Integration with `stable-ts` for word-level timestamp stabilization. This can be enabled via the `--stabilize` flag in the CLI or corresponding options in the API and WebUI.

#### 🔧 **Improvements in v0.10.0**

- **Updated**: Dependencies to include `stable-ts`.
- **Improved**: The ASR backend now conditionally applies `stable-ts` to refine timestamps when requested.

#### 📝 **Key Commits in v0.10.0**

`2e132f1`, `aede396`, `44492cd`, `5e48f44`, `2fc9fa0`

---

### `v0.9.1` - *19-07-2025*

#### 🐛 **Bug Fix Release: Translation & Model Override**

This patch fixes translation to English across all Whisper models and ensures CLI `--model` overrides the `.env` default in the WebUI.

#### 🐛 **Bug Fixes in v0.9.1**

- **Fixed**: Multilingual detection logic causing translation block.
- **Fixed**: CLI model override not respected by WebUI.
- **Fixed**: Excessive warnings during translation now suppressed.

#### 🔧 **Improvements in v0.9.1**

- **Improved**: Warning handling to single concise notice.
- **Improved**: Added automatic timestamp disable for distil models.

#### 📝 **Key Commits in v0.9.1**

`70d744d`

---

### `v0.9.0` - *06-07-2025*

#### ✨ **Feature Release: CLI Benchmarking, Export Options & Translation**

This release introduces CLI benchmarking and performance monitoring, export format options for CLI, and translation via CLI. Numerous refactors and bug fixes improve reliability and maintainability.

#### ✨ **New Features in v0.9.0**

- **Added**: CLI benchmarking and performance monitoring utilities
- **Added**: Export format options for CLI (TXT, SRT, JSON)
- **Added**: Translation functionality via CLI

#### 🐛 **Bug Fixes in v0.9.0**

- **Fixed**: Formatter/import issues in WebUI handlers
  - **Issue**: Incorrect import and usage of formatters in WebUI
  - **Root Cause**: Outdated import paths and handler logic
  - **Solution**: Refactored imports and updated handler logic

#### 🔧 **Improvements in v0.9.0**

- **Refactored**: CLI command structure and benchmarking integration
- **Refactored**: Modularized formatters and audio processing

#### 📝 **Key Commits in v0.9.0**

`32c6d73`

---

### `v0.8.0` - *06-07-2025*

#### ✨ **Feature Release: Entrypoints, CLI Export Formats & Translation**

This release focuses on standardizing application entrypoints, enhancing the CLI with new export options, and adding translation capabilities.

#### ✨ **New Features in v0.8.0**

- **Added**: Export format options (`--export-txt`, `--export-srt`, `--export-json`) to the CLI for saving transcription results.
- **Added**: Translation functionality to the CLI via the `translate` command.
- **Enhanced**: Replaced `BetterTransformer` with native PyTorch `SDPA` for attention optimization, improving performance.

#### 🐛 **Bug Fixes in v0.8.0**

- **Fixed**: Corrected formatter imports and usage within the WebUI handlers to resolve import errors and ensure proper functionality.

#### 🔧 **Improvements in v0.8.0**

- **Refactored**: Standardized entrypoints for the API, WebUI, and CLI to use `__main__.py` modules, simplifying execution.
- **Refactored**: Moved `formatters.py` to `insanely_fast_whisper_rocm/core/` and updated all relevant imports.
- **Refactored**: Unified audio processing logic to support both transcription and translation seamlessly.

#### 📝 Key Commits in v0.8.0

`537e788`

---

### `v0.7.0` - *06-07-2025*

#### **♻️ Refactor Release: Unified Audio Processing for CLI**

#### ✨ **New Features & Improvement in v0.7.0s**

- **Unified CLI Audio Processing**: `transcribe_audio` replaced by `process_audio` in CLI facade, supporting both transcription and translation with a single interface.
- **Consistent CLI Commands**: Both `transcribe` and `translate` commands now use the unified method for improved maintainability.
- **Improved Documentation & Logging**: Updated docstrings, CLI help, and logging best practices.

#### 🔧 **Refactor & Maintenance in v0.7.0**

- **Refactored**: CLI and core pipeline to use a single audio processing method, reducing code duplication and confusion.
- **Updated**: Version badges and documentation for v0.7.0.

#### 📝 **Key Commits in v0.7.0**

`f09d3ce`

---

### `v0.6.0` - *05-07-2025*

#### ✨ Minor Release: Translation CLI, SDPA attention, ASR refactors

#### ✨ **New Features in v0.6.0**

- **Added**: Translation functionality to CLI tool
- **Enhanced**: Replaced BetterTransformer with native PyTorch SDPA attention

#### 🔧 **Improvements in v0.6.0**

- **Refactored**: ASR pipeline and backend for improved model loading and processing
- **Refactored**: Removed BetterTransformer configuration
- **Improved**: Language processing logic

#### 📝 **Commits**

`496f49a`, `bbd78e4`, `e16511b`, `3e78fe4`, `ec08c5e`

---

### `v0.5.0` - *07-06-2025*

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

#### 📝 Key Commits in v0.5.0

`2154bdf`

---

### `v0.4.1` - *06-06-2025*

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

`76252e4`

---

### `v0.4.0` - *06-06-2025*

#### ✨ Enhanced Versioning & Logging

#### ✨ New Features in v0.4.0

- **Enhanced**: Improved versioning system
- **Enhanced**: Standardized logging format across the application

#### 📝 Commits in v0.4.0

`6ad709c`

---

### `v0.3.1` - *04-06-2025*

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

`a4bbe37`

---

### `v0.3.0` - *27-05-2025*

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

`3e78875`

---

### `v0.2.1` - *29-05-2025*

#### ♻️ Import Standardization & Core Refinements

#### 🏗️ Major Changes in v0.2.1

- **Refactored all imports** to absolute paths for improved IDE support:

  ```python
  # ✅ New absolute imports
  from insanely_fast_whisper_rocm.core.pipeline import WhisperPipeline
  
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

### `v0.2.0` - *20-05-2025*

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

`9dfb30f`

---

### `v0.1.2` - *08-03-2025*

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

`3cd8552`

---

### `v0.1.1` - *19-01-2025*

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

`6e41010`

---

### `v0.1.0` - *18-01-2025*

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

`67667cd`

---
