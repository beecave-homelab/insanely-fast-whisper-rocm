# Project Restructuring Plan

## Overview

This document outlines the step-by-step plan to restructure the Insanely Fast Whisper ROCm project for better maintainability and scalability.

## Phase 1: Project Setup

### 1.1 Directory Structure

```md
├── .github/
│   └── workflows/
│       └── tests.yml
├── config/
│   ├── __init__.py
│   ├── settings.py
│   └── logging_config.py
├── src/
│   ├── core/
│   ├── models/
│   ├── services/
│   ├── utils/
│   └── web/
├── tests/
│   ├── unit/
│   └── integration/
├── scripts/
├── .env.example
└── pyproject.toml
```

**Status**: Directory structure has been created as specified.

### 1.2 Initial Setup

- [x] Create new directory structure
- [x] Set up `pyproject.toml` with project metadata and dependencies
- [x] Create `.gitignore` file
- [x] Set up pre-commit hooks

**Notes**:

- The `.gitignore` has been restructured to exclude everything by default and only include necessary files
- Pre-commit hooks are configured in `.pre-commit-config.yaml`

## Phase 2: Core Functionality Migration

### 2.1 Configuration

- [x] Create `config/settings.py` with Pydantic settings
- [x] Move environment variables to centralized configuration
- [x] Implement logging configuration

**Notes**:

- Settings are managed using Pydantic with environment variable support
- Logging configuration includes Rich for enhanced console output
- Unit tests for settings and logging configuration are in place

### 2.2 Core Modules

- [ ] **Migrate `insanely-fast-whisper.py` (referred to as `main.py` previously) functionality to new project structure:**
  - [x] **A. Configuration & Setup (`config/` and initial setup):**
    - [x] Centralize all settings in `config/settings.py` using Pydantic (replaces `get_env_config` from `insanely-fast-whisper.py:41-50`, `constants.py` defaults, and Click option defaults from `insanely-fast-whisper.py:173-206`).
    - [x] Manage logging strategy via `config/logging_config.py` (replaces `transformers_logging` and `warnings` setup from `insanely-fast-whisper.py:14-18`).
    - [x] Ensure `os.environ["TOKENIZERS_PARALLELISM"] = "false"` is handled in `TranscriptionEngine.__init__`.
  - [x] **B. Custom Exceptions (`src/core/transcription.py`):**
    - [x] Define `TranscriptionError(Exception)` (moved from `click.ClickException` for better separation of concerns).
    - [x] Define `DeviceNotFoundError(TranscriptionError)` for device-related issues.
  - [x] **C. Utility Functions (`src/utils/`):**
    - [x] Create `src/utils/device_utils.py` (or similar):
      - [x] Implement `convert_device_string(device_id: str) -> str` (from `insanely-fast-whisper.py:29-39`).
      - [x] Implement device availability check logic (e.g., `torch.cuda.is_available()`, `torch.backends.mps.is_available()`) (from `insanely-fast-whisper.py:86-91`).
  - [x] **D. Core Transcription Logic (`src/core/transcription.py`):**
    - [x] File `core/transcription.py` created.
    - [x] Define `TranscriptionConfig` Pydantic model with all necessary parameters.
    - [x] Define `TranscriptionResult` Pydantic model for structured output.
    - [x] Create `TranscriptionEngine` class with the following features:
      - [x] **Constructor (`__init__`):**
        - [x] Store configuration and initialize logging.
        - [x] Set environment variables.
        - [x] Validate device availability.
        - [x] Initialize the transformers pipeline with progress reporting.
      - [x] **Method `transcribe_audio`:**
        - [x] Implement audio file validation.
        - [x] Execute the ASR pipeline with configurable parameters.
        - [x] Handle exceptions and provide meaningful error messages.
        - [x] Calculate and include runtime metrics.
        - [x] Return a `TranscriptionResult` object with the transcription.
      - [x] Add comprehensive type hints and docstrings to the class and its methods (extends original general task).
      - [x] Implement detailed logging for key operations within the engine (replaces `click.secho` for internal status).
  - [x] **E. File Handling & Input Validation (`src/core/file_handlers.py` and/or `src/utils/validation.py`):**
    - [x] File `core/file_handlers.py` created.
    - [x] Implement audio file format validation against `constants.SUPPORTED_AUDIO_FORMATS` (from `insanely-fast-whisper.py:243-248`). This could be a utility function used by the CLI or `TranscriptionEngine`.
    - [x] Support for additional audio formats (if any beyond initial `constants.SUPPORTED_AUDIO_FORMATS`) to be investigated and added here.
    - [x] Implement other file system operations, directory monitoring, file type detection, safe file operations as originally planned if still relevant beyond audio validation.
  - [x] **F. Command-Line Interface (`scripts/cli.py`):**
    - [x] Create `scripts/cli.py` with a Click-based CLI.
    - [x] Implement Click command group with `@click.group()`.
    - [x] Implement `transcribe` command with all necessary options:
      - [x] Audio file input validation.
      - [x] Model configuration options (model, device, batch size, etc.).
      - [x] Output file handling.
      - [x] Progress reporting with rich.
      - [x] Error handling and user feedback.
    - [x] Add entry point in `pyproject.toml` for easy command-line access.
    - [x] Implement proper logging and error handling.
    - [x] Add rich console output for better user experience.
    - [x] Implement proper exit codes for different error conditions.

- [x] **Output Conversion (`core/conversion.py` - For Future Enhancements, not in `insanely-fast-whisper.py`):**
  - [x] File `core/conversion.py` created.
  - [ ] Design and implement logic for converting `TranscriptionResult` (specifically its chunks) to TXT, SRT, VTT formats.
  - [ ] Add format validation for conversion outputs.
  - [ ] Implement formatters for each output type.
  - [ ] Add batch conversion functionality for multiple `TranscriptionResult` objects.

### 2.3 Core Module Enhancements

- [ ] All relevant enhancements from the original `insanely-fast-whisper.py` (like BetterTransformer, chunk length, task type, specific error handling, language spec, model config) are now integrated into the detailed tasks for `src/core/transcription.py` and `scripts/cli.py` above. Review if any abstract concepts like 'batch processing of audio files' (original line 67) need further distinct tasks beyond what `TranscriptionEngine`'s batch_size implies (e.g., a higher-level service for managing multiple files).

## Current Status Summary

### Completed Tasks

- Successfully restructured the project into a modular Python package
- Implemented core transcription functionality in `src/core/transcription.py`
- Created a robust CLI interface in `scripts/cli.py` with rich output
- Added comprehensive error handling and logging
- Implemented file handling and validation
- Set up proper project configuration and dependencies
- Added type hints and docstrings throughout the codebase
- Fixed linting and formatting issues
- Implemented proper exit codes for different error conditions
- Added rich console output for better user experience
- Completed the main transcription workflow

### Next Steps

- **Testing**: Write unit and integration tests for the core functionality
- **Documentation**: Update README with new usage instructions
- **Docker**: Update Dockerfile to use the new project structure
- **CI/CD**: Set up GitHub Actions for testing and deployment

1. **Testing**:
   - [ ] Write unit tests for the core functionality
   - [ ] Add integration tests for the CLI
   - [ ] Test with different audio formats and languages
   - [ ] Test error conditions and edge cases
2. **Documentation**:
   - [ ] Update README with new usage instructions
   - [ ] Add examples for common use cases
   - [ ] Document environment variables and configuration options
3. **Packaging**:
   - [ ] Update `pyproject.toml` with all dependencies
   - [ ] Test installation from source
   - [ ] Consider publishing to PyPI
4. **Performance Optimization**:
   - [ ] Profile the transcription process
   - [ ] Optimize memory usage for large files
   - [ ] Implement batch processing for multiple files
5. **Docker Support**:
   - [ ] Update Dockerfile to use the new project structure
   - [ ] Optimize the Docker image size
   - [ ] Add multi-stage builds for smaller images

## Phase 3: Services Layer [P0]

### 3.1 Processing Service

- [ ] Create `services/processor.py`
  - [ ] Handle file processing pipeline
  - [ ] Manage worker threads
  - [ ] Progress tracking
  - [ ] Task prioritization
  - [ ] Resource management

### 3.2 Conversion Service

- [ ] Create `services/converter.py`
  - [ ] Format conversion pipeline
  - [ ] Batch processing
  - [ ] Error recovery
  - [ ] Format validation
  - [ ] Progress reporting

### 3.3 Error Handling [P1]

- [ ] Define error hierarchy
  - [ ] Input validation errors
  - [ ] Processing errors
  - [ ] System errors
- [ ] Implement error recovery
  - [ ] Retry mechanisms
  - [ ] Fallback strategies
  - [ ] Error reporting
- [ ] Error logging
  - [ ] Structured logging
  - [ ] Error context
  - [ ] Alerting thresholds

### 3.4 Performance Optimization [P2]

- [ ] Profile critical paths
  - [ ] Transcription pipeline
  - [ ] File I/O operations
  - [ ] Memory usage
- [ ] Implement optimizations
  - [ ] Caching
  - [ ] Batch processing
  - [ ] Parallel processing
- [ ] Monitoring
  - [ ] Performance metrics
  - [ ] Resource utilization
  - [ ] Bottleneck detection

## Phase 4: Web Interface [P1]

### 4.1 Web Application

- [ ] Create `web/app.py`
  - [ ] Gradio interface
    - [ ] File upload component
    - [ ] Progress display
    - [ ] Results visualization
  - [ ] User authentication
  - [ ] Session management

### 4.2 API Endpoints

- [ ] Create `web/routes.py`
  - [ ] File operations
    - [ ] Upload
    - [ ] Status check
    - [ ] Download
  - [ ] Processing
    - [ ] Start processing
    - [ ] Check status
    - [ ] Cancel processing
  - [ ] System
    - [ ] Health check
    - [ ] Metrics
    - [ ] Configuration

### 4.3 API Documentation [P2]

- [ ] OpenAPI/Swagger documentation
- [ ] Example requests/responses
- [ ] Authentication details
- [ ] Rate limiting information

## Phase 5: Testing

### 5.1 Unit Tests [P0]

- [x] Test configuration settings
- [x] Test logging setup
- [ ] Test core functionality
  - [ ] Transcription module
  - [ ] Conversion module
  - [ ] File handlers
- [ ] Test utility functions
- [ ] Test error cases
- [ ] Test model configurations
- [ ] Test format validations

### 5.2 Integration Tests [P1]

- [ ] Test file processing pipeline
  - [ ] Audio file ingestion
  - [ ] Batch processing
  - [ ] Format conversion flow
- [ ] Test web interface integration
  - [ ] File uploads
  - [ ] Progress reporting
  - [ ] Error handling
- [ ] Test API endpoints
  - [ ] Authentication
  - [ ] Rate limiting
  - [ ] Response formats

### 5.3 Performance Testing [P1]

- [ ] Benchmark transcription speed
  - [ ] Small files (<1 min)
  - [ ] Medium files (1-5 min)
  - [ ] Large files (>5 min)
- [ ] Memory usage profiling
  - [ ] Single file processing
  - [ ] Batch processing
- [ ] Load testing
  - [ ] Concurrent users
  - [ ] Large batch processing
  - [ ] API endpoint stress testing

**Notes**:

- Test infrastructure uses `pytest` with `pytest-cov` for coverage
- Test configuration in `pytest.ini` and `conftest.py`
- Performance benchmarks should be run on consistent hardware

## Phase 6: Documentation [P1]

### 6.1 Code Documentation

- [x] Add docstrings to configuration modules
- [ ] Add docstrings to all functions
  - [ ] Core modules
  - [ ] Services
  - [ ] Web interface
- [ ] Add module-level documentation
  - [ ] Purpose
  - [ ] Dependencies
  - [ ] Usage examples
- [ ] Document public API
  - [ ] REST endpoints
  - [ ] Python API
  - [ ] WebSocket interface (if any)

### 6.2 User Documentation [P1]

- [x] Initial README.md structure
- [ ] Setup Guide
  - [x] Basic installation
  - [ ] System requirements
  - [ ] Configuration options
  - [ ] Troubleshooting
- [ ] Usage Guide
  - [ ] Basic usage
  - [ ] Advanced features
  - [ ] Examples
  - [ ] Common issues
- [ ] API Reference
  - [ ] Endpoint documentation
  - [ ] Request/response formats
  - [ ] Authentication

### 6.3 Developer Documentation [P2]

- [ ] Architecture overview
  - [ ] Component diagram
  - [ ] Data flow

## Phase 7: Docker Setup and Development Workflow

### 7.1 Docker Development Environment

- [ ] Update Dockerfile to support development workflow:
  - [ ] Ensure all development dependencies are installed in the development image
  - [ ] Set up volume mounts for live code reloading
  - [ ] Configure debugging tools and ports

- [ ] Update docker-compose-dev.yaml:
  - [ ] Add service for development with proper volume mounts
  - [ ] Configure environment variables for development
  - [ ] Set up debugging ports

### 7.2 Development Workflow

- [ ] Document Docker-based development workflow:
  - [ ] Building the development container
  - [ ] Running tests inside the container
  - [ ] Debugging with VS Code
  - [ ] Running the application in development mode

- [ ] Testing in Docker:
  - [ ] Document how to run unit and integration tests in the container
  - [ ] Add test coverage reporting
  - [ ] Set up test automation in the container

### 7.3 Production Build

- [ ] Optimize Dockerfile for production:
  - [ ] Multi-stage build to reduce image size
  - [ ] Security hardening
  - [ ] Proper user permissions
  - [ ] Health checks

## Phase 8: Development Setup

- [ ] Environment setup
  - [x] Document Docker-based development environment setup
  - [ ] Document required VS Code extensions
  - [ ] Document recommended settings for development

- [ ] Testing guidelines
  - [ ] Document how to run tests in Docker
  - [ ] Add test coverage reporting
  - [ ] Document test writing guidelines

- [ ] Code style guide
  - [ ] Document code style rules
  - [ ] Document pre-commit hooks
  - [ ] Document commit message conventions

- [ ] Contribution guidelines
  - [ ] Pull request process
  - [ ] Code review checklist
  - [ ] Release process

**Notes**:

- Use Sphinx for API documentation
- Include code examples in docstrings
- Keep documentation in sync with code changes

## Phase 7: Deployment

### 7.1 Containerization

- [ ] Modify `Dockerfile` to use local project source:
  - [ ] Remove `pipx install insanely-fast-whisper` and related `pipx runpip` commands.
  - [ ] Ensure `COPY` commands correctly transfer local project files (e.g., `src/`, `pyproject.toml`, `requirements.txt`) into the image.
  - [ ] Implement local project dependency installation (e.g., via `pip install .` if `pyproject.toml` defines an installable package, or `pip install -r requirements.txt`).
  - [ ] Update `ENTRYPOINT` and/or `CMD` to execute the restructured application's entry point (e.g., `python3 -m src.web.app` for the web interface, or a designated CLI script).
- [ ] Update docker-compose.yml
- [ ] Add health checks

### 7.2 CI/CD

- [ ] Set up GitHub Actions
- [ ] Add test automation
- [ ] Add release process

## Phase 8: Final Steps [P1]

### 8.1 Code Quality

- [ ] Code review
  - [ ] Peer review process
  - [ ] Automated code analysis
  - [ ] Security audit
- [ ] Testing
  - [ ] Full test suite
  - [ ] Performance benchmarks
  - [ ] Edge case validation
- [ ] Documentation
  - [ ] Update README
  - [ ] Update API docs
  - [ ] Add/update examples

### 8.2 Release Preparation [P1]

- [ ] Version bumping
- [ ] Changelog update
- [ ] Dependency updates
- [ ] Release notes

### 8.3 Migration [P0]

- [ ] Data migration plan
- [ ] Downtime planning
- [ ] Rollback strategy
- [ ] Post-migration validation

### 8.4 Post-Release [P2]

- [ ] Monitoring setup
- [ ] Performance tracking
- [ ] User feedback collection
- [ ] Bug triage process

- [ ] Create migration guide
- [ ] Update dependent projects
- [ ] Archive old code
- [ ] Update `project-overview.md` with new project structure and components

### 8.5 User Approval Points

- [ ] Phase 1 completion review
- [ ] Phase 2-3 core functionality review
- [ ] Phase 4-5 UI/Testing review
- [ ] Final review before deployment
  - [ ] Code review sign-off
  - [ ] Documentation review
  - [ ] Performance testing results

## Dependencies

### Required

- Python 3.9+
- Pydantic
- Gradio
- pytest
- mypy
- black
- isort

## Timeline

| Phase | Estimated Time | Status |
|-------|----------------|--------|
| 1. Setup | 1 day | ✅ Completed |
| 2. Core Migration | 2 days | 🟡 In Progress |
| 3. Services | 1 day | ⏳ Not Started |
| 4. Web Interface | 1 day | ⏳ Not Started |
| 5. Testing | 2 days | 🟡 Partially Started |
| 6. Documentation | 1 day | 🟡 Partially Started |
| 7. Deployment | 1 day | ⏳ Not Started |
| 8. Final Steps | 1 day | ⏳ Not Started |

**Legend**:

- ✅ Completed
- 🟡 In Progress
- ⏳ Not Started

## Notes

- Each task should be completed in its own branch
- Create PRs for review after each major phase
- Update documentation as you go
- Keep commits atomic and well-described
- Await user approval at designated checkpoints before proceeding
- Update `project-overview.md` to reflect any structural changes
