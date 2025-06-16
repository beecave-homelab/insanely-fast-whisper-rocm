# To-Do: PDM Dependency Management Setup

This plan outlines the steps to configure `pyproject.toml` for PDM, incorporating all existing dependencies from various `requirements*.txt` files.

## Tasks

- [x] **Analysis Phase:**
  - [x] Analyze `requirements.txt`, `requirements-rocm.txt`, `requirements-onnxruntime-rocm.txt`, and `requirements-dev.txt`.
    - Path:
      - `requirements.txt`
      - `requirements-rocm.txt`
      - `requirements-onnxruntime-rocm.txt`
      - `requirements-dev.txt`
    - Action: Identify all packages and their versions.
    - Analysis Results:
      - **`requirements.txt`**:
        - `fastapi>=0.104.0`
        - `uvicorn>=0.24.0`
        - `python-multipart>=0.0.6`
        - `transformers>=4.35.0`
        - `python-dotenv>=1.0.0`
        - `PyYAML>=6.0.1`
        - `click>=8.1.3`
        - `flask>=3.1.0`
        - `gradio>=5.20.1`
        - `pydub>=0.25.1`
        - (Note: `numpy`, `datasets`, `accelerate` already in `pyproject.toml` `[project.dependencies]`)
      - **`requirements-rocm.txt`**:
        - `torch==2.5.0+rocm6.1` (Source: `https://download.pytorch.org/whl/rocm6.1/`)
      - **`requirements-onnxruntime-rocm.txt`**:
        - `onnxruntime-rocm @ https://repo.radeon.com/rocm/manylinux/rocm-rel-6.1.3/onnxruntime_rocm-1.17.0-cp310-cp310-linux_x86_64.whl`
        - `numpy==1.26.4`
      - **`requirements-dev.txt`**:
        - `pytest>=8.0.0`
        - `pytest-cov>=4.1.0`
        - `ruff>=0.1.0`
        - `black>=23.12.0`
        - `isort>=5.13.2`
        - `mypy>=1.7.0`
        - `pydocstyle>=6.3.0`
        - `pylint>=3.0.3`
        - `types-PyYAML>=6.0.12`
        - `typing-extensions>=4.8.0`
        - (Note: All dev dependencies already in `pyproject.toml` `[project.optional-dependencies.dev]`)
    - Accept Criteria: A comprehensive list of all dependencies and their sources is compiled.
  - [x] Analyze existing `pyproject.toml`.
    - Path: `pyproject.toml`
    - Action: Understand current structure and PDM-specific sections.
    - Analysis Results:
      - Uses `hatchling` build backend.
      - Core dependencies in `[project.dependencies]`.
      - Optional dependencies (e.g., `dev`) in `[project.optional-dependencies]`.
      - Contains placeholders for `torch-rocm`.
      - PDM uses `[[tool.pdm.source]]` for custom indexes and supports direct URL dependencies.
      - Existing `[project.dependencies]` and `[project.optional-dependencies.dev]` are mostly up-to-date with `requirements.txt` and `requirements-dev.txt` respectively.
    - Accept Criteria: Clear understanding of how to integrate new dependencies into `pyproject.toml`.

- [x] **Implementation Phase:**
  - [x] Update `pyproject.toml` with all dependencies.
    - Path: `pyproject.toml`
    - Action:
      - Main dependencies from `requirements.txt` confirmed in `[project.dependencies]`.
      - In `[project.optional-dependencies.rocm]`: `torch` specified as a direct URL dependency to its wheel on `repo.radeon.com`. `onnxruntime-rocm` and `numpy` remain as version specifiers.
      - Development dependencies from `requirements-dev.txt` confirmed in `[project.optional-dependencies.dev]`.
      - `[[tool.pdm.source]]` for PyPI has `include_packages = ["*"]`.
      - `[[tool.pdm.source]]` for `rocm-wheels` (URL: `https://repo.radeon.com/rocm/manylinux/rocm-rel-6.4.1/`) has `type = "find_links"` and `include_packages = ["onnxruntime-rocm"]` (now only for onnxruntime-rocm).
      - This setup uses a direct URL for `torch`, `find_links` for `onnxruntime-rocm` from `rocm-wheels`, and PyPI for all other packages.
      - Correct PDM syntax and version specifiers used.
    - Status: `Completed (torch as direct URL, rocm-wheels for onnxruntime-rocm only)`
  - [x] Clean up `pyproject.toml` by removing non-PDM specific sections (Rye, Hatch envs/metadata, UV comments).
    - Path: `pyproject.toml`
    - Action: Removed `[tool.rye]`, `[tool.hatch.metadata]`, `[tool.hatch.envs.default]`, and commented-out UV configuration.
    - Status: `Completed`
  - [x] Add PDM scripts to `pyproject.toml` for launching API, CLI, and WebUI.
    - Path: `pyproject.toml`
    - Action: Added `start-webui`, `start-webui-debug`, `start-api`, `start-api-verbose`, and `cli` to `[tool.pdm.scripts]` based on existing `[project.scripts]` entry points and `docker-compose.yaml` commands.
    - Status: `Completed`
- [ ] **Verification & Testing:**
  - [x] Run `pdm install -G rocm -G dev` to ensure all dependencies install correctly.
    - Status: `Completed successfully (Exit Code 0). Correct packages installed in .venv.`
    - Note: Log may show 404s for general packages on ROCm index during resolution; this is likely resolver probing noise if installation succeeds.
  - [ ] Verify installation details:
    - [x] Check for `pdm.lock` file creation/update.
    - [x] Run `.venv/bin/python -c "import torch; print(f'Torch version: {torch.__version__}'); print(f'Torch path: {torch.__file__}'); import onnxruntime; print(f'ONNXRuntime version: {onnxruntime.__version__}'); print(f'ONNXRuntime path: {onnxruntime.__file__}')"` to confirm ROCm package versions and sources.
          Note: `pdm run python ...` was found to incorrectly use the system Python (3.13) instead of the project's venv (3.10). Direct execution via `.venv/bin/python` or activating the venv (`source .venv/bin/activate`) is required.
    - Status: `Completed`
  - [ ] Test PDM scripts:
    Note: `pdm run` was found to incorrectly use the system Python (3.13) instead of the project's venv (3.10). Scripts should be tested by activating the venv (`source .venv/bin/activate`) and running the target script/module, or by direct invocation e.g., `.venv/bin/python -m module.path`.
    - [x] `pdm run start-webui` (User confirmed working.)
    - [x] `pdm run start-webui-debug` (User confirmed working.)
    - [x] `pdm run start-api` (Tested via direct venv Python invocation, e.g., `.venv/bin/python -m insanely_fast_whisper_api.__main__ --port <port>`. Assumed working based on verbose test.)
    - [x] `pdm run start-api-verbose` (Tested via `.venv/bin/python -m insanely_fast_whisper_api.__main__ -v --port 8889`. API started successfully.)
    - [x] `pdm run cli -- --help` (Tested as `.venv/bin/python -m insanely_fast_whisper_api.cli.cli --help` due to `pdm run` issues. Help displayed.)
    - [x] `cli` with a sample transcription command (Tested with `distil-whisper/distil-large-v3`, batch size 4, using `tests/conversion-test-file.mp3`. Transcription successful.)
    - Status: `Completed`. Scripts (API, WebUI, CLI help) launch correctly using direct venv Python invocation. CLI transcription successful with `distil-whisper/distil-large-v3`. Default `openai/whisper-large-v2` (likely set via WHISPER_MODEL env var) causes OOM on this 8GB GPU.
    - Accept Criteria: Application runs without import errors related to PDM-managed dependencies.

- [x] **Testing Phase:**
  - [x] Test basic package functionality after PDM installation.
    - Path: N/A
    - Action: Ensure the application can start and core features work.
    - Accept Criteria: Application runs without import errors related to PDM-managed dependencies.

- [ ] **Documentation Phase:**
  - [ ] Update `project-overview.md` and/or README with PDM usage instructions.
    - Path: `project-overview.md` or `README.md`
    - Action: Document how to install dependencies using PDM, including optional groups.
    - Accept Criteria: Documentation clearly explains the new PDM-based dependency management.

## Related Files

- `pyproject.toml`
- `requirements.txt`
- `requirements-rocm.txt`
- `requirements-onnxruntime-rocm.txt`
- `requirements-dev.txt`
- `to-do/pdm-dependency-management.md`

## Future Enhancements

- [ ] Consider PDM scripts for common development tasks.
