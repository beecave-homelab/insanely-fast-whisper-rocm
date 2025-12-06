# Pull Request: Modular ROCm Refactor & Test Suite Expansion

## Summary

This PR refactors the original `insanely_fast_whisper_rocm` layout into a modular, ROCm-focused package `insanely_fast_whisper_rocm` and aligns the distribution/CLI name to `insanely-fast-whisper-rocm`. It introduces a clearer separation of concerns across `api`, `cli`, `core`, `audio`, `utils`, and `webui` modules, adds extensive test coverage, and modernizes configuration, documentation, and tooling.

The changes include:

- Migrating implementation code from the legacy `insanely_fast_whisper_api` package into the new `insanely_fast_whisper_rocm` package.
- Updating Dockerfiles, compose files, scripts, and requirements to use the new package and distribution names.
- Expanding and reorganizing tests to mirror the new modular structure and improve coverage.
- Updating documentation (README, VERSIONS, AGENTS, project-overview) and CI helper scripts.

---

## Files Changed

### Added

1. **`AGENTS.md`**  
   - Introduces detailed coding standards (Ruff, pytest, SOLID, configuration rules) for contributors and agents.

2. **`.windsurf/rules/testing.md`**  
   - Documents local CI workflow, test layout, and expectations for coverage and deterministic tests.

3. **`.windsurf/workflows/srt-benchmark.md`**  
   - Adds documentation for benchmark workflows, metrics, and scripts.

4. **`insanely_fast_whisper_rocm/api/*`**  
   - New modular API package including `__init__.py`, `__main__.py`, `app.py`, `dependencies.py`, `middleware.py`, `models.py`, `responses.py`, `routes.py`.
   - Replaces the legacy `insanely_fast_whisper_rocm` API implementation with a renamed and slightly reorganized structure.

5. **`insanely_fast_whisper_rocm/audio/*`**  
   - New `audio` package containing audio conversion, processing, and result helpers.
   - Encapsulates audio-specific concerns previously mixed into broader modules.

6. **`insanely_fast_whisper_rocm/benchmarks/*`**  
   - Adds a benchmarks package and collector utilities for structured benchmarking.

7. **`insanely_fast_whisper_rocm/cli/*`**  
   - New CLI package including `__init__.py`, `__main__.py`, `cli.py`, `commands.py`, `common_options.py`, `errors.py`, `facade.py`, `progress_tqdm.py`.
   - Centralizes CLI entry points, orchestration logic, error handling, and progress reporting.

8. **`insanely_fast_whisper_rocm/core/*`**  
   - New core package including `__init__.py`, `asr_backend.py`, `backend_cache.py`, `cancellation.py`, `errors.py`, `formatters.py`, `stable_ts.py`, `pipeline.py`, `progress.py`, `segmentation.py`, `storage.py`, and helpers.
   - Encapsulates ASR backend logic, chunking, segmentation, SRT/VTT formatting, caching, and storage.

9. **`insanely_fast_whisper_rocm/utils/*`**  
   - New utilities: `benchmark.py`, `constants.py`, `download_hf_model.py`, `env_loader.py`, `file_utils.py`, `filename_generator.py`, `format_time.py`, `formatting.py`, `srt_quality.py`, `timestamp_utils.py`.

10. **`insanely_fast_whisper_rocm/webui/*`**  
    - New WebUI package including `__init__.py`, `__main__.py`, `app.py`, `errors.py`, `handlers.py`, `merge_handler.py`, `ui.py`, `utils.py`, `zip_creator.py`.

11. **`requirements-reviewer.txt`**  
    - Adds a focused requirements set used by AI tooling/reviewers.

12. **`constraints-no-heavy.txt`**  
    - Constraints file to forbid heavy ML/RL packages during lightweight installs (e.g., for reviewers).

13. **`scripts/*`**  
    - `scripts/benchmark.sh`, `scripts/clean_benchmark.sh`, `scripts/clean_codebase.sh`, `scripts/clean_codebase_sorted.sh`, `scripts/compare_benchmarks.sh`, `scripts/hf_models.py`, `scripts/local-ci.sh`, `scripts/mww.sh`.
    - Provide helpers for CI-like checks, benchmarks, cleaning artifacts, and model utilities.

14. **`tests/api/*`, `tests/audio/*`, `tests/benchmarks/*`, `tests/cli/*`, `tests/core/*`, `tests/utils/*`, `tests/webui/*`, `tests/data/*`, `tests/helpers.py`**  
    - Large expansion and reorganization of tests to match the new modular package layout.
    - Adds focused tests for api routes, CLI entry points, ASR backend behavior, segmentation, formatting, benchmarks, utilities, and WebUI handlers.

15. **`to-do/rich-progress-callback-integration.md`**  
    - Adds a detailed implementation plan for future Rich-based progress callbacks spanning CLI and backend.

### Modified

1. **`.env.example`**  
   - Updated environment variable hints and example values to align with the new package name and configuration locations.

2. **`.gitignore`**  
   - Adjusted ignore rules for new directories, artifacts, benchmark outputs, and coverage files.

3. **`Dockerfile` / `Dockerfile.dev`**  
   - Updated to use `insanely_fast_whisper_rocm` as the working directory and entry point.
   - Ensures containers run the new CLI/WebUI module paths and install the updated package.

4. **`docker-compose.yaml` / `docker-compose.dev.yaml`**  
   - Adjusted service names, image names, commands, and volume paths to the new package/distribution naming.

5. **`pyproject.toml`**  
   - Renamed project to `insanely-fast-whisper-rocm`.
   - Set version to `2.0.0` (major bump for breaking rename).
   - Updated `[project.scripts]`, `[tool.pdm.scripts]`, Ruff config, and type-checking paths to the new package.

6. **`pdm.lock`, `requirements-all.txt`**  
   - Regenerated / updated dependency lockfile and aggregate requirements for the new layout.

7. **`pyrightconfig.json`**  
   - Updated `include` and `extraPaths` to point at `insanely_fast_whisper_rocm` instead of the legacy package.

8. **`README.md`**  
   - Updated project name, badges, installation and usage examples to reference `insanely-fast-whisper-rocm` / `insanely_fast_whisper_rocm`.
   - Documents new CLI commands and module paths.

9. **`VERSIONS.md`**  
   - Adds `v2.0.0` entry as the current release, describing the rename, modular refactor, and major features.
   - Keeps `v1.0.2` and earlier versions as historical.

10. **`project-overview.md`**  
    - Updates current version badge and summary for `v2.0.0`.
    - Describes the modular layout, new testing strategy, and benchmark tooling.

11. **`scripts/convert_json_to_txt_srt.py`**  
    - Adjusted imports and paths to use the new package structure.

12. **`scripts/setup_config.py`**  
    - Updated configuration directory naming and defaults to match the ROCm-focused package.

13. **`tests/conftest.py`, `tests/test_cuda.py`, `tests/test_webui.py`**  
    - Updated to import from `insanely_fast_whisper_rocm` and integrate with the new test modules.

### Deleted

1. **Legacy `insanely_fast_whisper_rocm/*` package**  
   - Removed old `api`, `cli`, `core`, `utils`, `webui`, and related modules under the previous package name.
   - Functionality is preserved and expanded in the new `insanely_fast_whisper_rocm` package.

2. **Legacy monolithic tests**  
   - Removed older test files: `tests/test_api.py`, `tests/test_cli.py`, `tests/test_cli_exports.py`, `tests/test_download_hf_model.py`, `tests/test_stable_ts.py`, `tests/test_response_formats.py`, and others now covered by new modular tests.
   - Replaced by more granular tests under `tests/api`, `tests/cli`, `tests/core`, `tests/utils`, and `tests/webui`.

---

## Code Changes

### `insanely_fast_whisper_rocm/core/pipeline.py`

```python
# Example: high-level pipeline refactor (illustrative snippet)
class WhisperPipeline:
    """Coordinate ASR inference, segmentation, and formatting."""

    def transcribe(self, audio_path: Path, *, task: str = "transcribe") -> dict[str, Any]:
        """Run the end-to-end transcription pipeline for the given audio file."""
        # Delegates to ASR backend, segmentation, and SRT/VTT formatting helpers.
```

- Introduces a clearer pipeline orchestration class that wires together backend inference, segmentation, and output formatting.
- Centralizes options and configuration for downstream consumers (CLI, API, WebUI).

### `insanely_fast_whisper_rocm/cli/commands.py`

```python
@click.group()
@click.version_option(__version__, prog_name="insanely-fast-whisper-rocm")
def main() -> None:
    """Entry point for the insanely-fast-whisper-rocm CLI."""
```

- Defines the main click group for the new CLI.
- Adds subcommands for transcription, translation, WebUI launch, and benchmark helpers.
- Normalizes options and flags across commands.

### `insanely_fast_whisper_rocm/api/routes.py`

```python
@router.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe(
    request: TranscriptionRequest,
    deps: ApiDependencies = Depends(get_dependencies),
) -> TranscriptionResponse:
    """Transcribe audio using the configured ASR backend."""
```

- Exposes typed FastAPI routes for transcription and translation.
- Uses dependency injection to access the pipeline, storage, and configuration.

### `insanely_fast_whisper_rocm/utils/env_loader.py`

```python
def load_env() -> None:
    """Load environment configuration from XDG config dir and `.env` files."""
    # Resolves `~/.config/insanely-fast-whisper-rocm/` and applies dotenv loading.
```

- Centralizes environment loading and configuration discovery for the renamed app.

### `tests/cli/test_cli.py`

```python
def test_main_help(runner: CliRunner) -> None:
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "insanely-fast-whisper-rocm" in result.output
```

- Ensures the new CLI entry point works as expected and surfaces the correct name.

> Additional files follow similar patterns: updated imports to the new package, expanded tests, and clearer separation of concerns.

---

## Reason for Changes

- **Package/Distribution Rename**: Align the Python import path and distribution/CLI name with the ROCm-focused branding `insanely_fast_whisper_rocm` / `insanely-fast-whisper-rocm`.
- **Modular Refactor**: Improve maintainability by separating concerns into dedicated packages (`api`, `cli`, `core`, `audio`, `utils`, `webui`, `benchmarks`).
- **Testing & Tooling**: Expand test coverage, introduce deterministic local CI scripts, and document contributor expectations.
- **Versioning**: Bump to `v2.0.0` as a major release with breaking changes and clear migration notes.

---

## Impact of Changes

### Positive Impacts

- **Clearer architecture**: Easier to navigate and reason about the codebase due to modular layout.
- **Improved test coverage**: Granular tests across modules reduce regression risk (local CI reports ~90% coverage overall).
- **Better developer experience**: Documented workflows, helper scripts, and coding standards make contributions safer and more consistent.
- **User-facing clarity**: Consistent naming across imports, CLI commands, Docker, and configuration directories.

### Potential Issues

- **Breaking imports**: External code importing `insanely_fast_whisper_api` must migrate to `insanely_fast_whisper_rocm`.
- **Config directory migration**: Existing configuration under `~/.config/insanely-fast-whisper-rocm/` needs to be moved or recreated under `~/.config/insanely-fast-whisper-rocm/`.
- **Ecosystem updates**: Downstream scripts, Docker deployments, or automation referring to the old distribution/CLI name must be updated.

---

## Test Plan

1. **Unit Testing**  
   - Run the full test suite:
     - `pdm run pytest --maxfail=1 -q`
     - `pdm run pytest --cov=. --cov-report=term-missing:skip-covered --cov-report=xml`
   - Key coverage areas:
     - API routes and models under `tests/api/`.
     - CLI behavior and entry points under `tests/cli/`.
     - Core ASR backend, segmentation, and formatting under `tests/core/`.
     - Utility helpers and configuration loaders under `tests/utils/`.
     - WebUI behavior under `tests/webui/`.

2. **Integration Testing**  
   - Docker/WebUI:
     - `docker compose up --build -d`
     - Access the WebUI and perform a sample transcription.
   - CLI smoke tests (via PDM):
     - `pdm run insanely-fast-whisper-rocm --help`
     - `pdm run insanely-fast-whisper-cli --help`
     - `pdm run insanely-fast-whisper-webui --help`
   - Module entry points:
     - `pdm run python -m insanely_fast_whisper_rocm.api --help`
     - `pdm run python -m insanely_fast_whisper_rocm.webui --help`
     - `pdm run python -m insanely_fast_whisper_rocm.cli --help`

3. **Manual Testing**  
   - Verify migration:
     - Install the package locally and ensure imports from `insanely_fast_whisper_rocm` succeed.
     - Confirm config is read from `~/.config/insanely-fast-whisper-rocm/`.
   - CLI usage:
     - Run a short audio transcription and ensure outputs (SRT/VTT/TXT) are created as expected.
   - WebUI usage:
     - Upload a test file via WebUI, verify progress and resulting subtitles.

---

## Additional Notes

- This PR is the foundational modular refactor and rename; follow-up work (e.g., Rich-powered progress callbacks, JSONL progress reporters) is tracked in `to-do/rich-progress-callback-integration.md`.
- Downstream consumers should follow the migration notes in `VERSIONS.md` and `project-overview.md` when upgrading from `v1.x` to `v2.0.0`.
