# Pull Request: Merge dev into main — ASR robustness, pipeline fixes, and test hardening

## Summary

This PR merges the `dev` branch into `main` focusing on substantive code changes that improve ASR robustness, fix generation/timestamp edge cases, streamline CLI and utilities, and harden tests. It also includes housekeeping for configuration and CI hygiene. Documentation-only changes are de-emphasized per request.

Commits between `12b1cacb86be0a066fe4f06652121c4c6a1daa0d` and `44bd33d96db8ca924fb151d3acb82aa2c9fa8b78` have been considered and the relevant code changes are summarized below.

---

## Files Changed

### Added

1. `scripts/clean_codebase_sorted.sh`  
   - Script to sort and clean the codebase (lint/order consistency support).
2. `tests/test_asr_backend_generation_config.py`  
   - Adds explicit tests for generation-config backfilling and task/language mapping safety.
3. `tests/test_asr_backend_timestamp.py`  
   - Adds tests that cover fallback behavior when word-level timestamps fail.

### Modified

1. `insanely_fast_whisper_rocm/core/asr_backend.py`  
   - Major robustness improvements for generation-config handling, timestamp safety checks, and graceful fallbacks.
2. `insanely_fast_whisper_rocm/core/pipeline.py`, `core/utils.py`, `core/formatters.py`, `core/storage.py`, `core/integrations/stable_ts.py`  
   - Refactors/readability improvements and alignment with updated backend behavior.
3. `insanely_fast_whisper_rocm/audio/processing.py`, `audio/conversion.py`, `audio/results.py`  
   - Improved error handling and conversion logic; clarified paths for temp files.
4. `insanely_fast_whisper_rocm/cli/cli.py`, `cli/commands.py`, `cli/common_options.py`, `cli/facade.py`  
   - CLI refactors for clarity and argument handling improvements.
5. `insanely_fast_whisper_rocm/api/app.py`, `api/routes.py`, `api/models.py`, `api/responses.py`, `api/__main__.py`  
   - API plumbing adjustments to match backend and utils changes.
6. `insanely_fast_whisper_rocm/utils/constants.py`, `utils/env_loader.py`, `utils/file_utils.py`, `utils/filename_generator.py`, `utils/format_time.py`, `utils/benchmark.py`, `utils/download_hf_model.py`  
   - Moved environment-dependent constants behind `env_loader` and general utility refactors.
7. `insanely_fast_whisper_rocm/webui/app.py`, `webui/ui.py`, `webui/handlers.py`, `webui/merge_handler.py`, `webui/utils.py`, `webui/zip_creator.py`, `webui/errors.py`  
   - Keeps WebUI in sync with backend/CLI changes; stability and consistency updates.
8. `tests/test_api_integration.py`, `tests/test_api_modules.py`, `tests/test_asr_pipeline.py`, `tests/test_cli.py`, `tests/test_download_hf_model.py`, `tests/test_response_formats.py`, `tests/test_webui.py`  
   - Test refactors and additions to reflect new behavior and increase coverage.
9. `pyproject.toml`  
   - Fixes setup-config invocation; lint/test configuration refinements.
10. `docker-compose.yaml`, `docker-compose.dev.yaml`  

    - Update image tags (previously fixed in v0.10.1) and ensure dev/main parity.

11. `pdm.lock`  

    - Dependency lockfile updates consistent with code changes.

### Deleted

1. `to-do/` directory and all contained items  
   - Housekeeping: removed stale to-do plans migrated into code/tests and project docs.

---

## Code Changes

### `insanely_fast_whisper_rocm/core/asr_backend.py`

```python
# Key highlights from dev vs main
# 1) Safer type hints and signatures
-        language: Optional[str],
-        return_timestamps_value: Union[bool, str],
-    ) -> Dict[str, Any]:
+        language: str | None,
+        return_timestamps_value: bool | str,
+    ) -> dict[str, Any]:

# 2) Guard task/language forwarding based on generation_config capabilities
+        gen_cfg = getattr(self.asr_pipe.model, "generation_config", None)
+        has_task_mappings = False
+        if gen_cfg is not None:
+            has_task_mappings = any(
+                getattr(gen_cfg, attr, None) is not None
+                for attr in ("task_to_id", "lang_to_id")
+            )
+
+        if has_task_mappings:
+            pipeline_kwargs["generate_kwargs"]["task"] = task
+            if language and language.lower() != "none":
+                pipeline_kwargs["generate_kwargs"]["language"] = language
+            elif task == "translate":
+                pipeline_kwargs["generate_kwargs"]["language"] = "en"
+        elif task != "transcribe" or language:
+            logger.warning(
+                "Generation config for model %s lacks task/language mappings; "
+                "falling back to default transcription.",
+                self.config.model_name,
+            )

# 3) Timestamp safety check before enabling word-level timestamps
+        if _return_timestamps_value:
+            gen_cfg = getattr(self.asr_pipe.model, "generation_config", None)
+            no_ts_token_id = getattr(gen_cfg, "no_timestamps_token_id", None)
+            if no_ts_token_id is None:
+                logger.warning(
+                    (
+                        "Timestamp generation not properly configured for model %s; disabling."
+                    ),
+                    self.config.model_name,
+                )
+                _return_timestamps_value = False

# 4) Graceful fallback for known word-level timestamp tensor mismatch
             try:
                 outputs = self.asr_pipe(str(converted_path), **pipeline_kwargs)
             except RuntimeError as e:
                 if "expanded size of the tensor" in str(e) and "must match the existing size" in str(e):
                     logger.warning(
-                        "Word-level timestamp extraction failed due to tensor size mismatch. Falling back to chunk-level timestamps for %s: %s",
+                        (
+                            "Word-level timestamp extraction failed due to tensor size mismatch. "
+                            "Falling back to chunk-level timestamps for %s: %s"
+                        ),
                         audio_file_path,
                         str(e),
                     )
                     pipeline_kwargs["return_timestamps"] = "chunk"
                     outputs = self.asr_pipe(str(converted_path), **pipeline_kwargs)
                 else:
                     raise
```

- **What changed and why**
  - **Generation-config backfill/safety:** Older checkpoints may lack `task_to_id`/`lang_to_id`; the backend now checks for these and only forwards `task`/`language` when supported, avoiding `ValueError`s.
  - **Timestamp robustness:** Before enabling word-level timestamps, ensure `no_timestamps_token_id` exists; otherwise, disable timestamps safely. Also adds a targeted fallback from word-level to `"chunk"` on a known tensor size mismatch error pattern.
  - **Type hints/formatting:** Modern typing (`|` unions, `dict[str, Any]`) and clearer logging/messages.

### `insanely_fast_whisper_rocm/utils/constants.py` and `utils/env_loader.py`

```python
# Key highlights
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as pkg_version

from insanely_fast_whisper_rocm.utils.env_loader import (
    PROJECT_ROOT,
    SHOW_DEBUG_PRINTS,
    USER_CONFIG_DIR,
    USER_ENV_EXISTS,
    USER_ENV_FILE,
    debug_print,
)

# Re-export selected symbols to keep external imports stable
__all__ = [
    "PROJECT_ROOT",
]

# Resolve API version from package metadata without importing the package
try:
    API_VERSION = pkg_version("insanely-fast-whisper-rocm")
except PackageNotFoundError:
    API_VERSION = "unknown"

# Modern typing for supported formats
SUPPORTED_AUDIO_FORMATS: set[str] = {".mp3", ".flac", ".wav", ".m4a"}
SUPPORTED_VIDEO_FORMATS: set[str] = {".mp4", ".mkv", ".webm", ".mov"}
SUPPORTED_UPLOAD_FORMATS: set[str] = SUPPORTED_AUDIO_FORMATS | SUPPORTED_VIDEO_FORMATS
```

- **What changed and why**
  - Centralizes env-derived values via `env_loader` to avoid import-time side effects and ensure consistent resolution of `PROJECT_ROOT` and related constants.
  - Uses `importlib.metadata` to get `API_VERSION` without importing the package (avoids circular imports during initialization).
  - Modernizes type hints (`set[str]`) and cleans up debug logging for clarity.

### `insanely_fast_whisper_rocm/cli/*`

```python
# commands.py
from pathlib import Path

def _run_task(*, task: str, audio_file: Path, **kwargs) -> None:  # noqa: C901
    """Execute *task* (“transcribe” or “translate”) on *audio_file*."""
    output: Path | None = kwargs.pop("output", None)
    # ...

def _handle_output_and_benchmarks(
    audio_file: Path,
    result: dict,
    total_time: float,
    output: Path | None,
    export_format: str,
    export_format_explicit: bool,
    benchmark_enabled: bool,
    benchmark_extra: list[str] | None,
    temp_files: list[Path],
) -> None:
    # ...
    extra_dict: dict[str, str] | None = None
    if benchmark_extra:
        extra_dict = dict(item.split("=", 1) for item in benchmark_extra)
```

- **What changed and why**
  - Modern typing and cleaner CLI internals (`Path | None`, `dict[str, str]`).
  - Keeps command wrappers thin; shared options are still provided by `cli/common_options.py`.

### `insanely_fast_whisper_rocm/webui/*`

```python
# handlers.py (selected changes)
all_results_data.append({
    "audio_original_path": str(audio_file_path),
    "error": str(e),
})

# Use returned path from BatchZipBuilder and propagate via gr.update
all_zip_path, _ = all_zip_builder.build()
zip_btn_update = gr.update(value=all_zip_path, visible=True, interactive=True,
                           label=f"Download All ({len(successful_results)} files) as ZIP")
raw_result_state_val["all_zip"] = all_zip_path
json_output_val["zip_archive_all"] = Path(all_zip_path).name
```

- **What changed and why**
  - Fixes state propagation for generated ZIP archives using `gr.update` values and consistent variable naming (`all_zip_path`).
  - Cleans up exception handling and progress descriptions for multi-file processing.

### `pyproject.toml`

```toml
[project]
version = "0.10.1"

[project.scripts]
setup-config = {call = "scripts.setup_config:main"}

[tool.ruff]
preview = true
src = ["insanely_fast_whisper_rocm"]

[tool.ruff.lint]
select = ["F", "E", "W", "N", "I", "D", "DOC", "ANN", "TID", "UP", "FA"]
extend-select = ["ANN401"]

[tool.pytest.ini_options]
addopts = "-q --cov=insanely_fast_whisper_rocm --cov-branch --cov-report=term-missing:skip-covered --cov-report=xml --cov-report=html --cov-fail-under=85"
```

- **What changed and why**
  - Bumps project version to `0.10.1` and corrects `setup-config` script invocation.
  - Adopts Ruff as the unified linter/formatter/import sorter; enables docstring and annotation rules, with targeted test-line-length ignore.
  - Strengthens pytest defaults with coverage thresholds for CI reliability.

---

## Reason for Changes

- **Fixing bugs:** Prevent errors on older model checkpoints lacking generation-config mappings; handle timestamp extraction edge cases robustly.
- **Refactoring for clarity/maintainability:** Unify CLI/common options, improve utilities, and modernize typing/logging.
- **Test hardening:** Add targeted tests covering generation-config backfill and timestamp fallbacks; refactor existing tests for clearer intent.
- **Tooling/CI hygiene:** Ensure linters and lockfiles reflect current code paths and constraints.

---

## Impact of Changes

- **Stability:** ASR runs more reliably across a wider range of Whisper/HF model checkpoints.
- **Developer velocity:** Clearer CLI and utilities reduce friction for extending features.
- **Observability:** More precise warnings and fallback behavior make issues diagnosable without hard failures.
- **Backward compatibility:** Behavior is unchanged for supported models; unsupported mappings now degrade gracefully instead of throwing.

---

## Testing Plan

- **Unit/Integration tests:**
  - Run: `pytest -q` to execute updated and new tests, including:
    - `tests/test_asr_backend_generation_config.py`
    - `tests/test_asr_backend_timestamp.py`
    - Updated API/CLI/WebUI tests.
- **Manual validation:**
  - Verify transcription on models with and without `task_to_id`/`lang_to_id`.
  - Confirm timestamp behavior: word-level works when supported; falls back to `chunk` when the known tensor size mismatch occurs.
  - Smoke test WebUI and CLI flows.
- **Lint/format:**
  - `pdm run ruff check --fix .`
  - `pdm run ruff format .`

---

## Backward Compatibility

- No breaking API changes. Where model capabilities are missing, the system logs warnings and proceeds with safe defaults.

---

## Related Commits (selection within provided range)

- `12b1cac`: fix 🐛: Update setup-config call in pyproject.toml
- `680ff5a`: fix 🐛: Refactor constants.py to use env_loader for PROJECT_ROOT
- `87ed4e1`: feat ✨: Add Ruff code linting script and JSON conversion tool
- `f2c5390`: fix 🐛: Fix timestamp error and add fallback (#17)
- `87ed4e1`, `25a81d8`: feature/chore updates reflecting tooling and dependencies
- Multiple `refactor ♻️` commits aligning CLI/core/utils/webui with new behavior

---

## Checklist

- [x] Code changes focused over docs changes
- [x] Tests updated/added
- [x] Lint/format rules respected (Ruff)
- [x] No breaking public API changes
