---
trigger: glob
description: Keep tests fast, deterministic, and hermetic by default; mirror package layout; mock external tools at boundaries; mark heavy/integration/gpu tests opt-in. Designed for consistent, reviewable tests across the repo.
globs: tests/**/*.py
---

# Testing Guidelines for `insanely_fast_whisper_rocm`

Description: Keep tests fast, deterministic, and hermetic by default; mirror package layout; mock external tools at boundaries; mark heavy/integration/gpu tests opt-in. Designed for consistent, reviewable tests across the repo.

---

## General principles

- Prefer fast, deterministic, hermetic tests by default.  
- Mirror the package structure under `tests/` for locality and easier refactors.  
- Separate unit tests (pure logic) from integration/e2e tests (boundary interactions).  
- Mock external boundaries (ffmpeg, pydub, torch, HTTP, filesystem adapters); avoid mocking pure internal logic.  
- Use Arrange–Act–Assert (AAA), one behavior per test, and descriptive test names.  
- Favor parametrization and property tests over duplicated cases.  
- Normalize timestamps, UUIDs, absolute paths, and collection ordering before snapshot assertions.

---

## Layout & scoping

- Unit tests: `tests/<subpkg>/test_*.py` (for example, `tests/api`, `tests/audio`, `tests/cli`).  
- Boundary / integration tests: `tests/integration/**` (API, CLI subprocesses, real ffmpeg, etc.)  
- Optional E2E smoke tests: `tests/integration/**` or `tests/e2e/**`  
- Keep integration/E2E tests minimal and opt-in via markers.

---

## pytest configuration (recommended)

Use markers to opt-in heavy suites:

- `@pytest.mark.integration` — filesystem / subprocess / network / external tools  
- `@pytest.mark.slow` — long audio processing, full model init  
- `@pytest.mark.e2e` — end-to-end black-box flows  
- `@pytest.mark.gpu` — ROCm/GPU required

Default `pytest` run excludes `integration`, `slow`, `e2e`, and `gpu`.  
Measure coverage for `insanely_fast_whisper_rocm` and add a fail-under threshold after the suite stabilizes.

Example `pytest.ini` entry:

```ini
[pytest]
testpaths = tests
python_files = test_*.py
addopts = -q --strict-markers
markers =
    integration: filesystem/subprocess/network/external tools
    slow: long-running audio/model tasks
    e2e: end-to-end smoke tests
    gpu: ROCm/GPU required
```

---

## Fixtures & test data

- Provide a tiny WAV fixture (~0.25 s, 16 kHz, mono, 16-bit PCM) for fast audio tests.  
- Provide a deterministic fake ASR backend fixture with a minimal `.config` (e.g., `chunk_length`) and stable `process_audio()` output.  
- Use `tmp_path` for file artifacts and ensure cleanup of temporary files.  
- Generate fixtures on-the-fly; avoid large binary fixtures checked into the repo.

---

## FastAPI / API testing rules

- Construct the app via the factory and override dependencies in tests:  
  - Replace `get_asr_pipeline()` with the fake backend.  
  - Replace `get_file_handler()` with a lightweight fake that writes to `tmp_path`.  
  - Mock startup actions (for example, `download_model_if_needed()`).
- Test endpoints `/v1/audio/transcriptions` and `/v1/audio/translations` for:  
  - Happy path (multipart upload of tiny WAV).  
  - Invalid `response_format` and missing form fields.  
- Verify `ResponseFormatter` outputs for `json`, `text`, `srt`, and `vtt`. Normalize variable data (timestamps, paths, IDs) before snapshot assertions.  
- Prefer `httpx.AsyncClient` with `asgi_lifespan.LifespanManager` for API tests and mark them `integration` when they hit dependency overrides that exercise more than pure logic.

---

## Audio module rules

- Unit tests should mock external tools: ffmpeg and pydub.  
- `ensure_wav()`:
  - If input ends with `.wav`, return unchanged.  
  - For non-wav inputs, assert ffmpeg is invoked (mocked) and output path is returned.  
- `extract_audio_from_video()`:
  - Assert FileNotFoundError for missing video file.  
  - Mock ffmpeg for success and failure; ensure cleanup on exception.  
- `split_audio()`:
  - Short audio returns original path.  
  - Longer audio yields deterministic chunk files; assert ordering and overlap behavior.  
  - Ensure no leaked temp files on error.  
- `merge_chunk_results()`:
  - Concatenate texts, flatten chunk lists, and sum runtime seconds; add property tests for ordering/associativity when appropriate.  
- Real ffmpeg runs only in `@pytest.mark.integration` tests with tiny inputs.

---

## Core & pipeline rules

- `WhisperPipeline.process()` tests should cover:  
  - Observable event sequence: `pipeline_start` → `chunk_start` / `chunk_complete` → `pipeline_complete`.  
  - Presence of `pipeline_runtime_seconds` and `output_file_path` when saving is enabled.  
  - Error cases: missing audio or backend exceptions should surface as `TranscriptionError` and notify listeners of `pipeline_error`.  
  - Multi-chunk flows: assert `merge_chunk_results()` usage and temporary chunk cleanup.
- Formatters (SRT/VTT): property tests for timestamp correctness (monotonic, non-negative) and snapshot tests after normalizing variable fields.  
- Storage: assert correct directory layout, collision handling, and atomic write semantics under `tmp_path`.  
- Utils: device-string parsing and small helpers; prefer property tests for edge cases.

---

## CLI rules

- Unit tests: use `click.testing.CliRunner`.  
  - Assert `transcribe` and `translate` call the facade with normalized arguments.  
  - Verify legacy export flags map to `export_format`.  
  - Confirm `--help` exits with code 0; invalid flag combos exit non-zero with helpful messages.  
- Integration CLI tests (marked `integration`): run the CLI as a subprocess on a tiny WAV inside a temp dir and assert output artifacts (e.g., `.json`, `.srt`) are created.

---

## WebUI rules

- `webui.utils`:
  - Test `save_temp_file()` with and without `desired_filename`. Freeze time when asserting timestamped filenames.  
  - Mock `torch.cuda.is_available()` and `torch.mps.is_available()` to avoid hardware coupling.
- Handlers and merge logic:
  - Mock the pipeline and filesystem interactions. Snapshot merged JSON/text outputs after normalizing variable values.

---

## Property & snapshot tests

- Use `hypothesis` for parsers, encoders/decoders, and formatters.  
- Use snapshot testing for stable textual outputs (API responses, CLI exports). Normalize timestamps, absolute paths, and UUIDs prior to asserting.

---

## Determinism & isolation

- Freeze time for any tests asserting timestamps (fixture or `freezegun`).  
- Inject `now_func`, `uuid_func`, and RNG where feasible to avoid flaky randomness.  
- Sort unordered collections before assertions; never rely on dict iteration order.

---

## When to mark integration or slow

- Mark `integration` when invoking subprocess/CLI, real FastAPI HTTP roundtrips, real ffmpeg/pydub decoding, or network calls.  
- Mark `slow` for non-trivial audio processing, full stable-ts runs, or model initialization.  
- Mark `gpu` when tests require ROCm/GPU or CUDA/MPS availability.

---

## Quality gates & PR guidance

- Add coverage reporting for `insanely_fast_whisper_rocm` and enforce a threshold once stable.  
- Keep E2E tests few and focused; rely on unit + integration tests for breadth and depth.  
- Assert stable fragments or structured attributes instead of entire error message strings.

---

## Do / Don't (short)

### DO

- Generate tiny fixtures programmatically and clean up temp files.  
- Mock external tool entry points in unit tests.  
- Use FastAPI dependency overrides instead of patching internals.

### DON'T

- Download models or hit the network in unit tests.  
- Depend on real GPUs in default CI or local runs.  
- Leave timestamps, UUIDs, or path separators non-deterministic in snapshot assertions.

---

## How to run

- Default unit suite: `pdm run pytest`  
- Include boundaries: `pdm run pytest -m "integration or e2e"`  
- Heavy/GPU tests: `pdm run pytest -m "slow or gpu"`

---

## Scope

- Applies to: `insanely_fast_whisper_rocm/**` and `tests/**` in this workspace.  
- Purpose: produce tests that are fast by default, hermetic, and aligned with repository architecture and external dependencies.
