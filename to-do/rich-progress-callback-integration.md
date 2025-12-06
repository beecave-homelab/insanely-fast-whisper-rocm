# To-Do: Rich, Granular Progress via Callback Integration

This plan outlines the steps to implement a Rich-powered, granular progress reporting system for the CLI by threading a typed callback interface from the CLI layer down into the ASR backend.

## Tasks

- [ ] **Analysis Phase:**
  - [ ] Review existing CLI flow and backend call sites
    - Path: `insanely_fast_whisper_rocm/cli/commands.py`
    - Action: Identify phases and long-running sections that can emit progress (pre-processing, inference, post-processing, export). Confirm how errors and exits are handled to properly close progress rendering.
    - Analysis Results:
      - CLI orchestration is centralized in `_run_task()`.
      - Backend call is `cli_facade.process_audio()` which delegates to `HuggingFaceBackend.process_audio()`.
      - Export happens in `_handle_output_and_benchmarks()` with a finite set of formats.
    - Accept Criteria: Clear mapping of phases and insertion points for progress events without altering command semantics.
  - [ ] Inspect backend chunking/inference loops for hook points
    - Path: `insanely_fast_whisper_rocm/core/asr_backend.py`
    - Action: Locate model load, audio load, chunking, batching, and postprocess boundaries to invoke callback events.
    - Analysis Results:
      - Determine feasibility to surface total chunks/batches before processing begins.
    - Accept Criteria: List of concrete callback invocations with arguments and expected ordering.

- [ ] **Implementation Phase:**
  - [ ] Define ProgressCallback protocol and events
    - Path: `insanely_fast_whisper_rocm/core/progress.py`
    - Action: Create a `ProgressCallback` Protocol (PEP 544) with events:
      - `on_model_load_started()` / `on_model_load_finished()`
      - `on_audio_loading_started(path: str)` / `on_audio_loading_finished(duration_sec: float)`
      - `on_chunking_started(total_chunks: int)` / `on_chunk_done(index: int)`
      - `on_inference_started(total_batches: int)` / `on_inference_batch_done(index: int)`
      - `on_postprocess_started(name: str)` / `on_postprocess_finished(name: str)`
      - `on_export_started(total_items: int)` / `on_export_item_done(index: int, label: str)`
      - `on_completed()` / `on_error(message: str)`
    - Status: Pending
  - [ ] Implement RichProgressReporter in CLI
    - Path: `insanely_fast_whisper_rocm/cli/progress_rich.py`
    - Action: Implement a concrete reporter using `rich.progress.Progress` with columns `SpinnerColumn`, `TextColumn`, `BarColumn` (when totals known), `TaskProgressColumn`, `TimeElapsedColumn`.
    - Details:
      - Create tasks lazily on first event for each phase.
      - Auto-disable rendering on `--no-progress` or when not a TTY.
      - Ensure context-manager style usage for proper cleanup on exceptions.
    - Status: Pending
  - [ ] Thread callback through CLI facade
    - Path: `insanely_fast_whisper_rocm/cli/facade.py`
    - Action: Add optional `progress_cb: ProgressCallback | None` to `CLIFacade.process_audio()` and pass it to backend.
    - Status: Pending
  - [ ] Instrument backend with callback hooks
    - Path: `insanely_fast_whisper_rocm/core/asr_backend.py`
    - Action: Accept `progress_cb: ProgressCallback | None` in `HuggingFaceBackend.process_audio()` and invoke hooks during model load, audio read, chunking, inference batches, and optional postprocess inside backend (if any).
    - Status: Pending
  - [ ] Wire export progress updates in CLI
    - Path: `insanely_fast_whisper_rocm/cli/commands.py`
    - Action: In `_handle_output_and_benchmarks()`, compute `total_items` from formats and call `on_export_started/…done`. If no callback provided, no-op.
    - Status: Pending
  - [ ] Add CLI flags
    - Path: `insanely_fast_whisper_rocm/cli/common_options.py`
    - Action: Add `--no-progress` flag. Optionally add `--quiet` that implies `--no-progress`. Pass flag down to construct or skip Rich reporter.
    - Status: Pending

- [ ] **Testing Phase:**
  - [ ] Unit tests for callback invocation order and counts
    - Path: `tests/test_progress_callback.py`
    - Action: Create a fake callback collector to assert events are emitted in expected order and with correct totals for chunking/batching/export.
    - Accept Criteria: Tests cover success path and error path (ensuring `on_error` is called and Rich progress context is closed).
  - [ ] CLI smoke test for Rich reporter
    - Path: `tests/test_cli_progress.py`
    - Action: Run CLI commands with a tiny audio fixture and `--no-progress` toggled to ensure no regressions and that progress rendering doesn’t crash.
    - Accept Criteria: Exit code zero, expected files created, no Rich traceback.

- [ ] **Documentation Phase:**
  - [ ] Update `project-overview.md` and README
    - Path: `project-overview.md`, `README.md`
    - Action: Document `--no-progress`, show a short animated example or screenshot, describe phases and what the bar indicates. Mention auto-disable behavior on non-TTY environments.
    - Accept Criteria: Docs clearly explain usage and behavior; references to relevant modules/functions.

## Related Files

- `insanely_fast_whisper_rocm/cli/commands.py`
- `insanely_fast_whisper_rocm/cli/common_options.py`
- `insanely_fast_whisper_rocm/cli/facade.py`
- `insanely_fast_whisper_rocm/core/asr_backend.py`
- `insanely_fast_whisper_rocm/core/progress.py` (new)
- `insanely_fast_whisper_rocm/cli/progress_rich.py` (new)
- `tests/test_progress_callback.py` (new)
- `tests/test_cli_progress.py` (new)

## Future Enhancements

- [ ] Add per-segment or per-word progress events for integrations that compute timestamps incrementally.
- [ ] Provide a JSONL machine-readable progress reporter for programmatic consumption (non-TTY environments).
- [ ] Add environment variable overrides, e.g., `IFW_NO_PROGRESS=1`.
- [ ] Support nested tasks (e.g., multiple files batch mode) when batch-CLI is introduced.
