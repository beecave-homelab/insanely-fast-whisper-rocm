# To-Do: Instrument Debug Logging Across Segmentation Pipeline

This plan outlines the steps to implement comprehensive debug logging for the CLI, pipeline, segmentation, and benchmark components to diagnose subtitle quality regressions.

## Tasks

- [ ] **Analysis Phase:**
  - [ ] Research and evaluate existing logging patterns in the codebase
    - Path: `[insanely_fast_whisper_rocm/]`
    - Action: Review current logging levels and conventions to ensure new statements align with project practices
    - Analysis Results:
      - Confirm modules use `logging.getLogger(__name__)`
      - Identify safe payload summaries (counts, sample entries) for large data structures
      - Determine environment configuration needed to surface debug logs via CLI
    - Accept Criteria: Document recommended logging style and data truncation approach within task notes

- [ ] **Implementation Phase:**
  - [x] Implement core functionality
    - Path: `[insanely_fast_whisper_rocm/cli/commands.py]`
    - Action: Log inputs/outputs around `_run_task()` and `_handle_output_and_benchmarks()`
    - Status: **Complete** - Added DEBUG logging for task inputs, facade calls/returns, GPU stats, export formats, benchmark collection
  - [x] Implement core functionality
    - Path: `[insanely_fast_whisper_rocm/cli/facade.py]`
    - Action: Log backend and pipeline return payloads in `process_audio()`
    - Status: **Complete** - Added DEBUG logging for backend initialization, direct backend vs pipeline routing, and return payloads
  - [x] Implement core functionality
    - Path: `[insanely_fast_whisper_rocm/core/pipeline.py]`
    - Action: Log `_execute_asr()` results and `_postprocess_output()` outputs
    - Status: **Complete** - Added DEBUG logging for ASR params, chunk processing, merge results, and postprocessing inputs
  - [x] Implement core functionality
    - Path: `[insanely_fast_whisper_rocm/core/asr_backend.py]`
    - Action: Log normalized `outputs` prior to returning from `process_audio()`
    - Status: **Complete** - Added DEBUG logging for pipeline calls, raw outputs, and normalized result before return
  - [x] Implement core functionality
    - Path: `[insanely_fast_whisper_rocm/core/formatters.py]`
    - Action: Log `_result_to_words()` decisions and formatted segment summaries
    - Status: **Complete** - Added DEBUG logging for word extraction heuristics, duration checks, and quality segment building
  - [x] Implement core functionality
    - Path: `[insanely_fast_whisper_rocm/core/segmentation.py]`
    - Action: Log sentence chunking, gap handling, and final `Segment` collection
    - Status: **Complete** - Added DEBUG logging for word expansion, sanitization, sentence chunks, merge operations, CPS enforcement, and final segment count
  - [x] Implement core functionality
    - Path: `[insanely_fast_whisper_rocm/benchmarks/collector.py]`
    - Action: Log benchmark payloads written to disk, including `format_quality`
    - Status: **Complete** - Added DEBUG logging for collect parameters, format_quality keys, and output path

- [x] **Testing Phase:**
  - [x] Unit or integration tests
    - Path: `[tests/core/]`
    - Action: Add or update tests to ensure logging does not break control flow (e.g., using `caplog` to validate key messages in targeted modules)
    - Accept Criteria: Tests confirm critical debug statements trigger without altering functional output
    - Status: **Complete** - All 618 tests pass successfully with new logging in place
  - [x] Fix --debug flag to actually enable DEBUG-level output
    - Path: `[insanely_fast_whisper_rocm/cli/commands.py]`
    - Action: Add code to set log level to DEBUG when --debug flag is True
    - Status: **Complete** - Added logging configuration at lines 243-248 to enable DEBUG level when --debug is passed

- [x] **Documentation Phase:**
  - [x] Update `project-overview.md` and/or README
    - Path: `[project-overview.md]`
    - Action: Document new debug instrumentation workflow and how to enable it during benchmarking
    - Accept Criteria: Documentation explains relevant environment variables and CLI flags for viewing logs
    - Status: **Complete** - Added comprehensive Debug Logging section with examples, instrumented modules list, and usage instructions

- [x] **Refinement Phase:**
  - [x] Fix debug log ordering issues
    - Path: `[insanely_fast_whisper_rocm/cli/commands.py]`
    - Action: Ensure DEBUG logs appear BEFORE user-facing emoji messages for proper chronological flow
    - Status: **Complete** - Moved DEBUG logs to appear before click.secho() calls; added file size to export logs
  - [x] Improve ENV_LOADER_DEBUG logging
    - Path: `[insanely_fast_whisper_rocm/utils/env_loader.py]`, `[insanely_fast_whisper_rocm/utils/constants.py]`
    - Action: Replace print() with proper logger.debug(), clarify CLI override behavior, fix contradictory LOG_LEVEL messages
    - Status: **Complete** - Converted to proper Python logging, added CLI override notice, consolidated environment variable logs
  - [x] Fix third-party logger spam
    - Path: `[insanely_fast_whisper_rocm/utils/env_loader.py]`
    - Action: Prevent urllib3 and torio from flooding DEBUG output when --debug is used
    - Status: **Complete** - Set root logger to INFO, only enable DEBUG for insanely_fast_whisper_rocm.* loggers, suppress noisy third-party loggers
  - [x] Final debug output polish
    - Path: Multiple files
    - Action: Remove redundant/duplicate logs, reduce noise, improve clarity
    - Status: **Complete** - Simplified environment loading flow, removed kwargs_keys list, removed duplicate ASR completion and benchmark logs, consolidated messages

## Related Files

- `insanely_fast_whisper_rocm/cli/commands.py`
- `insanely_fast_whisper_rocm/cli/facade.py`
- `insanely_fast_whisper_rocm/core/pipeline.py`
- `insanely_fast_whisper_rocm/core/asr_backend.py`
- `insanely_fast_whisper_rocm/core/formatters.py`
- `insanely_fast_whisper_rocm/core/segmentation.py`
- `insanely_fast_whisper_rocm/benchmarks/collector.py`

## Future Enhancements

- [ ] Add structured logging or trace IDs to correlate events across modules
