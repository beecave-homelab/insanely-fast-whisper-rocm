# To-Do: Refactor codebase with Design Patterns

This plan outlines the steps to implement the Builder, Strategy, Observer, and Decorator patterns to improve maintainability, extensibility, and observability of the Insanely Fast Whisper API codebase while moving refactored components into dedicated sub-directories.

## Tasks

- [ ] **Analysis Phase:**
  - [ ] Research and evaluate Builder pattern for configuration
    - Path: `[insanely_fast_whisper_api/core.py]`
    - Action: Audit repeated configuration parsing across CLI, FastAPI, and Gradio; decide on fields and validation rules for a central builder.
    - Analysis Results:
      - Repetition of default parameters and validation logic in three entry-points.
      - Opportunity to return a validated `ASRConfig` dataclass.
    - Accept Criteria: Design document describing builder API, migration plan, and chosen sub-dir (`config/`).
  - [ ] Research and evaluate Strategy pattern for audio chunking
    - Path: `[insanely_fast_whisper_api/audio_utils.py]`
    - Action: Compare fixed-length, silence-based, and VAD-based chunking algorithms; draft `ChunkStrategy` interface.
    - Analysis Results:
      - Current fixed-length loop embedded in `split_audio`.
      - `pydub.silence.split_on_silence` and `pyannote.audio` viable for alternatives.
    - Accept Criteria: Chosen interface with at least two concrete strategies documented and target sub-dir (`chunking/`).
  - [ ] Research and evaluate Observer pattern for progress events
    - Path: `[insanely_fast_whisper_api/core.py]`
    - Action: Investigate lightweight event-bus options (`blinker`, custom async queue) for pipeline progress.
    - Analysis Results:
      - `blinker` offers zero-dependency synchronous signals.
      - Async queue integrates with FastAPI/Gradio.
    - Accept Criteria: Selected mechanism, event taxonomy defined, and target sub-dir (`events/`).
  - [ ] Research and evaluate Decorator pattern for cross-cutting concerns
    - Path: `[insanely_fast_whisper_api/core.py]`
    - Action: Identify functions for `@timed`, `@retry`, or `@memoize` decorators.
    - Analysis Results:
      - Manual timing scattered in several functions.
      - Transient Hugging Face network errors suggest need for retries.
    - Accept Criteria: List of decorators and target sub-dir (`utils/`).

- [ ] **Implementation Phase:**
  - [ ] Implement `ASRConfigBuilder` and `ASRConfig` dataclass
    - Path: `[insanely_fast_whisper_api/config/builder.py]`
    - Action: Create fluent builder with validation; refactor CLI (`cli.py`), FastAPI (`main.py`), and Gradio (`webui.py`) to use it.
    - Status:
  - [ ] Implement `ChunkStrategy` base class with concrete strategies
    - Path: `[insanely_fast_whisper_api/chunking/strategies.py]`
    - Action: Extract current fixed-length logic; add `SilenceChunks`; expose registry for future VAD strategy.
    - Status:
  - [ ] Integrate Strategy into `ASRPipeline`
    - Path: `[insanely_fast_whisper_api/core/pipeline.py]`
    - Action: Split >200-line `core.py`; move pipeline-specific code into `core/pipeline.py`; accept `chunker: ChunkStrategy` parameter and delegate splitting.
    - Status:
  - [ ] Implement lightweight `EventBus`
    - Path: `[insanely_fast_whisper_api/events/bus.py]`
    - Action: Provide `subscribe(event, cb)` and `publish(event, data)` helpers with minimal dependency footprint.
    - Status:
  - [ ] Emit progress events from pipeline
    - Path: `[insanely_fast_whisper_api/core/pipeline.py]`
    - Action: Publish `chunk_started`, `chunk_finished`, and `pipeline_finished` events.
    - Status:
  - [ ] Consume events in Gradio UI
    - Path: `[insanely_fast_whisper_api/webui.py]`
    - Action: Subscribe to events; update `progress_bar` via streaming/yield.
    - Status:
  - [ ] Create Decorators module
    - Path: `[insanely_fast_whisper_api/utils/decorators.py]`
    - Action: Implement `@timed` and `@retry`; apply to `run_asr_pipeline`, `split_audio`.
    - Status:

- [ ] **Testing Phase:**
  - [ ] Unit tests for Builder
    - Path: `[tests/test_config_builder.py]`
    - Action: Validate builder outputs with valid and invalid input.
    - Accept Criteria: All assertions pass; branch coverage ≥ 90 %.
  - [ ] Unit tests for Strategy implementations
    - Path: `[tests/test_chunking_strategies.py]`
    - Action: Feed sample audio files and assert expected chunk counts and boundaries.
    - Accept Criteria: Strategies return correct chunk lists within tolerance.
  - [ ] Integration tests for EventBus
    - Path: `[tests/test_event_bus.py]`
    - Action: Publish dummy events and verify subscriber callbacks execute in order.
    - Accept Criteria: All subscribers triggered; no memory leaks.
  - [ ] Regression benchmark
    - Path: `[tests/test_performance.py]`
    - Action: Compare transcription latency before and after refactor.
    - Accept Criteria: p95 latency degradation ≤ 5 %.

- [ ] **Documentation Phase:**
  - [ ] Update architecture diagrams and README
    - Path: `[project-overview.md]`
    - Action: Add sections on new sub-modules, sequence diagram for Observer flow, builder usage examples.
    - Accept Criteria: Documentation is up-to-date and explains the new components clearly.

## Related Files

- `[insanely_fast_whisper_api/core.py]`
- `[insanely_fast_whisper_api/core/pipeline.py]`
- `[insanely_fast_whisper_api/config/builder.py]`
- `[insanely_fast_whisper_api/chunking/strategies.py]`
- `[insanely_fast_whisper_api/events/bus.py]`
- `[insanely_fast_whisper_api/utils/decorators.py]`
- `[insanely_fast_whisper_api/webui.py]`
- `[insanely_fast_whisper_api/audio_utils.py]`
- `[tests/*]`
- `[project-overview.md]`

## Future Enhancements

- [ ] Replace `constants.py` with a Pydantic-based `Settings` singleton for runtime overrides.
