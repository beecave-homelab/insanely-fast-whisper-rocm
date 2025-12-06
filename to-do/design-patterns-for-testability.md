# Design patterns for testability

Assessment of what’s already helping testability in the repo and where things fall short from a TDD perspective, with concrete references.

## What’s already helping testing

- Formatter strategy + registry
  - Code: [insanely_fast_whisper_rocm/core/formatters.py](cci:7://file:///home/elvee/Local-AI/insanely-fast-whisper-rocm/insanely_fast_whisper_rocm/core/formatters.py:0:0-0:0) (`FORMATTERS`, [BaseFormatter](cci:2://file:///home/elvee/Local-AI/insanely-fast-whisper-rocm/insanely_fast_whisper_rocm/core/formatters.py:17:0-35:74), [TxtFormatter](cci:2://file:///home/elvee/Local-AI/insanely-fast-whisper-rocm/insanely_fast_whisper_rocm/core/formatters.py:38:0-71:20), [SrtFormatter](cci:2://file:///home/elvee/Local-AI/insanely-fast-whisper-rocm/insanely_fast_whisper_rocm/core/formatters.py:74:0-210:20), [VttFormatter](cci:2://file:///home/elvee/Local-AI/insanely-fast-whisper-rocm/insanely_fast_whisper_rocm/core/formatters.py:213:0-304:20), [JsonFormatter](cci:2://file:///home/elvee/Local-AI/insanely-fast-whisper-rocm/insanely_fast_whisper_rocm/core/formatters.py:307:0-336:21))
  - Why it helps: Clear Strategy/Registry pattern; pure formatting functions are easy to unit test and reuse. Consumers (API, WebUI, ZIP builder, CLI tools) all go through the same surface.

- API response formatter as a facade
  - Code: [insanely_fast_whisper_rocm/api/responses.py](cci:7://file:///home/elvee/Local-AI/insanely-fast-whisper-rocm/insanely_fast_whisper_rocm/api/responses.py:0:0-0:0) ([ResponseFormatter](cci:2://file:///home/elvee/Local-AI/insanely-fast-whisper-rocm/insanely_fast_whisper_rocm/api/responses.py:22:0-234:9))
  - Why it helps: Encapsulates response payload shaping (JSON/Text/SRT/VTT). Helpers like [_seconds_to_timestamp](cci:1://file:///home/elvee/Local-AI/insanely-fast-whisper-rocm/insanely_fast_whisper_rocm/api/responses.py:26:4-43:67), [_segments_to_srt](cci:1://file:///home/elvee/Local-AI/insanely-fast-whisper-rocm/insanely_fast_whisper_rocm/api/responses.py:45:4-66:43), [_segments_to_vtt](cci:1://file:///home/elvee/Local-AI/insanely-fast-whisper-rocm/insanely_fast_whisper_rocm/api/responses.py:68:4-88:43) are pure and deterministic. Tests in [tests/test_responses.py](cci:7://file:///home/elvee/Local-AI/insanely-fast-whisper-rocm/tests/test_responses.py:0:0-0:0) validate this.

- FastAPI dependency injection for boundary overrides
  - Code: [insanely_fast_whisper_rocm/api/routes.py](cci:7://file:///home/elvee/Local-AI/insanely-fast-whisper-rocm/insanely_fast_whisper_rocm/api/routes.py:0:0-0:0) (uses `Depends(get_asr_pipeline)`, `Depends(get_file_handler)`)
  - Why it helps: Easy to override dependencies in tests with FastAPI’s `dependency_overrides`. This is aligned with the Testing Standards memory.

- WebUI ZIP Builder pattern
  - Code: [insanely_fast_whisper_rocm/webui/zip_creator.py](cci:7://file:///home/elvee/Local-AI/insanely-fast-whisper-rocm/insanely_fast_whisper_rocm/webui/zip_creator.py:0:0-0:0) ([BatchZipBuilder](cci:2://file:///home/elvee/Local-AI/insanely-fast-whisper-rocm/insanely_fast_whisper_rocm/webui/zip_creator.py:53:0-804:37), [ZipConfiguration](cci:2://file:///home/elvee/Local-AI/insanely-fast-whisper-rocm/insanely_fast_whisper_rocm/webui/zip_creator.py:26:0-38:31))
  - Why it helps: Builder pattern with clear state and methods ([create](cci:1://file:///home/elvee/Local-AI/insanely-fast-whisper-rocm/insanely_fast_whisper_rocm/webui/zip_creator.py:85:4-130:17), [add_batch_files](cci:1://file:///home/elvee/Local-AI/insanely-fast-whisper-rocm/insanely_fast_whisper_rocm/webui/zip_creator.py:132:4-170:17), [add_merged_files](cci:1://file:///home/elvee/Local-AI/insanely-fast-whisper-rocm/insanely_fast_whisper_rocm/webui/zip_creator.py:172:4-217:17), [add_summary](cci:1://file:///home/elvee/Local-AI/insanely-fast-whisper-rocm/insanely_fast_whisper_rocm/webui/zip_creator.py:250:4-286:23), [build](cci:1://file:///home/elvee/Local-AI/insanely-fast-whisper-rocm/insanely_fast_whisper_rocm/webui/zip_creator.py:288:4-361:33)). Easy to unit test in isolation; minimal side effects (writes via `writestr`).

- Clear CLI boundary tests
  - Code: [tests/test_cli_exports.py](cci:7://file:///home/elvee/Local-AI/insanely-fast-whisper-rocm/tests/test_cli_exports.py:0:0-0:0), [tests/test_cli.py](cci:7://file:///home/elvee/Local-AI/insanely-fast-whisper-rocm/tests/test_cli.py:0:0-0:0)
  - Why it helps: Uses `click.testing.CliRunner` and patches the processing boundary (`cli_facade.process_audio`) to stay hermetic. Asserts artifacts and outputs without touching ASR.

- Centralized configuration
  - Code: [insanely_fast_whisper_rocm/utils/constants.py](cci:7://file:///home/elvee/Local-AI/insanely-fast-whisper-rocm/insanely_fast_whisper_rocm/utils/constants.py:0:0-0:0)
  - Why it helps: Single source of truth for config and env loading yields consistent behavior and enables test overrides via environment.

Where TDD falls short (and why)

- SRT/VTT segmentation and readability logic not modularized or injectable
  - Code: [SrtFormatter.format()](cci:1://file:///home/elvee/Local-AI/insanely-fast-whisper-rocm/insanely_fast_whisper_rocm/core/formatters.py:77:4-200:21) in [core/formatters.py](cci:7://file:///home/elvee/Local-AI/insanely-fast-whisper-rocm/insanely_fast_whisper_rocm/core/formatters.py:0:0-0:0)
  - Issue: Formatting method mixes selection of data source (`segments` vs `chunks`), timestamp extraction, “word-like” grouping, and final formatting. There’s no separate, testable “segmenter” component or policy injection (e.g., max chars per line, CPS, min/max duration).
  - TDD impact: Hard to write fine-grained unit tests for segmentation rules, clause-aware splitting, CPS and duration constraints. Tests can only validate the whole formatter, not the underlying decisions.

- Missing readability policy object for subtitles
  - Code: [utils/constants.py](cci:7://file:///home/elvee/Local-AI/insanely-fast-whisper-rocm/insanely_fast_whisper_rocm/utils/constants.py:0:0-0:0) doesn’t define subtitle constraints (MAX_LINE_CHARS, MAX_LINES_PER_BLOCK, MIN/MAX_CPS, MIN/MAX_SEGMENT_DURATION_SEC, DISPLAY_BUFFER_SEC, SOFT_BOUNDARY_WORDS).
  - Issue: No structured, injectable policy; everything is inside heuristics.
  - TDD impact: Tests can’t parametrically verify line length, CPS, and duration behavior by swapping policies. This blocks fast, deterministic unit tests described in your TDD standards.

- Global formatting registry not easily overridden in API tests
  - Code: [api/responses.py](cci:7://file:///home/elvee/Local-AI/insanely-fast-whisper-rocm/insanely_fast_whisper_rocm/api/responses.py:0:0-0:0) does `from insanely_fast_whisper_rocm.core.formatters import FORMATTERS` and uses that mapping directly.
  - Issue: In [tests/test_responses.py](cci:7://file:///home/elvee/Local-AI/insanely-fast-whisper-rocm/tests/test_responses.py:0:0-0:0), there’s an attempt to monkeypatch `ResponseFormatter.FORMATTERS`, but [ResponseFormatter](cci:2://file:///home/elvee/Local-AI/insanely-fast-whisper-rocm/insanely_fast_whisper_rocm/api/responses.py:22:0-234:9) doesn’t actually own that mapping—it imports the global one. That mock works only because the test assigns attributes to [ResponseFormatter](cci:2://file:///home/elvee/Local-AI/insanely-fast-whisper-rocm/insanely_fast_whisper_rocm/api/responses.py:22:0-234:9) and then the method references the imported `FORMATTERS` indirectly through class attributes by coincidence in some places. This is brittle.
  - TDD impact: You should be able to cleanly inject/override the formatter mapping/factory per test without relying on import-time globals.

- Time is not abstracted (non-deterministic filenames/ZIP names)
  - Code: [webui/handlers.py](cci:7://file:///home/elvee/Local-AI/insanely-fast-whisper-rocm/insanely_fast_whisper_rocm/webui/handlers.py:0:0-0:0) (e.g., `datetime.now().strftime(...)`), [zip_creator.py](cci:7://file:///home/elvee/Local-AI/insanely-fast-whisper-rocm/insanely_fast_whisper_rocm/webui/zip_creator.py:0:0-0:0) (timestamps in `_batch_id`, [build](cci:1://file:///home/elvee/Local-AI/insanely-fast-whisper-rocm/insanely_fast_whisper_rocm/webui/zip_creator.py:288:4-361:33)/`summary`).
  - Issue: Real clock usage makes outputs time-dependent.
  - TDD impact: Snapshot-like tests and filename assertions are flaky/hard. A `Clock` interface with a test fake would make outputs deterministic.

- Filesystem/IO not abstracted in WebUI paths
  - Code: [webui/handlers.py](cci:7://file:///home/elvee/Local-AI/insanely-fast-whisper-rocm/insanely_fast_whisper_rocm/webui/handlers.py:0:0-0:0) writes temp files; [zip_creator.py](cci:7://file:///home/elvee/Local-AI/insanely-fast-whisper-rocm/insanely_fast_whisper_rocm/webui/zip_creator.py:0:0-0:0) writes ZIPs with `zipfile.ZipFile`.
  - Issue: Direct IO complicates unit tests; tests need temp dirs or to mock file system calls wholesale.
  - TDD impact: Introducing a small `FileStore` port (write_text, writestr_to_zip, mktempdir) enables in-memory or temp-backed fakes for fast, hermetic tests.

- Domain data structures are plain dicts
  - Code: Results, words, segments are mostly dict-based across formatters/response formatter/webui.
  - Issue: Lack of typed value objects (`Word`, `Segment`) reduces clarity and makes unit tests verbose and error-prone.
  - TDD impact: Dataclasses with clear types make fixtures and assertions simpler and safer.

- God functions in handlers
  - Code: [webui/handlers.py](cci:7://file:///home/elvee/Local-AI/insanely-fast-whisper-rocm/insanely_fast_whisper_rocm/webui/handlers.py:0:0-0:0) [process_transcription_request](cci:1://file:///home/elvee/Local-AI/insanely-fast-whisper-rocm/insanely_fast_whisper_rocm/webui/handlers.py:365:0-954:5) is long and branchy.
  - Issue: Multiple responsibilities (pipeline orchestration, UI progress, temp file creation, ZIP transitions).
  - TDD impact: Hard to unit test specific branches; this should be decomposed into smaller commands/services (e.g., a “PrepareDownloadsCommand” depending on `FileStore`, `FormatterFactory`, `Clock`).

- ASR backend abstraction vs injection
  - Code: [webui/handlers.py](cci:7://file:///home/elvee/Local-AI/insanely-fast-whisper-rocm/insanely_fast_whisper_rocm/webui/handlers.py:0:0-0:0) constructs `HuggingFaceBackend` directly and `WhisperPipeline` directly.
  - Issue: There is a port in API routes via DI, but WebUI builds concrete backends in place.
  - TDD impact: WebUI tests have to patch deep symbols; better to inject an `ASRBackend`/`WhisperPipeline` interface to isolate UI from inference in tests.

Quick, TDD-friendly improvements to consider

- Introduce `SubtitleReadabilityPolicy` and a `Segmenter` interface (e.g., `ClauseAwareSegmenter`, `ChunkFallbackSegmenter`).
  - Wire them into [SrtFormatter](cci:2://file:///home/elvee/Local-AI/insanely-fast-whisper-rocm/insanely_fast_whisper_rocm/core/formatters.py:74:0-210:20)/[VttFormatter](cci:2://file:///home/elvee/Local-AI/insanely-fast-whisper-rocm/insanely_fast_whisper_rocm/core/formatters.py:213:0-304:20) via a small factory or constructor injection.
  - Add pure functions for `split_lines`, `_respect_limits`, `_merge_short_segments`, `_fix_overlaps`.

- Provide a `FormatterFactory` dependency for [ResponseFormatter](cci:2://file:///home/elvee/Local-AI/insanely-fast-whisper-rocm/insanely_fast_whisper_rocm/api/responses.py:22:0-234:9) (instead of using the global `FORMATTERS` mapping directly).
  - Allow clean per-test overrides with a small DI hook.

- Add `Clock` and `FileStore` ports.
  - Use in [webui/handlers.py](cci:7://file:///home/elvee/Local-AI/insanely-fast-whisper-rocm/insanely_fast_whisper_rocm/webui/handlers.py:0:0-0:0) and [webui/zip_creator.py](cci:7://file:///home/elvee/Local-AI/insanely-fast-whisper-rocm/insanely_fast_whisper_rocm/webui/zip_creator.py:0:0-0:0); tests inject fakes for deterministic names and IO-less behavior.

- Add simple dataclasses for `Word` and `Segment` in a `core/models.py` or `core/segmentation.py`.
  - Provide conversion helpers from the existing dicts.

- Decompose [process_transcription_request()](cci:1://file:///home/elvee/Local-AI/insanely-fast-whisper-rocm/insanely_fast_whisper_rocm/webui/handlers.py:365:0-954:5) into small functions or commands.
  - Each with narrow responsibilities and unit tests.

---
