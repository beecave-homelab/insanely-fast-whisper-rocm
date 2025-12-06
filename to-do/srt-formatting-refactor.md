# SRT/VTT Formatter Refactor Plan

## Overview

This plan summarises the completed analysis for bringing the readability-focused segmentation pipeline into every SRT and VTT export path. The refactor applies across the CLI, HTTP API, and WebUI because all three already delegate subtitle rendering to the shared formatters in `insanely_fast_whisper_rocm/core/formatters.py`.

## Findings

- **Shared formatter usage**: `FORMATTERS` are invoked from CLI exports in `insanely_fast_whisper_rocm/cli/commands.py`, API responses in `insanely_fast_whisper_rocm/api/responses.py`, and WebUI downloads/ZIP creation in `insanely_fast_whisper_rocm/webui/handlers.py` and `insanely_fast_whisper_rocm/webui/zip_creator.py`. Updating the formatters once propagates improvements everywhere.
- **Segmentation capabilities**: `insanely_fast_whisper_rocm/core/segmentation.py` already implements `Word`, `Segment`, `segment_words()`, and line-wrapping helpers that enforce character limits, CPS bounds, and duration constraints. These utilities are not yet consumed by the subtitle formatters.
- **Formatter shortcomings**: `SrtFormatter.format()` and `VttFormatter.format()` still operate on raw `segments`/`chunks`, relying on narrow heuristics. They neither call `segment_words()` nor guarantee balanced two-line captions, CPS limits, or minimum/maximum durations.
- **Testing gaps**: Current tests verify segmentation helpers directly, but no suite asserts that formatter outputs (and thus CLI/API/WebUI artefacts) meet readability rules. Snapshot coverage for rendered SRT/VTT files is also absent.

## Integration Strategy

- **Normalize inputs**: Introduce a helper inside `insanely_fast_whisper_rocm/core/formatters.py` that converts `result` payloads into `list[Word]` when word timestamps exist, while preserving a chunk-based fallback for legacy outputs.
- **Call segmentation pipeline**: When `Word` data is available, invoke `segment_words()` to obtain readable `Segment` objects, then render cues using `format_srt_time()` or `format_vtt_time()`.
- **Fallback behaviour**: When only chunk-level timestamps exist, continue the existing chunk formatting path but apply the `split_lines()` helper so captions remain readable.
- **Backward compatibility**: Keep formatter method signatures and return structures unchanged so existing CLI/API/WebUI code, as well as downstream consumers, remain compatible by default.

## Testing Strategy

- **Formatter unit tests**: Extend `tests/core/test_srt_formatting.py` and `tests/core/test_srt_formatting_realworld.py` to exercise `SrtFormatter.format()` and `VttFormatter.format()` with both word-level and chunk-level inputs, asserting line-length, CPS, cue numbering, and timestamp formatting.
- **API integration tests**: Add FastAPI-level tests (or expand `tests/test_responses.py`) that request `RESPONSE_FORMAT_SRT` and `RESPONSE_FORMAT_VTT`, storing normalised snapshots to confirm readability constraints.
- **CLI regression tests**: Update `tests/cli/test_cli_exports.py` so exported `.srt` and `.vtt` files from the CLI reflect segmented captions instead of raw chunks.
- **WebUI coverage**: Add focused tests for `_prepare_temp_downloadable_file()` (and ZIP creation if feasible) to guarantee WebUI downloads include segmented subtitles.
- **Snapshot fixtures**: Create representative SRT/VTT fixtures that capture expected formatting to detect future regressions quickly.

## Implementation Milestones & Dependencies

- **Milestone 1 – Input normalisation helper**: Add a private helper in `insanely_fast_whisper_rocm/core/formatters.py` that extracts `Word` objects from results and falls back to chunk dictionaries. Depends on the `Word` dataclass from `insanely_fast_whisper_rocm/core/segmentation.py` and readability constants in `insanely_fast_whisper_rocm/utils/constants.py`.
- **Milestone 2 – SRT formatter upgrade**: Refactor `SrtFormatter.format()` to use the helper, call `segment_words()`, and render cues from `Segment` objects while keeping the chunk fallback intact.
- **Milestone 3 – VTT formatter parity**: Mirror the SRT changes inside `VttFormatter.format()` so all surfaces share identical readability behaviour.
- **Milestone 4 – Optional feature flag**: If rollout needs a safety switch, expose a `USE_READABLE_SUBTITLES` constant in `insanely_fast_whisper_rocm/utils/constants.py` and gate the new path accordingly.
- **Milestone 5 – Documentation updates**: Refresh `.env.example`, `README.md`, and `project-overview.md` once new formatter behaviour ships so configuration options and architecture are documented.
- **Dependent consumers**: CLI exports (`insanely_fast_whisper_rocm/cli/commands.py`), API responses (`insanely_fast_whisper_rocm/api/responses.py`), and WebUI flows (`insanely_fast_whisper_rocm/webui/handlers.py`, `insanely_fast_whisper_rocm/webui/zip_creator.py`) automatically benefit after Milestones 1–3.

## Regression Safeguards

- **Automated testing**: Ensure new formatter tests run in default CI, while broader CLI/API/WebUI tests carry targeted markers (e.g., `@pytest.mark.integration`).
- **Snapshots**: Maintain baseline SRT/VTT fixtures so future code changes cannot silently regress readability constraints.
  - `benchmarks/audio-overview-hoe-je-iets-vraagt-bepaalt-wat-je-krijgt-5_transcribe_20250926T213049Z_chunk_timestmaps.json`
  - `benchmarks/audio-overview-hoe-je-iets-vraagt-bepaalt-wat-je-krijgt-5_transcribe_20250926T213521Z_word_timestamps.json`
  - `benchmarks/Onboarding_App__Blauwdruk_transcribe_20250926T204933Z.json`
  - `benchmarks/Onboarding_App__Blauwdruk_transcribe_20250926T205208Z.json`
  - `transcripts-srt/audio-overview-hoe-je-iets-vraagt-bepaalt-wat-je-krijgt-5_transcribe_20250926T213049Z_chunk_timestamps.srt`
  - `transcripts-srt/audio-overview-hoe-je-iets-vraagt-bepaalt-wat-je-krijgt-5_transcribe_20250926T213521Z_word_timestamps.srt`
  - `transcripts-srt/Onboarding_App__Blauwdruk_transcribe_20250926T204933Z_chunk_timestamps.srt`
  - `transcripts-srt/Onboarding_App__Blauwdruk_transcribe_20250926T205208Z_word_timestamps.srt`
- **Performance monitoring**: Segmentation runs in O(n); add lightweight timing asserts or profiling hooks if test runs suggest a slowdown while processing large transcripts.

## Next Steps

- **Implement milestones sequentially**, landing associated tests with each change to maintain a TDD cadence.
- **Add documentation and env guidance** immediately after the formatter refactor merges so operators know how to tune readability constants.
- **Monitor rollout** using the optional feature flag if necessary; otherwise, deprecate the legacy chunk heuristics once confidence is established.
