# Report Summary (to-do/srt-formatting.md)

Scope: Industry best practices (CPL ≤ 42, up to 2 lines, natural line breaks, CPS caps ~15–17, min/max durations, merging short segments) and a reference segmentation/formatting pipeline (clause-aware splits, CPS/duration enforcement, line balancing, modular formatters).
Key ideas to adopt: Centralized readability constraints, typed word/segment models, clause-aware segmentation with backtracking + greedy fallbacks, modular SRT/VTT/TXT formatters, and clean integration in the ASR pipeline.
Codebase Analysis

## Formatters strategy/registry exists and is shared across CLI/API/WebUI

- SRT: `insanely_fast_whisper_rocm/core/formatters.py:74` formats via heuristics and uses `format_time.format_seconds` for timestamps.
- VTT: `insanely_fast_whisper_rocm/core/formatters.py:216`.
- TXT/JSON: straightforward.
- API facade: `insanely_fast_whisper_rocm/api/responses.py:22` contains pure helpers (`_seconds_to_timestamp`, `_segments_to_srt`, `_segments_to_vtt`), but SRT/VTT responses ultimately call the core formatters (so formatter correctness matters everywhere).
- WebUI merge/template: `insanely_fast_whisper_rocm/webui/merge_handler.py` uses FORMATTERS for SRT and renumbers entries; VTT is rendered inline.
- CLI export: `insanely_fast_whisper_rocm/cli/commands.py` writes outputs using FORMATTERS.
- Tests: verify presence/shape of outputs but do not assert SRT punctuation in core formatting; API tests mock SRT formatting.

## Findings (Gaps/Bugs)

### SRT punctuation bug

- SrtFormatter uses insanely_fast_whisper_rocm/utils/format_time.py:6 which returns HH:MM:SS.mmm (dot). SRT requires a comma. API’s _seconds_to_timestamp does it correctly, but SrtFormatter (used by API/CLI/WebUI) does not. This yields non‑compliant SRT across consumers.
- Heuristics embedded and not configurable: thresholds (gap 0.6s, max len 42, duration 3.5s) are hardcoded in SrtFormatter. No central “readability policy”.
- No balanced two‑line splitting: SrtFormatter sanitizes newlines and keeps one line per cue; there’s no split_lines step respecting CPL and natural breaks.
- No validator stage: No post‑check to enforce CPS/duration constraints on the grouped output; merging logic is minimal and non‑configurable.
Testable Design Proposal

### SubtitlePolicy (config object)

- Encapsulate readability constraints: max_line_chars, max_cps, min_cps, min_duration_s, max_duration_s, max_lines=2, gap_threshold_s, punctuation/soft break sets, plus language-specific overrides later.
- Pure components (no I/O, easy to unit test)
- Segment models: minimal Segment(start, end, text) value object; optional Word(start, end, text) if we later ingest word-level timing.
- SegmentGrouper: groups fine-grained chunks into readable segments using SubtitlePolicy (gap threshold, max duration, CPL/CPS checks; sentence-end flush; greedy fallback).
- LineSplitter: balanced 1–2 line split for each segment; break on punctuation/conjunctions; honor CPL and min-line-length constraints.
- SubtitleValidator: CPS/duration/line-length checks; can also normalize/merge or flag violations.
TimeFormatter: unify SRT/VTT formatting in one place (comma vs dot), used by both core formatters and API.
Strategy stays: Keep FORMATTERS registry; wire SrtFormatter to the pure pipeline above. WebUI merger and API continue to depend on the single SrtFormatter implementation, improving consistency.

## Plan (Incremental and Low-Risk)

### PR 1 — SRT punctuation correctness

- Add a unified time formatter in insanely_fast_whisper_rocm/utils/format_time.py with explicit mode or separator (e.g., format_seconds(seconds, sep=',')).
- Update SrtFormatter to use comma; keep VttFormatter on dot; no behavior change to VTT.
- Add unit tests for time formatting and an SRT cue sample (ensures commas).

### PR 2 — Extract pure helpers

- Add SubtitlePolicy (sane defaults matching report) and LineSplitter (balanced two-line split).
- Unit tests: CPL splitting, punctuation break preference, edge cases.

### PR 3 — Segment grouping and validation

- Implement SegmentGrouper to replace ad hoc grouping in SrtFormatter. Support gap threshold, CPL, CPS, and duration enforcement from SubtitlePolicy. Add SubtitleValidator.
- Update SrtFormatter.format to: pick segments/chunks → group → split lines → render with TimeFormatter.
- Unit tests: grouping on “word-like” inputs vs chunked inputs; CPS/duration enforcement; validator behavior.

### PR 4 — Config and docs

- Consider exposing policy fields via env vars/CLI (optional for now). Update README/CLI help; add tests for default config parsing. Ensure API, CLI, and WebUI all produce aligned SRT.
Optional later — API facade unification
Migrate _seconds_to_timestamp in api/responses.py to the shared time util; adjust tests accordingly (no semantic change).
Acceptance Criteria (per step)

- PR 1: Any SRT produced by API/CLI/WebUI uses commas in times; tests verify.
- PR 2–3: Produces two-line cues as needed, honors CPL, CPS, duration limits; unit tests cover split/group edge cases.
- PR 4: Documented defaults and optional overrides; no breaking changes to defaults without explicit request.
Key File Touch Points (later, when approved)

- insanely_fast_whisper_rocm/core/formatters.py:74 (SRT), :216 (VTT)
- insanely_fast_whisper_rocm/utils/format_time.py
- New: insanely_fast_whisper_rocm/core/subtitles/policy.py, segmenter.py, line_splitter.py, validator.py
- Tests: tests/test_srt_time_format.py, tests/test_line_splitter.py, tests/test_segment_grouper.py, and selective updates to existing CLI/API tests to assert commas and basic shape.

## Questions (before I change anything)

- Should I fix the SRT timestamp comma bug first as a focused PR?
- Are you happy with introducing SubtitlePolicy plus SegmentGrouper/LineSplitter/Validator as small, test-driven PRs?
- Any preferred defaults (e.g., max_cps=17, min_duration=1.2s, max_duration=5.5s, max_line_chars=42, max_lines=2) you want changed?
- Do you want policy exposure via CLI/env now, or keep defaults internal for the initial refactor?
- If you confirm the plan and the first step, I’ll proceed with PR 1 (timestamp separator fix) and add a minimal test to lock it in.
