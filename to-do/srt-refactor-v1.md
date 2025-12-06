# srt-refactor-v1.md

## 1) Findings from the report: to-do/srt-formatting.md

I read [to-do/srt-formatting.md](cci:7://file:///home/elvee/Local-AI/insanely-fast-whisper-rocm/to-do/srt-formatting.md:0:0-0:0) end-to-end. Key takeaways:

- Industry standards:
  - Characters per line ≈ 32–42; max two lines per block.
  - Break lines at natural pauses (punctuation, before conjunctions/prepositions), avoid breaking names/compounds.
  - CPS caps around 15–17 with a minimum threshold to avoid flashes; duration typically 0.8–7s, with most guidance around 1.2–6s.
  - Avoid orphan short segments; merge considering soft/hard limits.
- Reference implementation (parakeet_nemo_asr_rocm):
  - Segmenter enforces CPS, line-length, duration limits with backtracking clause-aware splits and greedy fallbacks.
  - `split_lines(text)` balances two lines with min-line-length constraints.
  - Modular formatters for SRT/VTT/TXT with a registry.
- Recommended application:
  - Centralized, configurable readability constraints.
  - Typed `Word`/`Segment` models with timing.
  - Clause-aware segmentation + validator + formatter as separable modules.

## 2) Current codebase analysis (SRT/VTT path)

I audited the SRT-related code paths:

- Formatters live in [insanely_fast_whisper_rocm/core/formatters.py](cci:7://file:///home/elvee/Local-AI/insanely-fast-whisper-rocm/insanely_fast_whisper_rocm/core/formatters.py:0:0-0:0):
  - [SrtFormatter.format(...)](cci:1://file:///home/elvee/Local-AI/insanely-fast-whisper-rocm/insanely_fast_whisper_rocm/core/formatters.py:77:4-200:21) heuristics:
    - Chooses `segments` vs `chunks` based on valid timestamps.
    - Detects “word-like” data (duration ≤ 0.6s and ≤ 2 words) and groups into captions with simple heuristics:
      - Flush group if gap > 0.6s OR length > 42 OR duration > 3.5s, and at sentence-ending punctuation.
    - Formats timestamps via [insanely_fast_whisper_rocm.utils.format_time.format_seconds(...)](cci:1://file:///home/elvee/Local-AI/insanely-fast-whisper-rocm/insanely_fast_whisper_rocm/utils/format_time.py:6:0-24:76).
  - [VttFormatter.format(...)](cci:1://file:///home/elvee/Local-AI/insanely-fast-whisper-rocm/insanely_fast_whisper_rocm/core/formatters.py:216:4-294:31) is simpler and uses the same time formatter.
  - [TxtFormatter](cci:2://file:///home/elvee/Local-AI/insanely-fast-whisper-rocm/insanely_fast_whisper_rocm/core/formatters.py:38:0-71:20) is straightforward.
- Time formatting util: [insanely_fast_whisper_rocm/utils/format_time.py](cci:7://file:///home/elvee/Local-AI/insanely-fast-whisper-rocm/insanely_fast_whisper_rocm/utils/format_time.py:0:0-0:0)
  - [format_seconds(seconds)](cci:1://file:///home/elvee/Local-AI/insanely-fast-whisper-rocm/insanely_fast_whisper_rocm/utils/format_time.py:6:0-24:76) returns `HH:MM:SS.mmm` (with a dot).
- API layer: [insanely_fast_whisper_rocm/api/responses.py](cci:7://file:///home/elvee/Local-AI/insanely-fast-whisper-rocm/insanely_fast_whisper_rocm/api/responses.py:0:0-0:0)
  - Uses `FORMATTERS["srt" | "vtt"]` for SRT/VTT responses (good).
  - Also has internal [_seconds_to_timestamp](cci:1://file:///home/elvee/Local-AI/insanely-fast-whisper-rocm/insanely_fast_whisper_rocm/api/responses.py:26:4-43:67) that emits:
    - SRT with comma: `HH:MM:SS,mmm`
    - VTT with dot: `HH:MM:SS.mmm`
- CLI converter: [scripts/convert_json_to_txt_srt.py](cci:7://file:///home/elvee/Local-AI/insanely-fast-whisper-rocm/scripts/convert_json_to_txt_srt.py:0:0-0:0) uses the formatters directly.

Findings and gaps:

- SRT timestamp separator bug:
  - [SrtFormatter](cci:2://file:///home/elvee/Local-AI/insanely-fast-whisper-rocm/insanely_fast_whisper_rocm/core/formatters.py:74:0-210:20) uses [format_time.format_seconds](cci:1://file:///home/elvee/Local-AI/insanely-fast-whisper-rocm/insanely_fast_whisper_rocm/utils/format_time.py:6:0-24:76) which returns `HH:MM:SS.mmm` (dot), but SRT requires a comma separator. This is inconsistent with [ResponseFormatter._seconds_to_timestamp](cci:1://file:///home/elvee/Local-AI/insanely-fast-whisper-rocm/insanely_fast_whisper_rocm/api/responses.py:26:4-43:67) which correctly uses commas for SRT.
  - This will produce non-compliant SRT when the core formatter is used directly (e.g., CLI or batch export). Tests don’t assert the delimiter.
- Heuristic grouping is hardcoded:
  - Magic numbers (42, 3.5, 0.6) are embedded in [SrtFormatter](cci:2://file:///home/elvee/Local-AI/insanely-fast-whisper-rocm/insanely_fast_whisper_rocm/core/formatters.py:74:0-210:20). They are not configurable, making tuning and testing harder.
  - No centralized “readability policy” for CPS, duration, line-length, max lines, min line length, etc.
- Line splitting:
  - [SrtFormatter](cci:2://file:///home/elvee/Local-AI/insanely-fast-whisper-rocm/insanely_fast_whisper_rocm/core/formatters.py:74:0-210:20) preserves a single-line text per cue (`.replace("\n"," ")`), so it never attempts the balanced 1–2 line split required by best practices.
  - There’s no `split_lines(text)` step to break within a cue respecting max chars/line and natural breaks.
- Missing validator:
  - No pass where CPS, duration, line length, and max lines are validated post-grouping. Instead, grouping uses early stopping heuristics.
- Data modeling:
  - Formatters and API work with `dict`-like segments instead of typed `Word`/`Segment` value objects, limiting static testability and clarity.

## 3) Design patterns to improve testability

To make this feature truly testable and extensible, I recommend these patterns:

- Strategy + Policy Objects:
  - Keep `FORMATTERS` Strategy map (`txt`, `srt`, `vtt`) as is.
  - Introduce a `SubtitlePolicy` value object encapsulating all readability constraints (max line chars, max lines per block, min/max duration, min line length %, max/hard/soft block chars, min/max CPS, gap threshold, sentence-ending punctuation, soft boundary words).
  - Inject `SubtitlePolicy` into the segmenter and (optionally) into the formatter for line-breaking decisions.
- Functional core, imperative shell:
  - Move grouping/segmentation out of [SrtFormatter](cci:2://file:///home/elvee/Local-AI/insanely-fast-whisper-rocm/insanely_fast_whisper_rocm/core/formatters.py:74:0-210:20) into a pure, deterministic `SegmentGrouper` (no I/O, pure inputs→outputs).
  - [SrtFormatter](cci:2://file:///home/elvee/Local-AI/insanely-fast-whisper-rocm/insanely_fast_whisper_rocm/core/formatters.py:74:0-210:20) becomes a thin orchestrator calling pure components (grouper → line-splitter → timestamp formatter).
- Value Objects (dataclasses):
  - Define `Word(text: str, start: float, end: float)` and `Segment(text: str, start: float, end: float)` in a small `models` module. Provide converters from dicts to models and vice versa.
  - This enables type-safe tests and property-based tests.
- TimestampFormatter (Strategy or simple module):
  - Provide `format_srt_time(float) -> str` and `format_vtt_time(float) -> str`. Avoid a single function with format-agnostic output. This prevents the comma/dot bug category.
- LineSplitter component:
  - Implement a balanced `split_lines(text, policy)` function (pure), with explicit min line length threshold and preferred split points (punctuation, before conjunctions, whitespace).
- Validator component:
  - `SubtitleValidator` with pure checks: CPS in range, duration within bounds, chars per line <= max, lines <= max, monotonic timestamps, no overlaps.
  - Enables unit tests that validate outputs and produce fix suggestions or fail reasons.
- Ports & Adapters for configuration:
  - Inject `SubtitlePolicy` from either env or arguments. Keep defaults in one place.
- Tests via Golden Files and Properties:
  - Snapshot tests for formatter outputs (SRT/VTT) on seeded inputs.
  - Property tests for line splitting (e.g., never exceed max line length; minimal raggedness; no empty lines).
- Facade:
  - Optionally expose a `SubtitleFormatterFacade` that accepts raw ASR dict and returns SRT/VTT text using policy + components. This keeps CLI/API thin and easy to test independently.

## 4) Plan to enhance SRT formatting (no code changes yet)

I propose a small, incremental plan. This keeps risk low and lets us validate behavior with tests.

Phase 1 — Correctness and safety nets

- Fix SRT timestamp separator in the core path:
  - Introduce `format_srt_time()` and `format_vtt_time()` utilities or make [format_seconds](cci:1://file:///home/elvee/Local-AI/insanely-fast-whisper-rocm/insanely_fast_whisper_rocm/utils/format_time.py:6:0-24:76) accept a `style` enum to return the correct delimiter. Update [SrtFormatter](cci:2://file:///home/elvee/Local-AI/insanely-fast-whisper-rocm/insanely_fast_whisper_rocm/core/formatters.py:74:0-210:20) to use the SRT-compliant formatter.
  - Add unit tests asserting SRT uses commas and VTT uses dots.
- Add minimal tests around [SrtFormatter](cci:2://file:///home/elvee/Local-AI/insanely-fast-whisper-rocm/insanely_fast_whisper_rocm/core/formatters.py:74:0-210:20):
  - Given chunk-level timestamps, verify output numbering, time ordering, and the delimiter.
  - Given segments without timestamps, ensure graceful handling (skip or return empty with a warning).

Phase 2 — Extract policy and pure components

- Create `SubtitlePolicy` with defaults (aligned to the report’s guidance):
  - `max_line_chars=42`, `max_lines_per_block=2`, `max_cps=17`, `min_cps=12`, `min_duration=1.2`, `max_duration=5.5`, `soft_block_chars=90`, `gap_threshold=0.6`, `flush_punct=.:;!?`.
- Extract `SegmentGrouper` (pure):
  - Replace hard-coded magic numbers in [SrtFormatter](cci:2://file:///home/elvee/Local-AI/insanely-fast-whisper-rocm/insanely_fast_whisper_rocm/core/formatters.py:74:0-210:20) with a grouper taking the policy and producing readable segments from word-like chunks.
  - Maintain existing heuristics initially; add TODO hooks for clause-aware splitting.
- Add `LineSplitter`:
  - Implement balanced 2-line split with a minimum per-line length threshold (e.g., ≥ 10 chars and ≥ 25% of `max_line_chars`) and natural break preference.
  - Use in [SrtFormatter](cci:2://file:///home/elvee/Local-AI/insanely-fast-whisper-rocm/insanely_fast_whisper_rocm/core/formatters.py:74:0-210:20) when rendering cue text: join lines with newline to produce multi-line SRT entries.
- Tests:
  - Unit tests for `SegmentGrouper` (various word streams → merged segments).
  - Unit tests for `LineSplitter` (texts near edge cases).
  - Snapshot tests for SRT/VTT outputs on fixed inputs.

Phase 3 — Clause-aware segmentation and validator

- Add clause-aware split:
  - Prefer splitting at commas/semicolons/colons; backtrack to balance text and timing with CPS and duration constraints.
- Add `SubtitleValidator`:
  - Verify each output cue meets policy; if not, adjust splitting or flag regression in tests.
- Expand tests:
  - Property-based tests for CPS caps and max line length.
  - Snapshot tests covering:
    - Very fast speech
    - Long sentence broken by commas
    - Orphan short segments merged correctly.

Phase 4 — Integration and UX

- Make `SubtitlePolicy` configurable:
  - Optional env backing or CLI flags for `--max-line-chars`, `--max-cps`, etc. without breaking current defaults.
- Documentation:
  - Update README/CLI help to document readable defaults and new tuning knobs.

Test alignment with our repo standards:

- I’ll follow the “Testing Standards (pytest)” memory:
  - Unit tests fast/deterministic in `tests/`.
  - Use fixtures for tiny inputs; avoid heavy deps.
  - Add marker scaffolding only if needed.
  - Prefer pure function tests for the new components and snapshot-based tests for end-to-end outputs.

## 5) Risks and considerations

- Changing grouping/splitting can alter outputs, potentially affecting users relying on older structure. We’ll gate behavior under the new `SubtitlePolicy` defaults and write explicit tests to prevent regressions.
- Comma/dot fix for SRT is a correctness change and should be applied ASAP; it may change existing SRT outputs, but it’s required by spec.

## 6) Questions for you (before code changes)

- Do you want the SRT timestamp separator bug fixed immediately in [SrtFormatter](cci:2://file:///home/elvee/Local-AI/insanely-fast-whisper-rocm/insanely_fast_whisper_rocm/core/formatters.py:74:0-210:20) as a focused first PR?
- Are you OK with introducing `SubtitlePolicy` and extracting pure components (`SegmentGrouper`, `LineSplitter`, `SubtitleValidator`) over 2–3 small PRs?
- Any language-specific defaults you want different from the report (e.g., `MAX_CPS` 14 vs 17; `min_duration` 1.5s vs 1.2s; `max_lines_per_block` 2 vs 3)?
- Should we expose policy via CLI flags now or keep it internal defaults first?

---
