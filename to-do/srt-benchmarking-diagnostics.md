# To-Do: Enhance SRT Benchmark Diagnostics

This plan outlines the steps to implement richer benchmarking diagnostics for SRT formatting and segmentation.

## Tasks

- [x] **Analysis Phase:**
  - [x] Research current metrics in `insanely_fast_whisper_rocm/utils/srt_quality.py`
    - Path: `insanely_fast_whisper_rocm/utils/srt_quality.py`
    - Action: Document existing sub-metrics and scoring logic
    - Analysis Results:
      - [x] Clarify overlap, hyphen, line-length, and CPS checks
      - [x] Identify integration points for new metrics
    - Accept Criteria: Summary explains current computation and identifies extension hooks (documented in dev notes dated 2025-10-07)
  - [x] Evaluate benchmark serialization pipeline
    - Path: `insanely_fast_whisper_rocm/cli/commands.py`
    - Action: Trace how SRT quality results are embedded into benchmark JSON
    - Analysis Results:
      - [x] Confirm structure of `format_quality` blocks
      - [x] List fields requiring updates for new diagnostics
    - Accept Criteria: Documented plan for augmenting benchmark output while maintaining backward compatibility (see inline comments added in PR #NNN)

- [x] **Implementation Phase:**
  - [x] Implement extended diagnostics in `compute_srt_quality()`
    - Path: `insanely_fast_whisper_rocm/utils/srt_quality.py`
    - Action: Add duration statistics, CPS histograms, boundary counts, and sample offenders
    - Status: Complete (merged via latest commit; see `compute_srt_quality()` output structure)
  - [x] Persist new diagnostics in benchmarks
    - Path: `insanely_fast_whisper_rocm/cli/commands.py`
    - Action: Ensure benchmark JSON captures additional `details` fields without breaking existing consumers
    - Status: Complete (`_handle_output_and_benchmarks()` passes `format_quality` with new details)

- [ ] **Testing Phase:**
  - [x] Unit tests for enhanced metrics
    - Path: `tests/core/test_srt_quality.py`
    - Action: Cover new detail fields and validate calculations with representative segments
    - Accept Criteria: Tests fail without new metrics and pass with implementation (`TestSrtQuality` now asserts duration stats, histogram, boundary counts, and offenders)
  - [x] Integration tests for benchmark serialization
    - Path: `tests/cli/test_commands.py` (or create new benchmark-focused tests)
    - Action: Assert benchmark JSON includes extended diagnostics when running mocked transcription
    - Accept Criteria: Snapshot or structured assertion confirms new fields (see `test_benchmark_includes_srt_quality` additions)

- [x] **Documentation Phase:**
  - [x] Update developer docs describing diagnostics
    - Path: `[project-overview.md]`
    - Action: Document SRT benchmarking metrics and interpretation guidance
    - Accept Criteria: Documentation explains each metric, thresholds, and debugging usage (see "SRT Formatting Diagnostics" subsection added on 2025-10-08)

## Related Files

- `insanely_fast_whisper_rocm/utils/srt_quality.py`
- `insanely_fast_whisper_rocm/cli/commands.py`
- `tests/core/test_srt_quality.py`
- `tests/cli/`
- `project-overview.md`

## Future Enhancements

- [ ] Expose diagnostics via CLI reporting or dashboards for quicker inspection
