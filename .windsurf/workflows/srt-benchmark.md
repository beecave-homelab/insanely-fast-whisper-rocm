---
description: Benchmark subtitles and inspect srt_quality metrics
auto_execution_mode: 3
---

# SRT quality benchmark workflow

## Purpose

Guide to regenerate benchmarks, collect SRT quality metrics, and compare word vs chunk timestamp behavior.

## Steps

1. **Reset artifacts** (only after recent code changes that could effect benchmarks)

   ```bash
   scripts/clean_benchmark.sh
   ```

   Removes cached benchmark JSON, SRT, and TXT files so the new run is isolated.

2. **Run word-level benchmark variants**

   Choose one of the files below that best matches what kind of benchmark we want to test:

   - `uploads/Weighted_Scorecard_vs_5_minute_test.mp3` (short)
   - `uploads/Weighted_Scorecard_vs_medium_test.mp3` (medium)
   - `uploads/Weighted_Scorecard_vs.mp3` (large +- 17 minutes)

   ```bash
   scripts/benchmark.sh -a "uploads/Weighted_Scorecard_vs_5_minute_test.mp3" \
     -m distil-whisper/distil-large-v3.5 \
     --timestamp-types word \
     --batch-word 4
   ```

   Captures baseline, stabilize, demucs, VAD, and combination variants with word timestamps.

3. **Run chunk-level baseline (fallback validation)**

   ```bash
   scripts/benchmark.sh -a "uploads/Weighted_Scorecard_vs_5_minute_test.mp3" \
     -m distil-whisper/distil-large-v3.5 \
     --timestamp-types chunk
   ```

   Produces JSON/SRT outputs that rely on the chunk fallback path to confirm duration splitting behavior.

4. **Inspect benchmark JSON metrics**

   ```bash
   # Example: open baseline JSON
   sed -n '1,160p' benchmarks/Weighted_Scorecard_vs_5_minute_test_transcribe_<timestamp>.json
   ```

   Focus on `format_quality.srt.score`, `duration_stats`, `boundary_counts`, `cps_histogram`, and `sample_offenders`.

5. **Compare word vs chunk results**

   - **Quality score**: confirm chunk fallback raises `score` when word segmentation still has long spans.
   - **Durations**: note `duration_stats.max_seconds` and `boundary_counts.too_long` differences.
   - **CPS**: verify `cps_histogram` and offender lists reflect improvements with fallback.

6. **Review SRT outputs for timing accuracy**

   ```bash
   head -n 80 transcripts-srt/Weighted_Scorecard_vs_5_minute_test_transcribe_<timestamp>.srt
   ```

   Compare word-based vs chunk-based SRT files to assess alignment vs readability trade-offs.

7. **Document findings**

   Summarize observed improvements or regressions (e.g., more balanced durations, remaining boundary violations) and note follow-up actions for `segment_words()` or `compute_srt_quality()`.
