"""Tests for timing sanitation in segmentation to improve CPS and stability."""

from __future__ import annotations

from insanely_fast_whisper_rocm.core.segmentation import Word, segment_words
from insanely_fast_whisper_rocm.utils import constants


class TestSegmentationTimingSanitization:
    """Validate that segmentation sanitizes word timings for stability.

    The expectations are:
    - Zero-duration words are expanded to a minimal positive duration.
    - Word timings are monotonically non-decreasing across the sequence.
    - Produced segments have non-zero duration and plausible CPS.
    """

    def test_zero_duration_word_is_expanded(self) -> None:
        """A word with start==end must be expanded to have positive duration."""
        words = [Word(text="We're", start=9.28, end=9.28)]
        segments = segment_words(words)
        assert len(segments) >= 1
        seg = segments[0]
        assert seg.end > seg.start
        assert (seg.end - seg.start) > 0.0

    def test_monotonicity_across_words(self) -> None:
        """Overlapping or zero-duration words should be made monotonic."""
        words = [
            Word(text="A", start=1.0, end=1.2),
            Word(text="B", start=1.2, end=1.2),  # zero-duration
            Word(text="C", start=1.19, end=1.25),  # overlap
            Word(text="D", start=1.25, end=1.24),  # inverted
        ]
        segments = segment_words(words)
        # Ensure segments come out with non-decreasing times
        prev_end = -1.0
        for seg in segments:
            assert seg.start >= 0.0
            assert seg.end >= seg.start
            assert seg.start >= prev_end
            prev_end = seg.end

    def test_cps_positive_after_sanitization(self) -> None:
        """CPS for a short caption should be positive and finite after sanitation."""
        words = [
            Word(text="We", start=0.0, end=0.0),  # zero-duration
            Word(text="are", start=0.0, end=0.0),  # zero-duration
            Word(text="here.", start=0.0, end=0.0),  # zero-duration
        ]
        segments = segment_words(words)
        assert segments, "Expected at least one segment"
        total_text = sum(len(s.text) for s in segments)
        total_dur = sum((s.end - s.start) for s in segments)
        assert total_dur > 0.0
        cps = total_text / total_dur
        # Must be finite and > 0; do not assert tight bounds to avoid flakiness
        assert cps > 0.0
        assert cps < constants.MAX_CPS * 4
