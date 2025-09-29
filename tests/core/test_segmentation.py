"""Tests for the segmentation module."""

from __future__ import annotations

from insanely_fast_whisper_api.core import segmentation
from insanely_fast_whisper_api.core.segmentation import Word


def test_segment_words_does_not_drop_words() -> None:
    """Verify that segment_words does not drop words when forming lines."""
    # This simulates a long sentence that would previously cause words to be dropped.
    words = [
        Word(text="Welcome", start=0.0, end=0.5),
        Word(text="to", start=0.5, end=0.7),
        Word(text="the", start=0.7, end=0.9),
        Word(text="deep", start=0.9, end=1.2),
        Word(text="dive", start=1.2, end=1.5),
        Word(text="where", start=1.5, end=1.8),
        Word(text="your", start=1.8, end=2.1),
        Word(text="shortcut", start=2.1, end=2.7),
        Word(text="through", start=2.7, end=3.0),
        Word(text="all", start=3.0, end=3.2),
        Word(text="that", start=3.2, end=3.5),
        Word(text="information", start=3.5, end=4.2),
        # The bug occurs here: these words would be dropped.
        Word(text="noise", start=4.2, end=4.6),
        Word(text="getting", start=4.6, end=5.0),
        Word(text="you", start=5.0, end=5.2),
        Word(text="straight", start=5.2, end=5.7),
    ]

    grouped = segmentation.segment_words(words)

    # Reconstruct the full text from the grouped segments
    result_text = " ".join(seg.text for seg in grouped).replace("\n", " ")
    expected_text = " ".join(w.text for w in words)

    assert result_text == expected_text


def test_segment_words_produces_monotonic_segments() -> None:
    """Segments should have non-decreasing start times."""
    words = [
        Word(text="Alpha", start=0.0, end=0.04),
        Word(text="ends.", start=0.04, end=0.08),
        Word(text="Beta", start=0.08, end=0.12),
        Word(text="starts.", start=0.12, end=0.16),
    ]

    segments = segmentation.segment_words(words)

    prev_end = 0.0
    for seg in segments:
        assert seg.start >= prev_end
        assert seg.end >= seg.start
        prev_end = seg.end
