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


def test_segment_words_enforces_max_duration_with_short_fragments() -> None:
    """Merging many short segments should not exceed MAX_SEGMENT_DURATION_SEC.

    This reproduces the benchmark bug where _merge_short_segments() combines
    multiple short segments (each < MIN_SEGMENT_DURATION_SEC) into one segment
    that exceeds MAX_SEGMENT_DURATION_SEC.
    """
    from insanely_fast_whisper_api.utils import constants

    # Create many SHORT segments (each < MIN_SEGMENT_DURATION_SEC = 0.9s)
    # WITHOUT sentence-ending punctuation to trigger merging
    # If merged unchecked, they create a 30s+ segment
    words = [
        Word(text="It", start=0.0, end=0.5),
        Word(text="demands", start=3.0, end=3.7),
        Word(text="that", start=6.0, end=6.2),
        Word(text="we", start=9.0, end=9.1),
        Word(text="as", start=12.0, end=12.1),
        Word(text="the", start=15.0, end=15.1),
        Word(text="procuring", start=18.0, end=18.7),
        Word(text="entity", start=21.0, end=21.5),
        Word(text="precisely", start=24.0, end=24.6),
        Word(text="priorities", start=27.0, end=27.7),
        Word(text="up", start=30.0, end=30.1),
        Word(text="front", start=33.0, end=33.3),  # NO period!
    ]

    segments = segmentation.segment_words(words)

    # Debug: print created segments
    print(f"\nMAX_SEGMENT_DURATION_SEC = {constants.MAX_SEGMENT_DURATION_SEC}")
    print(f"MIN_SEGMENT_DURATION_SEC = {constants.MIN_SEGMENT_DURATION_SEC}")
    print(f"Created {len(segments)} segments:")
    for i, seg in enumerate(segments, 1):
        duration = seg.end - seg.start
        print(
            f"  Seg {i}: {duration:.2f}s ({seg.start:.2f} → {seg.end:.2f}) | {seg.text!r}"
        )

    # Assert all segments respect MAX_SEGMENT_DURATION_SEC
    for seg in segments:
        duration = seg.end - seg.start
        assert duration <= constants.MAX_SEGMENT_DURATION_SEC, (
            f"Segment exceeds max duration: {duration:.2f}s > "
            f"{constants.MAX_SEGMENT_DURATION_SEC}s. Text: {seg.text!r}"
        )


def test_segment_words_splits_on_large_gaps() -> None:
    """segment_words should split when gap between words exceeds threshold.

    This reproduces the benchmark bug where words separated by long silences
    (e.g., 2.5s gaps) are grouped into a single 27s+ segment.
    """
    from insanely_fast_whisper_api.core.segmentation import Word, segment_words
    from insanely_fast_whisper_api.utils import constants

    # Words with large gaps (simulating sparse audio with long silences)
    words = [
        Word(text="It", start=0.0, end=0.5),
        Word(text="demands", start=3.0, end=3.7),  # 2.5s gap
        Word(text="that", start=6.0, end=6.2),  # 2.3s gap
        Word(text="we,", start=9.0, end=9.1),  # 2.8s gap
        Word(text="as", start=12.0, end=12.1),  # 2.9s gap
        Word(text="the", start=15.0, end=15.1),  # 2.9s gap
        Word(text="procuring", start=18.0, end=18.7),  # 2.9s gap
        Word(text="entity,", start=21.0, end=21.5),  # 2.3s gap
        Word(text="precisely", start=24.0, end=24.6),  # 2.4s gap
        Word(text="priorities", start=27.0, end=27.7),  # 2.3s gap
        Word(text="up", start=30.0, end=30.1),  # 2.3s gap
        Word(text="front.", start=33.0, end=33.3),  # 2.9s gap
    ]

    segments = segment_words(words)

    # Debug output
    print(f"\nCreated {len(segments)} segments:")
    for i, seg in enumerate(segments, 1):
        duration = seg.end - seg.start
        print(
            f"  Seg {i}: {duration:.2f}s ({seg.start:.2f} → {seg.end:.2f}) | {seg.text!r}"
        )

    # Should split into multiple segments due to large gaps
    # At minimum, should not create a single 33s segment
    assert len(segments) > 1, (
        f"Expected multiple segments due to large gaps, got {len(segments)}"
    )

    # All segments must respect MAX_SEGMENT_DURATION_SEC
    for seg in segments:
        duration = seg.end - seg.start
        assert duration <= constants.MAX_SEGMENT_DURATION_SEC, (
            f"Segment exceeds max duration: {duration:.2f}s > "
            f"{constants.MAX_SEGMENT_DURATION_SEC}s. Text: {seg.text!r}"
        )


def test_merge_short_segments_respects_max_duration() -> None:
    """_merge_short_segments should not create segments exceeding MAX_SEGMENT_DURATION_SEC."""
    from insanely_fast_whisper_api.core.segmentation import (
        Segment,
        _merge_short_segments,
    )
    from insanely_fast_whisper_api.utils import constants

    # Create many short segments (each < MIN_SEGMENT_DURATION_SEC) without punctuation
    # These WILL be merged by _merge_short_segments()
    segments = [
        Segment(
            text="It", start=0.0, end=0.5, words=[Word(text="It", start=0.0, end=0.5)]
        ),
        Segment(
            text="demands",
            start=3.0,
            end=3.7,
            words=[Word(text="demands", start=3.0, end=3.7)],
        ),
        Segment(
            text="that",
            start=6.0,
            end=6.2,
            words=[Word(text="that", start=6.0, end=6.2)],
        ),
        Segment(
            text="we", start=9.0, end=9.1, words=[Word(text="we", start=9.0, end=9.1)]
        ),
        Segment(
            text="as",
            start=12.0,
            end=12.1,
            words=[Word(text="as", start=12.0, end=12.1)],
        ),
        Segment(
            text="the",
            start=15.0,
            end=15.1,
            words=[Word(text="the", start=15.0, end=15.1)],
        ),
        Segment(
            text="procuring",
            start=18.0,
            end=18.7,
            words=[Word(text="procuring", start=18.0, end=18.7)],
        ),
        Segment(
            text="entity",
            start=21.0,
            end=21.5,
            words=[Word(text="entity", start=21.0, end=21.5)],
        ),
        Segment(
            text="precisely",
            start=24.0,
            end=24.6,
            words=[Word(text="precisely", start=24.0, end=24.6)],
        ),
        Segment(
            text="priorities",
            start=27.0,
            end=27.7,
            words=[Word(text="priorities", start=27.0, end=27.7)],
        ),
        Segment(
            text="up",
            start=30.0,
            end=30.1,
            words=[Word(text="up", start=30.0, end=30.1)],
        ),
        Segment(
            text="front",
            start=33.0,
            end=33.3,
            words=[Word(text="front", start=33.0, end=33.3)],
        ),
    ]

    merged = _merge_short_segments(segments)

    print(f"\nBefore merge: {len(segments)} segments")
    print(f"After merge: {len(merged)} segments")
    for i, seg in enumerate(merged, 1):
        duration = seg.end - seg.start
        print(f"  Seg {i}: {duration:.2f}s | {seg.text!r}")

    # Assert merged segments still respect MAX_SEGMENT_DURATION_SEC
    for seg in merged:
        duration = seg.end - seg.start
        assert duration <= constants.MAX_SEGMENT_DURATION_SEC, (
            f"Merged segment exceeds max duration: {duration:.2f}s > "
            f"{constants.MAX_SEGMENT_DURATION_SEC}s. Text: {seg.text!r}"
        )


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
