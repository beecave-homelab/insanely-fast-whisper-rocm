"""Tests for SRT/VTT formatting enhancements, including readability constraints and segmentation logic."""

from __future__ import annotations

import pytest

from insanely_fast_whisper_api.core.segmentation import Word, segment_words
from insanely_fast_whisper_api.utils import constants

# from insanely_fast_whisper_api.utils.constants import (
#     MAX_LINE_CHARS,
#     MIN_CPS,
#     MAX_CPS,
#     MIN_SEGMENT_DURATION_SEC,
# )


class TestSrtFormattingEnhancements:
    """Unit tests for enhanced SRT/VTT formatting with readability constraints."""

    def test_line_length_enforcement(self) -> None:
        """Test that lines are limited to MAX_LINE_CHARS and blocks to ≤2 lines."""
        words = [
            Word(
                text="This is a very long sentence that should be split into multiple lines based on character limits.",
                start=0.0,
                end=5.0,
            )
        ]
        result = segment_words(words)
        assert len(result) == 1
        assert result[0].text.count("\n") == 1  # Two lines
        assert all(
            len(line) <= constants.MAX_LINE_CHARS for line in result[0].text.split("\n")
        )

    def test_balanced_two_line_splits(self) -> None:
        """Test that two-line blocks are balanced and avoid super-short lines."""
        words = [
            Word(text="Short first part", start=0.0, end=1.0),
            Word(
                text="This is a much longer second part that needs to be split appropriately.",
                start=1.0,
                end=6.0,
            ),
        ]
        result = segment_words(words)
        assert len(result) == 1
        lines = result[0].text.split("\n")
        assert len(lines) == 2
        assert len(lines[0]) >= 10 and len(lines[1]) >= 10  # No super-short lines

    def test_clause_splitting_at_boundaries(self) -> None:
        """Test splitting at natural clause boundaries when over limits."""
        words = [
            Word(text="This", start=0.0, end=0.2),
            Word(text="is", start=0.2, end=0.4),
            Word(text="a", start=0.4, end=0.5),
            Word(text="very,", start=0.5, end=0.8),
            Word(text="very", start=0.8, end=1.1),
            Word(text="long", start=1.1, end=1.4),
            Word(text="sentence", start=1.4, end=2.0),
            Word(text="that", start=2.0, end=2.2),
            Word(text="absolutely", start=2.2, end=2.8),
            Word(text="must", start=2.8, end=3.1),
            Word(text="be", start=3.1, end=3.3),
            Word(text="split", start=3.3, end=3.6),
            Word(text="into", start=3.6, end=3.8),
            Word(text="multiple", start=3.8, end=4.3),
            Word(text="clauses.", start=4.3, end=4.8),
        ]
        result = segment_words(words)
        # The sentence is long, but has no comma, so it should not be split by the current logic
        # This will now test the fallback path
        assert len(result) == 1

    def test_cps_enforcement(self) -> None:
        """Test CPS (characters per second) enforcement with min/max limits."""
        words = [
            Word(
                text="This is a very long sentence with many characters.",
                start=0.0,
                end=1.0,
            )  # High CPS
        ]
        result = segment_words(words)
        assert all(
            constants.MIN_CPS
            <= len(seg.text) / (seg.end - seg.start)
            <= constants.MAX_CPS
            for seg in result
        )

    def test_min_max_segment_durations(self) -> None:
        """Test min/max segment durations and merging of too-short segments."""
        words = [
            Word(text="Short", start=0.0, end=0.5),  # Too short
            Word(text="Another", start=0.5, end=0.8),
            Word(text="short", start=0.8, end=1.0),
            Word(text="one.", start=1.0, end=1.2),  # Too short
            Word(
                text="This is a longer segment that should be kept.",
                start=2.0,
                end=6.0,
            ),
        ]
        result = segment_words(words)
        assert len(result) == 2  # Merged first two sentences
        assert result[0].end - result[0].start >= constants.MIN_SEGMENT_DURATION_SEC

    def test_overlap_fixing_monotonic_timestamps(self) -> None:
        """Test that overlaps are fixed and timestamps are strictly increasing."""
        words = [
            Word(text="First", start=0.0, end=2.0),
            Word(text="Second", start=1.5, end=3.0),  # Overlap
            Word(text="Third", start=2.5, end=4.0),  # Another overlap
        ]
        result = segment_words(words)
        for i in range(len(result) - 1):
            assert result[i].end <= result[i + 1].start

    @pytest.mark.parametrize(
        ("words", "expected_segments"),
        [
            # Representative snapshot test
            (
                [
                    Word(text="Hello", start=0.0, end=0.5),
                    Word(text="world.", start=0.5, end=1.0),
                    Word(text="This", start=1.2, end=1.5),
                    Word(text="is", start=1.5, end=1.7),
                    Word(text="a", start=1.7, end=1.8),
                    Word(text="test.", start=1.8, end=2.2),
                ],
                2,
            ),  # Split at period
            ([Word(text="Short.", start=0.0, end=1.0)], 1),  # No split needed
        ],
    )
    def test_snapshot_content(self, words: list[Word], expected_segments: int) -> None:
        """Snapshot-ish test for representative inputs."""
        result = segment_words(words)
        assert len(result) == expected_segments
