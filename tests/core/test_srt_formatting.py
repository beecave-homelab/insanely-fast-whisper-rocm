"""Tests for SRT/VTT formatting enhancements, including readability constraints and segmentation logic."""

from __future__ import annotations

from insanely_fast_whisper_rocm.core.formatters import SrtFormatter
from insanely_fast_whisper_rocm.core.segmentation import Word, segment_words
from insanely_fast_whisper_rocm.utils import constants


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

        # For extremely long text, it may need to be broken into multiple segments
        if len(result) == 1:
            # Single segment should have properly wrapped lines
            assert result[0].text.count("\n") <= 1  # At most 2 lines
            lines = result[0].text.split("\n")
            for line in lines:
                assert len(line) <= constants.MAX_LINE_CHARS, (
                    f"Line too long: {len(line)} > {constants.MAX_LINE_CHARS}"
                )
        else:
            # Multiple segments should each respect character limits
            for seg in result:
                assert len(seg.text) <= constants.MAX_BLOCK_CHARS, (
                    f"Segment too long: {len(seg.text)} > {constants.MAX_BLOCK_CHARS}"
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
        assert len(result) <= 2

        for segment in result:
            lines = segment.text.split("\n")
            assert len(lines) <= 2
            for line in lines:
                assert len(line) <= constants.MAX_LINE_CHARS

    def test_paragraph_with_comma_only_lines(self) -> None:
        """Ensure multi-clause narration avoids comma-only lines and overlong blocks."""
        words = [
            Word(text="We've", start=60.379, end=60.700),
            Word(text="got", start=60.700, end=60.900),
            Word(text="articles,", start=60.900, end=61.100),
            Word(text="some", start=61.200, end=61.400),
            Word(text="detailed", start=61.400, end=61.700),
            Word(text="research,", start=61.700, end=62.000),
            Word(text="and", start=62.000, end=62.200),
            Word(text="practical", start=62.200, end=62.500),
            Word(text="notes", start=62.500, end=62.700),
            Word(text="on", start=62.700, end=62.900),
            Word(text="co-pilot", start=62.900, end=63.300),
            Word(text="studio", start=63.300, end=63.600),
            Word(text="itself.", start=63.600, end=63.900),
        ]

        segments = segment_words(words)
        assert len(segments) >= 2, (
            "Comma-rich narration should be split into multiple segments"
        )

        for segment in segments:
            assert segment.text.strip(",") != "", (
                "Segments must contain lexical content beyond commas"
            )
            for line in segment.text.split("\n"):
                assert len(line) <= constants.MAX_LINE_CHARS, (
                    f"Line too long: {len(line)} > {constants.MAX_LINE_CHARS}"
                )

    def test_comma_rich_clause_wrapping(self) -> None:
        """Ensure individual comma-heavy clauses comply with MAX_LINE_CHARS."""
        words = [
            Word(text="Our", start=0.0, end=0.3),
            Word(text="mission,", start=0.3, end=0.6),
            Word(text="really,", start=0.6, end=0.9),
            Word(text="is", start=0.9, end=1.2),
            Word(text="to", start=1.2, end=1.5),
            Word(text="unpack", start=1.5, end=1.9),
            Word(text="how", start=1.9, end=2.2),
            Word(text="these", start=2.2, end=2.5),
            Word(text="declarative", start=2.5, end=2.9),
            Word(text="agents", start=2.9, end=3.2),
            Word(text="work,", start=3.2, end=3.5),
            Word(text="what", start=3.5, end=3.8),
            Word(text="they", start=3.8, end=4.1),
            Word(text="are.", start=4.1, end=4.5),
        ]
        segments = segment_words(words)
        assert segments, "Segmentation should return at least one segment"

        for segment in segments:
            for line in segment.text.split("\n"):
                assert len(line) <= constants.MAX_LINE_CHARS, (
                    f"Line too long: {len(line)} > {constants.MAX_LINE_CHARS}"
                )

    def test_performance_based_clause_wrapping(self) -> None:
        """Ensure hyphenated clauses stay within MAX_LINE_CHARS."""
        words = [
            Word(text="I", start=0.0, end=0.3),
            Word(text="believe", start=0.3, end=0.6),
            Word(text="the", start=0.6, end=0.9),
            Word(text="performance-based", start=0.9, end=1.2),
            Word(text="selection", start=1.2, end=1.5),
            Word(text="method,", start=1.5, end=1.8),
            Word(text="BVP", start=1.8, end=2.1),
            Word(text="offers", start=2.1, end=2.4),
            Word(text="the", start=2.4, end=2.7),
            Word(text="essential", start=2.7, end=3.0),
            Word(text="mechanism", start=3.0, end=3.3),
        ]

        segments = segment_words(words)
        assert segments, "Segmentation should produce at least one segment"

        for segment in segments:
            for line in segment.text.split("\n"):
                assert len(line) <= constants.MAX_LINE_CHARS, (
                    f"Line too long: {len(line)} > {constants.MAX_LINE_CHARS}"
                )

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

    def test_fallback_splits_long_chunks_by_duration(self) -> None:
        """Chunk fallback should split long durations into bounded SRT blocks."""
        long_duration = constants.MAX_SEGMENT_DURATION_SEC * 3.0
        chunk_text = " ".join([
            "Focusing",
            "on",
            "distilled",
            "insights",
            "helps",
            "teams",
            "move",
            "faster",
            "with",
            "confidence",
            "around",
            "mission",
            "critical",
            "software",
            "decisions",
            "and",
            "keeps",
            "stakeholders",
            "aligned",
            "throughout",
            "the",
            "evaluation",
            "process",
            "even",
            "when",
            "constraints",
            "shift",
            "unexpectedly",
            "mid",
            "project",
        ])

        result = {
            "chunks": [
                {
                    "text": chunk_text,
                    "start": 0.0,
                    "end": long_duration,
                }
            ]
        }

        srt_output = SrtFormatter.format(result)
        blocks = [block for block in srt_output.strip().split("\n\n") if block.strip()]
        assert len(blocks) > 1, "Expected long chunk to split into multiple segments"

        tolerance = 1e-6
        for block in blocks:
            lines = block.split("\n")
            assert len(lines) >= 2
            timing_line = lines[1]
            assert "-->" in timing_line
            start_raw, end_raw = timing_line.split("-->")
            start_seconds = self._parse_srt_time(start_raw.strip())
            end_seconds = self._parse_srt_time(end_raw.strip())
            duration = end_seconds - start_seconds
            assert duration <= constants.MAX_SEGMENT_DURATION_SEC + tolerance

    def test_srt_overlapping_timestamps_validation(self) -> None:
        """Test that SRT formatter handles overlapping timestamps correctly.

        This test demonstrates the issue where overlapping word-level timestamps
        create invalid SRT output with overlapping time ranges.
        """
        # Create chunks with overlapping timestamps (simulating the problematic output)
        overlapping_chunks = [
            {"text": "Welcome", "timestamp": [0.14, 0.44]},
            {"text": "to", "timestamp": [0.44, 0.76]},
            {"text": "the", "timestamp": [0.76, 1.0]},
            {"text": "deep", "timestamp": [1.0, 1.3]},
            {"text": "dive", "timestamp": [1.3, 1.6]},
            # These overlap with previous segments
            {
                "text": "Where",
                "timestamp": [0.379, 0.732],
            },  # Overlaps with "to" and "the"
            {
                "text": "are",
                "timestamp": [0.939, 1.05],
            },  # Overlaps with "the" and "deep"
            {
                "text": "you",
                "timestamp": [1.499, 1.617],
            },  # Overlaps with "deep" and "dive"
        ]

        result = {"chunks": overlapping_chunks}
        srt = SrtFormatter.format(result)

        # Parse SRT to check for overlaps
        blocks = [b for b in srt.strip().split("\n\n") if b.strip()]
        timestamps = []

        for block in blocks:
            lines = block.split("\n")
            if len(lines) >= 2:
                timing_line = lines[1]
                if "-->" in timing_line:
                    start, end = timing_line.split("-->")
                    start = self._parse_srt_time(start.strip())
                    end = self._parse_srt_time(end.strip())
                    timestamps.append((start, end))

        # Check that no timestamps overlap
        for i in range(len(timestamps) - 1):
            assert timestamps[i][1] <= timestamps[i + 1][0], (
                f"Overlapping timestamps found: {timestamps[i]} overlaps with {timestamps[i + 1]}"
            )

    def test_long_text_segmentation(self) -> None:
        """Test that very long text is properly broken into multiple segments.

        This test addresses the issue where extremely long sentences create
        SRT blocks that exceed readability limits.
        """
        # Create a very long sentence that should be broken into multiple segments
        long_text = (
            "Where are your shortcut through all that information noise, "
            "getting you straight to the knowledge that really counts and "
            "making sure you never miss the important details that could "
            "change everything for your business and help you stay ahead "
            "of the competition in this fast-paced digital world?"
        )

        words = [Word(text=long_text, start=0.0, end=10.0)]
        segments = segment_words(words)

        # Should create multiple segments to keep each under character limits
        assert len(segments) > 1, "Long text should be broken into multiple segments"

        # Each segment should respect character limits
        for seg in segments:
            assert len(seg.text) <= constants.MAX_BLOCK_CHARS, (
                f"Segment too long: {len(seg.text)} chars > {constants.MAX_BLOCK_CHARS}"
            )

        # Segments should be in chronological order
        for i in range(len(segments) - 1):
            assert segments[i].end <= segments[i + 1].start, (
                "Segments should be in chronological order"
            )

    def _parse_srt_time(self, time_str: str) -> float:
        """Parse SRT time format (HH:MM:SS,mmm) to seconds.

        Args:
            time_str: SRT timestamp string in ``HH:MM:SS,mmm`` format.

        Returns:
            float: Timestamp in seconds.
        """
        parts = time_str.replace(",", ".").split(":")
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = float(parts[2])
        return hours * 3600 + minutes * 60 + seconds
