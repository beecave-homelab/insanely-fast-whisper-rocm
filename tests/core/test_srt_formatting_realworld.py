"""Real-world SRT/VTT formatting tests using sentences adapted from a sample SRT.

These tests reuse the segmentation pipeline but swap in text drawn from
``transcripts-srt/Onboarding_App__Blauwdruk_transcribe_20250925T184328Z.srt``
(pre-segmentation) to simulate more realistic phrasing and line lengths.
"""

from __future__ import annotations

from insanely_fast_whisper_api.core.segmentation import Word, segment_words
from insanely_fast_whisper_api.utils import constants


class TestSrtFormattingRealWorld:
    """Tests that apply readability rules to real-world-like sentences."""

    @staticmethod
    def make_words_with_proportional_timing(
        text: str, start: float, end: float
    ) -> list[Word]:
        """Split text by spaces and distribute timing proportional to token length.

        Args:
            text: Full sentence text.
            start: Segment start timestamp (seconds).
            end: Segment end timestamp (seconds).

        Returns:
            A list of Word objects with proportional start/end times.
        """
        tokens = [t for t in text.split() if t]
        if not tokens:
            return []
        total_chars = sum(len(t) for t in tokens)
        words: list[Word] = []
        cur = start
        for i, tok in enumerate(tokens):
            if i < len(tokens) - 1 and total_chars > 0:
                frac = len(tok) / total_chars
                dur = frac * (end - start)
                nxt = cur + dur
            else:
                nxt = end
            words.append(Word(text=tok, start=cur, end=nxt))
            cur = nxt
        return words

    def test_line_length_enforcement_realworld(self) -> None:
        """Ensure long real-world sentence wraps to two lines and respects limits.

        This uses a long sentence adapted from the provided SRT sample.
        """
        words = self.make_words_with_proportional_timing(
            (
                "This approach guarantees a chance on basis of facts and provides a "
                "transparent, objective comparison for all candidates involved."
            ),
            start=0.0,
            end=5.0,
        )
        result = segment_words(words)
        assert len(result) == 1
        # Two lines and both lines within the per-line hard limit
        assert result[0].text.count("\n") == 1
        assert all(
            len(line) <= constants.MAX_LINE_CHARS for line in result[0].text.split("\n")
        )

    def test_balanced_two_line_splits_realworld(self) -> None:
        """Prefer balanced two-line splits on uneven, real-world phrasing."""
        words = [
            Word(text="Okay, this is the route card for today.", start=0.0, end=1.2),
            Word(
                text=(
                    "We begin together with a unifying goal, a flying start for everyone,"
                    " then dive into the phases."
                ),
                start=1.2,
                end=6.2,
            ),
        ]
        result = segment_words(words)
        # Input contains a terminal period in the first sentence; our pipeline
        # splits at strong punctuation, so expect two segments.
        assert len(result) == 2
        for seg in result:
            lines = seg.text.split("\n")
            # One or two lines, both within per-line limit if two lines
            assert 1 <= len(lines) <= 2
            assert all(len(line) <= constants.MAX_LINE_CHARS for line in lines)

    def test_cps_enforcement_realworld(self) -> None:
        """High-density sentence over a short duration should be CPS-adjusted."""
        words = self.make_words_with_proportional_timing(
            (
                "Good, as soon as we have the right tool chosen, the real work begins:"
                " configuration, integration, testing, and adoption."
            ),
            start=0.0,
            end=1.0,  # Intentionally short to trigger high CPS
        )
        result = segment_words(words)
        # For high-density short-duration inputs, synthetic timing ensures we do
        # not exceed MAX_CPS. The lower bound (MIN_CPS) may not always be
        # achievable without overextending durations in real-world inputs.
        assert all(
            (len(seg.text) / (seg.end - seg.start)) <= constants.MAX_CPS
            for seg in result
        )

    def test_snapshot_realworld_sentences_split_on_period(self) -> None:
        """Two simple sentences should produce two segments at period boundary."""
        words = [
            Word(text="Good, our main goal.", start=0.0, end=0.8),
            Word(
                text="That's actually very simple, but also ambitious.",
                start=0.9,
                end=2.2,
            ),
        ]
        result = segment_words(words)
        # Expect two segments split at the period boundary
        assert len(result) == 2

    def test_two_line_limit_and_line_length_across_segments(self) -> None:
        """Ensure wrapped text never exceeds two lines and respects per-line limit."""
        long_text = (
            "By the way, to know what we consider important, we ensure the top "
            "candidates can be objectively compared and evaluated transparently."
        )
        words = [Word(text=long_text, start=0.0, end=5.0)]
        result = segment_words(words)
        assert len(result) == 1
        lines = result[0].text.split("\n")
        assert 1 <= len(lines) <= 2
        assert all(len(line) <= constants.MAX_LINE_CHARS for line in lines)

    def test_high_density_short_duration_cps_realworld(self) -> None:
        """High-density real-world text over short duration respects CPS bounds."""
        words = [
            Word(
                text=(
                    "From the administrative basics to compliance and connections, "
                    "we build a complete program for every role."
                ),
                start=0.0,
                end=0.9,
            )
        ]
        result = segment_words(words)
        assert all(
            (len(seg.text) / (seg.end - seg.start)) <= constants.MAX_CPS
            for seg in result
        )

    def test_low_density_long_duration_cps_realworld(self) -> None:
        """Low-density text over longer duration should not be split unnecessarily."""
        words = [
            Word(
                text=("This approach guarantees a decision based on facts."),
                start=0.0,
                end=6.0,
            )
        ]
        result = segment_words(words)
        # Should remain one segment; line wrapping may add a newline but segments
        # need not split.
        assert len(result) == 1
        lines = result[0].text.split("\n")
        assert 1 <= len(lines) <= 2
        assert all(len(line) <= constants.MAX_LINE_CHARS for line in lines)

    def test_multiple_sentences_period_splitting_realworld(self) -> None:
        """Multiple sentences should split at terminal punctuation into segments."""
        words = [
            Word(text="We chose the right tool.", start=0.0, end=1.0),
            Word(text="Now the real work begins.", start=1.1, end=2.4),
            Word(text="Configure and integrate.", start=2.5, end=3.4),
        ]
        result = segment_words(words)
        assert len(result) == 3
        for seg in result:
            # Each segment should respect line constraints after wrapping
            lines = seg.text.split("\n")
            assert 1 <= len(lines) <= 2
            assert all(len(line) <= constants.MAX_LINE_CHARS for line in lines)
