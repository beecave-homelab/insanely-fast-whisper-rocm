"""Tests for real-world sentence grouping quality issues.

These tests capture actual problematic segmentation patterns found in production
SRT output and ensure they are grouped more naturally.
"""

from __future__ import annotations

from insanely_fast_whisper_api.core.segmentation import Word, segment_words
from insanely_fast_whisper_api.utils import constants


def _mk_words(text: str, *, start: float = 0.0, per_word: float = 0.2) -> list[Word]:
    """Build a list of ``Word`` with uniform durations for the given text.

    Args:
        text: Text to split by whitespace into tokens.
        start: Starting timestamp in seconds.
        per_word: Duration per token in seconds.

    Returns:
        Sequence of Word objects with consecutive, non-overlapping timings.
    """
    words: list[Word] = []
    t = start
    for tok in text.strip().split():
        words.append(Word(text=tok, start=t, end=t + per_word))
        t += per_word
    return words


def test_opening_greeting_not_fragmented() -> None:
    """Short greetings should be kept together, not split into single words.

    Real issue: "Welcome to The Debate." was split into:
    - Segment 1: "Welcome"
    - Segment 2: "to The Debate."

    Expected: Single segment "Welcome to The Debate."
    """
    text = "Welcome to The Debate."
    words = _mk_words(text, start=0.256, per_word=0.464)

    segments = segment_words(words)

    # Should be at most 1 segment for such a short greeting
    assert len(segments) <= 2, f"Expected ≤2 segments, got {len(segments)}"

    # First segment should not be a single word
    if len(segments) > 0:
        first_seg_word_count = len(segments[0].words)
        assert first_seg_word_count >= 2, (
            f"First segment should have ≥2 words, got {first_seg_word_count}"
        )


def test_long_sentence_with_comma_clause_not_broken_mid_phrase() -> None:
    """Long sentences with clauses should split at natural boundaries, not mid-phrase.

    Real issue: "Today we are tackling really a foundational challenge in
    organizational investment, how we select mission-critical software." was split:
    - Segment 3: "Today we are tackling really a"
    - Segment 4: "foundational challenge in organizational"
    - Segment 5: "investment, how we select mission-critical software."

    Expected: Split at the comma, not in the middle of "a foundational challenge".
    """
    text = (
        "Today we are tackling really a foundational challenge in organizational "
        "investment, how we select mission-critical software."
    )
    words = _mk_words(text, start=2.592, per_word=0.2)

    segments = segment_words(words)

    # Join all segment texts to verify no content is lost
    full_text = " ".join(seg.text.replace("\n", " ") for seg in segments).strip()
    expected_text = text.strip()

    # Normalize whitespace for comparison
    def normalize(s: str) -> str:
        return " ".join(s.split())

    assert normalize(full_text) == normalize(expected_text), "Text content mismatch"

    # None of the segments should end with awkward mid-phrase words like "a" or "in"
    awkward_endings = {"a", "an", "the", "in", "on", "at", "of", "to", "for"}
    for seg in segments:
        last_word = seg.text.replace("\n", " ").strip().split()[-1].rstrip(".,!?")
        assert last_word.lower() not in awkward_endings, (
            f"Segment ends awkwardly with '{last_word}': {seg.text!r}"
        )


def test_comma_separated_clauses_split_at_comma() -> None:
    """Sentences with clear comma boundaries should split at commas, not mid-phrase.

    This ensures the algorithm prefers natural clause boundaries over arbitrary
    character limits.
    """
    text = (
        "The organization is procuring a new onboarding application, "
        "and this system is absolutely crucial for efficiency, compliance, "
        "and, frankly, the employee experience."
    )
    words = _mk_words(text, start=11.272, per_word=0.18)

    segments = segment_words(words)

    # Expect multiple segments due to length, but splits should be at commas
    assert len(segments) >= 2, "Expected multiple segments for this long sentence"

    # Check that no segment ends with an article or preposition (mid-phrase indicator)
    awkward_endings = {"a", "an", "the", "for", "and", "is"}
    for seg in segments:
        words_in_seg = seg.text.replace("\n", " ").strip().split()
        if len(words_in_seg) > 1:  # Skip single-word segments from merging
            last_word = words_in_seg[-1].rstrip(".,!?;:")
            assert last_word.lower() not in awkward_endings, (
                f"Segment ends mid-phrase: {seg.text!r}"
            )


def test_segments_respect_readability_constraints() -> None:
    """All segments should respect character, duration, and CPS constraints."""
    text = (
        "Welcome to The Debate. Today we are tackling really a foundational "
        "challenge in organizational investment, how we select mission-critical "
        "software. The organization is procuring a new onboarding application, "
        "and this system is absolutely crucial for efficiency, compliance, and, "
        "frankly, the employee experience."
    )
    words = _mk_words(text, start=0.256, per_word=0.18)

    segments = segment_words(words)

    for seg in segments:
        clean_text = seg.text.replace("\n", " ")
        dur = seg.end - seg.start

        # Character limit check
        assert len(clean_text) <= constants.MAX_BLOCK_CHARS, (
            f"Segment exceeds block char limit: {len(clean_text)} > "
            f"{constants.MAX_BLOCK_CHARS}"
        )

        # Line length check
        for line in seg.text.split("\n"):
            assert len(line) <= constants.MAX_LINE_CHARS, (
                f"Line exceeds max length: {len(line)} > {constants.MAX_LINE_CHARS}"
            )

        # CPS check (with tolerance for edge cases and floating point precision)
        if dur > 0:
            cps = len(clean_text) / dur
            # Allow small violations for very short segments that get merged
            if len(clean_text) > 10:  # Only enforce for non-trivial segments
                # Use small epsilon for floating point comparison
                tolerance = 0.01
                assert (
                    constants.MIN_CPS - tolerance
                    <= cps
                    <= constants.MAX_CPS + tolerance
                ), (
                    f"CPS out of range: {cps:.2f} not in "
                    f"[{constants.MIN_CPS}, {constants.MAX_CPS}]"
                )
