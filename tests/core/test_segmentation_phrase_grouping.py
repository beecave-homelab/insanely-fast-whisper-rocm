"""Tests for phrase-level cue grouping and natural split points.

These tests ensure that word-level timestamps are grouped into readable,
phrase-length subtitle cues, and that natural split points (e.g., conjunctions)
are respected when splitting long text.
"""

from __future__ import annotations

import pytest

from insanely_fast_whisper_api.core.formatters import build_quality_segments
from insanely_fast_whisper_api.core.segmentation import (
    Word,
    _find_natural_split_points,
    segment_words,
)
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


def _as_word_chunks(words: list[Word]) -> list[dict]:
    """Convert ``Word`` list into a Whisper-like ``chunks`` list.

    The formatter's ``_result_to_words`` supports extracting words from a
    ``chunks`` payload where each item has ``text`` and ``timestamp``.

    Args:
        words: Word objects to map into chunk dicts.

    Returns:
        List of chunk dictionaries suitable for ``build_quality_segments``.
    """
    return [{"text": w.text, "timestamp": [w.start, w.end]} for w in words]


@pytest.mark.parametrize(
    "text, expected_split_after",
    [
        (
            # No commas; expect a natural split after the first conjunction
            "This system is absolutely crucial for efficiency and compliance and reliability",
            [6, 8, 10],  # include 'for' and both 'and' occurrences
        ),
        (
            "We evaluated options but selected the weighted scorecard methodology",
            [4],
        ),
    ],
)
def test_find_natural_split_points_detects_boundaries(
    text: str, expected_split_after: list[int]
) -> None:
    """Ensure ``_find_natural_split_points`` detects conjunction boundaries.

    The indices returned are positions to start the next chunk (i+1), so they
    represent split-after positions.
    """
    words = _mk_words(text)
    splits = _find_natural_split_points(words)
    assert splits == expected_split_after


def test_segment_words_groups_into_phrases_with_commas() -> None:
    """Segment a comma-rich sentence into clause-level cues.

    Uses a sentence from the provided transcript.
    """
    sentence = (
        "The organization is procuring a new onboarding application, and this system "
        "is absolutely crucial for efficiency, compliance, and, frankly, the employee "
        "experience."
    )
    words = _mk_words(sentence, per_word=0.18)

    segs = segment_words(words)

    # Expect at least 2 segments (split near commas/clauses), not word-by-word
    assert len(segs) >= 2
    assert all(len(s.words) >= 2 for s in segs)

    # Check limits: block chars, line chars, and CPS constraints
    for s in segs:
        clean_text = s.text.replace("\n", " ")
        assert len(clean_text) <= constants.MAX_BLOCK_CHARS
        assert all(len(line) <= constants.MAX_LINE_CHARS for line in s.text.split("\n"))
        dur = s.end - s.start
        assert dur > 0
        cps = len(clean_text) / dur
        assert constants.MIN_CPS <= cps <= constants.MAX_CPS


def test_build_quality_segments_groups_word_timestamps_into_readable_segments() -> None:
    """``build_quality_segments`` should merge word timestamps into readable cues.

    We simulate a word-level result via ``chunks`` and assert that multiple words
    are grouped per returned segment and that limits are respected.
    """
    text = (
        "While I absolutely respect the need for control especially with compliance "
        "the complexity of this project demands more than just checking boxes on a list."
    )
    words = _mk_words(text, per_word=0.16)
    result = {"text": text, "chunks": _as_word_chunks(words)}

    segments = build_quality_segments(result)
    assert segments, "Expected non-empty quality segments"

    # Ensure grouping: average words per segment should be > 3
    # Approximate words per segment by splitting text and dividing by segment count
    tokens = len(text.split())
    avg_words = tokens / len(segments)
    assert avg_words > 3, f"Segments appear too granular: avg_words={avg_words:.2f}"

    # Validate per-segment constraints
    for seg in segments:
        seg_text = seg["text"]
        assert seg_text.strip(), "Empty segment text"
        assert len(seg_text.replace("\n", " ")) <= constants.MAX_BLOCK_CHARS
        assert all(
            len(line) <= constants.MAX_LINE_CHARS for line in seg_text.split("\n")
        )
        assert isinstance(seg["start"], float) and isinstance(seg["end"], float)
        assert seg["end"] > seg["start"], "Non-positive duration"
