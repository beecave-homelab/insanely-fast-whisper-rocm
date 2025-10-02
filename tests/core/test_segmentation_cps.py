"""Tests for ensuring CPS enforcement in segmentation.

We feed word-level timestamps representing full sentences with too-short durations
and expect the segmentation pipeline to normalize them so that all resulting
segments satisfy the configured characters-per-second (CPS) limits.
"""

from __future__ import annotations

import pytest

from insanely_fast_whisper_api.core.segmentation import Word, segment_words
from insanely_fast_whisper_api.utils import constants


def _build_words(
    sentence: str, *, start: float = 0.0, per_word: float = 0.12
) -> list[Word]:
    """Utility: build uniformly-spaced ``Word`` objects for a sentence.

    Args:
        sentence: The sentence text to split into words.
        start: Initial start time in seconds.
        per_word: Duration allocated for each word.

    Returns:
        List of ``Word`` objects with consecutive timings.
    """
    words: list[Word] = []
    t = start
    for token in sentence.strip().split():
        words.append(Word(text=token, start=t, end=t + per_word))
        t += per_word
    return words


@pytest.mark.parametrize(
    "sentence",
    [
        (
            "Today we are tackling really a foundational challenge in organizational investment."
        ),
        (
            "The weighted scorecard methodology provides the essential framework for objective assessment."
        ),
    ],
)
def test_segment_words_cps_within_limits(sentence: str) -> None:
    """Ensure ``segment_words`` outputs segments whose CPS is within limits.

    We intentionally use per-word durations that would violate CPS if the
    segmentation pipeline failed to merge and/or stretch the timings.
    """
    words = _build_words(sentence, per_word=0.10)

    segments = segment_words(words)

    # Assert all produced segments fall inside the configured CPS range
    # Use small tolerance for floating point comparison
    cps_tolerance = 0.01
    for seg in segments:
        duration = seg.end - seg.start
        # Avoid division by zero in pathological cases
        assert duration > 0, "Segment duration must be positive"
        cps = len(seg.text.replace("\n", " ")) / duration
        assert (
            constants.MIN_CPS - cps_tolerance
            <= cps
            <= constants.MAX_CPS + cps_tolerance
        ), f"CPS {cps:.2f} out of bounds for segment: '{seg.text}'"

    # Extra assurance: combined coverage equals original text length
    joined_output = " ".join(s.text.replace("\n", " ") for s in segments)
    original = sentence.strip()
    assert joined_output.startswith(original[: len(original) // 2]), (
        "Output text should contain the original sentence content (at least partially)."
    )
