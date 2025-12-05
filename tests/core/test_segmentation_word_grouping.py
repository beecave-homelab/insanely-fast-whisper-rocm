"""Tests for word grouping in segmentation logic.

This test suite verifies that segment_words() correctly groups individual
word-level timestamps (from stable-ts) into readable phrases/sentences,
rather than creating one subtitle per word.
"""

from __future__ import annotations

from insanely_fast_whisper_api.core.segmentation import Word, segment_words


def test_groups_individual_words_into_phrases() -> None:
    """Ensure individual words are grouped into readable phrases.

    This test simulates what happens when stable-ts returns word-level
    timestamps where each word is a separate item. The segmentation logic
    should group these into readable subtitle segments, not one word per line.

    Args:
        None
    """
    # Simulate word-level data from stable-ts (like the bug scenario)
    words = [
        Word(text="Welcome", start=0.256, end=0.720),
        Word(text="to", start=0.720, end=1.120),
        Word(text="the", start=1.120, end=1.379),
        Word(text="debate.", start=1.379, end=2.259),
        Word(text="Today", start=2.592, end=2.839),
        Word(text="we", start=2.839, end=3.020),
        Word(text="are", start=3.020, end=3.220),
        Word(text="tackling", start=3.220, end=3.700),
        Word(text="really", start=3.700, end=5.000),
        Word(text="a", start=5.000, end=5.200),
        Word(text="foundational", start=5.200, end=5.759),
        Word(text="challenge", start=5.759, end=6.500),
        Word(text="in", start=6.500, end=6.700),
        Word(text="organizational", start=6.700, end=7.500),
        Word(text="investment.", start=7.500, end=8.000),
    ]

    segments = segment_words(words)

    # Should NOT create one segment per word
    assert len(segments) < len(words), (
        f"Expected fewer segments than words, got {len(segments)} segments "
        f"for {len(words)} words"
    )

    # The key test: segments should group words, not be one-word-per-segment
    # Check that at least some segments have multiple words
    multi_word_segments = [
        seg for seg in segments if seg.text.replace("\n", " ").count(" ") >= 1
    ]
    assert len(multi_word_segments) >= len(segments) // 2, (
        f"Expected at least half of segments to have multiple words, "
        f"got {len(multi_word_segments)} multi-word out of {len(segments)} total"
    )

    # First segment should ideally contain "Welcome" and start the sentence
    first_segment_text = segments[0].text.replace("\n", " ")
    assert "Welcome" in first_segment_text or "welcome" in first_segment_text.lower(), (
        f"First segment should contain 'Welcome', got: '{first_segment_text}'"
    )

    # Verify sentence boundaries are respected - segments ending with periods
    # should not be split mid-sentence
    period_segments = [seg for seg in segments if "." in seg.text]
    assert len(period_segments) >= 2, (
        f"Expected at least 2 segments with periods (sentence endings), "
        f"got {len(period_segments)}"
    )


def test_respects_sentence_boundaries() -> None:
    """Ensure segments respect sentence-ending punctuation.

    Args:
        None
    """
    words = [
        Word(text="Hello", start=0.0, end=0.5),
        Word(text="world.", start=0.5, end=1.0),
        Word(text="How", start=1.5, end=1.8),
        Word(text="are", start=1.8, end=2.0),
        Word(text="you?", start=2.0, end=2.5),
    ]

    segments = segment_words(words)

    # Should create at least 2 segments (two sentences)
    assert len(segments) >= 2, f"Expected at least 2 segments, got {len(segments)}"

    # First segment should end with period
    first_text = segments[0].text.replace("\n", " ")
    assert "world." in first_text or first_text.endswith(".")

    # Second segment should end with question mark
    second_text = segments[1].text.replace("\n", " ")
    assert "you?" in second_text or second_text.endswith("?")


def test_handles_long_sentences_with_splitting() -> None:
    """Ensure long sentences are split appropriately.

    Args:
        None
    """
    # Create a very long sentence that should be split
    words = []
    start_time = 0.0
    long_text = (
        "This is a very long sentence that contains many words and should be "
        "split into multiple subtitle segments for better readability and to "
        "respect the maximum character limits that we have defined."
    )

    for word in long_text.split():
        words.append(Word(text=word, start=start_time, end=start_time + 0.3))
        start_time += 0.35

    segments = segment_words(words)

    # Should create multiple segments due to length
    assert len(segments) > 1, (
        f"Expected multiple segments for long sentence, got {len(segments)}"
    )

    # Each segment should respect character limits (roughly)
    for seg in segments:
        seg_text = seg.text.replace("\n", " ")
        # Allow some flexibility but should generally be under 84 chars (2 lines)
        assert len(seg_text) < 100, (
            f"Segment too long ({len(seg_text)} chars): '{seg_text}'"
        )
