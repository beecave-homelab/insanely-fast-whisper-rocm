"""Additional tests for formatters.py coverage."""

from __future__ import annotations

import pytest

from insanely_fast_whisper_api.core.formatters import (
    BaseFormatter,
    JsonFormatter,
    SrtFormatter,
    TxtFormatter,
    VttFormatter,
    _result_to_words,
    build_quality_segments,
)


def test_base_formatter_abstract_methods() -> None:
    """Test BaseFormatter abstract methods raise NotImplementedError.

    Covers lines 162, 167.
    """
    with pytest.raises(NotImplementedError):
        BaseFormatter.format({})

    with pytest.raises(NotImplementedError):
        BaseFormatter.get_file_extension()


def test_txt_formatter_handles_exceptions() -> None:
    """Test TxtFormatter handles various exception cases.

    Covers lines 191-193.
    """
    # Test with malformed text field that causes TypeError in the try block
    result = TxtFormatter.format({"text": None})  # type: ignore[dict-item]
    assert result == ""


def test_result_to_words_nested_word_structure() -> None:
    """Test _result_to_words extracts nested word structures in segments.

    Covers lines 59-71.
    """
    result = {
        "segments": [
            {
                "words": [
                    {"word": "Hello", "start": 0.0, "end": 0.5},
                    {"word": "world", "start": 0.5, "end": 1.0},
                ]
            },
            {
                "words": [
                    {"word": "foo", "start": 1.5, "end": 2.0},
                    {"word": "bar", "start": 2.0, "end": 2.5},
                ]
            },
        ]
    }

    words = _result_to_words(result)
    assert words is not None
    assert len(words) == 4
    assert words[0].text == "Hello"
    assert words[1].text == "world"


def test_result_to_words_segments_word_level() -> None:
    """Test _result_to_words with segments that are word-level (short duration).

    Covers lines 72-94.
    """
    result = {
        "segments": [
            {"text": "Hello", "start": 0.0, "end": 0.5},  # Short duration
            {"text": "world", "start": 0.5, "end": 1.0},  # Short duration
        ]
    }

    words = _result_to_words(result)
    assert words is not None
    assert len(words) == 2


def test_result_to_words_segments_sentence_level() -> None:
    """Test _result_to_words returns None for sentence-level segments (long duration).

    Covers lines 87-94.
    """
    result = {
        "segments": [
            {"text": "This is a long sentence.", "start": 0.0, "end": 5.0},  # Long
            {"text": "Another sentence.", "start": 5.0, "end": 10.0},  # Long
        ]
    }

    words = _result_to_words(result)
    assert words is None  # Should return None for sentence-level data


def test_build_quality_segments_fallback_with_empty_text() -> None:
    """Test build_quality_segments skips segments with empty text.

    Covers lines 127-140.
    """
    result = {
        "segments": [
            {"text": "", "start": 0.0, "end": 1.0},  # Empty text
            {"text": "Valid", "start": 1.0, "end": 2.0},
            {"text": "   ", "start": 2.0, "end": 3.0},  # Whitespace only
        ]
    }

    segments = build_quality_segments(result)
    assert len(segments) == 1
    assert segments[0]["text"] == "Valid"


def test_build_quality_segments_fallback_with_invalid_types() -> None:
    """Test build_quality_segments handles non-numeric timestamps.

    Covers lines 136-140.
    """
    result = {
        "chunks": [
            {"text": "Invalid", "timestamp": ["not", "numeric"]},  # Bad types
            {"text": "Valid", "timestamp": [0.0, 1.0]},
        ]
    }

    segments = build_quality_segments(result)
    # Only the valid one should be included
    assert len(segments) >= 1
    assert any("Valid" in s["text"] for s in segments)


def test_srt_formatter_fallback_with_timestamp_field() -> None:
    """Test SrtFormatter fallback uses 'timestamp' field correctly.

    Covers lines 297, 302.
    """
    result = {
        "chunks": [
            {"text": "Test chunk", "timestamp": [0.0, 2.0]},
        ]
    }

    srt = SrtFormatter.format(result)
    assert "00:00:00,000 --> 00:00:02,000" in srt
    assert "Test chunk" in srt


def test_srt_formatter_fallback_handles_chunk_errors() -> None:
    """Test SrtFormatter logs but continues on chunk formatting errors.

    Covers lines 313-314.
    """
    result = {
        "chunks": [
            {"text": "Valid", "timestamp": [0.0, 1.0]},
            {"text": "Bad", "timestamp": "not a list"},  # Bad timestamp
            {"text": "Also valid", "timestamp": [2.0, 3.0]},
        ]
    }

    srt = SrtFormatter.format(result)
    # Should still produce output for valid chunks
    assert "Valid" in srt
    assert "Also valid" in srt


def test_srt_formatter_fallback_handles_top_level_errors() -> None:
    """Test SrtFormatter handles top-level exceptions gracefully.

    Covers lines 319-321.
    """
    # Create a result that will cause errors in the try block
    result = {"chunks": "not a list"}  # type: ignore[dict-item]

    srt = SrtFormatter.format(result)
    # Should handle the error gracefully
    assert srt == "" or "1\n" in srt  # May return empty or partial content


def test_vtt_formatter_fallback_no_chunks() -> None:
    """Test VttFormatter returns empty VTT when no chunks/segments found.

    Covers lines 495-498.
    """
    result = {}

    vtt = VttFormatter.format(result)
    assert vtt == "WEBVTT\n\n"


def test_vtt_formatter_fallback_with_timestamp_field() -> None:
    """Test VttFormatter fallback uses 'timestamp' field correctly.

    Covers lines 516, 520, 521.
    """
    result = {
        "chunks": [
            {"text": "Test chunk", "timestamp": [0.0, 2.0]},
            {"text": "Skip this", "timestamp": [-1, -1]},  # Invalid
        ]
    }

    vtt = VttFormatter.format(result)
    assert "WEBVTT" in vtt
    # The timestamp format might be slightly different due to precision
    assert "00:00:00" in vtt and "00:00:02" in vtt
    assert "Test chunk" in vtt


def test_vtt_formatter_fallback_handles_chunk_errors() -> None:
    """Test VttFormatter logs but continues on chunk formatting errors.

    Covers lines 531-532.
    """
    result = {
        "chunks": [
            {"text": "Valid", "timestamp": [0.0, 1.0]},
            {"text": "Bad", "timestamp": "not a list"},  # Bad timestamp
            {"text": "Also valid", "timestamp": [2.0, 3.0]},
        ]
    }

    vtt = VttFormatter.format(result)
    # Should still produce output for valid chunks
    assert "Valid" in vtt
    assert "Also valid" in vtt


def test_vtt_formatter_fallback_handles_top_level_errors() -> None:
    """Test VttFormatter handles top-level exceptions gracefully.

    Covers lines 535-537.
    """
    # Create a result that will cause errors in the try block
    result = {"chunks": "not a list"}  # type: ignore[dict-item]

    vtt = VttFormatter.format(result)
    # Should handle the error gracefully and return at least WEBVTT header
    assert "WEBVTT" in vtt


def test_json_formatter_handles_non_serializable() -> None:
    """Test JsonFormatter handles non-JSON-serializable data.

    Covers lines 567-569.
    """

    # Create a result with a non-serializable object
    class NonSerializable:
        pass

    result = {"text": "test", "obj": NonSerializable()}

    json_str = JsonFormatter.format(result)
    assert json_str == "{}"


def test_json_formatter_get_file_extension() -> None:
    """Test JsonFormatter returns correct extension."""
    assert JsonFormatter.get_file_extension() == "json"


def test_vtt_formatter_get_file_extension() -> None:
    """Test VttFormatter returns correct extension."""
    assert VttFormatter.get_file_extension() == "vtt"


def test_srt_formatter_get_file_extension() -> None:
    """Test SrtFormatter returns correct extension."""
    assert SrtFormatter.get_file_extension() == "srt"


def test_build_quality_segments_with_timestamp_extraction() -> None:
    """Test build_quality_segments extracts timestamps from timestamp field.

    Covers lines 133-135.
    """
    result = {
        "chunks": [
            {
                "text": "Test",
                "timestamp": [1.0, 2.0],
                # No start/end, should use timestamp
            }
        ]
    }

    segments = build_quality_segments(result)
    assert len(segments) == 1
    assert segments[0]["start"] == 1.0
    assert segments[0]["end"] == 2.0


def test_build_quality_segments_skips_invalid_end_before_start() -> None:
    """Test build_quality_segments skips segments where end <= start.

    Covers lines 138-139.
    """
    result = {
        "chunks": [  # Use chunks instead to avoid word-level merging
            {"text": "Valid", "start": 0.0, "end": 1.0},
            {"text": "Bad order", "start": 3.0, "end": 2.0},  # end < start
            {"text": "Equal", "start": 5.0, "end": 5.0},  # end == start
        ]
    }

    segments = build_quality_segments(result)
    # Only the valid one should be included (bad timestamps are filtered)
    assert len(segments) >= 1
    # Verify the bad ones are not included
    for seg in segments:
        assert seg["end"] > seg["start"]


def test_srt_formatter_chunk_format_error_handling() -> None:
    """Test SrtFormatter handles individual chunk formatting errors.

    Covers lines 313-314.
    """
    result = {
        "chunks": [
            {"text": "Good", "start": 0.0, "end": 1.0},
            {"text": "Missing timestamps"},  # Will fail in formatting
            {"text": "Also good", "start": 2.0, "end": 3.0},
        ]
    }

    srt = SrtFormatter.format(result)
    # Should contain the good chunks
    assert "Good" in srt
    assert "Also good" in srt


def test_vtt_formatter_skips_invalid_timestamps() -> None:
    """Test VttFormatter skips chunks with invalid timestamps.

    Covers lines 520-521.
    """
    result = {
        "chunks": [
            {"text": "No timestamp"},  # Missing both start/end and timestamp
            {"text": "Valid", "start": 0.0, "end": 1.0},
        ]
    }

    vtt = VttFormatter.format(result)
    assert "Valid" in vtt
    assert "No timestamp" not in vtt


def test_vtt_formatter_chunk_error_handling() -> None:
    """Test VttFormatter handles chunk formatting errors.

    Covers lines 531-532.
    """
    result = {
        "chunks": [
            {"text": "Good", "start": 0.0, "end": 1.0},
            {"text": "Missing end"},  # Missing timestamps, will skip
            {"text": "Also good", "start": 2.0, "end": 3.0},
        ]
    }

    vtt = VttFormatter.format(result)
    assert "Good" in vtt
    assert "Also good" in vtt
    # Missing end should be skipped, not causing complete failure
