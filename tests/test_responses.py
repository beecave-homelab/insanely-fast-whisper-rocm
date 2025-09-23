"""Tests for API response formatting."""

from __future__ import annotations

from fastapi.responses import JSONResponse, PlainTextResponse

from insanely_fast_whisper_api.api.responses import ResponseFormatter
from insanely_fast_whisper_api.utils import (
    RESPONSE_FORMAT_JSON,
    RESPONSE_FORMAT_SRT,
    RESPONSE_FORMAT_TEXT,
    RESPONSE_FORMAT_VERBOSE_JSON,
    RESPONSE_FORMAT_VTT,
)


def test_seconds_to_timestamp_srt_format() -> None:
    """Test _seconds_to_timestamp method with SRT format."""
    # Test basic conversion
    result = ResponseFormatter._seconds_to_timestamp(3661.5)  # 1:01:01,500
    assert result == "01:01:01,500"

    # Test zero seconds
    result = ResponseFormatter._seconds_to_timestamp(0.0)
    assert result == "00:00:00,000"

    # Test exact seconds
    result = ResponseFormatter._seconds_to_timestamp(65.0)
    assert result == "00:01:05,000"

    # Test milliseconds
    result = ResponseFormatter._seconds_to_timestamp(1.123)
    assert result == "00:00:01,123"


def test_seconds_to_timestamp_vtt_format() -> None:
    """Test _seconds_to_timestamp method with VTT format."""
    # Test VTT format (uses dot separator)
    result = ResponseFormatter._seconds_to_timestamp(3661.5, for_vtt=True)
    assert result == "01:01:01.500"

    # Test VTT with milliseconds
    result = ResponseFormatter._seconds_to_timestamp(1.123, for_vtt=True)
    assert result == "00:00:01.123"


def test_segments_to_srt() -> None:
    """Test _segments_to_srt method."""
    segments = [
        {"start": 0.0, "end": 2.5, "text": "Hello world"},
        {"start": 3.0, "end": 5.5, "text": "This is a test"},
    ]

    result = ResponseFormatter._segments_to_srt(segments)

    expected_lines = [
        "1",
        "00:00:00,000 --> 00:00:02,500",
        "Hello world",
        "",
        "2",
        "00:00:03,000 --> 00:00:05,500",
        "This is a test",
        "",
    ]
    expected = "\n".join(expected_lines)

    assert result == expected


def test_segments_to_srt_empty_segments() -> None:
    """Test _segments_to_srt with empty segments."""
    result = ResponseFormatter._segments_to_srt([])
    assert result == ""


def test_segments_to_srt_missing_keys() -> None:
    """Test _segments_to_srt with missing keys."""
    segments = [
        {"text": "Hello world"},  # Missing start/end
        {"start": 3.0, "text": "Test"},  # Missing end
    ]

    result = ResponseFormatter._segments_to_srt(segments)

    expected_lines = [
        "1",
        "00:00:00,000 --> 00:00:00,000",  # Default values
        "Hello world",
        "",
        "2",
        "00:00:03,000 --> 00:00:00,000",  # Default end
        "Test",
        "",
    ]
    expected = "\n".join(expected_lines)

    assert result == expected


def test_segments_to_vtt() -> None:
    """Test _segments_to_vtt method."""
    segments = [
        {"start": 0.0, "end": 2.5, "text": "Hello world"},
        {"start": 3.0, "end": 5.5, "text": "This is a test"},
    ]

    result = ResponseFormatter._segments_to_vtt(segments)

    expected_lines = [
        "WEBVTT",
        "",
        "00:00:00.000 --> 00:00:02.500",
        "Hello world",
        "",
        "00:00:03.000 --> 00:00:05.500",
        "This is a test",
        "",
    ]
    expected = "\n".join(expected_lines)

    assert result == expected


def test_segments_to_vtt_empty_segments() -> None:
    """Test _segments_to_vtt with empty segments."""
    result = ResponseFormatter._segments_to_vtt([])
    assert result == "WEBVTT"


def test_format_transcription_text_response() -> None:
    """Test format_transcription with text response format."""
    result = {"text": "Hello world"}

    response = ResponseFormatter.format_transcription(result, RESPONSE_FORMAT_TEXT)

    assert isinstance(response, PlainTextResponse)
    assert response.body == b"Hello world"
    assert response.media_type == "text/plain; charset=utf-8"


def test_format_transcription_json_response() -> None:
    """Test format_transcription with json response format."""
    result = {"text": "Hello world"}

    response = ResponseFormatter.format_transcription(result, RESPONSE_FORMAT_JSON)

    assert isinstance(response, JSONResponse)
    assert response.body == b'{"text":"Hello world"}'


def test_format_transcription_verbose_json_response() -> None:
    """Test format_transcription with verbose_json response format."""
    result = {
        "text": "Hello world",
        "chunks": [
            {
                "id": 0,
                "start": 0.0,
                "end": 2.5,
                "text": "Hello world",
                "tokens": [1, 2, 3],
                "temperature": 0.5,
                "avg_logprob": -0.3,
                "compression_ratio": 1.2,
                "no_speech_prob": 0.1,
            }
        ],
    }

    response = ResponseFormatter.format_transcription(result, RESPONSE_FORMAT_VERBOSE_JSON)

    assert isinstance(response, JSONResponse)

    content = response.body.decode()
    assert '"text":"Hello world"' in content
    assert '"segments":' in content
    assert '"id":0' in content
    assert '"tokens":[1,2,3]' in content


def test_format_transcription_verbose_json_with_language() -> None:
    """Test format_transcription with verbose_json and language detection."""
    result = {
        "text": "Hello world",
        "chunks": [{"start": 0.0, "end": 2.5, "text": "Hello world"}],
        "language": "en",
    }

    response = ResponseFormatter.format_transcription(result, RESPONSE_FORMAT_VERBOSE_JSON)

    content = response.body.decode()
    assert '"language":"en"' in content


def test_format_transcription_verbose_json_with_config_language() -> None:
    """Test format_transcription with verbose_json and config_used language."""
    result = {
        "text": "Hello world",
        "chunks": [{"start": 0.0, "end": 2.5, "text": "Hello world"}],
        "config_used": {"language": "fr"},
    }

    response = ResponseFormatter.format_transcription(result, RESPONSE_FORMAT_VERBOSE_JSON)

    content = response.body.decode()
    assert '"language":"fr"' in content


def test_format_transcription_srt_response() -> None:
    """Test format_transcription with SRT response format."""
    # Mock the FORMATTERS to avoid external dependencies
    original_formatters = ResponseFormatter.__dict__.get("FORMATTERS", {})
    ResponseFormatter.FORMATTERS = {"srt": type("MockFormatter", (), {"format": lambda x: "1\n00:00:00,000 --> 00:00:02,500\nHello world\n\n"})()}

    try:
        result = {"text": "Hello world", "chunks": [{"start": 0.0, "end": 2.5, "text": "Hello world"}]}

        response = ResponseFormatter.format_transcription(result, RESPONSE_FORMAT_SRT)

        assert isinstance(response, PlainTextResponse)
        assert response.media_type == "text/srt"
        assert response.body == b"1\n00:00:00,000 --> 00:00:02,500\nHello world\n\n"
    finally:
        # Restore original FORMATTERS
        if hasattr(ResponseFormatter, 'FORMATTERS'):
            delattr(ResponseFormatter, 'FORMATTERS')
        for key, value in original_formatters.items():
            setattr(ResponseFormatter, key, value)


def test_format_transcription_vtt_response() -> None:
    """Test format_transcription with VTT response format."""
    # Mock the FORMATTERS to avoid external dependencies
    original_formatters = ResponseFormatter.__dict__.get("FORMATTERS", {})
    ResponseFormatter.FORMATTERS = {"vtt": type("MockFormatter", (), {"format": lambda x: "WEBVTT\n\n00:00:00.000 --> 00:00:02.500\nHello world\n\n"})()}

    try:
        result = {"text": "Hello world", "chunks": [{"start": 0.0, "end": 2.5, "text": "Hello world"}]}

        response = ResponseFormatter.format_transcription(result, RESPONSE_FORMAT_VTT)

        assert isinstance(response, PlainTextResponse)
        assert response.media_type == "text/vtt"
        assert response.body == b"WEBVTT\n\n00:00:00.000 --> 00:00:02.500\nHello world\n\n"
    finally:
        # Restore original FORMATTERS
        if hasattr(ResponseFormatter, 'FORMATTERS'):
            delattr(ResponseFormatter, 'FORMATTERS')
        for key, value in original_formatters.items():
            setattr(ResponseFormatter, key, value)


def test_format_transcription_unsupported_format() -> None:
    """Test format_transcription with unsupported response format."""
    result = {"text": "Hello world"}

    response = ResponseFormatter.format_transcription(result, "unsupported")

    assert isinstance(response, JSONResponse)
    assert response.status_code == 400
    content = response.body.decode()
    assert '"error":"Unsupported response_format"' in content


def test_format_translation_text_response() -> None:
    """Test format_translation with text response format."""
    result = {"transcription": {"text": "Hello world"}}

    response = ResponseFormatter.format_translation(result, RESPONSE_FORMAT_TEXT)

    assert isinstance(response, PlainTextResponse)
    assert response.body == b"Hello world"
    assert response.media_type == "text/plain; charset=utf-8"


def test_format_translation_json_response() -> None:
    """Test format_translation with json response format."""
    result = {"transcription": {"text": "Hello world"}}

    response = ResponseFormatter.format_translation(result, RESPONSE_FORMAT_JSON)

    assert isinstance(response, JSONResponse)
    assert response.body == b'{"text":"Hello world"}'


def test_format_translation_verbose_json_response() -> None:
    """Test format_translation with verbose_json response format."""
    result = {
        "transcription": {
            "text": "Hello world",
            "chunks": [
                {
                    "id": 0,
                    "start": 0.0,
                    "end": 2.5,
                    "text": "Hello world",
                    "tokens": [1, 2, 3],
                    "temperature": 0.5,
                    "avg_logprob": -0.3,
                    "compression_ratio": 1.2,
                    "no_speech_prob": 0.1,
                }
            ],
            "language": "en",
        }
    }

    response = ResponseFormatter.format_translation(result, RESPONSE_FORMAT_VERBOSE_JSON)

    assert isinstance(response, JSONResponse)

    content = response.body.decode()
    assert '"text":"Hello world"' in content
    assert '"segments":' in content
    assert '"language":"en"' in content


def test_format_translation_srt_response() -> None:
    """Test format_translation with SRT response format."""
    # Mock the FORMATTERS to avoid external dependencies
    original_formatters = ResponseFormatter.__dict__.get("FORMATTERS", {})
    ResponseFormatter.FORMATTERS = {"srt": type("MockFormatter", (), {"format": lambda x: "1\n00:00:00,000 --> 00:00:02,500\nHello world\n\n"})()}

    try:
        result = {"transcription": {"text": "Hello world", "chunks": [{"start": 0.0, "end": 2.5, "text": "Hello world"}]}}

        response = ResponseFormatter.format_translation(result, RESPONSE_FORMAT_SRT)

        assert isinstance(response, PlainTextResponse)
        assert response.media_type == "text/srt"
        assert response.body == b"1\n00:00:00,000 --> 00:00:02,500\nHello world\n\n"
    finally:
        # Restore original FORMATTERS
        if hasattr(ResponseFormatter, 'FORMATTERS'):
            delattr(ResponseFormatter, 'FORMATTERS')
        for key, value in original_formatters.items():
            setattr(ResponseFormatter, key, value)


def test_format_translation_vtt_response() -> None:
    """Test format_translation with VTT response format."""
    # Mock the FORMATTERS to avoid external dependencies
    original_formatters = ResponseFormatter.__dict__.get("FORMATTERS", {})
    ResponseFormatter.FORMATTERS = {"vtt": type("MockFormatter", (), {"format": lambda x: "WEBVTT\n\n00:00:00.000 --> 00:00:02.500\nHello world\n\n"})()}

    try:
        result = {"transcription": {"text": "Hello world", "chunks": [{"start": 0.0, "end": 2.5, "text": "Hello world"}]}}

        response = ResponseFormatter.format_translation(result, RESPONSE_FORMAT_VTT)

        assert isinstance(response, PlainTextResponse)
        assert response.media_type == "text/vtt"
        assert response.body == b"WEBVTT\n\n00:00:00.000 --> 00:00:02.500\nHello world\n\n"
    finally:
        # Restore original FORMATTERS
        if hasattr(ResponseFormatter, 'FORMATTERS'):
            delattr(ResponseFormatter, 'FORMATTERS')
        for key, value in original_formatters.items():
            setattr(ResponseFormatter, key, value)


def test_format_translation_unsupported_format() -> None:
    """Test format_translation with unsupported response format."""
    result = {"transcription": {"text": "Hello world"}}

    response = ResponseFormatter.format_translation(result, "unsupported")

    assert isinstance(response, JSONResponse)
    assert response.status_code == 400
    content = response.body.decode()
    assert '"error":"Unsupported response_format"' in content


def test_format_translation_fallback_result() -> None:
    """Test format_translation when result doesn't have transcription key."""
    result = {"text": "Hello world"}  # No transcription key

    response = ResponseFormatter.format_translation(result, RESPONSE_FORMAT_JSON)

    assert isinstance(response, JSONResponse)
    assert response.body == b'{"text":"Hello world"}'


def test_format_transcription_empty_result() -> None:
    """Test format_transcription with empty result."""
    result = {}

    # Test text format
    response = ResponseFormatter.format_transcription(result, RESPONSE_FORMAT_TEXT)
    assert isinstance(response, PlainTextResponse)
    assert response.body == b""

    # Test JSON format
    response = ResponseFormatter.format_transcription(result, RESPONSE_FORMAT_JSON)
    assert isinstance(response, JSONResponse)
    assert response.body == b'{"text":""}'
