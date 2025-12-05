"""Tests for centralized timestamp utilities."""

import pytest

from insanely_fast_whisper_api.utils.timestamp_utils import (
    TimestampError,
    extract_timestamps,
    normalize_timestamp_format,
    validate_timestamps,
)


class TestTimestampValidation:
    """Test centralized timestamp validation functionality."""

    def test_validate_timestamps_basic(self) -> None:
        """Test basic timestamp validation with valid data."""
        segments = [
            {"text": "Hello", "start": 0.0, "end": 1.0},
            {"text": "World", "start": 1.0, "end": 2.0},
        ]
        result = validate_timestamps(segments)
        assert len(result) == 2
        assert result[0]["start"] == 0.0
        assert result[0]["end"] == 1.0

    def test_validate_timestamps_wrong_order(self) -> None:
        """Test timestamp validation with wrong order timestamps."""
        segments = [
            {"text": "World", "start": 2.0, "end": 1.0},  # Wrong order
            {"text": "Hello", "start": 0.0, "end": 1.0},
        ]
        result = validate_timestamps(segments)
        # Should fix the wrong order
        assert result[0]["start"] == 0.0
        assert result[0]["end"] == 1.0
        assert result[1]["start"] == 1.0
        assert result[1]["end"] == 2.0

    def test_validate_timestamps_overlapping(self) -> None:
        """Test timestamp validation with overlapping segments."""
        segments = [
            {"text": "Hello", "start": 0.0, "end": 2.0},
            {"text": "World", "start": 1.0, "end": 3.0},  # Overlaps
        ]
        result = validate_timestamps(segments)
        # Should fix overlapping by adjusting end times
        assert len(result) == 2
        assert result[0]["end"] <= result[1]["start"]

    def test_validate_timestamps_none_values(self) -> None:
        """Test timestamp validation with None values."""
        segments = [
            {"text": "Hello", "start": 0.0, "end": 1.0},
            {"text": "World", "start": None, "end": 2.0},  # None start
            {"text": "Test", "start": 2.0, "end": None},  # None end
        ]
        result = validate_timestamps(segments)
        # Should filter out segments with None timestamps
        assert len(result) == 1
        assert result[0]["text"] == "Hello"


class TestTimestampFormatConversion:
    """Test timestamp format conversion functionality."""

    def test_normalize_timestamp_format_chunks_to_segments(self) -> None:
        """Test conversion from chunks format to segments format."""
        data = {
            "text": "Hello world",
            "chunks": [
                {"text": "Hello", "timestamp": [0.0, 1.0]},
                {"text": "World", "timestamp": [1.0, 2.0]},
            ],
        }
        result = normalize_timestamp_format(data)
        assert "segments" in result
        assert "chunks" not in result
        assert len(result["segments"]) == 2
        assert result["segments"][0]["start"] == 0.0
        assert result["segments"][0]["end"] == 1.0

    def test_normalize_timestamp_format_already_segments(self) -> None:
        """Test that segments format is preserved when already present."""
        data = {
            "text": "Hello world",
            "segments": [
                {"text": "Hello", "start": 0.0, "end": 1.0},
                {"text": "World", "start": 1.0, "end": 2.0},
            ],
        }
        result = normalize_timestamp_format(data)
        assert result == data  # Should be unchanged

    def test_extract_timestamps_from_tuple(self) -> None:
        """Test extracting timestamps from tuple format."""
        segment = {"text": "Hello", "timestamp": [0.0, 1.0]}
        result = extract_timestamps(segment)
        assert result == (0.0, 1.0)

    def test_extract_timestamps_from_fields(self) -> None:
        """Test extracting timestamps from start/end fields."""
        segment = {"text": "Hello", "start": 0.0, "end": 1.0}
        result = extract_timestamps(segment)
        assert result == (0.0, 1.0)

    def test_extract_timestamps_invalid(self) -> None:
        """Test extracting timestamps from invalid data."""
        segment = {"text": "Hello"}  # No timestamp info
        with pytest.raises(TimestampError):
            extract_timestamps(segment)


class TestTimestampErrorHandling:
    """Test standardized timestamp error handling."""

    def test_timestamp_error_exception(self) -> None:
        """Test that TimestampError is properly raised.

        Raises:
            TimestampError: Always raised for testing purposes.
        """
        with pytest.raises(TimestampError):
            raise TimestampError("Test error message")
