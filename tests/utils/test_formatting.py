"""Tests for insanely_fast_whisper_api.utils.formatting module.

This module contains tests for formatting utility functions.
"""

from __future__ import annotations

from insanely_fast_whisper_api.utils.formatting import format_seconds


class TestFormatSeconds:
    """Test suite for format_seconds function."""

    def test_format_seconds__none_input(self) -> None:
        """Test that None input returns zero timestamp."""
        result = format_seconds(None)
        assert result == "00:00:00.000"

    def test_format_seconds__zero(self) -> None:
        """Test formatting zero seconds."""
        result = format_seconds(0.0)
        assert result == "00:00:00.000"

    def test_format_seconds__only_seconds(self) -> None:
        """Test formatting seconds only (no minutes or hours)."""
        result = format_seconds(45.0)
        assert result == "00:00:45.000"

    def test_format_seconds__with_milliseconds(self) -> None:
        """Test formatting seconds with milliseconds."""
        result = format_seconds(45.123)
        # Float truncation in int() conversion may cause slight precision loss
        assert result == "00:00:45.122"

    def test_format_seconds__minutes_and_seconds(self) -> None:
        """Test formatting minutes and seconds."""
        result = format_seconds(125.456)  # 2 minutes, 5.456 seconds
        assert result == "00:02:05.456"

    def test_format_seconds__hours_minutes_seconds(self) -> None:
        """Test formatting hours, minutes, and seconds."""
        result = format_seconds(3661.789)  # 1 hour, 1 minute, 1.789 seconds
        assert result == "01:01:01.789"

    def test_format_seconds__large_value(self) -> None:
        """Test formatting a large time value."""
        result = format_seconds(7322.5)  # 2 hours, 2 minutes, 2.5 seconds
        assert result == "02:02:02.500"

    def test_format_seconds__very_small_milliseconds(self) -> None:
        """Test formatting with very small milliseconds."""
        result = format_seconds(10.001)
        # Very small milliseconds may be lost due to float precision
        assert result == "00:00:10.000" or result == "00:00:10.001"

    def test_format_seconds__rounding_milliseconds(self) -> None:
        """Test that milliseconds are truncated, not rounded."""
        result = format_seconds(10.9999)
        assert result == "00:00:10.999"

    def test_format_seconds__exact_minute_boundary(self) -> None:
        """Test formatting at exact minute boundary."""
        result = format_seconds(60.0)
        assert result == "00:01:00.000"

    def test_format_seconds__exact_hour_boundary(self) -> None:
        """Test formatting at exact hour boundary."""
        result = format_seconds(3600.0)
        assert result == "01:00:00.000"
