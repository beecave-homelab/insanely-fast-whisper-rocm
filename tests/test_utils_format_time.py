"""Unit tests for time formatting utilities.

Covers ``format_srt_time``, ``format_vtt_time``, and legacy ``format_seconds``.
"""

from __future__ import annotations

from insanely_fast_whisper_rocm.utils.format_time import (
    format_seconds,
    format_srt_time,
    format_vtt_time,
)


def test_format_srt_time_basic() -> None:
    """SRT uses comma as the milliseconds separator."""
    assert format_srt_time(0.0) == "00:00:00,000"
    assert format_srt_time(1.234) == "00:00:01,234"
    assert format_srt_time(3661.007) == "01:01:01,007"


def test_format_vtt_time_basic() -> None:
    """VTT uses dot as the milliseconds separator."""
    assert format_vtt_time(0.0) == "00:00:00.000"
    assert format_vtt_time(1.234) == "00:00:01.234"
    assert format_vtt_time(3661.007) == "01:01:01.007"


def test_format_seconds_legacy_equivalence_to_vtt() -> None:
    """Legacy ``format_seconds`` matches ``format_vtt_time`` output."""
    for value in [0.0, 1.234, 3661.007, None]:
        assert format_seconds(value) == format_vtt_time(value)
