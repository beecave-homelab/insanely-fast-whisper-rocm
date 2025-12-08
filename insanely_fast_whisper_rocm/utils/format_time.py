"""Formatting utility functions for Insanely Fast Whisper API.

This module contains utility functions for formatting data, such as timestamps.
"""

from __future__ import annotations


def format_seconds(seconds: float | None) -> str:
    """Format seconds as ``HH:MM:SS.mmm`` using a dot separator.

    This function is kept for backward compatibility and for VTT-style
    timestamps. Prefer using the explicit helpers ``format_srt_time`` and
    ``format_vtt_time`` when you need a specific subtitle format.

    Args:
        seconds: Time in seconds (can be None).

    Returns:
        A dot-separated timestamp string.
    """
    if seconds is None:
        return "00:00:00.000"

    whole_seconds = int(seconds)
    milliseconds = int((seconds - whole_seconds) * 1000)
    hours = whole_seconds // 3600
    minutes = (whole_seconds % 3600) // 60
    seconds_val = whole_seconds % 60

    return f"{hours:02d}:{minutes:02d}:{seconds_val:02d}.{milliseconds:03d}"


def format_vtt_time(seconds: float | None) -> str:
    """Format seconds as WebVTT timestamp ``HH:MM:SS.mmm``.

    Args:
        seconds: Time in seconds (can be None).

    Returns:
        Dot-separated timestamp suitable for VTT.
    """
    return format_seconds(seconds)


def format_srt_time(seconds: float | None) -> str:
    """Format seconds as SRT timestamp ``HH:MM:SS,mmm``.

    Args:
        seconds: Time in seconds (can be None).

    Returns:
        Comma-separated timestamp suitable for SRT.
    """
    if seconds is None:
        return "00:00:00,000"

    whole_seconds = int(seconds)
    milliseconds = int((seconds - whole_seconds) * 1000)
    hours = whole_seconds // 3600
    minutes = (whole_seconds % 3600) // 60
    seconds_val = whole_seconds % 60

    return f"{hours:02d}:{minutes:02d}:{seconds_val:02d},{milliseconds:03d}"
