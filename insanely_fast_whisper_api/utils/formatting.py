"""Formatting utility functions for Insanely Fast Whisper API.

This module contains utility functions for formatting data, such as timestamps.
"""


def format_seconds(seconds: float | None) -> str:
    """Format seconds to a human-readable time string (HH:MM:SS.mmm).

    Args:
        seconds: Time in seconds (can be None).

    Returns:
        Formatted time string.
    """
    if seconds is None:
        return "00:00:00.000"

    whole_seconds = int(seconds)
    milliseconds = int((seconds - whole_seconds) * 1000)
    hours = whole_seconds // 3600
    minutes = (whole_seconds % 3600) // 60
    seconds_val = whole_seconds % 60

    return f"{hours:02d}:{minutes:02d}:{seconds_val:02d}.{milliseconds:03d}"
