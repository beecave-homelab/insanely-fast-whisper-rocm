"""Formatting utility functions for Insanely Fast Whisper API.

This module contains utility functions for formatting data, such as timestamps.
"""


def format_seconds(seconds: float | None) -> str:
    """Format a duration in seconds into an HH:MM:SS.mmm formatted string.

    Parameters:
        seconds (float | None): Duration in seconds; if `None`, returns the zero-time string "00:00:00.000".

    Returns:
        str: Time formatted as "HH:MM:SS.mmm" with hours, minutes, and seconds zero-padded and milliseconds shown with three digits.
    """
    if seconds is None:
        return "00:00:00.000"

    whole_seconds = int(seconds)
    milliseconds = int((seconds - whole_seconds) * 1000)
    hours = whole_seconds // 3600
    minutes = (whole_seconds % 3600) // 60
    seconds_val = whole_seconds % 60

    return f"{hours:02d}:{minutes:02d}:{seconds_val:02d}.{milliseconds:03d}"
