"""Centralized timestamp utilities for Insanely Fast Whisper API.

This module provides centralized functions for timestamp validation, normalization,
and format conversion to avoid duplication across the codebase.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class TimestampError(Exception):
    """Exception raised for timestamp-related errors."""

    pass


def validate_timestamps(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Validate and correct timestamp issues in a list of segments.

    This function centralizes timestamp validation logic that was previously
    duplicated across multiple modules (stable_ts, segmentation, formatters).

    Args:
        segments: List of segment dictionaries with timestamp information.

    Returns:
        List of segments with corrected timestamps.
    """
    if not segments:
        return []

    # Sort segments by start time (handle None starts by pushing them last)
    segments.sort(
        key=lambda s: s.get("start") if s.get("start") is not None else float("inf")
    )

    cleaned, last_end = [], 0.0
    for seg in segments:
        start, end = seg.get("start"), seg.get("end")

        # Skip segments with None timestamps (preserve original behavior)
        if start is None or end is None:
            continue

        # Fix wrong order timestamps
        if start > end:
            start, end = end, start

        # Fix overlapping segments
        if start < last_end:
            duration = max(0.0, end - start)
            start = last_end
            end = last_end + duration

        # Only include segments that start after the previous one ends
        if start >= last_end:
            seg["start"], seg["end"] = start, end
            cleaned.append(seg)
            last_end = end

    return cleaned


def normalize_timestamp_format(data: dict[str, Any]) -> dict[str, Any]:
    """Normalize timestamp data to use consistent 'segments' format.

    This function centralizes the conversion between different timestamp
    formats (chunks vs segments, timestamp tuples vs start/end fields).

    Args:
        data: Dictionary containing transcription data.

    Returns:
        Dictionary with normalized timestamp format.
    """
    # Create a copy to avoid modifying the original
    result = data.copy()

    # If segments already exist, return as-is
    if "segments" in result:
        return result

    # Convert chunks to segments if present
    if "chunks" in result:
        chunks = result["chunks"]
        segments = []

        for chunk in chunks:
            # Handle chunks with timestamp tuples
            if "timestamp" in chunk and isinstance(chunk["timestamp"], (list, tuple)):
                start, end = chunk["timestamp"]
                if isinstance(start, (int, float)) and isinstance(end, (int, float)):
                    # Create segment format
                    segment = chunk.copy()
                    segment["start"] = start
                    segment["end"] = end
                    segment.pop("timestamp", None)
                    segments.append(segment)
            # Handle chunks that already have start/end fields
            elif "start" in chunk and "end" in chunk:
                start = chunk["start"]
                end = chunk["end"]
                if isinstance(start, (int, float)) and isinstance(end, (int, float)):
                    segments.append(chunk.copy())

        result["segments"] = segments
        # Remove chunks to avoid duplication
        result.pop("chunks", None)

    return result


def extract_timestamps(segment: dict[str, Any]) -> tuple[float, float] | None:
    """Extract start and end timestamps from a segment dictionary.

    This function centralizes timestamp extraction logic that was previously
    duplicated across formatters and other modules.

    Args:
        segment: Segment dictionary with timestamp information.

    Returns:
        Tuple of (start, end) timestamps, or None if not found.

    Raises:
        TimestampError: If timestamp data is invalid.
    """
    # Try timestamp tuple first
    if "timestamp" in segment:
        timestamp = segment["timestamp"]
        if isinstance(timestamp, (list, tuple)) and len(timestamp) == 2:
            start, end = timestamp
            if isinstance(start, (int, float)) and isinstance(end, (int, float)):
                return (start, end)

    # Try start/end fields
    if "start" in segment and "end" in segment:
        start = segment["start"]
        end = segment["end"]
        if isinstance(start, (int, float)) and isinstance(end, (int, float)):
            return (start, end)

    # No valid timestamps found
    raise TimestampError(f"No valid timestamps found in segment: {segment}")
