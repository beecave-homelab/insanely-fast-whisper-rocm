"""Formatter classes for Insanely Fast Whisper API WebUI.

This module contains classes for formatting transcription results
in different output formats (text, SRT subtitles, JSON).
"""

import json
from typing import Any, Dict

from insanely_fast_whisper_api.webui.utils import format_seconds as util_format_seconds


class BaseFormatter:
    """Base class for all formatters."""

    @classmethod
    def format(cls, result: Dict[str, Any]) -> str:
        """Format the transcription result.

        Args:
            result: The transcription result from ASRPipeline

        Returns:
            Formatted string
        """
        raise NotImplementedError("Subclasses must implement this method")

    @classmethod
    def get_file_extension(cls) -> str:
        """Get the file extension for this format."""
        raise NotImplementedError("Subclasses must implement this method")


class TxtFormatter(BaseFormatter):
    """Formatter for plain text output."""

    @classmethod
    def format(cls, result: Dict[str, Any]) -> str:
        """Format as plain text."""
        return result.get("text", "")

    @classmethod
    def get_file_extension(cls) -> str:
        return "txt"


class SrtFormatter(BaseFormatter):
    """Formatter for SRT (SubRip) subtitles."""

    @classmethod
    def format(cls, result: Dict[str, Any]) -> str:
        """Format as SRT subtitles with timestamps."""
        chunks = result.get("chunks", [])
        if not chunks:
            return ""

        srt_content = []
        for i, chunk in enumerate(chunks, 1):
            text = chunk.get("text", "").strip()
            if not text:
                continue

            timestamps = chunk.get("timestamp", [None, None])
            start, end = timestamps[0] if len(timestamps) > 0 else None, (
                timestamps[1] if len(timestamps) > 1 else None
            )

            srt_content.append(
                f"{i}\n"
                f"{util_format_seconds(start)} --> {util_format_seconds(end)}\n"
                f"{text}\n"
            )

        return "\n".join(srt_content)

    @classmethod
    def get_file_extension(cls) -> str:
        return "srt"


class JsonFormatter(BaseFormatter):
    """Formatter for JSON output."""

    @classmethod
    def format(cls, result: Dict[str, Any]) -> str:
        """Format as pretty-printed JSON."""
        return json.dumps(result, indent=2, ensure_ascii=False)

    @classmethod
    def get_file_extension(cls) -> str:
        return "json"


# Available formatters
FORMATTERS = {
    "txt": TxtFormatter,
    "srt": SrtFormatter,
    "json": JsonFormatter,
}
