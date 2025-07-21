"""Formatter classes for Insanely Fast Whisper API.

This module contains classes for formatting transcription results
in different output formats (text, SRT subtitles, JSON).
"""

import json
import logging
from typing import Any, Dict

from insanely_fast_whisper_api.utils.formatting import (
    format_seconds as util_format_seconds,
)

logger = logging.getLogger(__name__)


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
        logger.debug(f"[TxtFormatter] Formatting result: keys={list(result.keys())}")
        try:
            text = result.get("text", "")
            if not isinstance(text, str):
                logger.error("[TxtFormatter] 'text' field is not a string.")
                return ""
            return text
        except Exception as e:
            logger.exception(f"[TxtFormatter] Failed to format TXT: {e}")
            return ""

    @classmethod
    def get_file_extension(cls) -> str:
        return "txt"


class SrtFormatter(BaseFormatter):
    """Formatter for SRT (SubRip) subtitles."""

    @classmethod
    def format(cls, result: Dict[str, Any]) -> str:
        """Format as SRT subtitles with timestamps."""
        logger.debug(f"[SrtFormatter] Formatting result: keys={list(result.keys())}")
        try:
            chunks = result.get("segments") or result.get("chunks", [])
            if not chunks:
                logger.warning(
                    "[SrtFormatter] No 'segments' or 'chunks' found in result."
                )
                return ""

            srt_content = []
            for i, chunk in enumerate(chunks, 1):
                try:
                    if (
                        "start" not in chunk
                        or "end" not in chunk
                        or chunk["start"] is None
                        or chunk["end"] is None
                    ):
                        logger.warning(
                            "[Formatter] Skipping chunk #%d with missing timestamp", i
                        )
                        continue
                    start = util_format_seconds(chunk["start"])
                    end = util_format_seconds(chunk["end"])
                    text = chunk["text"].replace("\n", " ").strip()
                    srt_content.append(f"{i}\n{start} --> {end}\n{text}\n")
                except Exception as chunk_e:
                    logger.error(
                        f"[SrtFormatter] Failed to format chunk #{i}: {chunk_e}"
                    )
            return "\n".join(srt_content)
        except Exception as e:
            logger.exception(f"[SrtFormatter] Failed to format SRT: {e}")
            return ""

    @classmethod
    def get_file_extension(cls) -> str:
        return "srt"


class VttFormatter(BaseFormatter):
    """Formatter for WebVTT subtitles."""

    @classmethod
    def format(cls, result: Dict[str, Any]) -> str:
        """Format as WebVTT subtitles with timestamps."""
        logger.debug(f"[VttFormatter] Formatting result: keys={list(result.keys())}")
        try:
            chunks = result.get("segments") or result.get("chunks", [])
            if not chunks:
                logger.warning(
                    "[VttFormatter] No 'segments' or 'chunks' found in result."
                )
                return "WEBVTT\n\n"

            vtt_content = ["WEBVTT\n"]
            for i, chunk in enumerate(chunks, 1):
                try:
                    if (
                        "start" not in chunk
                        or "end" not in chunk
                        or chunk["start"] is None
                        or chunk["end"] is None
                    ):
                        logger.warning(
                            "[Formatter] Skipping chunk #%d with missing timestamp", i
                        )
                        continue
                    start = util_format_seconds(chunk["start"])
                    end = util_format_seconds(chunk["end"])
                    text = chunk["text"].replace("\n", " ").strip()
                    vtt_content.append(f"{start} --> {end}\n{text}\n")
                except Exception as chunk_e:
                    logger.error(
                        f"[VttFormatter] Failed to format chunk #{i}: {chunk_e}"
                    )
            return "\n".join(vtt_content)
        except Exception as e:
            logger.exception(f"[VttFormatter] Failed to format VTT: {e}")
            return "WEBVTT\n\n"

    @classmethod
    def get_file_extension(cls) -> str:
        return "vtt"


class JsonFormatter(BaseFormatter):
    """Formatter for JSON output."""

    @classmethod
    def format(cls, result: Dict[str, Any]) -> str:
        """Format as pretty-printed JSON."""
        logger.debug(f"[JsonFormatter] Formatting result: type={type(result)}")
        try:
            return json.dumps(result, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.exception(f"[JsonFormatter] Failed to format JSON: {e}")
            return "{}"

    @classmethod
    def get_file_extension(cls) -> str:
        return "json"


# Available formatters
FORMATTERS = {
    "txt": TxtFormatter,
    "srt": SrtFormatter,
    "vtt": VttFormatter,
    "json": JsonFormatter,
}
