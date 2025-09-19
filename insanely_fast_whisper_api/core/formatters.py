"""Formatter classes for Insanely Fast Whisper API.

This module contains classes for formatting transcription results
in different output formats (text, SRT subtitles, JSON).
"""

import json
import logging
from typing import Any

from insanely_fast_whisper_api.utils.format_time import (
    format_seconds as util_format_seconds,
)

logger = logging.getLogger(__name__)


class BaseFormatter:
    """Base class for all formatters."""

    @classmethod
    def format(cls, result: dict[str, Any]) -> str:
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
    def format(cls, result: dict[str, Any]) -> str:
        """Format as plain text.

        Args:
            result: The transcription result from ASRPipeline

        Returns:
            str: The formatted text.

        """
        logger.debug(f"[TxtFormatter] Formatting result: keys={list(result.keys())}")
        try:
            text = result.get("text", "")
            if not isinstance(text, str):
                logger.error("[TxtFormatter] 'text' field is not a string.")
                return ""
            return text
        except (TypeError, KeyError, AttributeError) as e:
            logger.exception(f"[TxtFormatter] Failed to format TXT: {e}")
            return ""

    @classmethod
    def get_file_extension(cls) -> str:
        """Get the file extension for this format.

        Returns:
            str: The file extension for this format ("txt").

        """
        return "txt"


class SrtFormatter(BaseFormatter):
    """Formatter for SRT (SubRip) subtitles."""

    @classmethod
    def format(cls, result: dict[str, Any]) -> str:
        """Format as SRT subtitles with timestamps.

        Args:
            result: The transcription result from ASRPipeline

        Returns:
            str: The formatted SRT subtitles.

        Raises:
            TypeError: If the input is of incorrect type.
            KeyError: If required keys are missing from the result.
            AttributeError: If the result object does not support expected operations.
            IndexError: If accessing list elements fails.

        """
        logger.debug(f"[SrtFormatter] Formatting result: keys={list(result.keys())}")
        try:
            segments = result.get("segments", [])
            chunks_candidate = result.get("chunks", [])

            def _has_valid_timestamp(lst: list[dict]) -> bool:
                return any(
                    (
                        (c.get("start") is not None and c.get("end") is not None)
                        or (
                            isinstance(c.get("timestamp"), (list, tuple))
                            and len(c.get("timestamp")) == 2
                            and c.get("timestamp")[0] is not None
                            and c.get("timestamp")[1] is not None
                        )
                    )
                    for c in lst
                )

            # Select whichever list yields more valid entries
            segments_valid = [c for c in segments if _has_valid_timestamp([c])]
            chunks_valid = [c for c in chunks_candidate if _has_valid_timestamp([c])]
            chunks = (
                segments_valid
                if len(segments_valid) >= len(chunks_valid)
                else chunks_valid
            )
            if not chunks:
                logger.warning(
                    "[SrtFormatter] No 'segments' or 'chunks' found in result."
                )
                return ""

            srt_content = []
            for i, chunk in enumerate(chunks, 1):
                try:
                    # Support both {'start': ... , 'end': ...} and
                    # {'timestamp': [start, end]}
                    ts_pair = (
                        chunk.get("timestamp")
                        if isinstance(chunk.get("timestamp"), (list, tuple))
                        else None
                    )
                    start_sec = chunk.get("start")
                    end_sec = chunk.get("end")
                    if (
                        (start_sec is None or end_sec is None)
                        and ts_pair
                        and len(ts_pair) == 2
                    ):
                        start_sec, end_sec = ts_pair[0], ts_pair[1]
                    if start_sec is None or end_sec is None:
                        logger.warning(
                            "[Formatter] Skipping chunk #%d with missing timestamp", i
                        )
                        continue
                    start = util_format_seconds(start_sec)
                    end = util_format_seconds(end_sec)
                    text = chunk["text"].replace("\n", " ").strip()
                    srt_content.append(f"{i}\n{start} --> {end}\n{text}\n")
                except (TypeError, KeyError, AttributeError, IndexError) as chunk_e:
                    logger.error(
                        f"[SrtFormatter] Failed to format chunk #{i}: {chunk_e}"
                    )
            return "\n".join(srt_content)
        except (TypeError, KeyError, AttributeError, IndexError) as e:
            logger.exception(f"[SrtFormatter] Failed to format SRT: {e}")
            return ""

    @classmethod
    def get_file_extension(cls) -> str:
        """Get the file extension for this format.

        Returns:
            str: The file extension for this format ("srt").

        """
        return "srt"


class VttFormatter(BaseFormatter):
    """Formatter for WebVTT subtitles."""

    @classmethod
    def format(cls, result: dict[str, Any]) -> str:
        """Format as WebVTT subtitles with timestamps.

        Args:
            result: The transcription result from ASRPipeline

        Returns:
            str: The formatted WebVTT subtitles.

        Raises:
            TypeError: If the input is of incorrect type.
            KeyError: If required keys are missing from the result.
            AttributeError: If the result object does not support expected operations.
            IndexError: If accessing list elements fails.

        """
        logger.debug(f"[VttFormatter] Formatting result: keys={list(result.keys())}")
        try:
            segments = result.get("segments", [])
            chunks_candidate = result.get("chunks", [])

            def _has_valid_timestamp(lst: list[dict]) -> bool:
                return any(
                    (
                        (c.get("start") is not None and c.get("end") is not None)
                        or (
                            isinstance(c.get("timestamp"), (list, tuple))
                            and len(c.get("timestamp")) == 2
                            and c.get("timestamp")[0] is not None
                            and c.get("timestamp")[1] is not None
                        )
                    )
                    for c in lst
                )

            # Select whichever list yields more valid entries
            segments_valid = [c for c in segments if _has_valid_timestamp([c])]
            chunks_valid = [c for c in chunks_candidate if _has_valid_timestamp([c])]
            chunks = (
                segments_valid
                if len(segments_valid) >= len(chunks_valid)
                else chunks_valid
            )
            if not chunks:
                logger.warning(
                    "[VttFormatter] No 'segments' or 'chunks' found in result."
                )
                return "WEBVTT\n\n"

            vtt_content = ["WEBVTT\n"]
            for i, chunk in enumerate(chunks, 1):
                try:
                    # Support both {'start': ..., 'end': ...} and
                    # {'timestamp': [start, end]}
                    ts_pair = (
                        chunk.get("timestamp")
                        if isinstance(chunk.get("timestamp"), (list, tuple))
                        else None
                    )
                    start_sec = chunk.get("start")
                    end_sec = chunk.get("end")
                    if (
                        (start_sec is None or end_sec is None)
                        and ts_pair
                        and len(ts_pair) == 2
                    ):
                        start_sec, end_sec = ts_pair[0], ts_pair[1]
                    if start_sec is None or end_sec is None:
                        logger.warning(
                            "[Formatter] Skipping chunk #%d with missing timestamp", i
                        )
                        continue
                    start = util_format_seconds(start_sec)
                    end = util_format_seconds(end_sec)
                    text = chunk["text"].replace("\n", " ").strip()
                    vtt_content.append(f"{start} --> {end}\n{text}\n")
                except (TypeError, KeyError, AttributeError, IndexError) as chunk_e:
                    logger.error(
                        f"[VttFormatter] Failed to format chunk #{i}: {chunk_e}"
                    )
            return "\n".join(vtt_content)
        except (TypeError, KeyError, AttributeError, IndexError) as e:
            logger.exception(f"[VttFormatter] Failed to format VTT: {e}")
            return "WEBVTT\n\n"

    @classmethod
    def get_file_extension(cls) -> str:
        """Get the file extension for this format.

        Returns:
            str: The file extension for this format ("vtt")

        """
        return "vtt"


class JsonFormatter(BaseFormatter):
    """Formatter for JSON output."""

    @classmethod
    def format(cls, result: dict[str, Any]) -> str:
        """Format as pretty-printed JSON.

        Args:
            result: The result to format.

        Returns:
            str: The formatted JSON string.

        Raises:
            TypeError: If the result contains non-serializable objects.
            ValueError: If the result cannot be serialized to JSON.

        """
        logger.debug(f"[JsonFormatter] Formatting result: type={type(result)}")
        try:
            return json.dumps(result, ensure_ascii=False, indent=2)
        except (TypeError, ValueError) as e:
            logger.exception(f"[JsonFormatter] Failed to format JSON: {e}")
            return "{}"

    @classmethod
    def get_file_extension(cls) -> str:
        """Get the file extension for this formatter.

        Returns:
            str: The file extension for this formatter ("json")

        """
        return "json"


# Available formatters
FORMATTERS = {
    "txt": TxtFormatter,
    "srt": SrtFormatter,
    "vtt": VttFormatter,
    "json": JsonFormatter,
}
