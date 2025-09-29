"""Formatter classes for Insanely Fast Whisper API.

This module contains classes for formatting transcription results
in different output formats (text, SRT subtitles, JSON).
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from insanely_fast_whisper_api.core.segmentation import Word, segment_words, split_lines
from insanely_fast_whisper_api.utils.constants import USE_READABLE_SUBTITLES
from insanely_fast_whisper_api.utils.format_time import format_srt_time, format_vtt_time

logger = logging.getLogger(__name__)


def _result_to_words(result: dict[str, Any]) -> list[Word] | None:
    """Extract `Word` objects from a transcription result if available.

    This helper checks for word-level timestamps and normalizes them into a
    list of `Word` objects. It supports both `chunks` and `segments` keys.

    Args:
        result: The transcription result from the ASR pipeline.

    Returns:
        A list of `Word` objects if word-level timestamps are found, otherwise
        ``None``.
    """
    words_list = []
    # Prioritize 'chunks' if it looks like word-level data
    chunks = result.get("chunks")
    if isinstance(chunks, list) and chunks and "timestamp" in chunks[0]:
        for chunk in chunks:
            text = chunk.get("text", "").strip()
            timestamp = chunk.get("timestamp")
            if text and isinstance(timestamp, (list, tuple)) and len(timestamp) == 2:
                start, end = timestamp
                if isinstance(start, (int, float)) and isinstance(end, (int, float)):
                    words_list.append(Word(text=text, start=start, end=end))

    if words_list:
        # Heuristic: if the average word duration is very short, it's likely word-level
        avg_duration = sum(w.end - w.start for w in words_list) / len(words_list)
        if avg_duration < 1.5:  # Words are typically short
            return words_list

    # Fallback to 'segments' if they contain word-level data
    segments = result.get("segments")
    if isinstance(segments, list):
        for segment in segments:
            words = segment.get("words")
            if isinstance(words, list) and words and isinstance(words[0], dict):
                for word_data in words:
                    text = word_data.get("word", "").strip()
                    start = word_data.get("start")
                    end = word_data.get("end")
                    if (
                        text
                        and isinstance(start, (int, float))
                        and isinstance(end, (int, float))
                    ):
                        words_list.append(Word(text=text, start=start, end=end))

    return words_list if words_list else None


def build_quality_segments(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Build readability-aware segments for quality scoring.

    Args:
        result: Transcription result containing raw word-level timestamps.

    Returns:
        Segments with ``start``, ``end``, and ``text`` keys suitable for
        `compute_srt_quality`. When word-level timestamps are available the
        segments are synthesized via `segment_words` so they satisfy timing and
        character-per-second constraints; otherwise the original segments/chunks
        are returned.
    """
    words = _result_to_words(result)
    if words:
        quality_segments: list[dict[str, Any]] = []
        for seg in segment_words(words):
            quality_segments.append({
                "start": float(seg.start),
                "end": float(seg.end),
                # Preserve line breaks for downstream line-length evaluation
                "text": seg.text.strip(),
            })
        if quality_segments:
            return quality_segments

    fallback_segments: list[dict[str, Any]] = []
    for chunk in result.get("segments") or result.get("chunks") or []:
        text = str(chunk.get("text", "")).strip()
        if not text:
            continue
        start = chunk.get("start")
        end = chunk.get("end")
        timestamp = chunk.get("timestamp")
        if (start is None or end is None) and isinstance(timestamp, (list, tuple)):
            if len(timestamp) == 2:
                start, end = timestamp
        if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
            continue
        if end <= start:
            continue
        fallback_segments.append({
            "start": float(start),
            "end": float(end),
            "text": text,
        })

    return fallback_segments


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

    _HYPHEN_SPACING_PATTERN = re.compile(r"(?<=\w)\s*-\s*(?=\w)")

    @classmethod
    def _normalize_hyphen_spacing(cls, text: str) -> str:
        """Collapse stray spaces around hyphens within words.

        Args:
            text: Raw caption text possibly containing spaced hyphens.

        Returns:
            Text with intra-word hyphen spacing normalized, e.g. 'co -pilot'
            becomes 'co-pilot'.
        """
        return cls._HYPHEN_SPACING_PATTERN.sub("-", text)

    @classmethod
    def format(cls, result: dict[str, Any]) -> str:
        """Format as SRT subtitles with timestamps.

        This method uses a segmentation pipeline to create readable subtitles
        if word-level timestamps are available. Otherwise, it falls back to
        formatting raw chunks.

        Args:
            result: The transcription result from ASRPipeline.

        Returns:
            The formatted SRT subtitles as a string.
        """
        logger.debug(f"[SrtFormatter] Formatting result: keys={list(result.keys())}")

        # Attempt to use the new segmentation pipeline first
        if USE_READABLE_SUBTITLES:
            words = _result_to_words(result)
            if words:
                logger.debug("[SrtFormatter] Found words, using segmentation.")
                segments = segment_words(words)
                srt_content = []
                for i, segment in enumerate(segments, 1):
                    start = format_srt_time(segment.start)
                    end = format_srt_time(segment.end)
                    wrapped = split_lines(segment.text)
                    normalized_text = cls._normalize_hyphen_spacing(wrapped)
                    srt_content.append(f"{i}\n{start} --> {end}\n{normalized_text}\n")
                return "\n".join(srt_content)

        # Fallback to old chunk-based formatting if no words are found
        logger.debug("[SrtFormatter] No word-level timestamps, using chunk fallback.")
        try:
            # Handle both 'chunks' from whisper and 'segments' from stable-ts
            chunks = result.get("chunks") or result.get("segments", [])
            if not chunks:
                logger.warning(
                    "[SrtFormatter] No 'chunks' or 'segments' found in result."
                )
                return ""

            # Apply timestamp validation to clean up overlapping segments
            from insanely_fast_whisper_api.utils.timestamp_utils import (
                validate_timestamps,
            )

            chunks = validate_timestamps(chunks)

            srt_content = []
            for i, chunk in enumerate(chunks, 1):
                try:
                    # Adapt to different timestamp formats
                    start_sec, end_sec = -1, -1
                    if (
                        "timestamp" in chunk
                        and isinstance(chunk["timestamp"], (list, tuple))
                        and len(chunk["timestamp"]) == 2
                    ):
                        start_sec, end_sec = chunk["timestamp"]
                    elif "start" in chunk and "end" in chunk:
                        start_sec, end_sec = chunk["start"], chunk["end"]

                    if start_sec == -1 or end_sec == -1:
                        continue

                    start = format_srt_time(start_sec)
                    end = format_srt_time(end_sec)
                    text = chunk.get("text", "").strip()

                    # Apply line splitting for readability
                    formatted_text = split_lines(text)
                    formatted_text = cls._normalize_hyphen_spacing(formatted_text)

                    srt_content.append(f"{i}\n{start} --> {end}\n{formatted_text}\n")
                except (TypeError, KeyError, AttributeError, IndexError) as chunk_e:
                    logger.error(
                        f"[SrtFormatter] Failed to format chunk #{i}: {chunk_e}"
                    )
            return "\n".join(srt_content)

        except (TypeError, KeyError, AttributeError, IndexError) as e:
            logger.exception(f"[SrtFormatter] Failed to format SRT: {e}")
            return ""

        # Fallback to old chunk-based formatting if no words are found
        logger.debug("[SrtFormatter] No word-level timestamps, using chunk fallback.")
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

            # Apply timestamp validation to clean up overlapping segments
            from insanely_fast_whisper_api.utils.timestamp_utils import (
                validate_timestamps,
            )

            chunks = validate_timestamps(chunks)

            def _get_start_end(c: dict) -> tuple[float | None, float | None]:
                ts_val = c.get("timestamp")
                ts_pair = ts_val if isinstance(ts_val, (list, tuple)) else None
                s = c.get("start")
                e = c.get("end")
                if (s is None or e is None) and ts_pair and len(ts_pair) == 2:
                    s, e = ts_pair[0], ts_pair[1]
                return s, e

            # Heuristic: detect "word-like" segments (very short duration, tiny text)
            durations = []
            short_count = 0
            for c in chunks:
                s, e = _get_start_end(c)
                if isinstance(s, (int, float)) and isinstance(e, (int, float)):
                    d = max(0.0, e - s)
                    durations.append(d)
                    if d <= 0.6 and len(c.get("text", "").strip().split()) <= 2:
                        short_count += 1
            word_like_ratio = short_count / max(1, len(chunks))

            # If more than 60% look like word-level items, group into readable captions
            if word_like_ratio >= 0.6:
                grouped: list[dict] = []
                cur: dict | None = None

                def flush_current() -> None:
                    nonlocal cur
                    if cur is not None:
                        grouped.append(cur)
                        cur = None

                for c in chunks:
                    s, e = _get_start_end(c)
                    if s is None or e is None:
                        continue
                    t = c.get("text", "").strip()
                    if not t:
                        continue
                    if cur is None:
                        cur = {"start": s, "end": e, "text": t}
                        continue
                    # If gap is big or line too long, flush
                    gap = s - float(cur["end"])  # type: ignore[arg-type]
                    new_text = (cur["text"] + " " + t).strip()  # type: ignore[index]
                    duration = float(e) - float(cur["start"])  # type: ignore[arg-type]
                    if gap > 0.6 or len(new_text) > 42 or duration > 3.5:
                        flush_current()
                        cur = {"start": s, "end": e, "text": t}
                    else:
                        cur["end"] = e  # type: ignore[index]
                        cur["text"] = new_text  # type: ignore[index]
                        # Flush at sentence-ending punctuation
                        if new_text.endswith((".", "!", "?", ":", ";")):
                            flush_current()
                            cur = None
                flush_current()
                chunks = grouped

            srt_content = []
            for i, chunk in enumerate(chunks, 1):
                try:
                    start_sec, end_sec = _get_start_end(chunk)
                    if start_sec is None or end_sec is None:
                        logger.warning(
                            "[Formatter] Skipping chunk #%d with missing timestamp",
                            i,
                        )
                        continue
                    start = format_srt_time(start_sec)
                    end = format_srt_time(end_sec)
                    text = chunk.get("text", "").replace("\n", " ").strip()
                    text = cls._normalize_hyphen_spacing(text)
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

        This method uses a segmentation pipeline to create readable subtitles
        if word-level timestamps are available. Otherwise, it falls back to
        formatting raw chunks.

        Args:
            result: The transcription result from ASRPipeline.

        Returns:
            The formatted WebVTT subtitles as a string.
        """
        logger.debug(f"[VttFormatter] Formatting result: keys={list(result.keys())}")

        # Attempt to use the new segmentation pipeline first
        if USE_READABLE_SUBTITLES:
            words = _result_to_words(result)
            if words:
                logger.debug("[VttFormatter] Found words, using segmentation.")
                segments = segment_words(words)
                vtt_content = ["WEBVTT\n"]
                for segment in segments:
                    start = format_vtt_time(segment.start)
                    end = format_vtt_time(segment.end)
                    vtt_content.append(f"{start} --> {end}\n{segment.text}\n")
                return "\n".join(vtt_content)

        # Fallback to old chunk-based formatting if no words are found
        logger.debug("[VttFormatter] No word-level timestamps, using chunk fallback.")
        try:
            chunks = result.get("chunks") or result.get("segments", [])
            if not chunks:
                logger.warning(
                    "[VttFormatter] No 'chunks' or 'segments' found in result."
                )
                return "WEBVTT\n\n"

            # Apply timestamp validation to clean up overlapping segments
            from insanely_fast_whisper_api.utils.timestamp_utils import (
                validate_timestamps,
            )

            chunks = validate_timestamps(chunks)

            vtt_content = ["WEBVTT\n"]
            for chunk in chunks:
                try:
                    start_sec, end_sec = -1, -1
                    if (
                        "timestamp" in chunk
                        and isinstance(chunk["timestamp"], (list, tuple))
                        and len(chunk["timestamp"]) == 2
                    ):
                        start_sec, end_sec = chunk["timestamp"]
                    elif "start" in chunk and "end" in chunk:
                        start_sec, end_sec = chunk["start"], chunk["end"]

                    if start_sec == -1 or end_sec == -1:
                        continue

                    start = format_vtt_time(start_sec)
                    end = format_vtt_time(end_sec)
                    text = chunk.get("text", "").strip()

                    # Apply line splitting for readability
                    formatted_text = split_lines(text)

                    vtt_content.append(f"{start} --> {end}\n{formatted_text}\n")
                except (TypeError, KeyError, AttributeError, IndexError) as chunk_e:
                    logger.error(f"[VttFormatter] Failed to format chunk: {chunk_e}")
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
