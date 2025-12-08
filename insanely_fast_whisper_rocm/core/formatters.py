"""Formatter classes for Insanely Fast Whisper API.

This module contains classes for formatting transcription results
in different output formats (text, SRT subtitles, JSON).
"""

from __future__ import annotations

import json
import logging
import math
import re
from typing import Any

from insanely_fast_whisper_rocm.core.segmentation import (
    Word,
    segment_words,
    split_lines,
)
from insanely_fast_whisper_rocm.utils import constants
from insanely_fast_whisper_rocm.utils.constants import USE_READABLE_SUBTITLES
from insanely_fast_whisper_rocm.utils.format_time import (
    format_srt_time,
    format_vtt_time,
)

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
    logger.debug(
        "_result_to_words: examining result with chunks=%s, segments=%s",
        "chunks" in result,
        "segments" in result,
    )
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
        logger.debug(
            "Found %d potential words from chunks, avg_duration=%.3fs",
            len(words_list),
            avg_duration,
        )
        if avg_duration < 1.5:  # Words are typically short
            logger.debug("Returning %d words (avg_duration < 1.5s)", len(words_list))
            return words_list
        else:
            # Reject as sentence-level data; clear the list to avoid returning it later
            logger.debug(
                "Rejecting chunks as sentence-level (avg_duration=%.3fs >= 1.5s)",
                avg_duration,
            )
            words_list = []

    # Fallback to 'segments' if they contain word-level data
    segments = result.get("segments")
    if isinstance(segments, list) and segments:
        # Check if segments have a 'words' field (nested word-level data)
        first_segment = segments[0]
        if "words" in first_segment:
            # Nested structure: segments contain words arrays
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
        elif "start" in first_segment and "end" in first_segment:
            # Flat structure: each segment IS a word (from stable-ts)
            # Check if these look like word-level segments (short duration)
            for segment in segments:
                text = segment.get("text", "").strip()
                start = segment.get("start")
                end = segment.get("end")
                if (
                    text
                    and isinstance(start, (int, float))
                    and isinstance(end, (int, float))
                ):
                    words_list.append(Word(text=text, start=start, end=end))

            # Only return if average duration suggests word-level data
            if words_list:
                total_duration = sum(w.end - w.start for w in words_list)
                avg_duration = total_duration / len(words_list)
                logger.debug(
                    "Flat segment structure: %d items, avg_duration=%.3fs",
                    len(words_list),
                    avg_duration,
                )
                if avg_duration < 1.5:  # Words are typically short
                    logger.debug(
                        "Returning %d words from segments (avg_duration < 1.5s)",
                        len(words_list),
                    )
                    return words_list
                else:
                    # These are sentence-level segments, not words
                    logger.debug(
                        "Rejecting as sentence-level (avg_duration=%.3fs >= 1.5s)",
                        avg_duration,
                    )
                    return None

    result_count = len(words_list) if words_list else 0
    logger.debug(
        "_result_to_words returning: %s words",
        result_count if result_count else "None",
    )
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
    logger.info("[build_quality_segments] Processing result for quality scoring")
    words = _result_to_words(result)
    if words:
        word_span = words[-1].end - words[0].start if words else 0
        avg_word_dur = sum(w.end - w.start for w in words) / len(words)
        logger.info(
            "[build_quality_segments] Found %d words (span=%.1fs, avg_dur=%.3fs)",
            len(words),
            word_span,
            avg_word_dur,
        )
        quality_segments: list[dict[str, Any]] = []
        for seg in segment_words(words):
            quality_segments.append({
                "start": float(seg.start),
                "end": float(seg.end),
                # Preserve line breaks for downstream line-length evaluation
                "text": seg.text.strip(),
            })
        if quality_segments:
            max_dur = max(seg["end"] - seg["start"] for seg in quality_segments)
            # Find the longest segment for debugging
            longest_seg = max(quality_segments, key=lambda s: s["end"] - s["start"])
            logger.info(
                "[build_quality_segments] Returning %d segments, max_duration=%.2fs",
                len(quality_segments),
                max_dur,
            )
            if max_dur > 10:
                logger.warning(
                    "[build_quality_segments] Long segment: %.2fs, text=%r",
                    max_dur,
                    longest_seg.get("text", "")[:100],
                )
            return quality_segments

    logger.info(
        "[build_quality_segments] No words found, using fallback from raw "
        "chunks/segments"
    )
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
                logger.info(
                    "[SrtFormatter] Found %d words, using segmentation pipeline.",
                    len(words),
                )
                # CRITICAL FIX: For word-level timestamps:
                # Check if they appear corrupted (e.g., all timestamps
                # are the same value indicating a backend bug)
                word_timestamps = [w.start for w in words[:10]]  # Check first 10 words
                has_timing_bug = len(set(word_timestamps)) <= 1  # All same timestamp

                if has_timing_bug:
                    logger.warning(
                        "[SrtFormatter] Detected word-level timestamp bug "
                        "(all words have same timestamp). "
                        "Using fallback chunk-based formatting to avoid gaps."
                    )
                else:
                    segments = segment_words(words)
                    logger.info(
                        "[SrtFormatter] segment_words() produced %d segments "
                        "from %d words.",
                        len(segments),
                        len(words),
                    )
                    srt_content = []
                    for i, segment in enumerate(segments, 1):
                        start = format_srt_time(segment.start)
                        end = format_srt_time(segment.end)
                        wrapped = split_lines(segment.text)
                        normalized_text = cls._normalize_hyphen_spacing(wrapped)
                        srt_content.append(
                            f"{i}\n{start} --> {end}\n{normalized_text}\n"
                        )
                    logger.info(
                        "[SrtFormatter] Returning %d SRT segments.", len(srt_content)
                    )
                    return "\n".join(srt_content)
            else:
                logger.info("[SrtFormatter] No words found, using fallback.")

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

            # Normalize timestamp format (convert "timestamp" tuples to "start"/"end")
            normalized_chunks: list[dict[str, Any]] = []
            for chunk in chunks:
                # Skip non-dict chunks (error injection edge case)
                if not isinstance(chunk, dict):
                    continue
                normalized = dict(chunk)
                has_start = isinstance(normalized.get("start"), (int, float))
                has_end = isinstance(normalized.get("end"), (int, float))
                timestamp = normalized.get("timestamp")
                if (
                    (not has_start or not has_end)
                    and isinstance(timestamp, (list, tuple))
                    and len(timestamp) == 2
                ):
                    start_val, end_val = timestamp
                    if isinstance(start_val, (int, float)) and isinstance(
                        end_val, (int, float)
                    ):
                        normalized["start"] = start_val
                        normalized["end"] = end_val
                normalized_chunks.append(normalized)

            # Apply timestamp validation to clean up overlapping segments
            from insanely_fast_whisper_rocm.utils.timestamp_utils import (
                validate_timestamps,
            )

            chunks = validate_timestamps(normalized_chunks)
            chunks = cls._split_chunks_by_duration(chunks)

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

    @classmethod
    def get_file_extension(cls) -> str:
        """Get the file extension for this format.

        Returns:
            str: The file extension for this format ("srt").

        """
        return "srt"

    @staticmethod
    def _split_chunks_by_duration(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Split chunks so each duration respects ``MAX_SEGMENT_DURATION_SEC``.

        Args:
            chunks: Normalized chunk dictionaries containing ``text`` and timing
                information.

        Returns:
            Chunks with durations capped at ``constants.MAX_SEGMENT_DURATION_SEC``.
        """
        if not chunks:
            return []

        max_duration = constants.MAX_SEGMENT_DURATION_SEC
        epsilon = 1e-6
        split: list[dict[str, Any]] = []

        for chunk in chunks:
            text = str(chunk.get("text", "")).strip()
            if not text:
                split.append(chunk)
                continue

            start = chunk.get("start")
            end = chunk.get("end")
            timestamp = chunk.get("timestamp")
            if (start is None or end is None) and isinstance(timestamp, (list, tuple)):
                if len(timestamp) == 2:
                    start, end = timestamp

            if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
                split.append(chunk)
                continue

            duration = float(end) - float(start)
            if duration <= max_duration + epsilon:
                split.append(chunk)
                continue

            words = text.split()
            if not words:
                split.append(chunk)
                continue

            segment_count = max(1, math.ceil(duration / max_duration))
            segment_count = min(segment_count, len(words))
            # Ensure monotonic coverage of tokens.
            for index in range(segment_count):
                token_start = math.floor(index * len(words) / segment_count)
                token_end = math.floor((index + 1) * len(words) / segment_count)
                if token_start == token_end:
                    token_end = min(token_start + 1, len(words))
                token_slice = words[token_start:token_end]
                if not token_slice:
                    continue

                sub_start = float(start) + (duration * index) / segment_count
                sub_end = float(start) + (duration * (index + 1)) / segment_count
                if sub_end > float(end):
                    sub_end = float(end)

                new_chunk = dict(chunk)
                new_chunk["text"] = " ".join(token_slice)
                new_chunk["start"] = sub_start
                new_chunk["end"] = sub_end
                has_timestamp = isinstance(new_chunk.get("timestamp"), (list, tuple))
                if has_timestamp:
                    new_chunk["timestamp"] = [sub_start, sub_end]
                split.append(new_chunk)

        return split


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

            # Normalize timestamp format (convert "timestamp" tuples to "start"/"end")
            normalized_chunks: list[dict[str, Any]] = []
            for chunk in chunks:
                # Skip non-dict chunks (error injection edge case)
                if not isinstance(chunk, dict):
                    continue
                normalized = dict(chunk)
                timestamp = normalized.get("timestamp")
                if (
                    isinstance(timestamp, (list, tuple))
                    and len(timestamp) == 2
                    and isinstance(timestamp[0], (int, float))
                    and isinstance(timestamp[1], (int, float))
                ):
                    normalized["start"], normalized["end"] = timestamp
                normalized_chunks.append(normalized)

            # Apply timestamp validation to clean up overlapping segments
            from insanely_fast_whisper_rocm.utils.timestamp_utils import (
                validate_timestamps,
            )

            chunks = validate_timestamps(normalized_chunks)

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
