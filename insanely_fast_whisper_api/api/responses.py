"""Response formatting utilities for the API.

This module implements the Strategy pattern for formatting different
types of responses (JSON, text) from ASR processing results.
"""

from typing import Any

from fastapi.responses import JSONResponse, PlainTextResponse

from insanely_fast_whisper_api.core.formatters import (
    FORMATTERS,
)
from insanely_fast_whisper_api.utils import (
    RESPONSE_FORMAT_JSON,
    RESPONSE_FORMAT_SRT,
    RESPONSE_FORMAT_TEXT,
    RESPONSE_FORMAT_VERBOSE_JSON,
    RESPONSE_FORMAT_VTT,
)


class ResponseFormatter:
    """Strategy pattern implementation for response formatting."""

    # --- Internal helper utilities -------------------------------------------------
    @staticmethod
    def _seconds_to_timestamp(seconds: float, for_vtt: bool = False) -> str:
        """Convert float seconds to SRT/VTT timestamp string.

        Args:
            seconds (float): The timestamp in seconds.
            for_vtt (bool): If True, format for WebVTT (uses dot as separator).

        Returns:
            str: The formatted timestamp string (SRT or VTT style).
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds - int(seconds)) * 1000)
        if for_vtt:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

    @staticmethod
    def _segments_to_srt(segments: list[dict]) -> str:
        """Convert segments list to SRT formatted string.

        Args:
            segments (list[dict]): ASR segments with 'start', 'end', and 'text'.

        Returns:
            str: A string containing SRT formatted subtitle cues.
        """
        srt_lines: list[str] = []
        for i, seg in enumerate(segments, start=1):
            start_ts = ResponseFormatter._seconds_to_timestamp(seg.get("start", 0.0))
            end_ts = ResponseFormatter._seconds_to_timestamp(seg.get("end", 0.0))
            text = seg.get("text", "").strip()
            srt_lines.extend([
                str(i),
                f"{start_ts} --> {end_ts}",
                text,
                "",
            ])  # blank line after cue
        return "\n".join(srt_lines).strip()

    @staticmethod
    def _segments_to_vtt(segments: list[dict]) -> str:
        """Convert segments list to WebVTT formatted string.

        Args:
            segments (list[dict]): ASR segments with 'start', 'end', and 'text'.

        Returns:
            str: A string containing WebVTT formatted subtitle cues.
        """
        vtt_lines: list[str] = ["WEBVTT", ""]
        for seg in segments:
            start_ts = ResponseFormatter._seconds_to_timestamp(
                seg.get("start", 0.0), for_vtt=True
            )
            end_ts = ResponseFormatter._seconds_to_timestamp(
                seg.get("end", 0.0), for_vtt=True
            )
            text = seg.get("text", "").strip()
            vtt_lines.extend([f"{start_ts} --> {end_ts}", text, ""])  # blank line
        return "\n".join(vtt_lines).strip()

    @staticmethod
    def format_transcription(
        result: dict[str, Any], response_format: str = RESPONSE_FORMAT_JSON
    ) -> JSONResponse | PlainTextResponse:
        """Format transcription result based on requested format.

        Args:
            result: The transcription result dictionary from ASR pipeline
            response_format: Desired response format ("json" or "text")

        Returns:
            Union[JSONResponse, PlainTextResponse]: Formatted response
        """
        # Plain text response
        if response_format == RESPONSE_FORMAT_TEXT:
            text_output = result.get("text", "")
            return PlainTextResponse(
                text_output, media_type="text/plain; charset=utf-8"
            )

        # Minimal JSON – only top-level text
        if response_format == RESPONSE_FORMAT_JSON:
            return JSONResponse(content={"text": result.get("text", "")})

        # Verbose JSON – map `chunks` to `segments` & add language
        if response_format == RESPONSE_FORMAT_VERBOSE_JSON:
            # Build verbose JSON per OpenAI Whisper specification
            # Normalise each chunk to include all expected keys so downstream
            # clients (e.g. MacWhisper) can rely on their presence even if
            # some values are only best-effort defaults.
            chunks = result.get("chunks", [])
            segments: list[dict] = []
            for idx, chunk in enumerate(chunks):
                seg: dict[str, Any] = {
                    "id": chunk.get("id", idx),
                    "seek": chunk.get("seek", 0),
                    "start": chunk.get("start", 0.0),
                    "end": chunk.get("end", 0.0),
                    "text": chunk.get("text", ""),
                    "tokens": chunk.get("tokens", []),
                    "temperature": chunk.get("temperature", 0.0),
                    "avg_logprob": chunk.get("avg_logprob", 0.0),
                    "compression_ratio": chunk.get("compression_ratio", 0.0),
                    "no_speech_prob": chunk.get("no_speech_prob", 0.0),
                }
                segments.append(seg)

            verbose_payload: dict[str, Any] = {
                "text": result.get("text", ""),
                "segments": segments,
            }

            # Attempt to include detected language if available
            language = result.get("language") or result.get("config_used", {}).get(
                "language"
            )
            if language:
                verbose_payload["language"] = language
            return JSONResponse(content=verbose_payload)

        # Subtitle formats (SRT/VTT)
        if response_format in (RESPONSE_FORMAT_SRT, RESPONSE_FORMAT_VTT):
            if response_format == RESPONSE_FORMAT_SRT:
                text_output = FORMATTERS["srt"].format(result)
                mime = "text/srt"
            else:
                text_output = FORMATTERS["vtt"].format(result)
                mime = "text/vtt"
            return PlainTextResponse(text_output, media_type=mime)

        # Unsupported format – return 400 handled by caller or fallback
        return JSONResponse(
            status_code=400, content={"error": "Unsupported response_format"}
        )

    @staticmethod
    def format_translation(
        result: dict[str, Any], response_format: str = RESPONSE_FORMAT_JSON
    ) -> JSONResponse | PlainTextResponse:
        """Format translation result based on requested format.

        Args:
            result: The translation result dictionary from ASR pipeline
            response_format: Desired response format ("json" or "text")

        Returns:
            Union[JSONResponse, PlainTextResponse]: Formatted response
        """
        # Plain text response
        if response_format == RESPONSE_FORMAT_TEXT:
            transcription_output = result.get("transcription", result)
            text_output = transcription_output.get("text", "")
            return PlainTextResponse(
                text_output, media_type="text/plain; charset=utf-8"
            )

        # Minimal JSON response
        if response_format == RESPONSE_FORMAT_JSON:
            transcription_output = result.get("transcription", result)
            return JSONResponse(content={"text": transcription_output.get("text", "")})

        # Verbose JSON response – reuse logic similar to transcription
        if response_format == RESPONSE_FORMAT_VERBOSE_JSON:
            transcription_output = result.get("transcription", result)
            chunks = transcription_output.get("chunks", [])
            segments: list[dict] = []
            for idx, chunk in enumerate(chunks):
                segments.append({
                    "id": chunk.get("id", idx),
                    "seek": chunk.get("seek", 0),
                    "start": chunk.get("start", 0.0),
                    "end": chunk.get("end", 0.0),
                    "text": chunk.get("text", ""),
                    "tokens": chunk.get("tokens", []),
                    "temperature": chunk.get("temperature", 0.0),
                    "avg_logprob": chunk.get("avg_logprob", 0.0),
                    "compression_ratio": chunk.get("compression_ratio", 0.0),
                    "no_speech_prob": chunk.get("no_speech_prob", 0.0),
                })
            verbose_payload = {
                "text": transcription_output.get("text", ""),
                "segments": segments,
            }
            language = transcription_output.get("language") or transcription_output.get(
                "config_used", {}
            ).get("language")
            if language:
                verbose_payload["language"] = language
            return JSONResponse(content=verbose_payload)

        # Subtitle formats
        if response_format in (RESPONSE_FORMAT_SRT, RESPONSE_FORMAT_VTT):
            transcription_output = result.get("transcription", result)
            if response_format == RESPONSE_FORMAT_SRT:
                text_output = FORMATTERS["srt"].format(transcription_output)
                mime = "text/srt"
            else:
                text_output = FORMATTERS["vtt"].format(transcription_output)
                mime = "text/vtt"
            return PlainTextResponse(text_output, media_type=mime)

        # Fallback unsupported
        return JSONResponse(
            status_code=400, content={"error": "Unsupported response_format"}
        )
