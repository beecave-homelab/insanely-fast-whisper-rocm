"""Format API responses for various payload styles."""

from collections.abc import Callable
from typing import Any

from fastapi.responses import JSONResponse, PlainTextResponse

from insanely_fast_whisper_rocm.core.formatters import FORMATTERS, BaseFormatter
from insanely_fast_whisper_rocm.utils import (
    RESPONSE_FORMAT_JSON,
    RESPONSE_FORMAT_SRT,
    RESPONSE_FORMAT_TEXT,
    RESPONSE_FORMAT_VERBOSE_JSON,
    RESPONSE_FORMAT_VTT,
)
from insanely_fast_whisper_rocm.utils.format_time import (
    format_srt_time,
    format_vtt_time,
)

FormatterCallable = Callable[[dict[str, Any]], str]
FormatterLike = FormatterCallable | type[BaseFormatter] | BaseFormatter


class ResponseFormatter:
    """Strategy pattern implementation for response formatting."""

    # Note: Subtitle formatting (SRT/VTT) is delegated to core FORMATTERS to
    # avoid duplicate logic and ensure consistent behavior across API/CLI/WebUI.

    # --- Backward-compatibility helpers (used in tests) -------------------------
    @staticmethod
    def _seconds_to_timestamp(seconds: float, for_vtt: bool = False) -> str:
        """Convert seconds to a timestamp string.

        Delegates to explicit utilities and exists for test compatibility.

        Args:
            seconds: Timestamp in seconds.
            for_vtt: Whether to return a VTT-style timestamp.

        Returns:
            Timestamp string in SRT or VTT style.
        """
        return format_vtt_time(seconds) if for_vtt else format_srt_time(seconds)

    @staticmethod
    def _segments_to_srt(segments: list[dict]) -> str:
        """Convert segments to SRT string.

        Thin wrapper that mirrors previous behavior while relying on
        the shared timestamp utilities.

        Args:
            segments: List of segment dicts containing 'start', 'end', 'text'.

        Returns:
            A string containing SRT-formatted cues.
        """
        srt_lines: list[str] = []
        for i, seg in enumerate(segments, start=1):
            start_ts = ResponseFormatter._seconds_to_timestamp(
                seg.get("start", 0.0), for_vtt=False
            )
            end_ts = ResponseFormatter._seconds_to_timestamp(
                seg.get("end", 0.0), for_vtt=False
            )
            text = seg.get("text", "").strip()
            srt_lines.extend([str(i), f"{start_ts} --> {end_ts}", text, ""])
        if not srt_lines:
            return ""
        return "\n".join(srt_lines)

    @staticmethod
    def _segments_to_vtt(segments: list[dict]) -> str:
        """Convert segments to WebVTT string.

        Thin wrapper that mirrors previous behavior while relying on
        the shared timestamp utilities.

        Args:
            segments: List of segment dicts containing 'start', 'end', 'text'.

        Returns:
            A string containing WebVTT-formatted cues.
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
            vtt_lines.extend([f"{start_ts} --> {end_ts}", text, ""])
        rendered = "\n".join(vtt_lines)
        return rendered.rstrip("\n") if not segments else rendered

    @staticmethod
    def _get_formatter(name: str) -> FormatterLike:
        """Return the formatter associated with ``name``.

        Prefers overrides attached to the class (used in tests) before
        falling back to the module-level `FORMATTERS` mapping.

        Args:
            name: Key identifying the formatter to retrieve.

        Returns:
            FormatterLike: Either a callable, formatter instance, or formatter class.
        """
        formatter_map = getattr(ResponseFormatter, "FORMATTERS", None)
        if formatter_map is None:
            return FORMATTERS[name]
        return formatter_map[name]

    @staticmethod
    def _call_formatter(formatter: FormatterLike, payload: dict[str, Any]) -> str:
        """Invoke a formatter that may be a class, instance, or callable.

        Args:
            formatter: Formatter candidate retrieved via `_get_formatter`.
            payload: Result dictionary passed to the formatter implementation.

        Returns:
            str: Formatted subtitle text.
        """
        if isinstance(formatter, type) and issubclass(formatter, BaseFormatter):
            return formatter.format(payload)
        if isinstance(formatter, BaseFormatter):
            return formatter.format(payload)

        format_callable: Callable[..., str] = getattr(formatter, "format", formatter)

        try:
            return format_callable(payload)
        except TypeError:
            return format_callable()

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
                formatter = ResponseFormatter._get_formatter("srt")
                text_output = ResponseFormatter._call_formatter(formatter, result)
                mime = "text/srt"
            else:
                formatter = ResponseFormatter._get_formatter("vtt")
                text_output = ResponseFormatter._call_formatter(formatter, result)
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
                formatter = ResponseFormatter._get_formatter("srt")
                text_output = ResponseFormatter._call_formatter(
                    formatter, transcription_output
                )
                mime = "text/srt"
            else:
                formatter = ResponseFormatter._get_formatter("vtt")
                text_output = ResponseFormatter._call_formatter(
                    formatter, transcription_output
                )
                mime = "text/vtt"
            return PlainTextResponse(text_output, media_type=mime)

        # Fallback unsupported
        return JSONResponse(
            status_code=400, content={"error": "Unsupported response_format"}
        )
