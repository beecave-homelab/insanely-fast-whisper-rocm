"""Stable-ts integration wrapper.

Provides `stabilize_timestamps` to refine Whisper transcription results using
`stable-whisper`'s `transcribe_any` convenience function.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

try:
    import stable_whisper  # type: ignore

    _postprocess = getattr(stable_whisper, "postprocess_word_timestamps", None)
except ImportError as err:  # pragma: no cover
    logger.error(
        "stable-whisper is not installed – stabilize_timestamps will be a no-op: %s",
        err,
    )
    stable_whisper = None  # type: ignore
    _postprocess = None

__all__ = ["stabilize_timestamps"]


def _to_dict(obj: Any) -> dict[str, Any]:
    """Convert the result object returned by *stable-whisper* to a dictionary."""
    logger.info("_to_dict called with type=%s", type(obj))
    if isinstance(obj, dict):
        return obj
    for attr in ("to_dict", "model_dump", "_asdict"):
        if hasattr(obj, attr):
            return getattr(obj, attr)()  # type: ignore[arg-type]
    return {"text": str(obj)}


def _convert_to_stable(result: dict[str, Any]) -> dict[str, Any]:
    """Return *result* reshaped to match Whisper JSON expected by stable-ts."""
    logger.info("_convert_to_stable input keys=%s", list(result.keys()))
    converted = deepcopy(result)
    logger.info("_convert_to_stable: deep-copied result")
    # Rename chunks -> segments if present
    if "segments" not in converted and "chunks" in converted:
        converted["segments"] = converted.pop("chunks")
        logger.info("Renamed 'chunks' to 'segments'")

    # Fix individual segment fields
    segments = converted.get("segments", [])
    for seg in segments:
        if (
            isinstance(seg, dict)
            and "timestamp" in seg
            and isinstance(seg["timestamp"], (list, tuple))
            and len(seg["timestamp"]) == 2
        ):
            seg["start"], seg["end"] = seg.pop("timestamp")
        # Swap timestamps if they are in the wrong order and both are not None
        if (
            isinstance(seg, dict)
            and seg.get("start") is not None
            and seg.get("end") is not None
            and seg["start"] > seg["end"]
        ):
            seg["start"], seg["end"] = seg["end"], seg["start"]

    # Sort and de-overlap (handle missing/None starts by pushing them last)
    segments.sort(
        key=lambda s: s.get("start") if s.get("start") is not None else float("inf")
    )
    cleaned, last_end = [], 0.0
    for seg in segments:
        start, end = seg.get("start"), seg.get("end")
        if start is None or end is None:
            continue
        if start < last_end:
            duration = max(0.0, end - start)
            start = last_end
            end = last_end + duration
        if start >= last_end:
            seg["start"], seg["end"] = start, end
            cleaned.append(seg)
            last_end = end

    converted["segments"] = cleaned
    logger.info("_convert_to_stable returning %d cleaned segments", len(cleaned))
    return converted


def stabilize_timestamps(
    result: dict[str, Any],
    *,
    demucs: bool = False,
    vad: bool = False,
    vad_threshold: float = 0.35,
) -> dict[str, Any]:
    """Return a copy of *result* with word-level timestamps via stable-ts."""
    logger.info(
        "stabilize_timestamps called demucs=%s vad=%s vad_threshold=%s",
        demucs,
        vad,
        vad_threshold,
    )
    if stable_whisper is None:
        logger.warning(
            "stable-whisper not available – returning original result unchanged"
        )
        return result

    audio_path_str = result.get("original_file") or result.get("audio_file_path")
    if not audio_path_str:
        logger.error(
            "Audio path missing from transcription result; cannot stabilize timestamps"
        )
        return result

    audio_path = Path(audio_path_str).expanduser().resolve()
    if not audio_path.exists():
        logger.error("Audio file not found for stabilization: %s", audio_path)
        return result

    # Prepare a stable-whisper–compatible dict
    converted = _convert_to_stable(result)
    inference_func = lambda *_a, **_k: converted  # type: ignore

    # 1️⃣ Primary path: lambda-inference via transcribe_any
    try:
        refined = stable_whisper.transcribe_any(
            inference_func,
            audio=str(audio_path),
            denoiser="demucs" if demucs else None,
            vad=vad,
            vad_threshold=vad_threshold,
            check_sorted=False,
        )
        refined_dict = _to_dict(refined)
        logger.info(
            "stable-ts succeeded: segments=%d", len(refined_dict.get("segments", []))
        )
        merged = {**result, **refined_dict, "stabilized": True}

        def _segments_have_timestamps(seg_list: list[dict]) -> bool:
            return any(
                (s.get("start") is not None and s.get("end") is not None)
                for s in seg_list
            )

        if "segments" in merged and _segments_have_timestamps(merged["segments"]):
            # Only discard original chunks if we actually obtained usable timestamps
            merged.pop("chunks", None)
        # Enrich with metadata before returning (lazy logging ready)
        merged.setdefault("segments_count", len(refined_dict.get("segments", [])))
        merged.setdefault("stabilization_path", "lambda")
        return merged
    except Exception as exc:  # pragma: no cover
        logger.error("stable-ts lambda inference path failed: %s", exc, exc_info=True)

    # 2️⃣ Fallback: postprocess_word_timestamps if available
    if _postprocess is not None:
        try:
            refined = _postprocess(
                converted,
                audio=str(audio_path),
                demucs=demucs,
                vad=vad,
                vad_threshold=vad_threshold,
            )
            refined_dict = _to_dict(refined)
            merged = {**result, **refined_dict, "stabilized": True}
            if "segments" in merged:
                merged.pop("chunks", None)
            merged.setdefault("segments_count", len(refined_dict.get("segments", [])))
            merged.setdefault("stabilization_path", "postprocess")
            return merged
        except Exception as exc:  # pragma: no cover
            logger.warning("postprocess_word_timestamps fallback failed: %s", exc)

    # 3️⃣ Give up
    logger.error("stable-ts processing failed; returning original result")
    return result
