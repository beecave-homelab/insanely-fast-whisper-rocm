"""Stable-ts integration wrapper.

Provides `stabilize_timestamps` to refine Whisper transcription results using
`stable-whisper`'s `transcribe_any` convenience function.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from insanely_fast_whisper_api.utils.timestamp_utils import (
    normalize_timestamp_format,
    validate_timestamps,
)

logger = logging.getLogger(__name__)

try:
    import stable_whisper  # type: ignore

    # Prefer explicit function if available; also support common alias 'postprocess'.
    _postprocess = getattr(stable_whisper, "postprocess_word_timestamps", None)
    _postprocess_alt = getattr(stable_whisper, "postprocess", None)
except ImportError as err:  # pragma: no cover
    logger.error(
        "stable-whisper is not installed – stabilize_timestamps will be a no-op: %s",
        err,
    )
    stable_whisper = None  # type: ignore
    _postprocess = None
    _postprocess_alt = None


def _to_dict(obj: object) -> dict[str, Any]:
    """Convert the result object returned by *stable-whisper* to a dictionary.

    Args:
        obj: The object returned by stable-whisper; can be a dict or a model-like
            object exposing ``to_dict``/``model_dump``/``_asdict``.

    Returns:
        dict[str, Any]: A dictionary representation of the input object.
    """
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

    # Use centralized normalization function
    converted = normalize_timestamp_format(result)
    logger.info("_convert_to_stable: normalized timestamp format")

    # Fix individual segment fields and validate timestamps
    segments = converted.get("segments", [])
    validated_segments = validate_timestamps(segments)
    converted["segments"] = validated_segments

    logger.info(
        "_convert_to_stable returning %d validated segments", len(validated_segments)
    )
    return converted


def stabilize_timestamps(
    result: dict[str, Any],
    *,
    demucs: bool = False,
    vad: bool = False,
    vad_threshold: float = 0.35,
    progress_cb: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Return a copy of *result* with word-level timestamps via stable-ts.

    Args:
        result: Base transcription result to refine.
        demucs: Whether to run Demucs denoising.
        vad: Whether to run Voice Activity Detection.
        vad_threshold: VAD threshold when ``vad`` is True.
        progress_cb: Optional callback receiving human-readable status updates.

    Returns:
        A refined result dictionary. If stabilization fails, the original
        ``result`` is returned.
    """
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
        if progress_cb:
            progress_cb("stable-ts unavailable; skipping stabilization")
        return result

    audio_path_str = result.get("original_file") or result.get("audio_file_path")
    if not audio_path_str:
        logger.error(
            "Audio path missing from transcription result; cannot stabilize timestamps"
        )
        if progress_cb:
            progress_cb("stable-ts: audio path missing; skipping")
        return result

    audio_path = Path(audio_path_str).expanduser().resolve()
    if not audio_path.exists():
        logger.error("Audio file not found for stabilization: %s", audio_path)
        if progress_cb:
            progress_cb("stable-ts: audio file not found; skipping")
        return result

    # Prepare a stable-whisper–compatible dict
    converted = _convert_to_stable(result)

    def inference_func(*_a: object, **_k: object) -> dict[str, Any]:
        """Return the precomputed converted dict regardless of inputs.

        This mirrors the previous lambda behavior used for transcribe_any.
        """
        return converted

    # 1. Preferred paths: postprocess_* style APIs (avoid torchaudio.save)
    for func_name, func in (
        ("postprocess_word_timestamps", _postprocess),
        ("postprocess", _postprocess_alt),
    ):
        if func is None:
            continue
        try:
            if progress_cb:
                progress_cb(f"stable-ts: {func_name} running")
            try:
                refined = func(  # type: ignore[misc]
                    converted,
                    audio=str(audio_path),
                    demucs=demucs,
                    vad=vad,
                    vad_threshold=vad_threshold,
                )
            except TypeError:
                # Some versions may not accept the same kwargs; retry with minimal args.
                refined = func(converted, audio=str(audio_path))  # type: ignore[misc]
            refined_dict = _to_dict(refined)
            merged = {**result, **refined_dict, "stabilized": True}
            if "segments" in merged:
                merged.pop("chunks", None)
            merged.setdefault("segments_count", len(refined_dict.get("segments", [])))
            merged.setdefault("stabilization_path", func_name)
            if progress_cb:
                progress_cb(f"stable-ts: refinement successful ({func_name})")
            return merged
        except Exception as exc:  # pragma: no cover
            logger.warning("%s failed: %s", func_name, exc)
            if progress_cb:
                progress_cb(f"stable-ts: {func_name} failed; trying alt path")

    # 2. Alternative path: lambda-inference via transcribe_any
    # (may require torchaudio.save)
    try:
        if progress_cb:
            progress_cb(
                f"stable-ts: running (demucs={demucs}, vad={vad}, thr={vad_threshold})"
            )
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
        if progress_cb:
            progress_cb("stable-ts: refinement successful")
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
        if progress_cb:
            progress_cb("stable-ts: merging results")
        return merged
    except Exception as exc:  # pragma: no cover
        logger.error("stable-ts lambda inference path failed: %s", exc, exc_info=True)
        if progress_cb:
            progress_cb("stable-ts: alternative path failed")

    # 3. Give up
    logger.error("stable-ts processing failed; returning original result")
    if progress_cb:
        progress_cb("stable-ts: failed; returning original result")
    return result
