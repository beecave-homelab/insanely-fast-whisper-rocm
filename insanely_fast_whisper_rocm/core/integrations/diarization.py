"""Speaker diarization integration via pyannote.audio.

Provides ``diarize`` to assign speaker labels to Whisper transcription chunks
as an optional post-processing step.  The module follows the same integration
pattern as ``stable_ts.py``: guarded import, pipeline caching, and graceful
degradation when the optional dependency is absent.
"""

from __future__ import annotations

import logging
import threading
import warnings
from typing import Any

from insanely_fast_whisper_rocm.core.errors import DiarizationError
from insanely_fast_whisper_rocm.utils.constants import DEFAULT_DIARIZATION_MODEL

# Check if torchcodec is available (required by pyannote.audio v4 for
# built-in audio decoding).  On ROCm, torchcodec is incompatible with
# the custom PyTorch build, so we preload audio via torchaudio instead.
try:
    import torchcodec  # noqa: F401

    _TORCHCODEC_AVAILABLE = True
except (ImportError, OSError):
    # ImportError: package not installed.
    # OSError: package installed but native libs fail to load (e.g. ROCm
    # PyTorch build has incompatible symbols, or FFmpeg .so missing).
    _TORCHCODEC_AVAILABLE = False

logger = logging.getLogger(__name__)

# Optional dependency guard
# Suppress the misleading torchcodec warning from pyannote.audio.core.io.
# On ROCm, torchcodec is intentionally excluded; the torchaudio fallback
# in diarize() handles audio loading correctly.
with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        message="torchcodec is not installed correctly",
        category=UserWarning,
    )
    try:
        from pyannote.audio import Pipeline  # type: ignore[import-untyped]
    except ImportError:
        Pipeline = None  # type: ignore[assignment, misc]

# Pipeline cache (simple dict + RLock, no refcounting)
_CACHE: dict[tuple[str, str], Pipeline] = {}  # type: ignore[type-arg]
_LOCK = threading.RLock()


def _get_or_create_pipeline(
    model_name: str,
    device: str,
    hf_token: str,
) -> Pipeline:  # type: ignore[valid-type]
    """Return a cached pyannote Pipeline or create a new one.

    Args:
        model_name: HuggingFace model identifier (e.g.
            ``pyannote/speaker-diarization-3.1``).
        device: Torch device string (``"cpu"`` or ``"cuda"``).
        hf_token: HuggingFace access token for gated models.

    Returns:
        A ``pyannote.audio.Pipeline`` instance.
    """
    # Normalize device string: PyTorch uses "cuda" even for ROCm AMD GPUs.
    if device.lower() == "gpu":
        device = "cuda"

    key = (model_name, device)
    with _LOCK:
        cached = _CACHE.get(key)
        if cached is not None:
            return cached

    # Create outside the lock to avoid blocking other callers during download.
    try:
        pipeline = Pipeline.from_pretrained(
            model_name,
            token=hf_token,
            revision="main",
        )
    except Exception as exc:
        _raise_load_error(model_name, exc)

    # Move to requested device.
    try:
        import torch

        pipeline = pipeline.to(torch.device(device))  # type: ignore[union-attr]
    except Exception as exc:  # pragma: no cover — defensive
        _raise_load_error(model_name, exc)

    with _LOCK:
        # Another thread may have created the same entry meanwhile.
        if key not in _CACHE:
            _CACHE[key] = pipeline
        return _CACHE[key]


def _raise_load_error(model_name: str, exc: Exception) -> None:
    """Translate a pipeline-load exception into a ``DiarizationError``.

    Detects HTTP 403 (gated-model license not accepted) and provides an
    actionable message.  All other errors are wrapped generically.

    Args:
        model_name: The model that was being loaded.
        exc: The original exception.

    Raises:
        DiarizationError: Always.
    """
    status_code = getattr(exc, "status_code", None)
    # Also check for requests-style .response.status_code
    if status_code is None and hasattr(exc, "response"):
        status_code = getattr(exc.response, "status_code", None)  # type: ignore[union-attr]

    if status_code == 403:
        raise DiarizationError(
            f"Access denied for model '{model_name}'. You must visit "
            f"https://huggingface.co/{model_name} and accept the license "
            "terms before using it. Then ensure HF_TOKEN is set to a valid "
            "HuggingFace access token.",
            model=model_name,
            reason="license_not_accepted",
        ) from exc

    raise DiarizationError(
        f"Failed to load diarization model '{model_name}': {exc}",
        model=model_name,
        reason="load_error",
    ) from exc


def clear_diarization_cache() -> None:
    """Invalidate all cached diarization pipelines.

    Useful for OOM recovery or when switching HuggingFace tokens.
    """
    with _LOCK:
        _CACHE.clear()
    logger.info("Diarization pipeline cache cleared")


# --- Speaker-to-segment alignment -------------------------------------------


def _align_speakers_to_segments(
    chunks: list[dict[str, Any]],
    speaker_turns: list[tuple[float, float, str]],
) -> list[dict[str, Any]]:
    """Assign speaker labels to transcription chunks by time overlap.

    For each Whisper chunk, the speaker with the maximum time overlap is
    assigned.  If a chunk has no overlap with any speaker turn, ``speaker``
    is set to ``None``.

    Args:
        chunks: Whisper output chunks, each with ``start`` and ``end`` keys.
        speaker_turns: List of ``(start, end, speaker_label)`` tuples from
            pyannote.

    Returns:
        A new list of chunk dicts with an added ``speaker`` key.
    """
    aligned: list[dict[str, Any]] = []
    for chunk in chunks:
        # Whisper chunks may use "timestamp" (tuple) or "start"/"end" keys.
        ts = chunk.get("timestamp")
        if ts is not None:
            chunk_start, chunk_end = ts[0], ts[1]
        else:
            chunk_start = chunk.get("start", 0.0)
            chunk_end = chunk.get("end", 0.0)

        best_speaker: str | None = None
        best_overlap = 0.0

        for turn_start, turn_end, speaker in speaker_turns:
            overlap_start = max(chunk_start, turn_start)
            overlap_end = min(chunk_end, turn_end)
            overlap = max(0.0, overlap_end - overlap_start)

            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = speaker

        aligned.append({**chunk, "speaker": best_speaker})
    return aligned


# --- Public API --------------------------------------------------------------


def diarize(
    result: dict[str, Any],
    audio_path: str,
    *,
    num_speakers: int | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
    hf_token: str | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    """Apply speaker diarization to a transcription result.

    Uses ``pyannote.audio`` to identify speaker turns in the audio and
    assigns speaker labels to each chunk in the result.

    Args:
        result: Whisper transcription result dict (must contain ``chunks``).
        audio_path: Path to the audio file on disk.
        num_speakers: Exact number of speakers (``None`` = auto-detect).
        min_speakers: Minimum number of speakers.
        max_speakers: Maximum number of speakers.
        hf_token: HuggingFace access token for gated models.
        device: Device for diarization inference (``"cpu"`` or ``"cuda"``).

    Returns:
        The result dict with ``speaker`` added to each chunk and
        ``diarized`` set to ``True``.  If pyannote is not installed, the
        original result is returned unchanged.

    Raises:
        DiarizationError: If the HuggingFace token is missing or the model
            cannot be loaded.
    """
    if Pipeline is None:
        logger.warning("pyannote.audio is not installed – returning result unchanged")
        return result

    if not hf_token:
        raise DiarizationError(
            "A HuggingFace access token (HF_TOKEN) is required for speaker "
            "diarization. Set the HF_TOKEN environment variable or pass "
            "hf_token explicitly.",
            model=DEFAULT_DIARIZATION_MODEL,
            reason="missing_token",
        )

    # Load pipeline (cached).
    pipeline = _get_or_create_pipeline(DEFAULT_DIARIZATION_MODEL, device, hf_token)

    # Prepare audio input for the pipeline.
    # When torchcodec is unavailable (common on ROCm), pyannote's built-in
    # audio decoding fails.  We preload the audio as a tensor dict instead.
    audio_input: str | dict[str, Any] = audio_path
    if not _TORCHCODEC_AVAILABLE:
        try:
            import torchaudio

            waveform, sample_rate = torchaudio.load(audio_path)
            audio_input = {"waveform": waveform, "sample_rate": sample_rate}
            logger.debug(
                "Preloaded audio via torchaudio: shape=%s, sr=%d",
                waveform.shape,
                sample_rate,
            )
        except Exception as exc:
            logger.warning(
                "Failed to preload audio via torchaudio: %s. "
                "Falling back to file-path input (may fail without torchcodec).",
                exc,
            )

    # Run diarization.
    try:
        kwargs: dict[str, Any] = {}
        if num_speakers is not None:
            kwargs["num_speakers"] = num_speakers
        if min_speakers is not None:
            kwargs["min_speakers"] = min_speakers
        if max_speakers is not None:
            kwargs["max_speakers"] = max_speakers

        diarization_result = pipeline(audio_input, **kwargs)
    except Exception as exc:
        logger.error("Diarization inference failed: %s", exc, exc_info=True)
        raise DiarizationError(
            f"Diarization inference failed: {exc}",
            model=DEFAULT_DIARIZATION_MODEL,
            reason="inference_error",
        ) from exc

    # Extract speaker turns as (start, end, label) tuples.
    # pyannote.audio v4 returns a DiarizeOutput dataclass; v3 returns
    # an Annotation directly.  Handle both.
    annotation = diarization_result  # default: v3-style Annotation
    if hasattr(diarization_result, "speaker_diarization"):
        # pyannote.audio v4 — DiarizeOutput dataclass
        annotation = diarization_result.speaker_diarization  # type: ignore[union-attr]

    speaker_turns: list[tuple[float, float, str]] = []
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        speaker_turns.append((turn.start, turn.end, speaker))

    if not speaker_turns:
        logger.warning("Diarization produced no speaker turns")
        return result

    # Align speakers to chunks.
    chunks = result.get("chunks", [])
    if not chunks:
        logger.warning("No chunks in result – nothing to diarize")
        return result

    aligned_chunks = _align_speakers_to_segments(chunks, speaker_turns)

    return {
        **result,
        "chunks": aligned_chunks,
        "segments": aligned_chunks,
        "diarized": True,
    }
