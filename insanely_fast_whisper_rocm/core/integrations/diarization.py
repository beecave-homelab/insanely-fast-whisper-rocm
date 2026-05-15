"""Speaker diarization integration via pyannote.audio.

Provides ``diarize`` to assign speaker labels to Whisper transcription chunks
as an optional post-processing step.  The module follows the same integration
pattern as ``stable_ts.py``: guarded import, pipeline caching, and graceful
degradation when the optional dependency is absent.
"""

from __future__ import annotations

import hashlib
import logging
import subprocess
import tempfile
import threading
import warnings
from pathlib import Path
from typing import Any

from insanely_fast_whisper_rocm.core.errors import DiarizationError
from insanely_fast_whisper_rocm.utils.constants import DEFAULT_DIARIZATION_MODEL

# Check if torchcodec is available (required by pyannote.audio v4 for
# built-in audio decoding).  On ROCm, torchcodec is incompatible with
# the custom PyTorch build, so we preload audio via torchaudio instead.
try:
    import torchcodec as _torchcodec  # noqa: F401  # type: ignore[import-untyped]

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
# Key includes a hash of hf_token so different tokens get separate entries.
_CACHE: dict[tuple[str, str, str], Pipeline] = {}  # type: ignore[type-arg]
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

    token_hash = hashlib.sha256(hf_token.encode()).hexdigest()[:16]
    key = (model_name, device, token_hash)
    with _LOCK:
        cached = _CACHE.get(key)
        if cached is not None:
            return cached

    # Create outside the lock to avoid blocking other callers during download.
    pipeline: Pipeline | None = None  # type: ignore[type-arg]
    try:
        pipeline = Pipeline.from_pretrained(  # type: ignore[union-attr]
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
        else:
            # Another thread won the race; move orphaned pipeline off GPU
            # to avoid leaked device memory.
            try:
                import torch

                if pipeline is not None:
                    pipeline.to(torch.device("cpu"))
                    del pipeline
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                logger.warning(
                    "Duplicate pipeline created for key=%s; orphan freed", key
                )
            except Exception:  # pragma: no cover
                logger.debug("Failed to clean up orphaned pipeline", exc_info=True)
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
            + f"https://huggingface.co/{model_name} and accept the license "
            + "terms before using it. Then ensure HF_TOKEN is set to a valid "
            + "HuggingFace access token.",
            model=model_name,
            reason="license_not_accepted",
        ) from exc

    raise DiarizationError(
        f"Failed to load diarization model '{model_name}': {exc}",
        model=model_name,
        reason="load_error",
    ) from exc


def _is_rocm_miopen_runtime_error(exc: Exception) -> bool:
    """Return whether an exception is a ROCm/MIOpen inference failure.

    Args:
        exc: Runtime exception raised by pyannote/PyTorch.

    Returns:
        True when the error matches a known ROCm/MIOpen runtime failure.
    """
    message = str(exc).lower()
    return (
        "miopenstatusunknownerror" in message
        or "miopen" in message
        or "rocrand" in message
    )


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
            chunk_start = ts[0]
            chunk_end = ts[1] if len(ts) > 1 and ts[1] is not None else chunk_start
        else:
            chunk_start = chunk.get("start", 0.0)
            chunk_end = chunk.get("end", 0.0)

        if chunk_end is None:
            chunk_end = chunk_start

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


def _preload_audio_as_waveform(audio_path: str) -> str | dict[str, Any]:
    """Load audio as a waveform dict for pyannote when torchcodec is absent.

    Tries ``torchaudio.load`` first (fast path for WAV/FLAC).  When that
    fails because the format is unsupported by the soundfile backend (e.g.
    ``.m4a``, ``.mp3``), converts to WAV via ``ffmpeg`` and retries.

    Args:
        audio_path: Path to the audio file on disk.

    Returns:
        The original ``audio_path`` string if loading fails, or a dict
        ``{"waveform": Tensor, "sample_rate": int}`` suitable for pyannote.
    """
    # Fast path: torchaudio can read it directly (WAV, FLAC, etc.)
    try:
        import torchaudio  # pyright: ignore[reportMissingTypeStubs]

        waveform, sample_rate = torchaudio.load(audio_path)
        # Ensure mono — pyannote expects single-channel audio.
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        logger.debug(
            "Preloaded audio via torchaudio: shape=%s, sr=%d",
            waveform.shape,
            sample_rate,
        )
        return {"waveform": waveform, "sample_rate": sample_rate}
    except Exception as direct_exc:
        logger.debug(
            "torchaudio.load failed for %s: %s – trying ffmpeg conversion",
            audio_path,
            direct_exc,
        )

    # Slow path: convert to WAV via ffmpeg, then load.
    suffix = Path(audio_path).suffix.lower()
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False, prefix="diarize_")
    tmp_path = tmp.name
    tmp.close()
    try:
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            audio_path,
            "-ar",
            "16000",
            "-ac",
            "1",
            tmp_path,
        ]
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, check=False, timeout=30
            )
        except subprocess.TimeoutExpired:
            logger.warning(
                "ffmpeg conversion timed out for %s (30s). "
                + "Falling back to file-path input (may fail without torchcodec).",
                audio_path,
            )
            return audio_path

        if result.returncode != 0:
            logger.warning(
                "ffmpeg conversion failed for %s: %s. "
                + "Falling back to file-path input (may fail without torchcodec).",
                audio_path,
                result.stderr[:300],
            )
            return audio_path

        waveform, sample_rate = torchaudio.load(tmp_path)  # type: ignore[possibly-undefined]
        logger.debug(
            "Preloaded audio via ffmpeg→torchaudio: shape=%s, sr=%d, src=%s",
            waveform.shape,
            sample_rate,
            suffix,
        )
        return {"waveform": waveform, "sample_rate": sample_rate}
    except Exception as exc:
        logger.warning(
            "Failed to preload audio (ffmpeg fallback): %s. "
            + "Falling back to file-path input (may fail without torchcodec).",
            exc,
        )
        return audio_path
    finally:
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except OSError:
            pass


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
        ``diarized`` set to ``True``.

    Raises:
        DiarizationError: If pyannote is not installed, the HuggingFace
            token is missing, or the model cannot be loaded.
    """
    if Pipeline is None:
        raise DiarizationError(
            "Speaker diarization was requested but pyannote.audio is not installed. "
            + "Install the diarization extras to enable --diarize support.",
            model=DEFAULT_DIARIZATION_MODEL,
            reason="pyannote_not_installed",
        )

    if not hf_token:
        raise DiarizationError(
            "A HuggingFace access token (HF_TOKEN) is required for speaker "
            + "diarization. Set the HF_TOKEN environment variable or pass "
            + "hf_token explicitly.",
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
        audio_input = _preload_audio_as_waveform(audio_path)

    # Run diarization.
    try:
        kwargs: dict[str, Any] = {}
        if num_speakers is not None:
            kwargs["num_speakers"] = num_speakers
        if min_speakers is not None:
            kwargs["min_speakers"] = min_speakers
        if max_speakers is not None:
            kwargs["max_speakers"] = max_speakers

        try:
            diarization_result = pipeline(audio_input, **kwargs)
        except Exception as exc:
            is_gpu_device = device.lower() in {"cuda", "gpu"}
            if not is_gpu_device or not _is_rocm_miopen_runtime_error(exc):
                raise

            logger.warning(
                "Diarization failed on ROCm GPU with %s; retrying on CPU",
                exc,
            )
            cpu_pipeline = _get_or_create_pipeline(
                DEFAULT_DIARIZATION_MODEL, "cpu", hf_token
            )
            diarization_result = cpu_pipeline(audio_input, **kwargs)
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

    # Align speakers to chunks, falling back to segments when chunks are
    # absent (e.g. after stabilization removes the ``chunks`` key).
    chunks = result.get("chunks") or []
    segments = result.get("segments") or []

    if not chunks and not segments:
        logger.warning("No chunks or segments in result - nothing to diarize")
        return result

    # Use chunks as the primary alignment target; fall back to segments.
    primary = chunks if chunks else segments
    aligned_primary = _align_speakers_to_segments(primary, speaker_turns)

    out: dict[str, Any] = {**result, "chunks": aligned_primary, "diarized": True}

    if chunks:
        # Also align segments if they exist and differ from chunks.
        if segments and segments is not chunks:
            out["segments"] = _align_speakers_to_segments(segments, speaker_turns)
        else:
            out["segments"] = aligned_primary
    else:
        # No original chunks — segments were the primary target.
        out["segments"] = aligned_primary

    return out
