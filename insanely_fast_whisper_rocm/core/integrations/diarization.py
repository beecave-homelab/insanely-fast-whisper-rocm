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
import time
import warnings
from pathlib import Path
from typing import Any

from insanely_fast_whisper_rocm.core.backend_cache import invalidate_gpu_cache
from insanely_fast_whisper_rocm.core.errors import DiarizationError
from insanely_fast_whisper_rocm.utils.constants import (
    DEFAULT_DIARIZATION_MODEL,
    DIARIZATION_ALLOW_CPU_FALLBACK,
    DIARIZATION_FFMPEG_TIMEOUT_SECONDS,
    DIARIZATION_PRELOAD_AUDIO,
)

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
        message=r"\s*torchcodec is not installed correctly.*",
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
    device = _normalize_diarization_device(device)

    # Normalize hf_token to bytes before hashing; handle None/bytes gracefully.
    if isinstance(hf_token, bytes):
        token_bytes = hf_token
    else:
        token_bytes = str(hf_token).encode()
    token_hash = hashlib.sha256(token_bytes).hexdigest()[:16]
    key = (model_name, device, token_hash)
    with _LOCK:
        cached = _CACHE.get(key)
        if cached is not None:
            logger.info(
                "Diarization timing: pipeline_cache_hit model=%s device=%s",
                model_name,
                device,
            )
            return cached

    # Create outside the lock to avoid blocking other callers during download.
    pipeline: Pipeline | None = None  # type: ignore[type-arg]
    started_at = time.perf_counter()
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
    load_elapsed = time.perf_counter() - started_at
    logger.info(
        "Diarization timing: pipeline_load_seconds=%.3f model=%s device=%s",
        load_elapsed,
        model_name,
        device,
    )

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
                    pipeline = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                logger.warning(
                    "Duplicate pipeline created for key=%s; orphan freed", key
                )
            except Exception as exc:  # pragma: no cover
                logger.warning(
                    "Failed to clean up orphaned pipeline for key=%s",
                    key,
                    exc_info=exc,
                )
        return _CACHE[key]


def _normalize_diarization_device(device: str) -> str:
    """Return a torch-compatible diarization device string.

    Args:
        device: User-facing device string.

    Returns:
        Torch device string for pyannote inference.
    """
    normalized = device.strip().lower()
    if normalized in {"gpu", "0"}:
        return "cuda"
    return normalized


def _log_timing(label: str, started_at: float, **fields: object) -> None:
    """Log a concise diarization timing event.

    Args:
        label: Event label.
        started_at: ``time.perf_counter()`` value captured before the work.
        **fields: Additional structured fields to append to the log line.
    """
    elapsed = time.perf_counter() - started_at
    details = " ".join(f"{key}={value}" for key, value in fields.items())
    if details:
        logger.info("Diarization timing: %s_seconds=%.3f %s", label, elapsed, details)
    else:
        logger.info("Diarization timing: %s_seconds=%.3f", label, elapsed)


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


def _is_retriable_gpu_runtime_error(exc: Exception) -> bool:
    """Return whether an exception is a ROCm/MIOpen or GPU OOM failure.

    Args:
        exc: Runtime exception raised by pyannote/PyTorch.

    Returns:
        True when the error matches a known ROCm/MIOpen runtime failure
        or GPU out-of-memory condition.
    """
    message = str(exc).lower()
    return (
        "miopenstatusunknownerror" in message
        or "miopen" in message
        or "rocrand" in message
        or "out of memory" in message
    )


def clear_diarization_cache() -> None:
    """Invalidate all cached diarization pipelines.

    Useful for OOM recovery or when switching HuggingFace tokens.
    """
    with _LOCK:
        _CACHE.clear()
    logger.info("Diarization pipeline cache cleared")


def _clear_gpu_diarization_cache() -> None:
    """Remove GPU-based diarization pipelines from the cache.

    Moves cached GPU pipelines to CPU and clears the cache entries so
    VRAM is freed for the ASR backend on the next transcription request.
    """
    with _LOCK:
        keys_to_remove = []
        for key, cached_pipeline in _CACHE.items():
            device = key[1]
            if isinstance(device, str) and device == "cuda":
                try:
                    import torch

                    cached_pipeline.to(torch.device("cpu"))
                except Exception as exc:  # noqa: BLE001  # defensive: cleanup must not raise
                    logger.debug(
                        "Failed to move cached pipeline %s to CPU: %s", key, exc
                    )
                keys_to_remove.append(key)
        for key in keys_to_remove:
            _CACHE.pop(key, None)
    if keys_to_remove:
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as exc:  # noqa: BLE001  # defensive: cache cleanup must not raise
            logger.debug("Failed to empty CUDA cache after GPU cleanup: %s", exc)
        logger.info(
            "Cleared %d GPU diarization pipeline(s) from cache",
            len(keys_to_remove),
        )


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
    normalized_turns = _normalize_speaker_turns(speaker_turns)
    for chunk in chunks:
        chunk_start, chunk_end = _chunk_time_bounds(chunk)

        best_speaker: str | None = None
        best_overlap = 0.0

        for turn_start, turn_end, speaker in normalized_turns:
            overlap = _time_overlap(chunk_start, chunk_end, turn_start, turn_end)

            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = speaker

        aligned.append({**chunk, "speaker": best_speaker})
    return aligned


def _chunk_time_bounds(chunk: dict[str, Any]) -> tuple[float, float]:
    """Return stable start and end times for an ASR chunk or segment.

    Args:
        chunk: Whisper chunk or segment dictionary.

    Returns:
        Normalized ``(start, end)`` seconds.
    """
    ts = chunk.get("timestamp")
    if ts is not None:
        chunk_start = ts[0]
        chunk_end = ts[1] if len(ts) > 1 and ts[1] is not None else chunk_start
    else:
        chunk_start = chunk.get("start", 0.0)
        chunk_end = chunk.get("end", chunk_start)

    start = _coerce_time(chunk_start)
    end = _coerce_time(chunk_end, default=start)
    if end < start:
        return end, start
    return start, end


def _coerce_time(value: object, *, default: float = 0.0) -> float:
    """Return ``value`` as a finite float timestamp.

    Args:
        value: Timestamp-like value.
        default: Fallback when ``value`` is missing or invalid.

    Returns:
        Finite float timestamp.
    """
    try:
        coerced = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    if coerced != coerced:
        return default
    return coerced


def _normalize_speaker_turns(
    speaker_turns: list[tuple[float, float, str]],
) -> list[tuple[float, float, str]]:
    """Return valid speaker turns sorted by start time.

    Args:
        speaker_turns: Raw pyannote speaker turns.

    Returns:
        Sorted ``(start, end, speaker)`` tuples with invalid spans removed.
    """
    normalized: list[tuple[float, float, str]] = []
    for start, end, speaker in speaker_turns:
        turn_start = _coerce_time(start)
        turn_end = _coerce_time(end, default=turn_start)
        if turn_end <= turn_start:
            continue
        normalized.append((turn_start, turn_end, speaker))
    return sorted(normalized, key=lambda item: (item[0], item[1], item[2]))


def _time_overlap(
    chunk_start: float,
    chunk_end: float,
    turn_start: float,
    turn_end: float,
) -> float:
    """Return overlap seconds, including point-in-turn timestamp matches.

    Args:
        chunk_start: Chunk start time in seconds.
        chunk_end: Chunk end time in seconds.
        turn_start: Speaker turn start time in seconds.
        turn_end: Speaker turn end time in seconds.

    Returns:
        Positive overlap weight for speaker selection.
    """
    if chunk_end == chunk_start and turn_start <= chunk_start < turn_end:
        return 1e-6
    overlap_start = max(chunk_start, turn_start)
    overlap_end = min(chunk_end, turn_end)
    return max(0.0, overlap_end - overlap_start)


def _preload_audio_as_waveform(audio_path: str) -> dict[str, Any]:
    """Load audio as a waveform dict for pyannote when torchcodec is absent.

    Tries ``torchaudio.load`` first (fast path for WAV/FLAC).  When that
    fails because the format is unsupported by the soundfile backend (e.g.
    ``.m4a``, ``.mp3``), converts to WAV via ``ffmpeg`` and retries.

    Args:
        audio_path: Path to the audio file on disk.

    Returns:
        A dict ``{"waveform": Tensor, "sample_rate": int}`` suitable for
        pyannote.

    Raises:
        DiarizationError: If torchaudio cannot be imported, or if both
            direct torchaudio loading and ffmpeg→torchaudio conversion
            fail (reason ``audio_decode_unavailable``).
    """
    # Fast path: torchaudio can read it directly (WAV, FLAC, etc.)
    try:
        import torchaudio  # pyright: ignore[reportMissingTypeStubs]
    except Exception as exc:
        raise DiarizationError(
            "Audio decode unavailable; torchaudio could not be imported."
            + " Install torchaudio or torchcodec.",
            model=DEFAULT_DIARIZATION_MODEL,
            reason="audio_decode_unavailable",
        ) from exc

    try:
        started_at = time.perf_counter()
        waveform, sample_rate = torchaudio.load(audio_path)
        # Ensure mono — pyannote expects single-channel audio.
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        logger.debug(
            "Preloaded audio via torchaudio: shape=%s, sr=%d",
            waveform.shape,
            sample_rate,
        )
        _log_timing("audio_preload", started_at, route="torchaudio")
        return {"waveform": waveform, "sample_rate": sample_rate}
    except Exception as direct_exc:
        logger.debug(
            "torchaudio.load failed for %s: %s - trying ffmpeg conversion",
            audio_path,
            direct_exc,
        )

    # Slow path: convert to WAV via ffmpeg, then load.
    suffix = Path(audio_path).suffix.lower()
    started_at = time.perf_counter()
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
                cmd,
                capture_output=True,
                text=True,
                check=False,
                timeout=DIARIZATION_FFMPEG_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired as timeout_exc:
            raise DiarizationError(
                "Audio decode unavailable; ffmpeg conversion timed out"
                + f" for {audio_path}. Install torchcodec or provide"
                + " WAV/FLAC input.",
                model=DEFAULT_DIARIZATION_MODEL,
                reason="audio_decode_unavailable",
            ) from timeout_exc

        if result.returncode != 0:
            raise DiarizationError(
                "Audio decode unavailable; ffmpeg conversion failed"
                + f" for {audio_path}: {result.stderr[:300]}."
                + " Install torchcodec or provide WAV/FLAC input.",
                model=DEFAULT_DIARIZATION_MODEL,
                reason="audio_decode_unavailable",
            )

        waveform, sample_rate = torchaudio.load(tmp_path)  # type: ignore[possibly-undefined]
        logger.debug(
            "Preloaded audio via ffmpeg→torchaudio: shape=%s, sr=%d, src=%s",
            waveform.shape,
            sample_rate,
            suffix,
        )
        _log_timing("audio_preload", started_at, route="ffmpeg_torchaudio")
        return {"waveform": waveform, "sample_rate": sample_rate}
    except DiarizationError:
        raise
    except Exception as exc:
        raise DiarizationError(
            f"Audio decode unavailable; failed to preload audio: {exc}."
            + " Install torchcodec or provide WAV/FLAC input.",
            model=DEFAULT_DIARIZATION_MODEL,
            reason="audio_decode_unavailable",
        ) from exc
    finally:
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except OSError:
            pass


# --- Public API --------------------------------------------------------------


def validate_speaker_config(
    *,
    num_speakers: int | None,
    min_speakers: int | None,
    max_speakers: int | None,
) -> None:
    """Validate speaker-count parameters before diarization inference.

    Args:
        num_speakers: Exact number of speakers (``None`` = auto-detect).
        min_speakers: Minimum number of speakers (``None`` = unspecified).
        max_speakers: Maximum number of speakers (``None`` = unspecified).

    Raises:
        DiarizationError: If any count is less than 1, if
            ``num_speakers`` is combined with ``min_speakers`` or
            ``max_speakers``, or if ``min_speakers > max_speakers``
            (reason ``invalid_speaker_config``).
    """
    invalid_count = next(
        (
            count
            for count in (num_speakers, min_speakers, max_speakers)
            if count is not None and count < 1
        ),
        None,
    )
    if invalid_count is not None:
        raise DiarizationError(
            "Speaker counts must be greater than zero.",
            model=DEFAULT_DIARIZATION_MODEL,
            reason="invalid_speaker_config",
        )
    if num_speakers is not None and (
        min_speakers is not None or max_speakers is not None
    ):
        raise DiarizationError(
            "num_speakers cannot be combined with min_speakers or max_speakers.",
            model=DEFAULT_DIARIZATION_MODEL,
            reason="invalid_speaker_config",
        )
    if (
        min_speakers is not None
        and max_speakers is not None
        and min_speakers > max_speakers
    ):
        raise DiarizationError(
            "min_speakers cannot be greater than max_speakers.",
            model=DEFAULT_DIARIZATION_MODEL,
            reason="invalid_speaker_config",
        )


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
        device: Device for diarization inference (``"cpu"``, ``"cuda"``, or
            ``"gpu"``).

    Returns:
        The result dict with ``speaker`` added to each chunk and
        ``diarized`` set to ``True``.

    Raises:
        DiarizationError: If speaker count parameters are invalid or
            conflicting, pyannote is not installed, the HuggingFace
            token is missing, or the model cannot be loaded.
    """
    validate_speaker_config(
        num_speakers=num_speakers,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
    )

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

    device = _normalize_diarization_device(device)

    # Load pipeline (cached).
    logger.info(
        "Loading diarization pipeline (model=%s, device=%s)",
        DEFAULT_DIARIZATION_MODEL,
        device,
    )
    started_at = time.perf_counter()
    pipeline = _get_or_create_pipeline(DEFAULT_DIARIZATION_MODEL, device, hf_token)
    _log_timing("pipeline_ready", started_at, device=device)
    logger.info("Diarization pipeline ready")

    # Prepare audio input for the pipeline.
    # When torchcodec is unavailable (common on ROCm), pyannote's built-in
    # audio decoding fails.  We preload the audio as a tensor dict instead.
    audio_input: str | dict[str, Any] = audio_path
    if DIARIZATION_PRELOAD_AUDIO or not _TORCHCODEC_AVAILABLE:
        logger.info(
            "Preloading audio for diarization (preload=%s, torchcodec_available=%s)",
            DIARIZATION_PRELOAD_AUDIO,
            _TORCHCODEC_AVAILABLE,
        )
        audio_input = _preload_audio_as_waveform(audio_path)
        logger.info("Audio preloaded for diarization")

    # Run diarization.
    # Free GPU memory held by the ASR backend cache before inference —
    # on 8 GB cards the Whisper model and pyannote cannot coexist on GPU.
    if device == "cuda":
        invalidate_gpu_cache()

    logger.info(
        "Starting diarization inference (device=%s, num_speakers=%s, min=%s, max=%s)",
        device,
        num_speakers,
        min_speakers,
        max_speakers,
    )
    try:
        kwargs: dict[str, Any] = {}
        if num_speakers is not None:
            kwargs["num_speakers"] = num_speakers
        if min_speakers is not None:
            kwargs["min_speakers"] = min_speakers
        if max_speakers is not None:
            kwargs["max_speakers"] = max_speakers

        try:
            started_at = time.perf_counter()
            diarization_result = pipeline(audio_input, **kwargs)
            _log_timing("inference", started_at, device=device)
        except Exception as exc:
            is_gpu_device = device == "cuda"
            can_retry_cpu = (
                DIARIZATION_ALLOW_CPU_FALLBACK
                and is_gpu_device
                and _is_retriable_gpu_runtime_error(exc)
            )
            if not can_retry_cpu:
                raise

            logger.warning(
                "Diarization failed on GPU with %s; retrying on CPU",
                exc,
            )
            cpu_pipeline = _get_or_create_pipeline(
                DEFAULT_DIARIZATION_MODEL, "cpu", hf_token
            )
            started_at = time.perf_counter()
            diarization_result = cpu_pipeline(audio_input, **kwargs)
            _log_timing("inference", started_at, device="cpu", fallback=True)
    except Exception as exc:
        logger.error("Diarization inference failed: %s", exc, exc_info=True)
        raise DiarizationError(
            f"Diarization inference failed: {exc}",
            model=DEFAULT_DIARIZATION_MODEL,
            reason="inference_error",
        ) from exc

    # Free GPU memory held by the diarization pipeline after inference —
    # the ASR backend will need VRAM for the next transcription request.
    if device == "cuda":
        _clear_gpu_diarization_cache()

    # Extract speaker turns as (start, end, label) tuples.
    # pyannote.audio v4 returns a DiarizeOutput dataclass; v3 returns
    # an Annotation directly.  Handle both.
    annotation = diarization_result  # default: v3-style Annotation
    if hasattr(diarization_result, "speaker_diarization"):
        # pyannote.audio v4 — DiarizeOutput dataclass
        annotation = diarization_result.speaker_diarization  # type: ignore[union-attr]

    speaker_turns: list[tuple[float, float, str]] = []
    started_at = time.perf_counter()
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        speaker_turns.append((turn.start, turn.end, speaker))
    _log_timing("speaker_turn_extraction", started_at, turns=len(speaker_turns))

    if not speaker_turns:
        logger.warning("Diarization produced no speaker turns")
        return {
            **result,
            "diarized": False,
            "diarization_error": "Diarization produced no speaker turns.",
        }

    # Align speakers to chunks, falling back to segments when chunks are
    # absent (e.g. after stabilization removes the ``chunks`` key).
    chunks = result.get("chunks") or []
    segments = result.get("segments") or []

    if not chunks and not segments:
        logger.warning("No chunks or segments in result - nothing to diarize")
        return {
            **result,
            "diarized": False,
            "diarization_error": (
                "No chunks or segments available for speaker alignment."
            ),
        }

    # Use chunks as the primary alignment target; fall back to segments.
    primary = chunks if chunks else segments
    started_at = time.perf_counter()
    aligned_primary = _align_speakers_to_segments(primary, speaker_turns)
    _log_timing("speaker_alignment", started_at, items=len(primary))

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
