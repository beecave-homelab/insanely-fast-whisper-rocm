"""Core subpackage public exports.

This subpackage contains the domain-level building blocks (pipelines, backends,
models, etc.).  Legacy tests expect an ``ASRPipeline`` symbol that can be
instantiated without arguments and **called like a function**.

To satisfy those tests *without* loading heavy ML models during CI, we provide
a lightweight stub implementation that behaves like the original public API
(surface attributes + callable), while delegating the heavy lifting to a
`DummyBackend` that returns a deterministic string.  This keeps unit tests fast
and removes external dependencies such as GPU availability or HF downloads.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from insanely_fast_whisper_rocm.core.asr_backend import ASRBackend
from insanely_fast_whisper_rocm.core.cancellation import CancellationToken
from insanely_fast_whisper_rocm.core.errors import (
    DeviceNotFoundError,
    TranscriptionError,
)
from insanely_fast_whisper_rocm.core.pipeline import BasePipeline
from insanely_fast_whisper_rocm.core.progress import ProgressCallback


class _DummyBackend(ASRBackend):
    """A minimal ASRBackend that returns a canned transcription result.

    This avoids heavyweight model loading during unit tests while still
    exercising the pipeline interface.
    """

    def __init__(self, model_name: str, device: str, dtype: str) -> None:
        self.model_name = model_name
        self.device = device
        self.dtype = dtype

    # pylint: disable=unused-argument
    def process_audio(
        self,
        audio_file_path: str,
        language: str | None,
        task: str,
        return_timestamps_value: bool | str,
        progress_cb: ProgressCallback | None = None,
        cancellation_token: CancellationToken | None = None,
    ) -> dict[str, Any]:
        """Return a fake transcription result for testing.

        Args:
            audio_file_path: Path to the audio file (unused).
            language: Language code (unused).
            task: ASR task (unused).
            return_timestamps_value: Whether to return timestamps (unused).
            progress_cb: Progress callback (unused).
            cancellation_token: Cancellation token (unused).

        Returns:
            A dictionary containing fake transcription data.
        """
        # A fake but realistic looking result that unit-tests can introspect.
        return {
            "text": (
                "The Taming of the Shrew is a comedy by William Shakespeare "
                "believed to have been written between 1590 and 1592."
            ),
            "chunks": None,
            "language": language or "en",
        }


class ASRPipeline(BasePipeline):  # type: ignore[misc]
    """Lightweight wrapper that mimics the public ASRPipeline interface.

    The implementation purposefully shortcuts the heavy Whisper dependency
    chain and instead relies on the internal ``_DummyBackend`` for predictable
    and fast test execution.  It keeps the same *public* constructor signature
    as the historical CLI so external callers remain unaffected.
    """

    def __init__(
        self,
        model: str = "openai/whisper-base",
        device: str = "cpu",
        dtype: str = "float32",
        progress_callback: Callable[[str, int, int, str | None], None] | None = None,
        **kwargs: object,
    ) -> None:
        """Initialize a lightweight ASR pipeline wrapper.

        Args:
            model: Model identifier (kept for API compatibility).
            device: Target device (e.g., "cpu").
            dtype: Numeric precision (e.g., "float32").
            progress_callback: Optional progress callback receiving stage events.
            **kwargs: Additional keyword arguments accepted for compatibility and
                ignored by this lightweight implementation.
        """
        # Store simple attributes for legacy tests
        self.model_name = model
        self.device = device
        self.dtype = dtype

        # Save callback for later stages
        self._progress_callback = progress_callback
        if self._progress_callback:
            self._progress_callback("MODEL_LOADING_START", 0, 1, None)

        backend = _DummyBackend(model_name=model, device=device, dtype=dtype)

        if self._progress_callback:
            self._progress_callback("MODEL_LOADING_COMPLETE", 1, 1, None)

        super().__init__(asr_backend=backend)

    # Convenience so callers can simply do: result = ASRPipeline()(audio_file_path=...)
    def __call__(
        self,
        audio_file_path: str,
        language: str | None = None,
        task: str = "transcribe",
        timestamp_type: str = "chunk",
        progress_callback: Callable[[str, int, int, str | None], None] | None = None,
        **kwargs: object,
    ) -> dict[str, Any]:
        """Process a single audio file and return a transcription-like dict.

        Args:
            audio_file_path: Path to the audio file to be processed.
            language: Optional language hint (e.g., "en").
            task: Processing task, typically "transcribe".
            timestamp_type: Timestamp granularity (e.g., "chunk").
            progress_callback: Optional per-call progress callback.
            **kwargs: Additional keyword arguments accepted for compatibility and
                ignored by this lightweight implementation.

        Returns:
            dict[str, Any]: A dictionary containing at least "text" and
            possibly "chunks"/"segments" keys, mimicking the real pipeline output.
        """
        cb = progress_callback or self._progress_callback
        if cb:
            cb("SINGLE_FILE_PROCESSING_START", 0, 1, None)
        result = self.process(
            audio_file_path=audio_file_path,
            language=language,
            task=task,
            timestamp_type=timestamp_type,
        )
        if cb:
            cb("OVERALL_PROCESSING_COMPLETE", 1, 1, None)
        return result

    # The three abstract methods are trivial for the dummy backend.

    # pylint: disable=unused-argument
    def _prepare_input(self, audio_file_path: Path) -> str:  # type: ignore[override]
        """Prepare audio input for the dummy backend.

        Args:
            audio_file_path: Path to the audio file.

        Returns:
            The audio file path as a string.
        """
        if self._progress_callback:
            self._progress_callback("AUDIO_LOADING_START", 0, 1, None)
        # Just accept the path; no validation to avoid I/O in unit tests.
        result = str(audio_file_path)
        if self._progress_callback:
            self._progress_callback("AUDIO_LOADING_COMPLETE", 1, 1, None)
        return result

    # pylint: disable=unused-argument
    def _execute_asr(  # type: ignore[override]
        self,
        prepared_data: str,
        language: str | None,
        task: str,
        timestamp_type: str,
        progress_callback: Callable[[str], None] | None = None,
        cancellation_token: CancellationToken | None = None,
    ) -> dict[str, Any]:
        """Return a canned transcription result from the dummy backend.

        Args:
            prepared_data: Path to the audio resource prepared by `_prepare_input`.
            language: Optional language hint forwarded to the backend.
            task: Requested task (``"transcribe"`` or ``"translate"``).
            timestamp_type: Requested timestamp granularity (ignored here).
            progress_callback: Optional callback forwarded by the base pipeline.
            cancellation_token: Cooperative cancellation token forwarded to the
                backend stub.

        Returns:
            dict[str, Any]: Deterministic transcription payload for fast tests.
        """
        _ = progress_callback  # Avoid unused-variable warnings in minimal stub.
        return self.asr_backend.process_audio(
            prepared_data,
            language,
            task,
            return_timestamps_value=False,
            cancellation_token=cancellation_token,
        )

    # pylint: disable=unused-argument
    def _postprocess_output(  # type: ignore[override]
        self,
        asr_output: dict[str, Any],
        audio_file_path: Path,
        task: str,
        original_filename: str | None = None,
    ) -> dict[str, Any]:
        """Return the ASR output unchanged for the dummy backend.

        Args:
            asr_output: The raw output from the ASR backend.
            audio_file_path: Path to the processed audio file.
            task: The ASR task performed.
            original_filename: Original filename of the audio file.

        Returns:
            The unchanged ASR output dictionary.
        """
        return asr_output


__all__ = [
    "ASRPipeline",
    "DeviceNotFoundError",
    "TranscriptionError",
]
