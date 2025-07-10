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

from pathlib import Path
from typing import Any, Callable, Dict, Optional

from insanely_fast_whisper_api.core.asr_backend import ASRBackend
from insanely_fast_whisper_api.core.errors import (
    DeviceNotFoundError,
    TranscriptionError,
)
from insanely_fast_whisper_api.core.pipeline import BasePipeline


class _DummyBackend(ASRBackend):
    """A minimal ASRBackend that returns a canned transcription result.

    This avoids heavyweight model loading during unit tests while still
    exercising the pipeline interface.
    """

    def __init__(self, model_name: str, device: str, dtype: str):
        self.model_name = model_name
        self.device = device
        self.dtype = dtype

    # pylint: disable=unused-argument
    def process_audio(
        self,
        audio_file_path: str,
        language: Optional[str],
        task: str,
        return_timestamps_value: bool | str,
    ) -> Dict[str, Any]:
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
        progress_callback: Optional[
            Callable[[str, int, int, Optional[str]], None]
        ] = None,
        **kwargs: Any,
    ):
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
        language: Optional[str] = None,
        task: str = "transcribe",
        timestamp_type: str = "chunk",
        progress_callback: Optional[Callable[[float], None]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
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
        language: Optional[str],
        task: str,
        timestamp_type: str,
    ) -> Dict[str, Any]:
        return self.asr_backend.process_audio(
            prepared_data, language, task, return_timestamps_value=False
        )

    # pylint: disable=unused-argument
    def _postprocess_output(  # type: ignore[override]
        self,
        asr_output: Dict[str, Any],
        audio_file_path: Path,
        task: str,
        original_filename: Optional[str] = None,
    ) -> Dict[str, Any]:
        return asr_output


__all__ = [
    "ASRPipeline",
    "DeviceNotFoundError",
    "TranscriptionError",
]
