"""Unit tests for the CLI facade covering backend and pipeline flows."""

from __future__ import annotations

from pathlib import Path
from typing import NoReturn

import pytest

from insanely_fast_whisper_rocm.cli.facade import CLIFacade
from insanely_fast_whisper_rocm.core.asr_backend import HuggingFaceBackendConfig
from insanely_fast_whisper_rocm.core.cancellation import CancellationToken
from insanely_fast_whisper_rocm.core.errors import TranscriptionError
from insanely_fast_whisper_rocm.core.progress import ProgressCallback


class RecordingBackend:
    """Test double that records calls to ``process_audio``."""

    instances: list[RecordingBackend] = []

    def __init__(self, config: HuggingFaceBackendConfig) -> None:
        """Store config passed by the facade for later inspection."""
        self.config = config
        self.calls: list[dict[str, str | None]] = []
        RecordingBackend.instances.append(self)

    def process_audio(
        self,
        *,
        audio_file_path: str,
        language: str | None,
        task: str,
        return_timestamps_value: bool | str,
        progress_cb: ProgressCallback | None = None,
        cancellation_token: CancellationToken | None = None,
    ) -> dict[str, str | None]:
        """Record the invocation and emit a predictable payload.

        Returns:
            dict[str, str | None]: Context captured from the invocation.
        """
        payload = {
            "source": "backend",
            "path": audio_file_path,
            "language": language,
            "task": task,
            "timestamps": return_timestamps_value,
        }
        self.calls.append(payload)
        return payload


class RecordingPipeline:
    """Test double mirroring the pipeline API."""

    def __init__(
        self,
        *,
        asr_backend: RecordingBackend,
        storage_backend: object,
        save_transcriptions: bool,
    ) -> None:
        """Capture constructor arguments and initialise call storage."""
        self.asr_backend = asr_backend
        self.storage_backend = storage_backend
        self.save_transcriptions = save_transcriptions
        self.calls: list[dict[str, str | None]] = []

    def process(
        self,
        *,
        audio_file_path: str,
        language: str | None,
        task: str,
        timestamp_type: str,
        original_filename: str,
        progress_callback: ProgressCallback | None = None,
        cancellation_token: CancellationToken | None = None,
    ) -> dict[str, str | None]:
        """Record processing calls and return a deterministic payload.

        Returns:
            dict[str, str | None]: Details captured for assertion.
        """
        call = {
            "audio_file_path": audio_file_path,
            "language": language,
            "task": task,
            "timestamp_type": timestamp_type,
            "original_filename": original_filename,
        }
        self.calls.append(call)
        return {"source": "pipeline", **call}


class ErrorPipeline(RecordingPipeline):
    """Pipeline double that raises a ``TranscriptionError``."""

    error_message = "unexpected failure"

    def process(self, **kwargs: object) -> NoReturn:  # type: ignore[override]
        """Raise a ``TranscriptionError`` to simulate pipeline errors.

        Raises:
            TranscriptionError: Always raised to exercise error handling.
        """
        raise TranscriptionError(self.error_message)


class FallbackPipeline(RecordingPipeline):
    """Pipeline double that triggers the backend fallback path."""

    facade_ref: CLIFacade | None = None

    def process(self, **kwargs: object) -> NoReturn:  # type: ignore[override]
        """Force the facade to fall back to backend processing.

        Raises:
            TranscriptionError: Raised to trigger backend fallback logic.
        """
        if self.facade_ref is not None:
            # Simulate late detection that skips strict file checking.
            self.facade_ref.check_file_exists = False
        raise TranscriptionError("audio file not found on disk")


class NoneBackend:
    """Backend double that fails to initialise properly."""

    def __new__(cls, config: HuggingFaceBackendConfig) -> None:
        """Return ``None`` to mimic backend construction failure."""
        return None


@pytest.fixture(autouse=True)
def _reset_recording_backend() -> None:
    """Reset shared state between tests."""
    RecordingBackend.instances.clear()


def test_process_audio_cpu_adjusts_configuration_and_uses_pipeline() -> None:
    """When running on CPU the facade adjusts configuration and calls pipeline."""
    facade = CLIFacade(
        backend_factory=RecordingBackend,
        pipeline_factory=RecordingPipeline,
        check_file_exists=True,
    )

    result = facade.process_audio(
        audio_file_path=Path("sample.wav"),
        device="cpu",
        batch_size=10,
        chunk_length=30,
        return_timestamps_value="word",
    )

    backend = facade.backend
    assert backend is not None
    assert backend.config.chunk_length == 15
    assert backend.config.batch_size == 2

    pipeline = facade.pipeline
    assert pipeline is not None
    assert pipeline.calls[0]["timestamp_type"] == "word"
    assert result["source"] == "pipeline"


def test_process_audio_reuses_backend_and_recreates_pipeline() -> None:
    """Facade reuses the backend config and rebuilds a missing pipeline."""
    facade = CLIFacade(
        backend_factory=RecordingBackend,
        pipeline_factory=RecordingPipeline,
        check_file_exists=True,
    )

    facade.process_audio(audio_file_path=Path("first.wav"))

    backend = facade.backend
    assert backend is not None
    assert len(RecordingBackend.instances) == 1

    facade.pipeline = None

    result = facade.process_audio(
        audio_file_path=Path("second.wav"),
        return_timestamps_value=False,
    )

    assert len(RecordingBackend.instances) == 1
    assert facade.pipeline is not None
    assert facade.pipeline.calls[-1]["timestamp_type"] == "none"
    assert result["source"] == "pipeline"


def test_process_audio_backend_initialisation_failure_raises() -> None:
    """Facade surfaces a runtime error when the backend fails to initialise."""
    facade = CLIFacade(backend_factory=NoneBackend, check_file_exists=False)

    with pytest.raises(RuntimeError, match="ASR backend failed to initialize"):
        facade.process_audio(audio_file_path=Path("broken.wav"))


def test_process_audio_pipeline_error_is_propagated_for_unhandled_failure() -> None:
    """Non-missing-file errors raised by the pipeline are propagated."""
    facade = CLIFacade(
        backend_factory=RecordingBackend,
        pipeline_factory=ErrorPipeline,
        check_file_exists=True,
    )

    with pytest.raises(TranscriptionError, match=ErrorPipeline.error_message):
        facade.process_audio(audio_file_path=Path("failure.wav"))


def test_process_audio_missing_file_falls_back_to_backend_when_available() -> None:
    """When file handling is relaxed the facade falls back to backend processing."""
    facade = CLIFacade(
        backend_factory=RecordingBackend,
        pipeline_factory=FallbackPipeline,
        check_file_exists=True,
    )
    FallbackPipeline.facade_ref = facade

    result = facade.process_audio(audio_file_path=Path("missing.wav"))

    backend = facade.backend
    assert backend is not None
    assert backend.calls[-1]["path"].endswith("missing.wav")
    assert result["source"] == "backend"

    # Restore facade behaviour for subsequent tests
    facade.check_file_exists = True
