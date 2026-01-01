"""Unit tests for the CLI facade covering backend and pipeline flows."""

from __future__ import annotations

from pathlib import Path
from typing import NoReturn
from unittest.mock import MagicMock

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
        """
        Record the invocation and return a payload reflecting the call.
        
        Returns:
            dict[str, str | None]: A dictionary with keys "source", "path", "language", "task", and "timestamps" captured from the invocation.
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
        timestamp_type: str | bool,
        original_filename: str,
        progress_callback: ProgressCallback | None = None,
        cancellation_token: CancellationToken | None = None,
    ) -> dict[str, str | None]:
        """
        Record the processing call and return a deterministic payload for assertions.
        
        Normalizes `timestamp_type` (False -> "none", True -> "chunk"; string values are unchanged) and appends a record to `self.calls`.
        
        Parameters:
            timestamp_type (str | bool): Type of timestamps requested; booleans are normalized as described.
        
        Returns:
            dict[str, str | None]: A payload containing `"source": "pipeline"` and the recorded fields
            `audio_file_path`, `language`, `task`, `timestamp_type`, and `original_filename`.
        """
        # Normalize timestamp_type for assertions
        ts_type = timestamp_type
        if ts_type is False:
            ts_type = "none"
        elif ts_type is True:
            ts_type = "chunk"

        call = {
            "audio_file_path": audio_file_path,
            "language": language,
            "task": task,
            "timestamp_type": ts_type,
            "original_filename": original_filename,
        }
        self.calls.append(call)
        return {"source": "pipeline", **call}


class ErrorPipeline(RecordingPipeline):
    """Pipeline double that raises a ``TranscriptionError``."""

    error_message = "unexpected failure"

    def process(self, **kwargs: object) -> NoReturn:  # type: ignore[override]
        """
        Simulates a pipeline failure by raising a TranscriptionError.
        
        Raises:
            TranscriptionError: Always raised with the pipeline's configured error message.
        """
        raise TranscriptionError(self.error_message)


class FallbackPipeline(RecordingPipeline):
    """Pipeline double that triggers the backend fallback path."""

    facade_ref: CLIFacade | None = None

    def process(self, **kwargs: object) -> NoReturn:  # type: ignore[override]
        """
        Trigger backend fallback by raising a TranscriptionError.
        
        If `facade_ref` is set, disables strict file existence checking on the referenced facade before raising the error.
        
        Raises:
            TranscriptionError: Always raised to force backend fallback (message: "audio file not found on disk").
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


def test_process_audio_cpu_adjusts_configuration_and_uses_orchestrator() -> None:
    """When running on CPU the facade adjusts configuration and calls orchestrator."""
    mock_orchestrator = MagicMock()
    mock_orchestrator.run_transcription.return_value = {"source": "mock"}

    facade = CLIFacade(
        orchestrator_factory=lambda: mock_orchestrator,
        check_file_exists=True,
    )

    result = facade.process_audio(
        audio_file_path=Path("sample.wav"),
        device="cpu",
        batch_size=10,
        chunk_length=30,
        return_timestamps_value="word",
    )

    assert result["source"] == "mock"
    mock_orchestrator.run_transcription.assert_called_once()
    call_args = mock_orchestrator.run_transcription.call_args[1]
    config = call_args["backend_config"]
    assert config.chunk_length == 15
    assert config.batch_size == 2
    assert call_args["timestamp_type"] == "word"


def test_process_audio_uses_orchestrator_factory_each_time() -> None:
    """Facade calls the orchestrator factory for every processing request."""
    factory_calls = 0

    def mock_factory() -> MagicMock:
        """
        Create a new MagicMock and record that the factory was invoked.
        
        Increments the enclosing scope's `factory_calls` counter each time it's called.
        
        Returns:
            MagicMock: A new MagicMock instance.
        """
        nonlocal factory_calls
        factory_calls += 1
        return MagicMock()

    facade = CLIFacade(orchestrator_factory=mock_factory, check_file_exists=True)

    facade.process_audio(audio_file_path=Path("first.wav"))
    facade.process_audio(audio_file_path=Path("second.wav"))

    assert factory_calls == 2


def test_process_audio_orchestrator_initialisation_failure_raises() -> None:
    """Facade surfaces errors when the orchestrator factory fails."""

    def broken_factory() -> None:
        """
        Factory function that always fails during initialization.
        
        Raises:
            RuntimeError: Always raised with the message "Factory failed".
        """
        raise RuntimeError("Factory failed")

    facade = CLIFacade(orchestrator_factory=broken_factory, check_file_exists=False)

    with pytest.raises(RuntimeError, match="Factory failed"):
        facade.process_audio(audio_file_path=Path("broken.wav"))


def test_process_audio_orchestrator_error_is_propagated() -> None:
    """Errors raised by the orchestrator are propagated."""
    mock_orchestrator = MagicMock()
    mock_orchestrator.run_transcription.side_effect = TranscriptionError("orch failure")

    facade = CLIFacade(
        orchestrator_factory=lambda: mock_orchestrator,
        check_file_exists=True,
    )

    with pytest.raises(TranscriptionError, match="orch failure"):
        facade.process_audio(audio_file_path=Path("failure.wav"))