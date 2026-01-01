"""Unit tests for the WebUI handler functions."""

from collections.abc import Generator
from unittest.mock import MagicMock, patch

import pytest

from insanely_fast_whisper_rocm.webui.handlers import TranscriptionConfig, transcribe


@pytest.fixture
def mock_pipeline_and_stabilizer() -> Generator[
    tuple[MagicMock, MagicMock], None, None
]:
    """
    Provide a pytest fixture that patches orchestrator creation and the timestamp stabilizer.
    
    Patches `create_orchestrator` and `stabilize_timestamps` in the handlers module, yields a mocked orchestrator whose `run_transcription` returns `{"text": "test transcription"}`, and a mocked `stabilize_timestamps` that returns its input.
    
    Returns:
        tuple[MagicMock, MagicMock]: A tuple of `(mock_orchestrator, mock_stabilize)` where `mock_orchestrator.run_transcription()` returns `{"text": "test transcription"}` and `mock_stabilize(result, **kwargs)` returns `result`.
    """
    with (
        patch(
            "insanely_fast_whisper_rocm.webui.handlers.create_orchestrator"
        ) as mock_create_orchestrator,
        patch(
            "insanely_fast_whisper_rocm.webui.handlers.stabilize_timestamps"
        ) as mock_stabilize,
    ):
        mock_orchestrator = MagicMock()
        mock_orchestrator.run_transcription.return_value = {
            "text": "test transcription"
        }
        mock_create_orchestrator.return_value = mock_orchestrator

        # Make the mocked stabilize_timestamps function return its input
        mock_stabilize.side_effect = lambda result, **kwargs: result

        yield mock_orchestrator, mock_stabilize


def test_transcribe_handler_with_stabilization(
    mock_pipeline_and_stabilizer: tuple[MagicMock, MagicMock],
) -> None:
    """Test that the transcribe handler calls stabilize_timestamps with correct args."""
    _mock_pipeline, mock_stabilize = mock_pipeline_and_stabilizer

    config = TranscriptionConfig(
        stabilize=True, demucs=True, vad=True, vad_threshold=0.6
    )

    transcribe(audio_file_path="dummy.wav", config=config, file_config=MagicMock())

    mock_stabilize.assert_called_once()
    _call_args, call_kwargs = mock_stabilize.call_args
    assert call_kwargs.get("demucs") is True
    assert call_kwargs.get("vad") is True
    assert call_kwargs.get("vad_threshold") == 0.6


def test_transcribe_handler_without_stabilization(
    mock_pipeline_and_stabilizer: tuple[MagicMock, MagicMock],
) -> None:
    """Test that stabilize_timestamps is not called when stabilization is disabled."""
    _mock_pipeline, mock_stabilize = mock_pipeline_and_stabilizer

    config = TranscriptionConfig(stabilize=False)

    transcribe(audio_file_path="dummy.wav", config=config, file_config=MagicMock())

    mock_stabilize.assert_not_called()