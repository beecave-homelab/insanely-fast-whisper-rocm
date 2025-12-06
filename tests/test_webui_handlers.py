"""Unit tests for the WebUI handler functions."""

from unittest.mock import MagicMock, patch

import pytest

from insanely_fast_whisper_rocm.webui.handlers import (
    TranscriptionConfig,
    transcribe,
)


@pytest.fixture
def mock_pipeline_and_stabilizer():
    """
    Provide a pytest fixture that mocks WhisperPipeline and stabilize_timestamps for handler tests.
    
    The mocked WhisperPipeline instance has its `process` method stubbed to return {"text": "test transcription"}. The mocked `stabilize_timestamps` function is configured to return its input unchanged.
    
    Returns:
        tuple: (mock_pipeline_instance, mock_stabilize) where `mock_pipeline_instance` is the mocked WhisperPipeline instance and `mock_stabilize` is the mocked stabilize_timestamps function.
    """
    with (
        patch(
            "insanely_fast_whisper_rocm.webui.handlers.WhisperPipeline"
        ) as mock_pipeline_class,
        patch(
            "insanely_fast_whisper_rocm.webui.handlers.stabilize_timestamps"
        ) as mock_stabilize,
    ):
        # Mock the pipeline's process method to return a dummy result
        mock_pipeline_instance = MagicMock()
        mock_pipeline_instance.process.return_value = {"text": "test transcription"}
        mock_pipeline_class.return_value = mock_pipeline_instance

        # Make the mocked stabilize_timestamps function return its input
        mock_stabilize.side_effect = lambda result, **kwargs: result

        yield mock_pipeline_instance, mock_stabilize


def test_transcribe_handler_with_stabilization(mock_pipeline_and_stabilizer):
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


def test_transcribe_handler_without_stabilization(mock_pipeline_and_stabilizer):
    """Test that stabilize_timestamps is not called when stabilization is disabled."""
    _mock_pipeline, mock_stabilize = mock_pipeline_and_stabilizer

    config = TranscriptionConfig(stabilize=False)

    transcribe(audio_file_path="dummy.wav", config=config, file_config=MagicMock())

    mock_stabilize.assert_not_called()