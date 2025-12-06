"""Unit tests for the WebUI handler functions."""

import contextlib
from collections.abc import Generator
from unittest.mock import MagicMock, patch

import pytest

from insanely_fast_whisper_rocm.webui.handlers import TranscriptionConfig, transcribe


@pytest.fixture
def mock_pipeline_and_stabilizer() -> Generator[
    tuple[MagicMock, MagicMock], None, None
]:
    """Fixture to mock borrow_pipeline and ``stabilize_timestamps``.

    Yields:
        tuple[MagicMock, MagicMock]: A tuple of (mocked pipeline instance,
        mocked ``stabilize_timestamps`` function).
    """
    with (
        patch(
            "insanely_fast_whisper_rocm.webui.handlers.borrow_pipeline"
        ) as mock_borrow,
        patch(
            "insanely_fast_whisper_rocm.webui.handlers.stabilize_timestamps"
        ) as mock_stabilize,
    ):
        # Mock the pipeline's process method to return a dummy result
        mock_pipeline_instance = MagicMock()
        mock_pipeline_instance.process.return_value = {"text": "test transcription"}
        mock_borrow.return_value = contextlib.nullcontext(mock_pipeline_instance)

        # Make the mocked stabilize_timestamps function return its input
        mock_stabilize.side_effect = lambda result, **kwargs: result

        yield mock_pipeline_instance, mock_stabilize


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
