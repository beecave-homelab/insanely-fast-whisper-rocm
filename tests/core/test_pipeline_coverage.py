"""Tests for core pipeline coverage improvements."""

from __future__ import annotations

import pathlib
from unittest.mock import MagicMock, patch

import pytest

from insanely_fast_whisper_api.core.asr_backend import ASRBackend
from insanely_fast_whisper_api.core.errors import TranscriptionError
from insanely_fast_whisper_api.core.pipeline import (
    ProgressEvent,
    WhisperPipeline,
)
from insanely_fast_whisper_api.core.progress import NoOpProgress
from insanely_fast_whisper_api.core.storage import BaseStorage


def test_pipeline_add_listener() -> None:
    """Test add_listener registers progress callbacks.

    Covers line 101.
    """
    mock_backend = MagicMock(spec=ASRBackend)
    pipeline = WhisperPipeline(
        asr_backend=mock_backend,
        storage_backend=None,
        save_transcriptions=False,
    )

    listener_called = []

    def test_listener(event: ProgressEvent) -> None:
        listener_called.append(event)

    pipeline.add_listener(test_listener)
    assert len(pipeline._listeners) == 1

    # Trigger a notification
    test_event = ProgressEvent(
        event_type="test",
        pipeline_id="test-id",
        file_path="test.wav",
    )
    pipeline._notify_listeners(test_event)
    assert len(listener_called) == 1
    assert listener_called[0] == test_event


def test_pipeline_notify_listeners_handles_errors() -> None:
    """Test _notify_listeners catches and logs listener errors.

    Covers lines 106-109.
    """
    mock_backend = MagicMock(spec=ASRBackend)
    pipeline = WhisperPipeline(
        asr_backend=mock_backend,
        storage_backend=None,
        save_transcriptions=False,
    )

    def failing_listener(event: ProgressEvent) -> None:
        raise RuntimeError("Listener error")

    def working_listener(event: ProgressEvent) -> None:
        working_listener.called = True  # type: ignore[attr-defined]

    working_listener.called = False  # type: ignore[attr-defined]

    pipeline.add_listener(failing_listener)
    pipeline.add_listener(working_listener)

    test_event = ProgressEvent(
        event_type="test",
        pipeline_id="test-id",
        file_path="test.wav",
    )

    # Should not raise, but log error
    pipeline._notify_listeners(test_event)

    # Working listener should still be called
    assert working_listener.called  # type: ignore[attr-defined]


def test_pipeline_save_result_adds_output_file_path(tmp_path: pathlib.Path) -> None:
    """Test that saved file path is added to result.

    Covers lines 163-168.
    """
    audio_file = tmp_path / "test.wav"
    audio_file.write_text("fake audio")

    mock_backend = MagicMock(spec=ASRBackend)
    mock_backend.config = MagicMock(chunk_length=30)

    mock_storage = MagicMock(spec=BaseStorage)
    saved_path = str(tmp_path / "output.json")
    mock_storage.save.return_value = saved_path

    pipeline = WhisperPipeline(
        asr_backend=mock_backend,
        storage_backend=mock_storage,
        save_transcriptions=True,
        output_dir=str(tmp_path),
    )

    # Mock the internal methods
    with (
        patch.object(pipeline, "_prepare_input", return_value=str(audio_file)),
        patch.object(
            pipeline,
            "_execute_asr",
            return_value={"text": "test", "chunks": []},
        ),
        patch.object(
            pipeline,
            "_postprocess_output",
            return_value={"text": "test", "task_type": "transcribe"},
        ),
    ):
        result = pipeline.process(
            audio_file_path=str(audio_file),
            language="en",
            task="transcribe",
            timestamp_type="chunk",
        )

    assert "output_file_path" in result
    assert result["output_file_path"] == saved_path


def test_pipeline_process_error_handling(tmp_path: pathlib.Path) -> None:
    """Test error handling during pipeline execution.

    Covers lines 184-220.
    """
    audio_file = tmp_path / "test.wav"
    audio_file.write_text("fake audio")

    mock_backend = MagicMock(spec=ASRBackend)

    pipeline = WhisperPipeline(
        asr_backend=mock_backend,
        storage_backend=None,
        save_transcriptions=False,
    )

    # Mock _prepare_input to raise an error
    with patch.object(
        pipeline, "_prepare_input", side_effect=RuntimeError("Test error")
    ):
        with pytest.raises(TranscriptionError, match="Pipeline failed"):
            pipeline.process(
                audio_file_path=str(audio_file),
                language="en",
                task="transcribe",
                timestamp_type="chunk",
            )


def test_pipeline_process_transcription_error_reraise(tmp_path: pathlib.Path) -> None:
    """Test that TranscriptionError is re-raised directly.

    Covers lines 206-220.
    """
    audio_file = tmp_path / "test.wav"
    audio_file.write_text("fake audio")

    mock_backend = MagicMock(spec=ASRBackend)

    pipeline = WhisperPipeline(
        asr_backend=mock_backend,
        storage_backend=None,
        save_transcriptions=False,
    )

    # Mock _prepare_input to raise TranscriptionError
    with patch.object(
        pipeline,
        "_prepare_input",
        side_effect=TranscriptionError("ASR failed"),
    ):
        with pytest.raises(TranscriptionError, match="ASR failed"):
            pipeline.process(
                audio_file_path=str(audio_file),
                language="en",
                task="transcribe",
                timestamp_type="chunk",
            )


def test_save_result_no_storage_backend() -> None:
    """Test _save_result warns when no storage backend is configured.

    Covers lines 274-275.
    """
    mock_backend = MagicMock(spec=ASRBackend)
    pipeline = WhisperPipeline(
        asr_backend=mock_backend,
        storage_backend=None,
        save_transcriptions=False,
    )

    # Temporarily set storage_backend to None to trigger the warning
    pipeline.storage_backend = None

    result = pipeline._save_result(
        result={"text": "test"},
        audio_file_path=pathlib.Path("test.wav"),
        task="transcribe",
    )

    assert result is None


def test_save_result_invalid_task_type(tmp_path: pathlib.Path) -> None:
    """Test _save_result handles invalid task type gracefully.

    Covers lines 279-287.
    """
    mock_backend = MagicMock(spec=ASRBackend)
    mock_storage = MagicMock(spec=BaseStorage)
    mock_storage.save.return_value = str(tmp_path / "output.json")

    pipeline = WhisperPipeline(
        asr_backend=mock_backend,
        storage_backend=mock_storage,
        save_transcriptions=True,
    )

    # Use an invalid task type
    result = pipeline._save_result(
        result={"text": "test"},
        audio_file_path=pathlib.Path("test.wav"),
        task="invalid_task",
    )

    # Should still save with fallback to TRANSCRIBE
    assert result is not None
    mock_storage.save.assert_called_once()


def test_save_result_handles_save_errors(tmp_path: pathlib.Path) -> None:
    """Test _save_result handles storage errors gracefully.

    Covers lines 325-334.
    """
    mock_backend = MagicMock(spec=ASRBackend)
    mock_storage = MagicMock(spec=BaseStorage)
    mock_storage.save.side_effect = OSError("Disk full")

    pipeline = WhisperPipeline(
        asr_backend=mock_backend,
        storage_backend=mock_storage,
        save_transcriptions=True,
    )

    result = pipeline._save_result(
        result={"text": "test"},
        audio_file_path=pathlib.Path("test.wav"),
        task="transcribe",
    )

    # Should return None on error, not raise
    assert result is None


def test_prepare_input_file_not_found() -> None:
    """Test _prepare_input raises FileNotFoundError for missing files.

    Covers line 357.
    """
    mock_backend = MagicMock(spec=ASRBackend)
    pipeline = WhisperPipeline(
        asr_backend=mock_backend,
        storage_backend=None,
        save_transcriptions=False,
    )

    # Temporarily disable SKIP_FS_CHECKS to test file not found error
    from insanely_fast_whisper_api.utils import constants

    original_skip = constants.SKIP_FS_CHECKS
    try:
        constants.SKIP_FS_CHECKS = False
        with pytest.raises(FileNotFoundError, match="Audio file not found"):
            pipeline._prepare_input(pathlib.Path("/nonexistent/file.wav"))
    finally:
        constants.SKIP_FS_CHECKS = original_skip


def test_execute_asr_timestamp_type_else_case(tmp_path: pathlib.Path) -> None:
    """Test _execute_asr with unknown timestamp_type defaults to False.

    Covers line 391.
    """
    audio_file = tmp_path / "test.wav"
    audio_file.write_text("fake audio")

    mock_backend = MagicMock(spec=ASRBackend)
    mock_backend.config = MagicMock(chunk_length=30)
    mock_backend.process_audio.return_value = {
        "text": "test",
        "chunks": [],
    }

    pipeline = WhisperPipeline(
        asr_backend=mock_backend,
        storage_backend=None,
        save_transcriptions=False,
    )

    with (
        patch(
            "insanely_fast_whisper_api.core.pipeline.audio_conversion.ensure_wav",
            return_value=str(audio_file),
        ),
        patch(
            "insanely_fast_whisper_api.core.pipeline.audio_processing.split_audio",
            return_value=[(str(audio_file), 0.0)],
        ),
    ):
        result = pipeline._execute_asr(
            prepared_data=str(audio_file),
            language="en",
            task="transcribe",
            timestamp_type="unknown",  # Unknown type
            progress_callback=NoOpProgress(),
        )

    # Verify return_timestamps_value was False (default)
    call_kwargs = mock_backend.process_audio.call_args[1]
    assert call_kwargs["return_timestamps_value"] is False
    assert result == {"text": "test", "chunks": []}


def test_execute_asr_no_chunks_error(tmp_path: pathlib.Path) -> None:
    """Test _execute_asr raises error when no chunks are produced.

    Covers line 410.
    """
    audio_file = tmp_path / "test.wav"
    audio_file.write_text("fake audio")

    mock_backend = MagicMock(spec=ASRBackend)
    mock_backend.config = MagicMock(chunk_length=30)

    pipeline = WhisperPipeline(
        asr_backend=mock_backend,
        storage_backend=None,
        save_transcriptions=False,
    )

    with (
        patch(
            "insanely_fast_whisper_api.core.pipeline.audio_conversion.ensure_wav",
            return_value=str(audio_file),
        ),
        patch(
            "insanely_fast_whisper_api.core.pipeline.audio_processing.split_audio",
            return_value=[],  # No chunks
        ),
    ):
        with pytest.raises(TranscriptionError, match="No audio chunks produced"):
            pipeline._execute_asr(
                prepared_data=str(audio_file),
                language="en",
                task="transcribe",
                timestamp_type="chunk",
                progress_callback=NoOpProgress(),
            )
