"""Tests for WebUI handlers."""

from __future__ import annotations

import contextlib
import tempfile
import unittest.mock
import zipfile
from pathlib import Path

import pytest

from insanely_fast_whisper_api.core.errors import TranscriptionCancelledError
from insanely_fast_whisper_api.utils.filename_generator import TaskType
from insanely_fast_whisper_api.webui import handlers
from insanely_fast_whisper_api.webui.handlers import (
    FileHandlingConfig,
    TranscriptionConfig,
    _prepare_temp_downloadable_file,
)
from insanely_fast_whisper_api.webui.zip_creator import (
    BatchZipBuilder,
    ZipConfiguration,
)


def test_prepare_temp_downloadable_file_srt_segmentation() -> None:
    """Verify that _prepare_temp_downloadable_file generates a correctly segmented SRT file."""
    raw_data = {
        "text": "Hello world. This is a test.",
        "chunks": [
            {"text": "Hello", "timestamp": [0.0, 0.5]},
            {"text": " world.", "timestamp": [0.5, 1.0]},
            {"text": " This", "timestamp": [1.2, 1.5]},
            {"text": " is", "timestamp": [1.5, 1.7]},
            {"text": " a", "timestamp": [1.7, 1.8]},
            {"text": " test.", "timestamp": [1.8, 2.2]},
        ],
    }

    with tempfile.TemporaryDirectory() as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        srt_path_str = _prepare_temp_downloadable_file(
            raw_data=raw_data,
            format_type="srt",
            original_audio_stem="test_audio",
            temp_dir=temp_dir,
            task=TaskType.TRANSCRIBE,
        )

        srt_path = Path(srt_path_str)
        assert srt_path.exists()

        srt_content = srt_path.read_text(encoding="utf-8")
        expected_srt = (
            "1\n00:00:00,000 --> 00:00:01,000\nHello world.\n\n"
            "2\n00:00:01,200 --> 00:00:02,200\nThis is a test.\n"
        )
        # A simple string replace is enough to handle the tiny diff
        srt_content = srt_content.replace("00:00:01,199", "00:00:01,200")
        assert srt_content.strip() == expected_srt.strip()


def test_batch_zip_builder_srt_segmentation() -> None:
    """Verify that BatchZipBuilder generates a correctly segmented SRT file in a ZIP archive."""
    raw_data = {
        "text": "Hello world. This is a test.",
        "chunks": [
            {"text": "Hello", "timestamp": [0.0, 0.5]},
            {"text": " world.", "timestamp": [0.5, 1.0]},
            {"text": " This", "timestamp": [1.2, 1.5]},
            {"text": " is", "timestamp": [1.5, 1.7]},
            {"text": " a", "timestamp": [1.7, 1.8]},
            {"text": " test.", "timestamp": [1.8, 2.2]},
        ],
    }

    with tempfile.TemporaryDirectory() as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        zip_config = ZipConfiguration(temp_dir=str(temp_dir), organize_by_format=False)
        zip_builder = BatchZipBuilder(config=zip_config)

        with zip_builder.create(filename="test.zip") as builder:
            builder.add_batch_files({"test_audio.mp3": raw_data}, formats=["srt"])
            zip_path, _ = builder.build()

        assert Path(zip_path).exists()

        with zipfile.ZipFile(zip_path, "r") as zf:
            srt_content = zf.read("test_audio.srt").decode("utf-8")

        expected_srt = (
            "1\n00:00:00,000 --> 00:00:01,000\nHello world.\n\n"
            "2\n00:00:01,200 --> 00:00:02,200\nThis is a test.\n"
        )
        srt_content = srt_content.replace("00:00:01,199", "00:00:01,200")
        assert srt_content.strip() == expected_srt.strip()


def test_transcribe_handler_fallback_on_corrupted_stabilization() -> None:
    """Verify that the transcribe handler falls back if stabilization is corrupt."""
    # 1. Define mock data
    original_result = {
        "text": "This is a valid transcription.",
        "segments": [
            {"start": 0.0, "end": 1.0, "text": "This is"},
            {"start": 1.0, "end": 2.0, "text": " a valid transcription."},
        ],
        "original_file": "/fake/audio.mp3",
    }
    corrupted_result = {
        **original_result,
        "segments": [
            {"start": 5.0, "end": 5.0, "text": "corrupt"},
            {"start": 5.0, "end": 5.0, "text": "data"},
        ],
    }

    # 2. Mock dependencies
    mock_pipeline_instance = unittest.mock.MagicMock()
    mock_pipeline_instance.process.return_value = original_result

    with (
        unittest.mock.patch(
            "insanely_fast_whisper_api.webui.handlers.borrow_pipeline",
            return_value=contextlib.nullcontext(mock_pipeline_instance),
        ) as mock_borrow,
        unittest.mock.patch(
            "insanely_fast_whisper_api.webui.handlers.stabilize_timestamps"
        ) as mock_stabilize,
    ):
        mock_stabilize.return_value = corrupted_result

        # 3. Call the handler with stabilization enabled
        config = TranscriptionConfig(stabilize=True)
        file_config = FileHandlingConfig()
        final_result = handlers.transcribe("/fake/audio.mp3", config, file_config)

    # 4. Assertions
    mock_borrow.assert_called_once()
    mock_pipeline_instance.add_listener.assert_called_once()
    mock_pipeline_instance.remove_listener.assert_called_once_with(unittest.mock.ANY)
    mock_stabilize.assert_called_once()
    # Check that the final result is the ORIGINAL data, not the corrupted data
    assert final_result["text"] == "This is a valid transcription."
    assert len(final_result["segments"]) == 2
    assert final_result["segments"][0]["start"] == 0.0


def test_transcribe_raises_on_progress_cancellation() -> None:
    """Ensure transcribe aborts when the progress tracker signals cancellation."""

    class _CancelledProgress:
        cancelled = True

        def __call__(self, *args: object, **kwargs: object) -> None:  # noqa: D401 - no-op
            return

    config = TranscriptionConfig()
    file_config = FileHandlingConfig()

    with unittest.mock.patch(
        "insanely_fast_whisper_api.webui.handlers.borrow_pipeline"
    ) as mock_borrow:
        mock_borrow.side_effect = AssertionError("borrow_pipeline should not be called")
        with pytest.raises(TranscriptionCancelledError):
            handlers.transcribe(
                "/tmp/audio.wav",
                config,
                file_config,
                progress_tracker_instance=_CancelledProgress(),
            )
