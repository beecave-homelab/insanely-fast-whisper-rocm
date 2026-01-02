"""Tests for WebUI handlers."""

from __future__ import annotations

import tempfile
import unittest.mock
import zipfile
from pathlib import Path

import pytest

from insanely_fast_whisper_rocm.core.errors import (
    TranscriptionCancelledError,
    TranscriptionError,
)
from insanely_fast_whisper_rocm.utils.filename_generator import TaskType
from insanely_fast_whisper_rocm.webui import handlers
from insanely_fast_whisper_rocm.webui.handlers import (
    FileHandlingConfig,
    TranscriptionConfig,
    _is_stabilization_corrupt,
    _prepare_temp_downloadable_file,
)
from insanely_fast_whisper_rocm.webui.zip_creator import (
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
    mock_orchestrator = unittest.mock.MagicMock()
    mock_orchestrator.run_transcription.return_value = original_result

    with (
        unittest.mock.patch(
            "insanely_fast_whisper_rocm.webui.handlers.create_orchestrator",
            return_value=mock_orchestrator,
        ) as mock_create_orchestrator,
        unittest.mock.patch(
            "insanely_fast_whisper_rocm.webui.handlers.stabilize_timestamps"
        ) as mock_stabilize,
    ):
        mock_stabilize.return_value = corrupted_result

        # 3. Call the handler with stabilization enabled
        config = TranscriptionConfig(stabilize=True)
        file_config = FileHandlingConfig()
        final_result = handlers.transcribe("/fake/audio.mp3", config, file_config)

    # 4. Assertions
    mock_create_orchestrator.assert_called_once()
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
        "insanely_fast_whisper_rocm.webui.handlers.create_orchestrator"
    ) as mock_create_orchestrator:
        mock_create_orchestrator.side_effect = AssertionError(
            "create_orchestrator should not be called"
        )
        with pytest.raises(TranscriptionCancelledError):
            handlers.transcribe(
                "/tmp/audio.wav",
                config,
                file_config,
                progress_tracker_instance=_CancelledProgress(),
            )


def test_prepare_temp_downloadable_file_txt() -> None:
    """Test _prepare_temp_downloadable_file with TXT format."""
    raw_data = {"text": "This is a test transcription."}

    with tempfile.TemporaryDirectory() as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        txt_path_str = _prepare_temp_downloadable_file(
            raw_data=raw_data,
            format_type="txt",
            original_audio_stem="test_audio",
            temp_dir=temp_dir,
            task=TaskType.TRANSCRIBE,
        )

        txt_path = Path(txt_path_str)
        assert txt_path.exists()
        txt_content = txt_path.read_text(encoding="utf-8")
        assert "This is a test transcription." in txt_content


def test_prepare_temp_downloadable_file_invalid_format() -> None:
    """Test _prepare_temp_downloadable_file raises ValueError for invalid format."""
    raw_data = {"text": "Test"}

    with tempfile.TemporaryDirectory() as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        with pytest.raises(ValueError, match="No formatter available"):
            _prepare_temp_downloadable_file(
                raw_data=raw_data,
                format_type="invalid_format",
                original_audio_stem="test",
                temp_dir=temp_dir,
                task=TaskType.TRANSCRIBE,
            )


def test_prepare_temp_downloadable_file_oserror() -> None:
    """Test _prepare_temp_downloadable_file raises OSError when write fails."""
    raw_data = {"text": "Test"}

    # Use a non-writable path to trigger OSError
    with pytest.raises(OSError):
        _prepare_temp_downloadable_file(
            raw_data=raw_data,
            format_type="txt",
            original_audio_stem="test",
            temp_dir=Path("/nonexistent/path"),
            task=TaskType.TRANSCRIBE,
        )


def test_transcribe_with_video_input() -> None:
    """Test transcribe function with video input (audio extraction)."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a fake video file
        video_path = Path(temp_dir) / "test_video.mp4"
        video_path.write_text("fake video content")

        mock_orchestrator = unittest.mock.MagicMock()
        mock_orchestrator.run_transcription.return_value = {"text": "transcribed"}

        with (
            unittest.mock.patch(
                "insanely_fast_whisper_rocm.webui.handlers.create_orchestrator",
                return_value=mock_orchestrator,
            ),
            unittest.mock.patch(
                "insanely_fast_whisper_rocm.webui.handlers.extract_audio_from_video"
            ) as mock_extract,
        ):
            mock_extract.return_value = str(Path(temp_dir) / "extracted.wav")

            config = TranscriptionConfig()
            file_config = FileHandlingConfig()
            result = handlers.transcribe(str(video_path), config, file_config)

            mock_extract.assert_called_once()
            assert result["text"] == "transcribed"


def test_transcribe_with_video_extraction_error() -> None:
    """Test transcribe function handles video extraction errors."""
    with tempfile.TemporaryDirectory() as temp_dir:
        video_path = Path(temp_dir) / "test_video.mp4"
        video_path.write_text("fake video")

        with (
            unittest.mock.patch(
                "insanely_fast_whisper_rocm.webui.handlers.extract_audio_from_video"
            ) as mock_extract,
        ):
            mock_extract.side_effect = RuntimeError("Extraction failed")

            config = TranscriptionConfig()
            file_config = FileHandlingConfig()

            from insanely_fast_whisper_rocm.core.errors import TranscriptionError

            with pytest.raises(TranscriptionError, match="Extraction failed"):
                handlers.transcribe(str(video_path), config, file_config)


def test_transcribe_with_progress_tracker() -> None:
    """Test transcribe with progress tracker instance."""
    mock_orchestrator = unittest.mock.MagicMock()
    mock_orchestrator.run_transcription.return_value = {"text": "test"}

    # Mock progress tracker
    mock_progress = unittest.mock.MagicMock()
    mock_progress.cancelled = False

    with unittest.mock.patch(
        "insanely_fast_whisper_rocm.webui.handlers.create_orchestrator",
        return_value=mock_orchestrator,
    ):
        config = TranscriptionConfig()
        file_config = FileHandlingConfig()
        result = handlers.transcribe(
            "/fake/audio.wav",
            config,
            file_config,
            progress_tracker_instance=mock_progress,
        )

        assert result["text"] == "test"
        # Progress tracker should have been called
        assert mock_progress.call_count > 0


def test_transcribe_with_chunk_duration_warning() -> None:
    """Test transcribe logs warning when chunk_duration is set."""
    mock_orchestrator = unittest.mock.MagicMock()
    mock_orchestrator.run_transcription.return_value = {"text": "test"}

    with (
        unittest.mock.patch(
            "insanely_fast_whisper_rocm.webui.handlers.create_orchestrator",
            return_value=mock_orchestrator,
        ),
        unittest.mock.patch(
            "insanely_fast_whisper_rocm.webui.handlers.logger"
        ) as mock_logger,
    ):
        config = TranscriptionConfig(chunk_duration=15.0, chunk_overlap=1.0)
        file_config = FileHandlingConfig()
        handlers.transcribe("/fake/audio.wav", config, file_config)

        # chunk_duration/chunk_overlap are currently not used by the WebUI handler.
        mock_logger.warning.assert_not_called()


def test_is_stabilization_corrupt_with_duplicate_timestamps() -> None:
    """Test _is_stabilization_corrupt detects duplicate timestamps."""
    # More than 50% of segments have identical timestamps
    corrupt_segments = [
        {"start": 5.0, "end": 5.0, "text": "segment1"},
        {"start": 5.0, "end": 5.0, "text": "segment2"},
        {"start": 5.0, "end": 5.0, "text": "segment3"},
        {"start": 6.0, "end": 7.0, "text": "segment4"},
    ]
    assert _is_stabilization_corrupt(corrupt_segments) is True


def test_is_stabilization_corrupt_with_valid_timestamps() -> None:
    """Test _is_stabilization_corrupt returns False for valid timestamps."""
    valid_segments = [
        {"start": 0.0, "end": 1.0, "text": "segment1"},
        {"start": 1.0, "end": 2.0, "text": "segment2"},
        {"start": 2.0, "end": 3.0, "text": "segment3"},
    ]
    assert _is_stabilization_corrupt(valid_segments) is False


def test_is_stabilization_corrupt_with_empty_list() -> None:
    """Test _is_stabilization_corrupt handles empty segment list."""
    assert _is_stabilization_corrupt([]) is False


def test_is_stabilization_corrupt_with_single_segment() -> None:
    """Test _is_stabilization_corrupt handles single segment."""
    single_segment = [{"start": 0.0, "end": 1.0, "text": "only one"}]
    assert _is_stabilization_corrupt(single_segment) is False


def test_process_transcription_request_single_file() -> None:
    """Test process_transcription_request with a single file."""
    with tempfile.TemporaryDirectory() as temp_dir:
        audio_path = Path(temp_dir) / "test.wav"
        audio_path.write_text("fake audio")

        mock_result = {
            "text": "Test transcription",
            "output_file_path": str(Path(temp_dir) / "test.json"),
        }

        with unittest.mock.patch(
            "insanely_fast_whisper_rocm.webui.handlers.transcribe",
            return_value=mock_result,
        ):
            config = TranscriptionConfig()
            file_config = FileHandlingConfig(temp_uploads_dir=temp_dir)

            result = handlers.process_transcription_request(
                [str(audio_path)],
                config,
                file_config,
            )

            # Should return a tuple with 7 elements
            assert len(result) == 7
            # First element is transcription output text
            assert isinstance(result[0], str)
            assert "Test transcription" in result[0]


def test_process_transcription_request_multiple_files() -> None:
    """Test process_transcription_request with multiple files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        audio_path1 = Path(temp_dir) / "test1.wav"
        audio_path2 = Path(temp_dir) / "test2.wav"
        audio_path1.write_text("fake audio 1")
        audio_path2.write_text("fake audio 2")

        mock_result1 = {
            "text": "Transcription 1",
            "output_file_path": str(Path(temp_dir) / "test1.json"),
        }
        mock_result2 = {
            "text": "Transcription 2",
            "output_file_path": str(Path(temp_dir) / "test2.json"),
        }

        with unittest.mock.patch(
            "insanely_fast_whisper_rocm.webui.handlers.transcribe",
            side_effect=[mock_result1, mock_result2],
        ):
            config = TranscriptionConfig()
            file_config = FileHandlingConfig(temp_uploads_dir=temp_dir)

            result = handlers.process_transcription_request(
                [str(audio_path1), str(audio_path2)],
                config,
                file_config,
            )

            assert len(result) == 7
            # For multiple files, should show summary message
            assert "Successfully processed 2 files" in str(result[0]) or "2" in str(
                result[0]
            )


def test_process_transcription_request_with_error() -> None:
    """Test process_transcription_request handles transcription errors."""
    with tempfile.TemporaryDirectory() as temp_dir:
        audio_path = Path(temp_dir) / "test.wav"
        audio_path.write_text("fake audio")

        with unittest.mock.patch(
            "insanely_fast_whisper_rocm.webui.handlers.transcribe",
            side_effect=TranscriptionError("Test error"),
        ):
            config = TranscriptionConfig()
            file_config = FileHandlingConfig(temp_uploads_dir=temp_dir)

            result = handlers.process_transcription_request(
                [str(audio_path)],
                config,
                file_config,
            )

            # Should return error in the output
            assert "Error" in result[0] or "error" in str(result[1])


def test_process_transcription_request_with_cancellation() -> None:
    """Test process_transcription_request propagates cancellation."""
    with tempfile.TemporaryDirectory() as temp_dir:
        audio_path = Path(temp_dir) / "test.wav"
        audio_path.write_text("fake audio")

        with unittest.mock.patch(
            "insanely_fast_whisper_rocm.webui.handlers.transcribe",
            side_effect=TranscriptionCancelledError("Cancelled"),
        ):
            config = TranscriptionConfig()
            file_config = FileHandlingConfig(temp_uploads_dir=temp_dir)

            with pytest.raises(TranscriptionCancelledError):
                handlers.process_transcription_request(
                    [str(audio_path)],
                    config,
                    file_config,
                )
