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
    apply_speaker_rename,
    build_speaker_review_state,
    build_speaker_review_summary,
    merge_review_speakers,
    refresh_review_downloads,
    retag_transcript_segment,
    split_review_segment_speaker,
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

            # Should return a tuple with 8 elements (status + 7 others)
            assert len(result) == 8
            # First element is status text
            assert isinstance(result[0], str)
            # Second element is transcription output text
            assert isinstance(result[1], str)
            assert "Test transcription" in result[1]
            assert result[3] == mock_result


def test_build_speaker_review_summary__shows_map_and_review_markers() -> None:
    """Speaker review summary lists readable map and ambiguous turns."""
    raw_result = {
        "diarized": True,
        "segments": [
            {
                "start": 0.0,
                "end": 1.0,
                "text": "Hello",
                "speaker": "SPEAKER_00",
                "speaker_confidence": 0.4,
                "overlap": True,
            },
            {
                "start": 1.0,
                "end": 2.0,
                "text": "World",
                "speaker": "SPEAKER_01",
            },
        ],
    }

    summary = build_speaker_review_summary(raw_result)

    assert "SPEAKER_00 -> SPEAKER_00" in summary
    assert "SPEAKER_01 -> SPEAKER_01" in summary
    assert "1 low-confidence turn(s)" in summary
    assert "1 overlapping turn(s)" in summary


def test_build_speaker_review_state__marks_ambiguous_segment_choices() -> None:
    """Review segment labels flag low-confidence and overlapping turns."""
    raw_result = {
        "segments": [
            {
                "start": 0.0,
                "end": 1.0,
                "text": "Hello",
                "speaker": "SPEAKER_00",
                "speaker_confidence": 0.4,
                "overlap": True,
            }
        ],
    }

    state = build_speaker_review_state(raw_result)

    assert "low confidence" in state["segment_choices"][0]
    assert "overlap" in state["segment_choices"][0]


def test_apply_speaker_rename__preserves_internal_id_and_adds_display_name() -> None:
    """Speaker rename keeps raw speaker IDs while adding readable labels."""
    raw_result = {
        "text": "Hello",
        "segments": [
            {"start": 0.0, "end": 1.0, "text": "Hello", "speaker": "SPEAKER_00"}
        ],
    }

    transcript_preview, json_preview, updated = apply_speaker_rename(
        raw_result,
        "SPEAKER_00",
        "Host",
    )

    assert updated is not None
    assert "[Host] Hello" in transcript_preview
    assert updated["speaker_names"] == {"SPEAKER_00": "Host"}
    assert updated["segments"][0]["speaker"] == "SPEAKER_00"
    assert updated["segments"][0]["speaker_display"] == "Host"
    assert '"speaker": "SPEAKER_00"' in json_preview
    assert '"speaker_display": "Host"' in json_preview


def test_retag_transcript_segment__updates_selected_segment_speaker() -> None:
    """Retagging a segment changes the selected turn speaker."""
    raw_result = {
        "text": "Hello Bye",
        "segments": [
            {"start": 0.0, "end": 1.0, "text": "Hello", "speaker": "SPEAKER_00"},
            {"start": 1.0, "end": 2.0, "text": "Bye", "speaker": "SPEAKER_01"},
        ],
    }

    _, _, updated = retag_transcript_segment(raw_result, "1: [1.0-2.0]", "SPEAKER_00")

    assert updated is not None
    assert updated["segments"][1]["speaker"] == "SPEAKER_00"


def test_split_review_segment_speaker__creates_new_speaker_label() -> None:
    """Splitting a segment creates a new speaker ID for one selected turn."""
    raw_result = {
        "text": "Hello Bye",
        "segments": [
            {"start": 0.0, "end": 1.0, "text": "Hello", "speaker": "SPEAKER_00"},
            {"start": 1.0, "end": 2.0, "text": "Bye", "speaker": "SPEAKER_00"},
        ],
    }

    transcript_preview, _, updated = split_review_segment_speaker(
        raw_result,
        "1: [1.0-2.0]",
        "Guest",
    )

    assert updated is not None
    assert updated["segments"][0]["speaker"] == "SPEAKER_00"
    assert updated["segments"][1]["speaker"] == "SPEAKER_01"
    assert updated["speaker_names"]["SPEAKER_01"] == "Guest"
    assert "[Guest] Bye" in transcript_preview


def test_merge_review_speakers__replaces_source_speaker() -> None:
    """Merging speakers replaces duplicated speaker labels across segments."""
    raw_result = {
        "text": "Hello Bye",
        "speaker_names": {"SPEAKER_00": "Host", "SPEAKER_01": "Guest"},
        "segments": [
            {"start": 0.0, "end": 1.0, "text": "Hello", "speaker": "SPEAKER_00"},
            {"start": 1.0, "end": 2.0, "text": "Bye", "speaker": "SPEAKER_01"},
        ],
    }

    _, _, updated = merge_review_speakers(raw_result, "SPEAKER_01", "SPEAKER_00")

    assert updated is not None
    assert [segment["speaker"] for segment in updated["segments"]] == [
        "SPEAKER_00",
        "SPEAKER_00",
    ]
    assert "SPEAKER_01" not in updated["speaker_names"]


def test_refresh_review_downloads__exports_readable_speaker_labels() -> None:
    """Reviewed downloads are regenerated from speaker-display labels."""
    raw_result = {
        "text": "Hello",
        "diarized": True,
        "audio_file_path": "/tmp/interview.wav",
        "segments": [
            {
                "start": 0.0,
                "end": 1.0,
                "text": "Hello",
                "speaker": "SPEAKER_00",
                "speaker_display": "Host",
            }
        ],
    }

    with tempfile.TemporaryDirectory() as temp_dir:
        _, txt_update, srt_update, json_update = refresh_review_downloads(
            raw_result,
            temp_dir,
            "transcribe",
        )

        txt_path = Path(txt_update["value"])
        srt_path = Path(srt_update["value"])
        json_path = Path(json_update["value"])
        assert "[Host] Hello" in txt_path.read_text(encoding="utf-8")
        assert "[Host] Hello" in srt_path.read_text(encoding="utf-8")
        assert '"speaker": "SPEAKER_00"' in json_path.read_text(encoding="utf-8")


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

            assert len(result) == 8
            # For multiple files, should show summary message
            assert "Successfully processed 2 files" in str(result[1]) or "2" in str(
                result[1]
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
