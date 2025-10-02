"""Tests for audio processing utilities."""

from __future__ import annotations

import os
import tempfile
from unittest.mock import Mock, patch

import pytest

from insanely_fast_whisper_api.audio.processing import (
    extract_audio_from_video,
    get_audio_duration,
    split_audio,
)


def test_get_audio_duration_valid_file() -> None:
    """Test get_audio_duration with a valid audio file."""
    # Create a mock audio file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        # Mock AudioSegment to avoid actual file processing
        mock_audio = Mock()
        mock_audio.__len__ = Mock(return_value=5000)  # 5 seconds in milliseconds

        with patch(
            "insanely_fast_whisper_api.audio.processing.AudioSegment"
        ) as mock_audio_segment:
            mock_audio_segment.from_file.return_value = mock_audio

            result = get_audio_duration(tmp_path)

            assert result == 5.0
            mock_audio_segment.from_file.assert_called_once_with(tmp_path)
    finally:
        os.unlink(tmp_path)


def test_get_audio_duration_file_not_found() -> None:
    """Test get_audio_duration with non-existent file."""
    non_existent_path = "/path/to/nonexistent/file.wav"

    with patch(
        "insanely_fast_whisper_api.audio.processing.AudioSegment"
    ) as mock_audio_segment:
        mock_audio_segment.from_file.side_effect = OSError("File not found")

        with pytest.raises(RuntimeError, match="Failed to get audio duration"):
            get_audio_duration(non_existent_path)


def test_get_audio_duration_processing_error() -> None:
    """Test get_audio_duration with processing error."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        with patch(
            "insanely_fast_whisper_api.audio.processing.AudioSegment"
        ) as mock_audio_segment:
            mock_audio_segment.from_file.side_effect = RuntimeError("Processing error")

            with pytest.raises(RuntimeError, match="Failed to get audio duration"):
                get_audio_duration(tmp_path)
    finally:
        os.unlink(tmp_path)


def test_extract_audio_from_video_valid_file() -> None:
    """Test extract_audio_from_video with valid video file."""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        with patch("insanely_fast_whisper_api.audio.processing.ffmpeg") as mock_ffmpeg:
            mock_ffmpeg_input = Mock()
            mock_ffmpeg_output = Mock()
            mock_run = Mock()

            mock_ffmpeg_input.output.return_value = mock_ffmpeg_output
            mock_ffmpeg_output.overwrite_output.return_value = mock_ffmpeg_output
            mock_ffmpeg_output.run.return_value = mock_run

            mock_ffmpeg.input.return_value = mock_ffmpeg_input

            result = extract_audio_from_video(tmp_path)

            # Verify ffmpeg was called correctly
            mock_ffmpeg.input.assert_called_once_with(tmp_path)
            call_args = mock_ffmpeg_input.output.call_args[1]
            assert call_args["acodec"] == "pcm_s16le"
            assert call_args["ac"] == 1
            assert call_args["ar"] == 16000
            assert call_args["vn"] is None

            # Verify result path format
            assert result.endswith(".wav")
            # Note: file doesn't exist since ffmpeg is mocked
    finally:
        os.unlink(tmp_path)


def test_extract_audio_from_video_file_not_found() -> None:
    """Test extract_audio_from_video with non-existent file."""
    non_existent_path = "/path/to/nonexistent/video.mp4"

    with pytest.raises(FileNotFoundError, match="Video file not found"):
        extract_audio_from_video(non_existent_path)


def test_extract_audio_from_video_ffmpeg_error() -> None:
    """Test extract_audio_from_video with FFmpeg error."""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        mock_error = Mock()
        mock_error.stderr = b"FFmpeg error message"

        with patch("insanely_fast_whisper_api.audio.processing.ffmpeg") as mock_ffmpeg:
            # Mock ffmpeg.Error to be a proper exception class
            mock_ffmpeg.Error = type(
                "FFmpegError", (Exception,), {"stderr": b"FFmpeg error"}
            )

            mock_ffmpeg_input = Mock()
            mock_ffmpeg_output = Mock()
            mock_ffmpeg_input.output.return_value = mock_ffmpeg_output
            mock_ffmpeg_output.overwrite_output.return_value = mock_ffmpeg_output

            # Make run() raise ffmpeg.Error
            error_instance = mock_ffmpeg.Error()
            error_instance.stderr = b"FFmpeg error message"
            mock_ffmpeg_output.run.side_effect = error_instance

            mock_ffmpeg.input.return_value = mock_ffmpeg_input

            with pytest.raises(
                RuntimeError, match="Failed to extract audio from video"
            ):
                extract_audio_from_video(tmp_path)
    finally:
        os.unlink(tmp_path)


def test_extract_audio_from_video_custom_parameters() -> None:
    """Test extract_audio_from_video with custom parameters."""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        with patch("insanely_fast_whisper_api.audio.processing.ffmpeg") as mock_ffmpeg:
            mock_ffmpeg_input = Mock()
            mock_ffmpeg_output = Mock()
            mock_run = Mock()

            mock_ffmpeg_input.output.return_value = mock_ffmpeg_output
            mock_ffmpeg_output.overwrite_output.return_value = mock_ffmpeg_output
            mock_ffmpeg_output.run.return_value = mock_run

            mock_ffmpeg.input.return_value = mock_ffmpeg_input

            _result = extract_audio_from_video(
                tmp_path, output_format="mp3", sample_rate=44100, channels=2
            )

            # Verify custom parameters were used
            call_args = mock_ffmpeg_input.output.call_args[1]
            assert call_args["acodec"] == "pcm_s16le"  # Should always be this
            assert call_args["ac"] == 2
            assert call_args["ar"] == 44100

            # Note: result path returned but file doesn't exist since ffmpeg is mocked
    finally:
        os.unlink(tmp_path)


def test_split_audio_valid_parameters() -> None:
    """Test split_audio with valid parameters."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        # Mock AudioSegment - must be subscriptable for slicing operations
        mock_audio = Mock()
        mock_audio.__len__ = Mock(return_value=30000)  # 30 seconds
        mock_chunk = Mock()
        mock_audio.__getitem__ = Mock(return_value=mock_chunk)
        mock_chunk.export = Mock()

        with patch(
            "insanely_fast_whisper_api.audio.processing.AudioSegment"
        ) as mock_audio_segment:
            with patch(
                "insanely_fast_whisper_api.audio.processing.tempfile.mkdtemp"
            ) as mock_mkdtemp:
                with patch(
                    "insanely_fast_whisper_api.audio.processing.os.path.join"
                ) as mock_join:
                    mock_audio_segment.from_file.return_value = mock_audio
                    mock_mkdtemp.return_value = "/tmp/test_dir"
                    mock_join.return_value = "/tmp/test_dir/chunk_0001.wav"

                    result = split_audio(
                        tmp_path, chunk_duration=10.0, chunk_overlap=1.0
                    )

                    assert len(result) == 3  # 30 seconds / 10 seconds = 3 chunks
                    assert all(isinstance(path, str) for path, _ in result)
                    assert all(
                        isinstance(start_time, float) for _, start_time in result
                    )
    finally:
        os.unlink(tmp_path)


def test_split_audio_short_audio() -> None:
    """Test split_audio with audio shorter than chunk duration."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        # Mock AudioSegment for short audio
        mock_audio = Mock()
        mock_audio.__len__ = Mock(return_value=5000)  # 5 seconds

        with patch(
            "insanely_fast_whisper_api.audio.processing.AudioSegment"
        ) as mock_audio_segment:
            mock_audio_segment.from_file.return_value = mock_audio

            result = split_audio(tmp_path, chunk_duration=10.0, chunk_overlap=1.0)

            assert len(result) == 1
            assert result[0][0] == tmp_path  # Should return original file
            assert result[0][1] == 0.0  # Start time should be 0
    finally:
        os.unlink(tmp_path)


def test_split_audio_invalid_parameters() -> None:
    """Test split_audio with invalid parameters."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        # Test negative chunk duration
        with pytest.raises(ValueError, match="chunk_duration must be greater than 0"):
            split_audio(tmp_path, chunk_duration=-1.0)

        # Test negative chunk overlap
        with pytest.raises(ValueError, match="chunk_overlap cannot be negative"):
            split_audio(tmp_path, chunk_overlap=-1.0)

        # Test overlap >= duration
        with pytest.raises(
            ValueError, match="chunk_overlap must be less than chunk_duration"
        ):
            split_audio(tmp_path, chunk_duration=10.0, chunk_overlap=15.0)

        # Test negative min chunk duration
        with pytest.raises(
            ValueError, match="min_chunk_duration must be greater than 0"
        ):
            split_audio(tmp_path, min_chunk_duration=-1.0)
    finally:
        os.unlink(tmp_path)


def test_split_audio_processing_error() -> None:
    """Test split_audio with processing error."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        with patch(
            "insanely_fast_whisper_api.audio.processing.AudioSegment"
        ) as mock_audio_segment:
            mock_audio_segment.from_file.side_effect = RuntimeError("Processing error")

            with pytest.raises(RuntimeError, match="Failed to split audio"):
                split_audio(tmp_path)
    finally:
        os.unlink(tmp_path)


def test_split_audio_memory_error() -> None:
    """Test split_audio with memory error."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        with patch(
            "insanely_fast_whisper_api.audio.processing.AudioSegment"
        ) as mock_audio_segment:
            with patch(
                "insanely_fast_whisper_api.audio.processing.tempfile.mkdtemp"
            ) as mock_mkdtemp:
                mock_audio_segment.from_file.side_effect = MemoryError("Out of memory")
                mock_mkdtemp.return_value = "/tmp/test_dir"

                with pytest.raises(RuntimeError, match="Failed to split audio"):
                    split_audio(tmp_path)
    finally:
        os.unlink(tmp_path)


def test_split_audio_chunk_export() -> None:
    """Test split_audio chunk export functionality."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        # Mock AudioSegment and chunk
        mock_audio = Mock()
        mock_audio.__len__ = Mock(return_value=15000)  # 15 seconds

        mock_chunk = Mock()
        mock_chunk.export = Mock()

        with patch(
            "insanely_fast_whisper_api.audio.processing.AudioSegment"
        ) as mock_audio_segment:
            with patch(
                "insanely_fast_whisper_api.audio.processing.tempfile.mkdtemp"
            ) as mock_mkdtemp:
                with patch(
                    "insanely_fast_whisper_api.audio.processing.os.path.join"
                ) as mock_join:
                    mock_audio_segment.from_file.return_value = mock_audio
                    mock_mkdtemp.return_value = "/tmp/test_dir"
                    mock_join.return_value = "/tmp/test_dir/chunk_0001.wav"

                    # Mock slicing to return our mock chunk
                    mock_audio.__getitem__ = Mock(return_value=mock_chunk)

                    split_audio(tmp_path, chunk_duration=5.0)

                    # Verify chunk.export was called
                    mock_chunk.export.assert_called()
                    call_args = mock_chunk.export.call_args
                    assert call_args[0][0].endswith("chunk_0001.wav")
                    assert call_args[1]["format"] == "wav"
    finally:
        os.unlink(tmp_path)
