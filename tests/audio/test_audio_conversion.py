"""Tests for audio conversion utilities."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from insanely_fast_whisper_rocm.audio.conversion import (
    DEFAULT_CHANNELS,
    DEFAULT_CODEC,
    DEFAULT_SAMPLE_RATE,
    ensure_wav,
)


def test_ensure_wav_wav_file_passthrough(tmp_path: Path) -> None:
    """Test ensure_wav with existing WAV file returns unchanged path."""
    tmp_file = tmp_path / "test.wav"
    tmp_file.write_bytes(b"fake wav data")
    tmp_path_str = str(tmp_file)

    result = ensure_wav(tmp_path_str)
    assert result == tmp_path_str


def test_ensure_wav_wav_file_case_insensitive(tmp_path: Path) -> None:
    """Test ensure_wav handles WAV files with different cases."""
    test_cases = [".WAV", ".Wav", ".wav"]

    for suffix in test_cases:
        tmp_file = tmp_path / f"test{suffix}"
        tmp_file.write_bytes(b"fake wav data")
        tmp_path_str = str(tmp_file)

        result = ensure_wav(tmp_path_str)
        assert result == tmp_path_str


def test_ensure_wav_non_wav_file_conversion() -> None:
    """Test ensure_wav converts non-WAV files to WAV."""
    fixture_path = Path(__file__).parent / "fixtures" / "test_clip.mp3"

    # Mock ffmpeg to avoid actual conversion
    mock_ffmpeg_input = Mock()
    mock_ffmpeg_output = Mock()
    mock_run = Mock()

    mock_ffmpeg_input.output.return_value = mock_ffmpeg_output
    mock_ffmpeg_output.overwrite_output.return_value = mock_ffmpeg_output
    mock_ffmpeg_output.run.return_value = mock_run

    with patch("insanely_fast_whisper_rocm.audio.conversion.ffmpeg") as mock_ffmpeg:
        mock_ffmpeg.input.return_value = mock_ffmpeg_input

        result = ensure_wav(fixture_path)

        # Verify ffmpeg was called correctly
        mock_ffmpeg.input.assert_called_once_with(str(fixture_path))
        mock_ffmpeg_input.output.assert_called_once()
        call_args = mock_ffmpeg_input.output.call_args[1]
        assert call_args["acodec"] == DEFAULT_CODEC
        assert call_args["ac"] == DEFAULT_CHANNELS
        assert call_args["ar"] == DEFAULT_SAMPLE_RATE

        # Verify result is a WAV file path
        assert result.endswith(".wav")
        assert os.path.exists(result)

        # Cleanup the created file
        os.unlink(result)


def test_ensure_wav_custom_sample_rate() -> None:
    """Test ensure_wav with custom sample rate."""
    fixture_path = Path(__file__).parent / "fixtures" / "test_clip.mp3"
    custom_sample_rate = 44100

    with patch("insanely_fast_whisper_rocm.audio.conversion.ffmpeg") as mock_ffmpeg:
        mock_ffmpeg_input = Mock()
        mock_ffmpeg_output = Mock()
        mock_run = Mock()

        mock_ffmpeg_input.output.return_value = mock_ffmpeg_output
        mock_ffmpeg_output.overwrite_output.return_value = mock_ffmpeg_output
        mock_ffmpeg_output.run.return_value = mock_run

        mock_ffmpeg.input.return_value = mock_ffmpeg_input

        result = ensure_wav(fixture_path, sample_rate=custom_sample_rate)

        # Verify custom sample rate was used
        call_args = mock_ffmpeg_input.output.call_args[1]
        assert call_args["ar"] == custom_sample_rate

        os.unlink(result)


def test_ensure_wav_custom_channels() -> None:
    """Test ensure_wav with custom channel count."""
    fixture_path = Path(__file__).parent / "fixtures" / "test_clip.mp3"
    custom_channels = 2

    with patch("insanely_fast_whisper_rocm.audio.conversion.ffmpeg") as mock_ffmpeg:
        mock_ffmpeg_input = Mock()
        mock_ffmpeg_output = Mock()
        mock_run = Mock()

        mock_ffmpeg_input.output.return_value = mock_ffmpeg_output
        mock_ffmpeg_output.overwrite_output.return_value = mock_ffmpeg_output
        mock_ffmpeg_output.run.return_value = mock_run

        mock_ffmpeg.input.return_value = mock_ffmpeg_input

        result = ensure_wav(fixture_path, channels=custom_channels)

        # Verify custom channels was used
        call_args = mock_ffmpeg_input.output.call_args[1]
        assert call_args["ac"] == custom_channels

        os.unlink(result)


def test_ensure_wav_pathlib_input() -> None:
    """Test ensure_wav handles Path objects as input."""
    fixture_path = Path(__file__).parent / "fixtures" / "test_clip.mp3"

    with patch("insanely_fast_whisper_rocm.audio.conversion.ffmpeg") as mock_ffmpeg:
        mock_ffmpeg_input = Mock()
        mock_ffmpeg_output = Mock()
        mock_run = Mock()

        mock_ffmpeg_input.output.return_value = mock_ffmpeg_output
        mock_ffmpeg_output.overwrite_output.return_value = mock_ffmpeg_output
        mock_ffmpeg_output.run.return_value = mock_run

        mock_ffmpeg.input.return_value = mock_ffmpeg_input

        result = ensure_wav(fixture_path)

        # Verify it was converted to string internally
        mock_ffmpeg.input.assert_called_once_with(str(fixture_path))

        os.unlink(result)


def test_ensure_wav_constants() -> None:
    """Test that default constants are properly defined."""
    assert DEFAULT_SAMPLE_RATE == 16000
    assert DEFAULT_CHANNELS == 1
    assert DEFAULT_CODEC == "pcm_s16le"


def test_ensure_wav_file_not_found() -> None:
    """Test ensure_wav raises FileNotFoundError for non-existent file."""
    with pytest.raises(FileNotFoundError, match="Audio file not found"):
        ensure_wav("/path/to/nonexistent/file.mp3")


def test_ensure_wav_ffmpeg_unavailable() -> None:
    """Test ensure_wav creates placeholder when ffmpeg is unavailable."""
    # Use the real audio test fixture
    fixture_path = Path(__file__).parent / "fixtures" / "test_clip.mp3"

    # Mock ffmpeg as None
    with patch("insanely_fast_whisper_rocm.audio.conversion.ffmpeg", None):
        result = ensure_wav(fixture_path)

        # Should create a WAV file (placeholder copy)
        assert result.endswith(".wav")
        assert os.path.exists(result)

        # Verify content was copied
        with open(result, "rb") as converted:
            converted_content = converted.read()

        # The converted file should contain the original audio data
        # (pydub converts it to WAV format, so it won't be identical)
        assert len(converted_content) > 0

        os.unlink(result)
        # Clean up the temp directory
        import shutil

        shutil.rmtree(Path(result).parent)


def test_ensure_wav_ffmpeg_error() -> None:
    """Test ensure_wav raises RuntimeError when ffmpeg fails."""

    # Create a custom ffmpeg.Error class for testing
    class FFmpegError(Exception):
        """Mock ffmpeg.Error exception."""

        def __init__(self, message: str) -> None:
            """Initialize with message and stderr attribute."""
            super().__init__(message)
            self.stderr = b"Conversion error details"

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        with patch("insanely_fast_whisper_rocm.audio.conversion.ffmpeg") as mock_ffmpeg:
            mock_error = FFmpegError("FFmpeg conversion failed")

            # Mock the chain to raise an error
            mock_ffmpeg_input = Mock()
            mock_ffmpeg_output = Mock()

            mock_ffmpeg_input.output.return_value = mock_ffmpeg_output
            mock_ffmpeg_output.overwrite_output.return_value = mock_ffmpeg_output
            mock_ffmpeg_output.run.side_effect = mock_error

            mock_ffmpeg.input.return_value = mock_ffmpeg_input
            mock_ffmpeg.Error = FFmpegError

            with pytest.raises(RuntimeError, match="Failed to convert.*to WAV"):
                ensure_wav(tmp_path)
    finally:
        os.unlink(tmp_path)


def test_ensure_wav_output_not_created() -> None:
    """Test ensure_wav creates empty file if conversion doesn't create output."""
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        with patch("insanely_fast_whisper_rocm.audio.conversion.ffmpeg") as mock_ffmpeg:
            mock_ffmpeg_input = Mock()
            mock_ffmpeg_output = Mock()

            mock_ffmpeg_input.output.return_value = mock_ffmpeg_output
            mock_ffmpeg_output.overwrite_output.return_value = mock_ffmpeg_output
            mock_ffmpeg_output.run.return_value = None

            mock_ffmpeg.input.return_value = mock_ffmpeg_input

            # Mock Path.exists to return False first time, True second time
            with patch("pathlib.Path.exists") as mock_exists:
                with patch("pathlib.Path.touch") as mock_touch:
                    # First call for original file check (True)
                    # Second call for output check (False)
                    mock_exists.side_effect = [True, False]

                    result = ensure_wav(tmp_path)

                    # Verify touch was called to create the file
                    mock_touch.assert_called_once()
                    assert result.endswith(".wav")
    finally:
        os.unlink(tmp_path)
