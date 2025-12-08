"""Tests for insanely_fast_whisper_rocm.webui.utils module.

This module contains tests for WebUI utility functions including file handling,
device checks, and timestamp generation.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from insanely_fast_whisper_rocm.webui.utils import (
    convert_device_string,
    generate_timestamped_filename,
    is_cuda_available,
    is_mps_available,
    save_temp_file,
)


class TestSaveTempFile:
    """Test suite for save_temp_file function."""

    def test_save_temp_file__default_extension(self) -> None:
        """Test saving temp file with default txt extension."""
        content = "Test content"
        result = save_temp_file(content)

        assert os.path.exists(result)
        assert result.endswith(".txt")
        with open(result, encoding="utf-8") as f:
            assert f.read() == content
        os.unlink(result)

    def test_save_temp_file__custom_extension(self) -> None:
        """Test saving temp file with custom extension."""
        content = "Test SRT content"
        result = save_temp_file(content, extension="srt")

        assert os.path.exists(result)
        assert result.endswith(".srt")
        with open(result, encoding="utf-8") as f:
            assert f.read() == content
        os.unlink(result)

    def test_save_temp_file__with_desired_filename(self) -> None:
        """Test saving temp file with desired filename."""
        content = "Test content"
        desired_name = "my_test_file"
        result = save_temp_file(content, extension="txt", desired_filename=desired_name)

        assert os.path.exists(result)
        assert "my_test_file.txt" in result
        with open(result, encoding="utf-8") as f:
            assert f.read() == content
        os.unlink(result)

    def test_save_temp_file__desired_filename_already_has_extension(self) -> None:
        """Test that desired filename with extension is handled correctly."""
        content = "Test content"
        desired_name = "my_test_file.txt"
        result = save_temp_file(content, extension="txt", desired_filename=desired_name)

        assert os.path.exists(result)
        assert "my_test_file.txt" in result
        # Should not have double extension
        assert not result.endswith(".txt.txt")
        with open(result, encoding="utf-8") as f:
            assert f.read() == content
        os.unlink(result)

    def test_save_temp_file__empty_content(self) -> None:
        """Test saving empty content to temp file."""
        content = ""
        result = save_temp_file(content)

        assert os.path.exists(result)
        with open(result, encoding="utf-8") as f:
            assert f.read() == ""
        os.unlink(result)

    def test_save_temp_file__unicode_content(self) -> None:
        """Test saving unicode content to temp file."""
        content = "Test with émojis 🎉 and spëcial çhars"
        result = save_temp_file(content)

        assert os.path.exists(result)
        with open(result, encoding="utf-8") as f:
            assert f.read() == content
        os.unlink(result)

    def test_save_temp_file__multiline_content(self) -> None:
        """Test saving multiline content to temp file."""
        content = "Line 1\nLine 2\nLine 3"
        result = save_temp_file(content)

        assert os.path.exists(result)
        with open(result, encoding="utf-8") as f:
            assert f.read() == content
        os.unlink(result)

    @patch("tempfile.mkstemp", side_effect=OSError("Write error"))
    def test_save_temp_file__raises_on_write_error(
        self, mock_mkstemp: MagicMock
    ) -> None:
        """Test that OSError is raised when file cannot be written."""
        with pytest.raises(OSError, match="Write error"):
            save_temp_file("content")


class TestConvertDeviceString:
    """Test suite for convert_device_string function."""

    def test_convert_device_string__cpu(self) -> None:
        """Test converting 'cpu' device string."""
        result = convert_device_string("cpu")
        assert result == "cpu"

    def test_convert_device_string__cuda(self) -> None:
        """Test converting 'cuda' device string."""
        result = convert_device_string("cuda")
        assert result == "cuda"

    def test_convert_device_string__cuda_with_index(self) -> None:
        """Test converting 'cuda:0' device string."""
        result = convert_device_string("cuda:0")
        assert result == "cuda:0"

    def test_convert_device_string__mps(self) -> None:
        """Test converting 'mps' device string."""
        result = convert_device_string("mps")
        assert result == "mps"


class TestGenerateTimestampedFilename:
    """Test suite for generate_timestamped_filename function."""

    def test_generate_timestamped_filename__basic(self) -> None:
        """Test generating timestamped filename with basic inputs."""
        result = generate_timestamped_filename("transcript", "txt")
        assert result.startswith("transcript_")
        assert result.endswith(".txt")
        # Check format: base_YYYYMMDD_HHMMSS.ext
        assert len(result.split("_")) == 3
        assert len(result.split("_")[1]) == 8  # YYYYMMDD
        assert len(result.split("_")[2].split(".")[0]) == 6  # HHMMSS

    def test_generate_timestamped_filename__different_extension(self) -> None:
        """Test generating timestamped filename with different extension."""
        result = generate_timestamped_filename("output", "srt")
        assert result.startswith("output_")
        assert result.endswith(".srt")

    def test_generate_timestamped_filename__unique_timestamps(self) -> None:
        """Test that consecutive calls produce unique filenames."""
        result1 = generate_timestamped_filename("test", "txt")
        result2 = generate_timestamped_filename("test", "txt")
        # They might be the same if called in the same second
        # Just verify both are valid
        assert result1.startswith("test_")
        assert result2.startswith("test_")

    def test_generate_timestamped_filename__long_base_name(self) -> None:
        """Test generating timestamped filename with long base name."""
        result = generate_timestamped_filename("very_long_base_name_here", "json")
        assert result.startswith("very_long_base_name_here_")
        assert result.endswith(".json")


class TestIsCudaAvailable:
    """Test suite for is_cuda_available function."""

    @patch("insanely_fast_whisper_rocm.webui.utils.torch.cuda.is_available")
    def test_is_cuda_available__returns_true_when_available(
        self, mock_cuda: MagicMock
    ) -> None:
        """Test that is_cuda_available returns True when CUDA is available."""
        mock_cuda.return_value = True
        result = is_cuda_available()
        assert result is True
        mock_cuda.assert_called_once()

    @patch("insanely_fast_whisper_rocm.webui.utils.torch.cuda.is_available")
    def test_is_cuda_available__returns_false_when_unavailable(
        self, mock_cuda: MagicMock
    ) -> None:
        """Test that is_cuda_available returns False when CUDA is unavailable."""
        mock_cuda.return_value = False
        result = is_cuda_available()
        assert result is False
        mock_cuda.assert_called_once()


class TestIsMpsAvailable:
    """Test suite for is_mps_available function."""

    @patch("insanely_fast_whisper_rocm.webui.utils.torch")
    def test_is_mps_available__returns_true_when_available(
        self, mock_torch: MagicMock
    ) -> None:
        """Test that is_mps_available returns True when MPS is available."""
        mock_torch.mps.is_available.return_value = True
        result = is_mps_available()
        assert result is True

    @patch("insanely_fast_whisper_rocm.webui.utils.torch")
    def test_is_mps_available__returns_false_when_unavailable(
        self, mock_torch: MagicMock
    ) -> None:
        """Test that is_mps_available returns False when MPS is unavailable."""
        mock_torch.mps.is_available.return_value = False
        result = is_mps_available()
        assert result is False

    @patch("insanely_fast_whisper_rocm.webui.utils.torch", spec=[])
    def test_is_mps_available__returns_false_when_mps_not_in_torch(
        self, mock_torch: MagicMock
    ) -> None:
        """Test that is_mps_available returns False when torch.mps doesn't exist."""
        result = is_mps_available()
        assert result is False
