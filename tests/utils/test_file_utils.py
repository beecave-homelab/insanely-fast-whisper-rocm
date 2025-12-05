"""Tests for utils/file_utils.py."""

from __future__ import annotations

import os
import tempfile
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException, UploadFile

from insanely_fast_whisper_rocm.utils.file_utils import (
    FileHandler,
    cleanup_temp_files,
    save_upload_file,
    validate_audio_file,
)


def test_validate_audio_file__valid_format() -> None:
    """Accept valid audio format."""
    file = MagicMock(spec=UploadFile)
    file.filename = "test.mp3"
    # Should not raise
    validate_audio_file(file)


def test_validate_audio_file__invalid_format__raises_http_exception() -> None:
    """Raise HTTPException for invalid audio format."""
    file = MagicMock(spec=UploadFile)
    file.filename = "test.txt"

    with pytest.raises(HTTPException, match="Unsupported file format"):
        validate_audio_file(file)


def test_save_upload_file__saves_file_successfully(tmp_path: Path) -> None:
    """Save uploaded file to disk successfully."""
    upload_dir = tmp_path / "uploads"

    file = MagicMock(spec=UploadFile)
    file.filename = "test.wav"
    file.file = BytesIO(b"fake audio data")

    with patch(
        "insanely_fast_whisper_rocm.utils.file_utils.UPLOAD_DIR", str(upload_dir)
    ):
        saved_path = save_upload_file(file)

    assert os.path.exists(saved_path)
    assert saved_path.startswith(str(upload_dir))
    assert "test.wav" in saved_path

    # Verify content
    with open(saved_path, "rb") as f:
        assert f.read() == b"fake audio data"


def test_save_upload_file__os_error__raises_http_exception(tmp_path: Path) -> None:
    """Raise HTTPException when OSError occurs during save."""
    upload_dir = tmp_path / "uploads"

    file = MagicMock(spec=UploadFile)
    file.filename = "test.wav"
    file.file = BytesIO(b"fake audio data")

    with patch(
        "insanely_fast_whisper_rocm.utils.file_utils.UPLOAD_DIR", str(upload_dir)
    ):
        with patch("builtins.open", side_effect=OSError("Disk full")):
            with pytest.raises(HTTPException, match="Error saving uploaded file"):
                save_upload_file(file)


def test_cleanup_temp_files__removes_files_successfully(tmp_path: Path) -> None:
    """Remove temporary files successfully."""
    file1 = tmp_path / "file1.txt"
    file2 = tmp_path / "file2.txt"
    file1.write_text("test")
    file2.write_text("test")

    cleanup_temp_files([str(file1), str(file2)])

    assert not file1.exists()
    assert not file2.exists()


def test_cleanup_temp_files__removes_empty_upload_dir(tmp_path: Path) -> None:
    """Remove empty parent directory from UPLOAD_DIR."""
    upload_dir = tmp_path / "temp_uploads"
    upload_dir.mkdir()
    test_file = upload_dir / "test.txt"
    test_file.write_text("test")

    with patch("insanely_fast_whisper_rocm.utils.file_utils.UPLOAD_DIR", str(tmp_path)):
        cleanup_temp_files([str(test_file)])

    assert not test_file.exists()
    assert not upload_dir.exists()  # Directory should be removed


def test_cleanup_temp_files__removes_empty_tempdir(tmp_path: Path) -> None:
    """Remove empty parent directory from tempfile.gettempdir()."""
    # Create a directory within the system temp
    with tempfile.TemporaryDirectory() as system_temp:
        test_dir = Path(system_temp) / "subdir"
        test_dir.mkdir()
        test_file = test_dir / "test.txt"
        test_file.write_text("test")

        cleanup_temp_files([str(test_file)])

        assert not test_file.exists()
        assert not test_dir.exists()


def test_cleanup_temp_files__handles_os_error_gracefully(tmp_path: Path) -> None:
    """Log warning but don't raise when cleanup fails."""
    non_existent_file = tmp_path / "does_not_exist.txt"

    # Should not raise, just log warning
    cleanup_temp_files([str(non_existent_file)])


def test_cleanup_temp_files__handles_rmdir_error_gracefully(tmp_path: Path) -> None:
    """Handle rmdir errors gracefully without raising."""
    test_file = tmp_path / "test.txt"
    test_file.write_text("test")

    # Mock os.rmdir to raise OSError
    with patch("os.rmdir", side_effect=OSError("Permission denied")):
        with patch(
            "insanely_fast_whisper_rocm.utils.file_utils.UPLOAD_DIR", str(tmp_path)
        ):
            # Should not raise
            cleanup_temp_files([str(test_file)])


def test_file_handler__init__creates_upload_dir(tmp_path: Path) -> None:
    """Initialize FileHandler and create upload directory."""
    upload_dir = tmp_path / "custom_uploads"
    handler = FileHandler(upload_dir=str(upload_dir))

    assert handler.upload_dir == str(upload_dir)
    assert upload_dir.exists()


def test_file_handler__validate_audio_file__calls_validate_function() -> None:
    """FileHandler.validate_audio_file calls the module-level function."""
    handler = FileHandler()
    file = MagicMock(spec=UploadFile)
    file.filename = "test.mp3"

    # Should not raise
    handler.validate_audio_file(file)


def test_file_handler__save_upload__saves_file_successfully(tmp_path: Path) -> None:
    """FileHandler saves uploaded file successfully."""
    upload_dir = tmp_path / "uploads"
    handler = FileHandler(upload_dir=str(upload_dir))

    file = MagicMock(spec=UploadFile)
    file.filename = "test.wav"
    file.file = BytesIO(b"test data")

    saved_path = handler.save_upload(file)

    assert os.path.exists(saved_path)
    assert saved_path.startswith(str(upload_dir))
    with open(saved_path, "rb") as f:
        assert f.read() == b"test data"


def test_file_handler__save_upload__logs_error_on_os_error(tmp_path: Path) -> None:
    """FileHandler logs error and raises HTTPException on OSError."""
    upload_dir = tmp_path / "uploads"
    handler = FileHandler(upload_dir=str(upload_dir))

    file = MagicMock(spec=UploadFile)
    file.filename = "test.wav"
    file.file = BytesIO(b"test data")

    with patch("builtins.open", side_effect=OSError("Disk error")):
        with pytest.raises(HTTPException, match="Error saving uploaded file"):
            handler.save_upload(file)


def test_file_handler__cleanup__removes_file_successfully(tmp_path: Path) -> None:
    """FileHandler cleanup removes file successfully."""
    handler = FileHandler(upload_dir=str(tmp_path))
    test_file = tmp_path / "test.txt"
    test_file.write_text("test")

    handler.cleanup(str(test_file))

    assert not test_file.exists()


def test_file_handler__cleanup__logs_warning_on_os_error(tmp_path: Path) -> None:
    """FileHandler cleanup logs warning on OSError without raising."""
    handler = FileHandler(upload_dir=str(tmp_path))
    test_file = tmp_path / "test.txt"
    test_file.write_text("test")

    with patch("os.remove", side_effect=OSError("Permission denied")):
        # Should not raise
        handler.cleanup(str(test_file))
