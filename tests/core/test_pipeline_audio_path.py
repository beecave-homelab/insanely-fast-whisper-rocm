"""Tests for audio path handling in the pipeline.

This test suite verifies that the pipeline correctly stores:
- original_file: user-facing display name
- audio_file_path: absolute filesystem path for post-processing (e.g., stabilization)
"""

from __future__ import annotations

import pathlib
from typing import Any
from unittest.mock import MagicMock

import pytest

from insanely_fast_whisper_rocm.core.asr_backend import ASRBackend
from insanely_fast_whisper_rocm.core.pipeline import WhisperPipeline


@pytest.fixture
def audio_fixture_path() -> pathlib.Path:
    """
    Provide the path to the test audio fixture file tests/audio/fixtures/test_clip.mp3.
    
    Returns:
        pathlib.Path: Absolute or relative Path pointing to the fixture file.
    """
    return pathlib.Path(__file__).parents[1] / "audio" / "fixtures" / "test_clip.mp3"


def test_postprocess_stores_absolute_audio_path(
    tmp_path: pathlib.Path,
    audio_fixture_path: pathlib.Path,
) -> None:
    """Ensure _postprocess_output stores absolute path in audio_file_path.

    This test verifies that audio_file_path contains the absolute filesystem
    path needed for post-processing (e.g., stabilization), while original_file
    contains the user-facing display name.

    Args:
        tmp_path: Pytest fixture for temporary directory.
        audio_fixture_path: Path to the test audio fixture.
    """
    # Create a test audio file in a subdirectory
    audio_dir = tmp_path / "uploads"
    audio_dir.mkdir()
    audio_file = audio_dir / "test_audio.wav"
    audio_file.write_bytes(audio_fixture_path.read_bytes())

    # Create a mock ASR backend
    mock_backend = MagicMock(spec=ASRBackend)

    # Create the pipeline
    pipeline = WhisperPipeline(
        asr_backend=mock_backend,
        storage_backend=None,
        save_transcriptions=False,
    )

    # Mock ASR output
    asr_output: dict[str, Any] = {
        "text": "Test transcription",
        "chunks": [{"text": "Test", "timestamp": [0.0, 1.0]}],
        "runtime_seconds": 1.5,
    }

    # Call _postprocess_output with a relative path
    # (simulating what happens when user provides relative path)
    relative_path = pathlib.Path("uploads/test_audio.wav")

    result = pipeline._postprocess_output(
        asr_output=asr_output,
        audio_file_path=relative_path,
        task="transcribe",
        original_filename=None,
    )

    # CRITICAL ASSERTION: audio_file_path should be an absolute path
    audio_path = result.get("audio_file_path")
    assert audio_path is not None, "audio_file_path must be set"

    # Convert to Path to check if it's absolute
    audio_path_obj = pathlib.Path(audio_path)
    assert audio_path_obj.is_absolute(), (
        f"audio_file_path must be absolute path, got: {audio_path}"
    )

    # Verify it points to the correct file
    assert audio_path_obj.name == "test_audio.wav"
    assert "uploads" in str(audio_path_obj)


def test_postprocess_preserves_absolute_audio_path(
    tmp_path: pathlib.Path,
    audio_fixture_path: pathlib.Path,
) -> None:
    """Ensure _postprocess_output preserves already-absolute paths.

    Args:
        tmp_path: Pytest fixture for temporary directory.
        audio_fixture_path: Path to the test audio fixture.
    """
    # Create a test audio file
    audio_file = tmp_path / "test_audio.wav"
    audio_file.write_bytes(audio_fixture_path.read_bytes())

    mock_backend = MagicMock(spec=ASRBackend)
    pipeline = WhisperPipeline(
        asr_backend=mock_backend,
        storage_backend=None,
        save_transcriptions=False,
    )

    asr_output: dict[str, Any] = {
        "text": "Test transcription",
        "chunks": [],
        "runtime_seconds": 1.0,
    }

    # Call with absolute path
    result = pipeline._postprocess_output(
        asr_output=asr_output,
        audio_file_path=audio_file.absolute(),
        task="transcribe",
        original_filename=None,
    )

    original_file = result.get("original_file")
    assert original_file is not None
    original_file_path = pathlib.Path(original_file)
    assert original_file_path.is_absolute()
    assert original_file_path == audio_file.absolute()


def test_postprocess_uses_original_filename_when_provided(
    tmp_path: pathlib.Path,
    audio_fixture_path: pathlib.Path,
) -> None:
    """
    Ensure that a provided original filename is used as the `original_file` in postprocessed output.
    
    Verifies that when `original_filename` is supplied (as with user uploads), the pipeline sets `original_file` to that value instead of deriving it from `audio_file_path`.
    """
    audio_file = tmp_path / "temp_chunk.wav"
    audio_file.write_bytes(audio_fixture_path.read_bytes())

    mock_backend = MagicMock(spec=ASRBackend)
    pipeline = WhisperPipeline(
        asr_backend=mock_backend,
        storage_backend=None,
        save_transcriptions=False,
    )

    asr_output: dict[str, Any] = {
        "text": "Test transcription",
        "chunks": [],
        "runtime_seconds": 1.0,
    }

    original_filename = "user_uploaded_file.mp3"

    result = pipeline._postprocess_output(
        asr_output=asr_output,
        audio_file_path=audio_file,
        task="transcribe",
        original_filename=original_filename,
    )

    # When original_filename is provided, it should be used as-is
    assert result.get("original_file") == original_filename


def test_postprocess_should_not_use_filename_only(
    tmp_path: pathlib.Path,
    audio_fixture_path: pathlib.Path,
) -> None:
    """
    Verify that passing only a filename as original_filename results in a non-absolute original_file, and that omitting original_filename produces an absolute, existing original_file path.
    
    This test writes a fixture audio file into a subdirectory and asserts the BAD case (original_filename set to the bare filename remains non-absolute) and the GOOD case (original_filename is None and the pipeline produces an absolute path suitable for stabilization).
    """
    # Create a test audio file in a subdirectory
    audio_dir = tmp_path / "uploads"
    audio_dir.mkdir()
    audio_file = audio_dir / "test_audio.wav"
    audio_file.write_bytes(audio_fixture_path.read_bytes())

    mock_backend = MagicMock(spec=ASRBackend)
    pipeline = WhisperPipeline(
        asr_backend=mock_backend,
        storage_backend=None,
        save_transcriptions=False,
    )

    asr_output: dict[str, Any] = {
        "text": "Test transcription",
        "chunks": [],
        "runtime_seconds": 1.0,
    }

    # BAD: Passing just the filename (this is what facade.py was doing)
    result_bad = pipeline._postprocess_output(
        asr_output=asr_output,
        audio_file_path=audio_file,
        task="transcribe",
        original_filename=audio_file.name,  # Just "test_audio.wav"
    )

    # This would break stabilization because it's not an absolute path
    original_file_bad = result_bad.get("original_file")
    assert original_file_bad == "test_audio.wav"  # Just the filename
    assert not pathlib.Path(original_file_bad).is_absolute()

    # GOOD: Not passing original_filename (let pipeline use absolute path)
    result_good = pipeline._postprocess_output(
        asr_output=asr_output,
        audio_file_path=audio_file,
        task="transcribe",
        original_filename=None,  # Don't override
    )

    # This works correctly for stabilization
    original_file_good = result_good.get("original_file")
    assert pathlib.Path(original_file_good).is_absolute()
    assert pathlib.Path(original_file_good).exists()