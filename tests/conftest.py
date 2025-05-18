"""Configuration and fixtures for pytest."""

import os
import shutil
import tempfile
from pathlib import Path
from typing import Generator

import pytest
from _pytest.monkeypatch import MonkeyPatch

from config.settings import Settings


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Return the path to the test data directory."""
    return Path(__file__).parent / "data"


@pytest.fixture(scope="function")
def temp_dir() -> Generator[Path, None, None]:
    """Create and cleanup a temporary directory for tests."""
    temp_dir = tempfile.mkdtemp()
    try:
        yield Path(temp_dir)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture(scope="function")
def test_settings(temp_dir: Path) -> Generator[Settings, None, None]:
    """Create test settings with temporary directories."""
    # Create subdirectories
    uploads_dir = temp_dir / "uploads"
    transcripts_dir = temp_dir / "transcripts"
    logs_dir = temp_dir / "logs"
    processed_txt_dir = temp_dir / "transcripts-txt"
    processed_srt_dir = temp_dir / "transcripts-srt"

    # Create settings with test directories
    settings = Settings(
        base_dir=temp_dir,
        uploads_dir=uploads_dir,
        transcripts_dir=transcripts_dir,
        logs_dir=logs_dir,
        processed_txt_dir=processed_txt_dir,
        processed_srt_dir=processed_srt_dir,
        debug=True,
        log_level="DEBUG",
    )

    # Ensure directories exist
    settings.ensure_dirs_exist()

    yield settings


@pytest.fixture(scope="function")
def mock_env(monkeypatch: MonkeyPatch) -> None:
    """Set up mock environment variables for testing."""
    # Clear existing environment variables
    for key in os.environ:
        if key.startswith("INSANELY_FAST_WHISPER_"):
            monkeypatch.delenv(key, raising=False)

    # Set test environment variables
    monkeypatch.setenv("MODEL", "distil-whisper/distil-tiny.en")
    monkeypatch.setenv("BATCH_SIZE", "2")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("DEBUG", "True")


@pytest.fixture(scope="function")
def sample_audio_file(test_data_dir: Path, temp_dir: Path) -> Path:
    """Copy a sample audio file to the temporary directory and return its path."""
    # Create a sample audio file if it doesn't exist in test data
    sample_dir = test_data_dir / "audio"
    sample_dir.mkdir(parents=True, exist_ok=True)

    sample_file = sample_dir / "sample.wav"
    if not sample_file.exists():
        # Create a small WAV file (44 bytes of silence for 1ms at 44.1kHz)
        sample_file.write_bytes(
            b'RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00"\xac\x00\x00"\xac\x00\x00\x01\x00\x08\x00data\x00\x00\x00\x00'
        )

    # Copy to temp dir
    dest_file = temp_dir / "sample.wav"
    shutil.copy2(sample_file, dest_file)
    return dest_file
