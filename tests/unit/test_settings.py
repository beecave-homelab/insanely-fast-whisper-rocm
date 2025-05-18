"""Tests for the settings module."""

import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from config.settings import Settings


def test_settings_defaults():
    """Test that default settings are loaded correctly."""
    settings = Settings()

    # Test some default values
    assert settings.model_name == "distil-whisper/distil-large-v3"
    assert settings.batch_size == 6
    assert settings.web_port == 7860
    assert settings.convert_output_formats == ["txt", "srt"]


def test_environment_variables(monkeypatch):
    """Test that environment variables override defaults."""
    # Set some environment variables
    monkeypatch.setenv("MODEL", "openai/whisper-large-v3")
    monkeypatch.setenv("BATCH_SIZE", "12")
    monkeypatch.setenv("CONVERT_OUTPUT_FORMATS", "txt,srt,vtt")

    settings = Settings()

    # Check that environment variables were used
    assert settings.model_name == "openai/whisper-large-v3"
    assert settings.batch_size == 12
    assert settings.convert_output_formats == ["txt", "srt", "vtt"]


def test_path_resolution():
    """Test that relative paths are resolved against base_dir."""
    base_dir = Path("/tmp/test_app")
    settings = Settings(base_dir=base_dir, uploads_dir="uploads")

    assert settings.uploads_dir == base_dir / "uploads"


def test_invalid_batch_size():
    """Test validation for batch size."""
    with pytest.raises(ValidationError):
        Settings(batch_size=0)  # Must be > 0

    with pytest.raises(ValidationError):
        Settings(batch_size=-1)  # Must be > 0


def test_ensure_dirs_exist(tmp_path):
    """Test that ensure_dirs_exist creates required directories."""
    # Create temporary directories for testing
    test_dirs = {
        "base_dir": tmp_path / "app",
        "uploads_dir": "test_uploads",
        "transcripts_dir": "test_transcripts",
        "logs_dir": "test_logs",
    }

    # Create settings with test directories
    settings = Settings(**test_dirs)

    # Ensure directories don't exist yet
    for dir_path in [
        test_dirs["base_dir"] / test_dirs["uploads_dir"],
        test_dirs["base_dir"] / test_dirs["transcripts_dir"],
        test_dirs["base_dir"] / test_dirs["logs_dir"],
    ]:
        assert not dir_path.exists()

    # Create directories
    settings.ensure_dirs_exist()

    # Verify directories were created
    for dir_path in [
        test_dirs["base_dir"] / test_dirs["uploads_dir"],
        test_dirs["base_dir"] / test_dirs["transcripts_dir"],
        test_dirs["base_dir"] / test_dirs["logs_dir"],
    ]:
        assert dir_path.exists()
        assert dir_path.is_dir()
