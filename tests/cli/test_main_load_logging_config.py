"""Tests for load_logging_config function in insanely_fast_whisper_rocm.__main__."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from insanely_fast_whisper_rocm.__main__ import load_logging_config


class TestLoadLoggingConfig:
    """Test the load_logging_config function."""

    def test_load_logging_config_debug_false(self) -> None:
        """Test loading logging config with debug=False."""
        # Setup mock config
        mock_config = {
            "version": 1,
            "root": {"level": "INFO", "handlers": ["console"]},
            "loggers": {"test_logger": {"level": "INFO", "handlers": ["console"]}},
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "level": "INFO",
                    "formatter": "simple",
                }
            },
            "formatters": {"simple": {"format": "%(levelname)s - %(message)s"}},
        }

        # Create temporary config file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(mock_config, f)
            config_path = Path(f.name)

        try:
            # Mock the open function to return our config file
            with patch(
                "builtins.open", return_value=open(config_path, encoding="utf-8")
            ):
                with patch(
                    "insanely_fast_whisper_rocm.__main__.yaml.safe_load",
                    return_value=mock_config,
                ) as mock_yaml_load:
                    with patch("insanely_fast_whisper_rocm.__main__.setup_timezone"):
                        # Execute
                        result = load_logging_config(debug=False)

                        # Verify
                        expected_config = mock_config.copy()
                        assert result == expected_config
                        mock_yaml_load.assert_called_once()

        finally:
            config_path.unlink()

    def test_load_logging_config_debug_true(self) -> None:
        """Test loading logging config with debug=True."""
        # Setup mock config
        mock_config = {
            "version": 1,
            "root": {"level": "INFO", "handlers": ["console"]},
            "loggers": {
                "test_logger": {"level": "INFO", "handlers": ["console"]},
                "another_logger": {"level": "WARNING", "handlers": ["console"]},
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "level": "INFO",
                    "formatter": "simple",
                }
            },
            "formatters": {"simple": {"format": "%(levelname)s - %(message)s"}},
        }

        # Create temporary config file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(mock_config, f)
            config_path = Path(f.name)

        try:
            # Mock the open function to return our config file
            with patch(
                "builtins.open", return_value=open(config_path, encoding="utf-8")
            ):
                with patch(
                    "insanely_fast_whisper_rocm.__main__.yaml.safe_load",
                    return_value=mock_config,
                ) as mock_yaml_load:
                    with patch("insanely_fast_whisper_rocm.__main__.setup_timezone"):
                        # Execute
                        result = load_logging_config(debug=True)

                        # Verify
                        expected_config = mock_config.copy()
                        expected_config["root"]["level"] = "DEBUG"
                        expected_config["loggers"]["test_logger"]["level"] = "DEBUG"
                        expected_config["loggers"]["another_logger"]["level"] = "DEBUG"

                        assert result == expected_config
                        mock_yaml_load.assert_called_once()

        finally:
            config_path.unlink()

    def test_load_logging_config_file_not_found(self) -> None:
        """Test load_logging_config when config file doesn't exist."""
        with patch(
            "insanely_fast_whisper_rocm.__main__.open",
            side_effect=FileNotFoundError("Config file not found"),
        ):
            with patch("insanely_fast_whisper_rocm.__main__.setup_timezone"):
                with pytest.raises(FileNotFoundError):
                    load_logging_config(debug=False)
