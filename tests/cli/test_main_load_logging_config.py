"""Tests for load_logging_config function in insanely_fast_whisper_api.__main__."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from insanely_fast_whisper_api.__main__ import load_logging_config


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
            # Mock the path resolution
            with patch("insanely_fast_whisper_api.__main__.Path") as mock_path_class:
                mock_path_instance = mock_path_class.return_value
                mock_path_instance.parent = (
                    Path(__file__).parent.parent / "insanely_fast_whisper_api"
                )
                mock_path_instance.__truediv__ = lambda self, x: config_path

                with patch(
                    "insanely_fast_whisper_api.__main__.yaml.safe_load",
                    return_value=mock_config,
                ) as mock_yaml_load:
                    with patch("insanely_fast_whisper_api.__main__.setup_timezone"):
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
            # Mock the path resolution
            with patch("insanely_fast_whisper_api.__main__.Path") as mock_path_class:
                mock_path_instance = mock_path_class.return_value
                mock_path_instance.parent = (
                    Path(__file__).parent.parent / "insanely_fast_whisper_api"
                )
                mock_path_instance.__truediv__ = lambda self, x: config_path

                with patch(
                    "insanely_fast_whisper_api.__main__.yaml.safe_load",
                    return_value=mock_config,
                ) as mock_yaml_load:
                    with patch("insanely_fast_whisper_api.__main__.setup_timezone"):
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
        with patch("insanely_fast_whisper_api.__main__.Path") as mock_path_class:
            mock_path_instance = mock_path_class.return_value
            mock_path_instance.parent = (
                Path(__file__).parent.parent / "insanely_fast_whisper_api"
            )
            mock_path_instance.__truediv__ = lambda self, x: Path("nonexistent.yaml")

            with patch("insanely_fast_whisper_api.__main__.setup_timezone"):
                # Execute & Verify
                with pytest.raises(FileNotFoundError):
                    load_logging_config(debug=False)
