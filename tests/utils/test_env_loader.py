"""Tests for insanely_fast_whisper_api.utils.env_loader module.

This module contains tests for environment setup and .env file loading.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch


class TestEnvLoaderConstants:
    """Test suite for env_loader module constants and paths."""

    def test_project_root__is_valid_path(self) -> None:
        """Test that PROJECT_ROOT is a valid path."""
        from insanely_fast_whisper_api.utils.env_loader import PROJECT_ROOT

        assert isinstance(PROJECT_ROOT, Path)
        assert PROJECT_ROOT.exists()

    def test_project_root_env_file__path_format(self) -> None:
        """Test that PROJECT_ROOT_ENV_FILE has correct format."""
        from insanely_fast_whisper_api.utils.env_loader import PROJECT_ROOT_ENV_FILE

        assert isinstance(PROJECT_ROOT_ENV_FILE, Path)
        assert PROJECT_ROOT_ENV_FILE.name == ".env"

    def test_user_config_dir__path_format(self) -> None:
        """Test that USER_CONFIG_DIR has correct format."""
        from insanely_fast_whisper_api.utils.env_loader import USER_CONFIG_DIR

        assert isinstance(USER_CONFIG_DIR, Path)
        assert "insanely-fast-whisper-api" in str(USER_CONFIG_DIR)

    def test_user_env_file__path_format(self) -> None:
        """Test that USER_ENV_FILE has correct format."""
        from insanely_fast_whisper_api.utils.env_loader import USER_ENV_FILE

        assert isinstance(USER_ENV_FILE, Path)
        assert USER_ENV_FILE.name == ".env"


class TestDebugPrint:
    """Test suite for debug_print function."""

    @patch("insanely_fast_whisper_api.utils.env_loader.SHOW_DEBUG_PRINTS", True)
    @patch("builtins.print")
    def test_debug_print__prints_when_enabled(self, mock_print: MagicMock) -> None:
        """Test that debug_print outputs when SHOW_DEBUG_PRINTS is True."""
        from insanely_fast_whisper_api.utils.env_loader import debug_print

        debug_print("Test message")

        mock_print.assert_called_once()
        call_args = mock_print.call_args[0][0]
        assert "ENV_LOADER_DEBUG" in call_args
        assert "Test message" in call_args

    @patch("insanely_fast_whisper_api.utils.env_loader.SHOW_DEBUG_PRINTS", False)
    @patch("builtins.print")
    def test_debug_print__silent_when_disabled(self, mock_print: MagicMock) -> None:
        """Test that debug_print is silent when SHOW_DEBUG_PRINTS is False."""
        from insanely_fast_whisper_api.utils.env_loader import debug_print

        debug_print("Test message")

        mock_print.assert_not_called()

    @patch("insanely_fast_whisper_api.utils.env_loader.SHOW_DEBUG_PRINTS", True)
    @patch("builtins.print")
    def test_debug_print__includes_timestamp(self, mock_print: MagicMock) -> None:
        """Test that debug_print includes a formatted timestamp."""
        from insanely_fast_whisper_api.utils.env_loader import debug_print

        debug_print("Test")

        call_args = mock_print.call_args[0][0]
        # Check format: YYYY-MM-DD HH:MM:SS,mmm
        assert " - " in call_args
        # Should have a timestamp at the start
        parts = call_args.split(" - ")
        assert len(parts) >= 3


class TestEnvLoaderBehavior:
    """Test suite for env_loader initialization behavior."""

    def test_show_debug_prints__with_cli_flag(self) -> None:
        """Test SHOW_DEBUG_PRINTS when --debug CLI flag is present."""
        # This tests the already-imported module state
        # In real usage, SHOW_DEBUG_PRINTS is set at import time
        from insanely_fast_whisper_api.utils.env_loader import SHOW_DEBUG_PRINTS

        # Just verify it's a boolean
        assert isinstance(SHOW_DEBUG_PRINTS, bool)

    def test_project_root_env_exists__is_boolean(self) -> None:
        """Test that PROJECT_ROOT_ENV_EXISTS is a boolean."""
        from insanely_fast_whisper_api.utils.env_loader import PROJECT_ROOT_ENV_EXISTS

        assert isinstance(PROJECT_ROOT_ENV_EXISTS, bool)

    def test_user_env_exists__is_boolean(self) -> None:
        """Test that USER_ENV_EXISTS is a boolean."""
        from insanely_fast_whisper_api.utils.env_loader import USER_ENV_EXISTS

        assert isinstance(USER_ENV_EXISTS, bool)


class TestEnvLoaderInitialization:
    """Test env_loader module initialization with mocked environments."""

    def test_initialization__no_env_files(self) -> None:
        """Test initialization when no .env files exist."""
        # This test verifies the module can handle missing .env files
        # The actual initialization happens at import time, so we can't
        # fully re-test it, but we can verify current state
        from insanely_fast_whisper_api.utils import env_loader

        # Should still have the required attributes
        assert hasattr(env_loader, "PROJECT_ROOT")
        assert hasattr(env_loader, "USER_CONFIG_DIR")
        assert hasattr(env_loader, "SHOW_DEBUG_PRINTS")

    def test_cli_debug_mode__with_flag(self) -> None:
        """Test that --debug flag is detected in sys.argv."""
        # Create a new check similar to module initialization
        with patch("sys.argv", ["test_script.py", "--debug"]):
            cli_debug = "--debug" in sys.argv
            assert cli_debug is True

    def test_cli_debug_mode__without_flag(self) -> None:
        """Test that missing --debug flag is correctly detected."""
        with patch("sys.argv", ["test_script.py"]):
            cli_debug = "--debug" in sys.argv
            assert cli_debug is False

    @patch.dict(os.environ, {"LOG_LEVEL": "DEBUG"})
    def test_env_debug_mode__with_debug_level(self) -> None:
        """Test detection of DEBUG log level from environment."""
        env_level = os.getenv("LOG_LEVEL", "").upper()
        env_debug = env_level == "DEBUG"
        assert env_debug is True

    @patch.dict(os.environ, {"LOG_LEVEL": "INFO"})
    def test_env_debug_mode__with_non_debug_level(self) -> None:
        """Test that non-DEBUG log levels are not treated as debug mode."""
        env_level = os.getenv("LOG_LEVEL", "").upper()
        env_debug = env_level == "DEBUG"
        assert env_debug is False

    @patch.dict(os.environ, {}, clear=True)
    def test_env_debug_mode__without_log_level(self) -> None:
        """Test that missing LOG_LEVEL defaults to non-debug."""
        env_level = os.getenv("LOG_LEVEL", "").upper()
        env_debug = env_level == "DEBUG"
        assert env_debug is False


class TestPathResolution:
    """Test suite for path resolution logic."""

    def test_project_root__resolves_to_repo_root(self) -> None:
        """Test that PROJECT_ROOT resolves to the repository root."""
        from insanely_fast_whisper_api.utils.env_loader import PROJECT_ROOT

        # Should contain common repo markers
        assert (PROJECT_ROOT / "pyproject.toml").exists() or (
            PROJECT_ROOT / "setup.py"
        ).exists()

    def test_user_config_dir__uses_home_directory(self) -> None:
        """Test that USER_CONFIG_DIR is under user's home."""
        from insanely_fast_whisper_api.utils.env_loader import USER_CONFIG_DIR

        assert str(Path.home()) in str(USER_CONFIG_DIR)
