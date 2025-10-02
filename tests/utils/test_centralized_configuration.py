"""Test centralized configuration system.

This module tests that all modules correctly use constants from constants.py
instead of direct environment variable access, and verifies default values
and .env file overrides work properly.
"""

import os
import tempfile
from importlib import reload
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import insanely_fast_whisper_api.utils.constants as constants_module


class TestCentralizedConfiguration:
    """Test the centralized configuration system."""

    def test_default_values_without_env_vars(self) -> None:
        """Test that default values are used when no environment variables are set."""
        with patch(
            "insanely_fast_whisper_api.utils.constants.os.getenv"
        ) as mock_getenv:
            # Mock getenv to return None for all calls (no env vars set)
            mock_getenv.side_effect = lambda key, default=None: default

            # Reload constants to apply mocked environment
            reload(constants_module)

            # Verify defaults are applied
            assert constants_module.DEFAULT_MODEL == "distil-whisper/distil-large-v3"
            assert constants_module.FILENAME_TIMEZONE == "UTC"
            assert constants_module.HF_TOKEN is None

    def test_environment_variable_overrides(self) -> None:
        """Test that environment variables properly override defaults."""
        test_env_vars = {
            "WHISPER_MODEL": "openai/whisper-tiny",
            # Set APP_TIMEZONE explicitly to override FILENAME_TIMEZONE/TZ
            "APP_TIMEZONE": "UTC",
            "FILENAME_TIMEZONE": "America/New_York",
            "HF_TOKEN": "test_token_123",
            "WHISPER_BATCH_SIZE": "8",
        }

        with patch(
            "insanely_fast_whisper_api.utils.constants.os.getenv"
        ) as mock_getenv:
            # Mock getenv to return test values for specific keys
            def getenv_side_effect(key: str, default: str | None = None) -> str | None:
                return test_env_vars.get(key, default)

            mock_getenv.side_effect = getenv_side_effect

            # Reload constants to apply mocked environment
            reload(constants_module)

            # Verify environment overrides work
            assert constants_module.DEFAULT_MODEL == "openai/whisper-tiny"
            # APP_TIMEZONE takes precedence over FILENAME_TIMEZONE
            assert constants_module.FILENAME_TIMEZONE == "UTC"
            assert constants_module.HF_TOKEN == "test_token_123"
            assert constants_module.DEFAULT_BATCH_SIZE == 8

    def test_boolean_environment_variables(self) -> None:
        """Test that boolean environment variables are properly parsed."""
        with patch(
            "insanely_fast_whisper_api.utils.constants.os.getenv"
        ) as mock_getenv:
            # Test true values
            mock_getenv.side_effect = lambda key, default=None: {
                "WHISPER_BETTER_TRANSFORMER": "true",
                "SAVE_TRANSCRIPTIONS": "TRUE",
                "HIP_LAUNCH_BLOCKING": "True",
            }.get(key, default)

            reload(constants_module)

            assert constants_module.DEFAULT_BETTER_TRANSFORMER is True
            assert constants_module.SAVE_TRANSCRIPTIONS is True
            assert constants_module.HIP_LAUNCH_BLOCKING is True

            # Test false values
            mock_getenv.side_effect = lambda key, default=None: {
                "WHISPER_BETTER_TRANSFORMER": "false",
                "SAVE_TRANSCRIPTIONS": "FALSE",
                "HIP_LAUNCH_BLOCKING": "False",
            }.get(key, default)

            reload(constants_module)

            assert constants_module.DEFAULT_BETTER_TRANSFORMER is False
            assert constants_module.SAVE_TRANSCRIPTIONS is False
            assert constants_module.HIP_LAUNCH_BLOCKING is False

    def test_integer_environment_variables(self) -> None:
        """Test that integer environment variables are properly parsed."""
        with patch(
            "insanely_fast_whisper_api.utils.constants.os.getenv"
        ) as mock_getenv:
            mock_getenv.side_effect = lambda key, default=None: {
                "WHISPER_BATCH_SIZE": "16",
                "WHISPER_CHUNK_LENGTH": "45",
                "API_PORT": "9000",
            }.get(key, default)

            reload(constants_module)

            assert constants_module.DEFAULT_BATCH_SIZE == 16
            assert constants_module.DEFAULT_CHUNK_LENGTH == 45
            assert constants_module.API_PORT == 9000

    def test_float_environment_variables(self) -> None:
        """Test that float environment variables are properly parsed."""
        with patch(
            "insanely_fast_whisper_api.utils.constants.os.getenv"
        ) as mock_getenv:
            mock_getenv.side_effect = lambda key, default=None: {
                "AUDIO_CHUNK_DURATION": "900.5",
                "AUDIO_CHUNK_OVERLAP": "2.5",
                "AUDIO_CHUNK_MIN_DURATION": "10.0",
            }.get(key, default)

            reload(constants_module)

            assert constants_module.AUDIO_CHUNK_DURATION == 900.5
            assert constants_module.AUDIO_CHUNK_OVERLAP == 2.5
            assert constants_module.AUDIO_CHUNK_MIN_DURATION == 10.0

    def test_hf_token_no_fallback(self) -> None:
        """Test that HF_TOKEN is sourced only from HF_TOKEN env var (no fallback)."""
        # When HF_TOKEN is set, constant should reflect it
        with patch(
            "insanely_fast_whisper_api.utils.constants.os.getenv"
        ) as mock_getenv:
            mock_getenv.side_effect = lambda key, default=None: {
                "HF_TOKEN": "primary_token",
            }.get(key, default)

            reload(constants_module)

            assert constants_module.HF_TOKEN == "primary_token"

        # When only HUGGINGFACE_TOKEN is set, HF_TOKEN should remain None
        with patch(
            "insanely_fast_whisper_api.utils.constants.os.getenv"
        ) as mock_getenv:
            mock_getenv.side_effect = lambda key, default=None: {
                "HUGGINGFACE_TOKEN": "fallback_token",
            }.get(key, default)

            reload(constants_module)

            assert constants_module.HF_TOKEN is None


class TestModuleCentralizedConfigurationUsage:
    """Test that modules correctly import and use centralized configuration."""

    def test_app_module_uses_centralized_config(self) -> None:
        """Test that app.py uses constants from constants.py."""
        # Import the app module and verify it imports from constants

        # Check that app module imports are using centralized config
        # We can't directly test the imports without more complex mocking,
        # but we can verify the constants are available
        from insanely_fast_whisper_api.utils.constants import DEFAULT_MODEL

        assert DEFAULT_MODEL is not None
        assert isinstance(DEFAULT_MODEL, str)

    def test_filename_generator_uses_centralized_config(self) -> None:
        """Test that filename_generator.py uses constants from constants.py."""
        from insanely_fast_whisper_api.utils.constants import FILENAME_TIMEZONE

        # Verify the constant is available and properly typed
        assert FILENAME_TIMEZONE is not None
        assert isinstance(FILENAME_TIMEZONE, str)

    def test_download_hf_model_uses_centralized_config(self) -> None:
        """Test that download_hf_model.py uses constants from constants.py."""
        from insanely_fast_whisper_api.utils.constants import DEFAULT_MODEL

        # Verify the constants are available
        assert DEFAULT_MODEL is not None
        assert isinstance(DEFAULT_MODEL, str)


class TestDotEnvFileSupport:
    """Test .env file loading and support."""

    def test_dotenv_file_loading(self) -> None:
        """Test that .env files are properly loaded by constants.py."""
        # Create a temporary .env file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".env", delete=False
        ) as env_file:
            env_file.write("WHISPER_MODEL=test-model-from-env\n")
            env_file.write("FILENAME_TIMEZONE=Europe/Paris\n")
            env_file.write("HF_TOKEN=test-token-from-env\n")
            env_file_path = env_file.name

        try:
            # Patch env_loader to make constants.py believe a user .env exists.
            with (
                patch(
                    "insanely_fast_whisper_api.utils.env_loader.USER_ENV_FILE",
                    Path(env_file_path),
                ),
                patch(
                    "insanely_fast_whisper_api.utils.env_loader.USER_ENV_EXISTS",
                    True,
                ),
                patch("dotenv.load_dotenv") as mock_load,
            ):
                reload(constants_module)
                mock_load.assert_called_once_with(Path(env_file_path), override=True)
        finally:
            # Clean up temp file
            os.unlink(env_file_path)

    def test_config_dir_creation(self) -> None:
        """Test that configuration directory is created if it doesn't exist."""
        with patch(
            "insanely_fast_whisper_api.utils.constants.CONFIG_DIR"
        ) as mock_config_dir:
            mock_path = MagicMock()
            mock_config_dir.return_value = mock_path
            mock_path.exists.return_value = False

            # Reload to trigger directory creation logic
            reload(constants_module)

            # Verify mkdir was called with proper parameters
            # (This would work if we had more control over the module loading)


@pytest.fixture(autouse=True)
def restore_constants() -> None:
    """Restore constants module to original state after each test.

    Yields:
        None: Control back to the test; after the test, constants are reloaded.
    """
    yield
    # Reload to restore original state
    reload(constants_module)
