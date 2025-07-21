import logging
import os
from pathlib import Path  # Used for Path object
from unittest.mock import patch  # Used for @patch decorator

import pytest

# Import centralized constants instead of local ones
from insanely_fast_whisper_api.utils.constants import DEFAULT_MODEL

# Adjust the import path based on your project structure
# This assumes 'insanely_fast_whisper_api' is a package in the project root
from insanely_fast_whisper_api.utils.download_hf_model import (
    download_model_if_needed,
)

# Constants for testing, mirroring values or concepts from the main script
TEST_HF_CACHE_ENV_VAR = "HF_HOME"  # Standard Hugging Face cache env var

# Configure a logger for tests, or capture logs via caplog fixture
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)  # Use DEBUG for more verbose test output if needed

# --- Test Fixtures (Optional, but good practice) ---


@pytest.fixture
def mock_hf_hub_cache(tmp_path_factory):
    """Creates a temporary, isolated Hugging Face cache directory for tests."""
    hf_cache_dir = tmp_path_factory.mktemp("hf_cache")
    original_hf_home = os.environ.get(TEST_HF_CACHE_ENV_VAR)
    os.environ[TEST_HF_CACHE_ENV_VAR] = str(hf_cache_dir)
    logger.info("Using temporary Hugging Face cache: %s", hf_cache_dir)
    yield str(hf_cache_dir)
    if original_hf_home:
        os.environ[TEST_HF_CACHE_ENV_VAR] = original_hf_home
    else:
        del os.environ[TEST_HF_CACHE_ENV_VAR]


@pytest.fixture
def custom_test_logger():
    """Provides a logger instance for tests."""
    return logging.getLogger("test_download_script")


# --- Test Cases ---


@patch("insanely_fast_whisper_api.utils.download_hf_model.snapshot_download")
def test_download_default_model_when_none_provided_and_env_var_not_set(
    mock_snapshot_download, mock_hf_hub_cache, custom_test_logger
):
    """
    Tests that download_model_if_needed attempts to download the centralized DEFAULT_MODEL
    when model_name is None.
    """
    dummy_model_path_str = str(
        Path(mock_hf_hub_cache)
        / "models--distil-whisper--distil-large-v2"  # The actual env var value in container
        / "snapshots"
        / "dummy_default_snapshot"
    )
    mock_snapshot_download.return_value = dummy_model_path_str

    model_path = download_model_if_needed(
        model_name=None, custom_logger=custom_test_logger, cache_dir=mock_hf_hub_cache
    )

    # Verify that the actual DEFAULT_MODEL constant is used (which gets its value from env var in container)
    _called_args, called_kwargs = mock_snapshot_download.call_args  # repo_id is a kwarg
    assert (
        called_kwargs.get("repo_id") == DEFAULT_MODEL
    )  # Use the actual imported constant
    assert called_kwargs.get("cache_dir") == mock_hf_hub_cache
    assert called_kwargs.get("local_files_only") is False
    assert called_kwargs.get("force_download") is False
    assert called_kwargs.get("token") is None

    assert model_path == dummy_model_path_str
    custom_test_logger.info("Test completed. Model path: %s", model_path)


# More tests for download_model_if_needed will be added here.
