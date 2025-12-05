"""Tests for Hugging Face model downloading utilities."""

import logging
import os
from collections.abc import Generator
from pathlib import Path  # Used for Path object
from unittest.mock import Mock, patch  # Used for @patch decorator

import pytest

# Import centralized constants instead of local ones
from insanely_fast_whisper_rocm.utils.constants import DEFAULT_MODEL

# Adjust the import path based on your project structure
# This assumes 'insanely_fast_whisper_rocm' is a package in the project root
from insanely_fast_whisper_rocm.utils.download_hf_model import download_model_if_needed

# Constants for testing, mirroring values or concepts from the main script
TEST_HF_CACHE_ENV_VAR = "HF_HOME"  # Standard Hugging Face cache env var

# Configure a logger for tests, or capture logs via caplog fixture
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)  # Use DEBUG for more verbose test output if needed

# --- Test Fixtures (Optional, but good practice) ---


@pytest.fixture
def mock_hf_hub_cache(
    tmp_path_factory: pytest.TempPathFactory,
) -> Generator[str, None, None]:
    """Create an isolated Hugging Face cache directory for tests.

    Yields:
        str: The temporary cache directory path set via the HF_HOME env var.
    """
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
def custom_test_logger() -> logging.Logger:
    """Provide a logger instance for tests.

    Returns:
        logging.Logger: Logger configured for tests in this module.
    """
    return logging.getLogger("test_download_script")


# --- Test Cases ---


@patch("insanely_fast_whisper_rocm.utils.download_hf_model.snapshot_download")
def test_download_default_model_when_none_provided_and_env_var_not_set(
    mock_snapshot_download: Mock,
    mock_hf_hub_cache: str,
    custom_test_logger: logging.Logger,
) -> None:
    """Test that DEFAULT_MODEL is used when no model_name is provided.

    When ``model_name`` is ``None``, the helper must resolve to the centralized
    ``DEFAULT_MODEL`` constant and call ``snapshot_download`` with it.
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


@patch("insanely_fast_whisper_rocm.utils.download_hf_model.snapshot_download")
def test_download_with_explicit_hf_token(
    mock_snapshot_download: Mock,
    mock_hf_hub_cache: str,
    custom_test_logger: logging.Logger,
) -> None:
    """Test download with explicit HF token.

    Verifies that an explicit token is passed through to snapshot_download.
    """
    dummy_path = str(Path(mock_hf_hub_cache) / "model" / "snapshot")
    mock_snapshot_download.return_value = dummy_path
    test_token = "test_hf_token_12345"

    model_path = download_model_if_needed(
        model_name="openai/whisper-tiny",
        hf_token=test_token,
        custom_logger=custom_test_logger,
        cache_dir=mock_hf_hub_cache,
    )

    _, called_kwargs = mock_snapshot_download.call_args
    assert called_kwargs.get("token") == test_token
    assert model_path == dummy_path


@patch("insanely_fast_whisper_rocm.utils.download_hf_model.snapshot_download")
def test_download_with_force_enabled(
    mock_snapshot_download: Mock,
    mock_hf_hub_cache: str,
    custom_test_logger: logging.Logger,
) -> None:
    """Test download with force=True.

    Verifies that force_download is passed correctly to snapshot_download.
    """
    dummy_path = str(Path(mock_hf_hub_cache) / "model" / "snapshot")
    mock_snapshot_download.return_value = dummy_path

    model_path = download_model_if_needed(
        model_name="openai/whisper-tiny",
        force=True,
        custom_logger=custom_test_logger,
        cache_dir=mock_hf_hub_cache,
    )

    _, called_kwargs = mock_snapshot_download.call_args
    assert called_kwargs.get("force_download") is True
    assert model_path == dummy_path


@patch("insanely_fast_whisper_rocm.utils.download_hf_model.snapshot_download")
def test_download_with_force_and_local_files_only(
    mock_snapshot_download: Mock,
    mock_hf_hub_cache: str,
    custom_test_logger: logging.Logger,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test download with force=True and local_files_only=True.

    Verifies warning is logged and force_download is False when local_files_only.
    """
    dummy_path = str(Path(mock_hf_hub_cache) / "model" / "snapshot")
    mock_snapshot_download.return_value = dummy_path

    with caplog.at_level(logging.WARNING):
        model_path = download_model_if_needed(
            model_name="openai/whisper-tiny",
            force=True,
            local_files_only=True,
            custom_logger=custom_test_logger,
            cache_dir=mock_hf_hub_cache,
        )

    # Verify warning is logged
    assert "Force re-download is ignored" in caplog.text

    # Verify force_download is False when local_files_only is True
    _, called_kwargs = mock_snapshot_download.call_args
    assert called_kwargs.get("force_download") is False
    assert called_kwargs.get("local_files_only") is True
    assert model_path == dummy_path


@patch("insanely_fast_whisper_rocm.utils.download_hf_model.snapshot_download")
def test_download_with_allow_and_ignore_patterns(
    mock_snapshot_download: Mock,
    mock_hf_hub_cache: str,
    custom_test_logger: logging.Logger,
) -> None:
    """Test download with allow_patterns and ignore_patterns.

    Verifies that patterns are passed correctly to snapshot_download.
    """
    dummy_path = str(Path(mock_hf_hub_cache) / "model" / "snapshot")
    mock_snapshot_download.return_value = dummy_path

    allow = ["*.bin", "*.json"]
    ignore = ["*.safetensors"]

    model_path = download_model_if_needed(
        model_name="openai/whisper-tiny",
        allow_patterns=allow,
        ignore_patterns=ignore,
        custom_logger=custom_test_logger,
        cache_dir=mock_hf_hub_cache,
    )

    _, called_kwargs = mock_snapshot_download.call_args
    assert called_kwargs.get("allow_patterns") == allow
    assert called_kwargs.get("ignore_patterns") == ignore
    assert model_path == dummy_path


@patch("insanely_fast_whisper_rocm.utils.download_hf_model.snapshot_download")
def test_download_raises_hf_validation_error(
    mock_snapshot_download: Mock,
    mock_hf_hub_cache: str,
    custom_test_logger: logging.Logger,
) -> None:
    """Test download with HFValidationError.

    Verifies that HFValidationError is re-raised properly.
    """
    from huggingface_hub.utils import HFValidationError

    mock_snapshot_download.side_effect = HFValidationError("Invalid repo ID")

    with pytest.raises(HFValidationError):
        download_model_if_needed(
            model_name="invalid/model",
            custom_logger=custom_test_logger,
            cache_dir=mock_hf_hub_cache,
        )


@patch("insanely_fast_whisper_rocm.utils.download_hf_model.snapshot_download")
def test_download_raises_http_error_401(
    mock_snapshot_download: Mock,
    mock_hf_hub_cache: str,
    custom_test_logger: logging.Logger,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test download with HfHubHTTPError 401 (authentication).

    Verifies that 401 errors are handled with appropriate logging.
    """
    from huggingface_hub.utils import HfHubHTTPError

    # Create a mock response with status_code
    mock_response = Mock()
    mock_response.status_code = 401
    error = HfHubHTTPError("Unauthorized", response=mock_response)
    mock_snapshot_download.side_effect = error

    with caplog.at_level(logging.ERROR):
        with pytest.raises(HfHubHTTPError):
            download_model_if_needed(
                model_name="private/model",
                custom_logger=custom_test_logger,
                cache_dir=mock_hf_hub_cache,
            )

    assert "Authentication failed" in caplog.text


@patch("insanely_fast_whisper_rocm.utils.download_hf_model.snapshot_download")
def test_download_raises_http_error_404(
    mock_snapshot_download: Mock,
    mock_hf_hub_cache: str,
    custom_test_logger: logging.Logger,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test download with HfHubHTTPError 404 (not found).

    Verifies that 404 errors are handled with appropriate logging.
    """
    from huggingface_hub.utils import HfHubHTTPError

    mock_response = Mock()
    mock_response.status_code = 404
    error = HfHubHTTPError("Not Found", response=mock_response)
    mock_snapshot_download.side_effect = error

    with caplog.at_level(logging.ERROR):
        with pytest.raises(HfHubHTTPError):
            download_model_if_needed(
                model_name="nonexistent/model",
                custom_logger=custom_test_logger,
                cache_dir=mock_hf_hub_cache,
            )

    assert "not found on Hugging Face Hub" in caplog.text


@patch("insanely_fast_whisper_rocm.utils.download_hf_model.snapshot_download")
def test_download_raises_file_not_found(
    mock_snapshot_download: Mock,
    mock_hf_hub_cache: str,
    custom_test_logger: logging.Logger,
) -> None:
    """Test download with FileNotFoundError.

    Verifies that FileNotFoundError is re-raised when model not in local cache.
    """
    mock_snapshot_download.side_effect = FileNotFoundError("Model not in cache")

    with pytest.raises(FileNotFoundError):
        download_model_if_needed(
            model_name="openai/whisper-tiny",
            local_files_only=True,
            custom_logger=custom_test_logger,
            cache_dir=mock_hf_hub_cache,
        )


@patch("insanely_fast_whisper_rocm.utils.download_hf_model.snapshot_download")
def test_download_raises_os_error(
    mock_snapshot_download: Mock,
    mock_hf_hub_cache: str,
    custom_test_logger: logging.Logger,
) -> None:
    """Test download with OSError.

    Verifies that OSError is re-raised properly.
    """
    mock_snapshot_download.side_effect = OSError("Network error")

    with pytest.raises(OSError):
        download_model_if_needed(
            model_name="openai/whisper-tiny",
            custom_logger=custom_test_logger,
            cache_dir=mock_hf_hub_cache,
        )


@patch("insanely_fast_whisper_rocm.utils.download_hf_model.snapshot_download")
def test_download_raises_runtime_error(
    mock_snapshot_download: Mock,
    mock_hf_hub_cache: str,
    custom_test_logger: logging.Logger,
) -> None:
    """Test download with RuntimeError.

    Verifies that RuntimeError is re-raised properly.
    """
    mock_snapshot_download.side_effect = RuntimeError("Unexpected error")

    with pytest.raises(RuntimeError):
        download_model_if_needed(
            model_name="openai/whisper-tiny",
            custom_logger=custom_test_logger,
            cache_dir=mock_hf_hub_cache,
        )


# CLI tests
@patch("insanely_fast_whisper_rocm.utils.download_hf_model.snapshot_download")
def test_cli_main_success(
    mock_snapshot_download: Mock,
    tmp_path: Path,
) -> None:
    """Test CLI main function with successful download.

    Verifies that the CLI properly invokes download_model_if_needed.
    """
    from click.testing import CliRunner

    from insanely_fast_whisper_rocm.utils.download_hf_model import main

    dummy_path = str(tmp_path / "model" / "snapshot")
    mock_snapshot_download.return_value = dummy_path

    runner = CliRunner()
    result = runner.invoke(main, ["--model", "openai/whisper-tiny"])

    assert result.exit_code == 0
    assert dummy_path in result.output


@patch("insanely_fast_whisper_rocm.utils.download_hf_model.snapshot_download")
def test_cli_main_with_force(
    mock_snapshot_download: Mock,
    tmp_path: Path,
) -> None:
    """Test CLI main function with --force flag.

    Verifies that --force flag is properly passed through.
    """
    from click.testing import CliRunner

    from insanely_fast_whisper_rocm.utils.download_hf_model import main

    dummy_path = str(tmp_path / "model" / "snapshot")
    mock_snapshot_download.return_value = dummy_path

    runner = CliRunner()
    result = runner.invoke(main, ["--model", "openai/whisper-tiny", "--force"])

    assert result.exit_code == 0
    _, called_kwargs = mock_snapshot_download.call_args
    assert called_kwargs.get("force_download") is True


@patch("insanely_fast_whisper_rocm.utils.download_hf_model.snapshot_download")
def test_cli_main_with_check_only(
    mock_snapshot_download: Mock,
    tmp_path: Path,
) -> None:
    """Test CLI main function with --check_only flag.

    Verifies that --check_only sets local_files_only=True.
    """
    from click.testing import CliRunner

    from insanely_fast_whisper_rocm.utils.download_hf_model import main

    dummy_path = str(tmp_path / "model" / "snapshot")
    mock_snapshot_download.return_value = dummy_path

    runner = CliRunner()
    result = runner.invoke(main, ["--model", "openai/whisper-tiny", "--check_only"])

    assert result.exit_code == 0
    _, called_kwargs = mock_snapshot_download.call_args
    assert called_kwargs.get("local_files_only") is True
    assert dummy_path in result.output


@patch("insanely_fast_whisper_rocm.utils.download_hf_model.snapshot_download")
def test_cli_main_with_verbose(
    mock_snapshot_download: Mock,
    tmp_path: Path,
) -> None:
    """Test CLI main function with --verbose flag.

    Verifies that --verbose enables debug logging.
    """
    from click.testing import CliRunner

    from insanely_fast_whisper_rocm.utils.download_hf_model import main

    dummy_path = str(tmp_path / "model" / "snapshot")
    mock_snapshot_download.return_value = dummy_path

    runner = CliRunner()
    result = runner.invoke(main, ["--model", "openai/whisper-tiny", "--verbose"])

    assert result.exit_code == 0
    # Verbose flag enables logging but logs go to stderr, not click output
    # Just verify the command succeeded
    assert dummy_path in result.output


@patch("insanely_fast_whisper_rocm.utils.download_hf_model.snapshot_download")
def test_cli_main_with_patterns(
    mock_snapshot_download: Mock,
    tmp_path: Path,
) -> None:
    """Test CLI main function with allow and ignore patterns.

    Verifies that patterns are converted from tuple to list.
    """
    from click.testing import CliRunner

    from insanely_fast_whisper_rocm.utils.download_hf_model import main

    dummy_path = str(tmp_path / "model" / "snapshot")
    mock_snapshot_download.return_value = dummy_path

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "--model",
            "openai/whisper-tiny",
            "--allow_patterns",
            "*.bin",
            "--allow_patterns",
            "*.json",
            "--ignore_patterns",
            "*.safetensors",
        ],
    )

    assert result.exit_code == 0
    _, called_kwargs = mock_snapshot_download.call_args
    assert called_kwargs.get("allow_patterns") == ["*.bin", "*.json"]
    assert called_kwargs.get("ignore_patterns") == ["*.safetensors"]


@patch("insanely_fast_whisper_rocm.utils.download_hf_model.snapshot_download")
def test_cli_main_hf_hub_error(
    mock_snapshot_download: Mock,
) -> None:
    """Test CLI main function with HfHubHTTPError.

    Verifies that CLI exits with code 4 on HF Hub errors.
    """
    from click.testing import CliRunner
    from huggingface_hub.utils import HfHubHTTPError

    from insanely_fast_whisper_rocm.utils.download_hf_model import main

    mock_response = Mock()
    mock_response.status_code = 404
    mock_snapshot_download.side_effect = HfHubHTTPError(
        "Not Found", response=mock_response
    )

    runner = CliRunner()
    result = runner.invoke(main, ["--model", "nonexistent/model"])

    assert result.exit_code == 4


@patch("insanely_fast_whisper_rocm.utils.download_hf_model.snapshot_download")
def test_cli_main_validation_error(
    mock_snapshot_download: Mock,
) -> None:
    """Test CLI main function with HFValidationError.

    Verifies that CLI exits with code 4 on validation errors.
    """
    from click.testing import CliRunner
    from huggingface_hub.utils import HFValidationError

    from insanely_fast_whisper_rocm.utils.download_hf_model import main

    mock_snapshot_download.side_effect = HFValidationError("Invalid repo ID")

    runner = CliRunner()
    result = runner.invoke(main, ["--model", "invalid/model"])

    assert result.exit_code == 4


@patch("insanely_fast_whisper_rocm.utils.download_hf_model.snapshot_download")
def test_cli_main_critical_error(
    mock_snapshot_download: Mock,
) -> None:
    """Test CLI main function with critical error (OSError, etc.).

    Verifies that CLI exits with code 5 on critical errors.
    """
    from click.testing import CliRunner

    from insanely_fast_whisper_rocm.utils.download_hf_model import main

    mock_snapshot_download.side_effect = OSError("Disk full")

    runner = CliRunner()
    result = runner.invoke(main, ["--model", "openai/whisper-tiny"])

    assert result.exit_code == 5
