"""Tests for dependency injection providers."""

from __future__ import annotations

from collections.abc import Generator
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(scope="session", autouse=True)
def mock_heavy_imports() -> Generator[None, None, None]:
    """Mock heavy imports before any test runs."""
    with patch.dict(
        "sys.modules",
        {
            "torch": MagicMock(),
            "transformers": MagicMock(),
            "transformers.utils": MagicMock(),
            "transformers.utils.logging": MagicMock(),
            "stable_whisper": None,
        },
    ):
        yield


def test_get_asr_pipeline_unwrapped_real() -> None:
    """Test _get_asr_pipeline_unwrapped by accessing the __wrapped__ attribute."""
    # Import after mocking
    from insanely_fast_whisper_api.api.dependencies import get_asr_pipeline

    # The function should have a __wrapped__ attribute that raises RuntimeError
    wrapped_func = get_asr_pipeline.__wrapped__

    with pytest.raises(
        RuntimeError, match="This placeholder should be monkeypatched in tests"
    ):
        wrapped_func()


def test_get_file_handler_unwrapped_real() -> None:
    """Test _get_file_handler_unwrapped by accessing the __wrapped__ attribute."""
    # Import after mocking
    from insanely_fast_whisper_api.api.dependencies import get_file_handler

    # The function should have a __wrapped__ attribute that raises RuntimeError
    wrapped_func = get_file_handler.__wrapped__

    with pytest.raises(
        RuntimeError, match="This placeholder should be monkeypatched in tests"
    ):
        wrapped_func()


def test_normalize_with_fastapi_param_real() -> None:
    """Test _normalize function by calling get_asr_pipeline with FastAPI param."""
    # Import after mocking
    from insanely_fast_whisper_api.api.dependencies import get_asr_pipeline

    # Mock the backend classes to avoid actual initialization
    with (
        patch(
            "insanely_fast_whisper_api.api.dependencies.HuggingFaceBackend"
        ) as mock_backend_class,
        patch(
            "insanely_fast_whisper_api.api.dependencies.WhisperPipeline"
        ) as mock_pipeline_class,
        patch(
            "insanely_fast_whisper_api.api.dependencies.HuggingFaceBackendConfig"
        ) as mock_config_class,
    ):
        # Create mock instances
        mock_config = MagicMock()
        mock_config_class.return_value = mock_config
        mock_backend = MagicMock()
        mock_backend_class.return_value = mock_backend
        mock_pipeline = MagicMock()
        mock_pipeline_class.return_value = mock_pipeline

        class MockFastAPIParam:
            """Mock FastAPI parameter object for testing _normalize function."""

            def __init__(self, default_value: object) -> None:  # noqa: D107
                self.default = default_value
                # Mock the class module to start with "fastapi."
                self.__class__.__module__ = "fastapi.params"

        # Call get_asr_pipeline with a FastAPI param - this should trigger the _normalize function
        mock_param = MockFastAPIParam("custom_model")
        result = get_asr_pipeline(model=mock_param)

        # Verify the pipeline was created
        assert result is mock_pipeline
        # Verify config was created with the normalized value
        mock_config_class.assert_called_once()
        call_kwargs = mock_config_class.call_args[1]
        assert (
            call_kwargs["model_name"] == "custom_model"
        )  # Should use the param's default


def test_get_file_handler_real() -> None:
    """Test get_file_handler function."""
    # Import after mocking
    from insanely_fast_whisper_api.api.dependencies import get_file_handler

    with patch(
        "insanely_fast_whisper_api.api.dependencies.FileHandler"
    ) as mock_file_handler_class:
        mock_handler = MagicMock()
        mock_file_handler_class.return_value = mock_handler

        result = get_file_handler()

        assert result is mock_handler
        mock_file_handler_class.assert_called_once_with()
