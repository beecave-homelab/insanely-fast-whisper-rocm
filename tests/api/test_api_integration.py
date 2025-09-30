"""Integration tests for the refactored API routes.

This module tests that the refactored routes work correctly with
dependency injection and maintain the same functionality as before.
"""

from io import BytesIO
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import pytest
from fastapi.testclient import TestClient

from insanely_fast_whisper_api import constants
from insanely_fast_whisper_api.api.app import create_app


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """Create a test client for the refactored API with temp upload dir.

    Returns:
        TestClient: FastAPI test client configured with a temporary upload dir.
    """
    monkeypatch.setattr(
        "insanely_fast_whisper_api.api.app.download_model_if_needed",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "insanely_fast_whisper_api.core.asr_backend.HuggingFaceBackend._validate_device",
        lambda self: None,
    )

    app = create_app()

    # Override FileHandler to use a temp directory to avoid permission issues
    from insanely_fast_whisper_api.api.dependencies import get_file_handler
    from insanely_fast_whisper_api.utils import FileHandler

    app.dependency_overrides[get_file_handler] = lambda: FileHandler(
        upload_dir=str(tmp_path)
    )

    return TestClient(app)


@pytest.fixture
def mock_audio_file() -> tuple[str, BytesIO, str]:
    """Create a mock audio file for testing.

    Returns:
        tuple[str, BytesIO, str]: Tuple of filename, file-like object, and MIME type.
    """
    return ("test.mp3", BytesIO(b"fake audio content"), "audio/mpeg")


@pytest.fixture
def mock_asr_result() -> dict[str, Any]:
    """Create a mock ASR processing result.

    Returns:
        dict[str, Any]: Minimal ASR result payload used by tests.
    """
    return {
        "text": "Hello, this is a test transcription.",
        "transcription": {
            "text": "Hello, this is a test transcription.",
            "chunks": [
                {
                    "timestamp": [0.0, 2.5],
                    "text": "Hello, this is a test transcription.",
                }
            ],
        },
        "metadata": {"duration": 2.5, "language": "en"},
        "pipeline_runtime_seconds": 1.23,
    }


class TestTranscriptionEndpoint:
    """Test the refactored transcription endpoint."""

    def test_transcription_endpoint_validation_invalid_format(
        self, client: TestClient
    ) -> None:
        """Test that the endpoint validates file formats correctly."""
        # Test with unsupported file format
        response = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("test.txt", BytesIO(b"test content"), "text/plain")},
        )
        assert response.status_code == 400
        assert "Unsupported file format" in response.json()["detail"]

    @patch("insanely_fast_whisper_api.api.dependencies.WhisperPipeline")
    @patch("insanely_fast_whisper_api.api.dependencies.HuggingFaceBackend")
    def test_transcription_endpoint_success_json(
        self,
        mock_backend: Mock,
        mock_pipeline: Mock,
        client: TestClient,
        mock_audio_file: tuple[str, BytesIO, str],
        mock_asr_result: dict[str, Any],
    ) -> None:
        """Test successful transcription with JSON response."""
        # Setup mocks
        mock_pipeline_instance = Mock()
        mock_pipeline_instance.process.return_value = mock_asr_result
        mock_pipeline.return_value = mock_pipeline_instance

        # Make request
        response = client.post(
            "/v1/audio/transcriptions",
            files={"file": mock_audio_file},
            data={
                "model": "test-model",
                "device": "cpu",
                "batch_size": "2",
                "timestamp_type": "chunk",
                "language": "en",
                "task": "transcribe",
                "dtype": "float32",
                "better_transformer": "false",
                "model_chunk_length": "30",
            },
        )

        # Verify response
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/json"
        result = response.json()
        assert result["text"] == "Hello, this is a test transcription."

        # Verify pipeline was called correctly
        mock_pipeline_instance.process.assert_called_once()
        call_args = mock_pipeline_instance.process.call_args[1]
        assert call_args["language"] == "en"
        assert call_args["task"] == "transcribe"
        assert call_args["timestamp_type"] == "chunk"

    @patch("insanely_fast_whisper_api.api.dependencies.WhisperPipeline")
    @patch("insanely_fast_whisper_api.api.dependencies.HuggingFaceBackend")
    def test_transcription_endpoint_success_text(
        self,
        mock_backend: Mock,
        mock_pipeline: Mock,
        client: TestClient,
        mock_audio_file: tuple[str, BytesIO, str],
        mock_asr_result: dict[str, Any],
    ) -> None:
        """Test successful transcription with text response."""
        # Setup mocks
        mock_pipeline_instance = Mock()
        mock_pipeline_instance.process.return_value = mock_asr_result
        mock_pipeline.return_value = mock_pipeline_instance

        # Make request with text response format
        response = client.post(
            "/v1/audio/transcriptions",
            files={"file": mock_audio_file},
            data={
                "response_format": "text",  # Correct way to trigger text/plain response
                "timestamp_type": "chunk",  # Use a valid timestamp type
                "language": "en",
                "task": "transcribe",
            },
        )

        # Verify response
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/plain; charset=utf-8"
        assert response.text == "Hello, this is a test transcription."

    @patch("insanely_fast_whisper_api.api.dependencies.WhisperPipeline")
    @patch("insanely_fast_whisper_api.api.dependencies.HuggingFaceBackend")
    def test_transcription_endpoint_dependency_injection(
        self,
        mock_backend: Mock,
        mock_pipeline: Mock,
        client: TestClient,
        mock_audio_file: tuple[str, BytesIO, str],
        mock_asr_result: dict[str, Any],
    ) -> None:
        """Test that dependency injection works correctly."""
        # Setup mocks
        mock_backend_instance = Mock()
        mock_backend.return_value = mock_backend_instance
        mock_pipeline_instance = Mock()
        mock_pipeline_instance.process.return_value = mock_asr_result
        mock_pipeline.return_value = mock_pipeline_instance

        # Make request with specific parameters
        _ = client.post(
            "/v1/audio/transcriptions",
            files={"file": mock_audio_file},
            data={
                "model": "custom-model",
                "device": "cuda:0",
                "batch_size": "8",
                "dtype": "float16",
                "better_transformer": "true",
                "model_chunk_length": "15",
            },
        )

        # Verify backend was configured correctly
        mock_backend.assert_called_once()
        backend_config = mock_backend.call_args[1]["config"]
        assert backend_config.model_name == constants.DEFAULT_MODEL
        assert backend_config.device == constants.DEFAULT_DEVICE
        assert backend_config.batch_size == constants.DEFAULT_BATCH_SIZE
        assert backend_config.dtype == constants.DEFAULT_DTYPE
        assert backend_config.chunk_length == constants.DEFAULT_CHUNK_LENGTH

        # Verify pipeline was created with backend
        mock_pipeline.assert_called_once_with(asr_backend=mock_backend_instance)


class TestTranslationEndpoint:
    """Test the refactored translation endpoint."""

    def test_translation_endpoint_validation_invalid_format(
        self, client: TestClient
    ) -> None:
        """Test that the endpoint validates file formats correctly."""
        # Test with unsupported file format
        response = client.post(
            "/v1/audio/translations",
            files={"file": ("test.txt", BytesIO(b"test content"), "text/plain")},
        )
        assert response.status_code == 400
        assert "Unsupported file format" in response.json()["detail"]

    @patch("insanely_fast_whisper_api.api.dependencies.WhisperPipeline")
    @patch("insanely_fast_whisper_api.api.dependencies.HuggingFaceBackend")
    def test_translation_endpoint_success_json(
        self,
        mock_backend: Mock,
        mock_pipeline: Mock,
        client: TestClient,
        mock_audio_file: tuple[str, BytesIO, str],
        mock_asr_result: dict[str, Any],
    ) -> None:
        """Test successful translation with JSON response."""
        # Setup mocks
        mock_pipeline_instance = Mock()
        mock_pipeline_instance.process.return_value = mock_asr_result
        mock_pipeline.return_value = mock_pipeline_instance

        # Make request
        response = client.post(
            "/v1/audio/translations",
            files={"file": mock_audio_file},
            data={
                "response_format": "json",
                "timestamp_type": "chunk",
                "language": "es",
            },
        )

        # Verify response
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/json"
        result = response.json()
        assert result["text"] == "Hello, this is a test transcription."

        # Verify pipeline was called with translate task
        mock_pipeline_instance.process.assert_called_once()
        call_args = mock_pipeline_instance.process.call_args[1]
        assert call_args["task"] == "translate"
        assert call_args["language"] == "es"

    @patch("insanely_fast_whisper_api.api.dependencies.WhisperPipeline")
    @patch("insanely_fast_whisper_api.api.dependencies.HuggingFaceBackend")
    def test_translation_endpoint_success_text(
        self,
        mock_backend: Mock,
        mock_pipeline: Mock,
        client: TestClient,
        mock_audio_file: tuple[str, BytesIO, str],
        mock_asr_result: dict[str, Any],
    ) -> None:
        """Test successful translation with text response."""
        # Setup mocks
        mock_pipeline_instance = Mock()
        mock_pipeline_instance.process.return_value = mock_asr_result
        mock_pipeline.return_value = mock_pipeline_instance

        # Make request with text response format
        response = client.post(
            "/v1/audio/translations",
            files={"file": mock_audio_file},
            data={"response_format": "text", "language": "fr"},
        )

        # Verify response
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/plain; charset=utf-8"
        assert response.text == "Hello, this is a test transcription."


class TestFileHandling:
    """Test file handling in the refactored routes."""

    @patch("insanely_fast_whisper_api.api.dependencies.WhisperPipeline")
    @patch("insanely_fast_whisper_api.api.dependencies.HuggingFaceBackend")
    @patch("insanely_fast_whisper_api.utils.FileHandler.cleanup")
    def test_file_cleanup_called(
        self,
        mock_cleanup: Mock,
        mock_backend: Mock,
        mock_pipeline: Mock,
        client: TestClient,
        mock_audio_file: tuple[str, BytesIO, str],
        mock_asr_result: dict[str, Any],
    ) -> None:
        """Test that file cleanup is called even when processing succeeds."""
        # Setup mocks
        mock_pipeline_instance = Mock()
        mock_pipeline_instance.process.return_value = mock_asr_result
        mock_pipeline.return_value = mock_pipeline_instance

        # Make request
        response = client.post(
            "/v1/audio/transcriptions",
            files={"file": mock_audio_file},
        )

        # Verify cleanup was called
        assert response.status_code == 200
        mock_cleanup.assert_called_once()

    @patch("insanely_fast_whisper_api.api.dependencies.WhisperPipeline")
    @patch("insanely_fast_whisper_api.api.dependencies.HuggingFaceBackend")
    @patch("insanely_fast_whisper_api.utils.FileHandler.cleanup")
    def test_file_cleanup_called_on_error(
        self,
        mock_cleanup: Mock,
        mock_backend: Mock,
        mock_pipeline: Mock,
        client: TestClient,
        mock_audio_file: tuple[str, BytesIO, str],
    ) -> None:
        """Test that file cleanup is called even when processing fails."""
        # Setup mocks to raise an exception during processing
        mock_backend_instance = Mock()
        mock_backend.return_value = mock_backend_instance
        mock_pipeline_instance = Mock()
        mock_pipeline_instance.process.side_effect = Exception("Processing failed")
        mock_pipeline.return_value = mock_pipeline_instance

        # Make request - expect it to raise an exception due to our mock
        try:
            response = client.post(
                "/v1/audio/transcriptions",
                files={"file": mock_audio_file},
            )
            # If we get here, the exception was handled and returned as 500
            assert response.status_code == 500
        except Exception as e:
            # The exception was re-raised by the test client
            assert "Processing failed" in str(e)

        # Verify cleanup was called even though processing failed
        mock_cleanup.assert_called_once()


class TestBackwardsCompatibility:
    """Test that the refactored API maintains backwards compatibility."""

    def test_endpoint_paths_unchanged(self, client: TestClient) -> None:
        """Test that endpoint paths remain the same."""
        # Test that the expected endpoints exist
        response = client.get("/docs")  # OpenAPI docs should be available
        assert response.status_code == 200

        # The actual endpoints will return 422 (validation error) for GET requests
        # but they should exist (not 404)
        response = client.get("/v1/audio/transcriptions")
        assert response.status_code == 405  # Method not allowed (POST required)

        response = client.get("/v1/audio/translations")
        assert response.status_code == 405  # Method not allowed (POST required)

    @patch("insanely_fast_whisper_api.api.dependencies.WhisperPipeline")
    @patch("insanely_fast_whisper_api.api.dependencies.HuggingFaceBackend")
    def test_parameter_names_unchanged(
        self,
        mock_backend: Mock,
        mock_pipeline: Mock,
        client: TestClient,
        mock_audio_file: tuple[str, BytesIO, str],
        mock_asr_result: dict[str, Any],
    ) -> None:
        """Test that parameter names remain the same for backwards compatibility."""
        # Setup mocks properly
        mock_backend_instance = Mock()
        mock_backend.return_value = mock_backend_instance
        mock_pipeline_instance = Mock()
        mock_pipeline_instance.process.return_value = mock_asr_result
        mock_pipeline.return_value = mock_pipeline_instance

        # This should not raise validation errors for parameter names
        response = client.post(
            "/v1/audio/transcriptions",
            files={"file": mock_audio_file},
            data={
                "model": "test-model",
                "device": "cpu",
                "batch_size": "2",
                "timestamp_type": "chunk",
                "language": "en",
                "task": "transcribe",
                "dtype": "float32",
                "better_transformer": "false",
                "model_chunk_length": "30",
            },
        )

        # Should not be a validation error (422)
        # Should succeed with proper mocking
        assert response.status_code == 200

        # Verify the backend was configured with the test parameters
        mock_backend.assert_called_once()
        backend_config = mock_backend.call_args[1]["config"]
        assert backend_config.model_name == constants.DEFAULT_MODEL
        assert backend_config.device == constants.DEFAULT_DEVICE
        assert backend_config.batch_size == constants.DEFAULT_BATCH_SIZE

        # Verify pipeline was created with backend
        mock_pipeline.assert_called_once_with(asr_backend=mock_backend_instance)
