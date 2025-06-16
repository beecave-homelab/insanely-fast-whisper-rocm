"""Unit tests for the new API modules.

This module tests the app factory, dependencies, response formatting,
and file handling components of the refactored API layer.
"""

import os
import tempfile
from io import BytesIO
from unittest.mock import MagicMock, Mock, patch

import pytest
from fastapi import FastAPI, UploadFile
from fastapi.testclient import TestClient

from insanely_fast_whisper_api.api.app import create_app
from insanely_fast_whisper_api.api.dependencies import (
    get_asr_pipeline,
    get_file_handler,
)
from insanely_fast_whisper_api.api.responses import ResponseFormatter
from insanely_fast_whisper_api.core.pipeline import TranscriptionResult, WhisperPipeline
from insanely_fast_whisper_api.main import app  # Assuming your FastAPI app is here
from insanely_fast_whisper_api.utils import (
    RESPONSE_FORMAT_JSON,
    RESPONSE_FORMAT_TEXT,
    FileHandler,
)


class TestAppFactory:
    """Test the FastAPI application factory."""

    def test_create_app_returns_fastapi_instance(self):
        """Test that create_app returns a FastAPI instance."""
        app = create_app()
        assert isinstance(app, FastAPI)

    def test_create_app_has_correct_metadata(self):
        """Test that the app has correct title, description, and version."""
        app = create_app()
        assert app.title == "Insanely Fast Whisper API"
        assert "FastAPI wrapper" in app.description
        assert app.version == "0.1.0"

    def test_create_app_has_routes(self):
        """Test that the app includes the expected routes."""
        app = create_app()
        route_paths = [route.path for route in app.routes]
        assert "/v1/audio/transcriptions" in route_paths
        assert "/v1/audio/translations" in route_paths

    def test_create_app_has_middleware(self):
        """Test that the app has middleware configured."""
        app = create_app()
        # Check that middleware is present (middleware stack is not empty)
        assert len(app.user_middleware) > 0

    def test_app_metadata(self):
        """Test that the app has correct title, description, and version."""
        assert app.title == "Insanely Fast Whisper API"
        assert (
            "FastAPI wrapper around a custom Whisper-based ASR pipeline"
            in app.description
        )
        assert app.version == "0.3.1"


class TestDependencies:
    """Test dependency injection providers."""

    @patch("insanely_fast_whisper_api.api.dependencies.HuggingFaceBackend")
    @patch("insanely_fast_whisper_api.api.dependencies.WhisperPipeline")
    def test_get_asr_pipeline_creates_pipeline(self, mock_pipeline, mock_backend):
        """Test that get_asr_pipeline creates a properly configured pipeline."""
        # Setup mocks
        mock_backend_instance = Mock()
        mock_backend.return_value = mock_backend_instance
        mock_pipeline_instance = Mock()
        mock_pipeline.return_value = mock_pipeline_instance

        # Call the dependency
        result = get_asr_pipeline(
            model="test-model",
            device="cpu",
            batch_size=2,
            dtype="float32",
            better_transformer=True,
            model_chunk_length=15,
        )

        # Verify backend configuration
        mock_backend.assert_called_once()
        backend_config = mock_backend.call_args[1]["config"]
        assert backend_config.model_name == "test-model"
        assert backend_config.device == "cpu"
        assert backend_config.batch_size == 2
        assert backend_config.dtype == "float32"
        assert backend_config.better_transformer == True
        assert backend_config.chunk_length == 15

        # Verify pipeline creation
        mock_pipeline.assert_called_once_with(asr_backend=mock_backend_instance)
        assert result == mock_pipeline_instance

    def test_get_file_handler_returns_file_handler(self):
        """Test that get_file_handler returns a FileHandler instance."""
        result = get_file_handler()
        assert isinstance(result, FileHandler)


class TestResponseFormatter:
    """Test response formatting strategies."""

    def test_format_transcription_json_response(self):
        """Test formatting transcription result as JSON."""
        result = {"text": "Hello world", "metadata": {"duration": 2.5}}

        response = ResponseFormatter.format_transcription(result, RESPONSE_FORMAT_JSON)

        assert response.status_code == 200
        assert response.headers["content-type"] == "application/json"
        # Note: JSONResponse content is accessible via response.body

    def test_format_transcription_text_response(self):
        """Test formatting transcription result as plain text."""
        result = {"text": "Hello world", "metadata": {"duration": 2.5}}

        response = ResponseFormatter.format_transcription(result, RESPONSE_FORMAT_TEXT)

        assert response.status_code == 200
        assert response.headers["content-type"] == "text/plain; charset=utf-8"
        assert response.body == b"Hello world"

    def test_format_translation_json_response(self):
        """Test formatting translation result as JSON."""
        result = {
            "transcription": {"text": "Hello world"},
            "metadata": {"duration": 2.5},
        }

        response = ResponseFormatter.format_translation(result, RESPONSE_FORMAT_JSON)

        assert response.status_code == 200
        assert response.headers["content-type"] == "application/json"

    def test_format_translation_text_response(self):
        """Test formatting translation result as plain text."""
        result = {
            "transcription": {"text": "Hello world"},
            "metadata": {"duration": 2.5},
        }

        response = ResponseFormatter.format_translation(result, RESPONSE_FORMAT_TEXT)

        assert response.status_code == 200
        assert response.headers["content-type"] == "text/plain; charset=utf-8"
        assert response.body == b"Hello world"

    def test_format_translation_text_response_missing_transcription(self):
        """Test formatting translation result when transcription key is missing."""
        result = {"metadata": {"duration": 2.5}}

        response = ResponseFormatter.format_translation(result, RESPONSE_FORMAT_TEXT)

        assert response.status_code == 200
        assert response.body == b""


class TestFileHandler:
    """Test file handling operations."""

    def test_file_handler_initialization(self, tmp_path):
        """Test FileHandler initialization creates upload directory."""
        upload_dir = str(tmp_path / "test_uploads")
        handler = FileHandler(upload_dir=upload_dir)

        assert handler.upload_dir == upload_dir
        assert os.path.exists(upload_dir)

    def test_validate_audio_file_valid_format(self):
        """Test validation passes for supported audio formats."""
        handler = FileHandler()

        # Create mock UploadFile with valid extension
        mock_file = Mock(spec=UploadFile)
        mock_file.filename = "test.mp3"

        # Should not raise exception
        handler.validate_audio_file(mock_file)

    def test_validate_audio_file_invalid_format(self):
        """Test validation fails for unsupported formats."""
        handler = FileHandler()

        # Create mock UploadFile with invalid extension
        mock_file = Mock(spec=UploadFile)
        mock_file.filename = "test.txt"

        # Should raise HTTPException
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            handler.validate_audio_file(mock_file)

        assert exc_info.value.status_code == 400
        assert "Unsupported file format" in str(exc_info.value.detail)

    def test_save_upload_creates_file(self, tmp_path):
        """Test saving uploaded file creates file with unique name."""
        upload_dir = str(tmp_path / "test_uploads")
        handler = FileHandler(upload_dir=upload_dir)

        # Create mock UploadFile
        mock_file = Mock(spec=UploadFile)
        mock_file.filename = "test.mp3"
        mock_file.file = BytesIO(b"test audio content")

        # Save file
        saved_path = handler.save_upload(mock_file)

        # Verify file was created
        assert os.path.exists(saved_path)
        assert saved_path.startswith(upload_dir)
        assert "test.mp3" in saved_path

        # Verify content
        with open(saved_path, "rb") as f:
            content = f.read()
        assert content == b"test audio content"

    def test_cleanup_removes_file(self, tmp_path):
        """Test cleanup removes the specified file."""
        upload_dir = str(tmp_path / "test_uploads")
        handler = FileHandler(upload_dir=upload_dir)

        # Create a test file
        test_file = tmp_path / "test_uploads" / "test.mp3"
        test_file.parent.mkdir(exist_ok=True)
        test_file.write_bytes(b"test content")

        # Verify file exists
        assert test_file.exists()

        # Cleanup file
        handler.cleanup(str(test_file))

        # Verify file is removed
        assert not test_file.exists()

    def test_cleanup_handles_missing_file(self):
        """Test cleanup gracefully handles missing files."""
        handler = FileHandler()

        # Should not raise exception for non-existent file
        handler.cleanup("/non/existent/file.mp3")


class TestMiddleware:
    """Test middleware functionality."""

    def test_request_timing_middleware_logs_requests(self):
        """Test that request timing middleware logs request information."""
        app = create_app()
        client = TestClient(app)

        with patch("insanely_fast_whisper_api.api.middleware.logger") as mock_logger:
            # Make a request to trigger middleware
            response = client.get("/docs")  # Use docs endpoint which should exist

            # Verify logger was called with timing information
            mock_logger.info.assert_called()
            log_call = mock_logger.info.call_args[0][0]
            assert "Request GET" in log_call
            assert "completed in" in log_call
            assert "with status" in log_call
