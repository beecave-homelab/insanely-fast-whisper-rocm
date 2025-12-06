"""Tests for the FastAPI endpoints."""

import io
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from insanely_fast_whisper_rocm.api.dependencies import get_asr_pipeline
from insanely_fast_whisper_rocm.main import app


@pytest.fixture
def mock_asr_pipeline():
    """Fixture to mock the ASR pipeline."""
    mock_pipeline = MagicMock()
    mock_pipeline.process.return_value = {"text": "test"}

    def get_mock_pipeline():
        return mock_pipeline

    app.dependency_overrides[get_asr_pipeline] = get_mock_pipeline
    yield mock_pipeline
    app.dependency_overrides = {}  # Clear overrides after test


client = TestClient(app)


# Dummy WAV header
DUMMY_WAV_HEADER = b"RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x80>\x00\x00\x00\xfa\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00"


def test_transcription_with_stabilization_options(mock_asr_pipeline):
    """Test that stabilization options are passed to the pipeline for transcription."""
    audio_file = io.BytesIO(DUMMY_WAV_HEADER)
    response = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("test.wav", audio_file, "audio/wav")},
        data={
            "stabilize": True,
            "demucs": True,
            "vad": True,
            "vad_threshold": 0.7,
        },
    )
    assert response.status_code == 200
    mock_asr_pipeline.process.assert_called_once()
    call_args, call_kwargs = mock_asr_pipeline.process.call_args
    assert call_kwargs.get("stabilize") is True
    assert call_kwargs.get("demucs") is True
    assert call_kwargs.get("vad") is True
    assert call_kwargs.get("vad_threshold") == 0.7


def test_translation_with_stabilization_options(mock_asr_pipeline):
    """Test that stabilization options are passed to the pipeline for translation."""
    audio_file = io.BytesIO(DUMMY_WAV_HEADER)
    response = client.post(
        "/v1/audio/translations",
        files={"file": ("test.wav", audio_file, "audio/wav")},
        data={
            "stabilize": True,
            "demucs": False,
            "vad": True,
            "vad_threshold": 0.8,
        },
    )
    assert response.status_code == 200
    mock_asr_pipeline.process.assert_called_once()
    call_args, call_kwargs = mock_asr_pipeline.process.call_args
    assert call_kwargs.get("stabilize") is True
    assert call_kwargs.get("demucs") is False
    assert call_kwargs.get("vad") is True
    assert call_kwargs.get("vad_threshold") == 0.8


def test_transcription_endpoint_validation():
    """Test input validation for the transcription endpoint."""
    # Test with unsupported file format
    with open("tests/data/test.txt", "wb") as f:
        f.write(b"test content")

    with open("tests/data/test.txt", "rb") as f:
        response = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("test.txt", f, "text/plain")},
        )
    assert response.status_code == 400
    assert "Unsupported file format" in response.json()["detail"]


def test_translation_endpoint_validation():
    """Test input validation for the translation endpoint."""
    # Test with unsupported file format
    with open("tests/data/test.txt", "wb") as f:
        f.write(b"test content")

    with open("tests/data/test.txt", "rb") as f:
        response = client.post(
            "/v1/audio/translations",
            files={"file": ("test.txt", f, "text/plain")},
        )
    assert response.status_code == 400
    assert "Unsupported file format" in response.json()["detail"]


@pytest.mark.integration
def test_transcription_endpoint_inference():
    """Test the transcription endpoint with a real audio file.

    This test requires a sample audio file and is marked as integration test.
    """
    # TODO: Add actual test with sample audio file
    pass


@pytest.mark.integration
def test_translation_endpoint_inference():
    """Test the translation endpoint with a real audio file.

    This test requires a sample audio file and is marked as integration test.
    """
    # TODO: Add actual test with sample audio file
    pass
