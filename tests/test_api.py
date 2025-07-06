"""Tests for the FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient

from insanely_fast_whisper_api.main import app

client = TestClient(app)


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
