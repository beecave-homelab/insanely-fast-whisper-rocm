"""Tests for the FastAPI endpoints."""

from __future__ import annotations

import io
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from insanely_fast_whisper_api.api.dependencies import (
    get_asr_pipeline,
    get_file_handler,
)
from insanely_fast_whisper_api.main import app
from insanely_fast_whisper_api.utils import FileHandler


@pytest.fixture
def mock_asr_pipeline() -> Iterator[MagicMock]:
    """Provide a mocked ASR pipeline and override dependency.

    Yields:
        MagicMock: The mocked pipeline instance with ``process`` configured.
    """
    mock_pipeline = MagicMock()
    mock_pipeline.process.return_value = {"text": "test"}

    def get_mock_pipeline() -> MagicMock:
        return mock_pipeline

    app.dependency_overrides[get_asr_pipeline] = get_mock_pipeline
    try:
        yield mock_pipeline
    finally:
        app.dependency_overrides = {}  # Clear overrides after test


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """Create a TestClient overriding FileHandler to use a temp dir.

    Args:
        tmp_path: Temporary directory provided by pytest.
        monkeypatch: Pytest fixture to temporarily modify attributes during tests.

    Yields:
        TestClient: Configured client instance.
    """
    monkeypatch.setattr(
        "insanely_fast_whisper_api.api.app.download_model_if_needed",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "insanely_fast_whisper_api.core.asr_backend.HuggingFaceBackend._validate_device",
        lambda self: None,
    )
    app.dependency_overrides[get_file_handler] = lambda: FileHandler(
        upload_dir=str(tmp_path)
    )
    client = TestClient(app)
    try:
        yield client
    finally:
        # Clean overrides to avoid leakage into other tests
        app.dependency_overrides.pop(get_file_handler, None)


# Dummy WAV header
DUMMY_WAV_HEADER = b"RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x80>\x00\x00\x00\xfa\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00"


def test_transcription_with_stabilization_options(
    mock_asr_pipeline: MagicMock, client: TestClient
) -> None:
    """Test that stabilization options are NOT passed to process(), but are handled separately.

    The stabilize/demucs/vad/vad_threshold params are handled by stabilize_timestamps()
    which is called AFTER asr_pipeline.process(). The process() method should only
    receive the core ASR parameters.
    """
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
    _, call_kwargs = mock_asr_pipeline.process.call_args

    # Verify stabilization params are NOT passed to process()
    # They are handled by separate stabilize_timestamps() post-processing
    assert "stabilize" not in call_kwargs
    assert "demucs" not in call_kwargs
    assert "vad" not in call_kwargs
    assert "vad_threshold" not in call_kwargs

    # Verify valid params ARE passed
    assert "audio_file_path" in call_kwargs
    assert call_kwargs.get("task") == "transcribe"


def test_translation_with_stabilization_options(
    mock_asr_pipeline: MagicMock, client: TestClient
) -> None:
    """Test that stabilization options are NOT passed to process(), but are handled separately.

    The stabilize/demucs/vad/vad_threshold params are handled by stabilize_timestamps()
    which is called AFTER asr_pipeline.process(). The process() method should only
    receive the core ASR parameters.
    """
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
    _, call_kwargs = mock_asr_pipeline.process.call_args

    # Verify stabilization params are NOT passed to process()
    # They are handled by separate stabilize_timestamps() post-processing
    assert "stabilize" not in call_kwargs
    assert "demucs" not in call_kwargs
    assert "vad" not in call_kwargs
    assert "vad_threshold" not in call_kwargs

    # Verify valid params ARE passed
    assert "audio_file_path" in call_kwargs
    assert call_kwargs.get("task") == "translate"


def test_transcription_endpoint_validation(client: TestClient, tmp_path: Path) -> None:
    """Test input validation for the transcription endpoint."""
    # Create unsupported file under a temp directory
    bad_file = tmp_path / "test.txt"
    bad_file.write_bytes(b"test content")

    with bad_file.open("rb") as f:
        response = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("test.txt", f, "text/plain")},
        )
    assert response.status_code == 400
    assert "Unsupported file format" in response.json()["detail"]


def test_translation_endpoint_validation(client: TestClient, tmp_path: Path) -> None:
    """Test input validation for the translation endpoint."""
    bad_file = tmp_path / "test.txt"
    bad_file.write_bytes(b"test content")

    with bad_file.open("rb") as f:
        response = client.post(
            "/v1/audio/translations",
            files={"file": ("test.txt", f, "text/plain")},
        )
    assert response.status_code == 400
    assert "Unsupported file format" in response.json()["detail"]


@pytest.mark.integration
def test_transcription_endpoint_inference() -> None:
    """Test the transcription endpoint with a real audio file.

    This test requires a sample audio file and is marked as integration test.
    """
    # TODO: Add actual test with sample audio file
    pass


@pytest.mark.integration
def test_translation_endpoint_inference() -> None:
    """Test the translation endpoint with a real audio file.

    This test requires a sample audio file and is marked as integration test.
    """
    # TODO: Add actual test with sample audio file
    pass
