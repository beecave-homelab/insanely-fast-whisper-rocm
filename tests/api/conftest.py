"""Shared fixtures for API tests."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from insanely_fast_whisper_rocm.api.dependencies import (
    get_asr_pipeline,
    get_file_handler,
)
from insanely_fast_whisper_rocm.main import app
from insanely_fast_whisper_rocm.utils import FileHandler


@pytest.fixture
def mock_asr_pipeline() -> Iterator[MagicMock]:
    """Provide a mocked ASR pipeline and override dependency.

    Yields:
        MagicMock: The mocked pipeline instance with ``process`` configured.
    """
    mock_pipeline = MagicMock()
    mock_pipeline.process.return_value = {"text": "test"}
    mock_pipeline.asr_backend.config.model_name = "test-model"

    def get_mock_pipeline() -> MagicMock:
        return mock_pipeline

    app.dependency_overrides[get_asr_pipeline] = get_mock_pipeline
    try:
        yield mock_pipeline
    finally:
        app.dependency_overrides.pop(get_asr_pipeline, None)


@pytest.fixture
def mock_orchestrator(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Mock the TranscriptionOrchestrator and override its factory.

    Args:
        monkeypatch: Pytest fixture for monkeypatching.

    Returns:
        MagicMock: The mocked orchestrator instance.
    """
    mock_orch = MagicMock()
    mock_orch.run_transcription.return_value = {
        "text": "test",
        "segments": [],
        "chunks": [],
    }
    monkeypatch.setattr(
        "insanely_fast_whisper_rocm.api.routes.create_orchestrator",
        lambda: mock_orch,
    )
    return mock_orch


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
        "insanely_fast_whisper_rocm.api.app.download_model_if_needed",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "insanely_fast_whisper_rocm.core.asr_backend.HuggingFaceBackend._initialize_pipeline",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "insanely_fast_whisper_rocm.core.asr_backend.HuggingFaceBackend.process_audio",
        lambda *args, **kwargs: {"text": "test", "segments": [], "chunks": []},
    )
    monkeypatch.setattr(
        "insanely_fast_whisper_rocm.api.routes.stabilize_timestamps",
        lambda result, **kwargs: result,
    )
    monkeypatch.setattr(
        "insanely_fast_whisper_rocm.core.asr_backend.HuggingFaceBackend._validate_device",
        lambda self: None,
    )
    app.dependency_overrides[get_file_handler] = lambda: FileHandler(
        upload_dir=str(tmp_path)
    )
    test_client = TestClient(app)
    try:
        yield test_client
    finally:
        app.dependency_overrides.pop(get_file_handler, None)


# Dummy WAV header for test uploads
DUMMY_WAV_HEADER = (
    b"RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00"
    b"\x01\x00\x01\x00\x80>\x00\x00\x00\xfa\x00\x00"
    b"\x02\x00\x10\x00data\x00\x00\x00\x00"
)
