"""Tests for the transcription orchestrator retry and fallback logic."""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from typing import Any
from unittest.mock import Mock

import pytest

from insanely_fast_whisper_rocm.core.asr_backend import HuggingFaceBackendConfig
from insanely_fast_whisper_rocm.core.errors import (
    InferenceOOMError,
    ModelLoadingOOMError,
    TranscriptionError,
)
from insanely_fast_whisper_rocm.core.orchestrator import TranscriptionOrchestrator


def _config(
    *, device: str, batch_size: int = 4, chunk_length: int = 30
) -> HuggingFaceBackendConfig:
    """
    Create a HuggingFaceBackendConfig for the "openai/whisper-tiny" model with sensible defaults.
    
    Parameters:
        device (str): Target device identifier (e.g., "cuda" or "cpu").
        batch_size (int): Number of items per batch; defaults to 4.
        chunk_length (int): Audio chunk length in seconds; defaults to 30.
    
    Returns:
        HuggingFaceBackendConfig: Configuration for "openai/whisper-tiny" using dtype "float16" and progress_group_size 4, with the provided device, batch_size, and chunk_length.
    """
    return HuggingFaceBackendConfig(
        model_name="openai/whisper-tiny",
        device=device,
        dtype="float16",
        batch_size=batch_size,
        chunk_length=chunk_length,
        progress_group_size=4,
    )


@contextmanager
def _borrow_pipeline_with_process(process: Mock) -> Generator[Any, None, None]:
    """
    Context manager that yields a Mock pipeline with its `process` attribute set to the provided mock.
    
    Parameters:
        process (Mock): The mock object to assign to `pipeline.process`.
    
    Returns:
        Generator[Any, None, None]: A generator (context manager) that yields the configured Mock pipeline.
    """
    pipeline = Mock()
    pipeline.process = process
    yield pipeline


def test_run_transcription_success_records_attempt_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Return successful result and attach orchestrator attempt history."""
    process = Mock(return_value={"text": "ok"})

    monkeypatch.setattr(
        "insanely_fast_whisper_rocm.core.orchestrator.borrow_pipeline",
        lambda *args, **kwargs: _borrow_pipeline_with_process(process),
    )

    orch = TranscriptionOrchestrator()
    warn = Mock()

    result = orch.run_transcription(
        audio_path="/tmp/audio.wav",
        backend_config=_config(device="cuda:0", batch_size=4),
        warning_callback=warn,
    )

    assert result["text"] == "ok"
    assert "orchestrator_attempts" in result
    assert result["orchestrator_attempts"][0]["status"] == "succeeded"
    assert warn.call_count == 1


def test_run_transcription_inference_oom_reduces_batch_then_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """First inference OOM reduces GPU batch size and retries."""
    process = Mock(side_effect=[InferenceOOMError("oom"), {"text": "ok"}])

    monkeypatch.setattr(
        "insanely_fast_whisper_rocm.core.orchestrator.borrow_pipeline",
        lambda *args, **kwargs: _borrow_pipeline_with_process(process),
    )
    invalidate = Mock()
    monkeypatch.setattr(
        "insanely_fast_whisper_rocm.core.orchestrator.invalidate_gpu_cache",
        invalidate,
    )

    orch = TranscriptionOrchestrator()
    warn = Mock()

    result = orch.run_transcription(
        audio_path="/tmp/audio.wav",
        backend_config=_config(device="cuda:0", batch_size=4),
        warning_callback=warn,
    )

    assert result["text"] == "ok"
    assert len(result["orchestrator_attempts"]) == 2

    attempt_1 = result["orchestrator_attempts"][0]
    assert attempt_1["status"] == "failed"
    assert attempt_1["error_type"] == "InferenceOOMError"
    assert "recovery_action" in attempt_1

    attempt_2 = result["orchestrator_attempts"][1]
    assert attempt_2["status"] == "succeeded"
    assert attempt_2["config"]["batch_size"] == 2

    invalidate.assert_not_called()
    assert warn.call_count == 3


def test_run_transcription_inference_oom_then_cpu_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Second inference OOM triggers CPU fallback and invalidates GPU cache."""
    process = Mock(
        side_effect=[
            InferenceOOMError("oom-1"),
            InferenceOOMError("oom-2"),
            {"text": "ok"},
        ]
    )

    monkeypatch.setattr(
        "insanely_fast_whisper_rocm.core.orchestrator.borrow_pipeline",
        lambda *args, **kwargs: _borrow_pipeline_with_process(process),
    )
    invalidate = Mock()
    monkeypatch.setattr(
        "insanely_fast_whisper_rocm.core.orchestrator.invalidate_gpu_cache",
        invalidate,
    )

    orch = TranscriptionOrchestrator()

    result = orch.run_transcription(
        audio_path="/tmp/audio.wav",
        backend_config=_config(device="cuda:0", batch_size=4, chunk_length=30),
    )

    assert result["text"] == "ok"
    assert len(result["orchestrator_attempts"]) == 3
    assert result["orchestrator_attempts"][2]["config"]["device"] == "cpu"
    assert result["orchestrator_attempts"][2]["config"]["dtype"] == "float32"
    assert result["orchestrator_attempts"][2]["config"]["batch_size"] == 2
    assert result["orchestrator_attempts"][2]["config"]["chunk_length"] == 15

    invalidate.assert_called_once()


def test_run_transcription_model_loading_oom_goes_directly_to_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Model loading OOM skips GPU retry and immediately falls back to CPU."""
    process = Mock(side_effect=[ModelLoadingOOMError("oom"), {"text": "ok"}])

    monkeypatch.setattr(
        "insanely_fast_whisper_rocm.core.orchestrator.borrow_pipeline",
        lambda *args, **kwargs: _borrow_pipeline_with_process(process),
    )
    invalidate = Mock()
    monkeypatch.setattr(
        "insanely_fast_whisper_rocm.core.orchestrator.invalidate_gpu_cache",
        invalidate,
    )

    orch = TranscriptionOrchestrator()

    result = orch.run_transcription(
        audio_path="/tmp/audio.wav",
        backend_config=_config(device="cuda:0", batch_size=4, chunk_length=30),
    )

    assert result["text"] == "ok"
    assert len(result["orchestrator_attempts"]) == 2
    assert result["orchestrator_attempts"][1]["config"]["device"] == "cpu"

    invalidate.assert_called_once()


def test_run_transcription_oom_on_cpu_is_not_recoverable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Raise when OOM occurs on CPU (no further fallbacks)."""
    process = Mock(side_effect=InferenceOOMError("oom"))

    monkeypatch.setattr(
        "insanely_fast_whisper_rocm.core.orchestrator.borrow_pipeline",
        lambda *args, **kwargs: _borrow_pipeline_with_process(process),
    )

    orch = TranscriptionOrchestrator()

    with pytest.raises(InferenceOOMError, match="oom"):
        orch.run_transcription(
            audio_path="/tmp/audio.wav",
            backend_config=_config(device="cpu", batch_size=4),
        )


def test_run_transcription_unexpected_exception_wraps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Wrap unexpected exceptions in TranscriptionError with chained cause."""
    process = Mock(side_effect=RuntimeError("boom"))

    monkeypatch.setattr(
        "insanely_fast_whisper_rocm.core.orchestrator.borrow_pipeline",
        lambda *args, **kwargs: _borrow_pipeline_with_process(process),
    )

    orch = TranscriptionOrchestrator()

    with pytest.raises(TranscriptionError, match="Unexpected error: boom") as exc:
        orch.run_transcription(
            audio_path="/tmp/audio.wav",
            backend_config=_config(device="cuda:0", batch_size=4),
        )

    assert isinstance(exc.value.__cause__, RuntimeError)