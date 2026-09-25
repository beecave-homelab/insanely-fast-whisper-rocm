"""Tests for ASR backend model device transfer and detection.

Covers model.to(device) invocation, device-detection logic after transfer,
and OOM handling during device movement.
"""

from __future__ import annotations

import types
from unittest.mock import MagicMock, patch

import pytest
import torch

from insanely_fast_whisper_rocm.core.asr_backend import (
    HuggingFaceBackend,
    HuggingFaceBackendConfig,
)
from insanely_fast_whisper_rocm.core.errors import (
    ModelLoadingOOMError,
    TranscriptionError,
)


def _make_backend(device: str) -> HuggingFaceBackend:
    """Create a backend with the given device, bypassing CUDA checks.

    Args:
        device: Target device string (e.g. "cpu", "cuda:0").

    Returns:
        HuggingFaceBackend: Configured backend instance.
    """
    config = HuggingFaceBackendConfig(
        model_name="openai/whisper-tiny",
        device=device,
        dtype="float32",
        batch_size=4,
        chunk_length=30,
        progress_group_size=4,
    )
    with patch("torch.cuda.is_available", return_value=True):
        return HuggingFaceBackend(config)


def _stub_model_load(
    backend: HuggingFaceBackend,
    mock_model: MagicMock,
) -> None:
    """Patch all model-load helpers and invoke _initialize_pipeline.

    Ensures ``model.to()`` returns the same mock so device attributes
    are preserved after the transfer call.
    """
    mock_model.to.return_value = mock_model
    with patch(
        "insanely_fast_whisper_rocm.core.asr_backend.AutoModelForSpeechSeq2Seq.from_pretrained",
        return_value=mock_model,
    ):
        with patch(
            "insanely_fast_whisper_rocm.core.asr_backend.AutoTokenizer.from_pretrained",
            return_value=MagicMock(),
        ):
            with patch(
                "insanely_fast_whisper_rocm.core.asr_backend.AutoFeatureExtractor.from_pretrained",
                return_value=MagicMock(),
            ):
                with patch(
                    "insanely_fast_whisper_rocm.core.asr_backend.pipeline",
                    return_value=MagicMock(model=mock_model),
                ):
                    backend._initialize_pipeline()


def test_model_transferred_to_non_cpu_device() -> None:
    """Assert model.to() is called with the effective device for non-CPU."""
    backend = _make_backend("cuda:0")
    mock_model = MagicMock()
    mock_model.device = torch.device("cuda:0")
    mock_model.generation_config = types.SimpleNamespace(
        no_timestamps_token_id=50363,
    )
    mock_model.config = types.SimpleNamespace(lang_to_id=None, task_to_id=None)

    _stub_model_load(backend, mock_model)

    mock_model.to.assert_called_once_with("cuda:0")
    assert backend.asr_pipe is not None


def test_model_not_transferred_for_cpu_device() -> None:
    """Assert model.to() is skipped when the effective device is CPU."""
    backend = _make_backend("cpu")
    mock_model = MagicMock()
    mock_model.device = torch.device("cpu")
    mock_model.generation_config = types.SimpleNamespace(
        no_timestamps_token_id=50363,
    )
    mock_model.config = types.SimpleNamespace(lang_to_id=None, task_to_id=None)

    _stub_model_load(backend, mock_model)

    mock_model.to.assert_not_called()
    assert backend.asr_pipe is not None


def test_device_detected_from_model_device_attribute() -> None:
    """Assert device detection reads model.device when present."""
    backend = _make_backend("cuda:0")
    mock_model = MagicMock()
    mock_model.device = torch.device("cuda:0")
    mock_model.generation_config = types.SimpleNamespace(
        no_timestamps_token_id=50363,
    )
    mock_model.config = types.SimpleNamespace(lang_to_id=None, task_to_id=None)

    _stub_model_load(backend, mock_model)

    assert backend.model_device == torch.device("cuda:0")


def test_device_detected_from_first_param_fallback() -> None:
    """Assert device detection falls back to first parameter device."""
    backend = _make_backend("cuda:0")
    mock_model = MagicMock()
    # No .device attribute on model itself
    del mock_model.device
    mock_model.generation_config = types.SimpleNamespace(
        no_timestamps_token_id=50363,
    )
    mock_model.config = types.SimpleNamespace(lang_to_id=None, task_to_id=None)

    # Simulate parameters() returning an iterable with a param that has .device
    mock_param = MagicMock()
    mock_param.device = torch.device("cuda:0")
    mock_model.parameters.return_value = [mock_param]

    _stub_model_load(backend, mock_model)

    assert backend.model_device == torch.device("cuda:0")


def test_warning_logged_when_model_on_cpu_despite_non_cpu_request() -> None:
    """Assert a warning is emitted when the model lands on CPU unexpectedly."""
    backend = _make_backend("cuda:0")
    mock_model = MagicMock()
    # Must be a real torch.device so str(device) == "cpu" evaluates True
    mock_model.device = torch.device("cpu")
    mock_model.generation_config = types.SimpleNamespace(
        no_timestamps_token_id=50363,
    )
    mock_model.config = types.SimpleNamespace(lang_to_id=None, task_to_id=None)

    with patch(
        "insanely_fast_whisper_rocm.core.asr_backend.logger.warning"
    ) as mock_warn:
        _stub_model_load(backend, mock_model)

    mock_warn.assert_any_call(
        "Requested device %s but model appears on CPU",
        "cuda:0",
    )


def test_oom_during_to_raises_model_loading_oom_error() -> None:
    """Assert ModelLoadingOOMError is raised when model.to() OOMs."""
    backend = _make_backend("cuda:0")
    mock_model = MagicMock()
    mock_model.to.side_effect = RuntimeError(
        "CUDA out of memory. Tried to allocate 2.00 GiB (GPU 0; 8.00 GiB total)"
    )
    mock_model.generation_config = types.SimpleNamespace(
        no_timestamps_token_id=50363,
    )
    mock_model.config = types.SimpleNamespace(lang_to_id=None, task_to_id=None)

    with pytest.raises(ModelLoadingOOMError) as exc_info:
        _stub_model_load(backend, mock_model)

    assert exc_info.value.device == "cuda:0"
    assert "OOM moving model to cuda:0" in str(exc_info.value)


def test_non_oom_runtime_error_during_to_propagates() -> None:
    """Assert non-OOM RuntimeError from model.to() is wrapped in TranscriptionError."""
    backend = _make_backend("cuda:0")
    mock_model = MagicMock()
    mock_model.to.side_effect = RuntimeError("some other CUDA error")
    mock_model.generation_config = types.SimpleNamespace(
        no_timestamps_token_id=50363,
    )
    mock_model.config = types.SimpleNamespace(lang_to_id=None, task_to_id=None)

    with pytest.raises(TranscriptionError, match="Failed to load ASR model"):
        _stub_model_load(backend, mock_model)
