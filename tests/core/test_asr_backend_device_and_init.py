"""Tests for ASR backend device validation and initialization.

Covers device validation errors (CUDA/MPS unavailable) and model
initialization logic including ROCm SDPA fallback.
"""

from __future__ import annotations

import pathlib
import types
from unittest.mock import MagicMock, patch

import pytest
import torch

from insanely_fast_whisper_rocm.core.asr_backend import (
    HuggingFaceBackend,
    HuggingFaceBackendConfig,
)
from insanely_fast_whisper_rocm.core.errors import (
    DeviceNotFoundError,
    TranscriptionError,
)


def test_cuda_device_not_available__raises_device_not_found_error() -> None:
    """Raise DeviceNotFoundError when CUDA is requested but not available."""
    config = HuggingFaceBackendConfig(
        model_name="openai/whisper-tiny",
        device="cuda:0",
        dtype="float16",
        batch_size=4,
        chunk_length=30,
        progress_group_size=4,
    )

    with patch(
        "insanely_fast_whisper_rocm.core.asr_backend.torch.cuda.is_available",
        return_value=False,
    ):
        with pytest.raises(
            DeviceNotFoundError,
            match="CUDA device cuda:0 requested but CUDA is not available",
        ):
            HuggingFaceBackend(config)


def test_mps_device_not_available__raises_device_not_found_error() -> None:
    """Raise DeviceNotFoundError when MPS is requested but not available."""
    config = HuggingFaceBackendConfig(
        model_name="openai/whisper-tiny",
        device="mps",
        dtype="float16",
        batch_size=4,
        chunk_length=30,
        progress_group_size=4,
    )

    with patch(
        "insanely_fast_whisper_rocm.core.asr_backend.torch.backends.mps.is_available",
        return_value=False,
    ):
        with pytest.raises(
            DeviceNotFoundError,
            match="MPS device requested but MPS.*is not available",
        ):
            HuggingFaceBackend(config)


def test_initialize_pipeline_loads_model_with_float16(
    tmp_path: pathlib.Path,
) -> None:
    """Verify pipeline initialization loads model with float16 dtype."""
    config = HuggingFaceBackendConfig(
        model_name="openai/whisper-tiny",
        device="cpu",
        dtype="float16",
        batch_size=4,
        chunk_length=30,
        progress_group_size=4,
    )
    backend = HuggingFaceBackend(config)

    mock_model = MagicMock()
    mock_model.generation_config = types.SimpleNamespace(
        no_timestamps_token_id=50363,
        task_to_id={"transcribe": 50359},
        lang_to_id={"en": 50259},
    )
    mock_model.config = types.SimpleNamespace(
        lang_to_id={"en": 50259}, task_to_id={"transcribe": 50359}
    )

    mock_tokenizer = MagicMock()
    mock_feature_extractor = MagicMock()
    mock_pipeline = MagicMock()
    mock_pipeline.model = mock_model

    with patch(
        "insanely_fast_whisper_rocm.core.asr_backend.AutoModelForSpeechSeq2Seq.from_pretrained",
        return_value=mock_model,
    ) as mock_model_load:
        with patch(
            "insanely_fast_whisper_rocm.core.asr_backend.AutoTokenizer.from_pretrained",
            return_value=mock_tokenizer,
        ):
            with patch(
                "insanely_fast_whisper_rocm.core.asr_backend.AutoFeatureExtractor.from_pretrained",
                return_value=mock_feature_extractor,
            ):
                with patch(
                    "insanely_fast_whisper_rocm.core.asr_backend.pipeline",
                    return_value=mock_pipeline,
                ):
                    backend._initialize_pipeline()

    # Verify float16 was passed
    call_kwargs = mock_model_load.call_args[1]
    assert call_kwargs["torch_dtype"] == torch.float16
    assert call_kwargs["use_safetensors"] is True
    assert backend.asr_pipe is not None


def test_initialize_pipeline_loads_model_with_float32(
    tmp_path: pathlib.Path,
) -> None:
    """Verify pipeline initialization loads model with float32 dtype."""
    config = HuggingFaceBackendConfig(
        model_name="openai/whisper-tiny",
        device="cpu",
        dtype="float32",
        batch_size=4,
        chunk_length=30,
        progress_group_size=4,
    )
    backend = HuggingFaceBackend(config)

    mock_model = MagicMock()
    mock_model.generation_config = types.SimpleNamespace(
        no_timestamps_token_id=50363,
    )
    mock_model.config = types.SimpleNamespace(lang_to_id=None, task_to_id=None)

    with patch(
        "insanely_fast_whisper_rocm.core.asr_backend.AutoModelForSpeechSeq2Seq.from_pretrained",
        return_value=mock_model,
    ) as mock_model_load:
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

    call_kwargs = mock_model_load.call_args[1]
    assert call_kwargs["torch_dtype"] == torch.float32


def test_initialize_pipeline_uses_sdpa_on_cuda(tmp_path: pathlib.Path) -> None:
    """Verify SDPA attention is used on CUDA devices."""
    config = HuggingFaceBackendConfig(
        model_name="openai/whisper-tiny",
        device="cuda:0",
        dtype="float16",
        batch_size=4,
        chunk_length=30,
        progress_group_size=4,
    )

    with patch("torch.cuda.is_available", return_value=True):
        backend = HuggingFaceBackend(config)

    mock_model = MagicMock()
    mock_model.generation_config = types.SimpleNamespace(no_timestamps_token_id=50363)
    mock_model.config = types.SimpleNamespace(lang_to_id=None, task_to_id=None)

    # Simulate non-ROCm (regular CUDA)
    with patch("torch.version.hip", None, create=True):
        with patch(
            "insanely_fast_whisper_rocm.core.asr_backend.AutoModelForSpeechSeq2Seq.from_pretrained",
            return_value=mock_model,
        ) as mock_model_load:
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

    call_kwargs = mock_model_load.call_args[1]
    assert call_kwargs["attn_implementation"] == "sdpa"


def test_initialize_pipeline_rocm_fallback_to_eager(tmp_path: pathlib.Path) -> None:
    """Verify ROCm falls back to eager attention if SDPA fails."""
    config = HuggingFaceBackendConfig(
        model_name="openai/whisper-tiny",
        device="cuda:0",
        dtype="float16",
        batch_size=4,
        chunk_length=30,
        progress_group_size=4,
    )

    with patch("torch.cuda.is_available", return_value=True):
        backend = HuggingFaceBackend(config)

    mock_model = MagicMock()
    mock_model.generation_config = types.SimpleNamespace(no_timestamps_token_id=50363)
    mock_model.config = types.SimpleNamespace(lang_to_id=None, task_to_id=None)

    call_count = 0

    def from_pretrained_side_effect(*args: object, **kwargs: object) -> object:
        """Simulate SDPA failure on ROCm, then succeed with eager.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            Mock model on second call.

        Raises:
            RuntimeError: On first call to simulate SDPA failure.
        """
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # First call with SDPA fails
            raise RuntimeError("SDPA not supported")
        # Second call with eager succeeds
        return mock_model

    # Simulate ROCm
    with patch("torch.version.hip", "5.7", create=True):
        with patch(
            "insanely_fast_whisper_rocm.core.asr_backend.AutoModelForSpeechSeq2Seq.from_pretrained",
            side_effect=from_pretrained_side_effect,
        ) as mock_model_load:
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

    # Should have been called twice: once with SDPA, once with eager
    assert mock_model_load.call_count == 2
    first_call_kwargs = mock_model_load.call_args_list[0][1]
    second_call_kwargs = mock_model_load.call_args_list[1][1]
    assert first_call_kwargs["attn_implementation"] == "sdpa"
    assert second_call_kwargs["attn_implementation"] == "eager"


def test_initialize_pipeline_raises_transcription_error_on_model_load_failure(
    tmp_path: pathlib.Path,
) -> None:
    """Raise TranscriptionError when model loading fails unrecoverably."""
    config = HuggingFaceBackendConfig(
        model_name="invalid/nonexistent-model",
        device="cpu",
        dtype="float32",
        batch_size=4,
        chunk_length=30,
        progress_group_size=4,
    )
    backend = HuggingFaceBackend(config)

    with patch(
        "insanely_fast_whisper_rocm.core.asr_backend.AutoModelForSpeechSeq2Seq.from_pretrained",
        side_effect=OSError("Model not found"),
    ):
        with pytest.raises(TranscriptionError, match="Failed to load ASR model"):
            backend._initialize_pipeline()


def test_initialize_pipeline_backfills_generation_config_for_whisper_large_v3(
    tmp_path: pathlib.Path,
) -> None:
    """Backfill generation config for whisper-large-v3 fine-tuned models."""
    config = HuggingFaceBackendConfig(
        model_name="user/custom-whisper-large-v3",
        device="cpu",
        dtype="float32",
        batch_size=4,
        chunk_length=30,
        progress_group_size=4,
    )
    backend = HuggingFaceBackend(config)

    mock_model = MagicMock()
    # Missing generation_config (simulates fine-tuned model without it)
    mock_model.generation_config = None
    mock_model.config = types.SimpleNamespace(lang_to_id=None, task_to_id=None)

    mock_gen_config = types.SimpleNamespace(no_timestamps_token_id=50363)

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
                    "insanely_fast_whisper_rocm.core.asr_backend.GenerationConfig.from_pretrained",
                    return_value=mock_gen_config,
                ) as mock_gen_load:
                    with patch(
                        "insanely_fast_whisper_rocm.core.asr_backend.pipeline",
                        return_value=MagicMock(model=mock_model),
                    ):
                        backend._initialize_pipeline()

    # Verify backfill was attempted for large-v3
    mock_gen_load.assert_called_once_with("openai/whisper-large-v3")
    assert mock_model.generation_config == mock_gen_config


def test_initialize_pipeline_backfills_generation_config_for_whisper_medium_en(
    tmp_path: pathlib.Path,
) -> None:
    """Backfill generation config for whisper-medium.en fine-tuned models."""
    config = HuggingFaceBackendConfig(
        model_name="user/custom-whisper-medium.en",
        device="cpu",
        dtype="float32",
        batch_size=4,
        chunk_length=30,
        progress_group_size=4,
    )
    backend = HuggingFaceBackend(config)

    mock_model = MagicMock()
    mock_model.generation_config = types.SimpleNamespace(no_timestamps_token_id=None)
    mock_model.config = types.SimpleNamespace(lang_to_id=None, task_to_id=None)

    mock_gen_config = types.SimpleNamespace(no_timestamps_token_id=50363)

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
                    "insanely_fast_whisper_rocm.core.asr_backend.GenerationConfig.from_pretrained",
                    return_value=mock_gen_config,
                ) as mock_gen_load:
                    with patch(
                        "insanely_fast_whisper_rocm.core.asr_backend.pipeline",
                        return_value=MagicMock(model=mock_model),
                    ):
                        backend._initialize_pipeline()

    # Verify backfill for medium.en
    mock_gen_load.assert_called_once_with("openai/whisper-medium.en")


def test_initialize_pipeline_skips_backfill_when_generation_config_exists(
    tmp_path: pathlib.Path,
) -> None:
    """Skip backfill when generation_config already has timestamp tokens."""
    config = HuggingFaceBackendConfig(
        model_name="user/custom-whisper-large-v3",
        device="cpu",
        dtype="float32",
        batch_size=4,
        chunk_length=30,
        progress_group_size=4,
    )
    backend = HuggingFaceBackend(config)

    mock_model = MagicMock()
    # Already has valid generation_config
    mock_model.generation_config = types.SimpleNamespace(no_timestamps_token_id=50363)
    mock_model.config = types.SimpleNamespace(lang_to_id=None, task_to_id=None)

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
                    "insanely_fast_whisper_rocm.core.asr_backend.GenerationConfig.from_pretrained",
                    return_value=MagicMock(),
                ) as mock_gen_load:
                    with patch(
                        "insanely_fast_whisper_rocm.core.asr_backend.pipeline",
                        return_value=MagicMock(model=mock_model),
                    ):
                        backend._initialize_pipeline()

    # Should not attempt backfill
    mock_gen_load.assert_not_called()
