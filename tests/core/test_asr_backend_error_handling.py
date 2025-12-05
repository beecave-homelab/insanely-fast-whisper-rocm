"""Tests for ASR backend error handling and fallback logic.

Covers RuntimeError handling with tensor size mismatch fallback and
other exception types during transcription.
"""

from __future__ import annotations

import pathlib
import types
from typing import Any

import pytest

from insanely_fast_whisper_rocm.core.asr_backend import (
    HuggingFaceBackend,
    HuggingFaceBackendConfig,
)
from insanely_fast_whisper_rocm.core.errors import TranscriptionError


def test_runtime_error_tensor_mismatch_falls_back_to_chunk_timestamps(
    tmp_path: pathlib.Path,
) -> None:
    """Fallback to chunk-level timestamps on tensor size mismatch error."""
    config = HuggingFaceBackendConfig(
        model_name="openai/whisper-tiny",
        device="cpu",
        dtype="float32",
        batch_size=4,
        chunk_length=30,
        progress_group_size=4,
    )
    backend = HuggingFaceBackend(config)

    call_count = 0
    captured_kwargs_list: list[dict[str, Any]] = []

    class DummyModel:
        generation_config = types.SimpleNamespace(
            no_timestamps_token_id=50363,
            task_to_id={"transcribe": 50359},
            lang_to_id={"en": 50259},
        )
        config = types.SimpleNamespace(
            lang_to_id={"en": 50259}, task_to_id={"transcribe": 50359}
        )

    class DummyPipe:
        def __init__(self) -> None:
            self.model = DummyModel()

        def __call__(self, path: str, **kwargs: dict[str, Any]) -> dict[str, Any]:
            """Simulate tensor mismatch error on first call, succeed on second.

            Args:
                path: Audio file path.
                **kwargs: Pipeline keyword arguments.

            Returns:
                Mock transcription result.

            Raises:
                RuntimeError: On first call with word-level timestamps.
            """
            nonlocal call_count
            nonlocal captured_kwargs_list
            call_count += 1
            captured_kwargs_list.append(kwargs.copy())

            if call_count == 1:
                # First call: raise tensor size mismatch error
                raise RuntimeError(
                    "The expanded size of the tensor (123) must match the existing size (456)"
                )
            # Second call: succeed with chunk-level timestamps
            return {
                "text": "Fallback success",
                "chunks": [{"text": "Fallback success", "timestamp": [0.0, 2.0]}],
            }

    backend.asr_pipe = DummyPipe()

    audio_file = tmp_path / "audio.wav"
    audio_file.write_bytes(b"0")

    result = backend.process_audio(
        str(audio_file),
        language=None,
        task="transcribe",
        return_timestamps_value="word",
    )

    # Should have been called twice: once with word, once with chunk fallback
    assert call_count == 2
    assert captured_kwargs_list[0]["return_timestamps"] == "word"
    assert captured_kwargs_list[1]["return_timestamps"] is True  # chunk-level
    assert result["text"] == "Fallback success"


def test_runtime_error_tensor_mismatch_fallback_also_fails__raises_transcription_error(
    tmp_path: pathlib.Path,
) -> None:
    """Raise TranscriptionError when fallback also fails after tensor mismatch."""
    config = HuggingFaceBackendConfig(
        model_name="openai/whisper-tiny",
        device="cpu",
        dtype="float32",
        batch_size=4,
        chunk_length=30,
        progress_group_size=4,
    )
    backend = HuggingFaceBackend(config)

    call_count = 0

    class DummyModel:
        generation_config = types.SimpleNamespace(
            no_timestamps_token_id=50363,
            task_to_id={"transcribe": 50359},
            lang_to_id={"en": 50259},
        )
        config = types.SimpleNamespace(
            lang_to_id={"en": 50259}, task_to_id={"transcribe": 50359}
        )

    class DummyPipe:
        def __init__(self) -> None:
            self.model = DummyModel()

        def __call__(self, path: str, **kwargs: dict[str, Any]) -> dict[str, Any]:
            """Simulate tensor mismatch error on first call, different error on second.

            Args:
                path: Audio file path.
                **kwargs: Pipeline keyword arguments.

            Raises:
                RuntimeError: On both calls.
            """
            nonlocal call_count
            call_count += 1

            if call_count == 1:
                raise RuntimeError(
                    "The expanded size of the tensor (123) must match the existing size (456)"
                )
            # Second call also fails
            raise RuntimeError("Fallback also failed")

    backend.asr_pipe = DummyPipe()

    audio_file = tmp_path / "audio.wav"
    audio_file.write_bytes(b"0")

    with pytest.raises(
        TranscriptionError,
        match="Failed to transcribe audio even with fallback",
    ):
        backend.process_audio(
            str(audio_file),
            language=None,
            task="transcribe",
            return_timestamps_value="word",
        )

    assert call_count == 2


def test_runtime_error_non_tensor_mismatch__raises_transcription_error(
    tmp_path: pathlib.Path,
) -> None:
    """Raise TranscriptionError on RuntimeError that is not tensor mismatch."""
    config = HuggingFaceBackendConfig(
        model_name="openai/whisper-tiny",
        device="cpu",
        dtype="float32",
        batch_size=4,
        chunk_length=30,
        progress_group_size=4,
    )
    backend = HuggingFaceBackend(config)

    class DummyModel:
        generation_config = types.SimpleNamespace(
            no_timestamps_token_id=50363,
            task_to_id={"transcribe": 50359},
            lang_to_id={"en": 50259},
        )
        config = types.SimpleNamespace(
            lang_to_id={"en": 50259}, task_to_id={"transcribe": 50359}
        )

    class DummyPipe:
        def __init__(self) -> None:
            self.model = DummyModel()

        def __call__(self, path: str, **kwargs: dict[str, Any]) -> dict[str, Any]:
            """Raise different RuntimeError.

            Args:
                path: Audio file path.
                **kwargs: Pipeline keyword arguments.

            Raises:
                RuntimeError: Always.
            """
            raise RuntimeError("Some other runtime error")

    backend.asr_pipe = DummyPipe()

    audio_file = tmp_path / "audio.wav"
    audio_file.write_bytes(b"0")

    with pytest.raises(TranscriptionError, match="Failed to transcribe audio"):
        backend.process_audio(
            str(audio_file),
            language=None,
            task="transcribe",
            return_timestamps_value=False,
        )


def test_os_error_during_transcription__raises_transcription_error(
    tmp_path: pathlib.Path,
) -> None:
    """Raise TranscriptionError on OSError during transcription."""
    config = HuggingFaceBackendConfig(
        model_name="openai/whisper-tiny",
        device="cpu",
        dtype="float32",
        batch_size=4,
        chunk_length=30,
        progress_group_size=4,
    )
    backend = HuggingFaceBackend(config)

    class DummyModel:
        generation_config = types.SimpleNamespace(
            no_timestamps_token_id=50363,
        )
        config = types.SimpleNamespace(lang_to_id=None, task_to_id=None)

    class DummyPipe:
        def __init__(self) -> None:
            self.model = DummyModel()

        def __call__(self, path: str, **kwargs: dict[str, Any]) -> dict[str, Any]:
            """Raise OSError.

            Args:
                path: Audio file path.
                **kwargs: Pipeline keyword arguments.

            Raises:
                OSError: Always.
            """
            raise OSError("File not found or corrupted")

    backend.asr_pipe = DummyPipe()

    audio_file = tmp_path / "audio.wav"
    audio_file.write_bytes(b"0")

    with pytest.raises(TranscriptionError, match="Failed to transcribe audio"):
        backend.process_audio(
            str(audio_file),
            language=None,
            task="transcribe",
            return_timestamps_value=False,
        )


def test_value_error_during_transcription__raises_transcription_error(
    tmp_path: pathlib.Path,
) -> None:
    """Raise TranscriptionError on ValueError during transcription."""
    config = HuggingFaceBackendConfig(
        model_name="openai/whisper-tiny",
        device="cpu",
        dtype="float32",
        batch_size=4,
        chunk_length=30,
        progress_group_size=4,
    )
    backend = HuggingFaceBackend(config)

    class DummyModel:
        generation_config = types.SimpleNamespace(
            no_timestamps_token_id=50363,
        )
        config = types.SimpleNamespace(lang_to_id=None, task_to_id=None)

    class DummyPipe:
        def __init__(self) -> None:
            self.model = DummyModel()

        def __call__(self, path: str, **kwargs: dict[str, Any]) -> dict[str, Any]:
            """Raise ValueError.

            Args:
                path: Audio file path.
                **kwargs: Pipeline keyword arguments.

            Raises:
                ValueError: Always.
            """
            raise ValueError("Invalid configuration")

    backend.asr_pipe = DummyPipe()

    audio_file = tmp_path / "audio.wav"
    audio_file.write_bytes(b"0")

    with pytest.raises(TranscriptionError, match="Failed to transcribe audio"):
        backend.process_audio(
            str(audio_file),
            language=None,
            task="transcribe",
            return_timestamps_value=False,
        )


def test_memory_error_during_transcription__raises_transcription_error(
    tmp_path: pathlib.Path,
) -> None:
    """Raise TranscriptionError on MemoryError during transcription."""
    config = HuggingFaceBackendConfig(
        model_name="openai/whisper-tiny",
        device="cpu",
        dtype="float32",
        batch_size=4,
        chunk_length=30,
        progress_group_size=4,
    )
    backend = HuggingFaceBackend(config)

    class DummyModel:
        generation_config = types.SimpleNamespace(
            no_timestamps_token_id=50363,
        )
        config = types.SimpleNamespace(lang_to_id=None, task_to_id=None)

    class DummyPipe:
        def __init__(self) -> None:
            self.model = DummyModel()

        def __call__(self, path: str, **kwargs: dict[str, Any]) -> dict[str, Any]:
            """Raise MemoryError.

            Args:
                path: Audio file path.
                **kwargs: Pipeline keyword arguments.

            Raises:
                MemoryError: Always.
            """
            raise MemoryError("Out of memory")

    backend.asr_pipe = DummyPipe()

    audio_file = tmp_path / "audio.wav"
    audio_file.write_bytes(b"0")

    with pytest.raises(TranscriptionError, match="Failed to transcribe audio"):
        backend.process_audio(
            str(audio_file),
            language=None,
            task="transcribe",
            return_timestamps_value=False,
        )


def test_type_error_during_transcription__raises_transcription_error(
    tmp_path: pathlib.Path,
) -> None:
    """Raise TranscriptionError on TypeError during transcription."""
    config = HuggingFaceBackendConfig(
        model_name="openai/whisper-tiny",
        device="cpu",
        dtype="float32",
        batch_size=4,
        chunk_length=30,
        progress_group_size=4,
    )
    backend = HuggingFaceBackend(config)

    class DummyModel:
        generation_config = types.SimpleNamespace(
            no_timestamps_token_id=50363,
        )
        config = types.SimpleNamespace(lang_to_id=None, task_to_id=None)

    class DummyPipe:
        def __init__(self) -> None:
            self.model = DummyModel()

        def __call__(self, path: str, **kwargs: dict[str, Any]) -> dict[str, Any]:
            """Raise TypeError.

            Args:
                path: Audio file path.
                **kwargs: Pipeline keyword arguments.

            Raises:
                TypeError: Always.
            """
            raise TypeError("Unexpected type")

    backend.asr_pipe = DummyPipe()

    audio_file = tmp_path / "audio.wav"
    audio_file.write_bytes(b"0")

    with pytest.raises(TranscriptionError, match="Failed to transcribe audio"):
        backend.process_audio(
            str(audio_file),
            language=None,
            task="transcribe",
            return_timestamps_value=False,
        )


def test_index_error_during_transcription__raises_transcription_error(
    tmp_path: pathlib.Path,
) -> None:
    """Raise TranscriptionError on IndexError during transcription."""
    config = HuggingFaceBackendConfig(
        model_name="openai/whisper-tiny",
        device="cpu",
        dtype="float32",
        batch_size=4,
        chunk_length=30,
        progress_group_size=4,
    )
    backend = HuggingFaceBackend(config)

    class DummyModel:
        generation_config = types.SimpleNamespace(
            no_timestamps_token_id=50363,
        )
        config = types.SimpleNamespace(lang_to_id=None, task_to_id=None)

    class DummyPipe:
        def __init__(self) -> None:
            self.model = DummyModel()

        def __call__(self, path: str, **kwargs: dict[str, Any]) -> dict[str, Any]:
            """Raise IndexError.

            Args:
                path: Audio file path.
                **kwargs: Pipeline keyword arguments.

            Raises:
                IndexError: Always.
            """
            raise IndexError("List index out of range")

    backend.asr_pipe = DummyPipe()

    audio_file = tmp_path / "audio.wav"
    audio_file.write_bytes(b"0")

    with pytest.raises(TranscriptionError, match="Failed to transcribe audio"):
        backend.process_audio(
            str(audio_file),
            language=None,
            task="transcribe",
            return_timestamps_value=False,
        )
