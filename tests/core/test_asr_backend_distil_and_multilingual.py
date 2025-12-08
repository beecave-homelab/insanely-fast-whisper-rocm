"""Tests for distil-whisper timestamp detection and multilingual model checks.

Covers timestamp capability detection for distil-whisper variants and
multilingual model detection logic.
"""

from __future__ import annotations

import pathlib
import types
from typing import Any

from insanely_fast_whisper_rocm.core.asr_backend import (
    HuggingFaceBackend,
    HuggingFaceBackendConfig,
)


def test_distil_whisper_v2_supports_timestamps(tmp_path: pathlib.Path) -> None:
    """Verify distil-whisper-large-v2 supports timestamps."""
    config = HuggingFaceBackendConfig(
        model_name="distil-whisper/distil-large-v2",
        device="cpu",
        dtype="float32",
        batch_size=4,
        chunk_length=30,
        progress_group_size=4,
    )
    backend = HuggingFaceBackend(config)

    captured_kwargs: dict[str, Any] = {}

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
            """Capture kwargs.

            Args:
                path: Audio file path.
                **kwargs: Pipeline keyword arguments.

            Returns:
                Mock transcription result.
            """
            nonlocal captured_kwargs
            captured_kwargs = kwargs.copy()
            return {
                "text": "Hello",
                "chunks": [{"text": "Hello", "timestamp": [0.0, 1.0]}],
            }

    backend.asr_pipe = DummyPipe()

    audio_file = tmp_path / "audio.wav"
    audio_file.write_bytes(b"0")

    result = backend.process_audio(
        str(audio_file),
        language=None,
        task="transcribe",
        return_timestamps_value=True,
    )

    # Timestamps should be enabled for v2
    assert captured_kwargs.get("return_timestamps") is True
    assert result["text"] == "Hello"


def test_distil_whisper_v1_disables_timestamps(tmp_path: pathlib.Path) -> None:
    """Verify distil-whisper v1 (without v2/v3 suffix) disables timestamps."""
    config = HuggingFaceBackendConfig(
        model_name="distil-whisper/distil-large",  # No -v2 suffix
        device="cpu",
        dtype="float32",
        batch_size=4,
        chunk_length=30,
        progress_group_size=4,
    )
    backend = HuggingFaceBackend(config)

    captured_kwargs: dict[str, Any] = {}

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
            """Capture kwargs.

            Args:
                path: Audio file path.
                **kwargs: Pipeline keyword arguments.

            Returns:
                Mock transcription result.
            """
            nonlocal captured_kwargs
            captured_kwargs = kwargs.copy()
            return {"text": "Hello"}

    backend.asr_pipe = DummyPipe()

    audio_file = tmp_path / "audio.wav"
    audio_file.write_bytes(b"0")

    result = backend.process_audio(
        str(audio_file),
        language=None,
        task="transcribe",
        return_timestamps_value=True,  # Requested, but should be disabled
    )

    # Timestamps should be disabled for v1 (no -v2/-v3/-v4 suffix)
    assert captured_kwargs.get("return_timestamps") is False
    assert result["text"] == "Hello"


def test_distil_whisper_medium_en_v2_supports_timestamps(
    tmp_path: pathlib.Path,
) -> None:
    """Verify distil-whisper/distil-medium.en-v2 supports timestamps."""
    config = HuggingFaceBackendConfig(
        model_name="distil-whisper/distil-medium.en-v2",
        device="cpu",
        dtype="float32",
        batch_size=4,
        chunk_length=30,
        progress_group_size=4,
    )
    backend = HuggingFaceBackend(config)

    captured_kwargs: dict[str, Any] = {}

    class DummyModel:
        generation_config = types.SimpleNamespace(
            no_timestamps_token_id=50363,
        )
        config = types.SimpleNamespace(lang_to_id=None, task_to_id=None)

    class DummyPipe:
        def __init__(self) -> None:
            self.model = DummyModel()

        def __call__(self, path: str, **kwargs: dict[str, Any]) -> dict[str, Any]:
            """Capture kwargs.

            Args:
                path: Audio file path.
                **kwargs: Pipeline keyword arguments.

            Returns:
                Mock transcription result.
            """
            nonlocal captured_kwargs
            captured_kwargs = kwargs.copy()
            return {"text": "Hello"}

    backend.asr_pipe = DummyPipe()

    audio_file = tmp_path / "audio.wav"
    audio_file.write_bytes(b"0")

    backend.process_audio(
        str(audio_file),
        language=None,
        task="transcribe",
        return_timestamps_value=True,
    )

    # Timestamps should be enabled for v2
    assert captured_kwargs.get("return_timestamps") is True


def test_multilingual_detection_via_lang_to_id_dict(tmp_path: pathlib.Path) -> None:
    """Detect multilingual model via lang_to_id dict with multiple languages."""
    config = HuggingFaceBackendConfig(
        model_name="openai/whisper-small",
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
            task_to_id=None,
            lang_to_id=None,
        )
        config = types.SimpleNamespace(
            # Multiple languages -> multilingual
            lang_to_id={"en": 50259, "es": 50260, "fr": 50261},
            task_to_id=None,
        )

    class DummyPipe:
        def __init__(self) -> None:
            self.model = DummyModel()

        def __call__(self, path: str, **kwargs: dict[str, Any]) -> dict[str, Any]:
            """Mock pipeline call that allows translate task.

            Args:
                path: Audio file path.
                **kwargs: Pipeline keyword arguments.

            Returns:
                Mock transcription result.
            """
            # Should not raise warning for translate on multilingual model
            return {"text": "Hello"}

    backend.asr_pipe = DummyPipe()

    audio_file = tmp_path / "audio.wav"
    audio_file.write_bytes(b"0")

    # Should not raise or log error for translate task
    result = backend.process_audio(
        str(audio_file),
        language="es",
        task="translate",
        return_timestamps_value=False,
    )

    assert result["text"] == "Hello"


def test_english_only_model_warns_on_translate_task(tmp_path: pathlib.Path) -> None:
    """Warn when translate task is requested on English-only model."""
    config = HuggingFaceBackendConfig(
        model_name="openai/whisper-tiny.en",
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
            task_to_id=None,
            lang_to_id=None,
        )
        config = types.SimpleNamespace(
            lang_to_id={"en": 50259},  # Only one language -> not multilingual
            task_to_id=None,
        )

    class DummyPipe:
        def __init__(self) -> None:
            self.model = DummyModel()

        def __call__(self, path: str, **kwargs: dict[str, Any]) -> dict[str, Any]:
            """Mock pipeline call.

            Args:
                path: Audio file path.
                **kwargs: Pipeline keyword arguments.

            Returns:
                Mock transcription result.
            """
            return {"text": "Hello"}

    backend.asr_pipe = DummyPipe()

    audio_file = tmp_path / "audio.wav"
    audio_file.write_bytes(b"0")

    # Should log warning but still proceed
    result = backend.process_audio(
        str(audio_file),
        language=None,
        task="translate",
        return_timestamps_value=False,
    )

    assert result["text"] == "Hello"


def test_language_parameter_passed_when_has_task_mappings(
    tmp_path: pathlib.Path,
) -> None:
    """Pass language parameter to generate_kwargs when model has task mappings."""
    config = HuggingFaceBackendConfig(
        model_name="openai/whisper-small",
        device="cpu",
        dtype="float32",
        batch_size=4,
        chunk_length=30,
        progress_group_size=4,
    )
    backend = HuggingFaceBackend(config)

    captured_kwargs: dict[str, Any] = {}

    class DummyModel:
        generation_config = types.SimpleNamespace(
            no_timestamps_token_id=50363,
            task_to_id={"transcribe": 50359, "translate": 50358},
            lang_to_id={"en": 50259, "es": 50260},
        )
        config = types.SimpleNamespace(
            lang_to_id={"en": 50259, "es": 50260},
            task_to_id={"transcribe": 50359, "translate": 50358},
        )

    class DummyPipe:
        def __init__(self) -> None:
            self.model = DummyModel()

        def __call__(self, path: str, **kwargs: dict[str, Any]) -> dict[str, Any]:
            """Capture kwargs.

            Args:
                path: Audio file path.
                **kwargs: Pipeline keyword arguments.

            Returns:
                Mock transcription result.
            """
            nonlocal captured_kwargs
            captured_kwargs = kwargs.copy()
            return {"text": "Hola"}

    backend.asr_pipe = DummyPipe()

    audio_file = tmp_path / "audio.wav"
    audio_file.write_bytes(b"0")

    result = backend.process_audio(
        str(audio_file),
        language="es",
        task="transcribe",
        return_timestamps_value=False,
    )

    # Language should be passed to generate_kwargs
    gen_kwargs = captured_kwargs.get("generate_kwargs", {})
    assert gen_kwargs.get("language") == "es"
    assert gen_kwargs.get("task") == "transcribe"
    assert result["text"] == "Hola"


def test_translate_task_defaults_to_english_when_no_language_specified(
    tmp_path: pathlib.Path,
) -> None:
    """Default to English language for translate task when none specified."""
    config = HuggingFaceBackendConfig(
        model_name="openai/whisper-small",
        device="cpu",
        dtype="float32",
        batch_size=4,
        chunk_length=30,
        progress_group_size=4,
    )
    backend = HuggingFaceBackend(config)

    captured_kwargs: dict[str, Any] = {}

    class DummyModel:
        generation_config = types.SimpleNamespace(
            no_timestamps_token_id=50363,
            task_to_id={"transcribe": 50359, "translate": 50358},
            lang_to_id={"en": 50259},
        )
        config = types.SimpleNamespace(
            lang_to_id={"en": 50259},
            task_to_id={"transcribe": 50359, "translate": 50358},
        )

    class DummyPipe:
        def __init__(self) -> None:
            self.model = DummyModel()

        def __call__(self, path: str, **kwargs: dict[str, Any]) -> dict[str, Any]:
            """Capture kwargs.

            Args:
                path: Audio file path.
                **kwargs: Pipeline keyword arguments.

            Returns:
                Mock transcription result.
            """
            nonlocal captured_kwargs
            captured_kwargs = kwargs.copy()
            return {"text": "Hello"}

    backend.asr_pipe = DummyPipe()

    audio_file = tmp_path / "audio.wav"
    audio_file.write_bytes(b"0")

    backend.process_audio(
        str(audio_file),
        language=None,  # No language specified
        task="translate",
        return_timestamps_value=False,
    )

    # Should default to English for translate task
    gen_kwargs = captured_kwargs.get("generate_kwargs", {})
    assert gen_kwargs.get("language") == "en"
    assert gen_kwargs.get("task") == "translate"
