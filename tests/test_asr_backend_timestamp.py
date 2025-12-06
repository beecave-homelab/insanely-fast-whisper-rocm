import pathlib
import types

from insanely_fast_whisper_rocm.core.asr_backend import (
    HuggingFaceBackend,
    HuggingFaceBackendConfig,
)


def test_disables_timestamps_when_generation_config_missing(monkeypatch, tmp_path):
    config = HuggingFaceBackendConfig(
        model_name="dummy-model",
        device="cpu",
        dtype="float32",
        batch_size=1,
        chunk_length=30,
    )
    backend = HuggingFaceBackend(config)

    class DummyModel:
        generation_config = types.SimpleNamespace(no_timestamps_token_id=None)
        config = types.SimpleNamespace(lang_to_id=None, task_to_id=None)

    class DummyPipe:
        def __init__(self):
            self.model = DummyModel()

        def __call__(self, path, **kwargs):
            assert kwargs["return_timestamps"] is False
            return {"text": "hello"}

    backend.asr_pipe = DummyPipe()

    monkeypatch.setattr(
        "insanely_fast_whisper_rocm.audio.conversion.ensure_wav",
        lambda p: pathlib.Path(p),
    )

    audio_file = tmp_path / "audio.wav"
    audio_file.write_bytes(b"0")

    result = backend.process_audio(
        str(audio_file), language=None, task="transcribe", return_timestamps_value=True
    )

    assert result["chunks"] is None
