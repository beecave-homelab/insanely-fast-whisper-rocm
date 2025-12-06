import pathlib
import types

from insanely_fast_whisper_rocm.core.asr_backend import (
    HuggingFaceBackend,
    HuggingFaceBackendConfig,
)


def test_disables_timestamps_when_generation_config_missing(monkeypatch, tmp_path):
    """Verify that when the model's generation_config.no_timestamps_token_id is None, the backend disables timestamps even if `return_timestamps_value=True`.

    The test injects a dummy ASR pipeline that asserts `return_timestamps` is False, patches audio conversion to accept any path, calls `process_audio` with `return_timestamps_value=True`, and asserts the returned result has `chunks` set to `None`.
    """
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
