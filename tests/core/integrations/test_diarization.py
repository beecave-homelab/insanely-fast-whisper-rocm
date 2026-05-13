"""Tests for the speaker diarization integration module."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch

from insanely_fast_whisper_rocm.core.errors import DiarizationError
from insanely_fast_whisper_rocm.core.integrations.diarization import (
    _align_speakers_to_segments,
    _preload_audio_as_waveform,
    clear_diarization_cache,
    diarize,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_cache() -> None:
    """Ensure the pipeline cache is empty before each test."""
    clear_diarization_cache()


def _sample_result() -> dict[str, Any]:
    """Return a minimal Whisper result dict with chunks."""
    return {
        "text": "Hello world. Goodbye world.",
        "chunks": [
            {"start": 0.0, "end": 2.0, "text": "Hello world."},
            {"start": 2.5, "end": 5.0, "text": "Goodbye world."},
        ],
    }


# ---------------------------------------------------------------------------
# diarize() — pyannote not installed
# ---------------------------------------------------------------------------


def test_diarize_raises_when_pyannote_not_installed() -> None:
    """diarize() raises DiarizationError when Pipeline is None."""
    result = _sample_result()
    with patch(
        "insanely_fast_whisper_rocm.core.integrations.diarization.Pipeline",
        None,
    ):
        with pytest.raises(DiarizationError, match="pyannote.audio is not installed"):
            diarize(result, audio_path="/fake.wav", hf_token="tok")


# ---------------------------------------------------------------------------
# diarize() — missing HF token
# ---------------------------------------------------------------------------


def test_diarize_raises_when_hf_token_missing() -> None:
    """diarize() raises DiarizationError when hf_token is None/empty."""
    result = _sample_result()
    with patch(
        "insanely_fast_whisper_rocm.core.integrations.diarization.Pipeline",
        MagicMock(),  # not None → pyannote "installed"
    ):
        with pytest.raises(DiarizationError, match="HF_TOKEN"):
            diarize(result, audio_path="/fake.wav", hf_token=None)

        with pytest.raises(DiarizationError, match="HF_TOKEN"):
            diarize(result, audio_path="/fake.wav", hf_token="")


# ---------------------------------------------------------------------------
# diarize() — 403 error (gated model license)
# ---------------------------------------------------------------------------


def test_diarize_raises_on_403_license_not_accepted() -> None:
    """diarize() raises DiarizationError with license link on HTTP 403."""
    result = _sample_result()
    mock_pipeline_cls = MagicMock()
    load_exc = Exception("forbidden")
    load_exc.status_code = 403  # type: ignore[attr-defined]
    mock_pipeline_cls.from_pretrained.side_effect = load_exc

    with patch(
        "insanely_fast_whisper_rocm.core.integrations.diarization.Pipeline",
        mock_pipeline_cls,
    ):
        with pytest.raises(DiarizationError, match=r"huggingface\.co"):
            diarize(result, audio_path="/fake.wav", hf_token="tok")


# ---------------------------------------------------------------------------
# diarize() — successful run
# ---------------------------------------------------------------------------


def test_diarize_assigns_speakers_to_chunks() -> None:
    """diarize() adds speaker labels to chunks and sets diarized=True."""
    result = _sample_result()

    # Build a fake pyannote annotation
    mock_annotation = MagicMock()
    mock_annotation.itertracks.return_value = [
        (MagicMock(start=0.0, end=2.5), None, "SPEAKER_00"),
        (MagicMock(start=2.5, end=5.0), None, "SPEAKER_01"),
    ]

    # Simulate pyannote v4 DiarizeOutput dataclass
    mock_output = MagicMock(spec=[])
    mock_output.speaker_diarization = mock_annotation

    mock_pipeline_instance = MagicMock()
    mock_pipeline_instance.return_value = mock_output
    mock_pipeline_instance.to.return_value = mock_pipeline_instance

    mock_pipeline_cls = MagicMock()
    mock_pipeline_cls.from_pretrained.return_value = mock_pipeline_instance

    with (
        patch(
            "insanely_fast_whisper_rocm.core.integrations.diarization.Pipeline",
            mock_pipeline_cls,
        ),
        patch(
            "insanely_fast_whisper_rocm.core.integrations.diarization._TORCHCODEC_AVAILABLE",
            False,
        ),
        patch("torchaudio.load") as mock_load,
    ):
        mock_load.return_value = (MagicMock(shape=torch.Size([1, 48000])), 16000)
        out = diarize(
            result,
            audio_path="/fake.wav",
            hf_token="tok",
            num_speakers=2,
        )

    assert out["diarized"] is True
    assert out["chunks"][0]["speaker"] == "SPEAKER_00"
    assert out["chunks"][1]["speaker"] == "SPEAKER_01"


# ---------------------------------------------------------------------------
# _align_speakers_to_segments()
# ---------------------------------------------------------------------------


def test_align_full_overlap() -> None:
    """Chunk fully inside a speaker turn gets that speaker."""
    chunks = [{"start": 1.0, "end": 2.0, "text": "hi"}]
    turns = [(0.0, 3.0, "A")]
    aligned = _align_speakers_to_segments(chunks, turns)
    assert aligned[0]["speaker"] == "A"


def test_align_partial_overlap_dominant() -> None:
    """Chunk overlapping two speakers gets the one with most overlap."""
    chunks = [{"start": 1.0, "end": 3.0, "text": "hi"}]
    turns = [
        (0.0, 2.0, "A"),  # 1s overlap
        (2.0, 4.0, "B"),  # 1s overlap — tie, A wins (first)
    ]
    aligned = _align_speakers_to_segments(chunks, turns)
    # Both have 1.0s overlap; A wins the tie (strict > keeps first match)
    assert aligned[0]["speaker"] == "A"


def test_align_no_overlap() -> None:
    """Chunk with no overlap gets speaker=None."""
    chunks = [{"start": 10.0, "end": 12.0, "text": "hi"}]
    turns = [(0.0, 3.0, "A")]
    aligned = _align_speakers_to_segments(chunks, turns)
    assert aligned[0]["speaker"] is None


def test_align_multiple_speakers_dominant() -> None:
    """Chunk gets the speaker with the longest overlap duration."""
    chunks = [{"start": 0.0, "end": 5.0, "text": "hi"}]
    turns = [
        (0.0, 1.0, "A"),  # 1s overlap
        (1.0, 4.5, "B"),  # 3.5s overlap
        (4.5, 5.0, "C"),  # 0.5s overlap
    ]
    aligned = _align_speakers_to_segments(chunks, turns)
    assert aligned[0]["speaker"] == "B"


def test_align_handles_none_timestamp_end() -> None:
    """Chunk with ``timestamp=(start, None)`` does not crash alignment."""
    chunks = [{"timestamp": (2.0, None), "text": "tail"}]
    turns = [(1.0, 3.0, "A")]

    aligned = _align_speakers_to_segments(chunks, turns)

    assert aligned[0]["speaker"] is None


# ---------------------------------------------------------------------------
# Pipeline caching
# ---------------------------------------------------------------------------


def test_pipeline_cache_reuses_same_config() -> None:
    """Same (model, device) returns the same cached pipeline object."""
    mock_pipeline_instance = MagicMock()
    mock_pipeline_instance.to.return_value = mock_pipeline_instance

    mock_pipeline_cls = MagicMock()
    mock_pipeline_cls.from_pretrained.return_value = mock_pipeline_instance

    with patch(
        "insanely_fast_whisper_rocm.core.integrations.diarization.Pipeline",
        mock_pipeline_cls,
    ):
        from insanely_fast_whisper_rocm.core.integrations.diarization import (
            _get_or_create_pipeline,
        )

        p1 = _get_or_create_pipeline("model-a", "cpu", "tok")
        p2 = _get_or_create_pipeline("model-a", "cpu", "tok")
        assert p1 is p2
        # from_pretrained called only once
        assert mock_pipeline_cls.from_pretrained.call_count == 1


def test_pipeline_cache_different_config() -> None:
    """Different (model, device) creates different pipeline objects."""
    mock_a = MagicMock()
    mock_a.to.return_value = mock_a
    mock_b = MagicMock()
    mock_b.to.return_value = mock_b

    mock_pipeline_cls = MagicMock()
    mock_pipeline_cls.from_pretrained.side_effect = [mock_a, mock_b]

    with patch(
        "insanely_fast_whisper_rocm.core.integrations.diarization.Pipeline",
        mock_pipeline_cls,
    ):
        from insanely_fast_whisper_rocm.core.integrations.diarization import (
            _get_or_create_pipeline,
        )

        p1 = _get_or_create_pipeline("model-a", "cpu", "tok")
        p2 = _get_or_create_pipeline("model-b", "cpu", "tok")
        assert p1 is not p2


def test_pipeline_cache_different_token_creates_separate_entry() -> None:
    """Different HF tokens produce separate cache entries."""
    mock_a = MagicMock()
    mock_a.to.return_value = mock_a
    mock_b = MagicMock()
    mock_b.to.return_value = mock_b

    mock_pipeline_cls = MagicMock()
    mock_pipeline_cls.from_pretrained.side_effect = [mock_a, mock_b]

    with patch(
        "insanely_fast_whisper_rocm.core.integrations.diarization.Pipeline",
        mock_pipeline_cls,
    ):
        from insanely_fast_whisper_rocm.core.integrations.diarization import (
            _get_or_create_pipeline,
        )

        p1 = _get_or_create_pipeline("model-a", "cpu", "token-1")
        p2 = _get_or_create_pipeline("model-a", "cpu", "token-2")
        assert p1 is not p2
        assert mock_pipeline_cls.from_pretrained.call_count == 2


def test_clear_diarization_cache_invalidates() -> None:
    """clear_diarization_cache() empties the cache."""
    mock_pipeline_instance = MagicMock()
    mock_pipeline_instance.to.return_value = mock_pipeline_instance

    mock_pipeline_cls = MagicMock()
    mock_pipeline_cls.from_pretrained.return_value = mock_pipeline_instance

    with patch(
        "insanely_fast_whisper_rocm.core.integrations.diarization.Pipeline",
        mock_pipeline_cls,
    ):
        from insanely_fast_whisper_rocm.core.integrations.diarization import (
            _CACHE,
            _get_or_create_pipeline,
        )

        _get_or_create_pipeline("model-x", "cpu", "tok")
        assert len(_CACHE) > 0

        clear_diarization_cache()
        assert len(_CACHE) == 0


# ---------------------------------------------------------------------------
# diarize() — graceful degradation on runtime error
# ---------------------------------------------------------------------------


def test_diarize_raises_on_inference_error() -> None:
    """diarize() raises DiarizationError when pipeline inference fails."""
    result = _sample_result()

    mock_pipeline_instance = MagicMock()
    mock_pipeline_instance.to.return_value = mock_pipeline_instance
    mock_pipeline_instance.side_effect = RuntimeError("OOM")

    mock_pipeline_cls = MagicMock()
    mock_pipeline_cls.from_pretrained.return_value = mock_pipeline_instance

    with (
        patch(
            "insanely_fast_whisper_rocm.core.integrations.diarization.Pipeline",
            mock_pipeline_cls,
        ),
        patch(
            "insanely_fast_whisper_rocm.core.integrations.diarization._TORCHCODEC_AVAILABLE",
            False,
        ),
        patch("torchaudio.load") as mock_load,
    ):
        mock_load.return_value = (MagicMock(shape=torch.Size([1, 48000])), 16000)
        with pytest.raises(DiarizationError, match="inference"):
            diarize(result, audio_path="/fake.wav", hf_token="tok")


def test_diarize_retries_cpu_on_rocm_miopen_error() -> None:
    """diarize() retries on CPU for known ROCm/MIOpen GPU inference failures."""
    result = _sample_result()

    mock_annotation = MagicMock()
    mock_annotation.itertracks.return_value = [
        (MagicMock(start=0.0, end=2.5), None, "SPEAKER_00"),
    ]

    mock_output = MagicMock(spec=[])
    mock_output.speaker_diarization = mock_annotation

    gpu_pipeline = MagicMock()
    gpu_pipeline.to.return_value = gpu_pipeline
    gpu_pipeline.side_effect = RuntimeError("miopenStatusUnknownError")

    cpu_pipeline = MagicMock()
    cpu_pipeline.to.return_value = cpu_pipeline
    cpu_pipeline.return_value = mock_output

    mock_pipeline_cls = MagicMock()
    mock_pipeline_cls.from_pretrained.side_effect = [gpu_pipeline, cpu_pipeline]

    with (
        patch(
            "insanely_fast_whisper_rocm.core.integrations.diarization.Pipeline",
            mock_pipeline_cls,
        ),
        patch(
            "insanely_fast_whisper_rocm.core.integrations.diarization._TORCHCODEC_AVAILABLE",
            False,
        ),
        patch("torchaudio.load") as mock_load,
    ):
        mock_load.return_value = (MagicMock(shape=torch.Size([1, 48000])), 16000)
        out = diarize(
            result,
            audio_path="/fake.wav",
            hf_token="tok",
            device="cuda",
        )

    assert out["diarized"] is True
    assert out["chunks"][0]["speaker"] == "SPEAKER_00"
    assert gpu_pipeline.call_count == 1
    assert cpu_pipeline.call_count == 1


def test_diarize_no_chunks_returns_unchanged() -> None:
    """diarize() returns result unchanged when there are no chunks."""
    result: dict[str, Any] = {"text": "Hello", "chunks": []}

    mock_annotation = MagicMock()
    mock_annotation.itertracks.return_value = [
        (MagicMock(start=0.0, end=1.0), None, "SPEAKER_00"),
    ]

    # Simulate pyannote v4 DiarizeOutput
    mock_output = MagicMock(spec=[])
    mock_output.speaker_diarization = mock_annotation

    mock_pipeline_instance = MagicMock()
    mock_pipeline_instance.return_value = mock_output
    mock_pipeline_instance.to.return_value = mock_pipeline_instance

    mock_pipeline_cls = MagicMock()
    mock_pipeline_cls.from_pretrained.return_value = mock_pipeline_instance

    with (
        patch(
            "insanely_fast_whisper_rocm.core.integrations.diarization.Pipeline",
            mock_pipeline_cls,
        ),
        patch(
            "insanely_fast_whisper_rocm.core.integrations.diarization._TORCHCODEC_AVAILABLE",
            False,
        ),
        patch("torchaudio.load") as mock_load,
    ):
        mock_load.return_value = (MagicMock(shape=torch.Size([1, 48000])), 16000)
        out = diarize(result, audio_path="/fake.wav", hf_token="tok")

    # No chunks → no alignment, returns original
    assert "diarized" not in out


def test_diarize_no_speaker_turns_returns_unchanged() -> None:
    """diarize() returns result unchanged when pyannote finds no turns."""
    result = _sample_result()

    mock_annotation = MagicMock()
    mock_annotation.itertracks.return_value = []

    # Simulate pyannote v4 DiarizeOutput
    mock_output = MagicMock(spec=[])
    mock_output.speaker_diarization = mock_annotation

    mock_pipeline_instance = MagicMock()
    mock_pipeline_instance.return_value = mock_output
    mock_pipeline_instance.to.return_value = mock_pipeline_instance

    mock_pipeline_cls = MagicMock()
    mock_pipeline_cls.from_pretrained.return_value = mock_pipeline_instance

    with (
        patch(
            "insanely_fast_whisper_rocm.core.integrations.diarization.Pipeline",
            mock_pipeline_cls,
        ),
        patch(
            "insanely_fast_whisper_rocm.core.integrations.diarization._TORCHCODEC_AVAILABLE",
            False,
        ),
        patch("torchaudio.load") as mock_load,
    ):
        mock_load.return_value = (MagicMock(shape=torch.Size([1, 48000])), 16000)
        out = diarize(result, audio_path="/fake.wav", hf_token="tok")

    assert "diarized" not in out


# ---------------------------------------------------------------------------
# diarize() — torchcodec preload path
# ---------------------------------------------------------------------------


def test_diarize_preloads_audio_when_torchcodec_unavailable() -> None:
    """diarize() preloads audio via torchaudio when torchcodec is missing."""
    result = _sample_result()

    mock_annotation = MagicMock()
    mock_annotation.itertracks.return_value = [
        (MagicMock(start=0.0, end=2.5), None, "SPEAKER_00"),
    ]

    mock_output = MagicMock(spec=[])
    mock_output.speaker_diarization = mock_annotation

    mock_pipeline_instance = MagicMock()
    mock_pipeline_instance.return_value = mock_output
    mock_pipeline_instance.to.return_value = mock_pipeline_instance

    mock_pipeline_cls = MagicMock()
    mock_pipeline_cls.from_pretrained.return_value = mock_pipeline_instance

    mock_waveform = MagicMock(shape=torch.Size([1, 48000]))
    with (
        patch(
            "insanely_fast_whisper_rocm.core.integrations.diarization.Pipeline",
            mock_pipeline_cls,
        ),
        patch(
            "insanely_fast_whisper_rocm.core.integrations.diarization._TORCHCODEC_AVAILABLE",
            False,
        ),
        patch("torchaudio.load") as mock_load,
    ):
        mock_load.return_value = (mock_waveform, 16000)
        out = diarize(result, audio_path="/fake.wav", hf_token="tok")

    # Verify torchaudio.load was called with the audio path
    mock_load.assert_called_once_with("/fake.wav")
    # Verify pipeline received a dict (not a string path)
    call_args = mock_pipeline_instance.call_args
    assert isinstance(call_args[0][0], dict)
    assert "waveform" in call_args[0][0]
    assert "sample_rate" in call_args[0][0]
    assert out["diarized"] is True


def test_diarize_passes_file_path_when_torchcodec_available() -> None:
    """diarize() passes file path directly when torchcodec is available."""
    result = _sample_result()

    mock_annotation = MagicMock()
    mock_annotation.itertracks.return_value = [
        (MagicMock(start=0.0, end=2.5), None, "SPEAKER_00"),
    ]

    mock_output = MagicMock(spec=[])
    mock_output.speaker_diarization = mock_annotation

    mock_pipeline_instance = MagicMock()
    mock_pipeline_instance.return_value = mock_output
    mock_pipeline_instance.to.return_value = mock_pipeline_instance

    mock_pipeline_cls = MagicMock()
    mock_pipeline_cls.from_pretrained.return_value = mock_pipeline_instance

    with (
        patch(
            "insanely_fast_whisper_rocm.core.integrations.diarization.Pipeline",
            mock_pipeline_cls,
        ),
        patch(
            "insanely_fast_whisper_rocm.core.integrations.diarization._TORCHCODEC_AVAILABLE",
            True,
        ),
    ):
        out = diarize(result, audio_path="/fake.wav", hf_token="tok")

    # Verify pipeline received the string path directly
    call_args = mock_pipeline_instance.call_args
    assert call_args[0][0] == "/fake.wav"
    assert out["diarized"] is True


def test_diarize_preserves_stabilized_segments_structure() -> None:
    """Diarization keeps existing ``segments`` shape while adding speakers."""
    result = {
        "text": "Hello world. Goodbye world.",
        "chunks": [
            {"timestamp": (0.0, 2.0), "text": "Hello world."},
            {"timestamp": (2.0, 4.0), "text": "Goodbye world."},
        ],
        "segments": [
            {"start": 0.0, "end": 2.0, "text": "Hello world."},
            {"start": 2.0, "end": 4.0, "text": "Goodbye world."},
        ],
    }

    mock_annotation = MagicMock()
    mock_annotation.itertracks.return_value = [
        (MagicMock(start=0.0, end=2.0), None, "SPEAKER_00"),
        (MagicMock(start=2.0, end=4.0), None, "SPEAKER_01"),
    ]

    mock_output = MagicMock(spec=[])
    mock_output.speaker_diarization = mock_annotation

    mock_pipeline_instance = MagicMock()
    mock_pipeline_instance.return_value = mock_output
    mock_pipeline_instance.to.return_value = mock_pipeline_instance

    mock_pipeline_cls = MagicMock()
    mock_pipeline_cls.from_pretrained.return_value = mock_pipeline_instance

    with (
        patch(
            "insanely_fast_whisper_rocm.core.integrations.diarization.Pipeline",
            mock_pipeline_cls,
        ),
        patch(
            "insanely_fast_whisper_rocm.core.integrations.diarization._TORCHCODEC_AVAILABLE",
            False,
        ),
        patch("torchaudio.load") as mock_load,
    ):
        mock_load.return_value = (MagicMock(shape=torch.Size([1, 48000])), 16000)
        out = diarize(result, audio_path="/fake.wav", hf_token="tok")

    assert out["diarized"] is True
    assert "timestamp" in out["chunks"][0]
    assert out["segments"][0]["start"] == 0.0
    assert out["segments"][0]["end"] == 2.0
    assert out["segments"][0]["speaker"] == "SPEAKER_00"


# ---------------------------------------------------------------------------
# _preload_audio_as_waveform — audio preloading for torchcodec-less envs
# ---------------------------------------------------------------------------


def test_preload_audio_direct_load() -> None:
    """torchaudio.load succeeds on first try for WAV files."""
    fake_waveform = MagicMock(shape=torch.Size([1, 48000]))
    with patch("torchaudio.load", return_value=(fake_waveform, 16000)):
        result = _preload_audio_as_waveform("/audio.wav")

    assert isinstance(result, dict)
    assert result["sample_rate"] == 16000
    assert result["waveform"] is fake_waveform


def test_preload_audio_ffmpeg_fallback() -> None:
    """Falls back to ffmpeg conversion when torchaudio.load fails (e.g. m4a)."""
    fake_waveform = MagicMock(shape=torch.Size([1, 48000]))
    mock_completed = MagicMock()
    mock_completed.returncode = 0

    with (
        patch(
            "torchaudio.load",
            side_effect=[RuntimeError("Format not recognised"), (fake_waveform, 16000)],
        ),
        patch("subprocess.run", return_value=mock_completed),
        patch("tempfile.NamedTemporaryFile") as mock_tmp,
    ):
        mock_file = MagicMock()
        mock_file.name = "/tmp/diarize_test.wav"
        mock_file.__enter__ = MagicMock(return_value=mock_file)
        mock_file.__exit__ = MagicMock(return_value=False)
        mock_tmp.return_value = mock_file

        result = _preload_audio_as_waveform("/audio.m4a")

    assert isinstance(result, dict)
    assert result["sample_rate"] == 16000
    assert result["waveform"] is fake_waveform


def test_preload_audio_ffmpeg_fails_returns_path() -> None:
    """Returns the original path string when both torchaudio and ffmpeg fail."""
    mock_completed = MagicMock()
    mock_completed.returncode = 1
    mock_completed.stderr = "Conversion failed"

    with (
        patch("torchaudio.load", side_effect=RuntimeError("Format not recognised")),
        patch("subprocess.run", return_value=mock_completed),
        patch("tempfile.NamedTemporaryFile") as mock_tmp,
    ):
        mock_file = MagicMock()
        mock_file.name = "/tmp/diarize_test.wav"
        mock_file.__enter__ = MagicMock(return_value=mock_file)
        mock_file.__exit__ = MagicMock(return_value=False)
        mock_tmp.return_value = mock_file

        result = _preload_audio_as_waveform("/audio.m4a")

    assert result == "/audio.m4a"
