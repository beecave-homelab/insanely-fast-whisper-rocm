"""Tests for the speaker diarization integration module."""

from __future__ import annotations

import subprocess
from pathlib import Path
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
def _clean_cache() -> None:  # pyright: ignore[reportUnusedFunction]
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


def test_diarize__raises_when_pyannote_not_installed() -> None:
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


def test_diarize__raises_when_hf_token_missing() -> None:
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


def test_diarize__raises_on_403_license_not_accepted() -> None:
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


def test_diarize__assigns_speakers_to_chunks() -> None:
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


def test_align_speakers_to_segments__assigns_speaker_with_largest_overlap() -> None:
    """Chunk fully inside a speaker turn gets that speaker."""
    chunks = [{"start": 1.0, "end": 2.0, "text": "hi"}]
    turns = [(0.0, 3.0, "A")]
    aligned = _align_speakers_to_segments(chunks, turns)
    assert aligned[0]["speaker"] == "A"


def test_align_speakers_to_segments__prefers_first_on_tie() -> None:
    """Chunk overlapping two speakers equally gets the first turn (strict > tie-break)."""
    chunks = [{"start": 1.0, "end": 3.0, "text": "hi"}]
    turns = [
        (0.0, 2.0, "A"),  # 1s overlap
        (2.0, 4.0, "B"),  # 1s overlap — tie, A wins (first)
    ]
    aligned = _align_speakers_to_segments(chunks, turns)
    # Both have 1.0s overlap; A wins the tie (strict > keeps first match)
    assert aligned[0]["speaker"] == "A"


def test_align_speakers_to_segments__returns_none_when_no_overlap() -> None:
    """Chunk with no overlap gets speaker=None."""
    chunks = [{"start": 10.0, "end": 12.0, "text": "hi"}]
    turns = [(0.0, 3.0, "A")]
    aligned = _align_speakers_to_segments(chunks, turns)
    assert aligned[0]["speaker"] is None


def test_align_speakers_to_segments__selects_speaker_with_longest_overlap() -> None:
    """Chunk gets the speaker with the longest overlap duration."""
    chunks = [{"start": 0.0, "end": 5.0, "text": "hi"}]
    turns = [
        (0.0, 1.0, "A"),  # 1s overlap
        (1.0, 4.5, "B"),  # 3.5s overlap
        (4.5, 5.0, "C"),  # 0.5s overlap
    ]
    aligned = _align_speakers_to_segments(chunks, turns)
    assert aligned[0]["speaker"] == "B"


def test_align_speakers_to_segments__handles_none_timestamp_end() -> None:
    """Chunk with ``timestamp=(start, None)`` uses point-in-turn alignment."""
    chunks = [{"timestamp": (2.0, None), "text": "tail"}]
    turns = [(1.0, 3.0, "A")]

    aligned = _align_speakers_to_segments(chunks, turns)

    assert aligned[0]["speaker"] == "A"


# ---------------------------------------------------------------------------
# Pipeline caching
# ---------------------------------------------------------------------------


def test_get_or_create_pipeline__reuses_cached_instance_for_same_config() -> None:
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


def test_get_or_create_pipeline__creates_separate_instance_for_different_config() -> (
    None
):
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


def test_get_or_create_pipeline__creates_separate_entry_for_different_token() -> None:
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


def test_clear_diarization_cache__empties_cache() -> None:
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


def test_diarize__raises_on_inference_error() -> None:
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


@pytest.mark.parametrize(
    "gpu_error_message",
    [
        "miopenStatusUnknownError",
        "miopenStatusInvalidValue: bad param",
        "rocrand/rocrand_xorwow.h: not found",
        "HIP out of memory. Tried to allocate 312.00 MiB. GPU 0 has a total capacity of 7.98 GiB",
    ],
    ids=["miopenstatusunknownerror", "miopen-generic", "rocrand", "oom"],
)
def test_diarize__retries_cpu_on_rocm_miopen_error(
    gpu_error_message: str,
) -> None:
    """diarize() retries on CPU for known ROCm/MIOpen or OOM GPU failures."""
    result = _sample_result()

    mock_annotation = MagicMock()
    mock_annotation.itertracks.return_value = [
        (MagicMock(start=0.0, end=2.5), None, "SPEAKER_00"),
    ]

    mock_output = MagicMock(spec=[])
    mock_output.speaker_diarization = mock_annotation

    gpu_pipeline = MagicMock()
    gpu_pipeline.to.return_value = gpu_pipeline
    gpu_pipeline.side_effect = RuntimeError(gpu_error_message)

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


def test_diarize__does_not_retry_cpu_when_fallback_disabled() -> None:
    """diarize() honors the ROCm CPU fallback configuration switch."""
    result = _sample_result()

    gpu_pipeline = MagicMock()
    gpu_pipeline.to.return_value = gpu_pipeline
    gpu_pipeline.side_effect = RuntimeError("miopenStatusUnknownError")

    mock_pipeline_cls = MagicMock()
    mock_pipeline_cls.from_pretrained.return_value = gpu_pipeline

    with (
        patch(
            "insanely_fast_whisper_rocm.core.integrations.diarization.Pipeline",
            mock_pipeline_cls,
        ),
        patch(
            "insanely_fast_whisper_rocm.core.integrations.diarization._TORCHCODEC_AVAILABLE",
            False,
        ),
        patch(
            "insanely_fast_whisper_rocm.core.integrations.diarization.DIARIZATION_ALLOW_CPU_FALLBACK",
            False,
        ),
        patch("torchaudio.load") as mock_load,
    ):
        mock_load.return_value = (MagicMock(shape=torch.Size([1, 48000])), 16000)
        with pytest.raises(DiarizationError, match="inference"):
            diarize(
                result,
                audio_path="/fake.wav",
                hf_token="tok",
                device="cuda",
            )

    assert mock_pipeline_cls.from_pretrained.call_count == 1


def test_diarize__emits_timing_logs(caplog: pytest.LogCaptureFixture) -> None:
    """diarize() logs concise timing events without changing output."""
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

    caplog.set_level("INFO")
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
    messages = "\n".join(record.getMessage() for record in caplog.records)
    assert "Diarization timing: pipeline_ready_seconds=" in messages
    assert "Diarization timing: audio_preload_seconds=" in messages
    assert "Diarization timing: inference_seconds=" in messages
    assert "Diarization timing: speaker_turn_extraction_seconds=" in messages
    assert "Diarization timing: speaker_alignment_seconds=" in messages


def test_diarize__returns_unchanged_when_no_chunks_and_no_segments() -> None:
    """diarize() returns result unchanged when there are no chunks or segments."""
    result: dict[str, Any] = {"text": "Hello", "chunks": [], "segments": []}

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

    # No chunks and no segments → no alignment, returns original
    assert "diarized" not in out


def test_diarize__returns_unchanged_when_no_speaker_turns() -> None:
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


def test_diarize__preloads_audio_when_torchcodec_unavailable() -> None:
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


def test_diarize__passes_file_path_when_torchcodec_available() -> None:
    """diarize() can pass file paths when preload is explicitly disabled."""
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
        patch(
            "insanely_fast_whisper_rocm.core.integrations.diarization.DIARIZATION_PRELOAD_AUDIO",
            False,
        ),
    ):
        out = diarize(result, audio_path="/fake.wav", hf_token="tok")

    # Verify pipeline received the string path directly
    call_args = mock_pipeline_instance.call_args
    assert call_args[0][0] == "/fake.wav"
    assert out["diarized"] is True


def test_diarize__preloads_audio_when_torchcodec_available_by_default() -> None:
    """diarize() still uses the in-memory preload fast path with TorchCodec present."""
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
            True,
        ),
        patch("torchaudio.load", return_value=(mock_waveform, 16000)),
    ):
        out = diarize(result, audio_path="/fake.wav", hf_token="tok")

    call_args = mock_pipeline_instance.call_args
    assert isinstance(call_args[0][0], dict)
    assert out["diarized"] is True


def test_diarize__falls_back_to_segments_when_chunks_removed_by_stabilization() -> (  # noqa: E501
    None
):
    """diarize() aligns speakers via segments when chunks are absent.

    Regression test: after stabilization removes the ``chunks`` key,
    diarization must still process the ``segments`` key instead of
    silently returning the result unchanged.
    """
    # Simulate a result after stabilization: chunks removed, segments present
    result: dict[str, Any] = {
        "text": "Hello world. Goodbye world.",
        "segments": [
            {"start": 0.0, "end": 2.0, "text": "Hello world."},
            {"start": 2.5, "end": 5.0, "text": "Goodbye world."},
        ],
        "stabilized": True,
    }

    mock_annotation = MagicMock()
    mock_annotation.itertracks.return_value = [
        (MagicMock(start=0.0, end=2.5), None, "SPEAKER_00"),
        (MagicMock(start=2.5, end=5.0), None, "SPEAKER_01"),
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
    # Segments should have speaker labels
    assert out["segments"][0]["speaker"] == "SPEAKER_00"
    assert out["segments"][1]["speaker"] == "SPEAKER_01"
    # Chunks should also be populated from the aligned segments
    assert out["chunks"][0]["speaker"] == "SPEAKER_00"
    assert out["chunks"][1]["speaker"] == "SPEAKER_01"


def test_diarize__preserves_stabilized_segments_structure() -> None:
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


def test_preload_audio_as_waveform__loads_wav_directly() -> None:
    """torchaudio.load succeeds on first try for WAV files."""
    fake_waveform = MagicMock(shape=torch.Size([1, 48000]))
    with patch("torchaudio.load", return_value=(fake_waveform, 16000)):
        result = _preload_audio_as_waveform("/audio.wav")

    assert isinstance(result, dict)
    assert result["sample_rate"] == 16000
    assert result["waveform"] is fake_waveform


def test_preload_audio_as_waveform__uses_ffmpeg_decode_for_m4a(
    tmp_path: Path,
) -> None:
    """Falls back to ffmpeg conversion when torchaudio.load fails (e.g. m4a)."""
    fake_waveform = MagicMock(shape=torch.Size([1, 48000]))
    mock_completed = MagicMock()
    mock_completed.returncode = 0

    with (
        patch(
            "torchaudio.load",
            side_effect=[RuntimeError("Format not recognised"), (fake_waveform, 16000)],
        ),
        patch("subprocess.run", return_value=mock_completed) as mock_run,
        patch("tempfile.NamedTemporaryFile") as mock_tmp,
        patch(
            "insanely_fast_whisper_rocm.core.integrations.diarization.DIARIZATION_FFMPEG_TIMEOUT_SECONDS",
            12,
        ),
    ):
        mock_file = MagicMock()
        mock_file.name = str(tmp_path / "diarize_test.wav")
        mock_file.__enter__ = MagicMock(return_value=mock_file)
        mock_file.__exit__ = MagicMock(return_value=False)
        mock_tmp.return_value = mock_file

        result = _preload_audio_as_waveform("/audio.m4a")

    assert isinstance(result, dict)
    assert result["sample_rate"] == 16000
    assert result["waveform"] is fake_waveform
    mock_run.assert_called_once()
    assert mock_run.call_args.kwargs["timeout"] == 12


def test_preload_audio_as_waveform__raises_error_on_ffmpeg_failure(
    tmp_path: Path,
) -> None:
    """Raises DiarizationError when both torchaudio and ffmpeg fail."""
    mock_completed = MagicMock()
    mock_completed.returncode = 1
    mock_completed.stderr = "Conversion failed"

    with (
        patch("torchaudio.load", side_effect=RuntimeError("Format not recognised")),
        patch("subprocess.run", return_value=mock_completed),
        patch("tempfile.NamedTemporaryFile") as mock_tmp,
    ):
        mock_file = MagicMock()
        mock_file.name = str(tmp_path / "diarize_test.wav")
        mock_file.__enter__ = MagicMock(return_value=mock_file)
        mock_file.__exit__ = MagicMock(return_value=False)
        mock_tmp.return_value = mock_file

        with pytest.raises(DiarizationError) as exc_info:
            _preload_audio_as_waveform("/audio.m4a")
        assert exc_info.value.reason == "audio_decode_unavailable"


def test_preload_audio_as_waveform__raises_error_on_ffmpeg_timeout(
    tmp_path: Path,
) -> None:
    """Raises DiarizationError when ffmpeg conversion times out."""
    with (
        patch("torchaudio.load", side_effect=RuntimeError("Format not recognised")),
        patch(
            "subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd=["ffmpeg"], timeout=30),
        ),
        patch("tempfile.NamedTemporaryFile") as mock_tmp,
    ):
        mock_file = MagicMock()
        mock_file.name = str(tmp_path / "diarize_test.wav")
        mock_file.__enter__ = MagicMock(return_value=mock_file)
        mock_file.__exit__ = MagicMock(return_value=False)
        mock_tmp.return_value = mock_file

        with pytest.raises(DiarizationError) as exc_info:
            _preload_audio_as_waveform("/audio.m4a")
        assert exc_info.value.reason == "audio_decode_unavailable"
