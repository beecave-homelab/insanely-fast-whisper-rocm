"""Tests for the core ASR pipeline functionality."""

from pathlib import Path

import pytest

from insanely_fast_whisper_rocm import ASRPipeline


def test_asr_pipeline_initialization():
    """Test that the ASR pipeline can be initialized with default parameters."""
    asr = ASRPipeline()
    assert asr.model_name == "openai/whisper-base"
    assert asr.device == "cpu"
    assert asr.dtype == "float32"


def test_asr_pipeline_custom_params():
    """Test that the ASR pipeline can be initialized with custom parameters."""
    asr = ASRPipeline(
        model="openai/whisper-large-v3",
        device="cuda:0",
        dtype="float16",
        better_transformer=True,
    )
    assert asr.model_name == "openai/whisper-large-v3"
    assert asr.device == "cuda:0"
    assert asr.dtype == "float16"


@pytest.mark.integration
def test_asr_pipeline_inference(tmp_path):
    """Test the ASR pipeline inference on a sample audio file.

    This test requires a sample audio file and is marked as integration test.
    """
    audio_file = Path(__file__).parent / "test.mp3"

    assert audio_file.exists(), f"Test audio file not found: {audio_file}"

    # Initialize with a small model, CPU, and float32 for faster and robust testing.
    asr = ASRPipeline(model="openai/whisper-tiny", device="cpu", dtype="float32")

    result = asr(audio_file_path=str(audio_file))

    assert isinstance(result, dict), "ASR result should be a dictionary."
    assert "text" in result, "ASR result dictionary should contain a 'text' key."
    assert isinstance(result["text"], str), "'text' field should be a string."
    assert len(result["text"].strip()) > 0, "'text' field should not be empty."
    # Content of test.mp3: "The Taming of the Shrew is a comedy by William Shakespeare..."
    assert "Shrew" in result["text"], "Transcription should contain 'Shrew'."
