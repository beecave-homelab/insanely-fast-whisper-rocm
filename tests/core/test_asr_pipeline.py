"""ASR pipeline tests focusing on progress callbacks and chunking behavior."""

import wave
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from insanely_fast_whisper_api.core import ASRPipeline, TranscriptionError

# A smaller model for faster testing if available, otherwise default.
# You might need to adjust this if 'openai/whisper-tiny' is not suitable or always available.
TEST_MODEL = "openai/whisper-tiny"


@pytest.fixture
def dummy_audio_file(tmp_path: Path) -> Callable[[float], str]:
    """Factory fixture to create a dummy WAV audio file of specified duration.

    Returns:
        Callable[[float], str]: Function that generates a WAV file of the given
        duration (in seconds) and returns its filesystem path as a string.
    """

    def _create_audio(duration_seconds: float = 1.0) -> str:
        file_path = tmp_path / f"test_audio_{duration_seconds}s.wav"
        sample_rate = 16000  # Hz
        frequency = 440  # Hz (A4 note)
        n_samples = int(sample_rate * duration_seconds)

        t = np.linspace(0, duration_seconds, n_samples, False)
        audio_data = (np.sin(2 * np.pi * frequency * t) * 32767).astype(np.int16)

        with wave.open(str(file_path), "w") as wf:
            wf.setnchannels(1)  # Mono
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(sample_rate)
            wf.writeframes(audio_data.tobytes())
        return str(file_path)

    return _create_audio


@pytest.fixture
def short_audio_file(dummy_audio_file: Callable[[float], str]) -> str:
    """Provide a short (1s) dummy audio file for non-chunking tests.

    Returns:
        str: Filesystem path to the generated WAV file.
    """
    return dummy_audio_file(1.0)


@pytest.fixture
def long_audio_file_for_chunking(dummy_audio_file: Callable[[float], str]) -> str:
    """Provide a longer (e.g., 5s) dummy audio file to trigger chunking.

    Returns:
        str: Filesystem path to the generated WAV file.
    """
    # Duration should be longer than the ASRPipeline's default chunk_length if we want to test chunking.
    # However, ASRPipeline default chunk_length is 30s. For faster tests, we'll use a shorter chunk_length
    # in the test setup itself.
    return dummy_audio_file(5.0)  # 5 seconds audio


def test_asr_pipeline_callback_non_chunked(short_audio_file: str) -> None:
    """Test ASRPipeline progress callbacks for non-chunked transcription."""
    progress_updates: list[tuple[str, int, int, str | None]] = []

    def mock_callback(
        stage: str, current_step: int, total_steps: int, message: str | None = None
    ) -> None:
        progress_updates.append((stage, current_step, total_steps, message))

    pipeline = ASRPipeline(
        model=TEST_MODEL,
        device="cpu",  # Use CPU for testing to avoid GPU dependencies
        progress_callback=mock_callback,
    )

    try:
        pipeline(short_audio_file)
    except TranscriptionError as e:
        # If transcription itself fails, that's a separate issue for this model/setup.
        # For this test, we are primarily interested in callbacks *before* a potential error.
        print(f"TranscriptionError during test: {e}")

    print(f"Progress updates received: {progress_updates}")

    # Assertions for callback stages
    # Stage 1: Model Loading Start
    assert len(progress_updates) > 0, "No progress updates received"
    assert progress_updates[0][0] == "MODEL_LOADING_START"
    assert progress_updates[0][1] == 0  # current_step
    assert progress_updates[0][2] == 1  # total_steps

    # Stage 2: Model Loading Complete
    assert len(progress_updates) > 1, "Model loading complete not reported"
    assert progress_updates[1][0] == "MODEL_LOADING_COMPLETE"
    assert progress_updates[1][1] == 1  # current_step
    assert progress_updates[1][2] == 1  # total_steps

    # Stage 3: Single File Processing Start
    # This might be the 3rd or later update depending on other potential internal callbacks
    # We'll search for it.
    single_file_start_found = any(
        update[0] == "SINGLE_FILE_PROCESSING_START"
        and update[1] == 0
        and update[2] == 1
        for update in progress_updates
    )
    assert single_file_start_found, (
        "Single file processing start not reported correctly"
    )

    # Stage 4: Overall Processing Complete (should be the last one)
    # This assumes transcription doesn't error out before completion.
    # If TranscriptionError occurs, this stage might not be reached.
    if not any(
        "TranscriptionError" in str(update) for update in progress_updates
    ):  # Cheap check if error was logged in updates
        overall_complete_found = any(
            update[0] == "OVERALL_PROCESSING_COMPLETE"
            and update[1] == 1
            and update[2] == 1
            for update in progress_updates
        )
        assert overall_complete_found, (
            "Overall processing complete not reported correctly"
        )


# More tests will be added here
