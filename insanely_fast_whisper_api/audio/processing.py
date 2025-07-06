"""Audio processing utilities for chunking and manipulation."""

import os
import tempfile
from typing import List

from pydub import AudioSegment

from insanely_fast_whisper_api.utils import cleanup_temp_files


def get_audio_duration(audio_path: str) -> float:
    """
    Get the duration of an audio file in seconds.

    Args:
        audio_path: Path to the audio file.

    Returns:
        float: Duration of the audio in seconds.
    """
    try:
        audio = AudioSegment.from_file(audio_path)
        return len(audio) / 1000.0  # Convert ms to seconds
    except (OSError, IOError, RuntimeError) as e:
        raise RuntimeError(
            f"Failed to get audio duration for {audio_path}: {str(e)}"
        ) from e


def split_audio(
    audio_path: str,
    chunk_duration: float = 600.0,
    chunk_overlap: float = 1.0,
    min_chunk_duration: float = 5.0,
) -> List[str]:
    """
    Split an audio file into chunks of specified duration.

    Args:
        audio_path: Path to the input audio file.
        chunk_duration: Target duration of each chunk in seconds.
        chunk_overlap: Overlap between chunks in seconds.
        min_chunk_duration: Minimum duration of a chunk in seconds.

    Returns:
        List of paths to the generated audio chunks.
    """
    try:
        # Validate inputs
        if chunk_duration <= 0:
            raise ValueError("chunk_duration must be greater than 0")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")
        if chunk_overlap >= chunk_duration:
            raise ValueError("chunk_overlap must be less than chunk_duration")
        if min_chunk_duration <= 0:
            raise ValueError("min_chunk_duration must be greater than 0")

        # Load the audio file
        audio = AudioSegment.from_file(audio_path)
        duration_ms = len(audio)
        chunk_duration_ms = int(chunk_duration * 1000)
        overlap_ms = int(chunk_overlap * 1000)
        min_chunk_duration_ms = int(min_chunk_duration * 1000)

        # If audio is shorter than chunk duration, return the original file
        if duration_ms <= chunk_duration_ms + overlap_ms:
            return [audio_path]

        # Create a temporary directory for chunks
        temp_dir = tempfile.mkdtemp(prefix="audio_chunks_")
        chunk_paths = []
        start_ms = 0
        chunk_num = 1

        while start_ms < duration_ms - min_chunk_duration_ms:
            # Calculate end time with overlap
            end_ms = min(start_ms + chunk_duration_ms + overlap_ms, duration_ms)

            # Extract chunk
            chunk = audio[start_ms:end_ms]

            # Save chunk
            chunk_path = os.path.join(temp_dir, f"chunk_{chunk_num:04d}.wav")
            chunk.export(chunk_path, format="wav")
            chunk_paths.append(chunk_path)

            # Move to next chunk (accounting for overlap)
            start_ms += chunk_duration_ms
            chunk_num += 1

        return chunk_paths

    except (OSError, IOError, RuntimeError, MemoryError) as e:
        # Clean up any created files before re-raising
        if "chunk_paths" in locals() and locals()["chunk_paths"]:
            cleanup_temp_files(locals()["chunk_paths"])
        raise RuntimeError(f"Failed to split audio: {str(e)}") from e
