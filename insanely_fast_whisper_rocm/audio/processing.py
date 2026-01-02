"""Audio processing utilities for chunking and manipulation."""

import os
import tempfile

import ffmpeg
from pydub import AudioSegment

from insanely_fast_whisper_rocm.utils.file_utils import cleanup_temp_files


def get_audio_duration(audio_path: str) -> float:
    """Get the duration of an audio file in seconds.

    Args:
        audio_path: Path to the audio file.

    Returns:
        float: Duration of the audio in seconds.

    Raises:
        RuntimeError: If fetching the audio duration fails.

    """
    try:
        audio = AudioSegment.from_file(audio_path)
        return len(audio) / 1000.0  # Convert ms to seconds
    except (OSError, RuntimeError) as e:
        raise RuntimeError(
            f"Failed to get audio duration for {audio_path}: {str(e)}"
        ) from e


def extract_audio_from_video(
    video_path: str,
    output_format: str = "wav",
    sample_rate: int = 16000,
    channels: int = 1,
) -> str:
    """Extract audio from a video file using FFmpeg.

    Args:
        video_path: Path to the input video file.
        output_format: Desired audio format (e.g., "wav").
        sample_rate: Target sample rate in Hz.
        channels: Number of audio channels (1 = mono, 2 = stereo).

    Returns:
        str: Path to the extracted audio file.

    Raises:
        FileNotFoundError: If the input video file does not exist.
        RuntimeError: If FFmpeg fails during extraction.

    """
    try:
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        tmp_dir = tempfile.mkdtemp(prefix="extracted_audio_")
        output_path = os.path.join(
            tmp_dir,
            f"{os.path.splitext(os.path.basename(video_path))[0]}.{output_format}",
        )

        (
            ffmpeg
            .input(video_path)
            .output(
                output_path,
                acodec="pcm_s16le",  # uncompressed WAV PCM 16-bit
                ac=channels,
                ar=sample_rate,
                vn=None,
            )
            .overwrite_output()
            .run(quiet=True)
        )
        return output_path
    except ffmpeg.Error as e:
        # Clean up potentially partially-written file
        if "output_path" in locals() and os.path.exists(output_path):
            cleanup_temp_files([output_path])
        raise RuntimeError(
            f"Failed to extract audio from video {video_path}: "
            f"{e.stderr.decode() if hasattr(e, 'stderr') else str(e)}"
        ) from e


def split_audio(
    audio_path: str,
    chunk_duration: float = 600.0,
    chunk_overlap: float = 1.0,
    min_chunk_duration: float = 5.0,
) -> list[tuple[str, float]]:
    """Split an audio file into chunks of specified duration.

    Args:
        audio_path: Path to the input audio file.
        chunk_duration: Target duration of each chunk in seconds.
        chunk_overlap: Overlap between chunks in seconds.
        min_chunk_duration: Minimum duration of a chunk in seconds.

    Returns:
        A list of tuples, where each tuple contains the path to the generated
        audio chunk and its start time in seconds.

    Raises:
        ValueError: If input parameters are invalid.
        RuntimeError: If splitting the audio fails due to an OS, runtime,
            or memory error.

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
            return [(audio_path, 0.0)]

        # Create a temporary directory for chunks
        temp_dir = tempfile.mkdtemp(prefix="audio_chunks_")
        chunk_paths: list[tuple[str, float]] = []
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
            chunk_paths.append((chunk_path, start_ms / 1000.0))

            # Move to next chunk (accounting for overlap)
            start_ms += chunk_duration_ms
            chunk_num += 1

        return chunk_paths

    except (OSError, RuntimeError, MemoryError) as e:
        # Clean up any created files before re-raising
        if "chunk_paths" in locals() and locals()["chunk_paths"]:
            # Extract just the file paths for cleanup
            paths_to_clean = [p[0] for p in locals()["chunk_paths"]]
            cleanup_temp_files(paths_to_clean)
        raise RuntimeError(f"Failed to split audio: {str(e)}") from e
