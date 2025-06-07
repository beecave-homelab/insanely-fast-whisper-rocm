"""Insanely Fast Whisper API.

A FastAPI wrapper around a custom Whisper-based ASR pipeline with audio chunking support.
"""

__version__ = "0.4.0"
__author__ = "elvee"
__email__ = "lowie@beecave.nl"

# Core functionality
# from .core import ASRPipeline, run_asr_pipeline # Old import
from insanely_fast_whisper_api.core.pipeline import (
    WhisperPipeline as ASRPipeline,
)  # New import

# Audio utilities
from insanely_fast_whisper_api.audio import (
    get_audio_duration,
    split_audio,
    cleanup_temp_files,
    merge_chunk_results,
)

__all__ = [
    "ASRPipeline",
    "get_audio_duration",
    "split_audio",
    "cleanup_temp_files",
    "merge_chunk_results",
]
