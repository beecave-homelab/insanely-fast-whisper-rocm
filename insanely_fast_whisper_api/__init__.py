"""Insanely Fast Whisper API.

A FastAPI wrapper around a custom Whisper-based ASR pipeline with audio
chunking support.
"""

from insanely_fast_whisper_api.utils.constants import API_VERSION as __version__

__author__ = "elvee"
__email__ = "lowie@beecave.nl"

# Audio utilities
import insanely_fast_whisper_api.utils.constants as constants
from insanely_fast_whisper_api.audio import (
    cleanup_temp_files,
    get_audio_duration,
    merge_chunk_results,
    split_audio,
)

# Core functionality
# from .core import ASRPipeline, run_asr_pipeline # Old import
from insanely_fast_whisper_api.core import ASRPipeline

__all__ = [
    "ASRPipeline",
    "constants",
    "get_audio_duration",
    "split_audio",
    "cleanup_temp_files",
    "merge_chunk_results",
    "__version__",
]
