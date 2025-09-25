"""Public surface for the Insanely Fast Whisper API package.

The module re-exports common audio utilities, the lightweight test-friendly
`ASRPipeline`, and (when available) the optional `benchmarks` subpackage to
preserve backward compatibility with existing integrations.
"""

from __future__ import annotations

from insanely_fast_whisper_api.audio import (
    cleanup_temp_files,
    get_audio_duration,
    merge_chunk_results,
    split_audio,
)
from insanely_fast_whisper_api.core import ASRPipeline
from insanely_fast_whisper_api.utils import constants

try:
    from insanely_fast_whisper_api import benchmarks
except ModuleNotFoundError:  # pragma: no cover
    benchmarks = None  # type: ignore[assignment]

__all__ = [
    "ASRPipeline",
    "benchmarks",
    "cleanup_temp_files",
    "constants",
    "get_audio_duration",
    "merge_chunk_results",
    "split_audio",
]
