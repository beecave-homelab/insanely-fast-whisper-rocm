"""Public surface for the Insanely Fast Whisper API package.

The module re-exports common audio utilities, the lightweight test-friendly
`ASRPipeline`, and (when available) the optional `benchmarks` subpackage to
preserve backward compatibility with existing integrations.
"""

from __future__ import annotations

from importlib import metadata

from insanely_fast_whisper_rocm.audio import (
    cleanup_temp_files,
    get_audio_duration,
    merge_chunk_results,
    split_audio,
)
from insanely_fast_whisper_rocm.core import ASRPipeline
from insanely_fast_whisper_rocm.utils import constants


def _resolve_package_version() -> str:
    """Return the package version string for the distribution.

    Returns:
        str: Semantic version read from installed metadata or project constants.
    """
    try:
        return metadata.version("insanely-fast-whisper-rocm")
    except metadata.PackageNotFoundError:
        candidate = getattr(constants, "API_VERSION", "")
        return candidate or "0.0.0-dev"


__version__ = _resolve_package_version()

try:
    from insanely_fast_whisper_rocm import benchmarks
except ModuleNotFoundError:  # pragma: no cover
    benchmarks = None  # type: ignore[assignment]

__all__ = [
    "ASRPipeline",
    "__version__",
    "benchmarks",
    "cleanup_temp_files",
    "constants",
    "get_audio_duration",
    "merge_chunk_results",
    "split_audio",
]
