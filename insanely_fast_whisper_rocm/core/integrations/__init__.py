"""Subpackage for optional integrations (e.g., stable-ts, diarization).

Each integration is imported lazily by its consumers to avoid pulling
optional dependencies (like pyannote.audio) at module load time.
"""

from .stable_ts import stabilize_timestamps  # noqa: F401

__all__ = [  # noqa: F822  # diarize/clear_diarization_cache via __getattr__
    "stabilize_timestamps",
    "diarize",
    "clear_diarization_cache",
]


def __getattr__(name: str) -> object:
    """Lazy re-export for diarization symbols.

    Prevents pyannote.audio from being imported until the diarization
    integration is actually used, avoiding spurious torchcodec warnings
    during CLI startup.

    Returns:
        The requested diarization symbol (``diarize`` or
        ``clear_diarization_cache``).

    Raises:
        AttributeError: If *name* is not a lazily re-exported symbol.
    """
    if name in ("diarize", "clear_diarization_cache"):
        from . import diarization as _diarization

        return getattr(_diarization, name)
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
