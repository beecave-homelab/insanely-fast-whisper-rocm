"""Test shim for `stable_ts` that forwards monkeypatches into production.

This file allows tests to import a path-local module but still exercise the
production implementation. When tests monkeypatch attributes like
`stable_whisper` or `_postprocess` on this shim module, we temporarily apply the
same patches to the production module during the call.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import insanely_fast_whisper_api.core.integrations.stable_ts as _prod

# Re-export helpers for direct testing
_to_dict: Callable[[object], dict[str, Any]] = _prod._to_dict  # type: ignore[attr-defined]
_convert_to_stable: Callable[[dict[str, Any]], dict[str, Any]] = (
    _prod._convert_to_stable
)  # type: ignore[attr-defined]


def stabilize_timestamps(
    result: dict[str, Any],
    *,
    demucs: bool = False,
    vad: bool = False,
    vad_threshold: float = 0.35,
    progress_cb: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Delegate to production stabilize_timestamps with test monkeypatches applied.

    Returns:
        The stabilized transcription result dictionary.
    """
    # Snapshot originals
    old_sw = getattr(_prod, "stable_whisper", None)
    old_pp = getattr(_prod, "_postprocess", None)
    old_pp_alt = getattr(_prod, "_postprocess_alt", None)

    # Apply any monkeypatched attributes set on this shim
    if "stable_whisper" in globals():  # type: ignore[truthy-function]
        setattr(_prod, "stable_whisper", globals()["stable_whisper"])  # type: ignore[arg-type]
    if "_postprocess" in globals():  # type: ignore[truthy-function]
        setattr(_prod, "_postprocess", globals()["_postprocess"])  # type: ignore[arg-type]
    if "_postprocess_alt" in globals():  # type: ignore[truthy-function]
        setattr(_prod, "_postprocess_alt", globals()["_postprocess_alt"])  # type: ignore[arg-type]

    try:
        return _prod.stabilize_timestamps(
            result,
            demucs=demucs,
            vad=vad,
            vad_threshold=vad_threshold,
            progress_cb=progress_cb,
        )
    finally:
        # Restore originals to avoid test cross-talk
        setattr(_prod, "stable_whisper", old_sw)
        setattr(_prod, "_postprocess", old_pp)
        setattr(_prod, "_postprocess_alt", old_pp_alt)


__all__ = [
    "_to_dict",
    "_convert_to_stable",
    "stabilize_timestamps",
]
