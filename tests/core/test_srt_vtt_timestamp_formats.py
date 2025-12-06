"""Tests for timestamp formatting in SRT and VTT formatters.

These tests ensure that SRT uses a comma as the milliseconds separator
(HH:MM:SS,mmm) while VTT uses a dot (HH:MM:SS.mmm).
"""

from __future__ import annotations

from insanely_fast_whisper_rocm.core.formatters import SrtFormatter, VttFormatter


def _sample_result() -> dict:
    # Minimal, deterministic payload with one cue
    return {
        "text": "Hello",
        "chunks": [
            {"text": "Hello", "timestamp": [0.0, 1.234]},
        ],
        "runtime_seconds": 0.01,
        "config_used": {},
    }


def test_srt_uses_comma_separator() -> None:
    """Ensure SRT timestamps use a comma separator.

    Verifies that the SRT formatter emits timestamps in the form
    ``HH:MM:SS,mmm`` as required by the SRT specification.
    """
    result = _sample_result()
    srt = SrtFormatter.format(result)
    # Expect a comma between seconds and milliseconds
    assert "00:00:00,000 --> 00:00:01,234" in srt


def test_vtt_uses_dot_separator() -> None:
    """Ensure VTT timestamps use a dot separator.

    Verifies that the VTT formatter emits timestamps in the form
    ``HH:MM:SS.mmm`` as required by the WebVTT specification.
    """
    result = _sample_result()
    vtt = VttFormatter.format(result)
    # Expect a dot between seconds and milliseconds
    assert "00:00:00.000 --> 00:00:01.234" in vtt
