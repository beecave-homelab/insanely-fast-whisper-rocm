"""Unit tests for SRT quality scoring.

These tests define the expected public contract for a future utility
`compute_srt_quality(segments, srt_text)` that returns a dictionary with a
numeric `score` in [0, 1] and a `details` mapping containing sub-metrics.
"""

from __future__ import annotations

from typing import Any

from insanely_fast_whisper_api.core.formatters import (
    SrtFormatter,
    build_quality_segments,
)
from insanely_fast_whisper_api.utils.srt_quality import compute_srt_quality


def _fake_segments_ok() -> list[dict[str, Any]]:
    # Two non-overlapping segments with reasonable durations and text lengths
    return [
        {"start": 0.0, "end": 2.0, "text": "Hello world."},
        {"start": 2.1, "end": 4.5, "text": "This is a test."},
    ]


def _fake_segments_with_overlap() -> list[dict[str, Any]]:
    # Overlap between first and second
    return [
        {"start": 0.0, "end": 2.5, "text": "Hello world."},
        {"start": 2.4, "end": 3.5, "text": "Overlap here."},
    ]


class TestSrtQuality:
    """Contract tests for the SRT quality scoring helper.

    Notes:
        The concrete scoring implementation can evolve; these tests assert the
        presence of a `score` key and some intuitive sub-metric behaviors.
    """

    def test_quality_ok_segments_high_score(self) -> None:
        """Well-formed segments and tidy SRT lines should yield high score."""
        segments = _fake_segments_ok()
        srt_text = (
            "1\n00:00:00,000 --> 00:00:02,000\nHello world.\n\n"
            "2\n00:00:02,100 --> 00:00:04,500\nThis is a test.\n"
        )

        # Intentionally import lazily to define the contract first
        quality = compute_srt_quality(segments=segments, srt_text=srt_text)
        assert isinstance(quality, dict)
        assert "score" in quality and isinstance(quality["score"], float)
        assert 0.0 <= quality["score"] <= 1.0
        assert quality["score"] >= 0.8  # Good output should be high
        assert isinstance(quality.get("details", {}), dict)

    def test_quality_penalizes_overlap_and_hyphen_spacing(self) -> None:
        """Overlaps and bad hyphen spacing should reduce the score."""
        segments = _fake_segments_with_overlap()
        # Include an intentional "co -pilot" spacing issue
        srt_text = (
            "1\n00:00:00,000 --> 00:00:02,500\nHello world.\n\n"
            "2\n00:00:02,400 --> 00:00:03,500\nWe look at co -pilot.\n"
        )

        quality = compute_srt_quality(segments=segments, srt_text=srt_text)
        assert 0.0 <= quality["score"] <= 1.0
        # Expect a noticeable penalty vs. the OK case
        assert quality["score"] <= 0.7
        details = quality.get("details", {})
        # Overlap should be detected (>0)
        assert details.get("overlap_violations", 0) >= 1
        # Hyphen normalization should fail
        assert details.get("hyphen_normalization_ok") is False

    def test_quality_word_level_chunks_improves_cps_ratio(self) -> None:
        """Ensure readable segments yield CPS ratio above zero."""
        chunks = [
            {"text": "Robust", "timestamp": [0.0, 0.15]},
            {"text": "weighted", "timestamp": [0.15, 0.35]},
            {"text": "scorecards", "timestamp": [0.35, 0.55]},
            {"text": "drive", "timestamp": [0.55, 0.75]},
            {"text": "clarity.", "timestamp": [0.75, 2.75]},
            {"text": "They", "timestamp": [3.0, 3.15]},
            {"text": "balance", "timestamp": [3.15, 3.35]},
            {"text": "risk", "timestamp": [3.35, 3.5]},
            {"text": "and", "timestamp": [3.5, 3.65]},
            {"text": "value.", "timestamp": [3.65, 4.8]},
        ]
        text = " ".join(chunk["text"] for chunk in chunks)
        result = {"text": text, "chunks": chunks}

        srt_text = SrtFormatter.format(result)

        raw_quality = compute_srt_quality(segments=chunks, srt_text=srt_text)
        assert raw_quality["details"]["cps_within_range_ratio"] == 0.0

        readable_segments = build_quality_segments(result)
        improved_quality = compute_srt_quality(
            segments=readable_segments,
            srt_text=srt_text,
        )
        assert improved_quality["details"]["cps_within_range_ratio"] > 0.0
