"""Unit tests for SRT quality scoring.

These tests define the expected public contract for a future utility
`compute_srt_quality(segments, srt_text)` that returns a dictionary with a
numeric `score` in [0, 1] and a `details` mapping containing sub-metrics.
"""

from __future__ import annotations

from typing import Any

import pytest

from insanely_fast_whisper_rocm.core.formatters import (
    SrtFormatter,
    build_quality_segments,
)
from insanely_fast_whisper_rocm.utils.srt_quality import compute_srt_quality


def _fake_segments_ok() -> list[dict[str, Any]]:
    # Two non-overlapping segments with reasonable durations and text lengths
    return [
        {
            "start": 0.0,
            "end": 2.0,
            "text": "Thoughtful metrics drive better clarity.",
        },
        {
            "start": 2.1,
            "end": 4.5,
            "text": "Insightful metrics keep decisions steady.",
        },
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
            "1\n00:00:00,000 --> 00:00:02,000\n"
            f"{segments[0]['text']}\n\n"
            "2\n00:00:02,100 --> 00:00:04,500\n"
            f"{segments[1]['text']}\n"
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

    def test_quality_details_include_duration_stats_and_histogram(self) -> None:
        """Details should expose duration stats, CPS histogram, and boundary counts."""
        segments = _fake_segments_ok()
        srt_text = (
            "1\n00:00:00,000 --> 00:00:02,000\n"
            f"{segments[0]['text']}\n\n"
            "2\n00:00:02,100 --> 00:00:04,500\n"
            f"{segments[1]['text']}\n"
        )

        quality = compute_srt_quality(segments=segments, srt_text=srt_text)
        details = quality["details"]

        duration_stats = details["duration_stats"]
        assert duration_stats["min_seconds"] == pytest.approx(2.0)
        assert duration_stats["max_seconds"] == pytest.approx(2.4)
        assert duration_stats["average_seconds"] == pytest.approx(2.2, rel=0.05)
        assert duration_stats["median_seconds"] == pytest.approx(2.2, rel=0.05)

        cps_histogram = details["cps_histogram"]
        assert set(cps_histogram) == {
            "below_min",
            "within_range",
            "above_max",
            "total",
        }
        assert cps_histogram["total"] == len(segments)

        boundary_counts = details["boundary_counts"]
        assert boundary_counts["within_range"] == len(segments)
        assert boundary_counts["too_short"] == 0
        assert boundary_counts["too_long"] == 0

        sample_offenders = details["sample_offenders"]
        assert sample_offenders["line_length"] == []
        assert sample_offenders["cps"] == []

    def test_quality_sample_offenders_capture_long_lines_and_cps(self) -> None:
        """Sample offenders list lines exceeding limits and CPS outliers."""
        long_line = "A" * 60
        segments = [
            {"start": 0.0, "end": 0.5, "text": "Hi."},
            {"start": 0.5, "end": 5.5, "text": long_line},
        ]
        srt_text = (
            "1\n00:00:00,000 --> 00:00:00,500\nHi.\n\n"
            f"2\n00:00:00,500 --> 00:00:05,500\n{long_line}\n"
        )

        quality = compute_srt_quality(segments=segments, srt_text=srt_text)
        details = quality["details"]

        boundary_counts = details["boundary_counts"]
        assert boundary_counts["too_short"] == 1
        assert boundary_counts["too_long"] == 1

        sample_offenders = details["sample_offenders"]
        assert any(
            long_line in offender["line"]
            for offender in sample_offenders["line_length"]
        )
        assert any(
            offender["category"] != "within_range"
            for offender in sample_offenders["cps"]
        )


class TestSrtQualityDurations:
    """Duration-focused regression tests for `compute_srt_quality()`."""

    def test_quality_penalizes_duration_outliers(self) -> None:
        """Segments outside duration bounds should lower the quality score."""
        long_text_line_1 = "A" * 40
        long_text_line_2 = "B" * 40
        readable_text = "Readable captions stay on pace."
        short_text = "OK"
        long_segment_text = f"{long_text_line_1}{long_text_line_2}"

        segments = [
            {"start": 0.0, "end": 3.0, "text": readable_text},
            {"start": 3.0, "end": 3.2, "text": short_text},
            {"start": 3.2, "end": 13.2, "text": long_segment_text},
        ]

        srt_text = (
            "1\n00:00:00,000 --> 00:00:03,000\n"
            f"{readable_text}\n\n"
            "2\n00:00:03,000 --> 00:00:03,200\n"
            f"{short_text}\n\n"
            "3\n00:00:03,200 --> 00:00:13,200\n"
            f"{long_text_line_1}\n"
            f"{long_text_line_2}\n"
        )

        quality = compute_srt_quality(segments=segments, srt_text=srt_text)
        details = quality["details"]

        assert details["boundary_counts"]["too_short"] == 1
        assert details["boundary_counts"]["too_long"] == 1
        assert quality["score"] <= 0.7
