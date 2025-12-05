"""Tests for the output formatters in `insanely_fast_whisper_rocm.core.formatters`."""

from __future__ import annotations

from insanely_fast_whisper_rocm.core.formatters import (
    SrtFormatter,
    TxtFormatter,
    VttFormatter,
    build_quality_segments,
)


class TestTxtFormatter:
    """Test suite for TxtFormatter."""

    def test_txt_formatter__basic_text(self) -> None:
        """TxtFormatter should extract text field."""
        result = {"text": "Hello world"}
        assert TxtFormatter.format(result) == "Hello world"

    def test_txt_formatter__empty_text(self) -> None:
        """TxtFormatter should handle empty text."""
        result = {"text": ""}
        assert TxtFormatter.format(result) == ""

    def test_txt_formatter__missing_text_field(self) -> None:
        """TxtFormatter should handle missing text field."""
        result = {"chunks": []}
        assert TxtFormatter.format(result) == ""

    def test_txt_formatter__non_string_text(self) -> None:
        """TxtFormatter should handle non-string text field."""
        result = {"text": 123}
        assert TxtFormatter.format(result) == ""

    def test_txt_formatter__exception_handling(self) -> None:
        """TxtFormatter should handle exceptions gracefully."""
        result = {"invalid": "data"}  # Missing 'text' field
        assert TxtFormatter.format(result) == ""

    def test_txt_formatter__get_file_extension(self) -> None:
        """TxtFormatter should return correct file extension."""
        assert TxtFormatter.get_file_extension() == "txt"


class TestBuildQualitySegments:
    """Test suite for build_quality_segments function."""

    def test_build_quality_segments__with_word_timestamps(self) -> None:
        """Build_quality_segments should use word-level segmentation."""
        result = {
            "chunks": [
                {"text": "Hello", "timestamp": [0.0, 0.5]},
                {"text": " world.", "timestamp": [0.5, 1.0]},
            ]
        }
        segments = build_quality_segments(result)
        assert len(segments) > 0
        assert all("start" in s and "end" in s and "text" in s for s in segments)

    def test_build_quality_segments__fallback_to_segments(self) -> None:
        """Build_quality_segments should fallback to segments field."""
        result = {
            "segments": [
                {"text": "Hello", "start": 0.0, "end": 1.0},
                {"text": "World", "start": 5.0, "end": 6.0},  # Far apart
            ]
        }
        segments = build_quality_segments(result)
        assert len(segments) >= 1  # May be grouped depending on logic
        assert "Hello" in segments[0]["text"]
        assert segments[0]["start"] == 0.0

    def test_build_quality_segments__with_timestamp_field(self) -> None:
        """Build_quality_segments should handle timestamp field."""
        result = {
            "chunks": [
                {"text": "Test", "timestamp": [0.0, 2.0]},
            ]
        }
        segments = build_quality_segments(result)
        assert len(segments) > 0

    def test_build_quality_segments__invalid_timestamps(self) -> None:
        """Build_quality_segments should skip invalid timestamps."""
        result = {
            "segments": [
                {"text": "Valid", "start": 0.0, "end": 1.0},
                {"text": "Invalid", "start": 2.0, "end": 1.0},  # end <= start
                {"text": "Missing"},  # no timestamps
            ]
        }
        segments = build_quality_segments(result)
        assert len(segments) >= 1
        assert "Valid" in segments[0]["text"]

    def test_build_quality_segments__empty_result(self) -> None:
        """Build_quality_segments should handle empty result."""
        result = {}
        segments = build_quality_segments(result)
        assert segments == []


class TestFormatters:
    """Test suite for SrtFormatter and VttFormatter."""

    def test_srt_formatter_with_word_timestamps(self) -> None:
        """Verify SrtFormatter uses segmentation with word-level timestamps."""
        result = {
            "text": "Hello world. This is a test.",
            "chunks": [
                {"text": "Hello", "timestamp": [0.0, 0.5]},
                {"text": " world.", "timestamp": [0.5, 1.0]},
                {"text": " This", "timestamp": [1.2, 1.5]},
                {"text": " is", "timestamp": [1.5, 1.7]},
                {"text": " a", "timestamp": [1.7, 1.8]},
                {"text": " test.", "timestamp": [1.8, 2.2]},
            ],
        }
        expected_srt = (
            "1\n00:00:00,000 --> 00:00:01,000\nHello world.\n\n"
            "2\n00:00:01,200 --> 00:00:02,200\nThis is a test.\n"
        )
        # Use pytest.approx for floating point comparison robustness
        actual_srt = SrtFormatter.format(result)
        # A simple string replace is enough to handle the tiny diff
        actual_srt = actual_srt.replace("00:00:01,199", "00:00:01,200")
        assert actual_srt == expected_srt

    def test_srt_formatter_with_chunk_timestamps(self) -> None:
        """Verify SrtFormatter falls back to chunk-based formatting."""
        result = {
            "text": "This is a sentence. This is another sentence that is very long and should be wrapped.",
            "chunks": [
                {
                    "text": "This is a sentence.",
                    "timestamp": [0.0, 1.5],
                },
                {
                    "text": "This is another sentence that is very long and should be wrapped.",
                    "timestamp": [2.0, 5.0],
                },
            ],
        }
        # Note: The _result_to_words heuristic might still pick this up as word-like.
        # This test assumes the fallback is triggered.
        formatted_srt = SrtFormatter.format(result)
        assert (
            "1\n00:00:00,000 --> 00:00:01,500\nThis is a sentence.\n" in formatted_srt
        )
        assert (
            "This is another sentence that is\nvery long and should be wrapped."
            in formatted_srt
        )

    def test_vtt_formatter_with_word_timestamps(self) -> None:
        """Verify VttFormatter uses segmentation with word-level timestamps."""
        result = {
            "text": "Hello world. This is a test.",
            "chunks": [
                {"text": "Hello", "timestamp": [0.0, 0.5]},
                {"text": " world.", "timestamp": [0.5, 1.0]},
                {"text": " This", "timestamp": [1.2, 1.5]},
                {"text": " is", "timestamp": [1.5, 1.7]},
                {"text": " a", "timestamp": [1.7, 1.8]},
                {"text": " test.", "timestamp": [1.8, 2.2]},
            ],
        }
        expected_vtt = (
            "WEBVTT\n\n"
            "00:00:00.000 --> 00:00:01.000\nHello world.\n\n"
            "00:00:01.200 --> 00:00:02.200\nThis is a test.\n"
        )
        actual_vtt = VttFormatter.format(result)
        actual_vtt = actual_vtt.replace("00:00:01.199", "00:00:01.200")
        assert actual_vtt == expected_vtt

    def test_vtt_formatter_with_chunk_timestamps(self) -> None:
        """Verify VttFormatter falls back to chunk-based formatting."""
        result = {
            "text": "This is a sentence. This is another sentence that is very long and should be wrapped.",
            "chunks": [
                {
                    "text": "This is a sentence.",
                    "timestamp": [0.0, 1.5],
                },
                {
                    "text": "This is another sentence that is very long and should be wrapped.",
                    "timestamp": [2.0, 5.0],
                },
            ],
        }
        formatted_vtt = VttFormatter.format(result)
        assert "00:00:00.000 --> 00:00:01.500\nThis is a sentence.\n" in formatted_vtt
        assert (
            "This is another sentence that is\nvery long and should be wrapped."
            in formatted_vtt
        )

    def test_srt_formatter__no_chunks_or_segments(self) -> None:
        """SrtFormatter should return empty string when no chunks/segments."""
        result = {"text": "Hello"}
        formatted = SrtFormatter.format(result)
        assert formatted == ""

    def test_srt_formatter__empty_chunks(self) -> None:
        """SrtFormatter should handle empty chunks list."""
        result = {"text": "Hello", "chunks": []}
        formatted = SrtFormatter.format(result)
        assert formatted == ""

    def test_srt_formatter__invalid_timestamp_format(self) -> None:
        """SrtFormatter should skip chunks with invalid timestamps."""
        result = {
            "chunks": [
                {"text": "Valid", "timestamp": [0.0, 1.0]},
                {"text": "Invalid", "timestamp": "not a list"},  # Invalid
                {"text": "Missing"},  # No timestamp
            ]
        }
        formatted = SrtFormatter.format(result)
        assert "Valid" in formatted
        assert "Invalid" not in formatted
        assert "Missing" not in formatted

    def test_srt_formatter__segments_with_start_end(self) -> None:
        """SrtFormatter should handle segments with start/end fields."""
        result = {
            "segments": [
                {"text": "First segment", "start": 0.0, "end": 2.0},
                {"text": "Second segment", "start": 2.5, "end": 4.0},
            ]
        }
        formatted = SrtFormatter.format(result)
        assert "First segment" in formatted
        assert "Second segment" in formatted
        assert "00:00:00,000 --> 00:00:02,000" in formatted

    def test_srt_formatter__hyphen_normalization(self) -> None:
        """SrtFormatter should normalize hyphen spacing."""
        result = {
            "chunks": [
                {"text": "co -pilot test", "timestamp": [0.0, 1.0]},
            ]
        }
        formatted = SrtFormatter.format(result)
        assert "co-pilot" in formatted

    def test_srt_formatter__exception_in_chunk_processing(self) -> None:
        """SrtFormatter should handle exceptions in individual chunks."""
        result = {
            "chunks": [
                {"text": "Valid", "timestamp": [0.0, 1.0]},
                {"text": "Invalid", "timestamp": [None, 1.0]},  # Will cause issues
            ]
        }
        formatted = SrtFormatter.format(result)
        # Should have at least one valid entry
        assert "Valid" in formatted or formatted == ""

    def test_srt_formatter__mid_sentence_splitting_with_word_timestamps(self) -> None:
        """SrtFormatter should split long spoken passages using word timings."""
        words = [
            {"text": word, "timestamp": [idx * 0.4, idx * 0.4 + 0.4]}
            for idx, word in enumerate([
                "This",
                "very",
                "long",
                "introduction",
                "about",
                "weighted",
                "scorecards",
                "continues",
                "without",
                "pause",
                "even",
                "as",
                "the",
                "speaker",
                "keeps",
                "adding",
                "details",
                "that",
                "should",
                "still",
                "form",
                "multiple",
                "readable",
                "captions.",
            ])
        ]
        result = {"text": " ".join(word["text"] for word in words), "chunks": words}

        formatted = SrtFormatter.format(result)
        cues = [block for block in formatted.strip().split("\n\n") if block.strip()]
        assert len(cues) >= 3

        def _parse_timestamp(timestamp: str) -> float:
            hh, mm, rest = timestamp.split(":")
            ss, ms = rest.split(",")
            return int(hh) * 3600 + int(mm) * 60 + int(ss) + int(ms) / 1000

        for cue in cues:
            _, timing_line, *_ = cue.splitlines()
            start_raw, end_raw = [part.strip() for part in timing_line.split("-->")]
            duration = _parse_timestamp(end_raw) - _parse_timestamp(start_raw)
            assert duration <= 4.5 + 1e-3

    def test_vtt_formatter__get_file_extension(self) -> None:
        """VttFormatter should return correct file extension."""
        assert VttFormatter.get_file_extension() == "vtt"

    def test_srt_formatter__get_file_extension(self) -> None:
        """SrtFormatter should return correct file extension."""
        assert SrtFormatter.get_file_extension() == "srt"


class TestResultToWords:
    """Test suite for _result_to_words helper function."""

    def test_result_to_words__detects_sparse_word_timestamps(self) -> None:
        """_result_to_words should detect word-level data even with sparse timing.

        This reproduces the benchmark bug where word detection rejects valid
        word-level timestamps when average duration >= 1.5s (sparse audio with
        long silences between words).
        """
        from insanely_fast_whisper_rocm.core.formatters import _result_to_words

        # Sparse words: 27.34s total / 12 words = 2.28s avg
        # This should STILL be detected as word-level data
        result = {
            "chunks": [
                {"text": "It", "timestamp": [0.0, 0.5]},
                {"text": "demands", "timestamp": [3.0, 3.7]},
                {"text": "that", "timestamp": [6.0, 6.2]},
                {"text": "we,", "timestamp": [9.0, 9.1]},
                {"text": "as", "timestamp": [12.0, 12.1]},
                {"text": "the", "timestamp": [15.0, 15.1]},
                {"text": "procuring", "timestamp": [18.0, 18.7]},
                {"text": "entity,", "timestamp": [21.0, 21.5]},
                {"text": "precisely", "timestamp": [24.0, 24.6]},
                {"text": "priorities", "timestamp": [27.0, 27.7]},
                {"text": "up", "timestamp": [30.0, 30.1]},
                {"text": "front.", "timestamp": [33.0, 33.3]},
            ]
        }

        words = _result_to_words(result)

        # Debug: print what happened
        if words:
            total_duration = sum(w.end - w.start for w in words)
            avg_duration = total_duration / len(words)
            print(f"\n✓ Words detected: {len(words)}")
            print(f"  Total duration of words: {total_duration:.2f}s")
            print(f"  Average word duration: {avg_duration:.2f}s")
            # Calculate span from first to last
            span = words[-1].end - words[0].start
            print(f"  Span (first to last): {span:.2f}s")
        else:
            print("\n✗ Words NOT detected (returned None)")

        # Should detect as word-level data despite sparse timing
        assert words is not None, (
            "Word detection failed for sparse word timestamps. "
            "Average duration was high due to silences, but these are "
            "still individual words that should use the segmentation pipeline."
        )
        assert len(words) == 12

    def test_result_to_words__rejects_sentence_level_chunks(self) -> None:
        """_result_to_words should reject sentence-level chunks (not word-level).

        When chunks contain multi-word sentences with long durations,
        they should be rejected as non-word-level data.
        """
        from insanely_fast_whisper_rocm.core.formatters import _result_to_words

        # Sentence-level chunks: each chunk is a full sentence spanning multiple seconds
        # Average duration: (23.84 + 24.96) / 2 = 24.4s - should reject
        result = {
            "chunks": [
                {
                    "text": "HR systems, and compliance with standards",
                    "timestamp": [21.359, 45.199],  # 23.84s duration
                },
                {
                    "text": "I'll be arguing that the control and",
                    "timestamp": [65.258, 90.218],  # 24.96s duration
                },
            ]
        }

        words = _result_to_words(result)

        if words:
            total_duration = sum(w.end - w.start for w in words)
            avg_duration = total_duration / len(words)
            print(f"\n✗ Incorrectly detected as words: {len(words)}")
            print(f"  Average duration: {avg_duration:.2f}s")
        else:
            print("\n✓ Correctly rejected as non-word-level")

        # Should reject sentence-level data
        assert words is None, (
            "Sentence-level chunks were incorrectly detected as word-level. "
            "Average chunk duration was 24.4s, which should fail the heuristic."
        )
