"""Tests for insanely_fast_whisper_api.webui.merge_handler module.

This module contains tests for merging multiple transcription results into
single files with format-specific implementations.
"""

from __future__ import annotations

import pytest

from insanely_fast_whisper_api.webui.merge_handler import (
    MERGE_HANDLERS,
    MergeConfiguration,
    MergeResult,
    SrtMerger,
    TxtMerger,
    VttMerger,
    get_merge_handler,
    merge_files,
)


class TestMergeConfiguration:
    """Test suite for MergeConfiguration dataclass."""

    def test_merge_configuration__default_values(self) -> None:
        """Test that MergeConfiguration has correct default values."""
        config = MergeConfiguration()

        assert config.include_file_headers is True
        assert config.include_section_separators is True
        assert config.header_style == "equals"
        assert config.section_separator == "\n\n"

    def test_merge_configuration__custom_values(self) -> None:
        """Test that MergeConfiguration accepts custom values."""
        config = MergeConfiguration(
            include_file_headers=False,
            include_section_separators=False,
            header_style="hash",
            section_separator="\n---\n",
        )

        assert config.include_file_headers is False
        assert config.include_section_separators is False
        assert config.header_style == "hash"
        assert config.section_separator == "\n---\n"


class TestMergeResult:
    """Test suite for MergeResult dataclass."""

    def test_merge_result__success_case(self) -> None:
        """Test MergeResult for successful merge."""
        result = MergeResult(
            success=True,
            merged_content="test content",
            source_files=["file1.txt", "file2.txt"],
            format_type="txt",
            merge_stats={"files_merged": 2},
        )

        assert result.success is True
        assert result.merged_content == "test content"
        assert result.source_files == ["file1.txt", "file2.txt"]
        assert result.format_type == "txt"
        assert result.merge_stats["files_merged"] == 2
        assert result.error_message is None
        assert result.warnings == []

    def test_merge_result__with_warnings(self) -> None:
        """Test MergeResult with warnings."""
        result = MergeResult(
            success=True,
            merged_content="test",
            source_files=["file1.txt"],
            format_type="txt",
            merge_stats={},
            warnings=["Warning 1", "Warning 2"],
        )

        assert len(result.warnings) == 2
        assert "Warning 1" in result.warnings

    def test_merge_result__failure_case(self) -> None:
        """Test MergeResult for failed merge."""
        result = MergeResult(
            success=False,
            merged_content="",
            source_files=["file1.txt"],
            format_type="txt",
            merge_stats={},
            error_message="Test error",
        )

        assert result.success is False
        assert result.error_message == "Test error"


class TestTxtMerger:
    """Test suite for TxtMerger class."""

    def test_txt_merger__merge_single_file(self) -> None:
        """Test merging a single TXT file."""
        merger = TxtMerger()
        file_results = {"test1.txt": {"text": "Hello world"}}

        result = merger.merge_files(file_results)

        assert result.success is True
        assert "Hello world" in result.merged_content
        assert result.format_type == "txt"
        assert result.merge_stats["files_merged"] == 1

    def test_txt_merger__merge_multiple_files(self) -> None:
        """Test merging multiple TXT files."""
        merger = TxtMerger()
        file_results = {
            "test1.txt": {"text": "First file"},
            "test2.txt": {"text": "Second file"},
        }

        result = merger.merge_files(file_results)

        assert result.success is True
        assert "First file" in result.merged_content
        assert "Second file" in result.merged_content
        assert result.merge_stats["files_merged"] == 2

    def test_txt_merger__includes_headers(self) -> None:
        """Test that TXT merger includes file headers."""
        merger = TxtMerger()
        file_results = {
            "test1.txt": {"text": "Content 1"},
            "test2.txt": {"text": "Content 2"},
        }

        result = merger.merge_files(file_results)

        assert "test1.txt" in result.merged_content
        assert "test2.txt" in result.merged_content

    def test_txt_merger__no_headers_when_disabled(self) -> None:
        """Test that headers are excluded when disabled in config."""
        config = MergeConfiguration(include_file_headers=False)
        merger = TxtMerger(config)
        file_results = {"test1.txt": {"text": "Content"}}

        result = merger.merge_files(file_results)

        assert result.success is True
        assert "test1.txt" not in result.merged_content

    def test_txt_merger__empty_text_filtered(self) -> None:
        """Test that files with empty text are filtered out."""
        merger = TxtMerger()
        file_results = {
            "test1.txt": {"text": "Valid content"},
            "test2.txt": {"text": ""},
            "test3.txt": {"text": "   "},
        }

        result = merger.merge_files(file_results)

        assert result.success is True
        assert result.merge_stats["files_merged"] == 1
        assert len(result.warnings) == 2

    def test_txt_merger__get_format_name(self) -> None:
        """Test that get_format_name returns 'txt'."""
        merger = TxtMerger()

        assert merger.get_format_name() == "txt"

    def test_txt_merger__header_styles(self) -> None:
        """Test different header styles."""
        # Test equals style
        config_equals = MergeConfiguration(header_style="equals")
        merger_equals = TxtMerger(config_equals)
        result_equals = merger_equals.merge_files({"test.txt": {"text": "Content"}})
        assert "===" in result_equals.merged_content

        # Test hash style
        config_hash = MergeConfiguration(header_style="hash")
        merger_hash = TxtMerger(config_hash)
        result_hash = merger_hash.merge_files({"test.txt": {"text": "Content"}})
        assert "# test.txt" in result_hash.merged_content

        # Test dashes style
        config_dashes = MergeConfiguration(header_style="dashes")
        merger_dashes = TxtMerger(config_dashes)
        result_dashes = merger_dashes.merge_files({"test.txt": {"text": "Content"}})
        assert "---" in result_dashes.merged_content


class TestSrtMerger:
    """Test suite for SrtMerger class."""

    def test_srt_merger__merge_single_file(self) -> None:
        """Test merging a single SRT file."""
        merger = SrtMerger()
        file_results = {
            "test1.srt": {
                "chunks": [
                    {"timestamp": [0.0, 1.0], "text": "First subtitle"},
                ]
            }
        }

        result = merger.merge_files(file_results)

        assert result.success is True
        assert result.format_type == "srt"

    def test_srt_merger__renumbers_entries(self) -> None:
        """Test that SRT entries are renumbered sequentially."""
        merger = SrtMerger()
        file_results = {
            "test1.srt": {
                "chunks": [
                    {"timestamp": [0.0, 1.0], "text": "Subtitle 1"},
                ]
            },
            "test2.srt": {
                "chunks": [
                    {"timestamp": [0.0, 1.0], "text": "Subtitle 2"},
                ]
            },
        }

        result = merger.merge_files(file_results)

        assert result.success is True
        # Check that entries are renumbered (1, 2...)
        assert "1\n" in result.merged_content
        assert "2\n" in result.merged_content
        assert result.merge_stats["files_merged"] == 2

    def test_srt_merger__handles_segments(self) -> None:
        """Test that SRT merger handles segments as well as chunks."""
        merger = SrtMerger()
        file_results = {
            "test1.srt": {
                "segments": [
                    {"start": 0.0, "end": 1.0, "text": "Segment 1"},
                ]
            }
        }

        result = merger.merge_files(file_results)

        assert result.success is True

    def test_srt_merger__get_format_name(self) -> None:
        """Test that get_format_name returns 'srt'."""
        merger = SrtMerger()

        assert merger.get_format_name() == "srt"

    def test_srt_merger__no_vtt_header(self) -> None:
        """Test that SRT merger doesn't add WEBVTT header."""
        merger = SrtMerger()
        file_results = {
            "test1.srt": {"chunks": [{"timestamp": [0.0, 1.0], "text": "Test"}]}
        }

        result = merger.merge_files(file_results)

        assert "WEBVTT" not in result.merged_content


class TestVttMerger:
    """Test suite for VttMerger class."""

    def test_vtt_merger__merge_single_file(self) -> None:
        """Test merging a single VTT file."""
        merger = VttMerger()
        file_results = {
            "test1.vtt": {
                "chunks": [
                    {"timestamp": [0.0, 1.0], "text": "First subtitle"},
                ]
            }
        }

        result = merger.merge_files(file_results)

        assert result.success is True
        assert result.format_type == "vtt"

    def test_vtt_merger__adds_webvtt_header(self) -> None:
        """Test that VTT merger adds WEBVTT header."""
        merger = VttMerger()
        file_results = {
            "test1.vtt": {"chunks": [{"timestamp": [0.0, 1.0], "text": "Test"}]}
        }

        result = merger.merge_files(file_results)

        assert result.merged_content.startswith("WEBVTT\n")

    def test_vtt_merger__empty_content_has_header(self) -> None:
        """Test that empty VTT still has WEBVTT header."""
        merger = VttMerger()
        file_results = {}

        result = merger.merge_files(file_results)

        # Should fail with no valid files but if we pass invalid data
        # it should handle gracefully
        assert "WEBVTT" in result.merged_content or result.success is False

    def test_vtt_merger__get_format_name(self) -> None:
        """Test that get_format_name returns 'vtt'."""
        merger = VttMerger()

        assert merger.get_format_name() == "vtt"


class TestMergeHandlerRegistry:
    """Test suite for MERGE_HANDLERS registry."""

    def test_merge_handlers__contains_all_formats(self) -> None:
        """Test that MERGE_HANDLERS contains all supported formats."""
        assert "txt" in MERGE_HANDLERS
        assert "srt" in MERGE_HANDLERS
        assert "vtt" in MERGE_HANDLERS

    def test_merge_handlers__classes_are_correct(self) -> None:
        """Test that MERGE_HANDLERS maps to correct classes."""
        assert MERGE_HANDLERS["txt"] is TxtMerger
        assert MERGE_HANDLERS["srt"] is SrtMerger
        assert MERGE_HANDLERS["vtt"] is VttMerger


class TestGetMergeHandler:
    """Test suite for get_merge_handler function."""

    def test_get_merge_handler__returns_txt_merger(self) -> None:
        """Test that get_merge_handler returns TxtMerger for 'txt'."""
        handler = get_merge_handler("txt")

        assert isinstance(handler, TxtMerger)

    def test_get_merge_handler__returns_srt_merger(self) -> None:
        """Test that get_merge_handler returns SrtMerger for 'srt'."""
        handler = get_merge_handler("srt")

        assert isinstance(handler, SrtMerger)

    def test_get_merge_handler__returns_vtt_merger(self) -> None:
        """Test that get_merge_handler returns VttMerger for 'vtt'."""
        handler = get_merge_handler("vtt")

        assert isinstance(handler, VttMerger)

    def test_get_merge_handler__raises_for_unsupported_format(self) -> None:
        """Test that get_merge_handler raises ValueError for unsupported format."""
        with pytest.raises(ValueError, match="Unsupported format"):
            get_merge_handler("pdf")

    def test_get_merge_handler__with_custom_config(self) -> None:
        """Test that get_merge_handler applies custom configuration."""
        config = MergeConfiguration(include_file_headers=False)

        handler = get_merge_handler("txt", config)

        assert handler.config.include_file_headers is False


class TestMergeFilesFunction:
    """Test suite for merge_files convenience function."""

    def test_merge_files__txt_format(self) -> None:
        """Test merge_files with txt format."""
        file_results = {"test.txt": {"text": "Test content"}}

        result = merge_files(file_results, "txt")

        assert result.success is True
        assert result.format_type == "txt"

    def test_merge_files__srt_format(self) -> None:
        """Test merge_files with srt format."""
        file_results = {
            "test.srt": {"chunks": [{"timestamp": [0.0, 1.0], "text": "Test"}]}
        }

        result = merge_files(file_results, "srt")

        assert result.success is True
        assert result.format_type == "srt"

    def test_merge_files__vtt_format(self) -> None:
        """Test merge_files with vtt format."""
        file_results = {
            "test.vtt": {"chunks": [{"timestamp": [0.0, 1.0], "text": "Test"}]}
        }

        result = merge_files(file_results, "vtt")

        assert result.success is True
        assert result.format_type == "vtt"


class TestMergeHandlerErrorHandling:
    """Test suite for error handling in merge operations."""

    def test_merge_handler__handles_invalid_file_data(self) -> None:
        """Test that merge handler gracefully handles invalid file data."""
        merger = TxtMerger()
        file_results = {
            "test1.txt": {"text": "Valid"},
            "test2.txt": None,  # Invalid
            "test3.txt": {},  # Missing text key
        }

        result = merger.merge_files(file_results)

        # Should succeed with only the valid file
        assert result.success is True
        assert result.merge_stats["files_merged"] == 1
        assert len(result.warnings) == 2

    def test_merge_handler__handles_exception_in_formatting(self) -> None:
        """Test that merge handler handles exceptions during formatting."""
        merger = TxtMerger()
        # Provide data that might cause formatting issues
        file_results = {"test.txt": {"wrong_key": "value"}}

        result = merger.merge_files(file_results)

        # Should handle the error gracefully
        assert result.success is False or len(result.warnings) > 0

    def test_merge_handler__empty_file_results(self) -> None:
        """Test that merge handler handles empty file results."""
        merger = TxtMerger()
        file_results = {}

        result = merger.merge_files(file_results)

        assert result.success is False
        assert result.error_message == "No valid files"

    def test_merge_handler__all_files_invalid(self) -> None:
        """Test that merge handler handles all files being invalid."""
        merger = TxtMerger()
        file_results = {
            "test1.txt": {"text": ""},
            "test2.txt": {"text": "   "},
        }

        result = merger.merge_files(file_results)

        assert result.success is False
        assert result.error_message == "No valid files"


class TestSectionOrdering:
    """Test suite for deterministic section ordering."""

    def test_txt_merger__sorts_files_alphabetically(self) -> None:
        """Test that files are merged in alphabetical order."""
        merger = TxtMerger()
        file_results = {
            "zebra.txt": {"text": "Z"},
            "alpha.txt": {"text": "A"},
            "beta.txt": {"text": "B"},
        }

        result = merger.merge_files(file_results)

        # Check that content appears in alphabetical order
        content = result.merged_content
        alpha_pos = content.find("alpha.txt")
        beta_pos = content.find("beta.txt")
        zebra_pos = content.find("zebra.txt")

        assert alpha_pos < beta_pos < zebra_pos


class TestCustomSeparators:
    """Test suite for custom section separators."""

    def test_txt_merger__custom_separator(self) -> None:
        """Test using a custom section separator."""
        config = MergeConfiguration(section_separator="\n---BREAK---\n")
        merger = TxtMerger(config)
        file_results = {
            "test1.txt": {"text": "First"},
            "test2.txt": {"text": "Second"},
        }

        result = merger.merge_files(file_results)

        assert "---BREAK---" in result.merged_content

    def test_txt_merger__no_separators(self) -> None:
        """Test disabling section separators."""
        config = MergeConfiguration(include_section_separators=False)
        merger = TxtMerger(config)
        file_results = {
            "test1.txt": {"text": "First"},
            "test2.txt": {"text": "Second"},
        }

        result = merger.merge_files(file_results)

        # Should still have content but without extra separators
        assert result.success is True
        assert "First" in result.merged_content
        assert "Second" in result.merged_content
