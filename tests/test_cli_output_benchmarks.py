"""Tests for CLI output and benchmark handling."""

from pathlib import Path
from unittest.mock import ANY, Mock, patch

from insanely_fast_whisper_api.cli.commands import _handle_output_and_benchmarks
from insanely_fast_whisper_api.core.progress import ProgressCallback


class TestOutputAndBenchmarks:
    """Test output file handling and benchmarking functionality."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.audio_file = Path("test.mp3")
        self.result = {
            "text": "Test transcription",
            "chunks": [{"text": "Test transcription", "timestamp": [0.0, 2.0]}],
            "runtime_seconds": 1.5,
            "config_used": {"model": "test"},
        }
        self.total_time = 2.0

    @patch("insanely_fast_whisper_api.cli.commands.FORMATTERS")
    def test_export_json_default(self, mock_formatters: Mock) -> None:
        """Test default JSON export to transcripts directory."""
        mock_formatter = Mock()
        mock_formatter.format.return_value = '{"test": "data"}'
        mock_formatter.get_file_extension.return_value = "json"
        mock_formatters.__getitem__.return_value = mock_formatter

        with patch("pathlib.Path.mkdir"):
            with patch("pathlib.Path.write_text") as mock_write:
                _handle_output_and_benchmarks(
                    task="transcribe",
                    audio_file=self.audio_file,
                    result=self.result,
                    total_time=self.total_time,
                    output=None,
                    export_format="json",
                    export_format_explicit=False,
                    benchmark_enabled=False,
                    benchmark_extra=(),
                    benchmark_flags=None,
                    benchmark_gpu_stats=None,
                    temp_files=[],
                    progress_cb=None,
                    quiet=False,
                )

                # Verify JSON formatter was used
                mock_formatters.__getitem__.assert_called_with("json")
                mock_formatter.format.assert_called_once()
                mock_write.assert_called_once()

    @patch("insanely_fast_whisper_api.cli.commands.FORMATTERS")
    def test_export_multiple_formats_all(self, mock_formatters: Mock) -> None:
        """Test exporting all formats."""
        mock_json_formatter = Mock()
        mock_json_formatter.format.return_value = '{"test": "json"}'
        mock_json_formatter.get_file_extension.return_value = "json"

        mock_txt_formatter = Mock()
        mock_txt_formatter.format.return_value = "Test transcription"
        mock_txt_formatter.get_file_extension.return_value = "txt"

        mock_srt_formatter = Mock()
        mock_srt_formatter.format.return_value = (
            "1\n00:00:00,000 --> 00:00:02,000\nTest transcription"
        )
        mock_srt_formatter.get_file_extension.return_value = "srt"

        def get_formatter(format_name: str) -> Mock:
            if format_name == "json":
                return mock_json_formatter
            elif format_name == "txt":
                return mock_txt_formatter
            elif format_name == "srt":
                return mock_srt_formatter
            return Mock()

        mock_formatters.__getitem__.side_effect = get_formatter

        with patch("pathlib.Path.mkdir"):
            with patch("pathlib.Path.write_text") as mock_write:
                _handle_output_and_benchmarks(
                    task="transcribe",
                    audio_file=self.audio_file,
                    result=self.result,
                    total_time=self.total_time,
                    output=None,
                    export_format="all",
                    export_format_explicit=False,
                    benchmark_enabled=False,
                    benchmark_extra=(),
                    benchmark_flags=None,
                    benchmark_gpu_stats=None,
                    temp_files=[],
                    progress_cb=None,
                    quiet=False,
                )

                # Verify all three formatters were called
                assert mock_formatters.__getitem__.call_count == 3
                assert mock_json_formatter.format.call_count == 1
                assert mock_txt_formatter.format.call_count == 1
                assert mock_srt_formatter.format.call_count == 1
                assert mock_write.call_count == 3

    @patch("insanely_fast_whisper_api.cli.commands.FORMATTERS")
    def test_custom_output_path(self, mock_formatters: Mock) -> None:
        """Test custom output file path."""
        mock_formatter = Mock()
        mock_formatter.format.return_value = "Test content"
        mock_formatter.get_file_extension.return_value = "json"
        mock_formatters.__getitem__.return_value = mock_formatter

        custom_output = Path("custom/output.json")

        with patch("pathlib.Path.mkdir"):
            with patch("pathlib.Path.write_text") as mock_write:
                _handle_output_and_benchmarks(
                    task="transcribe",
                    audio_file=self.audio_file,
                    result=self.result,
                    total_time=self.total_time,
                    output=custom_output,
                    export_format="json",
                    export_format_explicit=True,
                    benchmark_enabled=False,
                    benchmark_extra=(),
                    benchmark_flags=None,
                    benchmark_gpu_stats=None,
                    temp_files=[],
                    progress_cb=None,
                    quiet=False,
                )

                mock_write.assert_called_once()
                args = mock_write.call_args
                assert args[0][0] == "Test content"
                assert args[1]["encoding"] == "utf-8"

    @patch("insanely_fast_whisper_api.benchmarks.collector.BenchmarkCollector")
    def test_benchmark_enabled(self, mock_collector_class: Mock) -> None:
        """Test benchmark collection when enabled."""
        mock_collector = Mock()
        mock_collector.collect.return_value = Path("benchmark.json")
        mock_collector_class.return_value = mock_collector

        with patch("pathlib.Path.mkdir"):
            with patch("pathlib.Path.write_text"):
                with patch("click.secho") as mock_secho:
                    _handle_output_and_benchmarks(
                        task="transcribe",
                        audio_file=self.audio_file,
                        result=self.result,
                        total_time=self.total_time,
                        output=None,
                        export_format="json",
                        export_format_explicit=False,
                        benchmark_enabled=True,
                        benchmark_extra=("key1=value1", "key2=value2"),
                        benchmark_flags=None,
                        benchmark_gpu_stats=None,
                        temp_files=[],
                        progress_cb=None,
                        quiet=False,
                    )

                    # Verify benchmark collector was used
                    mock_collector_class.assert_called_once()
                    mock_collector.collect.assert_called_once()
                    mock_secho.assert_called()

    @patch("insanely_fast_whisper_api.cli.commands.FORMATTERS")
    def test_file_write_error_handling(self, mock_formatters: Mock) -> None:
        """Test handling of file write errors."""
        mock_formatter = Mock()
        mock_formatter.format.return_value = "Test content"
        mock_formatter.get_file_extension.return_value = "json"
        mock_formatters.__getitem__.return_value = mock_formatter

        with patch("pathlib.Path.mkdir"):
            with patch("pathlib.Path.write_text", side_effect=OSError("Disk full")):
                with patch("click.secho") as mock_secho:
                    _handle_output_and_benchmarks(
                        task="transcribe",
                        audio_file=self.audio_file,
                        result=self.result,
                        total_time=self.total_time,
                        output=None,
                        export_format="json",
                        export_format_explicit=False,
                        benchmark_enabled=False,
                        benchmark_extra=(),
                        benchmark_flags=None,
                        benchmark_gpu_stats=None,
                        temp_files=[],
                        progress_cb=None,
                        quiet=False,
                    )

                    # Verify error was reported
                    mock_secho.assert_called()
                    error_call = mock_secho.call_args
                    assert "Failed to save" in error_call[0][0]
                    assert error_call[1]["fg"] == "red"
                    assert error_call[1]["err"] is True

    @patch("insanely_fast_whisper_api.cli.commands.FORMATTERS")
    def test_progress_reporting(self, mock_formatters: Mock) -> None:
        """Test progress reporting during export."""
        mock_formatter = Mock()
        mock_formatter.format.return_value = "Test content"
        mock_formatter.get_file_extension.return_value = "json"
        mock_formatters.__getitem__.return_value = mock_formatter

        mock_progress = Mock(spec=ProgressCallback)

        with patch("pathlib.Path.mkdir"):
            with patch("pathlib.Path.write_text"):
                _handle_output_and_benchmarks(
                    task="transcribe",
                    audio_file=self.audio_file,
                    result=self.result,
                    total_time=self.total_time,
                    output=None,
                    export_format="json",
                    export_format_explicit=False,
                    benchmark_enabled=False,
                    benchmark_extra=(),
                    benchmark_flags=None,
                    benchmark_gpu_stats=None,
                    temp_files=[],
                    progress_cb=mock_progress,
                    quiet=False,
                )

                # Verify progress callbacks were made
                mock_progress.on_export_started.assert_called_once_with(1)
                mock_progress.on_export_item_done.assert_called_once_with(0, ANY)

    @patch("insanely_fast_whisper_api.cli.commands.cleanup_temp_files")
    def test_temp_file_cleanup(self, mock_cleanup: Mock) -> None:
        """Test cleanup of temporary files."""
        temp_files = [Path("temp1.wav"), Path("temp2.wav")]

        with patch("pathlib.Path.mkdir"):
            with patch("pathlib.Path.write_text"):
                _handle_output_and_benchmarks(
                    task="transcribe",
                    audio_file=self.audio_file,
                    result=self.result,
                    total_time=self.total_time,
                    output=None,
                    export_format="json",
                    export_format_explicit=False,
                    benchmark_enabled=False,
                    benchmark_extra=(),
                    benchmark_flags=None,
                    benchmark_gpu_stats=None,
                    temp_files=temp_files,
                    progress_cb=None,
                    quiet=False,
                )

                # Verify temp file cleanup was called
                mock_cleanup.assert_called_once_with(temp_files)
