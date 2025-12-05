"""Tests for CLI cancellation cleanup to prevent GPU memory leaks.

This module verifies that when a user cancels a transcription (CTRL+C),
the backend resources are properly released to prevent GPU memory leaks.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

from click.testing import CliRunner

from insanely_fast_whisper_rocm.cli.commands import transcribe
from insanely_fast_whisper_rocm.core.errors import TranscriptionCancelledError


class TestCliCancellationCleanup:
    """Test CLI resource cleanup on cancellation."""

    def test_transcribe_cancellation_calls_backend_close(self) -> None:
        """Verify that backend.close() is called when user cancels transcription.

        This test ensures that when TranscriptionCancelledError is raised
        (e.g., user presses CTRL+C), the CLI properly releases GPU memory
        by calling backend.close() before exiting.
        """
        runner = CliRunner()

        # Create a mock backend with a close method
        mock_backend = MagicMock()
        mock_backend.close = Mock()

        # Mock the facade to track cleanup calls
        with patch("insanely_fast_whisper_rocm.cli.commands.cli_facade") as mock_facade:
            # Setup the facade to have a backend attribute
            mock_facade.backend = mock_backend

            # Simulate a cancellation during processing
            mock_facade.process_audio.side_effect = TranscriptionCancelledError(
                "Transcription cancelled by user"
            )

            # Create a temporary test audio file
            with runner.isolated_filesystem():
                test_audio = Path("test.wav")
                test_audio.write_text("fake audio data")

                # Run the transcribe command (should exit with code 130)
                result = runner.invoke(transcribe, [str(test_audio)])

                # Verify the command exited with cancellation code
                assert result.exit_code == 130

                # Verify cancellation message was shown
                assert "Operation cancelled by user" in result.output

                # CRITICAL: Verify backend.close() was called to release GPU memory
                # This is the test that currently fails because the implementation
                # doesn't have cleanup logic yet
                mock_backend.close.assert_called_once()

    def test_transcribe_cancellation_handles_missing_backend(self) -> None:
        """Verify cleanup handles case where backend is None or missing.

        This test ensures that if cancellation happens before the backend
        is initialized, the cleanup logic doesn't crash.
        """
        runner = CliRunner()

        with patch("insanely_fast_whisper_rocm.cli.commands.cli_facade") as mock_facade:
            # Backend is None (not initialized yet)
            mock_facade.backend = None

            mock_facade.process_audio.side_effect = TranscriptionCancelledError(
                "Transcription cancelled by user"
            )

            with runner.isolated_filesystem():
                test_audio = Path("test.wav")
                test_audio.write_text("fake audio data")

                # Should not crash even though backend is None
                result = runner.invoke(transcribe, [str(test_audio)])

                # Should still exit with cancellation code
                assert result.exit_code == 130

    def test_transcribe_cancellation_logs_cleanup_failures(self) -> None:
        """Verify that cleanup failures are logged but don't crash shutdown.

        If backend.close() raises an exception, it should be caught and logged
        rather than propagating to the user.
        """
        runner = CliRunner()

        mock_backend = MagicMock()
        mock_backend.close = Mock(side_effect=RuntimeError("Close failed"))

        with patch("insanely_fast_whisper_rocm.cli.commands.cli_facade") as mock_facade:
            mock_facade.backend = mock_backend

            mock_facade.process_audio.side_effect = TranscriptionCancelledError(
                "Transcription cancelled by user"
            )

            with patch("insanely_fast_whisper_rocm.cli.commands.logger") as mock_logger:
                with runner.isolated_filesystem():
                    test_audio = Path("test.wav")
                    test_audio.write_text("fake audio data")

                    # Should not crash even though close() raised an exception
                    result = runner.invoke(transcribe, [str(test_audio)])

                    # Should still exit with cancellation code
                    assert result.exit_code == 130

                    # Verify the error was logged
                    warning_calls = [
                        call
                        for call in mock_logger.warning.call_args_list
                        if "Failed to cleanup backend" in str(call)
                    ]
                    assert len(warning_calls) > 0

    def test_transcribe_cancellation_checks_facade_attribute(self) -> None:
        """Verify cleanup checks that facade has backend attribute.

        This handles the case where the facade object doesn't have
        a backend attribute at all (edge case for compatibility).
        """
        runner = CliRunner()

        with patch("insanely_fast_whisper_rocm.cli.commands.cli_facade") as mock_facade:
            # Remove backend attribute entirely using delattr simulation
            type(mock_facade).backend = property(
                lambda self: (_ for _ in ()).throw(AttributeError("no backend"))
            )

            mock_facade.process_audio.side_effect = TranscriptionCancelledError(
                "Transcription cancelled by user"
            )

            with runner.isolated_filesystem():
                test_audio = Path("test.wav")
                test_audio.write_text("fake audio data")

                # Should not crash even though facade has no backend attribute
                result = runner.invoke(transcribe, [str(test_audio)])

                # Should still exit with cancellation code
                assert result.exit_code == 130
