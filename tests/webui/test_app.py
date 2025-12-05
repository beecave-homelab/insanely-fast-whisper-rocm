"""Tests for insanely_fast_whisper_rocm.webui.app module.

This module contains tests for the WebUI application launch functionality.
"""

from __future__ import annotations

from unittest.mock import MagicMock, Mock, patch

from click.testing import CliRunner

from insanely_fast_whisper_rocm.webui.app import launch_webui


class TestLaunchWebUI:
    """Test suite for launch_webui CLI command."""

    @patch("insanely_fast_whisper_rocm.webui.app.download_model_if_needed")
    @patch("insanely_fast_whisper_rocm.webui.app.create_ui_components")
    def test_launch_webui__basic_invocation(
        self, mock_create_ui: MagicMock, mock_download: MagicMock
    ) -> None:
        """Test basic invocation of launch_webui command."""
        runner = CliRunner()
        mock_iface = Mock()
        mock_iface.launch = Mock()
        mock_create_ui.return_value = mock_iface

        with runner.isolated_filesystem():
            result = runner.invoke(
                launch_webui,
                ["--host", "127.0.0.1", "--port", "7860"],
                catch_exceptions=False,
            )

        # Command should execute without errors
        assert result.exit_code == 0
        mock_download.assert_called_once()
        mock_create_ui.assert_called_once()
        mock_iface.launch.assert_called_once()

    @patch("insanely_fast_whisper_rocm.webui.app.download_model_if_needed")
    @patch("insanely_fast_whisper_rocm.webui.app.create_ui_components")
    def test_launch_webui__with_custom_host_and_port(
        self, mock_create_ui: MagicMock, mock_download: MagicMock
    ) -> None:
        """Test launch_webui with custom host and port."""
        runner = CliRunner()
        mock_iface = Mock()
        mock_iface.launch = Mock()
        mock_create_ui.return_value = mock_iface

        with runner.isolated_filesystem():
            result = runner.invoke(
                launch_webui,
                ["--host", "0.0.0.0", "--port", "8080"],
                catch_exceptions=False,
            )

        assert result.exit_code == 0
        # Verify launch was called with correct parameters
        mock_iface.launch.assert_called_once_with(
            server_name="0.0.0.0",
            server_port=8080,
            share=False,
        )

    @patch("insanely_fast_whisper_rocm.webui.app.download_model_if_needed")
    @patch("insanely_fast_whisper_rocm.webui.app.create_ui_components")
    def test_launch_webui__with_share_flag(
        self, mock_create_ui: MagicMock, mock_download: MagicMock
    ) -> None:
        """Test launch_webui with --share flag."""
        runner = CliRunner()
        mock_iface = Mock()
        mock_iface.launch = Mock()
        mock_create_ui.return_value = mock_iface

        with runner.isolated_filesystem():
            result = runner.invoke(launch_webui, ["--share"], catch_exceptions=False)

        assert result.exit_code == 0
        # Verify share=True was passed
        call_args = mock_iface.launch.call_args
        assert call_args.kwargs["share"] is True

    @patch("insanely_fast_whisper_rocm.webui.app.download_model_if_needed")
    @patch("insanely_fast_whisper_rocm.webui.app.create_ui_components")
    def test_launch_webui__with_custom_model(
        self, mock_create_ui: MagicMock, mock_download: MagicMock
    ) -> None:
        """Test launch_webui with custom model."""
        runner = CliRunner()
        mock_iface = Mock()
        mock_iface.launch = Mock()
        mock_create_ui.return_value = mock_iface
        custom_model = "openai/whisper-large-v3"

        with runner.isolated_filesystem():
            result = runner.invoke(
                launch_webui, ["--model", custom_model], catch_exceptions=False
            )

        assert result.exit_code == 0
        # Verify download was called with the custom model
        mock_download.assert_called_once()
        call_args = mock_download.call_args
        assert call_args.kwargs["model_name"] == custom_model

        # Verify UI was created with the custom model
        create_ui_args = mock_create_ui.call_args
        assert create_ui_args.kwargs["default_model"] == custom_model

    @patch("insanely_fast_whisper_rocm.webui.app.download_model_if_needed")
    @patch("insanely_fast_whisper_rocm.webui.app.create_ui_components")
    def test_launch_webui__with_stabilization_flags(
        self, mock_create_ui: MagicMock, mock_download: MagicMock
    ) -> None:
        """Test launch_webui with stabilization flags."""
        runner = CliRunner()
        mock_iface = Mock()
        mock_iface.launch = Mock()
        mock_create_ui.return_value = mock_iface

        with runner.isolated_filesystem():
            result = runner.invoke(
                launch_webui,
                ["--stabilize", "--demucs", "--vad", "--vad-threshold", "0.5"],
                catch_exceptions=False,
            )

        assert result.exit_code == 0
        # Verify UI was created with stabilization options
        create_ui_args = mock_create_ui.call_args
        assert create_ui_args.kwargs["default_stabilize"] is True
        assert create_ui_args.kwargs["default_demucs"] is True
        assert create_ui_args.kwargs["default_vad"] is True
        assert create_ui_args.kwargs["default_vad_threshold"] == 0.5

    @patch("insanely_fast_whisper_rocm.webui.app.download_model_if_needed")
    @patch("insanely_fast_whisper_rocm.webui.app.create_ui_components")
    def test_launch_webui__with_no_stabilize_flags(
        self, mock_create_ui: MagicMock, mock_download: MagicMock
    ) -> None:
        """Test launch_webui with --no-stabilize flags."""
        runner = CliRunner()
        mock_iface = Mock()
        mock_iface.launch = Mock()
        mock_create_ui.return_value = mock_iface

        with runner.isolated_filesystem():
            result = runner.invoke(
                launch_webui,
                ["--no-stabilize", "--no-demucs", "--no-vad"],
                catch_exceptions=False,
            )

        assert result.exit_code == 0
        # Verify UI was created with stabilization disabled
        create_ui_args = mock_create_ui.call_args
        assert create_ui_args.kwargs["default_stabilize"] is False
        assert create_ui_args.kwargs["default_demucs"] is False
        assert create_ui_args.kwargs["default_vad"] is False

    @patch("insanely_fast_whisper_rocm.webui.app.logging.basicConfig")
    @patch("insanely_fast_whisper_rocm.webui.app.download_model_if_needed")
    @patch("insanely_fast_whisper_rocm.webui.app.create_ui_components")
    def test_launch_webui__with_debug_flag(
        self,
        mock_create_ui: MagicMock,
        mock_download: MagicMock,
        mock_logging_config: MagicMock,
    ) -> None:
        """Test launch_webui with --debug flag."""
        runner = CliRunner()
        mock_iface = Mock()
        mock_iface.launch = Mock()
        mock_create_ui.return_value = mock_iface

        with runner.isolated_filesystem():
            result = runner.invoke(launch_webui, ["--debug"], catch_exceptions=False)

        assert result.exit_code == 0
        # Verify logging was configured with DEBUG level
        import logging

        mock_logging_config.assert_called_once()
        call_args = mock_logging_config.call_args
        assert call_args.kwargs["level"] == logging.DEBUG

    @patch("insanely_fast_whisper_rocm.webui.app.logging.basicConfig")
    @patch("insanely_fast_whisper_rocm.webui.app.download_model_if_needed")
    @patch("insanely_fast_whisper_rocm.webui.app.create_ui_components")
    def test_launch_webui__without_debug_flag(
        self,
        mock_create_ui: MagicMock,
        mock_download: MagicMock,
        mock_logging_config: MagicMock,
    ) -> None:
        """Test launch_webui without --debug flag."""
        runner = CliRunner()
        mock_iface = Mock()
        mock_iface.launch = Mock()
        mock_create_ui.return_value = mock_iface

        with runner.isolated_filesystem():
            result = runner.invoke(launch_webui, [], catch_exceptions=False)

        assert result.exit_code == 0
        # Verify logging was configured with INFO level
        import logging

        mock_logging_config.assert_called_once()
        call_args = mock_logging_config.call_args
        assert call_args.kwargs["level"] == logging.INFO

    @patch("insanely_fast_whisper_rocm.webui.app.download_model_if_needed")
    @patch("insanely_fast_whisper_rocm.webui.app.create_ui_components")
    @patch("insanely_fast_whisper_rocm.webui.app.DEFAULT_MODEL", "openai/whisper-tiny")
    def test_launch_webui__uses_default_model_when_not_specified(
        self, mock_create_ui: MagicMock, mock_download: MagicMock
    ) -> None:
        """Test that launch_webui uses DEFAULT_MODEL when model is not specified."""
        runner = CliRunner()
        mock_iface = Mock()
        mock_iface.launch = Mock()
        mock_create_ui.return_value = mock_iface

        with runner.isolated_filesystem():
            result = runner.invoke(launch_webui, [], catch_exceptions=False)

        assert result.exit_code == 0
        # Verify UI was created with the default model
        create_ui_args = mock_create_ui.call_args
        # When model is None, it should use DEFAULT_MODEL
        assert "default_model" in create_ui_args.kwargs

    @patch("insanely_fast_whisper_rocm.webui.app.download_model_if_needed")
    @patch("insanely_fast_whisper_rocm.webui.app.create_ui_components")
    def test_launch_webui__calls_download_before_ui_creation(
        self, mock_create_ui: MagicMock, mock_download: MagicMock
    ) -> None:
        """Test that model download is called before UI creation."""
        runner = CliRunner()
        mock_iface = Mock()
        mock_iface.launch = Mock()
        mock_create_ui.return_value = mock_iface

        call_order = []
        mock_download.side_effect = lambda **kwargs: call_order.append("download")
        mock_create_ui.side_effect = lambda **kwargs: (
            call_order.append("create_ui"),
            mock_iface,
        )[1]

        with runner.isolated_filesystem():
            result = runner.invoke(launch_webui, [], catch_exceptions=False)

        assert result.exit_code == 0
        assert call_order == ["download", "create_ui"]

    @patch("insanely_fast_whisper_rocm.webui.app.download_model_if_needed")
    @patch("insanely_fast_whisper_rocm.webui.app.create_ui_components")
    def test_launch_webui__all_parameters_combined(
        self, mock_create_ui: MagicMock, mock_download: MagicMock
    ) -> None:
        """Test launch_webui with all parameters combined."""
        runner = CliRunner()
        mock_iface = Mock()
        mock_iface.launch = Mock()
        mock_create_ui.return_value = mock_iface

        with runner.isolated_filesystem():
            result = runner.invoke(
                launch_webui,
                [
                    "--host",
                    "0.0.0.0",
                    "--port",
                    "9000",
                    "--share",
                    "--model",
                    "openai/whisper-medium",
                    "--stabilize",
                    "--demucs",
                    "--vad",
                    "--vad-threshold",
                    "0.4",
                    "--debug",
                ],
                catch_exceptions=False,
            )

        assert result.exit_code == 0
        # Verify all parameters were applied
        launch_args = mock_iface.launch.call_args
        assert launch_args.kwargs["server_name"] == "0.0.0.0"
        assert launch_args.kwargs["server_port"] == 9000
        assert launch_args.kwargs["share"] is True

        create_ui_args = mock_create_ui.call_args
        assert create_ui_args.kwargs["default_model"] == "openai/whisper-medium"
        assert create_ui_args.kwargs["default_stabilize"] is True
        assert create_ui_args.kwargs["default_demucs"] is True
        assert create_ui_args.kwargs["default_vad"] is True
        assert create_ui_args.kwargs["default_vad_threshold"] == 0.4
