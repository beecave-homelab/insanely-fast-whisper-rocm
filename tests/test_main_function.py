"""Tests for main function in insanely_fast_whisper_api.__main__."""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from click.testing import CliRunner

from insanely_fast_whisper_api.__main__ import main
from insanely_fast_whisper_api.utils import constants


class TestMainFunction:
    """Test the main function (click command)."""

    @patch('insanely_fast_whisper_api.__main__.uvicorn', None)
    def test_main_uvicorn_not_installed(self) -> None:
        """Test main function when uvicorn is not installed."""
        runner = CliRunner()

        # Execute
        result = runner.invoke(main, [])

        # Verify
        assert result.exit_code == 1
        assert "Uvicorn is not installed" in result.output
        assert "pip install uvicorn" in result.output

    @patch('insanely_fast_whisper_api.__main__.uvicorn')
    @patch('insanely_fast_whisper_api.__main__.load_logging_config')
    @patch('insanely_fast_whisper_api.__main__.logging')
    @patch('click.echo')
    @patch('click.secho')
    def test_main_default_options(
        self,
        mock_secho: Mock,
        mock_echo: Mock,
        mock_logging: Mock,
        mock_load_config: Mock,
        mock_uvicorn: Mock,
    ) -> None:
        """Test main function with default options."""
        # Setup mocks
        mock_config = {"test": "config"}
        mock_load_config.return_value = mock_config
        mock_uvicorn.run = Mock()

        runner = CliRunner()

        # Execute
        result = runner.invoke(main, [])

        # Verify
        assert result.exit_code == 0
        mock_load_config.assert_called_once_with(debug=False)
        mock_logging.config.dictConfig.assert_called_once_with(mock_config)
        mock_uvicorn.run.assert_called_once_with(
            "insanely_fast_whisper_api.main:app",
            host=constants.API_HOST,
            port=constants.API_PORT,
            workers=1,
            log_level=constants.LOG_LEVEL.lower(),
            reload=False,
            ssl_keyfile=None,
            ssl_certfile=None,
            log_config=mock_config,
        )

        # Check output messages
        assert any("🚀 Starting" in call[0][0] for call in mock_secho.call_args_list)
        assert any("🔗 Listening on:" in call[0][0] for call in mock_secho.call_args_list)

    @patch('insanely_fast_whisper_api.__main__.uvicorn')
    @patch('insanely_fast_whisper_api.__main__.load_logging_config')
    @patch('insanely_fast_whisper_api.__main__.logging')
    @patch('click.echo')
    @patch('click.secho')
    def test_main_custom_options(
        self,
        mock_secho: Mock,
        mock_echo: Mock,
        mock_logging: Mock,
        mock_load_config: Mock,
        mock_uvicorn: Mock,
    ) -> None:
        """Test main function with custom options."""
        # Setup mocks
        mock_config = {"test": "config"}
        mock_load_config.return_value = mock_config
        mock_uvicorn.run = Mock()

        runner = CliRunner()

        # Execute
        result = runner.invoke(main, [
            "--host", "127.0.0.1",
            "--port", "8080",
            "--workers", "4",
            "--log-level", "debug",
            "--reload"
        ])

        # Verify
        assert result.exit_code == 0
        mock_load_config.assert_called_once_with(debug=False)
        mock_uvicorn.run.assert_called_once_with(
            "insanely_fast_whisper_api.main:app",
            host="127.0.0.1",
            port=8080,
            workers=4,
            log_level="debug",
            reload=True,
            ssl_keyfile=None,
            ssl_certfile=None,
            log_config=mock_config,
        )

    @patch('insanely_fast_whisper_api.__main__.uvicorn')
    @patch('insanely_fast_whisper_api.__main__.load_logging_config')
    @patch('insanely_fast_whisper_api.__main__.logging')
    @patch('click.echo')
    @patch('click.secho')
    def test_main_ssl_options(
        self,
        mock_secho: Mock,
        mock_echo: Mock,
        mock_logging: Mock,
        mock_load_config: Mock,
        mock_uvicorn: Mock,
    ) -> None:
        """Test main function with SSL options."""
        # Setup mocks
        mock_config = {"test": "config"}
        mock_load_config.return_value = mock_config
        mock_uvicorn.run = Mock()

        # Create temporary SSL files
        with tempfile.NamedTemporaryFile(delete=False) as key_file, \
             tempfile.NamedTemporaryFile(delete=False) as cert_file:

            key_path = key_file.name
            cert_path = cert_file.name

        try:
            runner = CliRunner()

            # Execute
            result = runner.invoke(main, [
                "--ssl-keyfile", key_path,
                "--ssl-certfile", cert_path
            ])

            # Verify
            assert result.exit_code == 0
            mock_uvicorn.run.assert_called_once_with(
                "insanely_fast_whisper_api.main:app",
                host=constants.API_HOST,
                port=constants.API_PORT,
                workers=1,
                log_level=constants.LOG_LEVEL.lower(),
                reload=False,
                ssl_keyfile=key_path,
                ssl_certfile=cert_path,
                log_config=mock_config,
            )

            # Check SSL messages
            secho_calls = [call[0][0] for call in mock_secho.call_args_list]
            assert any("🔒 SSL is enabled." in call for call in secho_calls)
            assert any(f"Keyfile: {key_path}" in call for call in secho_calls)
            assert any(f"Certfile: {cert_path}" in call for call in secho_calls)

        finally:
            Path(key_path).unlink(missing_ok=True)
            Path(cert_path).unlink(missing_ok=True)

    @patch('insanely_fast_whisper_api.__main__.uvicorn')
    @patch('insanely_fast_whisper_api.__main__.load_logging_config')
    @patch('insanely_fast_whisper_api.__main__.logging')
    @patch('click.echo')
    @patch('click.secho')
    def test_main_debug_flag(
        self,
        mock_secho: Mock,
        mock_echo: Mock,
        mock_logging: Mock,
        mock_load_config: Mock,
        mock_uvicorn: Mock,
    ) -> None:
        """Test main function with debug flag."""
        # Setup mocks
        mock_config = {"test": "config"}
        mock_load_config.return_value = mock_config
        mock_uvicorn.run = Mock()

        runner = CliRunner()

        # Execute with debug flag
        result = runner.invoke(main, ["--debug"])

        # Verify
        assert result.exit_code == 0
        mock_load_config.assert_called_once_with(debug=True)
        mock_uvicorn.run.assert_called_once_with(
            "insanely_fast_whisper_api.main:app",
            host=constants.API_HOST,
            port=constants.API_PORT,
            workers=1,
            log_level="debug",  # Should be upgraded from default
            reload=False,
            ssl_keyfile=None,
            ssl_certfile=None,
            log_config=mock_config,
        )

        # Check debug messages
        secho_calls = [call[0][0] for call in mock_secho.call_args_list]
        assert any("🐛 Debug mode is ON" in call for call in secho_calls)

    @patch('insanely_fast_whisper_api.__main__.uvicorn')
    @patch('insanely_fast_whisper_api.__main__.load_logging_config')
    @patch('insanely_fast_whisper_api.__main__.logging')
    @patch('click.echo')
    @patch('click.secho')
    def test_main_debug_with_custom_log_level(
        self,
        mock_secho: Mock,
        mock_echo: Mock,
        mock_logging: Mock,
        mock_load_config: Mock,
        mock_uvicorn: Mock,
    ) -> None:
        """Test main function with debug flag and custom log level."""
        # Setup mocks
        mock_config = {"test": "config"}
        mock_load_config.return_value = mock_config
        mock_uvicorn.run = Mock()

        runner = CliRunner()

        # Execute with debug flag and custom log level
        result = runner.invoke(main, ["--debug", "--log-level", "warning"])

        # Verify
        assert result.exit_code == 0
        mock_uvicorn.run.assert_called_once_with(
            "insanely_fast_whisper_api.main:app",
            host=constants.API_HOST,
            port=constants.API_PORT,
            workers=1,
            log_level="warning",  # Should NOT be upgraded since it's not default
            reload=False,
            ssl_keyfile=None,
            ssl_certfile=None,
            log_config=mock_config,
        )

    @patch('insanely_fast_whisper_api.__main__.uvicorn')
    @patch('insanely_fast_whisper_api.__main__.load_logging_config')
    @patch('insanely_fast_whisper_api.__main__.logging')
    @patch('click.echo')
    @patch('click.secho')
    def test_main_workers_message(
        self,
        mock_secho: Mock,
        mock_echo: Mock,
        mock_logging: Mock,
        mock_load_config: Mock,
        mock_uvicorn: Mock,
    ) -> None:
        """Test main function with workers > 1 shows workers message."""
        # Setup mocks
        mock_config = {"test": "config"}
        mock_load_config.return_value = mock_config
        mock_uvicorn.run = Mock()

        runner = CliRunner()

        # Execute with multiple workers
        result = runner.invoke(main, ["--workers", "4"])

        # Verify
        assert result.exit_code == 0
        secho_calls = [call[0][0] for call in mock_secho.call_args_list]
        assert any("⚙️  Running with 4 worker processes." in call for call in secho_calls)

    @patch('insanely_fast_whisper_api.__main__.uvicorn')
    @patch('insanely_fast_whisper_api.__main__.load_logging_config')
    @patch('insanely_fast_whisper_api.__main__.logging')
    @patch('click.echo')
    @patch('click.secho')
    def test_main_reload_message(
        self,
        mock_secho: Mock,
        mock_echo: Mock,
        mock_logging: Mock,
        mock_load_config: Mock,
        mock_uvicorn: Mock,
    ) -> None:
        """Test main function with reload shows reload message."""
        # Setup mocks
        mock_config = {"test": "config"}
        mock_load_config.return_value = mock_config
        mock_uvicorn.run = Mock()

        runner = CliRunner()

        # Execute with reload
        result = runner.invoke(main, ["--reload"])

        # Verify
        assert result.exit_code == 0
        secho_calls = [call[0][0] for call in mock_secho.call_args_list]
        assert any("🔄 Auto-reload is enabled" in call for call in secho_calls)

    def test_main_version_option(self) -> None:
        """Test main function version option."""
        runner = CliRunner()

        # Execute
        result = runner.invoke(main, ["--version"])

        # Verify
        assert result.exit_code == 0
        assert constants.API_VERSION in result.output
        assert constants.API_TITLE in result.output

    def test_main_help_option(self) -> None:
        """Test main function help option."""
        runner = CliRunner()

        # Execute
        result = runner.invoke(main, ["--help"])

        # Verify
        assert result.exit_code == 0
        assert "Starts the Insanely Fast Whisper API server" in result.output
        # Check that all options are present
        expected_options = [
            "--host", "--port", "--workers", "--log-level",
            "--reload", "--ssl-keyfile", "--ssl-certfile", "--debug", "--help"
        ]
        for option in expected_options:
            assert option in result.output
