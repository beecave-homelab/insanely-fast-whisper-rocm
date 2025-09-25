"""Tests for the startup event handler in insanely_fast_whisper_api/api/app.py."""

import asyncio
import logging
from unittest.mock import Mock, patch

from insanely_fast_whisper_api.api.app import create_app
from insanely_fast_whisper_api.utils.constants import API_TITLE, DEFAULT_MODEL, HF_TOKEN


class TestAppStartupEvent:
    """Test the FastAPI application startup event handler."""

    @patch("insanely_fast_whisper_api.api.app.download_model_if_needed")
    @patch("insanely_fast_whisper_api.api.app.logger")
    def test_startup_event_calls_model_download(
        self, mock_logger: Mock, mock_download: Mock
    ) -> None:
        """Test that startup event calls download_model_if_needed with correct parameters."""
        # Create app to trigger startup event setup
        app = create_app()

        # Get the startup event handler
        startup_handlers = [
            handler
            for handler in app.router.on_startup
            if hasattr(handler, "__name__") and handler.__name__ == "startup_event"
        ]
        assert len(startup_handlers) == 1
        startup_event = startup_handlers[0]

        # Execute the startup event (it's async, so we need to run it in event loop)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(startup_event())
        finally:
            loop.close()

        # Verify download_model_if_needed was called with correct parameters
        mock_download.assert_called_once_with(
            model_name=DEFAULT_MODEL,
            hf_token=HF_TOKEN,
            custom_logger=mock_logger,
        )

    @patch("insanely_fast_whisper_api.api.app.download_model_if_needed")
    @patch("insanely_fast_whisper_api.api.app.logger")
    def test_startup_event_logs_startup_messages(
        self, mock_logger: Mock, mock_download: Mock
    ) -> None:
        """Test that startup event logs API title, version, and description."""
        # Create app to trigger startup event setup
        app = create_app()

        # Get the startup event handler
        startup_handlers = [
            handler
            for handler in app.router.on_startup
            if hasattr(handler, "__name__") and handler.__name__ == "startup_event"
        ]
        startup_event = startup_handlers[0]

        # Execute the startup event
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(startup_event())
        finally:
            loop.close()

        # Verify logging calls - check that key messages were logged
        call_args_list = mock_logger.info.call_args_list
        assert len(call_args_list) >= 5  # Should have multiple log calls

        # Check for specific log messages
        log_messages = [str(call[0][0]) for call in call_args_list]
        assert any("Attempting to download/verify" in msg for msg in log_messages)
        assert any("Model download/verification process" in msg for msg in log_messages)
        assert any("Starting" in msg and API_TITLE in msg for msg in log_messages)
        assert any("API Description:" in msg for msg in log_messages)
        assert any("Available endpoints:" in msg for msg in log_messages)

    @patch("insanely_fast_whisper_api.api.app.download_model_if_needed")
    @patch("insanely_fast_whisper_api.api.app.logger")
    def test_startup_event_logs_routes(
        self, mock_logger: Mock, mock_download: Mock
    ) -> None:
        """Test that startup event logs all available routes."""
        # Create app with some test routes
        app = create_app()

        # Add a test route for verification
        @app.get("/test-route")
        async def test_route() -> dict:
            return {"test": "data"}

        # Get the startup event handler
        startup_handlers = [
            handler
            for handler in app.router.on_startup
            if hasattr(handler, "__name__") and handler.__name__ == "startup_event"
        ]
        startup_event = startup_handlers[0]

        # Execute the startup event
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(startup_event())
        finally:
            loop.close()

        # Verify that routes were logged
        call_args_list = mock_logger.info.call_args_list

        # Should have logged "Available endpoints:"
        endpoints_logged = any(
            "Available endpoints:" in str(call[0]) for call in call_args_list
        )
        assert endpoints_logged

        # Should have processed routes (exact count depends on included router)
        route_calls = [call for call in call_args_list if len(call[0]) >= 2]
        assert len(route_calls) > 0

    @patch("insanely_fast_whisper_api.api.app.download_model_if_needed")
    @patch("insanely_fast_whisper_api.api.app.logger")
    def test_startup_event_logs_route_descriptions_in_debug(
        self, mock_logger: Mock, mock_download: Mock
    ) -> None:
        """Test that route descriptions are logged in debug mode."""
        # Set logger to DEBUG level
        mock_logger.isEnabledFor.side_effect = lambda level: level == logging.DEBUG

        # Create app with a route that has a description
        app = create_app()

        # Add a test route with description
        @app.get("/test-route", description="Test route description")
        async def test_route() -> dict:
            return {"test": "data"}

        # Get the startup event handler
        startup_handlers = [
            handler
            for handler in app.router.on_startup
            if hasattr(handler, "__name__") and handler.__name__ == "startup_event"
        ]
        startup_event = startup_handlers[0]

        # Execute the startup event
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(startup_event())
        finally:
            loop.close()

        # Verify debug logging was called for route descriptions
        debug_calls = [
            call
            for call in mock_logger.debug.call_args_list
            if "Description:" in str(call[0])
        ]
        assert len(debug_calls) > 0

    @patch("insanely_fast_whisper_api.api.app.download_model_if_needed")
    @patch("insanely_fast_whisper_api.api.app.logger")
    def test_startup_event_skips_route_descriptions_when_not_debug(
        self, mock_logger: Mock, mock_download: Mock
    ) -> None:
        """Test that route descriptions are not logged when not in debug mode."""
        # Set logger to INFO level (not DEBUG)
        mock_logger.isEnabledFor.side_effect = lambda level: level != logging.DEBUG

        # Create app with a route that has a description
        app = create_app()

        # Get the startup event handler
        startup_handlers = [
            handler
            for handler in app.router.on_startup
            if hasattr(handler, "__name__") and handler.__name__ == "startup_event"
        ]
        startup_event = startup_handlers[0]

        # Execute the startup event
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(startup_event())
        finally:
            loop.close()

        # Verify debug logging was NOT called for route descriptions
        debug_calls = [
            call
            for call in mock_logger.debug.call_args_list
            if "Description:" in str(call[0])
        ]
        assert len(debug_calls) == 0

    @patch("insanely_fast_whisper_api.api.app.download_model_if_needed")
    @patch("insanely_fast_whisper_api.api.app.logger")
    def test_startup_event_handles_multiple_route_types(
        self, mock_logger: Mock, mock_download: Mock
    ) -> None:
        """Test that startup event handles different types of routes."""
        # Create app
        app = create_app()

        # The app should have various routes from the included router
        # Get the startup event handler
        startup_handlers = [
            handler
            for handler in app.router.on_startup
            if hasattr(handler, "__name__") and handler.__name__ == "startup_event"
        ]
        startup_event = startup_handlers[0]

        # Execute the startup event
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(startup_event())
        finally:
            loop.close()

        # Verify that APIRoutes were processed (non-APIRoutes should be skipped)
        call_args_list = mock_logger.info.call_args_list

        # Should have logged "Available endpoints:"
        endpoints_logged = any(
            "Available endpoints:" in str(call[0]) for call in call_args_list
        )
        assert endpoints_logged

        # Should have processed routes (exact count depends on included router)
        route_calls = [call for call in call_args_list if len(call[0]) >= 2]
        assert len(route_calls) > 0
