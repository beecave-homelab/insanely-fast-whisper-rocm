"""Tests for setup_timezone function in insanely_fast_whisper_rocm.__main__."""

import os
from unittest.mock import Mock, patch

from insanely_fast_whisper_rocm.__main__ import setup_timezone
from insanely_fast_whisper_rocm.utils import constants


class TestSetupTimezone:
    """Test the setup_timezone function."""

    @patch("time.tzset")
    @patch("time.tzname", new=["EST", "EDT"])
    @patch("logging.info")
    def test_setup_timezone_success(
        self, mock_logging_info: Mock, mock_tzset: Mock
    ) -> None:
        """Test successful timezone setup."""
        # Setup
        expected_timezone = constants.APP_TIMEZONE

        # Execute
        setup_timezone()

        # Verify
        assert os.environ["TZ"] == expected_timezone
        mock_tzset.assert_called_once()
        mock_logging_info.assert_called_once_with(
            "Timezone set to: %s (%s) using APP_TIMEZONE='%s'",
            "EST",
            "EDT",
            expected_timezone,
        )

    @patch("os.environ.__setitem__", side_effect=OSError("Permission denied"))
    @patch("logging.warning")
    def test_setup_timezone_oserror(
        self, mock_logging_warning: Mock, mock_setitem: Mock
    ) -> None:
        """Test timezone setup with OSError."""
        # Execute
        setup_timezone()

        # Verify
        mock_setitem.assert_called_once_with("TZ", constants.APP_TIMEZONE)
        mock_logging_warning.assert_called_once_with(
            "Could not set timezone using APP_TIMEZONE='%s': %s. Using system default.",
            constants.APP_TIMEZONE,
            "Permission denied",
        )

    @patch("time.tzset", side_effect=TypeError("Invalid timezone"))
    @patch("logging.warning")
    def test_setup_timezone_typeerror(
        self, mock_logging_warning: Mock, mock_tzset: Mock
    ) -> None:
        """Test timezone setup with TypeError."""
        # Setup
        os.environ["TZ"] = constants.APP_TIMEZONE

        # Execute
        setup_timezone()

        # Verify
        mock_tzset.assert_called_once()
        mock_logging_warning.assert_called_once_with(
            "Could not set timezone using APP_TIMEZONE='%s': %s. Using system default.",
            constants.APP_TIMEZONE,
            "Invalid timezone",
        )

    @patch("time.tzname", new=["EST"])  # Missing EDT
    @patch("time.tzset")
    @patch("logging.warning")
    def test_setup_timezone_indexerror(
        self, mock_logging_warning: Mock, mock_tzset: Mock
    ) -> None:
        """Test timezone setup with IndexError (missing tzname[1])."""
        # Setup
        os.environ["TZ"] = constants.APP_TIMEZONE

        # Execute
        setup_timezone()

        # Verify
        mock_tzset.assert_called_once()
        mock_logging_warning.assert_called_once_with(
            "Could not set timezone using APP_TIMEZONE='%s': %s. Using system default.",
            constants.APP_TIMEZONE,
            "list index out of range",  # IndexError message
        )
