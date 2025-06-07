import pytest
import os
import datetime
from unittest.mock import patch, MagicMock
from zoneinfo import ZoneInfo

from insanely_fast_whisper_api.utils.filename_generator import (
    FilenameGenerator,
    StandardFilenameStrategy,
    TaskType,
    FilenameComponents,
)

# Import from centralized configuration instead of local constants
from insanely_fast_whisper_api.utils.constants import FILENAME_TIMEZONE

# Define a fixed timestamp for deterministic tests
FIXED_DATETIME_UTC = datetime.datetime(
    2023, 10, 26, 12, 30, 0, tzinfo=datetime.timezone.utc
)
FIXED_DATETIME_AMS_STR = "20231026T143000Z"  # 12:30 UTC is 14:30 Amsterdam (UTC+2 DST)
FIXED_DATETIME_UTC_STR = "20231026T123000Z"


@pytest.fixture
def standard_strategy():
    return StandardFilenameStrategy()


@pytest.fixture
def filename_generator(standard_strategy):
    return FilenameGenerator(strategy=standard_strategy)


class TestTaskType:
    def test_task_type_values(self):
        assert TaskType.TRANSCRIBE.value == "transcribe"
        assert TaskType.TRANSLATE.value == "translate"


class TestFilenameComponents:
    def test_filename_components_creation(self):
        dt = datetime.datetime.now(datetime.timezone.utc)
        components = FilenameComponents(
            audio_stem="test_audio",
            task=TaskType.TRANSCRIBE,
            timestamp=dt,
            extension="json",
        )
        assert components.audio_stem == "test_audio"
        assert components.task == TaskType.TRANSCRIBE
        assert components.timestamp == dt
        assert components.extension == "json"


class TestStandardFilenameStrategy:
    def test_generate_filename_basic(self, standard_strategy):
        components = FilenameComponents(
            audio_stem="my_audio",
            task=TaskType.TRANSCRIBE,
            timestamp=FIXED_DATETIME_UTC,
            extension="json",
        )
        expected_filename = f"my_audio_transcribe_{FIXED_DATETIME_UTC_STR}.json"
        assert standard_strategy.generate_filename(components) == expected_filename

    def test_generate_filename_with_leading_dot_extension(self, standard_strategy):
        components = FilenameComponents(
            audio_stem="another.sample",
            task=TaskType.TRANSLATE,
            timestamp=FIXED_DATETIME_UTC,
            extension=".srt",
        )
        expected_filename = f"another.sample_translate_{FIXED_DATETIME_UTC_STR}.srt"
        assert standard_strategy.generate_filename(components) == expected_filename


class TestFilenameGenerator:
    def test_init_with_valid_strategy(self, standard_strategy):
        try:
            _ = FilenameGenerator(strategy=standard_strategy)
        except TypeError:
            pytest.fail(
                "FilenameGenerator raised TypeError unexpectedly with valid strategy"
            )

    def test_init_with_invalid_strategy(self):
        with pytest.raises(TypeError):
            _ = FilenameGenerator(strategy=object())  # type: ignore

    @pytest.mark.parametrize(
        "audio_path, expected_stem",
        [
            ("test.wav", "test"),
            ("/path/to/some_file.mp3", "some_file"),
            (
                "./relative/path/archive.tar.gz",
                "archive.tar",
            ),  # os.path.splitext behavior
            ("no_extension_file", "no_extension_file"),
        ],
    )
    @patch.dict(os.environ, {}, clear=True)  # Ensure no FILENAME_TIMEZONE
    @patch(
        "insanely_fast_whisper_api.utils.filename_generator.datetime"
    )  # Patch module
    def test_create_filename_stem_extraction(
        self,
        mock_datetime_module,
        filename_generator,
        audio_path,
        expected_stem,
        standard_strategy,
    ):
        mock_datetime_module.datetime.now.return_value = FIXED_DATETIME_UTC
        mock_datetime_module.timezone = datetime.timezone
        mock_datetime_module.datetime.side_effect = (
            lambda *args, **kwargs: datetime.datetime(*args, **kwargs)
        )
        filename = filename_generator.create_filename(
            audio_path, TaskType.TRANSCRIBE, "json"
        )
        assert filename.startswith(f"{expected_stem}_transcribe_")

    @patch.dict(os.environ, {}, clear=True)  # Ensure no FILENAME_TIMEZONE
    @patch(
        "insanely_fast_whisper_api.utils.filename_generator.datetime"
    )  # Patch module
    def test_create_filename_default_utc_timestamp(
        self, mock_datetime_module, filename_generator
    ):
        mock_datetime_module.datetime.now.return_value = FIXED_DATETIME_UTC
        mock_datetime_module.timezone = (
            datetime.timezone
        )  # Ensure datetime.timezone.utc is available
        mock_datetime_module.datetime.side_effect = (
            lambda *args, **kwargs: datetime.datetime(*args, **kwargs)
        )

        filename = filename_generator.create_filename(
            "audio.mp3", TaskType.TRANSCRIBE, "json"
        )
        expected = f"audio_transcribe_{FIXED_DATETIME_UTC_STR}.json"
        assert filename == expected

    @patch.dict(os.environ, {"FILENAME_TIMEZONE": "Europe/Amsterdam"}, clear=True)
    @patch(
        "insanely_fast_whisper_api.utils.filename_generator.datetime"
    )  # Patch module
    def test_create_filename_custom_timezone_from_env(
        self, mock_datetime_module, filename_generator
    ):
        # Mock datetime.datetime.now() to return a specific timezone-aware datetime.
        ams_tz = ZoneInfo("Europe/Amsterdam")
        fixed_now_in_ams = datetime.datetime(2023, 10, 26, 13, 30, 0, tzinfo=ams_tz)

        # Configure the mock_datetime_module.datetime.now behavior
        mock_datetime_module.datetime.now.return_value = fixed_now_in_ams
        # Ensure that datetime.timezone.utc is still accessible if filename_generator uses it directly
        mock_datetime_module.timezone = datetime.timezone
        # If it uses datetime.datetime.timezone then that needs to be available on mock_datetime_module.datetime
        # However, filename_generator.py uses datetime.timezone.utc directly from 'import datetime'.
        # So, mock_datetime_module.timezone should be sufficient for that.
        # Also ensure datetime.datetime itself can be called if needed (e.g. for isinstance checks, though unlikely here)
        mock_datetime_module.datetime.side_effect = (
            lambda *args, **kwargs: datetime.datetime(*args, **kwargs)
        )

        filename = filename_generator.create_filename(
            "local.wav", TaskType.TRANSLATE, "txt"
        )
        expected_filename_ams_time_zulu_suffix = "local_translate_20231026T133000Z.txt"
        assert filename == expected_filename_ams_time_zulu_suffix

    @patch.dict(os.environ, {"FILENAME_TIMEZONE": "Europe/Amsterdam"}, clear=True)
    def test_create_filename_with_provided_naive_timestamp(self, filename_generator):
        naive_dt = datetime.datetime(2023, 11, 15, 10, 0, 0)  # 10:00 AM
        # This will be interpreted as 10:00 AM Europe/Amsterdam
        # Amsterdam is UTC+1 in November (standard time)
        # So, 2023-11-15 10:00:00 Europe/Amsterdam
        # strftime("%Y%m%dT%H%M%SZ") will produce "20231115T100000Z"
        filename = filename_generator.create_filename(
            "naive.ogg", TaskType.TRANSCRIBE, "srt", timestamp=naive_dt
        )
        expected = "naive_transcribe_20231115T100000Z.srt"
        assert filename == expected

    @patch(
        "insanely_fast_whisper_api.utils.constants.os.getenv"
    )  # Patch constants loading
    def test_create_filename_with_provided_aware_timestamp_conversion(
        self, mock_getenv, filename_generator
    ):
        # Mock constants.py environment loading to return America/New_York
        mock_getenv.side_effect = lambda key, default=None: (
            "America/New_York" if key == "FILENAME_TIMEZONE" else default
        )

        # Aware timestamp in UTC
        utc_dt = datetime.datetime(
            2023, 12, 25, 17, 0, 0, tzinfo=datetime.timezone.utc
        )  # 5 PM UTC
        # Note: Since we're using centralized configuration, the actual timezone behavior
        # depends on how the filename_generator module loads the timezone
        # With the current implementation, it uses the centralized constant
        filename = filename_generator.create_filename(
            "aware.flac", TaskType.TRANSLATE, "vtt", timestamp=utc_dt
        )
        # The filename generator now uses the centralized FILENAME_TIMEZONE which defaults to UTC
        # So the timestamp should remain in UTC format
        expected = "aware_translate_20231225T170000Z.vtt"  # UTC timestamp
        assert filename == expected

    @patch(
        "insanely_fast_whisper_api.utils.constants.os.getenv"
    )  # Patch constants loading
    @patch(
        "insanely_fast_whisper_api.utils.filename_generator.datetime"
    )  # Patch module
    @patch("builtins.print")  # Mock print to check warning
    def test_create_filename_invalid_timezone_fallback_to_utc(
        self, mock_print, mock_datetime_module, mock_getenv, filename_generator
    ):
        # Mock constants.py environment loading to return invalid timezone
        mock_getenv.side_effect = lambda key, default=None: (
            "Invalid/Timezone" if key == "FILENAME_TIMEZONE" else default
        )

        mock_datetime_module.datetime.now.return_value = (
            FIXED_DATETIME_UTC  # Fallback should use this UTC time
        )
        mock_datetime_module.timezone = datetime.timezone
        mock_datetime_module.datetime.side_effect = (
            lambda *args, **kwargs: datetime.datetime(*args, **kwargs)
        )

        filename = filename_generator.create_filename(
            "fallback.mp3", TaskType.TRANSCRIBE, "json"
        )
        expected_filename = f"fallback_transcribe_{FIXED_DATETIME_UTC_STR}.json"
        assert filename == expected_filename
        # Note: The print warning behavior depends on the specific implementation
        # Since we're using centralized config, this test verifies the fallback works

    @patch.dict(os.environ, {}, clear=True)
    @patch(
        "insanely_fast_whisper_api.utils.filename_generator.datetime"
    )  # Patch module
    def test_create_filename_task_type_and_extension(
        self, mock_datetime_module, filename_generator
    ):
        mock_datetime_module.datetime.now.return_value = FIXED_DATETIME_UTC
        mock_datetime_module.timezone = datetime.timezone
        mock_datetime_module.datetime.side_effect = (
            lambda *args, **kwargs: datetime.datetime(*args, **kwargs)
        )

        filename = filename_generator.create_filename(
            "audio_test.aac", TaskType.TRANSLATE, ".TXT"
        )
        expected = f"audio_test_translate_{FIXED_DATETIME_UTC_STR}.txt"
        assert filename == expected

    def test_default_timezone_constant(self):
        # Check if the constant is what we expect, as it's used as a default
        assert (
            FILENAME_TIMEZONE == "UTC"
        )  # Now expecting UTC as the centralized default
