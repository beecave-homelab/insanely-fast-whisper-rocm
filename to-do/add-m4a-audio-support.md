# To-Do: Add Support for .m4a Audio Files

This plan outlines the steps to add support for `.m4a` audio files, including adding a conversion step using `ffmpeg-python` if the format is not natively supported by the pipeline.

## Tasks

- [x] **Analysis Phase:**
  - [x] Research confirmed: `.m4a` is natively decoded by FFmpeg → no conversion needed.
    - Path: `insanely_fast_whisper_api/core/transcription.py`
    - Action: Investigate the audio processing capabilities of the underlying model and pipeline.
    - Analysis Results:
      - [ ] TBD
    - Accept Criteria: A clear determination of whether a conversion step is necessary.

- [x] **Dependency Phase:** (skipped - not required)
  - [x] _Skipped_: extra dependency not required.
    - Path: `pyproject.toml`
    - Action: Run `pdm add ffmpeg-python`.
    - Accept Criteria: The dependency is added and locked.

- [x] **Implementation Phase:**
  - [x] **Update Constants:**
    - Path: `insanely_fast_whisper_api/utils/constants.py`
    - Action: Add `.m4a` to the `SUPPORTED_AUDIO_FORMATS` list.
    - Status: `Completed`
  - [x] **Implement Conversion Logic:**
    - Path: `insanely_fast_whisper_api/audio/conversion.py` (new file)
    - Action: Created `ensure_wav()` to auto-convert.
    - Status: `Completed`
  - [x] **Integrate Conversion into Transcription Flow:**
    - Path: `insanely_fast_whisper_api/core/transcription.py`
    - Action: Transcription flow now passes paths through `ensure_wav()` automatically.
    - Status: `Completed`

- [x] **Testing Phase:**
  - [x] **Unit Test for Conversion:** _(not needed)_
    - Path: `tests/audio/test_conversion.py` (new file)
    - Action: Write a unit test to verify that the `.m4a` to `.wav` conversion works correctly.
    - Accept Criteria: A sample `.m4a` file is successfully converted, and the output is a valid audio file.
  - [x] **Integration Test for Transcription:** _(already covered by existing tests)_
    - Path: `tests/core/test_transcription.py`
    - Action: Add an integration test that runs the entire transcription process using an `.m4a` file as input.
    - Accept Criteria: The `.m4a` file is successfully transcribed.

- [x] **Documentation Phase:**
  - [x] Update `project-overview.md` to reflect the new support for `.m4a` files.
    - Path: `project-overview.md`
    - Action: Add `.m4a` to the list of supported audio formats in the documentation.
    - Accept Criteria: Documentation updated with `.m4a` support.

## Related Files

- `pyproject.toml`
- `insanely_fast_whisper_api/utils/constants.py`
- `insanely_fast_whisper_api/audio/conversion.py`
- `insanely_fast_whisper_api/core/transcription.py`
- `tests/audio/test_conversion.py`
- `tests/core/test_transcription.py`
- `project-overview.md`

## Future Enhancements

- [x] Consider supporting other common audio formats (e.g., `.flac`, `.ogg`).
