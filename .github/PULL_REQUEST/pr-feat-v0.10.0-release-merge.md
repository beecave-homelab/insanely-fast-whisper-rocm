# Pull Request: Release v0.10.0

## Summary

This PR introduces a major feature release, bumping the project version to `v0.10.0`. Key additions include the integration of `stable-ts` for highly accurate word-level timestamp stabilization, support for `.m4a` audio files, and the ability to upload video files directly in the WebUI for transcription. The release also includes significant refactoring, dependency updates, and new tests to ensure stability.

---

## Files Changed

### Added

1.  **`insanely_fast_whisper_api/core/integrations/stable_ts.py`**
    -   Adds the core logic for integrating `stable-ts` to stabilize and enhance word-level timestamps.
2.  **`insanely_fast_whisper_api/audio/conversion.py`**
    -   Introduces audio conversion utilities, specifically to handle `.m4a` files using `pydub`.
3.  **`insanely_fast_whisper_api/cli/common_options.py`**
    -   Centralizes CLI options for better reusability and consistency.
4.  **`tests/test_stable_ts.py`**, **`tests/test_webui_handlers.py`**, **`tests/webui/test_video_upload.py`**
    -   New test suites to validate the timestamp stabilization, WebUI video handling, and other new features.
5.  **`to-do/*.md`**
    -   Added several to-do files to track the implementation of new features.

### Modified

1.  **`pyproject.toml`**
    -   Bumped project version to `0.10.0` and added `stable-ts` as a dependency.
2.  **`insanely_fast_whisper_api/core/asr_backend.py`**
    -   Modified to conditionally apply `stable-ts` stabilization based on the `--stabilize` flag.
3.  **`insanely_fast_whisper_api/webui/handlers.py`**
    -   Updated to handle video file uploads by extracting the audio stream before transcription.
4.  **`README.md`**, **`VERSIONS.md`**, **`project-overview.md`**
    -   Updated documentation and changelogs to reflect the new version and features.
5.  **Numerous other files**
    -   Widespread refactoring and updates across the API, CLI, and core components to support the new features and improve code quality.

### Deleted

1.  **`requirements-onnxruntime-rocm.txt`**
    -   Removed obsolete requirements file as dependencies are managed via `pyproject.toml`.

---

## Code Changes

### `insanely_fast_whisper_api/core/asr_backend.py`

```python
# Snippet showing stable-ts integration
if transcribe_options.get("stabilize", False):
    from insanely_fast_whisper_api.core.integrations.stable_ts import (
        stabilize_timestamps,
    )

    result = stabilize_timestamps(result, model)
```

-   This change conditionally imports and applies `stable-ts` to the transcription result, allowing users to enable it via a flag for more precise word timings.

### `pyproject.toml`

```toml
[tool.pdm.project]
name = "insanely-fast-whisper-api"
version = "0.10.0"
description = "🔥 Insanely fast Whisper API supporting distil-whisper. Transcribe 1 hour of audio in less than 98 seconds."

...

[tool.pdm.dependencies]
stable-ts = {version = ">=1.1.1", optional = true}
```

-   The project version is officially bumped to `0.10.0`, and `stable-ts` is added as an optional dependency.

---

## Reason for Changes

This pull request merges several major feature enhancements from the `dev` branch into `main` to prepare for a new public release. The primary goals are to improve the accuracy of timestamps and expand supported media formats, making the tool more versatile.

---

## Impact of Changes

### Positive Impacts

-   **Enhanced Timestamp Accuracy**: Users can now get highly reliable word-level timestamps using `stable-ts`.
-   **Broader File Support**: Added support for `.m4a` audio and video files increases the tool's utility.
-   **Improved User Experience**: The WebUI is now more powerful, accepting video uploads directly.

### Potential Issues

-   **New Dependencies**: The addition of `pydub` (for M4A) and `stable-ts` introduces new dependencies that must be installed. These are handled via `pyproject.toml`.

---

## Test Plan

1.  **Unit & Integration Testing**
    -   New tests have been added in `tests/test_stable_ts.py` and `tests/test_webui_handlers.py` to cover the new functionality.
    -   Existing tests were updated to align with the refactored codebase.
2.  **Manual Testing**
    -   **CLI**: Run transcription with the `--stabilize` flag on an audio file and verify word timestamps in the output.
    -   **WebUI**: Upload an `.mp4` video file and confirm that it is transcribed successfully.
    -   **API**: Send a request with the `stabilize: true` option and check the response for word-level timestamps.

---

## Additional Notes

This release marks a significant step forward in functionality. Once merged, the `main` branch will be tagged as `v0.10.0`.
