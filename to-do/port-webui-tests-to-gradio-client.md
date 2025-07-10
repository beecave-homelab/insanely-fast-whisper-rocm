# To-Do: Port WebUI Tests to `gradio_client`

This plan outlines the steps to migrate existing WebUI integration tests from raw `requests` calls to the official `gradio_client` helper, ensuring robust file–upload handling and easier maintenance.

## Tasks

- [ ] **Analysis Phase:**
  - [ ] Investigate current Gradio version & API exposure
    - Path: `insanely_fast_whisper_api/webui/ui.py`, `tests/test_webui.py`
    - Action: Confirm the exact `api_name` values, component ordering, and server start-up requirements.
    - Analysis Results:
      - `api_name` for transcription is `/transcribe_audio_v2` (matches Gradio recorder output)
      - `gradio_client.handle_file()` must be used for audio/video uploads.
    - Accept Criteria: Clear mapping between component inputs and `Client.predict()` kwargs.

- [ ] **Implementation Phase:**
  - [ ] Add/modify test utilities to spin up WebUI once per session
    - Path: `tests/conftest.py`
    - Action: Factor current `setup_module` logic into a pytest fixture (`webui_server`).
    - Status: Pending
  - [ ] Rewrite transcription tests to use `gradio_client`
    - Path: `tests/webui/test_webui.py`
    - Action:
      1. Replace manual `requests.post` calls with `Client.predict()`.
      2. Use `handle_file()` for uploads.
      3. Remove hard-coded JSON payloads.
    - Status: Pending
  - [ ] Add new video upload integration test
    - Path: `tests/webui/test_video_upload.py`
    - Action: Use small `.mp4` sample + assert transcript contains words.
    - Status: Pending

- [ ] **Testing Phase:**
  - [ ] Run full test suite with `pytest -q tests/webui`  
    - Note: `gradio_client>=0.7.0` must be available; flag tests with `@pytest.mark.webui` so they can be selectively run in CI.
    - Path: `[N/A – CI]`
    - Action: Ensure all WebUI tests pass in <60 s using `openai/whisper-tiny`.
    - Accept Criteria: 0 failed, 0 skipped.

- [ ] **Documentation Phase:** *(Next)*
  - [ ] Update `project-overview.md` testing section
    - Path: `project-overview.md`
    - Action: Document use of `gradio_client` and new fixture.
    - Accept Criteria: Clear developer instructions for running WebUI tests.

## Related Files

- `tests/test_webui.py`
- `tests/webui/test_video_upload.py`
- `tests/conftest.py`
- `insanely_fast_whisper_api/webui/ui.py`

## Future Enhancements

- [ ] Parameterize model names via environment variable for faster CI runs.
- [ ] Explore asynchronous `client.submit()` to reduce total runtime.
