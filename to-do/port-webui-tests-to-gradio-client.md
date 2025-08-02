# To-Do: Port WebUI Tests to `gradio_client`

This plan outlines the steps to migrate existing WebUI integration tests from raw `requests` calls to the official `gradio_client` helper, ensuring robust file–upload handling and easier maintenance.

## Tasks

- [ ] **Analysis Phase:**
  - [ ] Investigate current Gradio version & API exposure
    - Path: `insanely_fast_whisper_api/webui/ui.py`, `tests/test_webui.py`
    - Action: Confirm the exact `api_name` values, component ordering, and server start-up requirements.
    - Analysis Results:
      - `api_name` for transcription is `/transcribe_audio_v2` (matches Gradio recorder output)
      - Component input order for `Client.predict()`:
        1. `audio_input` (List[str] filepaths)
        2. `model` (str)
        3. `device` (str)
        4. `batch_size` (int)
        5. `timestamp_type` (str)
        6. `language` (str)
        7. `task` (str)
        8. `dtype` (str)
        9. `chunk_length` (int)
        10. `save_transcriptions` (bool)
        11. `temp_uploads_dir` (str)
      - Gradio version pinned in `pyproject.toml`: `gradio>=5.20.1` → compatible with `gradio_client>=0.7.0`.
      - `gradio_client.handle_file()` must be used for audio/video uploads.
    - Accept Criteria: Clear mapping between component inputs and `Client.predict()` kwargs.

- [x] **Implementation Phase:**
  - [x] Add `gradio-client` dependency via PDM
    - Path: `pyproject.toml`
    - Action: Run `pdm add gradio-client && pdm lock && pdm sync`
    - Status: Done
  - [x] Add/modify test utilities to spin up WebUI once per session
    - Status: Done
    - Path: `tests/conftest.py`
    - Action: Factor current `setup_module` logic into a pytest fixture (`webui_server`).
  - [x] Rewrite transcription tests to use `gradio_client`
    - Path: `tests/webui/test_webui.py`
    - Action: Completed — tests now use `Client.predict()` + `handle_file()` and hard-coded payloads removed.
  - [x] Add new video upload integration test
    - Path: `tests/webui/test_video_upload.py`
    - Action: Implemented; skips if sample video missing.

- [x] **Testing Phase:**
  - Path: `[local]`
  - Result: Tests executed via `pytest -m webui` — 1 passed, others skipped due to missing media; performance < 10 s.

- [x] **Documentation Phase:** *(Next)*
  - [x] Update `project-overview.md` testing section
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
