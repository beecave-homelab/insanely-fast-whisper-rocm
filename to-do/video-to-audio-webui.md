# To-Do: Add Video-to-Audio Support in WebUI

This plan outlines the steps to enable video-file upload and conversion (via FFmpeg) within the Gradio WebUI, mirroring the new CLI capability.

## Tasks

- [ ] **Analysis Phase:**
  - [ ] Review current upload & handler flow
    - Path: `insanely_fast_whisper_api/webui/handlers.py`
    - Action: Identify where audio files are validated and passed to the backend.
    - Analysis Results:
      - The `process_transcription_request` function currently validates extensions against `SUPPORTED_AUDIO_FORMATS`, then sends the path to backend.
      - Needs extension of allowed video types and invocation of the new `extract_audio_from_video()` helper.
    - Accept Criteria: Clear plan for injection point and updated validation logic.

- [ ] **Implementation Phase:**
  - [ ] Update accepted file types in Gradio `File` component
    - Path: `insanely_fast_whisper_api/webui/app.py`
    - Action: Add video extensions (mp4, mkv, webm, mov) to `type="file"` accept list.
    - Status: Pending
  - [ ] Extend validation logic
    - Path: `insanely_fast_whisper_api/webui/handlers.py`
    - Action: Use `extract_audio_from_video()` when uploaded file has a video extension; track & cleanup temp WAV after processing.
    - Status: Pending
  - [ ] Ensure existing ZIP/Download workflow unchanged
    - Path: `insanely_fast_whisper_api/webui/handlers.py`
    - Action: Verify filenames stay unique; reuse previous fix for suffixes.
    - Status: Pending

- [ ] **Testing Phase:**
  - [ ] Manual E2E test with small `.mp4` upload
    - Path: `[N/A – manual]`
    - Action: Confirm UI accepts video, processes successfully, and transcripts/ZIPs are correct.
    - Accept Criteria: No errors; output identical to CLI.
  - [ ] Add integration test (optional)
    - Path: `tests/webui/test_video_upload.py`
    - Action: Spin up FastAPI TestClient to POST video, assert 200 + transcript text.
    - Accept Criteria: Test passes reliably.

- [ ] **Documentation Phase:**
  - [ ] Update `project-overview.md` and README
    - Path: `project-overview.md`
    - Action: Note WebUI now supports video uploads analogous to CLI.
    - Accept Criteria: Docs reflect new capability and usage examples.

## Related Files

- `insanely_fast_whisper_api/webui/app.py`
- `insanely_fast_whisper_api/webui/handlers.py`
- `insanely_fast_whisper_api/audio/processing.py`
- `tests/webui/test_video_upload.py`

## Future Enhancements

- [ ] Display preview thumbnail or duration of uploaded video.
- [ ] Allow batch video uploads with combined ZIP download.
