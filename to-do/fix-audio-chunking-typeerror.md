# To-Do: Fix TypeError in Audio Chunking (Float/String Comparison)

This plan outlines the steps to resolve a TypeError in audio chunking, caused by a string/float comparison for chunking parameters passed from the UI to the backend.

## Tasks

- [ ] ~~Analysis Phase:~~
  - [x] Reproduce and analyze bug: TypeError in audio chunking (`core.py`), caused by comparison of `audio_duration` (float) with `effective_chunk_duration` (string from UI input).
    - Path: `[insanely_fast_whisper_api/core.py]`
    - Action: Reproduce bug and trace parameter types through UI/backend
    - Analysis Results:
      - Bug occurs when UI or API passes `chunk_duration`/`chunk_overlap` as strings, causing a TypeError in the backend during float comparison.
    - Accept Criteria: Root cause is documented and code paths identified.

- [x] **Implementation Phase (Revised Approach):**
  - [x] Removed UI components for audio chunking from `insanely_fast_whisper_api/webui.py`.
    - Path: `[insanely_fast_whisper_api/webui.py]`
    - Action: Deleted Gradio Accordion, Checkbox, and Sliders related to UI-configurable audio chunking.
    - Status: Done
  - [x] Removed chunking parameter handling from `process_transcription` in `insanely_fast_whisper_api/webui.py`.
    - Path: `[insanely_fast_whisper_api/webui.py]`
    - Action: Adjusted `args` parsing, `pipeline_call_args` (to pass `None` for chunking params), `config_used_for_store`, and `final_result_for_store` to remove UI-driven chunking logic.
    - Status: Done
  - [x] Updated `submit_btn.click` inputs in `insanely_fast_whisper_api/webui.py`.
    - Path: `[insanely_fast_whisper_api/webui.py]`
    - Action: Removed chunking UI components from the `inputs` list for the main transcription submission.
    - Status: Done

- [ ] ~~Testing Phase:~~
  - [ ] ~~Test UI and backend to confirm bug is fixed and chunking works as expected.~~
    - Path: `[insanely_fast_whisper_api/webui.py]`, `[insanely_fast_whisper_api/core.py]`, `[tests/]`
    - Action: ~~Run UI and API tests for chunked audio uploads, verify no TypeError and correct chunking behavior.~~
    - Accept Criteria: ~~No TypeError, chunking works for both float and string inputs.~~

- [ ] ~~Documentation Phase:~~
  - [ ] ~~Update `project-overview.md` to reflect the bugfix and improved type safety in chunking parameters.~~
    - Path: `[project-overview.md]`
    - Action: ~~Document the bugfix, root cause, and the type safety improvement for chunking parameters.~~
    - Accept Criteria: ~~Documentation is up-to-date and explains the fix clearly.~~

## Related Files

- `insanely_fast_whisper_api/core.py`
- `insanely_fast_whisper_api/webui.py`
- `project-overview.md`
- Any relevant test files in `tests/`

## Future Enhancements

- [ ] ~~Add input validation and type enforcement for all user-supplied numeric parameters at the API/UI boundary.~~
