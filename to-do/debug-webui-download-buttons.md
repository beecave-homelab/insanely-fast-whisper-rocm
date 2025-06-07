# To-Do: Debug WebUI Download Buttons

This plan outlines the steps to debug why the download buttons (TXT, SRT, JSON) in `insanely_fast_whisper_api/webui.py` do not provide a file for download after a successful conversion.

## Tasks

- [x] **Analysis Phase:**
  - [x] Review `insanely_fast_whisper_api/webui.py` to understand how download functionality is implemented for TXT, SRT, and JSON formats.
    - Path: `insanely_fast_whisper_api/webui.py`
    - Action: Identify the Gradio components responsible for the download buttons and their associated callback functions. Trace the data flow from transcription result to file generation and download trigger.
    - Analysis Results:
      - [x] Identified key functions: `create_ui_components.export_as_format`, `export_transcription`, `save_temp_file`, and individual formatters (`TxtFormatter`, `SrtFormatter`, `JsonFormatter`).
      - [x] Gradio `gr.File` components (`txt_file`, `srt_file`, `json_file`) are used as outputs for button click events.
      - [x] A `gr.State` component (`result_store`) holds the transcription data for export.
      - [ ] ~~JavaScript console errors or backend errors when buttons are clicked (User reported issue; direct check pending if code review is inconclusive).~~
    - Accept Criteria: Clear understanding of the current download mechanism and potential points of failure.

- [x] **Debugging Phase (Code Review):**
  - [x] Investigated `process_transcription` and `transcribe_with_pipeline`.
    - Path: `insanely_fast_whisper_api/webui.py`
    - Action: Found that `chunks` data from `ASRPipeline` was not being propagated to `result_store`, affecting SRT and JSON exports.
    - Status: `Completed`
  - [x] Add logging statements to trace execution flow and variable states if downloads still fail.
    - Path: `insanely_fast_whisper_api/webui.py`
    - Action: Replaced all print statements with logging calls in export_transcription, save_temp_file, export_as_format, export_txt, export_srt, and export_json. Logger is now used consistently throughout download/export logic.
    - Status: Active task
    - Note: Formatting/syntax error found and fixed in save_temp_file (docstring properly closed, implementation restored, blank line added).
    - Note: Logging is now consistent; next step is to test the download buttons and check logs for issues.
  - [x] Code review confirms current download/export logic uses in-memory results, not the `.json` file.
    - Path: `insanely_fast_whisper_api/webui.py`
    - Action: The export/download functions currently operate on the in-memory transcription result. This can cause failures if the Gradio state is not correctly populated. To improve robustness, the logic should be refactored to always read from the saved `.json` file in `transcripts/` for TXT and SRT downloads.
    - Status: Completed (Code Review)
  - [x] Refactor export/download functions to read from the corresponding `.json` file in `transcripts/` for TXT and SRT downloads.
    - Path: `insanely_fast_whisper_api/webui.py`
    - Action: Update download/export logic to load the `.json` result from disk (using 'output_file_path' from result_store), then convert to the requested format on-demand for download.
    - Status: Completed
  - [x] Verify that `webui.py` correctly uses the `output_file_path` from `result_store` to generate download links for TXT, SRT, and JSON files.
  - **Confirmation**: `webui.py` logic appears correct and relies on `output_file_path` being present in `result_store`. The fix in `core.py` should address this.
  - [ ] ~~After refactor, verify that all download buttons work even if the in-memory result is unavailable.~~
    - Path: WebUI interface
    - Action: Perform a transcription, clear/reset the UI state if possible, and attempt to download results. All buttons should work as long as the `.json` file exists.
    - Status: Pending

- [x] **Implementation Phase (Fixes):**
  - [x] Modified `transcribe_with_pipeline` to return the full `ASRPipeline` result dictionary.
    - Path: `insanely_fast_whisper_api/webui.py`
    - Action: Changed return type and returned the raw dictionary from `asr_pipeline()` call.
    - Status: `Completed`
  - [x] Corrected duplicated assignment in `export_json_file`.
  - [x] Added detailed logging in `process_transcription` for `asr_output_dict` and `final_result_for_store` to trace `output_file_path` (refined f-string formatting and removed duplicates).
  - [x] Analyze logs after user testing to pinpoint where `output_file_path` becomes a dictionary or is mishandled.
    - Path: `insanely_fast_whisper_api/webui.py`
    - Action: Reviewed logs. Identified that `output_file_path` was not being added to the result dict in `ASRPipeline.__call__`.
    - Status: `Completed`
  - [x] Investigate `ASRPipeline` in `core.py` to see if `output_file_path` is consistently returned, especially with chunking.
  - **Observation**: `ASRPipeline.__call__` did not add `output_file_path` to the result dict when chunking was enabled and `save_transcriptions` was true.
  - **Fix**: Modified `insanely_fast_whisper_api/core.py` in `ASRPipeline.__call__` to add the `output_file_path` (obtained from saving the combined transcription) to the returned `result` dictionary. This ensures `webui.py` receives the necessary path for download functions.
    - Path: `insanely_fast_whisper_api/core.py`
    - Action: Captured the return value of `_save_transcription_result` and added it to the `result` dictionary with the key `output_file_path`.
    - Status: `Completed`
  - [x] Analyze new logs after user testing to confirm `output_file_path` is correctly propagated and used.
    - Path: `insanely_fast_whisper_api/webui.py`, `insanely_fast_whisper_api/core.py`
    - Action: Reviewed logs. Confirmed `output_file_path` is now correctly passed from `ASRPipeline` to `webui` export functions. JSON download works without fallback.
    - Status: `Completed`
  - [x] Test TXT and SRT download buttons in the WebUI.
    - Path: WebUI interface
    - Action: Performed a transcription and successfully downloaded TXT and SRT files. Logs confirm `output_file_path` is correctly used.
    - Status: `Completed`
  - [x] Modified `create_ui_components.process_transcription` to correctly populate `result_store`.
    - Path: `insanely_fast_whisper_api/webui.py`
    - Action: Used the full result from `transcribe_with_pipeline` to include `text`, `chunks`, and other necessary data in the dictionary stored in `result_store`. Constructed `config_used` within this function.
    - Status: `Completed`
  - [x] After successful transcription/translation, automatically convert the saved `.json` transcript in `transcripts/` to both `.srt` and `.txt` files and save them in the same directory.
    - Path: `insanely_fast_whisper_api/webui.py`
    - Action: Update the post-processing logic to trigger both conversions and save all three formats.
    - Status: Completed

  - [ ] ~~Fix bug: Handle both tuple and dict return values from ASRPipeline in `transcribe_with_pipeline` to avoid `'tuple' object has no attribute 'get'` error.~~
    - ~~Path: `insanely_fast_whisper_api/webui.py` (function: `transcribe_with_pipeline`)~~
    - ~~Action: Add a type check and extract the dict if a tuple is returned before further processing.~~
    - ~~Status: Completed~~
    - ~~Status: Pending~~
  - [x] Change the download button logic to use the locally saved `.json` file for conversion to `.txt` and `.srt` (instead of using the in-memory result).
    - Path: `insanely_fast_whisper_api/webui.py`
    - Action: Refactor the export/download functions to read from the corresponding `.json` file in `transcripts/` and convert it on-demand.
    - Status: Completed
  - [x] Review Gradio 5.20.1 File component documentation.
    - Path: <https://www.gradio.app/docs/gradio/file>
    - Action: Confirm requirements for making a file downloadable: the function must return a file path (str), and the output component must be `gr.File`.
    - Status: Completed
  - [x] Verify and fix Gradio download button wiring and outputs.
    - Path: `insanely_fast_whisper_api/webui.py`
    - Action: Modified `export_txt_from_json`, `export_srt_from_json`, and `export_json_file` to return file paths directly, resolving `AttributeError: type object 'File' has no attribute 'update'`. Added/updated checks for `transcription_result_data is None` to raise `gr.Error("Please run a transcription first.")`. Reverted `json_btn.click` handler to call `export_json_file` with `inputs=[result_store]`. Confirmed `dummy_export_json` was already removed.
    - Status: Completed
  - [x] Test the UI after wiring changes.
    - Path: WebUI interface
    - Action: Clicked each download button and confirmed that the browser prompts a file download for each format (TXT, SRT, JSON).
    - Status: Completed
  - [x] ~~Fix file visibility in the UI by updating export functions to handle visibility flags.~~
    - Path: `insanely_fast_whisper_api/webui.py`
    - Action: ~~Modified export functions to return both file path and visibility flag. Updated click handlers to use `.then()` for updating file component visibility.~~
    - Status: ~~Completed~~ Superseded by direct `gr.File.update()` return.
  - [x] Resolve `TypeError` and `ValueError` in download handlers.
    - Path: `insanely_fast_whisper_api/webui.py`
    - Action: Corrected export functions (`export_txt_from_json`, `export_srt_from_json`, `export_json_file`) to return a new Gradio File component instance (e.g., `gr.File(value=file_path, visible=True)`) instead of attempting to call `gr.File.update()`. This resolves the `AttributeError`. The `.then()` clauses remain removed as the export functions directly manage the `gr.File` component's state and visibility.
    - Status: Completed
  - [x] Update documentation if download mechanism or UI structure changes.
    - Path: `project-overview.md`
    - Action: Documented changes to download logic and UI components.
    - Status: Completed

- [x] **Error Handling Consistency:**
  - [x] Ensure `transcribe_with_pipeline` always returns a dictionary (never a tuple), even on error. Update `process_transcription` to handle an `"error"` key in the result dict and display appropriate messages in the UI.
    - Path: `insanely_fast_whisper_api/webui.py` (functions: `transcribe_with_pipeline`, `process_transcription`)
    - Action: Refactor error/exception handling so that all return values are dicts with consistent keys. Update UI logic to display error messages if present.
    - Accept Criteria: No `'tuple' object has no attribute 'get'` errors; robust error handling and clear UI error messages.

- [ ] ~~Formatting & Syntax Fixes:~~
  - [ ] ~~Fix the `save_temp_file` function formatting error (docstring closure, implementation, error handling).~~
    - Path: ~~`insanely_fast_whisper_api/webui.py` (function: `save_temp_file`)~~
    - Action: ~~Properly close the docstring, implement the function to save content to a temporary file, and add error handling/logging. Ensure the function handles different file extensions securely.~~
    - Note: ~~Check for any similar formatting or structural issues elsewhere in the file.~~
    - Accept Criteria: ~~No syntax errors; function works as intended.~~

- [x] **Testing Phase:**
  - [x] Test the download functionality thoroughly in the WebUI.
    - Path: WebUI interface
    - Action: Performed a transcription and successfully downloaded results in TXT, SRT, and JSON formats. Verified that the downloaded files are correct and complete.
    - Accept Criteria: All three download buttons (TXT, SRT, JSON) successfully provide the correct transcription file for download.
    - Status: Completed
  - [x] Test the download buttons for TXT, SRT, and JSON formats in the WebUI after changes.
    - Path: WebUI interface
    - Action: Performed multiple transcriptions and verified that all download buttons work consistently, even after UI state changes.
    - Accept Criteria: All three download buttons (TXT, SRT, JSON) successfully provide the correct transcription file for download.
    - Status: Completed

- [ ] ~~Documentation Phase:~~
  - [ ] ~~Update `project-overview.md` if any significant changes to the WebUI's functionality or structure are made.~~
    - Path: `project-overview.md`
    - Action: ~~Document any changes to download logic or UI components.~~
    - Status: ~~Pending~~
    - Accept Criteria: ~~Documentation accurately reflects the corrected functionality.~~

## Related Files

- `insanely_fast_whisper_api/webui.py`
- Potentially any utility files used for formatting outputs (e.g., SRT, JSON).

## Future Enhancements

- [ ] [Optional: Consider adding a "Download All" button if deemed useful]
