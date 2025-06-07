# To-Do: Refactor Gradio WebUI into Modular Package

This plan outlines the steps to refactor the large `webui.py` file into a clean, modular Python package for improved maintainability, testability, and code clarity.

## Tasks

- [x] **Analysis Phase:**
  - [x] Review current Gradio WebUI implementation and supporting modules
    - Path: `webui.py`, `core.py`, `download_hf_model.py`
    - Action: Analyze current structure, data flow, error handling, and duplication
    - Analysis Results:
      - Single monolithic file (~820 LoC) contains UI, logic, formatters, and state management
      - Redundant code in export functions and error guards
      - Defensive tuple handling no longer needed
      - Formatting logic belongs in separate classes
      - Logging and download code can be relocated
    - Accept Criteria: Clear list of responsibilities to isolate into modules
    - Status: Completed

- [x] **Implementation Phase:**
  - [x] Create new package layout
    - Path: `webui/`
    - Action: Add files `__init__.py`, `ui.py`, `handlers.py`, `formatters.py`, `utils.py`, `errors.py`, `cli.py`
    - Status: Completed

  - [x] Move and refactor formatter classes
    - Path: `webui/formatters.py`
    - Action: Relocate `BaseFormatter`, `TxtFormatter`, `SrtFormatter`, `JsonFormatter` from `webui.py`
    - Status: Completed

  - [x] Move and simplify utility functions
    - Path: `webui/utils.py`
    - Action: Move `save_temp_file`, `format_seconds`, and `convert_device_string` from `webui.py` and `core.py`
    - Status: Completed

  - [x] Create `handlers.py` for core UI logic
    - Path: `webui/handlers.py`
    - Action: Implement `transcribe()` and `export()` functions; remove tuple guards and file re-open logic
    - Status: Completed

  - [x] Refactor error classes into dedicated file
    - Path: `webui/errors.py`
    - Action: Move `TranscriptionError`, `DeviceNotFoundError` from `core.py`
    - Status: Completed

  - [x] Build new UI definition
    - Path: `webui/ui.py`
    - Action: Rebuild Gradio interface with clean layout and flat component tree
    - Status: Completed

  - [x] Simplify CLI entrypoint
    - Path: `webui/cli.py`
    - Action: Create launcher using `demo.launch(server_name="0.0.0.0")`
    - Status: Completed

- [ ] **Testing Phase:**
  - [ ] Write or migrate unit tests for moved modules
    - Path: `tests/test_handlers.py`, `tests/test_formatters.py`, `tests/test_utils.py`
    - Action: Verify exports, transcription results, and error handling
    - Accept Criteria: All core functionality is covered with automated tests

- [ ] **Documentation Phase:**
  - [ ] Update `README.md` and developer setup docs
    - Path: `README.md`
    - Action: Explain new structure, how to run UI, and where to contribute
    - Accept Criteria: Clear, up-to-date guide for new contributors

## Related Files

- `webui.py` (to be deprecated)
- `webui/ui.py`
- `webui/handlers.py`
- `webui/formatters.py`
- `webui/utils.py`
- `webui/errors.py`
- `webui/cli.py`
- `core.py`
- `download_hf_model.py`

## Future Enhancements

- [ ] Add `progress_cb` hook to `ASRPipeline` for real-time chunked transcription feedback
- [ ] Add config loading from file or env for defaults (e.g., model name, device)
- [ ] Add session history and in-browser playback of transcriptions
