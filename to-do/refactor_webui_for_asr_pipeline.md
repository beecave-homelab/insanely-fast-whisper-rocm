# To-Do: Refactor WebUI to Use ASRPipeline Directly

This plan outlines the steps to refactor the WebUI to use the `ASRPipeline` class from `core.py` directly, rather than making API calls to a separate server. This will simplify the architecture and improve performance by removing the HTTP overhead.

## Tasks

- [x] **Analysis Phase:**
  - [x] Analyze `insanely_fast_whisper_api/webui.py`
    - Path: `insanely_fast_whisper_api/webui.py`
    - Action: Review current implementation, identify UI components, and understand the API interaction flow.
    - Findings: The current implementation makes HTTP requests to a local API endpoint. It needs to be refactored to use ASRPipeline directly.
  - [x] Analyze `insanely_fast_whisper_api/core.py`
    - Path: `insanely_fast_whisper_api/core.py`
    - Action: Review `ASRPipeline` class interface, available methods, and configuration options.
    - Findings: The ASRPipeline class provides a comprehensive interface with support for various configurations including model, device, batch size, language, and more.
  - [x] Analyze `insanely_fast_whisper_api/constants.py`
    - Path: `insanely_fast_whisper_api/constants.py`
    - Action: Identify all configurable parameters and their default values.
    - Findings: Default values are defined for model, device, batch size, timestamp type, and language. Also includes limits for batch size and supported audio formats.

- [ ] **Implementation Phase:**
  - [x] Update WebUI Dependencies
    - Path: `insanely_fast_whisper_api/webui.py`
    - Action:
      - [x] Add imports for `ASRPipeline` and other required components
      - [x] Remove unused `requests` import
      - [x] Import additional constants as needed
  - [x] Refactor Transcribe Functionality
    - Path: `insanely_fast_whisper_api/webui.py`
    - Action:
      - [x] Create a new function `transcribe_with_pipeline` that uses `ASRPipeline`
      - [x] Initialize `ASRPipeline` with appropriate parameters
      - [x] Call the pipeline with the audio file and return results in the expected format
  - [x] Update UI Components
    - Path: `insanely_fast_whisper_api/webui.py`
    - Action: Add UI controls for all `ASRPipeline` parameters:
      - [x] Model configuration (model, device, batch size)
      - [x] Add processing options:
        - [x] dtype (float16/float32) dropdown
        - [x] better_transformer checkbox
        - [x] chunk_length slider
      - [x] Add file handling options:
        - [x] save_transcriptions checkbox
        - [x] temp_uploads_dir input (with default from constants)
      - [x] Diarization options (deferred for future implementation)
  - [x] Implement File Operations
    - Path: `insanely_fast_whisper_api/webui.py`
    - Action:
      - [x] Add logic to handle file saving when `save_transcriptions` is enabled
      - [x] Ensure proper directory structure exists
  - [x] Add Error Handling and Validation
    - Path: `insanely_fast_whisper_api/webui.py`
    - Action:
      - [x] Add input validation for all parameters
      - [x] Implement user-friendly error messages
      - [x] Add try-catch blocks for pipeline execution

  - [x] Add Export Functionality
    - Path: `insanely_fast_whisper_api/webui.py`
    - Action:
      - [x] Create export formatters for different file types:
        - [x] TXT formatter (basic text output)
        - [x] SRT formatter (SubRip subtitles with timestamps)
        - [x] JSON formatter (full structured data)
      - [x] Add export buttons to the UI:
        - [x] Download as TXT
        - [x] Download as SRT
        - [x] Download as JSON
      - [x] Implement file generation:
        - [x] Generate TXT file with transcription text
        - [x] Generate SRT file with timestamps
        - [x] Generate JSON file with full transcription data
      - [x] Add file handling:
        - [x] Create temporary files for download
        - [x] Clean up temporary files after download
        - [x] Handle file naming with timestamps
      - [x] Add user feedback:
        - [x] Show success/error messages
        - [x] Disable buttons during export
        - [x] Show progress indicators

- [ ] **Testing Phase:**
  - [ ] Manual Testing
    - Action: Test with various audio files and parameter combinations.
  - [ ] Error Case Testing
    - Action: Test error handling with invalid inputs and edge cases.
  - [ ] Docker Configuration Updates
    - [ ] Update `docker-compose.yaml` to include WebUI port mapping
      - Path: `docker-compose.yaml`
      - Action: Add port mapping for WebUI (default port 7860)
    - [ ] Add WebUI volume mount for development
      - Path: `docker-compose.yaml`
      - Action: Ensure WebUI files are mounted in development mode
    - [ ] Update default command to run WebUI
      - Path: `docker-compose.yaml`
      - Action: Update command to `python -m insanely_fast_whisper_api.webui`
  - [ ] WebUI Testingtes
    - [ ] Test TXT export functionality
    - [ ] Test SRT export functionality
    - [ ] Test JSON export functionality
    - [ ] Verify file downloads work in container environment
    - [ ] Test with different audio formats and lengths
    - [ ] Verify error handling for large files
  - [ ] Documentation
    - [ ] Update README with WebUI usage instructions
    - [ ] Add docker-compose example for WebUI deployment
    - [ ] Document any additional environment variables for WebUI
  - [ ] Performance Testing
    - Action: Compare performance with the previous API-based implementation.

- [ ] **Documentation Phase:**
  - [ ] Update Code Documentation
    - Path: `insanely_fast_whisper_api/webui.py`
    - Action: Add/update docstrings and comments.
  - [ ] Update User Documentation
    - Action: Document new UI features and parameters.
  - [ ] Update `project-overview.md`
    - Path: `project-overview.md`
    - Action: Document the WebUI refactoring and direct ASRPipeline integration.

## Related Files

- `insanely_fast_whisper_api/webui.py`
- `insanely_fast_whisper_api/core.py`
- `insanely_fast_whisper_api/constants.py`
- `project-overview.md`

## Future Enhancements

- [ ] Add support for speaker diarization when available
- [ ] Implement batch processing of multiple files
- [ ] Add more advanced audio processing options
- [ ] Support for live audio input
- [ ] Add support for more export formats (VTT, WebVTT, etc.)
- [ ] Implement batch export functionality
- [ ] Add option to customize SRT formatting (line length, characters per line)
- [ ] Add export templates for different use cases
