# To-Do: Implement Audio Chunking for Long Recordings

This plan outlines the implementation of an audio chunking feature that will automatically split long audio files into manageable segments (e.g., 10 minutes each) for processing.

## Tasks

- [x] **Analysis Phase:**
  - [x] Research audio processing libraries for Python that can handle audio file splitting
    - Path: `insanely_fast_whisper_api/audio_utils.py`
    - Action: Evaluate `pydub`, `librosa`, and `pysndfile` for audio splitting capabilities, format support, and performance.
    - Analysis Results:
      - **pydub**:
        - Pros: Simple high-level API, uses ffmpeg (already in our dependencies), lightweight, actively maintained
        - Cons: Limited advanced audio processing (not needed for our use case)
      - **librosa**:
        - Pros: Advanced audio analysis features
        - Cons: More complex API, heavier dependency, overkill for simple splitting
      - **pysndfile**:
        - Pros: Direct bindings to libsndfile
        - Cons: Lower-level API, less actively maintained
      - **Selected**: `pydub` - Best fit due to simplicity, ffmpeg support, and maintenance status
    - Accept Criteria: Selected library with best balance of features, performance, and maintenance status.
  - [x] Analyze the current `ASRPipeline` class
    - Path: `insanely_fast_whisper_api/core.py`
    - Action: Review how to integrate chunking into the existing pipeline while maintaining current functionality.
    - Analysis Results:
      - Current chunking in `chunk_length` is for model processing, not file splitting
      - Main integration point is the `__call__` method
      - Need to add chunking logic before `run_asr_pipeline`
      - Must handle result aggregation and cleanup
      - Existing error handling should be preserved
      - Performance considerations for large files
    - Accept Criteria: Documented integration points and any required modifications to the pipeline.
  
  - [x] Configure chunking settings in constants
    - Path: `insanely_fast_whisper_api/constants.py`
    - Action: Added default values for chunking configuration
      - `AUDIO_CHUNK_DURATION`: Default 600 (10 minutes in seconds)
      - `AUDIO_CHUNK_OVERLAP`: Default 1.0 (1 second overlap between chunks)
      - `AUDIO_CHUNK_MIN_DURATION`: Default 5.0 (minimum chunk duration in seconds)
    - Status: Implemented with environment variable fallbacks

  - [x] Update environment configuration
    - Path: `.env.example` and `.env`
    - Action: Added new configuration variables with documentation

      ```ini
      # Audio Chunking Configuration
      # Maximum duration of each audio chunk in seconds (default: 600 = 10 minutes)
      AUDIO_CHUNK_DURATION=600
      # Overlap duration between chunks in seconds (default: 1.0)
      AUDIO_CHUNK_OVERLAP=1.0
      # Minimum duration of a chunk in seconds (default: 5.0)
      AUDIO_CHUNK_MIN_DURATION=5.0
      ```

    - Status: Implemented with proper documentation
  - [x] Determine optimal chunking parameters
    - Action: Research best practices for chunk size (default: 10 minutes) and make it configurable. Consider memory usage and performance implications.
    - Analysis Results:
      - Default chunk size: 10 minutes (600s) - balances processing efficiency and memory usage
      - 1-second overlap between chunks to prevent word cutting
      - Minimum chunk duration of 5 seconds to avoid very small chunks
      - All parameters should be configurable via environment variables
    - Accept Criteria: Defined default chunk size and configuration options with rationale.

- [x] **Implementation Phase:**
  - [x] Create audio utility module
    - Path: `insanely_fast_whisper_api/audio_utils.py`
    - Action: Implemented core functions:
      - `get_audio_duration(audio_path: str) -> float`
      - `split_audio(audio_path: str, chunk_duration: int = 600, chunk_overlap: float = 1.0) -> List[str]`
      - `cleanup_temp_files(file_paths: List[str])`
      - `merge_chunk_results(chunk_results: List[dict]) -> dict`
    - Status: Implemented with comprehensive error handling and format support using pydub/ffmpeg.
  - [x] Update WebUI components
    - Path: `insanely_fast_whisper_api/webui.py`
    - Action:
      - Added chunking toggle and duration/overlap inputs
      - Implemented progress tracking UI
      - Added chunk processing status display
    - Status: Implemented with intuitive UI controls and real-time feedback.
  - [x] Modify transcription pipeline
    - Path: `insanely_fast_whisper_api/core.py`
    - Action: Updated `ASRPipeline` with:
      - New chunking parameters in `__init__`
      - Chunking logic in `__call__`
      - New `_process_single_chunk` helper method
    - Status: Seamless integration with backward compatibility maintained.

- [ ] **Testing Phase:**
  - [ ] Unit tests for audio utilities
    - Path: `tests/test_audio_utils.py`
    - Action: Test audio splitting, duration detection, and cleanup functions with various file formats.
    - Analysis Results:
    - Accept Criteria: Comprehensive test coverage with >90% code coverage for audio utilities.
  - [ ] Integration testing
    - Path: `tests/test_webui.py`
    - Action: Test chunking functionality through the WebUI interface.
    - Analysis Results:
    - Accept Criteria: Verified end-to-end functionality with various audio files and chunking configurations.
  - [ ] Performance testing
    - Action: Measure processing time and memory usage with different chunk sizes and audio formats.
    - Analysis Results:
    - Accept Criteria: Documented performance characteristics and optimal configurations for different scenarios.
  - [ ] Error case testing
    - Action: Verify proper error handling for corrupted files, unsupported formats, and interrupted processing.
    - Analysis Results:
    - Accept Criteria: Robust error handling with clear user feedback for all failure scenarios.

- [ ] **Documentation Phase:**
  - [ ] Update `project-overview.md`
    - Path: `project-overview.md`
    - Action: Document the new chunking feature and its configuration options.
    - Analysis Results:
    - Accept Criteria: Clear documentation of chunking capabilities and how they integrate with existing features.
  - [ ] Add API documentation
    - Path: `insanely_fast_whisper_api/audio_utils.py`
    - Action: Add comprehensive docstrings and usage examples.
    - Analysis Results:
    - Accept Criteria: Self-documenting code with examples for all public functions and classes.
  - [ ] Update WebUI documentation
    - Path: `README.md`
    - Action: Add section explaining chunking features and configuration.
    - Analysis Results:
    - Accept Criteria: User-friendly guide for configuring and using the chunking feature in the WebUI.

## Related Files

- `insanely_fast_whisper_api/webui.py`
- `insanely_fast_whisper_api/core.py`
- `insanely_fast_whisper_api/audio_utils.py` (to be created)
- `project-overview.md`
- `tests/test_audio_utils.py` (to be created)

## Future Enhancements

- [ ] Parallel processing of chunks
- [ ] Smart chunking at natural pauses
- [ ] Individual chunk export options
- [ ] Resume interrupted transcriptions
- [ ] Cloud storage integration for chunked files
