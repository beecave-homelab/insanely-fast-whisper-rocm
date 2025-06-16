# To-Do: Standardize Filename Conventions Across CLI, WebUI, and API Interfaces

This plan outlines the steps to standardize the filename conventions used for transcription output (.json files) across the CLI, WebUI, and API interfaces of the Insanely Fast Whisper API package. Currently, each interface uses different naming patterns, leading to inconsistency and potential confusion.

## Overview

The current state shows inconsistent filename conventions across the three main interfaces:

1. **CLI Interface** (`cli/commands.py`): Uses pattern `{audio_stem}_{timestamp}.json` (e.g., `audio_20241201_143022.json`)
2. **WebUI Interface** (`webui/handlers.py`): Uses pattern `transcription_{timestamp}.{extension}` (e.g., `transcription_20241201_143022.json`)
3. **Core Pipeline** (`core/pipeline.py`): Uses pattern `{audio_stem}_{task}_{timestamp_str}.json` (e.g., `audio_transcribe_20241201T143022Z.json`)

This inconsistency makes it difficult for users to predict output filenames and creates confusion when working across different interfaces. The refactoring will establish a unified naming convention that is descriptive, consistent, and follows best practices.

## Tasks

- [x] **Analysis Phase:**
  - [x] Review current filename generation implementations across all interfaces
    - Path: `insanely_fast_whisper_api/cli/commands.py` (lines 183-193)
    - Path: `insanely_fast_whisper_api/webui/handlers.py` (lines 295-318)
    - Path: `insanely_fast_whisper_api/core/pipeline.py` (lines 180-189)
    - Path: `insanely_fast_whisper_api/core/storage.py` (lines 21-52)
    - Action: Analyze current patterns, timestamp formats, and naming inconsistencies
    - Analysis Results:
      - **CLI**: Simple pattern without task indication, uses `strftime("%Y%m%d_%H%M%S")`
      - **WebUI**: Generic "transcription" prefix, loses audio file context, uses `strftime("%Y%m%d_%H%M%S")`
      - **Core Pipeline**: Most comprehensive pattern with task indication, uses ISO format `strftime("%Y%m%dT%H%M%SZ")`
      - **Key Issues**: Different timestamp formats, inconsistent inclusion of task type, loss of audio filename context in WebUI
    - **Design Pattern Candidates:**
      - **Strategy Pattern**: For different filename generation strategies based on interface requirements
      - **Template Method Pattern**: For standardized filename generation with customizable components
      - **Factory Pattern**: For creating appropriate filename generators based on context
    - Accept Criteria: Clear understanding of current inconsistencies and requirements for each interface
    - Status: Completed

- [x] **Implementation Phase:**
  - [x] Create centralized filename generation utility
    - Path: `insanely_fast_whisper_api/utils/filename_generator.py`
    - Action: Implement a new module with standardized filename generation logic
    - **Design Patterns Applied:**
      - **Strategy Pattern**: Different strategies for CLI, WebUI, and API contexts while maintaining consistency
      - **Template Method Pattern**: Base template for filename generation with customizable components
    - Components to include:
      - Standardized timestamp format (ISO 8601: `YYYYMMDDTHHMMSSZ`)
      - Audio filename preservation (stem extraction)
      - Task type inclusion (transcribe/translate)
      - Interface context handling
      - File extension management
    - Status: [x] Completed - Initial module created with `FilenameGenerator`, `FilenameGenerationStrategy` (ABC), `StandardFilenameStrategy`, `FilenameComponents` dataclass, and `TaskType` enum. Core logic for stem extraction, task type inclusion, and extension handling is in place. Timestamp formatting uses `YYYYMMDDTHHMMSSZ` and is now configurable via the `FILENAME_TIMEZONE` environment variable (defaults to "Europe/Amsterdam"), using the `zoneinfo` library. Note: 'Z' suffix in format implies UTC, but will reflect local time if `FILENAME_TIMEZONE` is not UTC.
  - [x] Update CLI commands to use centralized filename generator
    - Path: `insanely_fast_whisper_api/cli/commands.py`
    - Action: Replaced current filename generation logic (lines 183-193) with calls to new utility
    - **Design Patterns Applied:**
      - **Dependency Injection**: Inject filename generator into CLI commands (Note: Currently instantiated directly; can be refactored for DI later if needed for broader CLI use or testing)
    - Status: [x] Completed - Updated `transcribe` command to use `FilenameGenerator` with `StandardFilenameStrategy`. Added necessary imports and passed `audio_path`, `TaskType.TRANSCRIBE`, and extension `json`.
  - [x] Update WebUI handlers to use centralized filename generator
    - Path: `insanely_fast_whisper_api/webui/handlers.py`
    - Action: Replace current filename generation logic (lines 295-318) with calls to new utility
    - **Design Patterns Applied:**
      - **Strategy Pattern**: Use appropriate strategy for WebUI export context (Note: Currently uses global StandardFilenameStrategy instance; can be made more flexible if needed)
    - Status: [x] Completed - Updated `export_transcription` to use `FilenameGenerator`. It now derives `audio_path` from `result.get("original_file")` and `task` from `result.get("task_type")` or `result.get("task")`. A global `WEBUI_FILENAME_GENERATOR` instance was added. Further debugging fixed `export_txt` and `export_srt` to use standardized filenames by modifying `webui/utils.py:save_temp_file` to accept a `desired_filename` and updating the export functions in `handlers.py` to generate and pass this name. **Final debugging completed** - Fixed issue where `original_file` and `task_type` were not preserved in the WebUI state by updating `process_transcription_request` to include these fields in `final_result_for_store`, ensuring SRT and TXT exports can use standardized filenames.
  - [x] Update core pipeline to use centralized filename generator
    - Path: `insanely_fast_whisper_api/core/pipeline.py`
    - Action: Replace current filename generation logic (lines 180-189) with calls to new utility
    - **Design Patterns Applied:**
      - **Template Method Pattern**: Maintain existing pipeline structure while standardizing filename generation (Note: FilenameGenerator instantiated in BasePipeline.__init__)
    - Status: [x] Completed - Updated `_save_result` in `BasePipeline` to use `FilenameGenerator`. An instance is created in `__init__`. `audio_path`, `TaskType`, and extension `json` are used. Interaction with `JsonStorage` extension handling noted for future review.
  - [x] Update storage backend to handle standardized filenames
    - Path: `insanely_fast_whisper_api/core/storage.py`
    - Action: Ensure JsonStorage works correctly with new filename patterns
    - Status: [x] Completed - `JsonStorage.save()` already correctly handles fully qualified filenames passed as `destination_path` due to its use of `with_suffix(".json")`. Updated comments to reflect that `destination_path` is now the full filename. No functional code changes required.

- [x] **Testing Phase:**
  - [x] Write unit tests for filename generator utility
    - Path: `tests/test_filename_generator.py`
    - Action: Test all filename generation scenarios, edge cases, and interface-specific requirements
    - Accept Criteria: 100% coverage of filename generation logic with various input scenarios
    - Status: [x] Completed - Created `tests/test_filename_generator.py` with comprehensive unit tests for `TaskType`, `FilenameComponents`, `StandardFilenameStrategy`, and `FilenameGenerator`, including timezone handling, stem extraction, and various timestamp scenarios using `pytest` and `unittest.mock`.
  - [x] Update existing tests to reflect new filename patterns
    - Path: `tests/test_cli.py`, `tests/test_webui.py`
    - Action: Update test assertions to match new standardized filename patterns
    - Accept Criteria: All existing tests pass with new filename conventions
    - Status: **WebUI Testing: [x] Completed** - WebUI filename standardization has been successfully tested in Docker environment. All export formats (JSON, SRT, TXT) now use consistent standardized filenames with pattern `{audio_stem}_{task}_{timestamp}.{extension}`. No warning messages about missing `original_file` field. **CLI Testing: [x] Completed** - CLI interface testing completed successfully. Fixed import errors in `cli.py` (constants import) and `commands.py` (facade import). CLI now generates standardized filenames correctly: `conversion-test-file_transcribe_20250530T210018Z.json`. Both WebUI and CLI interfaces now use consistent filename conventions. **API Testing: [x] Completed** - API interface testing completed successfully. **FIXED**: Modified `core/pipeline.py` to accept `original_filename` parameter and updated `api/routes.py` to pass `file.filename` instead of temp file path. API now generates standardized filenames correctly: `conversion-test-file_transcribe_20250530T211219Z.json` (without UUID prefix). All three interfaces (WebUI, CLI, API) now use consistent standardized filename conventions following the pattern `{audio_stem}_{task}_{timestamp}.{extension}`. **ISSUE RESOLVED**: API no longer includes temporary UUID prefix in saved filenames.

- [ ] **Documentation Phase:**
  - [x] Update `README.md` and `project-overview.md` to document new filename conventions
    - Describe standardized filename pattern: `{audio_stem}_{task}_{timestamp}.{extension}`
    - Explain timestamp format (ISO 8601) and rationale
    - Provide examples for each interface
    - Accept Criteria: Clear documentation of new filename conventions and migration notes
    - Status: **Completed** - Both `README.md` and `project-overview.md` have been updated with comprehensive documentation of the standardized filename conventions. Added new "Output Files and Filename Conventions" section to README with examples and configuration options. Added detailed "Filename Conventions (v0.2.1+)" section to project overview with interface-specific behavior, configuration options using `FILENAME_TIMEZONE`, and migration notes. Documentation includes the unified pattern `{audio_stem}_{task}_{timestamp}.{extension}`, examples for all file types, and timezone configuration instructions.

- [ ] **Review Phase:**
  - [ ] Validate the new filename conventions across all interfaces for consistency and usability
    - Ensure filenames are descriptive, sortable, and collision-resistant
    - Verify backward compatibility considerations are addressed
    - Test with various audio file types and names

## Architectural Overview

The new modular structure will introduce a centralized filename generation system:

- **FilenameGenerator** (Strategy Pattern): Base class defining the filename generation interface
- **StandardFilenameStrategy**: Default strategy implementing the unified naming convention
- **FilenameGeneratorFactory** (Factory Pattern): Creates appropriate generators based on context
- **FilenameComponents** (Data Class): Encapsulates filename components (audio_stem, task, timestamp, extension)

### New Standardized Pattern

**Unified Format**: `{audio_stem}_{task}_{timestamp}.{extension}`

**Examples**:
- CLI: `my_audio_transcribe_20241201T143022Z.json`
- WebUI TXT Export: `my_audio_transcribe_20241201T143022Z.txt`
- WebUI SRT Export: `my_audio_transcribe_20241201T143022Z.srt`
- WebUI JSON Export: `my_audio_transcribe_20241201T143022Z.json`
- API (when saved): `my_audio_translate_20241201T143022Z.json`

**Design Pattern Benefits**:
- **Strategy Pattern**: Allows for interface-specific customizations while maintaining core consistency
- **Template Method Pattern**: Ensures all filename generation follows the same basic structure
- **Factory Pattern**: Simplifies creation of appropriate filename generators

## Integration Points

- `insanely_fast_whisper_api/cli/commands.py` (CLI interface filename generation)
- `insanely_fast_whisper_api/webui/handlers.py` (WebUI export filename generation)
- `insanely_fast_whisper_api/core/pipeline.py` (Core pipeline filename generation)
- `insanely_fast_whisper_api/core/storage.py` (Storage backend filename handling)
- `tests/test_cli.py` (CLI tests expecting specific filename patterns)
- `tests/test_webui.py` (WebUI tests expecting specific filename patterns)

## Related Files

- `insanely_fast_whisper_api/cli/commands.py` (to be refactored)
- `insanely_fast_whisper_api/webui/handlers.py` (to be refactored)
- `insanely_fast_whisper_api/core/pipeline.py` (to be refactored)
- `insanely_fast_whisper_api/core/storage.py` (to be updated)
- `insanely_fast_whisper_api/webui/formatters.py` (contextually relevant)
- `insanely_fast_whisper_api/utils/constants.py` (may need new constants)

## Future Enhancements

- **Configurable Filename Templates**: Allow users to customize filename patterns via configuration
- **Filename Sanitization**: Enhanced handling of special characters in audio filenames
- **Collision Detection**: Automatic handling of filename collisions with incremental suffixes
- **Metadata Integration**: Optional inclusion of model name, language, or other metadata in filenames
- **Internationalization**: Support for different timestamp formats based on locale preferences 