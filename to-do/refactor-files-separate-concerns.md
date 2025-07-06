# To-Do: Refactor Insanely Fast Whisper API Files

This plan outlines the steps to refactor several Python files within the `insanely_fast_whisper_api` project. The primary goal is to improve modularity by creating an `audio` subdirectory for audio-specific utilities and reorganizing other functionalities for better clarity, maintainability, and testability.

## Overview

The current structure has several utility and core functionality files directly under the `insanely_fast_whisper_api` directory. This refactoring aims to:
- Create an `insanely_fast_whisper_api/audio/` sub-package for dedicated audio processing logic.
- Relocate or consolidate other functionalities from the specified files to enhance separation of concerns without creating numerous additional subdirectories at this stage.
- Clarify responsibilities of each module.
- Make the codebase easier to navigate and extend.

## Tasks

- [x] **Analysis Phase:**
  - [x] Review current implementation of all target files.
    - Path: [`insanely_fast_whisper_api/audio_utils.py`, `insanely_fast_whisper_api/download_hf_model.py`, `insanely_fast_whisper_api/insanely_fast_whisper.py`, `insanely_fast_whisper_api/main.py`, `insanely_fast_whisper_api/models.py`, `insanely_fast_whisper_api/utils.py`]
    - Action: Analyze structure, data flow, error handling, and identify distinct responsibilities within each file. Confirm which components belong in the `audio/` subdirectory and decide the fate of others (new top-level modules or consolidation).
    - Analysis Results (Revised):
      - `audio_utils.py`: Contains audio processing (duration, splitting), temporary file cleanup, and transcription result merging. Functions `get_audio_duration` and `split_audio` are clear candidates for `audio/`. `cleanup_temp_files` is a general file utility. `merge_chunk_results` is specific to transcription post-processing.
      - `download_hf_model.py`: Handles Hugging Face model downloading and includes CLI options. This is model management, not audio processing.
      - `insanely_fast_whisper.py`: Provides the main CLI interface and ASR pipeline logic. Core transcription pipeline logic is distinct from pure audio file manipulation.
      - `main.py`: Defines the FastAPI application, routes, and startup/middleware logic. This is API specific.
      - `models.py`: Contains Pydantic models for API request/response. API specific.
      - `utils.py`: Contains general utility functions (file validation, saving uploads). Some are general, some API-related.
    - **Design Pattern Candidates:**
      - **Strategy Pattern:** Could be used within the `audio` module if different audio processing techniques are envisioned (e.g., different chunking strategies).
      - **Factory Pattern:** For creating ASR pipeline instances or model instances (likely outside the `audio` module).
      - **Command Pattern:** For CLI command handling in `insanely_fast_whisper.py` and `download_hf_model.py` (outside `audio` module).
    - Accept Criteria: Clear breakdown of components, responsibilities, and a confirmed plan for the `audio/` subdirectory and reorganization of other functionalities.
    - Status: Pending

- [ ] **Implementation Phase:**
  - [x] **Create `insanely_fast_whisper_api/audio/` sub-package:**
    -   `insanely_fast_whisper_api/audio/__init__.py`
    -   `insanely_fast_whisper_api/audio/processing.py` (for core audio functions)
    -   Potentially `insanely_fast_whisper_api/audio/effects.py` or similar if more advanced audio manipulations are added later.
    - Status: Completed - Directory and initial files created.

  - [x] **Refactor `insanely_fast_whisper_api/audio_utils.py`**
    - Path: `insanely_fast_whisper_api/audio_utils.py`
    - Action:
      - Moved `get_audio_duration` and `split_audio` to `insanely_fast_whisper_api/audio/processing.py`.
      - Relocated `cleanup_temp_files` to `insanely_fast_whisper_api/utils.py`.
      - Relocated `merge_chunk_results` to `insanely_fast_whisper_api/audio/results.py`.
      - Deleted `insanely_fast_whisper_api/audio_utils.py`.
    - **Design Patterns Applied:**
      - **Single Responsibility Principle (SRP):** Applied to `insanely_fast_whisper_api/audio/processing.py` and `insanely_fast_whisper_api/audio/results.py`.
      - **Strategy Pattern:** Considered for `split_audio` (in `audio/processing.py`).
    - Status: Completed

  - [ ] **Review and Refactor other specified files (`download_hf_model.py`, `insanely_fast_whisper.py`, `main.py`, `models.py`, `utils.py`)**
    - Path: Remaining files from the user's list.
    - Action:
      - Identify if any functions/classes within these files strictly perform audio input/output operations or low-level audio data manipulation that belong in `insanely_fast_whisper_api/audio/`.
      - For functionalities not fitting into `audio/`, ensure they are well-organized within their current files or consider minimal splitting into new top-level modules if a file has multiple, clearly distinct major responsibilities (e.g., `download_hf_model.py` could potentially separate its core downloader logic from its CLI logic into two files like `model_downloader.py` and `model_cli.py` at the top level of `insanely_fast_whisper_api/` if deemed necessary for clarity, but avoid creating many new subdirectories).
      - `models.py` (Pydantic models) likely remains as is, as it serves a distinct API-related purpose.
      - `utils.py` will absorb general file utilities like `cleanup_temp_files`. API-specific utilities like `validate_audio_file` can stay in `utils.py` or move to `main.py` if very tightly coupled to API request handling.
    - **Design Patterns Applied:** Continue to apply Single Responsibility Principle at the function/class level. For CLI parts, Command Pattern is relevant.
    - Status: Pending

  - [ ] Update all imports across the codebase to reflect the new locations of moved functions/classes.
    - Status: Pending

- [ ] **Testing Phase:**
  - [ ] Write or migrate unit tests for new/modified modules, especially for `insanely_fast_whisper_api/audio/processing.py`.
    - Path: `tests/` (e.g., `tests/audio/test_processing.py`, update existing tests for other files)
    - Action: Ensure comprehensive coverage for the new structure and verify that existing functionality remains unchanged.
    - Accept Criteria: All core functionality is covered by automated tests, and all tests pass.

- [ ] **Documentation Phase:**
  - [ ] Update `README.md` and `project-overview.md`.
    - Action: Describe the new `insanely_fast_whisper_api/audio/` package and other significant changes to module responsibilities.
    - Accept Criteria: Documentation is clear, up-to-date, and provides a good guide for new contributors.

- [ ] **Review Phase:**
  - [ ] Validate the new structure for clarity, maintainability, adherence to best practices, and effective use of design patterns.
  - [ ] Ensure all internal imports are updated correctly.

## Architectural Overview (Revised)

The primary structural change will be the introduction of the `insanely_fast_whisper_api/audio/` sub-package:

-   **`insanely_fast_whisper_api/audio/`**: Dedicated to audio file manipulation and processing.
    -   `__init__.py`
    -   `processing.py`: Functions for audio splitting, duration calculation, and other direct audio manipulations.

Other modules (`download_hf_model.py`, `insanely_fast_whisper.py`, `main.py`, `models.py`, `utils.py`) will remain at the top level of `insanely_fast_whisper_api/` but may undergo internal reorganization or minor splits into new top-level files if a single file currently handles overly disparate major concerns. The goal is to improve clarity without over-fragmenting the top-level directory structure.

**Design Pattern Rationale (Preliminary):**
- **Single Responsibility Principle** will guide the creation of the `audio/processing.py` module and internal reorganization of other files.
- **Command Pattern** for CLI aspects in `insanely_fast_whisper.py` and `download_hf_model.py` remains a consideration for their internal structure.

## Integration Points

- Existing test files in `tests/` will need significant updates to reflect new module paths and potentially new test files for `audio/processing.py`.
- `Dockerfile`, `docker-compose.yaml` might need path adjustments if entrypoints or copied files change structure significantly.
- Any scripts or external tools relying on the current paths of CLI scripts or modules will need updates.
- `pyproject.toml` might need updates if module entry points for scripts are defined there.
- All files within `insanely_fast_whisper_api/` that use functions from `audio_utils.py`

---

## Code review 28-05-2025

Based on my review of the completed tasks outlined in `to-do/refactor-files-separate-concerns.md`, I can provide the following assessment:

1. **Creation of `insanely_fast_whisper_api/audio/` Sub-Package**:
   - **Status**: Completed and verified.
   - **Review**: The directory structure has been created with `__init__.py`, `processing.py`, and `results.py`. This is a good step towards modularizing audio-related functionality, improving codebase organization, and making it easier to maintain and extend.

2. **Refactoring of `audio_utils.py`**:
   - **Status**: Completed and verified.
   - **Details**:
     - `get_audio_duration` and `split_audio` have been moved to `audio/processing.py`. The code is well-documented, with type hints and error handling, adhering to best practices. The `split_audio` function shows potential for the Strategy Pattern if different splitting techniques are needed in the future.
     - `merge_chunk_results` has been moved to `audio/results.py`. This function is focused on post-processing transcription results, maintaining a clear separation of concerns.
     - `cleanup_temp_files` has been moved to `utils.py`, which is appropriate for a general utility function not specific to audio processing.
     - `audio_utils.py` has been deleted, confirming the refactoring is complete.
   - **Review**: The refactoring adheres to the Single Responsibility Principle by separating audio processing logic from general utilities and transcription result merging. The code quality is high, with proper documentation and error handling. The deletion of the original file ensures no redundant code remains.

3. **Design Patterns Applied**:
   - **Single Responsibility Principle (SRP)**: Applied effectively in `audio/processing.py` and `audio/results.py` by ensuring each module has a specific focus.
   - **Strategy Pattern**: Considered for `split_audio` in `audio/processing.py`, which is a forward-thinking approach for potential future enhancements.

### Recommendations and Observations

- **Code Quality**: The moved functions retain their original quality with proper documentation, type hints, and error handling. This maintains consistency across the codebase.
- **Modularity**: The creation of the `audio` sub-package is a significant improvement in modularity, making it easier to locate and work on audio-specific code.
- **Future Considerations**: As noted in the plan, the Strategy Pattern could be implemented for `split_audio` if different chunking strategies are needed. This should be revisited if new requirements arise.
- **Testing**: While not yet completed as per the plan, ensure that unit tests for the new modules (`audio/processing.py` and `audio/results.py`) are written or migrated to cover the moved functionality. This is critical to maintain reliability.
- **Documentation**: Update `README.md` and `project-overview.md` to reflect these changes, as planned in the Documentation Phase. This will help new contributors understand the new structure.

Overall, the completed tasks have been executed well, with a clear focus on improving separation of concerns and modularity. The refactoring aligns with the outlined goals and adheres to best practices.