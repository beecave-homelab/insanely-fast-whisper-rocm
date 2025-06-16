# To-Do: Refactor and Modularise `core.py`

This plan outlines the steps to refactor the monolithic `core.py` into smaller, cohesive modules that are easier to maintain, test, and extend. **It prescribes concrete design-pattern choices** so any developer can execute the refactor in a uniform, idiomatic way.

> **Folder convention**  
> All new modules live in the sub-package **`insanely_fast_whisper_api/core/`**.  
> The legacy top-level `core.py` will be deprecated after migration.

---

## Tasks

- [x] **Analysis Phase**
  - [x] Map current responsibilities and pain-points  
    - Path: `insanely_fast_whisper_api/core.py`
    - Action: Produce a concise responsibility matrix and LoC breakdown
    - Analysis Results:
      - **Responsibility Spread**

        | Concern | Lines | Comment |
        |---------|-------|---------|
        | Error definitions | ~30 | Move to `core/errors.py`; decouple from business logic |
        | Device helper `convert_device_string` | ~25 | Generic util duplicated elsewhere; move to `core/utils.py` |
        | Low-level ASR (`run_asr_pipeline`) | ~120 | Pure backend; target `core/asr_backend.py` |
        | High-level orchestration (`ASRPipeline`) | ~730 | Mix of config, lazy init, chunk loop, progress, storage |
        | JSON saving | ~60 | Side-effect; extract to `core/storage.py` |
      - **Design Problems & Duplications**
        - Device conversion logic re-implemented in two places.
        - Timestamp mapping duplicated.
        - Hidden deps (`librosa`, `soundfile`) imported at run-time.
        - `click` used solely for exceptions → overweight.
      - **Metrics**
        - File length ~950 LoC, cyclomatic complexity of `ASRPipeline.__call__` ≈ 17.
      - **Refactor Goals**
        1. Split into 5 elegant modules under `core/`.
        2. Apply design patterns for extensibility/testability.
        3. Remove `click` dep from core runtime.
    - Accept Criteria: Responsibility matrix + LoC report in `docs/architecture.md`.

- [x] **Implementation Phase**
  *Apply the following design patterns while creating the new modules (all paths are inside* `insanely_fast_whisper_api/core/` *).*

  | Module | Pattern | Why & How |
  |--------|---------|-----------|
  | `asr_backend.py` | **Strategy Pattern** | Define abstract `ASRBackend` interface (`transcribe(audio, cfg) -> dict`). Concrete strategy `HuggingFaceBackend` wraps Transformers. Future backends plug-in without edits to the pipeline. |
  | `pipeline.py` | **Template Method** | `BasePipeline` defines the algorithm skeleton (`prepare→process→save`). `WhisperPipeline` overrides variable steps (chunking vs non-chunking) while reusing error & progress logic. |
  | `utils.py` | **Singleton / Borg for Config Cache** | Implement `_GlobalConfig` (shared state) to cache device info + global defaults, ensuring a single expensive read. |
  | `storage.py` | **Factory Method** | Provide `StorageFactory.create(kind="json")` → `JsonStorage` (implements `save(result)`). Future storage (SQLite, S3) simply add subclasses. |
  | Progress events | **Observer Pattern** | Define `ProgressEvent` dataclass. `WhisperPipeline` exposes `add_listener`. UI or CLI registers observers for real-time updates. |

  ### Concrete steps

  1. [x] **Create sub-package skeleton**  
     - `mkdir -p insanely_fast_whisper_api/core && touch __init__.py`
          - Note: Executed as specified.
  2. [x] **Extract utilities**
     - Path: `core/utils.py`  
     - Move `convert_device_string`, timestamp helpers, implement `_GlobalConfig` singleton.
          - Note: `convert_device_string` moved. `_GlobalConfig` and timestamp helpers are currently placeholders (TODOs) and require full implementation.
          - Note: `insanely_fast_whisper_api/webui/utils.py` was also updated to use `core.utils.convert_device_string`, centralizing this utility.
  3. [x] **Error module**
     - Path: `core/errors.py`  
     - Relocate `TranscriptionError`, `DeviceNotFoundError`; drop `click` inheritance, subclass `Exception`.
          - Note: Implemented as specified.
          - Note: `insanely_fast_whisper_api/webui/errors.py` was also updated: its local error classes now subclass `Exception` instead of `click.ClickException` to align with removing `click` from non-CLI runtime.
  4. [x] **Backend strategy**
     - Path: `core/asr_backend.py`  
     - Define `ASRBackend` ABC + `HuggingFaceBackend`.
          - Note: Implemented. The internal logic for `HuggingFaceBackend` (e.g., pipeline initialization, transcription execution) was adapted from the original `core.py`'s `run_asr_pipeline` function and `ASRPipeline` class.
  5. [x] **Storage factory**
     - Path: `core/storage.py`  
     - Implement factory + `JsonStorage`.
          - Note: Implemented. The internal logic for `JsonStorage.save` was adapted from the original `core.py`'s `ASRPipeline._save_transcription_result` method.
  6. [x] **Pipeline (template + observer)**
     - Path: `core/pipeline.py`  
     - Create `BasePipeline` + `WhisperPipeline` consuming any `ASRBackend` strategy and emitting `ProgressEvent`s.
          - Note: Implemented. `ProgressEvent` dataclass and observer pattern (`add_listener`) are new.
          - Note: The `WhisperPipeline`'s `_prepare_input` method is currently basic. It does **not** implement the application-level audio chunking (e.g., using `librosa`) from the legacy `core.py`. It relies on the ASR backend's internal chunking. This is a key functional difference from the old pipeline if detailed app-level chunking was previously used.
  7. [x] **Package façade**
     - Edit `insanely_fast_whisper_api/__init__.py`  
     - Re-export `WhisperPipeline` as `ASRPipeline` for backwards compatibility.
          - Note: Implemented as specified. The old `run_asr_pipeline` import from `core.py` was removed from `__init__.py` to resolve an ImportError.
  8. [x] **Remove `click` dep from runtime** (keep it only in CLI tooling).
      - Note: `click` import and usage removed from the legacy `insanely_fast_whisper_api/core.py` by changing its local error classes to subclass `Exception`. New core modules (`core/errors.py`) do not use `click`. `webui/errors.py` was also updated (see step 3).
  9. [x] **Update imports project-wide** – replace `from ... import ASRPipeline` with the new path if necessary.
      - Note: Basic import paths for `ASRPipeline` updated in `main.py`, `webui/handlers.py`, and `tests/test_core.py` (legacy `webui.py` was deleted by user).
      - **Detailed Implementation Notes for Project-Wide Updates:**
          - **Pipeline Instantiation**: Significant changes were required in `main.py` and `webui/handlers.py` to instantiate the new `WhisperPipeline` (aliased as `ASRPipeline`). This involved:
              - Importing and instantiating `insanely_fast_whisper_api.core.asr_backend.HuggingFaceBackend`.
              - Passing the backend instance to the `WhisperPipeline` constructor.
              - Updating parameters for the `WhisperPipeline.process()` method.
          - **New FastAPI Parameters**: Added `dtype`, `better_transformer`, and `model_chunk_length` as FastAPI `Form` parameters in `insanely_fast_whisper_api/main.py` to configure the `HuggingFaceBackend`.
          - **Progress Callback Adaptation**: In `insanely_fast_whisper_api/webui/handlers.py`, the `progress_callback` mechanism was adapted to the new `ProgressEvent` / observer pattern. A warning was added that app-level chunking parameters from the UI (`chunk_duration`, `chunk_overlap`) are not currently used by the refactored pipeline.
          - **Unused Parameters**: Diarization-related parameters in `insanely_fast_whisper_api/main.py` are currently not utilized by the new pipeline structure.
          - **Utility Centralization**: `insanely_fast_whisper_api/webui/utils.py` was updated to use `convert_device_string` from `insanely_fast_whisper_api.core.utils`.

- [ ] **Testing Phase**
  - [ ] Unit tests for each component  
    - Path: `tests/`  
    - Actions:  
      - Mock backend strategies to test pipeline logic isolation.  
      - Verify observer notifications.  
    - Accept Criteria: ≥ 90 % coverage on new modules and green CI.

- [ ] **Documentation Phase**
  - [ ] Update architecture docs  
    - Path: `docs/architecture.md`, `README.md`  
    - Action: Add module diagram & design-pattern explanations.
    - Accept Criteria: Docs build without warnings.

## Related Files

- `insanely_fast_whisper_api/core.py` (legacy, slated for removal)
- `insanely_fast_whisper_api/core/utils.py`
- `insanely_fast_whisper_api/core/errors.py`
- `insanely_fast_whisper_api/core/asr_backend.py`
- `insanely_fast_whisper_api/core/pipeline.py`
- `insanely_fast_whisper_api/core/storage.py`
- Tests under `tests/`
- Updated `insanely_fast_whisper_api/__init__.py`

## Future Enhancements

- [ ] Replace temp WAV writes with in-memory buffers (Decorator for cacheable transforms).
- [ ] Add streaming backend strategy (e.g., WebSocket) via Strategy interface.
- [ ] Emit GPU/CPU resource metrics through Observer events for dashboards.

---

- [ ] **Code Quality Phase**
  - Address Pylint errors in `insanely_fast_whisper_api/core/` modules:
    - **`insanely_fast_whisper_api/core/asr_backend.py`**
      - [x] C0301: Line too long (116/100) (line 57)
      - [x] C0301: Line too long (105/100) (line 61)
      - [x] C0301: Line too long (107/100) (line 67)
      - [x] C0114: Missing module docstring (line 1)
      - [x] W0107: Unnecessary pass statement (line 29, in `ASRBackend.transcribe`)
      - [x] R0903: Too few public methods (1/2) (class `ASRBackend` line 16)
      - [x] R0913: Too many arguments (7/5) (method `HuggingFaceBackend.__init__` line 35)
      - [x] R0917: Too many positional arguments (7/5) (method `HuggingFaceBackend.__init__` line 35)
      - [x] R1720: Unnecessary "elif" after "raise" (line 55, in `_validate_device`)
      - [x] W1203: Use lazy % formatting in logging functions (lines 66, 77, 85, 89, 106, 128, 155)
      - [x] W0718: Catching too general exception Exception (line 84, in `_initialize_pipeline`)
      - [x] W0707: Consider explicitly re-raising (lines 93, 131)
      - [x] C0415: Import outside toplevel (time) (line 107)
      - [x] R0903: Too few public methods (1/2) (class `HuggingFaceBackend` line 32) - *Note: This might be a duplicate or related to the __init__ params. Review if R0913/R0917 fix addresses this.*
      - [x] C0411: standard import "logging" should be placed before third party imports... (line 11)
    - **`insanely_fast_whisper_api/core/utils.py`**
      - [ ] W0511: TODO: Implement _GlobalConfig singleton (line 3)
      - [ ] W0511: TODO: Add timestamp helpers (line 4)
      - [x] C0114: Missing module docstring (line 1)
      - [x] R1705: Unnecessary "elif" after "return" (line 27, in `convert_device_string`)
      - [x] W0611: Unused import torch (line 1)
    - **`insanely_fast_whisper_api/core/storage.py`**
      - [x] C0114: Missing module docstring (line 1)
      - [x] W0107: Unnecessary pass statement (line 20, in `BaseStorage.save`)
      - [x] R0903: Too few public methods (1/2) (class `BaseStorage` line 12)
      - [x] W1203: Use lazy % formatting in logging functions (lines 42, 45)
      - [x] R0903: Too few public methods (1/2) (class `JsonStorage` line 23)
      - [x] R1705: Unnecessary "else" after "return" (line 57, in `StorageFactory.create`)
      - [x] R0903: Too few public methods (1/2) (class `StorageFactory` line 51)
    - **`insanely_fast_whisper_api/core/errors.py`**
      - [x] C0114: Missing module docstring (line 1)
      - [x] W0107: Unnecessary pass statement (line 6, in `TranscriptionError`)
      - [x] W0107: Unnecessary pass statement (line 14, in `DeviceNotFoundError`)
    - **`insanely_fast_whisper_api/core/pipeline.py`**
      - [x] C0301: Line too long (108/100) (line 210)
      - [x] C0301: Line too long (106/100) (line 226)
      - [x] C0301: Line too long (111/100) (line 228)
      - [x] C0114: Missing module docstring (line 1)
      - [x] W0718: Catching too general exception Exception (line 63, in `BasePipeline._notify_listeners`)
      - [x] W1203: Use lazy % formatting in logging functions (lines 64, 123, 184, 187, 212, 227, 277)
      - [x] W0107: Unnecessary pass statement (lines 143, 154, 161, in `BasePipeline` abstract methods)
      - [x] W0246: Useless parent or super() delegation in method '__init__' (line 196, in `WhisperPipeline.__init__`)
      - [x] C0411: standard import "logging" should be placed before local imports... (line 14)
      - [x] W0611: Unused field imported from dataclasses (line 2, `field`)
