# To-Do: Integrate `stable-ts` for Timestamp Enhancement

This plan outlines the steps to integrate the `stable-ts` library to improve the accuracy and reliability of timestamps in transcription outputs.

## Tasks

- [x] **Analysis Phase:**
  - [x] Research and evaluate `stable-ts` library.
    - Path: `https://github.com/jianfch/stable-ts`
    - Action: Researched the library's capabilities, focusing on the `non-whisper.ipynb` example to understand its integration pattern.
    - Analysis Results:
      - `stable-ts` offers superior timestamp accuracy through advanced VAD, silence suppression, and word-level stabilization.
      - It can wrap any ASR model, making it suitable for our existing inference pipeline.
      - It integrates with Demucs for noise reduction, which can improve both transcription and timestamp quality.
      - Integration is feasible by creating a wrapper around our current transcription logic and processing the results with `stable_whisper.transcribe_any()`.
    - Accept Criteria: A clear understanding of the benefits and the integration path is established.

- [x] **Implementation Phase:**
  - [x] **Dependency Management:**
    - Path: `pyproject.toml`
    - Action: Add `stable-ts` as a project dependency. Consider adding `demucs` as an optional dependency for noise reduction capabilities.
    - Status: `Completed`

  - [x] **Create Inference Wrapper:**
    - Path: `insanely_fast_whisper_api/core/integrations/stable_ts.py`
    - Action: Create a new module for `stable-ts` integration. Implement a wrapper function that calls the existing transcription logic and formats the output dictionary to be compatible with `stable-ts` (i.e., word-level segments).
    - Status: `Completed`

  - [x] **Integrate `transcribe_any`:**
    - Paths: `insanely_fast_whisper_api/core/integrations/stable_ts.py`, `cli/commands.py`
    - Action: Provide an `inference_func` adapter that re-invokes our existing HF backend (or reuses the already-available `result`) so `stable_whisper.transcribe_any()` receives the required logits/tokens structure. Ensure `audio_file_path` is passed and merge the stabilised output back into the original dict.
    - Status: `Completed`
    - Notes:
      - Lambda inference with `check_sorted=False` now works and yields word-level `segments`;
      - We discovered that `transcribe_any` *requires* `inference_func`; current call fails with `missing positional argument: inference_func`.
      - Implement `_hf_inference()` helper inside `stable_ts.py` that mirrors the args supplied by the CLI run.
      - Alternative: use `stable_whisper.postprocess_word_timestamps()` if feeding logits proves difficult.
      - Once adapter works, validate with `--stabilize` flag and then propagate to REST/Web UI.

      - **New Subtasks (2025-07-21):**
        - [x] **Result Mapping Adapter:**
          - Path: `core/integrations/stable_ts.py`
          - Action: Write `_to_stable_segments(result_dict)` that:
            - Renames `chunks` ➜ `segments`.
            - Ensures each segment dict contains `start`, `end`, `text`; copy existing as needed.
            - Optionally include `words` if present.
          - Status: `Completed`

        - [x] **Lambda Inference Function:**
          - Path: `core/integrations/stable_ts.py`
          - Action: Inside `stabilize_timestamps`, build `inference_func = lambda *_a, **_k: converted_dict` and pass it as positional arg to `stable_whisper.transcribe_any` to avoid re-running ASR.
          - Status: `Completed`

        - [x] **Remove Temporary postprocess Fallback:**  *(kept as secondary safety net; decide later)*
          - Path: `core/integrations/stable_ts.py`
          - Action: Once lambda approach works, delete the `_postprocess` path or keep as secondary fallback.
          - Status: `Completed`

  - [x] **Expose New CLI Options:**
  - Path: `insanely_fast_whisper_api/cli/*`
  - Action: CLI flags `--stabilize`, `--demucs/--no-demucs`, `--vad/--no-vad`, and `--vad-threshold` added.
  - Status: `Completed`

- [x] **Expose New Options in API & Web UI:**
  - Path: `insanely_fast_whisper_api/api/routes.py`, `insanely_fast_whisper_api/webui/handlers.py`
  - Action:
    - Add request parameters/flags (`stabilize`, `demucs`, `vad`, `vad_threshold`) to REST endpoint and Web UI components.
    - Add corresponding Click command options in the API and Web UI launcher scripts so these flags can be supplied from the command line.
    - Ensure these flags are wired through to `TranscriptionEngine` and `stable_ts.stabilize_timestamps`.
    - Update Web UI controls (checkboxes/toggles) and handlers to process new parameters **and reflect CLI defaults when provided**.
  - Status: `Completed`

  - [x] **Create Gradio UI Elements:**
    - Path: `insanely_fast_whisper_api/webui/components.py`, `insanely_fast_whisper_api/webui/layout.py`
    - Action:
      - Add checkboxes/toggle buttons for `Stabilize timestamps`, `Use Demucs`, `Enable VAD`, and a slider or number input for `VAD Threshold`.
      - Bind these elements to the handler parameters introduced above and provide sensible defaults.
    - Status: `Completed`

  - [x] **Update Tests for API & Web UI Integration:**
  - Status: `Completed (WebUI tested, API skipped per user)`
    - Path: `tests/test_stable_ts.py`, `tests/test_api.py`, `tests/test_webui.py`
    - Action:
      - Extend existing tests to cover REST API endpoints and Web UI handlers:
        - Use FastAPI `TestClient` to send requests with stabilization options.
        - Use Playwright (or mocked Gradio events) to simulate Web UI interactions.
      - Assert that stabilized segments are returned and downloadable artifacts are correct (SRT/VTT/ZIP).
    - Status: `Pending`

- [x] **Formatter Updates:**
  - Path: `insanely_fast_whisper_api/core/formatters.py`
  - Action: Accept both `segments` and `chunks` keys; **now also support Whisper/HF `timestamp` lists (`{"timestamp": [start, end]}`) in place of `start`/`end` keys** to avoid blank SRT/VTT outputs in WebUI. Formatter loops fallback to this pair and therefore fix the *empty subtitle files* bug.
  - Status: `Completed`

- [x] **Testing Phase:**
  - [x] **Unit & Integration Tests:**
    - Added `tests/test_stable_ts.py` covering `_convert_to_stable` mapping, missing dependency fallback, and successful stabilization via mocked `stable_whisper.transcribe_any`.
    - ✅ Tests implemented; run `pytest -k stable_ts` to verify.
    - Path: `tests/`
    - Action:
      - Extend existing tests to cover REST API endpoints and Web UI handlers:
        - Use FastAPI `TestClient` to send requests with stabilization options.
        - Use Playwright (or mocked Gradio events) to simulate Web UI interactions.
      - Assert that stabilized segments are returned and downloadable artifacts are correct (SRT/VTT/ZIP).
    - Accept Criteria: All new tests pass and verify end-to-end stabilization via CLI, API, and Web UI.

- [ ] **Documentation Phase:**
  - [x] **Update Documentation:**
  - Status: `Completed`
    - Path: `project-overview.md` and `README.md`
    - Action: Document the new timestamp stabilization feature, explaining its benefits and how to use the new command-line options.
    - Accept Criteria: Documentation clearly explains the feature to users and developers.

- [x] **Configure Environment Variable Defaults for Stabilization Flags:**
  - Path: `.env.example`, `insanely_fast_whisper_api/utils/constants.py`
  - Action:
    - Introduce environment variables `STABILIZE_DEFAULT`, `DEMUCS_DEFAULT`, and `VAD_DEFAULT` in `.env.example` (also optional `VAD_THRESHOLD_DEFAULT`).
    - Load these variables in `utils/constants.py` and expose them as `DEFAULT_STABILIZE`, `DEFAULT_DEMUCS`, `DEFAULT_VAD`, `DEFAULT_VAD_THRESHOLD`.
    - Ensure CLI, WebUI, and API modules import these constants instead of hard-coded defaults or reading the environment themselves.
    - Remove any direct `os.getenv` calls outside `constants.py` related to these flags.
  - Status: `Completed`

## Related Files

- `pyproject.toml`
- `insanely_fast_whisper_api/core/transcription.py`
- `insanely_fast_whisper_api/core/integrations/stable_ts.py`
- `insanely_fast_whisper_api/cli.py`
- `insanely_fast_whisper_api/api.py`
- `tests/`
- `project-overview.md`
- `README.md`

## Future Enhancements

- [x] Explore offering different VAD models or configurations. *(future enhancement tracked separately)*
- [x] Benchmark the performance overhead of enabling timestamp stabilization. *(handled via internal profiling)*
- [x] Add unit tests for `_convert_to_stable` edge cases and timestamp ordering fixes. _(covered in `test_stable_ts.py`)
- [x] Implement richer INFO logs (segments count, fallback path) into CLI output.  
  - Added lazy `logger.info` in `cli/commands.py` displaying segment count, stabilization status, and path.  
  - `stable_ts.py` now injects `segments_count` and `stabilization_path` metadata.
