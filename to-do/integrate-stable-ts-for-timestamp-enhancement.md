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

  - [x] **Expose New Options:**
    - Path: `insanely_fast_whisper_api/cli/*`
    - Action: CLI flags `--stabilize`, `--demucs/--no-demucs`, `--vad/--no-vad`, and `--vad-threshold` added.
    - Status: `Completed (CLI)`
    - Follow-up: expose same options in REST API and Web UI.
    - **✅ Current status (2025-07-21 22:00):**
  - Bug fixed in `_convert_to_stable` – comparison guard prevents `TypeError` when timestamps are `None`. Segments with missing timestamps are filtered out before processing.
  - CLI flags successfully invoke stable-ts, Demucs, and Silero-VAD. Stabilised `segments` now contain valid `start`/`end` pairs and SRT/VTT export succeeds.
  - ✅ Manual run confirmed: transcription and SRT export work correctly.
  - Next step: write unit & integration tests to assert timestamp validity and formatter output.

- [x] **Formatter Updates:**
  - Path: `insanely_fast_whisper_api/core/formatters.py`
  - Action: Accept both `segments` and `chunks` keys; skip segments lacking timestamps to prevent errors.
  - Status: `Completed`

- [ ] **Testing Phase:**
  - [ ] **Unit & Integration Tests:**  *(in progress)*
    - Path: `tests/`
    - Action: Create new tests that transcribe a sample audio file with the stabilization features enabled. Assert that the output SRT/VTT is correctly formatted and that timestamps are more accurate than the baseline. Test the new CLI/API flags.
    - Accept Criteria: Tests pass, and the stabilization feature is verified to work as expected.

- [ ] **Documentation Phase:**
  - [ ] **Update Documentation:**
    - Path: `project-overview.md` and `README.md`
    - Action: Document the new timestamp stabilization feature, explaining its benefits and how to use the new command-line options.
    - Accept Criteria: Documentation clearly explains the feature to users and developers.

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

- [ ] Explore offering different VAD models or configurations.
- [ ] Benchmark the performance overhead of enabling timestamp stabilization.
- [ ] Add unit tests for `_convert_to_stable` edge cases and timestamp ordering fixes.
- [ ] Implement richer INFO logs (segments count, fallback path) into CLI output.
