# To-Do: Integrate `stable-ts` for Timestamp Enhancement

This plan outlines the steps to integrate the `stable-ts` library to improve the accuracy and reliability of timestamps in transcription outputs.

## Tasks

- [ ] **Analysis Phase:**
  - [x] Research and evaluate `stable-ts` library.
    - Path: `https://github.com/jianfch/stable-ts`
    - Action: Researched the library's capabilities, focusing on the `non-whisper.ipynb` example to understand its integration pattern.
    - Analysis Results:
      - `stable-ts` offers superior timestamp accuracy through advanced VAD, silence suppression, and word-level stabilization.
      - It can wrap any ASR model, making it suitable for our existing inference pipeline.
      - It integrates with Demucs for noise reduction, which can improve both transcription and timestamp quality.
      - Integration is feasible by creating a wrapper around our current transcription logic and processing the results with `stable_whisper.transcribe_any()`.
    - Accept Criteria: A clear understanding of the benefits and the integration path is established.

- [ ] **Implementation Phase:**
  - [ ] **Dependency Management:**
    - Path: `pyproject.toml`
    - Action: Add `stable-ts` as a project dependency. Consider adding `demucs` as an optional dependency for noise reduction capabilities.
    - Status: `Pending`

  - [ ] **Create Inference Wrapper:**
    - Path: `insanely_fast_whisper_api/core/integrations/stable_ts.py`
    - Action: Create a new module for `stable-ts` integration. Implement a wrapper function that calls the existing transcription logic and formats the output dictionary to be compatible with `stable-ts` (i.e., word-level segments).
    - Status: `Pending`

  - [ ] **Integrate `transcribe_any`:**
    - Path: `insanely_fast_whisper_api/core/transcription.py`
    - Action: Modify the main transcription pipeline to import and conditionally call the `stable-ts` wrapper from the new `integrations` module. This should be triggered by a new API/CLI flag (e.g., `--stabilize`).
    - Status: `Pending`

  - [ ] **Expose New Options:**
    - Path: `insanely_fast_whisper_api/cli.py` and `insanely_fast_whisper_api/api.py`
    - Action: Add arguments to control `stable-ts` features, such as `--stabilize`, `--demucs`, `--vad`, and `--vad-threshold`.
    - Status: `Pending`

- [ ] **Testing Phase:**
  - [ ] **Unit & Integration Tests:**
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
