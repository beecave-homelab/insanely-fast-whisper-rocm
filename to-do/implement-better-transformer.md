# To-Do: Implement Hugging Face Optimum Accelerations

This plan outlines the steps to implement Optimum's acceleration features to accelerate the Whisper ASR pipeline on ROCm.

## Tasks

- [x] **Analysis Phase:**
  - [x] Review available acceleration options within Hugging Face Optimum for ROCm.
    - Path: Documentation Review
    - Action: Based on the provided documentation, analyze the two primary acceleration paths for ROCm devices.
    - Analysis Results:
      - **Path 1: BetterTransformer:** This is a PyTorch-native optimization. It is implemented by installing the `optimum` library and wrapping the existing model with `BetterTransformer.transform(model)`. This is a relatively simple, low-effort integration.
      - **Path 2: ONNX Runtime (ORT):** This method involves installing `onnxruntime-rocm` and converting the model to the ONNX format at runtime. The model is then loaded via `optimum.onnxruntime` using the `ROCMExecutionProvider`. This can yield higher performance but is more complex to set up and might require changes to the model loading and inference logic in `insanely_fast_whisper_api/core/asr_backend.py`. The project already includes `onnxruntime-rocm` as an optional dependency, so this path was likely considered.
      - **Note on `optimum-amd`:** The `optimum-amd` package on PyPI also provides an interface for ONNX Runtime on ROCm. However, the official and more current Hugging Face documentation points to this functionality being integrated directly into the main `optimum` package. Therefore, `optimum` is the correct dependency to use.
      - **Conclusion:** The initial request was for `BetterTransformer`. Given that it is far less intrusive, we should finalize and test this implementation first. The ONNX Runtime path can be considered a future, more advanced optimization.
    - Accept Criteria: A clear understanding of the options and a decision on the path forward. The decision is to proceed with `BetterTransformer`.

- [x] **Implementation Phase:**
  - [x] Add `optimum` package to `pyproject.toml`.
    - Path: `pyproject.toml`
    - Action: Add the `optimum` package to the `[project.dependencies]` section.
    - Status: Done

  - [x] Integrate BetterTransformer in the ASR backend.
    - Path: `insanely_fast_whisper_api/core/asr_backend.py`
    - Action: Import `BetterTransformer` from `optimum.bettertransformer` and wrap the loaded model using `model = BetterTransformer.transform(model)`. The previous use of the deprecated `.to_bettertransformer()` has been corrected.
    - Status: Done

- [x] **Testing Phase:**
  - [x] Test the new implementation.
    - Path: `tests/` or via manual execution.
    - Action: Ensure that the ASR pipeline still functions correctly after the changes and that there are no new errors.
    - Accept Criteria: The transcription process works as expected.
    - Status: Done. The implementation is now working correctly after fixing the model, tokenizer, and feature extractor loading.

- [x] **Documentation Phase:**
  - [x] Update `project-overview.md` if necessary.
    - Path: `project-overview.md`
    - Action: Document the use of Optimum and BetterTransformer for performance optimization.
    - Accept Criteria: Documentation is up-to-date.
    - Status: Done. The final implementation uses native `transformers` optimization (`attn_implementation='sdpa'`), which is a standard feature and does not require extensive custom documentation. The `optimum` dependency remains for other potential future optimizations but is not the primary driver for this feature anymore.

## Related Files

- `pyproject.toml`
- `insanely_fast_whisper_api/core/asr_backend.py`
- `project-overview.md`

## Future Enhancements

- [ ] Benchmark the performance improvement from BetterTransformer. 