# To-Do: Hugging Face Model Download Script

This plan outlines the steps to implement a robust script for downloading Whisper ASR models from Hugging Face, supporting CLI, Python module, and programmatic use, with force re-download and logging options.

## Tasks

- [x] **Analysis Phase:**
  - [x] Research and evaluate best practices for Hugging Face model downloads in Docker/container environments
    - Path: `[insanely_fast_whisper_api/download_hf_model.py]`
    - Action: Review Hugging Face docs, forums, and existing code for optimal download and cache management
    - Analysis Results:
      - Use `huggingface_hub.snapshot_download` for efficient, idempotent downloads
      - Support authentication via `HUGGINGFACE_TOKEN`
      - Respect cache env vars: `HF_HOME`, `TRANSFORMERS_CACHE`, `HUGGINGFACE_HUB_CACHE`
      - Ensure compatibility with both CLI and programmatic use
    - Accept Criteria: Script approach is robust, Docker/container-safe, and user-friendly

- [x] **Implementation Phase:**
  - [x] Implement script with the following features:
    - Path: `[insanely_fast_whisper_api/download_hf_model.py]`
    - Action:
      - Implement CLI using `click` package.
      - Accept model name via CLI (`-m`/`--model`) or `WHISPER_MODEL` env var
      - Support `--force` flag for forced re-download
      - Expose `download_model_if_needed(model_name, force=False, logger=None)` for programmatic use
      - Implement `__main__` for CLI and `-m` module invocation
      - Use Python `logging` for status and error output (stdout/stderr)
      - Handle errors gracefully (network, auth, invalid model, etc.)
      - Status: Completed
  - [x] Integrate the model download script into API and WebUI startup:
    - Path: `[main.py]`, `[insanely_fast_whisper_api/webui.py]`
    - Action: Import and call the download function on startup to automatically download the model defined by the `WHISPER_MODEL` environment variable
    - Accept Criteria: Model download runs automatically on API and WebUI startup, using the environment variable
      - Implement `__main__` for CLI and `-m` module invocation
      - Use Python `logging` for status and error output (stdout/stderr)
      - Handle errors gracefully (network, auth, invalid model, etc.)
    - Status: Completed

- [x] **Testing Phase:**
  - [x] Test integration in API and WebUI startup
    - Path: `[main.py]`, `[insanely_fast_whisper_api/webui.py]`
    - Action: Restart Docker container and check logs for download messages from both API and WebUI on startup
    - Accept Criteria: Model download/verification messages appear in logs from both main.py and webui.py, using WHISPER_MODEL env var
    - Status: Completed
  - [ ] Unit and integration tests for script behavior
    - Path: `[tests/test_download_hf_model.py]`,
    - Action:
      - [x] Test default model download when no model name or env var is provided
      - Test CLI, module, and programmatic invocation; test force re-download, error handling, and logging
    - Accept Criteria: All tests pass; script works as expected in Docker and locally

- [x] **Documentation Phase:**
  - [x] Update `project-overview.md` and/or README
    - Path: `[project-overview.md]`, `[README.md]`
    - Action: Document usage examples, environment variables, CLI/module/programmatic options, and logging
    - Accept Criteria: Documentation is up-to-date and clearly explains the new script and its options
    - Status: Completed

## Related Files

- `[insanely_fast_whisper_api/download_hf_model.py]`
- `[tests/test_download_hf_model.py]`
- `[project-overview.md]`
- `[README.md]`

## Future Enhancements

- [ ] Support downloading multiple models in one invocation
- [ ] Add progress bar or advanced logging options
- [ ] Allow custom cache directory via CLI
