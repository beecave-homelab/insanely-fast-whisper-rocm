# To-Do: Enhance Configuration Loading and Setup

This plan outlines the steps to improve the application's configuration handling. This includes implementing hierarchical .env loading and creating a user-friendly setup script.

## Tasks

### Phase 1: Hierarchical .env Loading

- [x] **Refinement: Isolate Debug Logic**
  - [x] Create a new utility file (`insanely_fast_whisper_api/utils/env_loader.py`).
    - Action: Move the logic for `_show_debug_prints` (checking CLI args and LOG_LEVEL) and the `_debug_print` function into this new file.
  - [x] Update `insanely_fast_whisper_api/utils/constants.py`.
    - Action: Import and use the environment loading helper functions from `env_loader.py` to keep `constants.py` cleaner.
  - Accept Criteria: Debug print logic is successfully encapsulated in the new utility, and `constants.py` is simplified.

- [x] **Implementation Phase:**
  - [x] Modify `constants.py` to implement the new `.env` loading logic.
    - Path: `insanely_fast_whisper_api/utils/constants.py`
    - Action:
      1. Add code to detect the project root directory.
      2. Load the `.env` file from the project root using `load_dotenv(..., override=True)`.
      3. Load the `.env` file from `~/.config/insanely-fast-whisper-api/.env` using `load_dotenv(..., override=True)` to ensure it overrides any previously set variables.
    - Status: `Completed`

- [x] **Conditional Debug Output (Successfully Refactored to `env_loader.py`)**
  - [x] Implement logic to show debug prints only if `--debug` CLI flag is present or `LOG_LEVEL=DEBUG` is in `.env` (now in `env_loader.py`).
  - [x] Format debug prints to mimic INFO log style with timestamps (now in `env_loader.py`).
  - [x] Verify conditional debug output behavior (post-refactor).

- [x] **Documentation Phase:**
  - [x] Update the docstring in `constants.py`.
    - Path: `insanely_fast_whisper_api/utils/constants.py`
    - Action: Clearly document the new loading order (project root first, then user-specific) and explain that user-specific settings will override project settings.
    - Accept Criteria: The docstring accurately reflects the implementation.

- [x] **Testing Phase:**
  - [x] Manually verify the new loading behavior.
    - Action:
      1. Create a test variable in the project root `.env`.
      2. Create the same test variable with a different value in `~/.config/insanely-fast-whisper-api/.env`.
      3. Run the application and confirm that the value from the user-specific file is used.
    - Accept Criteria: The override logic works as expected.

### Phase 2: User Configuration Setup Script

- [x] **Implementation Phase:**
  - [x] Create a new utility script.
    - Path: `scripts/setup_config.py`
    - Action:
      1. Define the source (`.env.example`) and destination (`~/.config/insanely-fast-whisper-api/.env`) paths.
      2. Check if the destination file already exists.
      3. If it exists, prompt the user to confirm if they want to overwrite it. If they decline, exit gracefully.
      4. If it doesn't exist or the user confirms overwrite, copy the contents from the source file to the destination file.
      5. Print a clear message to the user informing them that the file has been created/updated and that they should now edit it to add their specific values (e.g., `HUGGINGFACE_TOKEN`).
  - [x] Add a CLI entrypoint for the script.
    - Path: `pyproject.toml`
    - Action: Add a new script entry under `[tool.pdm.scripts]` to make the utility easily runnable (e.g., `pdm run setup-config`).

- [x] **Documentation Phase:**
  - [x] Update the `README.md` to document the new setup script.
    - Path: `README.md`
    - Action: Add a section explaining how to use the new setup script to create the user-specific configuration.

## Related Files

- `insanely_fast_whisper_api/utils/constants.py`
- `scripts/setup_config.py` (new file)
- `pyproject.toml`
- `README.md`
- `.env.example`

---

## Code Quality Test Results (As of 2025-06-10)

After completing the environment loading refactor, code quality checks were run from within the `insanely-fast-whisper-rocm-dev` Docker container.

- [x] **`black`**: Passed after auto-formatting 2 files (`constants.py`, `env_loader.py`).
- [x] **`isort`**: Passed after auto-fixing imports in 37 files.
- [ ] **`mypy`**: **Failed with 34 errors.**

### Mypy Errors & Blockers

A significant number of type-checking errors were found. The initial investigation was blocked by a **file permission error**, preventing automated fixes.

**Next Steps for Mypy:**

1. [ ] **Resolve file permission issues** to allow code modifications.
2. [ ] Apply the manual fixes already identified for:
    - `insanely_fast_whisper_api/utils/constants.py` (Incorrect `Literal` assignment)
    - `insanely_fast_whisper_api/utils/file_utils.py` (Potential `None` access on `file.filename`)
3. [ ] Systematically analyze and fix the remaining 32 `mypy` errors.
4. [ ] Re-run all code quality checks to confirm all issues are resolved.
