# Feasibility Report

**Generated**: 2025-07-06 00:40 (NL time)

This report validates the checked tasks in `to-do/add-txt-srt-flags-cli.md`.

## Analysis Phase

> - [x] **Analysis Phase:**
>   - [x] Investigate existing text/SRT generation logic used in WebUI
>   - [x] List reusable utilities or note gaps where new converter functions need writing

**Commentary:**
The analysis was executed successfully. Instead of just reusing snippets, the developer created a new, centralized module `insanely_fast_whisper_api/core/formatters.py`. This module contains `TxtFormatter`, `SrtFormatter`, and `JsonFormatter` classes, which are now consumed by both the CLI (`commands.py`) and the WebUI (`handlers.py`). This is a robust solution that improves maintainability and avoids code duplication.

**Conclusion:** ✅ Feasible

---

## Implementation Phase

> - [x] **Implementation Phase:**
>   - [x] Extend CLI export flags
>   - [x] Generate files based on selected export format(s)
>   - [x] Ensure `_run_task()` returns paths of created files or logs them.

**Commentary:**
The implementation in `insanely_fast_whisper_api/cli/commands.py` is excellent. It correctly introduces a single `--export-format` option with choices (`all`, `json`, `srt`, `txt`), deprecates the old boolean flags, and uses the new `formatters` module to handle file generation. The logic in `_run_task` correctly handles the different output directories and file naming conventions.

**Conclusion:** ✅ Feasible

---

## Testing Phase (Complete)

> - [x] **Testing Phase:**
>   - **Status: Complete** - The test suite was fixed and now successfully validates the export functionality.

**Commentary:**
Following the recommendations, the test suite in `tests/test_cli_exports.py` was refactored. The `AttributeError` was resolved by correctly initializing `self.model` and `self.batch_size` in the `setup_method`. Furthermore, the tests were updated to use a timestamp-supporting model (`openai/whisper-tiny.en`), which fixed the failing SRT export tests.

**Conclusion:** ✅ Feasible & Complete
