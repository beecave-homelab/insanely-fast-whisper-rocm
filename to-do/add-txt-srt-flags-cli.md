# To-Do: Add --txt / --srt Export Flags to CLI

This plan outlines the steps to let the CLI optionally export plain-text (`.txt`) and SubRip subtitle (`.srt`) versions of a transcription/translation, matching WebUI behaviour.

## Tasks

- [x] **Analysis Phase:**
  - [x] Investigate existing text/SRT generation logic used in WebUI
    - Path: `[insanely_fast_whisper_api/webui/handlers.py]`
    - Action: Identify helper functions/classes that convert the JSON result to `.txt` and `.srt` files (likely via `BatchZipBuilder` or similar)
    - Analysis Results:
      - [x] List reusable utilities or note gaps where new converter functions need writing
    - Accept Criteria: Clear decision whether to reuse or implement new converters

- [x] **Implementation Phase:**
  - [x] Extend CLI export flags
    - Path: `[insanely_fast_whisper_api/cli/commands.py]`
    - Action: Add mutually-exclusive options:
      - `--export-json` (default. Saved to `transcripts/`)
      - `--export-txt` (Saved to `transcripts-txt/`)
      - `--export-srt` (Saved to `transcripts-srt/`)
      - `--export-all` (writes json to `transcripts/`, txt to `transcripts-txt/`, and srt to `transcripts-srt/`)
      Use a `click.Choice` or `click.Option` group so the user can only choose one, unless `--export-all` is selected.
    - Status: Done
  - [x] Generate files based on selected export format(s)
    - Path: `[insanely_fast_whisper_api/cli/commands.py]`
    - Action: Implement helpers to write `.json` (existing logic), `.txt` (plain text), and `.srt` (using chunk timestamps). Ensure consistent filenames via `FilenameGenerator`.
    - Status: Done
  - [x] Ensure `_run_task()` returns paths of created files or logs them.

- [x] **Testing Phase:**
  - [x] CLI e2e tests
    - Path: `[tests/test_cli_exports.py]`
    - Action: Run CLI with the new flags on a short sample and assert that `.txt`/`.srt` files exist and non-empty.
    - **Status: Complete** - The test suite was fixed and now successfully validates the export functionality.
    - Accept Criteria: Tests pass in CI.

- [x] **Documentation Phase:**
  - [x] Update `project-overview.md` and README
    - Path: `[project-overview.md]`, `[README.md]`
    - Action: Mention new flags and show usage examples.
    - Accept Criteria: Docs reflect new functionality.

## Related Files

- `insanely_fast_whisper_api/cli/commands.py`
- `insanely_fast_whisper_api/webui/handlers.py` (reference logic)
- `insanely_fast_whisper_api/utils/filename_generator.py`
- `tests/conversion-test-file.mp3` (sample audio)

## Future Enhancements

- [ ] Add `--vtt` flag for WebVTT export
- [ ] Offer a single `--export fmt1,fmt2` option instead of multiple booleans
