# TODO: Add `translate` Command to CLI

This plan outlines the steps required to expose the existing translation functionality through a new `translate` command in the CLI.

## 1. Create the `translate` Command

- **File:** `insanely_fast_whisper_api/cli/commands.py`
- **Action:**
  - [ ] Create a new `click` command function named `translate`.
  - [ ] This function will be based on the existing `transcribe` command.
  - [ ] It must call the `CLIFacade.transcribe_audio` method with the `task` parameter explicitly set to `"translate"`.
  - [ ] The command should accept the same options as `transcribe` where applicable (e.g., `--file`, `--model`, `--device`, `--language`). Options that are irrelevant to translation can be omitted.

## 2. Register the `translate` Command

- **File:** `insanely_fast_whisper_api/cli/cli.py`
- **Action:**
  - [ ] Import the newly created `translate` command from `insanely_fast_whisper_api.cli.commands`.
  - [ ] Add the `translate` command to the `main` `click.group()` so it becomes a registered CLI command.

## 3. Verification

- **Action:**
  - [ ] After implementation, run `python -m insanely_fast_whisper_api.cli.cli --help` to confirm the `translate` command appears in the list of available commands.
  - [ ] Run a test translation from the CLI to ensure it works end-to-end.
