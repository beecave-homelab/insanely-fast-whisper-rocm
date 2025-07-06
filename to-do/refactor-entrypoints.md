# To-Do: Refactor Entrypoints to `__main__.py`

This plan outlines the steps to refactor the `webui`, `api`, and `cli` modules to use `__main__.py` for their startup logic. This will allow them to be run as packages using `python -m`.

## Tasks

- [x] **Refactor `webui` Module:**
  - [x] Move startup logic from `insanely_fast_whisper_api/webui/cli.py` to a new `insanely_fast_whisper_api/webui/__main__.py` file.
  - [x] Delete the now-empty `insanely_fast_whisper_api/webui/cli.py`.
  - [x] Update any references to the old `cli.py` file.

- [x] **Refactor `api` Module:**
  - [x] Move startup logic from `insanely_fast_whisper_api/api/app.py` to a new `insanely_fast_whisper_api/api/__main__.py` file.
  - [x] Ensure the `api` can still be run correctly after the move.

- [x] **Refactor `cli` Module:**
  - [x] Move startup logic from `insanely_fast_whisper_api/cli/cli.py` to a new `insanely_fast_whisper_api/cli/__main__.py` file.
  - [x] Ensure the `cli` can still be run correctly after the move.

## Related Files

- `insanely_fast_whisper_api/webui/cli.py`
- `insanely_fast_whisper_api/webui/__main__.py` (new)
- `insanely_fast_whisper_api/api/app.py`
- `insanely_fast_whisper_api/api/__main__.py` (new)
- `insanely_fast_whisper_api/cli/cli.py`
- `insanely_fast_whisper_api/cli/__main__.py` (new)
