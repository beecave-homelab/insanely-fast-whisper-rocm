# To-Do: Enhance API Server CLI with Click

**Status: All phases (Analysis, Implementation, Testing, Documentation) COMPLETE.**

This plan outlines the steps to implement a Click-based command-line interface for the API server defined in `insanely_fast_whisper_api/__main__.py`. This will allow users to configure uvicorn settings such as host, port, and other parameters via CLI options, with default values sourced from `insanely_fast_whisper_api/utils/constants.py`.

## Tasks

- [x] **Analysis Phase:**
  - [x] Review `insanely_fast_whisper_api/cli` to understand the existing Click usage pattern.
    - Path: `insanely_fast_whisper_api/cli/`
    - Action: Identify best practices and common patterns for Click integration in this project.
    - Analysis Results:
      - [x] Document key Click features used (e.g., groups, commands, options, type casting, help messages).
          Main CLI entry point uses `@click.group()` and `@click.version_option()`.
          Commands are added via `cli.add_command(command_function)`.
          Commands use `@click.command()` with `short_help`.
          Arguments use `@click.argument()` with `type` and `metavar`.
          Options use `@click.option()` with short/long forms, `help`, `show_default=True`, `default` (often from `constants.py`), `type` (e.g., `click.Choice`, `click.Path`, `int`), and `is_flag=True` for booleans.
          `click.secho()` and `click.echo()` for output.
          Error handling with `try/except` and `click.secho(..., fg="red", err=True)`.
    - Accept Criteria: Clear understanding of how Click is utilized in the `cli` module.
  - [x] Analyze `insanely_fast_whisper_api/__main__.py` for current uvicorn configuration.
    - Path: `insanely_fast_whisper_api/__main__.py`
    - Action: Identify how uvicorn parameters (host, port, workers, log_level, etc.) are currently set or hardcoded.
    - Analysis Results:
      - [x] List current uvicorn parameters and their sources.
          `app`: Hardcoded to `"insanely_fast_whisper_api.main:app"`.
          `host`: Hardcoded to `"0.0.0.0"`.
          `port`: From `argparse` (`--port`), defaults to `8888`.
          `reload`: Hardcoded to `True`.
          `log_level`: Derived from `argparse` (`--verbose`), `"debug"` or `"info"`.
          `log_config`: Loaded from `logging_config.yaml`, modified by `--verbose`.
          `workers`: Not explicitly set (uvicorn default).
          `ssl_keyfile`: Not explicitly set.
          `ssl_certfile`: Not explicitly set.
    - Accept Criteria: Full list of uvicorn parameters to be made configurable.
  - [x] Identify relevant default values in `insanely_fast_whisper_api/utils/constants.py`.
    - Path: `insanely_fast_whisper_api/utils/constants.py`
    - Action: Find constants related to API server host, port, etc.
    - Analysis Results:
      - [x] List constants to be used as default values for Click options.
        - `API_HOST`: Default `"0.0.0.0"` (from `os.getenv("API_HOST", "0.0.0.0")`)
        - `API_PORT`: Default `8000` (from `int(os.getenv("API_PORT", "8000"))`)
        - `LOG_LEVEL`: Default `"INFO"` (from `os.getenv("LOG_LEVEL", "INFO")`)
        - No explicit constants for `workers`, `reload` flag, `ssl_keyfile`, `ssl_certfile`. Sensible defaults will be used directly in Click options.
    - Accept Criteria: Complete mapping of constants to Click options.

- [x] **Implementation Phase:**
  - [x] Add `click` as a dependency if not already present for the main API entrypoint (check `pyproject.toml`).
    - Path: `pyproject.toml`
    - Action: Ensure `click` is listed under `[project.dependencies]`.
    - Status: Completed (already present)
  - [x] Refactor `insanely_fast_whisper_api/__main__.py` to use `click`.
    - Path: `insanely_fast_whisper_api/__main__.py`
    - Action:
      - [x] Import `click`.
      - [x] Define a main `click.group()` or `click.command()` for launching the API server.
      - [x] Add `click.option()` for uvicorn parameters:
        - `host` (default from `constants.py` or a sensible default like `DEFAULT_API_HOST`)
        - `port` (default from `constants.py` or `DEFAULT_API_PORT`)
        - `workers` (default from `constants.py` or `DEFAULT_API_WORKERS`)
        - `log_level` (default from `constants.py` or `DEFAULT_API_LOG_LEVEL`)
        - `reload` (boolean flag, default False)
        - `ssl_keyfile` (optional string)
        - `ssl_certfile` (optional string)
      - [x] Ensure help messages for options are clear and informative.
      - [x] Update the uvicorn server startup logic to use the values from these Click options.
      - [x] Enhance startup messages using `click.secho` for a more user-friendly and visually appealing output, similar to the main CLI.
    - Status: Completed

- [x] **Testing Phase:**
  - [x] Test the new CLI interface.
    - Path: Terminal
    - Action:
      - [x] Run `python -m insanely_fast_whisper_api --help` to verify options and help messages.
      - [x] Launch the server with default options: `python -m insanely_fast_whisper_api` (Verified: Started on [http://0.0.0.0:8888](http://0.0.0.0:8888), log level info)
      - [x] Launch the server with custom port: `python -m insanely_fast_whisper_api --port 8001` (Verified: Started on [http://0.0.0.0:8001](http://0.0.0.0:8001))
      - [x] Launch the server with custom host: `python -m insanely_fast_whisper_api --host 0.0.0.0` (Verified: Started on [http://0.0.0.0:8888](http://0.0.0.0:8888))
      - [x] Launch the server with a specified number of workers: `python -m insanely_fast_whisper_api --workers 4 --no-reload` (Verified: Started on [http://0.0.0.0:8888](http://0.0.0.0:8888) with 4 workers, reload explicitly false)
      - [x] Test `--reload` flag (Verified: Started on [http://0.0.0.0:8888](http://0.0.0.0:8888) with reload enabled, watching for changes)
      - [x] Test `--log-level debug` (Verified: Started on [http://0.0.0.0:8888](http://0.0.0.0:8888) with Uvicorn log level set to debug)
      - [ ] ~~Test `--ssl-keyfile dummy.key --ssl-certfile dummy.crt` (create dummy files first, expect server to attempt HTTPS).~~ (Skipped by user)
      - [x] Test `--debug` flag (Verified: Started on [http://0.0.0.0:8888](http://0.0.0.0:8888), Uvicorn log level set to 'debug', application logs show DEBUG messages, Click startup message confirms debug mode)
    - Status: Help message verified.

- [x] **Documentation Phase:**
  - [x] Update `project-overview.md` if server launch instructions change significantly.
    - Path: `project-overview.md`
    - Action: Reflect new CLI options for starting the API server.
    - Accept Criteria: Documentation accurately describes how to launch and configure the API server.
  - [x] Update README.md if it contains server launch instructions.
    - Path: `README.md`
    - Action: Ensure README instructions are consistent with the new Click-based CLI.
    - Accept Criteria: README is up-to-date.

## Related Files

- `insanely_fast_whisper_api/__main__.py`
- `insanely_fast_whisper_api/utils/constants.py`
- `pyproject.toml` (potentially)
- `project-overview.md` (potentially)
- `README.md` (potentially)

## Future Enhancements

- [ ] Consider adding subcommands if more server-related actions are needed in the future (e.g., `python -m insanely_fast_whisper_api config show`).
