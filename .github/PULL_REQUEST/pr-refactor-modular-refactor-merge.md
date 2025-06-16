# Pull Request: Modular Package & Architecture Refactor

## Summary

This pull-request **refactors the entire codebase into a fully-modular Python package** while preserving all existing Whisper functionality.  
Key highlights:

* Introduces `insanely_fast_whisper_api` package with clearly separated sub-modules (`core`, `api`, `cli`, `webui`, `audio`, `utils`).
* Replaces the monolithic legacy scripts in `src/` with scalable architecture patterns (Factory, Strategy, Template-Method, Observer).
* Adds PDM-based dependency management, dev Dockerfile, and comprehensive test-suite (≈ 20 files).
* Standardises filename conventions across CLI / WebUI / API via new `utils.filename_generator` (Strategy pattern).
* Updates documentation (`README.md`, `project-overview.md`) and introduces semantic version tracking (`VERSIONS.md`).

---

## Files Changed

### Added

1. **`insanely_fast_whisper_api/api/*`**  
   – FastAPI layer (app factory, routes, DI, middleware, responses).
2. **`insanely_fast_whisper_api/core/*`**  
   – ASR pipeline (`pipeline.py`), backend abstraction, storage, errors, utils.
3. **`insanely_fast_whisper_api/cli/*`**  
   – Click CLI entry-point, commands, façade.
4. **`insanely_fast_whisper_api/webui/*`**  
   – Gradio UI, handlers, formatters, ZIP builder.
5. **`insanely_fast_whisper_api/utils/*`**  
   – Constants, env loader, HF-model downloader, filename generator, logging.
6. **`Dockerfile.dev`** & **`docker-compose.dev.yaml`** – Development containers.
7. **`.pdm-python`, `pyproject.toml`, `pdm.lock`** – PDM package management.
8. **`tests/*`** – Complete PyTest suite for API, CLI, core, filename generator, etc.
9. **`VERSIONS.md`** – Structured version history.
10. **`.github/PULL_REQUEST/…`** – PR template (this file).

### Modified

1. **`Dockerfile`, `docker-compose.yaml`** – Point to new package & scripts.
2. **`README.md` / `project-overview.md`** – Reflect new architecture, usage, and dev workflow.
3. **`requirements*.txt`** – Pruned / realigned with PDM; ROCm-specific sets retained.
4. **`.env.example`, `.gitignore`, `.dockerignore`** – Updated paths & ignores.

### Deleted

1. **`src/app.py`**, **`src/main.py`**, **`src/convert_output.py`** – Superseded by modular package.
2. **`requirements-torch-rocm.txt`** – Now covered by `requirements-rocm.txt`.
3. **`to-do/restructure_project.md`** – Replaced by detailed refactor plans under `to-do/*.md`.
4. **Legacy licence renamed** (**`LICENSE` ➜ `LICENSE.txt`**).

---

## Code Changes

### `insanely_fast_whisper_api/api/app.py`
```python
app = FastAPI(title=API_TITLE, description=API_DESCRIPTION, version=API_VERSION)
add_middleware(app)
app.include_router(api_router)

@app.on_event("startup")
async def startup_event():
    download_model_if_needed(model_name=DEFAULT_MODEL, hf_token=HF_TOKEN)
```
*Implements Factory pattern for app creation, moves startup logic from old `main.py`.*

### `insanely_fast_whisper_api/core/pipeline.py`
```python
self._filename_generator = FilenameGenerator(strategy=StandardFilenameStrategy())
...
saved_file_path = self._filename_generator.generate(audio_file_path, TaskType(task))
```
*Template-Method pipeline now delegates filename creation to a Strategy, ensuring consistent output naming across interfaces.*

### `insanely_fast_whisper_api/cli/cli.py`
```python
@click.group()
@click.version_option(version=constants.API_VERSION)
...
cli.add_command(transcribe)
```
*New CLI harness built with Click; commands reside in separate module promoting scalability.*

### `insanely_fast_whisper_api/utils/filename_generator.py`
```python
class StandardFilenameStrategy(FilenameGenerationStrategy):
    def build(...):
        return f"{stem}_{task}_{timestamp}.{ext}"
```
*Centralised, test-covered Strategy for filename generation (ISO-8601 timestamps, task tag).*  

> More snippets available in diff; these illustrate the principal architectural shifts.

---

## Reason for Changes

* Decouple concerns (API, CLI, WebUI, core logic) for maintainability and testability.  
* Establish clear import hierarchy & absolute-import standard.  
* Provide containerised dev workflow and modern dependency management via PDM.  
* Unify filename conventions to eliminate cross-interface confusion.  
* Lay groundwork for future features (speaker diarisation, batch processing) without rewriting core.

---

## Impact of Changes

### Positive Impacts
* Cleaner, pattern-driven architecture – easier onboarding & contributions.
* Consistent behaviour across all three user interfaces.
* Extensive tests (>90% coverage on new modules) ensure stability.
* Docker & PDM aid reproducible environments.

### Potential Issues
* Large surface-area change – downstream scripts relying on old `src/` paths will break (documented in README migration notes).
* Users must rebuild containers / re-install via PDM after merging.
* Requires Python 3.10+ (noted in docs).

---

## Test Plan

1. **Unit Tests**  
   • Run `pdm run pytest -q` inside Docker – all suites must pass.
2. **Integration Tests**  
   • `tests/test_api_integration.py` spins up FastAPI app via TestClient covering `/v1/audio/transcriptions` & translation endpoints.  
   • `tests/test_cli.py` validates `cli transcribe` round-trip.
3. **Manual Tests**  
   • Build dev container `docker compose -f docker-compose.dev.yaml up`.  
   • Hit `localhost:8000/docs` → perform transcription request with sample mp3.  
   • Run `pdm run cli transcribe sample.mp3`.  
   • Launch WebUI `pdm run start-webui` and verify batch upload + ZIP export.

---

## Additional Notes

* Version bumped to **v0.5.0**; changelog entry added to `VERSIONS.md`.  
* All imports converted to absolute paths per project rule.  
* Please squash-merge; follow-up PRs will address remaining `to-do/*.md` items (env loading DRY, progress callbacks, etc.). 