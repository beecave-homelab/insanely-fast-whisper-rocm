# To-Do: Analyze and Align /v1/audio/transcriptions Output with OpenAPI Standard

This plan outlines the steps to analyze and update the API response for `/v1/audio/transcriptions` to ensure compatibility with the OpenAI Whisper endpoint, enabling MacWhisper and other clients to correctly parse results.

## Summary: Key Differences with OpenAI Whisper API

- **Response Format:**
  - OpenAI honors a `response_format` parameter (`text`, `json`, `verbose_json`, `srt`, `vtt`).
  - Current API always returns JSON, regardless of requested format.
- **Minimal JSON:**
  - OpenAI returns `{ "text": ... }` for `json` format.
  - Current API includes extra fields (e.g., `chunks`, `runtime_seconds`, etc.).
- **Verbose JSON:**
  - OpenAI returns `segments` (with timing and token info) and `language`.
  - Current API returns `chunks` (with timestamps/text), lacks `language`, and uses different field names.
- **Plain Text:**
  - OpenAI returns only the transcription as text if requested.
  - Current API may not support this, or always wraps in JSON.
- **SRT/VTT:**
  - OpenAI returns subtitle text in the requested format.
  - Current API support is unclear or missing.
- **Field Naming:**
  - OpenAI uses `segments` for detailed breakdowns; current API uses `chunks`.

### Table: Feature Comparison

| Feature                | OpenAI Whisper API                      | Current API                                 |
|------------------------|-----------------------------------------|---------------------------------------------|
| `response_format`      | text, json, verbose_json, srt, vtt      | json (default), others unclear              |
| Minimal JSON           | `{ "text": ... }`                       | `{ "text": ..., ...extra fields... }`       |
| Verbose JSON           | `segments`, `language`, `text`          | `chunks`, no `language`, extra fields       |
| Subtitle formats       | SRT/VTT supported                       | Unclear                                     |
| Field naming           | `segments` (verbose), `text` (simple)   | `chunks`, `text`, `runtime_seconds`, etc.   |

---

## Detailed Checklist for OpenAI Whisper API Compatibility

Below is a step-by-step checklist of all required changes and their code locations for aligning your API with OpenAI's `/v1/audio/transcriptions` endpoint:

1. **Support the `response_format` Parameter**
   - **Where:** `insanely_fast_whisper_api/api/routes.py` (`create_transcription`, `create_translation`)
   - **Action:** Parse and honor the `response_format` form parameter (`text`, `json`, `verbose_json`, `srt`, `vtt`). Pass it to the response formatting logic.

2. **Minimal JSON Response (`response_format=json`)**
   - **Where:** `insanely_fast_whisper_api/api/responses.py` (`ResponseFormatter.format_transcription`), `api/models.py`
   - **Action:** If `response_format=json`, return only `{ "text": ... }` (no extra fields).

3. **Verbose JSON Response (`response_format=verbose_json`)**
   - **Where:** `api/responses.py`, `api/models.py` (add `segments`, `language` if missing), transcription pipeline output
   - **Action:** Return a JSON object with `text`, `segments` (list of segment dicts with required fields), and `language`. Map `chunks` to `segments` and add missing fields.

4. **Plain Text Response (`response_format=text`)**
   - **Where:** `api/responses.py` (`format_transcription`)
   - **Action:** Return only the transcribed text as a plain text HTTP response.

5. **SRT/VTT Subtitle Responses (`response_format=srt` or `vtt`)**
   - **Where:** `api/responses.py` (add SRT/VTT formatting logic), transcription output
   - **Action:** Format and return transcription as SRT/VTT subtitle text. Set correct `Content-Type`.

6. **Field Naming and Structure**
   - **Where:** `api/responses.py`, `api/models.py`
   - **Action:** For verbose JSON, rename `chunks` to `segments` and ensure each segment matches OpenAI’s schema. Remove or conditionally include extra fields only in non-standard formats.

7. **Add/Extract Missing Fields**
   - **Where:** Transcription pipeline (language detection), `api/responses.py`
   - **Action:** Ensure detected language is included in verbose JSON. Populate all required segment fields, even if some are defaulted/approximated.

8. **Tests and Documentation**
   - **Where:** `tests/test_api.py`, `project-overview.md`, `README.md`
   - **Action:** Add/adjust tests for all supported formats. Document new behavior and any limitations.

9. **Content-Type Headers for Each Format**
   - **Where:** `api/routes.py`, `api/responses.py`
   - **Action:** Set correct `Content-Type` for `text/plain`, `application/json`, `text/srt`, `text/vtt` using FastAPI `response_class` or headers.

10. **Add Constants for New Response Formats**
   - **Where:** `insanely_fast_whisper_api/utils.py`
   - **Action:** Define constants/enums for `verbose_json`, `srt`, `vtt`, etc.

11. **OpenAPI Schema / response_model Adjustments**
   - **Where:** `api/models.py`, `api/routes.py`
   - **Action:** Update or conditionally remove `response_model` when returning non-JSON formats; regenerate schema docs.

12. **Error Handling for Unsupported `response_format`**
   - **Where:** `api/routes.py`, `api/responses.py`
   - **Action:** Return HTTP 400 when an unknown format is requested.

### Code Location Reference Table

| Task                             | Main File(s)                                   | Function/Class                |
|-----------------------------------|------------------------------------------------|-------------------------------|
| Parse `response_format`           | api/routes.py                                  | create_transcription          |
| Minimal JSON                      | api/responses.py, api/models.py                | format_transcription          |
| Verbose JSON                      | api/responses.py, api/models.py                | format_transcription          |
| Plain Text                        | api/responses.py                               | format_transcription          |
| SRT/VTT                           | api/responses.py                               | (add new formatters)          |
| Field renaming/structure          | api/responses.py, api/models.py                | format_transcription          |
| Add/extract missing fields        | core pipeline, api/responses.py                | as needed                     |
| Tests/documentation               | tests/test_api.py, project-overview.md, README | all relevant                  |
| Parse `response_format` (translation) | api/routes.py                                  | create_translation           |
| Content-Type handling              | api/routes.py, api/responses.py                | create_transcription/translation, ResponseFormatter |
| Format constants                   | insanely_fast_whisper_api/utils.py             | constant definitions         |
| OpenAPI schema adjustments         | api/models.py, api/routes.py                   | TranscriptionResponse / response_model |
| Invalid format error handling      | api/routes.py, api/responses.py                | create_* , ResponseFormatter |

---

## Tasks

- [ ] **Analysis Phase:**
  - [ ] Research and evaluate the current output structure and compare to OpenAI's OpenAPI spec
    - Path: `[insanely_fast_whisper_api/api/routes.py]`, `[insanely_fast_whisper_api/api/app.py]`
    - Action: Document the current response payload and log output
    - Analysis Results:
      - [ ] List of current fields returned
      - [ ] Reference OpenAI Whisper endpoint response format
      - [ ] Identify missing/extra fields or formatting issues
    - Accept Criteria: Clear mapping of differences and requirements for compatibility

- [ ] **Implementation Phase:**
  - [ ] Update API response to match OpenAI Whisper endpoint
    - Path: `[insanely_fast_whisper_api/api/routes.py]`
    - Action: Adjust response schema, field names, and values as needed
    - Status: Pending

- [ ] **Testing Phase:**
  - [ ] Unit or integration tests for new response format
    - Path: `[tests/test_api.py]`
    - Action: Test output against OpenAI-compatible clients (e.g., MacWhisper)
    - Accept Criteria: MacWhisper and other clients can parse the response without error

- [ ] **Documentation Phase:**
  - [ ] Update `project-overview.md` and/or README
    - Path: `[project-overview.md]`, `[README.md]`
    - Action: Document endpoint behavior, output format, and compatibility
    - Accept Criteria: Documentation is up-to-date and explains the new feature clearly

## Related Files

- `insanely_fast_whisper_api/api/routes.py`
- `insanely_fast_whisper_api/api/app.py`
- `tests/test_api.py`
- `project-overview.md`
- `README.md`

## Future Enhancements

- [ ] Add support for other OpenAI Whisper endpoint features (e.g., translation, SRT output, etc.)
