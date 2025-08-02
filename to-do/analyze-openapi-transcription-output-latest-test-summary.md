# Integration Test Results (2025-07-08, after test updates)

- **8 passed, 3 failed**

## ✅ Fixed (OpenAI spec alignment)

- Minimal JSON response tests for both `/v1/audio/transcriptions` and `/v1/audio/translations` now expect only `{ "text": ... }`, matching the real OpenAI Whisper API.
- Tests for `metadata` and `transcription` fields in minimal JSON have been removed.

## ❌ Remaining Failures

1. **test_transcription_endpoint_success_text**
   - Expects `Content-Type: text/plain; charset=utf-8`, but API returns `application/json`.
   - **Action:** Update API response formatting to set correct content-type for plain text.
2. **test_transcription_endpoint_dependency_injection**
   - Model name override not honored; expected `custom-model`, got `openai/whisper-medium`.
   - **Action:** Review/fix dependency injection or test setup.
3. **test_parameter_names_unchanged**
   - Model name override not honored; expected `test-model`, got `openai/whisper-medium`.
   - **Action:** Review/fix dependency injection or test setup.

## Next Actions

- [ ] Fix plain text response to return correct content-type header.
- [ ] Investigate and fix dependency injection/test override issues for model name in integration tests.

---

_This summary should be merged into the main to-do file upon next structural update to avoid context loss._
