# Fix ZIP Archive Overwrite Issue in WebUI

**Objective:** Resolve the issue where different types of ZIP archives (All, TXT, SRT, JSON) generated during multi-file transcription were overwriting each other due to non-unique filenames.

**File to Modify:** `/home/elvee/Local-AI/insanely-fast-whisper-rocm/insanely_fast_whisper_api/webui/handlers.py`

**Problem:**
In `process_transcription_request`, when multiple audio files are processed, several `BatchZipBuilder` instances are used to create different ZIP archives (e.g., "all formats", "TXT only", "SRT only", "JSON only"). The `create` method of `BatchZipBuilder` was being called with the same `batch_id` and no explicit unique `filename` for each type. This resulted in all ZIP files being created with the same default name (e.g., `batch_archive_<timestamp>.zip`), causing the last one generated to overwrite the others.

**Tasks:**

Within `process_transcription_request` in `insanely_fast_whisper_api/webui/handlers.py`:

- [x] **Ensure unique filenames for 'All Formats' ZIP:**
  - Modify the call to `all_zip_builder.create()` to pass a unique `filename` argument.
  - Example: `all_zip_filename = f"batch_archive_{timestamp_str}_all_formats.zip"`
  - Pass `filename=all_zip_filename` to `all_zip_builder.create()`.

- [x] **Ensure unique filenames for 'TXT Only' ZIP:**
  - Modify the call to `txt_zip_builder.create()` to pass a unique `filename` argument.
  - Example: `txt_zip_filename = f"batch_archive_{timestamp_str}_txt_only.zip"`
  - Pass `filename=txt_zip_filename` to `txt_zip_builder.create()`.

- [x] **Ensure unique filenames for 'SRT Only' ZIP:**
  - Modify the call to `srt_zip_builder.create()` to pass a unique `filename` argument.
  - Example: `srt_zip_filename = f"batch_archive_{timestamp_str}_srt_only.zip"`
  - Pass `filename=srt_zip_filename` to `srt_zip_builder.create()`.

- [x] **Ensure unique filenames for 'JSON Only' ZIP:**
  - Modify the call to `json_zip_builder.create()` to pass a unique `filename` argument.
  - Example: `json_zip_filename = f"batch_archive_{timestamp_str}_json_only.zip"`
  - Pass `filename=json_zip_filename` to `json_zip_builder.create()`.

- [x] **Fix 'Download All' button label:**
  - Modify the `label` for the main ZIP download button in batch mode.
  - Change from `label="Download All ({num_files} files) as ZIP"`
  - To: `label=f"Download All ({len(successful_results)} files) as ZIP"`

- [x] **Test thoroughly:**
  - Transcribe multiple files.
  - Verify that all four "Download ... (ZIP)" buttons (All, TXT, SRT, JSON) appear.
  - Download each ZIP file.
  - Confirm that each ZIP file has a unique name (e.g., includes `_all_formats`, `_txt_only`, etc.).
  - Confirm that each ZIP file contains the correct corresponding files (e.g., "TXT only" ZIP contains only .txt files).

- [x] Update `project-overview.md` (Not required for this bug fix, as it's not a structural/major functional change).
