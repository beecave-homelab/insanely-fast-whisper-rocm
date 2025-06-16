# Fix Gradio DownloadButton TypeError

**Objective:** Resolve the `TypeError: expected str, bytes or os.PathLike object, not function` occurring during Gradio UI updates for download buttons.

**File to Modify:** `/home/elvee/Local-AI/insanely-fast-whisper-rocm/insanely_fast_whisper_api/webui/handlers.py`

**Problem:**
The `process_transcription_request` function incorrectly returns new `gr.DownloadButton()` instances when it intends to update existing download button components in the UI. Gradio's internal `postprocess` method for `DownloadButton` then receives an entire `DownloadButton` component object instead of a file path or a callable returning a path, leading to the TypeError.

**Tasks:**

Within `process_transcription_request` in `insanely_fast_whisper_api/webui/handlers.py`:

- [x] **Modify `dl_btn_hidden` initialization:**
  - Change: `dl_btn_hidden = gr.DownloadButton(visible=False, interactive=False)` (approx. line 448)
  - To: `dl_btn_hidden_update = gr.update(visible=False, value=None, interactive=False)`
  - Update subsequent assignments: `zip_btn_update = txt_btn_update = srt_btn_update = json_btn_update = dl_btn_hidden_update`

- [x] **Modify error/no-results return paths:**
  - For lines returning `gr.DownloadButton(visible=False)` (e.g., approx. lines 439-444, and similar in other early exits for errors):
  - Change to: `gr.update(visible=False, value=None, interactive=False)` (Implemented by using `dl_btn_hidden_update`)

- [x] **Modify single-file success path button updates:**
  - For `txt_btn_update` (approx. line 494):
    - Change `gr.DownloadButton(...)` to `gr.update(...)`, using static label "Download .txt".
  - For `srt_btn_update` (approx. line 506):
    - Change `gr.DownloadButton(...)` to `gr.update(...)`, using static label "Download .srt".
  - For `json_btn_update` (approx. line 519):
    - Change `gr.DownloadButton(...)` to `gr.update(...)`, using static label "Download .json".
  - For `zip_btn_update` (single file, approx. line 554):
    - Change `gr.DownloadButton(...)` to `gr.update(...)`.

- [x] **Modify batch-file success path button updates:**
  - For `zip_btn_update` (batch mode, approx. line 628):
    - Change `gr.DownloadButton(...)` to `gr.update(...)`.
  - Ensure that `txt_btn_update`, `srt_btn_update`, `json_btn_update` in batch mode (for batch ZIPs) also use `gr.update(...)`. (Verified: TXT ZIP, SRT ZIP, JSON ZIP buttons were updated to use `gr.update`)

- [x] **Verify final return statement:**
  - Ensure the elements in the tuple returned by `process_transcription_request` that correspond to the download buttons are indeed these `gr.update()` dictionaries or the `dl_btn_hidden_update`.

- [x] Update `project-overview.md` (Not required for this bug fix, as it's not a structural/major functional change).
