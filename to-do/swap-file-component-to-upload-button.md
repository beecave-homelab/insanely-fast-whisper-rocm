# To-Do: Swap gr.File Component to gr.UploadButton (Attempted & Reverted)

This plan outlines the steps to replace the `gr.File` component with `gr.UploadButton` in the Insanely Fast Whisper API web UI to potentially enhance the file upload experience.
**Conclusion: This change was attempted but reverted due to issues with filename preservation for special characters (e.g., `()`, `[]`) when using `gr.UploadButton` with `os.path.basename()`. The original `gr.File` implementation using `GradioFileDataEvent.orig_name` was deemed more robust for this specific requirement.**

## Tasks

- [x] **Analysis Phase:**
  - [x] Research and evaluate the differences between `gr.File` and `gr.UploadButton`
    - Path: [insanely_fast_whisper_api/webui/ui.py]
    - Action: Analyze the current implementation of `gr.File` and how `gr.UploadButton` can be integrated. Refer to the Gradio documentation for `gr.File` (https://www.gradio.app/docs/gradio/file) which states it 'Creates a file component that allows uploading one or more generic files' and for `gr.UploadButton` (https://www.gradio.app/docs/gradio/uploadbutton) which mentions it is 'Used to create an upload button, when clicked allows a user to upload files that satisfy the specified file type'.
      - Verify how `gr.UploadButton` provides original filenames when `file_count="multiple"` and `type="filepath"`, and adapt the `handle_file_uploads` function or similar logic in `insanely_fast_whisper_api/webui/ui.py` accordingly.
    - Analysis Results:
      - The current `gr.File` component's `.upload` event provides `List[GradioFileDataEvent]`, where each event object has an `orig_name` attribute used by `handle_file_uploads`. This `orig_name` likely reflects the filename as sent by the browser, potentially with minimal server-side sanitization.
      - According to the `gr.UploadButton` documentation ([https://www.gradio.app/docs/gradio/uploadbutton](https://www.gradio.app/docs/gradio/uploadbutton)):
        - When `type="filepath"` (the default), the component's value (passed to its `.upload` event handler if the button itself is an input) will be a `str` (for single file) or `List[str]` (for `file_count="multiple"`) of temporary file paths.
        - The documentation for the `type` parameter explicitly states: "`filepath` returns a temporary file object with the same base name as the uploaded file..." This confirms that `os.path.basename(temp_file_path_string)` will yield the original filename *as it exists in the temporary file's path*.
      - **Uncertainty**: While `os.path.basename()` will correctly extract the basename from the temp path, it's not explicitly detailed if Gradio sanitizes special characters (like `( ) [ ]`) from the original filename *before* creating the temporary file's basename. If sanitization occurs, `os.path.basename()` would return the sanitized version, potentially losing some original characters. The `gr.File` component's `orig_name` might be more robust in this specific scenario.
      - Therefore, the `handle_file_uploads` function in `insanely_fast_whisper_api/webui/ui.py` would need modification if proceeding.
      - The swap was initially deemed feasible, with the caveat that special character handling needed verification.
    - Accept Criteria: Clear understanding of the impact on file upload handling and event processing, especially considering `gr.UploadButton` supports specific event listeners like `.upload()` which could streamline file handling compared to `gr.File`.

- [-] **Implementation Phase:** (Marked as reverted)
  - [ ] Replace `gr.File` with `gr.UploadButton` in the UI (Attempted)
    - Path: [insanely_fast_whisper_api/webui/ui.py]
    - Action: Update the UI component and adjust event handlers to accommodate `gr.UploadButton`. Ensure compatibility with file type restrictions and multiple file uploads as noted in the documentation where `file_count` can be set to 'multiple' (https://www.gradio.app/docs/gradio/uploadbutton).
    - Status: Reverted. Code changes were rolled back.

- [-] **Testing Phase:** (Led to reversion)
  - [x] Test file upload functionality with `gr.UploadButton`
    - Path: [tests/test_webui.py]
    - Action: Create or update tests to verify file uploads work correctly with the new component, ensuring the upload event triggers as expected based on the `.upload()` listener described in the Gradio docs (https://www.gradio.app/docs/gradio/uploadbutton).
    - Action: Specifically test uploads with filenames containing common special characters (e.g., spaces, parentheses `()`, square brackets `[]`) to ensure `os.path.basename()` on the temporary file path preserves these characters as expected and matches the fidelity of the current `gr.File`'s `orig_name`.
      - **Test Result (from user-provided logs & clarification)**: User uploaded files `conversion-test-file (2).mp3`, `conversion-test-file [2].mp3`, and `conversion-test-file.mp3`. The logs indicated that the derived `orig_name` using `os.path.basename()` on the temporary file paths resulted in `conversion-test-file 2.mp3` for the files with parentheses and brackets, and `conversion-test-file.mp3` for the plain one. 
      - **Conclusion**: This test shows that `os.path.basename()` with `gr.UploadButton` **does not preserve parentheses `()` or square brackets `[]`** in the same way `gr.File` with `GradioFileDataEvent.orig_name` likely did. These characters appear to be sanitized (stripped/replaced) when the temporary file is named by Gradio. This is a regression in filename preservation fidelity.
    - Accept Criteria: All file upload and processing functionalities work as expected with no regressions. Original filenames, including common special characters, are preserved with at least the same fidelity as the current implementation.
      - **Status**: **Failed**. Filenames with parentheses `()` and square brackets `[]` were NOT preserved correctly by the `gr.UploadButton` approach, leading to the decision to revert.

- [ ] **Documentation Phase:** (Not applicable as change was reverted)
  - [ ] Update project-overview.md
    - Path: [project-overview.md]
    - Action: Document the change in file upload component and any user-facing changes.
    - Accept Criteria: Documentation reflects the updated UI component and explains the new upload process clearly.

## Related Files

- insanely_fast_whisper_api/webui/ui.py
- tests/test_webui.py
- project-overview.md

## Future Enhancements

- [ ] Explore additional UI improvements for file selection and upload progress visualization (original item).
- [ ] If future Gradio versions improve `gr.UploadButton`'s ability to expose un-sanitized original filenames, this swap could be revisited. 