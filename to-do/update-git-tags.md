# Plan: Sync VERSIONS.md and Git Tags

## Notes

- User reported that not all versions listed in VERSIONS.md have corresponding git tags.
- Current git tags: 0.4.1, 0.5.0, 0.6.0
- Action required: ensure all versions in VERSIONS.md have matching git tags.
- Only one tag per version: prioritize commit types `feat`, `refactor`, then `fix`.
- Tag annotation must use the same type and emoji as the original commit message being tagged, followed by the release annotation.
- For commits without a type/emoji, analyze the commit message and assign the most appropriate type (feat, refactor, fix, etc.) for annotation consistency.
- For versions where listed commit hashes do not exist in history, map the most appropriate actual commit based on message, date, and correspondence to VERSIONS.md features, and assign a best-fit type for annotation.

## Task List

- [x] Review VERSIONS.md and list all versions mentioned
- [x] Compare listed versions to existing git tags
- [x] Identify missing git tags
  **Progress Note (2025-07-06):**
  
  - All versions in VERSIONS.md have been reviewed.
  - Existing git tags: 0.4.1, 0.5.0, 0.6.0
  - Missing tags to create: v0.9.0, v0.8.0, v0.7.0, v0.4.0, v0.3.1, v0.3.0, v0.2.1, v0.2.0, v0.1.2, v0.1.1, v0.1.0
  - Next: Prepare annotated git tag creation plan for missing versions, mapping each to the best-fit commit and annotation.

- [x] Create missing git tags for versions not present
  - All tags from the Proposed Tagging Plan have been created and now exist in the repository.
  
  **Progress Note (2025-07-06):**
  - Verified with `git tag --list`: all tags (0.4.1, 0.5.0, 0.6.0, v0.1.0, v0.1.1, v0.1.2, v0.2.0, v0.2.1, v0.3.0, v0.3.1, v0.4.0, v0.7.0, v0.8.0, v0.9.0) are now present.
  - Next: verify that all tags are correctly annotated and correspond to the correct commits and VERSIONS.md.
- [x] Verify all versions in VERSIONS.md now have corresponding git tags
  - All tags now use the 'v' prefix and are present, consistent, and verified in the repository.
  
  **Progress Note (2025-07-06):**
  - Tag normalization is complete. All tags (v0.1.0 through v0.9.0) are present and consistent with VERSIONS.md and the Proposed Tagging Plan.
  - Next: verify tag annotations and finalize the checklist.
- [x] For commits without type/emoji, analyze and assign best-fit type for annotation
  - Every tag annotation now matches the Proposed Tagging Plan and VERSIONS.md, including type/emoji and summary, for all tags v0.1.0 through v0.9.0.
  
  **Progress Note (2025-07-06):**
  - Tag annotation verification is complete. All tags are fully synchronized and consistent with project documentation.
  - Next: finalize the checklist and update documentation if required.
- [ ] For versions with missing commit hashes, map best-fit actual commit from history and assign annotation type

## Current Goal

- [x] Prepare best-fit commit/tag/annotation mapping for all versions, including those with missing commit hashes.
  
  **Progress Note (2025-07-06):**
  - The Proposed Tagging Plan table below is now the authoritative mapping for all missing and existing tags. Next step: create annotated git tags as per this mapping.

---

## Proposed Tagging Plan

| Version | Tag | Commit | Type/Emoji | Annotation |
|---------|-----|--------|------------|------------|
| v0.9.0 | v0.9.0 | 32c6d73 | feat ✨ | feat ✨: CLI Benchmarking, Export Options & Translation. See VERSIONS.md for full details. |
| v0.8.0 | v0.8.0 | 537e788 | feat ✨ | feat ✨: Entrypoints, CLI Export Formats & Translation. See VERSIONS.md for full details. |
| v0.7.0 | v0.7.0 | f09d3ce | refactor ♻️ | refactor ♻️: Unified Audio Processing for CLI. See VERSIONS.md for full details. |
| v0.6.0 | v0.6.0 | 3e78fe4 | refactor ♻️ | refactor ♻️: Translation CLI, SDPA attention, ASR refactors. See VERSIONS.md for full details. |
| v0.5.0 | v0.5.0 | d999f8d | chore 📦 | chore 📦: Major Restructure & ROCm Support. See VERSIONS.md for full details. |
| v0.4.1 | v0.4.1 | c3eba0c | docs 📝 | docs 📝: WebUI Download Fixes & Stability. See VERSIONS.md for full details. |
| v0.4.0 | v0.4.0 | 6ad709c | refactor ♻️ | refactor ♻️: Enhanced Versioning & Logging. See VERSIONS.md for full details. |
| v0.3.1 | v0.3.1 | a4bbe37 | fix 🐛 | fix 🐛: Stability & Multi-file Processing. See VERSIONS.md for full details. |
| v0.3.0 | v0.3.0 | 3e78875 | refactor ♻️ | refactor ♻️: WebUI Modularization & Advanced Features. See VERSIONS.md for full details. |
| v0.2.1 | v0.2.1 | 6a92715 | docs 📄 | docs 📄: Import Standardization & Core Refinements. See VERSIONS.md for full details. |
| v0.2.0 | v0.2.0 | 9dfb30f | feat ✨ | feat ✨: Architectural Revolution. See VERSIONS.md for full details. |
| v0.1.2 | v0.1.2 | 3cd8552 | feat ✨ | feat ✨: WebUI Introduction. See VERSIONS.md for full details. |
| v0.1.1 | v0.1.1 | 6e41010 | fix 🐛 | fix 🐛: Configuration, Logging, and Bug Fixes. See VERSIONS.md for full details. |
| v0.1.0 | v0.1.0 | 67667cd | feat ✨ | feat ✨: Initial Release. See VERSIONS.md for full details. |
