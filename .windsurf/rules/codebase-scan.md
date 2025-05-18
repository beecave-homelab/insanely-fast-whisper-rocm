---
trigger: always_on
---

# Codebase Scan Policy

## Initial Context Gathering

- Begin by reading `project-overview.md` to understand the codebase structure.
- Do not read through every file listed unless required for the user’s request or question.

## Documentation Update

- After any structural, behavioral, or functional change to the codebase, update `project-overview.md` to reflect those changes.
- Never skip this update, even for small refactors.

## Check-in Requirement

- Once the scan is complete, check in with the user before doing anything else.
- Provide a summary of:
  - The codebase structure
  - Key files and components
  - Interdependencies or patterns discovered
- Await explicit user approval before proceeding.

## Enforcement

Do not propose, modify, refactor, optimize, or generate new code until:

- You have demonstrated clear contextual understanding
- The user has explicitly approved your next step
