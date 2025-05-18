---
trigger: glob
description: When working
globs: to-do/, to-do/*.md
---

# TO-DO FIRST POLICY

All work begins in `to-do/`. This directory is the source of truth for all tasks.

## Workflow Rules

- Before making any change, you must update or create a checklist in `to-do/<name>.md`
- Use `[ ]`, `[x]`, and `~~...~~` for tasks, subtasks, and abandoned work
- Include relevant file paths where possible
- Add a task to update `project-overview.md` once the feature is complete
- No work outside `to-do/` unless the task exists in the checklist

### Example

```markdown
- [ ] Build login screen (`LoginForm.tsx`)
- [x] Implement backend route (`api/login.js`)
- [ ] ~~Add biometric login~~ <!-- removed from scope -->
- [ ] Update `project-overview.md`
```

## Task and File Deletion

- Never delete checklist items unless explicitly approved by the user or needed for formatting
- Confirm all deletions first
- Only the user may delete files in `to-do/`

## Approval Required

All checklist updates must be confirmed by the user before implementation begins

## Always Repeat

Every new feature or fix:
1. Starts in `to-do/`
2. Follows this workflow
3. Ends with updated documentation
