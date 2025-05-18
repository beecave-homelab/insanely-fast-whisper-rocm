---
description: Workflow when working on to-do files.
---

# Project Task Management Rules

## Communication

- AI assistance is enabled.

## Task Management

- **Source of Truth**: The `to-do/` directory is the authoritative source for all planned and in-progress tasks.
- **Task Initiation**: Before starting any work, consult the relevant `to-do/*.md` files to understand current priorities and workflows.
- **Task Creation**: If no existing `to-do` file corresponds to the intended change, create a new Markdown file in the `to-do/` directory detailing the scope, goals, and steps of the task.
- **Scope Enforcement**: Only work on tasks explicitly mentioned in `to-do` files. If a user request is out of scope, a new `to-do` file must be created and approved before implementation begins.
- **Checklist Format**: All `to-do` files must be structured as Markdown checklists with clear tasks and subtasks.
  - Each item should reference specific files and their paths.
  - Use clear, actionable language for each checklist item.
- **Project Overview Update**: Every `to-do` file must include a task to update the `project-overview.md` file upon feature completion to ensure documentation remains current.
- **Approval Process**: After updating or creating a `to-do` file, seek user confirmation before proceeding with any code changes. Await explicit approval before starting implementation.
- **Completion Tracking**: Upon task completion and verification, update the relevant `to-do` file to mark items as done. Add clarifications or notes as needed to maintain an accurate project history.
- **File Deletion Policy**: `to-do` files are never deleted by the agent. Only the user may delete them manually.
- **Iterative Process**: For every new feature, bugfix, or workflow change, repeat this process, ensuring the `to-do` directory remains the central coordination point for all work.

## Filesystem

- **Protected Path**: `to-do/`
  - Modifications to files in this directory should follow the task management rules outlined above.
