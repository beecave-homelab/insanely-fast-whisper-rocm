---
description: Workflow to follow when the user asks you to write pull requests.
---

# Pull Request Workflow

## Communication

AI assistance is enabled.

## Coding

- You must determine the current Git branch using `git branch --show-current`, unless explicitly overridden.
- You must run `git --no-pager diff <branch_name>` yourself to retrieve code changes.
- You only ask the user to run the command if it fails.
- You must use `git diff --name-status` to determine file status as Added, Modified, or Deleted.
- You must categorize all changes as: **Added**, **Modified**, or **Deleted**.
- You must interpret differences and summarize them in clear language.
- You must explain each change with reasoning, impacts, and a testing plan.
- You must include file-specific details and code snippets where applicable.
- If no meaningful changes are detected (i.e. diff is empty or trivial), you must abort the workflow and notify the user.

## Commit Messages

- You must only use the following commit types:
  - `feat ✨`: A new feature
  - `fix 🐛`: A bug fix
  - `docs 📝`: Documentation only changes
  - `style 💎`: Code style changes (formatting, missing semi-colons, etc.)
  - `refactor ♻️`: Code changes that neither fix a bug nor add a feature
  - `test 🧪`: Adding missing tests or correcting existing tests
  - `chore 📦`: Changes to the build process or auxiliary tools
  - `revert ⏪`: Reverts a previous commit

## Pull Request Formatting

- You must follow the exact Markdown format shown in the **Pull Request Template Format** section.
- You must fill in every section of the template with details derived from the diff output.
- You must use clear, consistent formatting and maintain section headers.

## Filesystem

- You must describe file changes under their respective categories: Added, Modified, Deleted.
- You must highlight key code changes per file and explain their purpose.

## Output Rules

- You must write the final pull request in Markdown format.
- You must use only the allowed commit types and emojis.
- You must follow the section structure exactly as defined in this ruleset.
- You must save the pull request as a file in the `.github/PULL_REQUEST/` directory.
- You must name the file using the pattern: `pr-<commit_type>-<short_name>-merge.md`.  
  Example: `pr-feature-badgeai-merge.md`
- You must also show the pull request to the user in a nested markdown codeblock using ` ````md `.

## Final Step: Dry-Run PR via GitHub CLI

- After the user confirms the pull request message is complete, you must simulate a pull request using the `gh` command line tool.
- You must use only allowed commit types in the `--title`.
- You must use the generated `.md` file for the `--body-file`.
- Example command format:

```sh
gh pr create --base dev --head frontmatter-badge-ai --draft --title "feat ✨: Add AI Usage Badge System via Frontmatter" --body-file .github/PULL_REQUEST/pr-feature-badgeai-merge.md
```

## Pull Request Template Format

````markdown
# Pull Request: [Short Title for the PR]

## Summary

Provide a brief and clear summary of the changes made in this pull request. For example:  
"This PR introduces [feature/fix] to achieve [goal]. It includes changes to [describe major components]."

---

## Files Changed

### Added

1. **`<file_name>`**  
   - Description of what was added and its purpose.

### Modified

1. **`<file_name>`**  
   - Description of what was modified and why. Include relevant details.

### Deleted

1. **`<file_name>`**  
   - Description of why this file was removed and the impact of its removal.

---

## Code Changes

### `<file_name>`

```<language>
# Provide a snippet of significant changes in the file if applicable.
# Highlight key changes, improvements, or new functionality.
```

- Explain the code changes in plain language, such as what functionality was added or modified and why.

---

## Reason for Changes

Provide the reasoning for making these changes. For example:  
- Fixing a bug  
- Adding a new feature  
- Refactoring for better performance or readability  

---

## Impact of Changes

### Positive Impacts

- List benefits, such as improved performance, new functionality, or bug fixes.

### Potential Issues

- Mention any known risks, dependencies, or edge cases introduced by these changes.

---

## Test Plan

1. **Unit Testing**  
   - Describe how unit tests were added or modified.  
   - Mention specific scenarios covered.

2. **Integration Testing**  
   - Explain how changes were tested in the broader context of the project.  

3. **Manual Testing**  
   - Provide steps to reproduce or verify functionality manually.

---

## Additional Notes

- Add any relevant context, known limitations, or future considerations.
- Include suggestions for enhancements or follow-up work if applicable.
