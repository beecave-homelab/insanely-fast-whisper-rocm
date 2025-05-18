---
description: Perform a codebase scan before making any changes.
---

# Codebase Analysis & Management Protocol

## BEFORE MAKING ANY CHANGES

### Initial Understanding Phase

- Perform a thorough examination of the codebase
- Read the `project-overview.md` file to understand the codebase structure
- Document the containerization approach using Docker and Docker Compose
- Map interdependencies between key components
- Identify core files and their functions

### Check-in Requirements

- Present a concise summary of codebase understanding
- Outline the structure, architecture, and project organization
- Highlight key files and their relationships
- Explain the containerization setup for development and production
- Wait for explicit user confirmation before proceeding further
- Do not propose any modifications, optimizations, refactors, or new code until this phase is complete and approved

## AFTER MAKING ANY CHANGES

### Post-Change Documentation

- Update the `project-overview.md` file to reflect all modifications made
- Document any minor OR major changes implemented in the codebase
- Record modifications to Docker/Docker Compose configurations
- Note new dependencies or structural changes
- Ensure the project overview remains accurate and up-to-date

### Change Summary

- Provide a detailed explanation of all implemented changes
- Explain the rationale behind modifications
- Document any potential impacts on other parts of the system
- Include instructions for testing the changes if applicable