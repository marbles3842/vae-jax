---
name: git-commit
description: >-
  Commit currently staged changes only. Use when the user says "commit", "commit changes", "commit my changes", "git commit", "save changes", or as the first step of
  "commit and push". Verifies that changes are staged, checks for atomic cohesive edits,
  generates a simple, short, but meaningful commit message, and runs the commit script.
---

# git-commit

## Overview

This skill commits **currently staged changes only** (the git index). It does not automatically stage unstaged changes. It encourages atomic, logically-cohesive commits and generates clean, simple, short, and meaningful commit messages.

---

## Workflow

### Step 1 — Check Staged Changes

Before committing, verify that there are staged changes:

```bash
git diff --cached --name-status
```

If the output is empty:
- **Stop** and inform the user: *"There are no staged changes. Please stage files first using `git add` and then request a commit."*
- Do not make a commit.

---

### Step 2 — Verify Atomic Scope

Review the list of changed files and the diff:

```bash
git diff --cached
```

- **Atomic Check**: Verify if the changes represent a cohesive, logical unit.
- If the changes cover multiple unrelated features, modules, or mix code refactoring with configuration/tool updates in a single chunk, split them. Unstage the unrelated files using `git restore --staged <paths>`, commit the first cohesive unit, and then stage and commit the remaining parts as subsequent atomic commits.
- If the changes are a cohesive unit, proceed to generating the commit message.

---

### Step 3 — Propose Commit Plan

Before running the commit, generate a commit plan and present it to the user in the chat:
- **Proposed Commit Message**: A simple, short, and meaningful commit message.
- **Files Staged**: A list of the files being committed.
- **Summary**: A brief (1-2 sentence) summary of what changes are being introduced.

Wait for the user's explicit confirmation/approval of the plan. Do not commit until the user approves.

---

### Step 4 — Run Commit Script

After the user approves the commit plan, run the helper script, passing the approved commit message as the argument:

```bash
bash .agents/skills/git-commit/scripts/git_commit.sh "your commit message"
```

---

### Step 5 — Confirm and Summarize

Report the result to the user:
- Show the commit hash and message.
- List the files that were committed.
