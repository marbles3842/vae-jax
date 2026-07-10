---
name: git-push
description: >-
  Push the current branch to origin. Use when the user says "push", "git push",
  "push branch", "push changes", or as the second step of "commit and push".
  Determines the current branch, pushes to remote origin, and sets upstream tracking if needed.
---

# git-push

## Overview

This skill pushes the current local Git branch to the remote repository (`origin`). If the branch doesn't have an upstream tracking branch configured, the skill automatically configures it.

---

## Workflow

### Step 1 — Check Branch Status

Optionally inspect the current branch name and status:

```bash
git status --short
git branch --show-current
```

- Warn the user if there are uncommitted changes, but proceed to push anyway (since pushing only affects committed changes).

---

### Step 2 — Run Push Script

Execute the helper script to push the current branch to `origin`:

```bash
bash .agents/skills/git-push/scripts/git_push.sh
```

---

### Step 3 — Verify Output

Report the output of the script to the user. Show if the upstream tracking was successfully configured or if the push succeeded.
