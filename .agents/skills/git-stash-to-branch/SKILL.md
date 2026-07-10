---
name: git-stash-to-branch
description: >-
  Move staged git changes from the current branch onto a new AI-named feature
  branch, then hard-reset the source branch to HEAD. Use when the user says
  things like "save my staged changes to a branch", "move this to a feature
  branch", "I staged changes on the wrong branch", "stash this to a new
  branch", or "create a branch for my staged work". Reads the staged diff,
  generates a descriptive kebab-case branch name starting with 'feature/',
  creates the branch directly, and hard-resets the source branch without prompting.
---

# git-stash-to-branch

## Overview

This skill automates the workflow of isolating staged changes onto a dedicated,
meaningfully-named branch without committing. It is scoped to **staged changes
only** (the git index). It does not touch unstaged edits or untracked files as
the basis for naming — but those changes will follow the working tree to the new
branch during checkout.

**High-level flow:**

1. Read the staged diff to understand what changed
2. Generate an AI-selected branch name following the `feature/` convention
3. Create new branch + move staged changes there
4. Hard-reset the source branch to `HEAD`

---

## Workflow

### Step 1 — Verify and read staged changes

Before running the migration script, check if there are staged changes and read the diff so you can generate a descriptive branch name:

```bash
git diff --cached
```

If the output is empty, **stop** and inform the user: _"There are no staged changes. Stage some files first with `git add` and then re-trigger this skill."_

---

### Step 2 — Generate the branch name

Based on the diff content, generate a single branch name following this naming convention:

```
feature/<short-kebab-description>
```

Rules for the description part:
- Lowercase only
- Hyphens instead of spaces or underscores
- Maximum 5 words / ~40 characters total
- Descriptive of the *change*, not the file name (e.g. `feature/remove-legacy-vae-files`)

Select this branch name (referred to as `BRANCH_NAME`) autonomously. Do NOT ask the user for confirmation or input.

---

### Step 3 — Execute the unified shell script

Run the unified shell script, passing the generated `BRANCH_NAME` as an argument. This script handles checking out/creating the branch, stashing/popping fallback if necessary, and leaving the working directory/index in the correct state:

```bash
bash .agents/skills/git-stash-to-branch/scripts/stash_to_branch.sh <BRANCH_NAME>
```

---

### Step 4 — Confirm and summarise

Run the following to verify the outcome:

```bash
git status --short
git branch --list <BRANCH_NAME>
```

Report back to the user:
- New branch `<BRANCH_NAME>` created/updated with the staged changes.
- The user is now on the `<BRANCH_NAME>` branch.
- Any remaining untracked or unstaged files.
- Next suggested command: `git commit -m "..."`

---

## Common Mistakes

1. **Triggering on unstaged changes**: This skill reads `git diff --cached`
   (staged/index only). If the user forgot to `git add` their files, Step 1
   will catch it and prompt them to stage first.

2. **Branch name already exists**: The unified script handles this by attempting
   to switch to the existing branch and popping the stash if the direct checkout
   fails. If the script still fails, ask the user to resolve the conflict or delete the old branch.

3. **Untracked files**: Keep in mind that untracked files are not stashed or
   moved unless git < 2.35 is used (where the script falls back to stashing everything).
   Communicate any remaining files/status clearly to the user.
