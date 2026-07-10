---
name: git-create-mr
description: >-
  Create a Merge Request (MR) on GitLab or Pull Request (PR) on GitHub for the current branch.
  Use when the user says "create an MR", "create a PR", "open a pull request", or "submit changes".
  Automatically determines the remote hosting platform, base branch, parses repository metadata,
  authenticates using a token, and opens the PR/MR.
---

# git-create-mr

## Overview

This skill creates a Pull Request (GitHub) or Merge Request (GitLab) for the current active branch. It automatically detects the hosting service, parses SSH config aliases, verifies credentials, pulls repository metadata (like default branch), and calls the API to create the PR/MR.

---

## Workflow

### Step 1 — Check Credentials

Before running, verify that `GITHUB_TOKEN` (for GitHub) or `GITLAB_TOKEN` (for GitLab) is configured in `~/.env` using the safe credentials protocol:

```bash
grep -sq "^GITHUB_TOKEN=" ~/.env
```

If it is missing, stop and run the following command to ask the user to add it securely (replacing GITHUB_TOKEN or GITLAB_TOKEN as appropriate):

```bash
printf "Enter GITHUB_TOKEN (typing hidden): " && read -s val && echo && echo "GITHUB_TOKEN=$val" >> "$HOME/.env" && echo "Saved."
```

---

### Step 2 — Run creation script (Dry Run / Plan)

Execute the Python script in dry-run mode to inspect the proposed PR/MR details:

```bash
python3 .agents/skills/git-create-mr/scripts/create_mr.py --dry-run
```

The script will output details like:
- Target Repository (Owner/Repo)
- Platform detected (GitHub/GitLab)
- Source Branch (e.g. current branch)
- Target/Base Branch (e.g. `main` or the default branch)
- Proposed PR/MR Title (auto-generated from latest commit)
- Proposed Description (auto-generated from commit log)

Verify these details and present them to the user. Ask the user for confirmation.

---

### Step 3 — Create PR/MR

After the user approves, run the script to create the PR/MR:

```bash
python3 .agents/skills/git-create-mr/scripts/create_mr.py
```

Optional flags:
- `--title "<Title>"`: Override the auto-generated title.
- `--description "<Description>"`: Override the auto-generated description.
- `--base "<Branch>"`: Override the base branch (default: remote's default branch).
- `--draft`: Create the pull request as a draft (GitHub only).
- `--push`: Push the current branch to origin before creating the PR/MR (if not already pushed).

---

### Step 4 — Confirm and Summarize

Provide the user with the direct link to the created PR/MR on GitHub/GitLab.
