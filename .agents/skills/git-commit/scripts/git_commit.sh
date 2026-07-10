#!/bin/bash
# Script for git-commit skill.
# Commits currently staged changes with a provided message.

set -e

if [ -z "$1" ]; then
  echo "Error: No commit message provided." >&2
  exit 1
fi

COMMIT_MSG="$1"

# Verify there are staged changes
if git diff --cached --quiet; then
  echo "Error: There are no staged changes. Stage files first." >&2
  exit 1
fi

git commit -m "$COMMIT_MSG"
echo "Successfully committed staged changes."
