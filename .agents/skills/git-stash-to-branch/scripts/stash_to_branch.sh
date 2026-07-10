#!/bin/bash
# Unified script for git-stash-to-branch skill.
# Moves staged changes from the current branch to a new feature branch.

set -e

if [ -z "$1" ]; then
  echo "Error: No branch name provided." >&2
  exit 1
fi

BRANCH_NAME="$1"
SOURCE_BRANCH=$(git rev-parse --abbrev-ref HEAD)

# Verify there are staged changes
if git diff --cached --quiet; then
  echo "Error: There are no staged changes. Stage some files first." >&2
  exit 1
fi

echo "Moving staged changes from '$SOURCE_BRANCH' to '$BRANCH_NAME'..."

# Strategy A: Try direct checkout
if git checkout -b "$BRANCH_NAME" 2>/dev/null; then
  echo "Success (Strategy A): Checked out to new branch '$BRANCH_NAME' with staged changes."
else
  echo "Strategy A failed (branch may exist or conflict). Trying Strategy B (stash fallback)..."
  
  # Check Git version
  GIT_VER=$(git --version | awk '{print $3}')
  MAJOR=$(echo "$GIT_VER" | cut -d. -f1)
  MINOR=$(echo "$GIT_VER" | cut -d. -f2)
  
  USE_STAGED=false
  if [ "$MAJOR" -gt 2 ]; then
    USE_STAGED=true
  elif [ "$MAJOR" -eq 2 ] && [ "$MINOR" -ge 35 ]; then
    USE_STAGED=true
  fi
  
  if [ "$USE_STAGED" = true ]; then
    echo "Stashing staged changes (Git >= 2.35)..."
    git stash --staged --message "wip: stash-to-branch $BRANCH_NAME"
  else
    echo "Stashing all changes (Git < 2.35)..."
    git stash --include-untracked --message "wip: stash-to-branch $BRANCH_NAME"
  fi
  
  # Checkout/create the branch
  if ! git checkout -b "$BRANCH_NAME" 2>/dev/null; then
    echo "Branch '$BRANCH_NAME' already exists. Switching to it..."
    if ! git checkout "$BRANCH_NAME"; then
      echo "Error: Failed to checkout branch '$BRANCH_NAME'." >&2
      git stash pop || true
      exit 1
    fi
  fi
  
  # Apply stash
  if ! git stash pop; then
    echo "Error: Failed to pop stash onto '$BRANCH_NAME'." >&2
    exit 1
  fi
  echo "Success (Strategy B): Moved staged changes to '$BRANCH_NAME' via stash."
fi
