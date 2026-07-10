#!/bin/bash
# Script for git-push skill.
# Pushes current branch to remote (origin), setting upstream if needed.

set -e

BRANCH_NAME=$(git rev-parse --abbrev-ref HEAD)
REMOTE="origin"

# Check if current branch has an upstream tracking branch
if ! git rev-parse --abbrev-ref @{u} >/dev/null 2>&1; then
  echo "No upstream tracking branch found. Pushing and setting upstream to '$REMOTE'..."
  git push --set-upstream "$REMOTE" "$BRANCH_NAME"
else
  echo "Upstream branch exists. Pushing to '$REMOTE'..."
  git push "$REMOTE" "$BRANCH_NAME"
fi
