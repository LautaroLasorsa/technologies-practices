#!/usr/bin/env bash
# update_and_push.sh -- Clone the Gitea repo, apply a change, and push.
#
# This helper script simplifies the "modify manifest in Git" workflow.
# It clones the repo, lets you make changes, commits, and pushes.
#
# Usage:
#   bash scripts/update_and_push.sh "commit message"
#
# The script clones to a temp dir, opens it, and lets you copy modified files in.
# For quick changes, you can also directly modify the command below.

set -euo pipefail

GITEA_USER="gitops-admin"
GITEA_PASS="gitops-admin"
REPO_URL="http://${GITEA_USER}:${GITEA_PASS}@localhost:3000/${GITEA_USER}/demo-app.git"
COMMIT_MSG="${1:-Update manifests}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PRACTICE_DIR="$(dirname "$SCRIPT_DIR")"

TMPDIR=$(mktemp -d)
trap "rm -rf $TMPDIR" EXIT

echo "=== Cloning repo from Gitea ==="
git clone "$REPO_URL" "$TMPDIR/demo-app"

echo "=== Copying updated k8s manifests ==="
cp -r "$PRACTICE_DIR/k8s/"* "$TMPDIR/demo-app/k8s/"

cd "$TMPDIR/demo-app"
git add .

if git diff --cached --quiet; then
    echo "No changes to commit."
    exit 0
fi

git commit -m "$COMMIT_MSG"
git push origin main

echo ""
echo "=== Pushed: $COMMIT_MSG ==="
echo "ArgoCD will detect this change and sync (if auto-sync is enabled)."
echo "Check status: argocd app get demo-app"
