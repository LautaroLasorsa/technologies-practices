#!/usr/bin/env bash
# push_to_gitea.sh -- Initialize a Git repo with k8s manifests and push to Gitea.
#
# This script:
#   1. Creates a temporary directory
#   2. Copies the k8s/ manifests into it
#   3. Initializes a Git repo, commits, and pushes to Gitea
#
# Prerequisites:
#   - Gitea is running and port-forwarded to localhost:3000
#   - A user "gitops-admin" with password "gitops-admin" exists in Gitea
#   - A repository "demo-app" has been created in Gitea
#
# Usage: bash scripts/push_to_gitea.sh

set -euo pipefail

GITEA_URL="http://localhost:3000"
GITEA_USER="gitops-admin"
GITEA_PASS="gitops-admin"
REPO_NAME="demo-app"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PRACTICE_DIR="$(dirname "$SCRIPT_DIR")"

echo "=== Pushing k8s manifests to Gitea ==="

# Create a temporary directory for the git repo
TMPDIR=$(mktemp -d)
trap "rm -rf $TMPDIR" EXIT

# Copy k8s manifests
cp -r "$PRACTICE_DIR/k8s" "$TMPDIR/k8s"

# Initialize git repo and push
cd "$TMPDIR"
git init
git checkout -b main
git add .
git commit -m "Initial commit: k8s manifests for demo-app"

# Add remote and push
git remote add origin "http://${GITEA_USER}:${GITEA_PASS}@localhost:3000/${GITEA_USER}/${REPO_NAME}.git"
git push -u origin main

echo ""
echo "=== Done! Manifests pushed to Gitea ==="
echo "  Repo URL (host):     ${GITEA_URL}/${GITEA_USER}/${REPO_NAME}"
echo "  Repo URL (cluster):  http://gitea.gitea.svc.cluster.local:3000/${GITEA_USER}/${REPO_NAME}.git"
echo ""
echo "You can now create an ArgoCD Application pointing to this repo."
