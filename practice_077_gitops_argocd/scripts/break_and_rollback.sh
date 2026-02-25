#!/usr/bin/env bash
# break_and_rollback.sh -- Push a broken image tag, then rollback via git revert.
#
# This script demonstrates the GitOps rollback workflow:
#   1. Clone the Gitea repo
#   2. Change the Deployment image to a non-existent tag (demo-app:v999)
#   3. Push the bad change (ArgoCD will sync it, causing Degraded health)
#   4. Wait for the user to observe the failure
#   5. Revert the bad commit and push (ArgoCD syncs back to healthy)
#
# Usage: bash scripts/break_and_rollback.sh

set -euo pipefail

GITEA_USER="gitops-admin"
GITEA_PASS="gitops-admin"
REPO_URL="http://${GITEA_USER}:${GITEA_PASS}@localhost:3000/${GITEA_USER}/demo-app.git"

TMPDIR=$(mktemp -d)
trap "rm -rf $TMPDIR" EXIT

echo "=== Step 1: Cloning repo ==="
git clone "$REPO_URL" "$TMPDIR/demo-app"
cd "$TMPDIR/demo-app"

echo ""
echo "=== Step 2: Breaking the deployment (image: demo-app:v999) ==="
# Replace the image tag with a non-existent one
if [[ "$OSTYPE" == "darwin"* ]]; then
    sed -i '' 's|image: demo-app:v1|image: demo-app:v999|g' k8s/deployment.yaml
else
    sed -i 's|image: demo-app:v1|image: demo-app:v999|g' k8s/deployment.yaml
fi

git add .
git commit -m "Deploy v999 (broken image -- intentional)"
git push origin main

echo ""
echo "=== Bad commit pushed! ==="
echo "ArgoCD will sync this and the app will become Degraded."
echo ""
echo "Observe the failure:"
echo "  - ArgoCD UI: https://localhost:8080"
echo "  - CLI: argocd app get demo-app"
echo "  - Pods: kubectl get pods -n demo-app"
echo ""
read -p "Press Enter to rollback via git revert..."

echo ""
echo "=== Step 3: Rolling back via git revert ==="
git revert --no-edit HEAD
git push origin main

echo ""
echo "=== Rollback pushed! ==="
echo "ArgoCD will detect the revert and sync back to the healthy state."
echo "Check: argocd app get demo-app"
