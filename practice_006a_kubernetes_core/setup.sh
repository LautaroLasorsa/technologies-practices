#!/usr/bin/env bash
# ============================================================
#  Practice 006a: Kubernetes Core Concepts — Linux/macOS Setup
# ============================================================
#
#  Prerequisites:
#    - Docker installed and running
#    - Internet connection (for downloading minikube/kubectl)
#
#  This script guides you through installing minikube and kubectl.
#  Run each section manually — do NOT run this file blindly.
# ============================================================

set -euo pipefail

echo ""
echo "=== Step 1: Install minikube ==="
echo ""
echo "macOS (Homebrew):   brew install minikube"
echo "Linux (binary):     curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64"
echo "                    sudo install minikube-linux-amd64 /usr/local/bin/minikube"
echo ""
echo "Verify: minikube version"
echo ""
read -p "Press Enter to continue..."

echo ""
echo "=== Step 2: Install kubectl ==="
echo ""
echo "macOS (Homebrew):   brew install kubectl"
echo "Linux (snap):       sudo snap install kubectl --classic"
echo ""
echo "Verify: kubectl version --client"
echo ""
read -p "Press Enter to continue..."

echo ""
echo "=== Step 3: Start minikube cluster ==="
echo ""
echo "  minikube start --driver=docker"
echo ""
echo "Verify:"
echo "  kubectl cluster-info"
echo "  kubectl get nodes"
echo ""
read -p "Press Enter to continue..."

echo ""
echo "=== Step 4: Point Docker CLI to minikube's daemon ==="
echo ""
echo "  eval \$(minikube docker-env)"
echo ""
echo "This makes 'docker build' target minikube's internal Docker."
echo ""
read -p "Press Enter to continue..."

echo ""
echo "=== Step 5: Build the app image ==="
echo ""
echo "  docker build -t task-tracker:v1 ./app"
echo ""
echo "Verify:"
echo "  docker images | grep task-tracker"
echo ""
read -p "Press Enter to continue..."

echo ""
echo "=== Ready! ==="
echo "Now open the k8s/ folder and start filling in the YAML manifests."
echo ""
