@echo off
REM ============================================================
REM  Practice 006a: Kubernetes Core Concepts — Windows Setup
REM ============================================================
REM
REM  Prerequisites:
REM    - Docker Desktop installed and running (with WSL2 backend)
REM    - Internet connection (for downloading minikube/kubectl)
REM
REM  This script guides you through installing minikube and kubectl.
REM  Run each section manually — do NOT run this file blindly.
REM ============================================================

echo.
echo === Step 1: Install minikube ===
echo.
echo Option A — winget (recommended):
echo   winget install Kubernetes.minikube
echo.
echo Option B — Chocolatey:
echo   choco install minikube
echo.
echo Option C — Manual download:
echo   https://minikube.sigs.k8s.io/docs/start/
echo.
echo After install, verify:
echo   minikube version
echo.
pause

echo.
echo === Step 2: Install kubectl ===
echo.
echo Option A — winget:
echo   winget install Kubernetes.kubectl
echo.
echo Option B — Chocolatey:
echo   choco install kubernetes-cli
echo.
echo After install, verify:
echo   kubectl version --client
echo.
pause

echo.
echo === Step 3: Start minikube cluster ===
echo.
echo   minikube start --driver=docker
echo.
echo This creates a single-node K8s cluster inside a Docker container.
echo It may take 1-3 minutes on first run (downloads the K8s image).
echo.
echo Verify:
echo   kubectl cluster-info
echo   kubectl get nodes
echo.
pause

echo.
echo === Step 4: Point Docker CLI to minikube's daemon ===
echo.
echo In CMD (current terminal):
echo   @FOR /f "tokens=*" %%i IN ('minikube -p minikube docker-env --shell cmd') DO @%%i
echo.
echo In PowerShell:
echo   ^& minikube -p minikube docker-env --shell powershell ^| Invoke-Expression
echo.
echo This makes 'docker build' target minikube's internal Docker, so images
echo are immediately available to the cluster without pushing to a registry.
echo.
pause

echo.
echo === Step 5: Build the app image ===
echo.
echo   docker build -t task-tracker:v1 ./app
echo.
echo Verify the image exists in minikube's Docker:
echo   docker images ^| findstr task-tracker
echo.
pause

echo.
echo === Ready! ===
echo.
echo Now open the k8s/ folder and start filling in the YAML manifests.
echo Follow the phases in CLAUDE.md.
echo.
echo Quick reference:
echo   kubectl apply -f k8s/pod.yaml          Apply a manifest
echo   kubectl get pods                        List pods
echo   kubectl describe pod task-tracker       Inspect a pod
echo   kubectl logs task-tracker               View pod logs
echo   kubectl port-forward pod/task-tracker 5000:5000   Expose locally
echo   kubectl delete -f k8s/pod.yaml          Delete resources from manifest
echo.
