# kubectl Cheatsheet for Practice 006a

## Cluster & Context

```bash
minikube start --driver=docker          # Start local cluster
minikube stop                           # Stop cluster (preserves state)
minikube delete                         # Delete cluster entirely
kubectl cluster-info                    # Show cluster endpoint
kubectl get nodes                       # List cluster nodes
```

## Building Images Inside minikube

```powershell
# CMD — point Docker CLI to minikube's daemon
@FOR /f "tokens=*" %i IN ('minikube -p minikube docker-env --shell cmd') DO @%i

# PowerShell
& minikube -p minikube docker-env --shell powershell | Invoke-Expression

# Linux/macOS
eval $(minikube docker-env)

# Then build normally — image is inside minikube
docker build -t task-tracker:v1 ./app
```

## Apply & Delete Manifests

```bash
kubectl apply -f k8s/pod.yaml                  # Create/update from file
kubectl apply -f k8s/ -n practice              # Apply all YAMLs in directory
kubectl delete -f k8s/pod.yaml                 # Delete resources defined in file
kubectl delete namespace practice              # Delete namespace + everything inside
```

## Pods

```bash
kubectl get pods                               # List pods (default namespace)
kubectl get pods -n practice                   # List pods in "practice" namespace
kubectl get pods --all-namespaces              # List pods across all namespaces
kubectl get pods -o wide                       # Show node, IP, etc.
kubectl get pods --show-labels                 # Show labels column

kubectl describe pod <pod-name> -n practice    # Detailed info (events, status)
kubectl logs <pod-name> -n practice            # View container stdout/stderr
kubectl logs <pod-name> -n practice -f         # Follow logs (like tail -f)

kubectl exec -it <pod-name> -n practice -- sh  # Shell into a container
```

## Deployments & ReplicaSets

```bash
kubectl get deployments -n practice            # List deployments
kubectl get rs -n practice                     # List ReplicaSets
kubectl describe deployment <name> -n practice # Detailed deployment info

kubectl scale deployment <name> --replicas=5 -n practice   # Scale up/down

kubectl set image deployment/<name> <container>=<image:tag> -n practice  # Update image
kubectl rollout status deployment/<name> -n practice       # Watch rollout
kubectl rollout history deployment/<name> -n practice      # View revision history
kubectl rollout undo deployment/<name> -n practice         # Rollback one revision
```

## ConfigMaps & Secrets

```bash
kubectl get configmaps -n practice             # List ConfigMaps
kubectl get secrets -n practice                # List Secrets
kubectl describe configmap <name> -n practice  # View ConfigMap contents
kubectl describe secret <name> -n practice     # View Secret metadata

# Decode a secret value (PowerShell)
kubectl get secret <name> -n practice -o jsonpath="{.data.SECRET_API_KEY}" | ForEach-Object { [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String($_)) }

# Decode a secret value (bash)
kubectl get secret <name> -n practice -o jsonpath="{.data.SECRET_API_KEY}" | base64 -d
```

## Namespaces

```bash
kubectl get namespaces                         # List all namespaces
kubectl create namespace practice              # Create imperatively
kubectl delete namespace practice              # Delete + all resources inside
```

## Labels & Selectors

```bash
kubectl get pods -l app=task-tracker -n practice          # Filter by label
kubectl get pods -l 'app=task-tracker,env=debug'          # Multiple labels (AND)
kubectl get pods -l 'app in (task-tracker,api)'           # Set-based selector
kubectl label pod <name> env=debug -n practice            # Add label
kubectl label pod <name> env- -n practice                 # Remove label
```

## Port Forwarding

```bash
kubectl port-forward pod/<pod-name> 5000:5000 -n practice
kubectl port-forward deployment/<deploy-name> 5000:5000 -n practice
# Then: curl http://localhost:5000/health
```

## Base64 Encoding (for Secrets)

```powershell
# PowerShell
[Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes("super-secret-key-12345"))
[Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes("db-password-67890"))
```

```bash
# Bash
echo -n "super-secret-key-12345" | base64
echo -n "db-password-67890" | base64
```

## Debugging

```bash
kubectl get events -n practice                 # Cluster events (scheduling, errors)
kubectl get events --sort-by='.lastTimestamp'   # Sorted by time
kubectl describe pod <name> -n practice        # Check "Events" section at the bottom
kubectl logs <name> -n practice --previous     # Logs from previous crash
```
