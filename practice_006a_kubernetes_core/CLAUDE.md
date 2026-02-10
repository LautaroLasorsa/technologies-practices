# Practice 006a: Kubernetes — Core Concepts & Deploy

## Technologies

- **Kubernetes** — Container orchestration platform for automating deployment, scaling, and management
- **minikube** — Local single-node Kubernetes cluster for development and learning
- **kubectl** — CLI tool to interact with the Kubernetes API server
- **Docker** — Container runtime used by minikube (driver: `docker`)

## Stack

- Python 3.12+ (Flask — minimal HTTP app)
- Docker (container image)
- minikube + kubectl (local Kubernetes cluster)

## Description

Deploy a **Task Tracker API** (a simple Flask app) to a local Kubernetes cluster, learning core K8s primitives hands-on: Pods, Deployments, ReplicaSets, Namespaces, ConfigMaps, Secrets, labels, and selectors.

The Python app and its Dockerfile are fully implemented. **You write the Kubernetes YAML manifests** — the part that teaches how K8s objects are declared and how they relate to each other.

### What you'll learn

1. **Pods** — Smallest deployable unit; wraps one or more containers
2. **Deployments & ReplicaSets** — Declarative desired-state management with rollout/rollback
3. **Namespaces** — Logical isolation of resources within a cluster
4. **Labels & Selectors** — How K8s objects find and group each other
5. **ConfigMaps** — Inject non-sensitive configuration into containers
6. **Secrets** — Inject sensitive data (base64-encoded, not encrypted at rest by default)
7. **kubectl** — Core commands: apply, get, describe, logs, delete, port-forward
8. **Image workflow** — Building images inside minikube's Docker daemon

## Instructions

### Phase 1: Setup & Cluster (~10 min)

1. Install minikube and kubectl (see `setup.bat`)
2. Start a local cluster: `minikube start --driver=docker`
3. Point your shell to minikube's Docker daemon so images built locally are available to the cluster
4. Build the app image: `docker build -t task-tracker:v1 ./app`
5. Verify: `docker images | findstr task-tracker`
6. Key question: Why do we build inside minikube's Docker daemon instead of our host's?

### Phase 2: Bare Pod (~15 min)

1. Open `k8s/pod.yaml` — fill in the `TODO(human)` sections
2. Apply: `kubectl apply -f k8s/pod.yaml`
3. Verify: `kubectl get pods`, `kubectl describe pod task-tracker`
4. Port-forward: `kubectl port-forward pod/task-tracker 5000:5000`
5. Test: `curl http://localhost:5000/health`
6. Delete the pod: `kubectl delete -f k8s/pod.yaml`
7. Key question: If the pod crashes, does anything restart it? Why or why not?

### Phase 3: Namespace (~10 min)

1. Open `k8s/namespace.yaml` — fill in the `TODO(human)` section
2. Apply: `kubectl apply -f k8s/namespace.yaml`
3. Verify: `kubectl get namespaces`
4. Re-deploy the pod into the new namespace (add `namespace` to metadata)
5. Verify: `kubectl get pods -n practice`
6. Key question: What happens if two teams deploy a pod named `task-tracker` in different namespaces?

### Phase 4: ConfigMap & Secret (~20 min)

1. Open `k8s/configmap.yaml` — fill in the `TODO(human)` sections
2. Open `k8s/secret.yaml` — fill in the `TODO(human)` sections (base64-encode values)
3. Apply both: `kubectl apply -f k8s/configmap.yaml -f k8s/secret.yaml -n practice`
4. Verify: `kubectl get configmaps -n practice`, `kubectl get secrets -n practice`
5. Key question: Secrets are base64-encoded but not encrypted. What additional measures would you use in production?

### Phase 5: Deployment (~25 min)

1. Open `k8s/deployment.yaml` — fill in the `TODO(human)` sections
2. Apply: `kubectl apply -f k8s/deployment.yaml -n practice`
3. Observe the ReplicaSet: `kubectl get rs -n practice`
4. Scale: `kubectl scale deployment task-tracker --replicas=5 -n practice`
5. Delete a pod and watch self-healing: `kubectl delete pod <pod-name> -n practice`
6. Port-forward to the deployment: `kubectl port-forward deployment/task-tracker 5000:5000 -n practice`
7. Test all endpoints: `/health`, `GET /tasks`, `POST /tasks`
8. Key question: When you scaled to 5 replicas, what happened to the ReplicaSet? What is the relationship between Deployment -> ReplicaSet -> Pods?

### Phase 6: Rolling Update & Rollback (~15 min)

1. Modify the app (e.g., change the version in `/health` response)
2. Rebuild: `docker build -t task-tracker:v2 ./app`
3. Update the deployment image: `kubectl set image deployment/task-tracker task-tracker=task-tracker:v2 -n practice`
4. Watch rollout: `kubectl rollout status deployment/task-tracker -n practice`
5. Rollback: `kubectl rollout undo deployment/task-tracker -n practice`
6. View history: `kubectl rollout history deployment/task-tracker -n practice`
7. Key question: How does K8s ensure zero-downtime during a rolling update?

### Phase 7: Labels & Selectors Exploration (~15 min)

1. List pods with labels: `kubectl get pods -n practice --show-labels`
2. Filter by label: `kubectl get pods -n practice -l app=task-tracker`
3. Add a custom label: `kubectl label pod <pod-name> env=debug -n practice`
4. Filter by new label: `kubectl get pods -n practice -l env=debug`
5. Key question: How does a Deployment's `selector.matchLabels` connect to the Pod template's `labels`? What breaks if they don't match?

### Phase 8: Cleanup & Discussion (~10 min)

1. Delete everything: `kubectl delete namespace practice`
2. Stop minikube: `minikube stop`
3. Discussion: How does declarative (YAML + `kubectl apply`) differ from imperative (`kubectl create`, `kubectl run`)? Which is better for production and why?
4. Discussion: What is the role of `etcd` in the Kubernetes architecture?

## Motivation

- **Industry standard**: Kubernetes is the de facto container orchestration platform — required knowledge for any backend/infra role
- **Declarative infrastructure**: Understanding K8s manifests is foundational for GitOps, Terraform, and CI/CD pipelines
- **Complements Docker practice (005)**: Moves from "run one container" to "orchestrate many containers with self-healing"
- **Foundation for 006b**: Services, Ingress, and auto-scaling build directly on these core concepts
- **Career relevance**: K8s appears in most DevOps/SRE/Backend job descriptions; hands-on experience with kubectl is expected

## References

- [Kubernetes Official Concepts](https://kubernetes.io/docs/concepts/)
- [Pods — Kubernetes Docs](https://kubernetes.io/docs/concepts/workloads/pods/)
- [Deployments — Kubernetes Docs](https://kubernetes.io/docs/concepts/workloads/controllers/deployment/)
- [Namespaces — Kubernetes Docs](https://kubernetes.io/docs/concepts/overview/working-with-objects/namespaces/)
- [Labels and Selectors — Kubernetes Docs](https://kubernetes.io/docs/concepts/overview/working-with-objects/labels/)
- [ConfigMaps — Kubernetes Docs](https://kubernetes.io/docs/concepts/configuration/configmap/)
- [Secrets — Kubernetes Docs](https://kubernetes.io/docs/concepts/configuration/secret/)
- [minikube Start Guide](https://minikube.sigs.k8s.io/docs/start/)
- [kubectl Cheat Sheet](https://kubernetes.io/docs/reference/kubectl/cheatsheet/)
- [Pushing Images to minikube](https://minikube.sigs.k8s.io/docs/handbook/pushing/)
- [Using Local Docker Images With Minikube](https://www.baeldung.com/ops/docker-local-images-minikube)

## Commands

### Phase 1: Setup & Cluster

| Command | Description |
|---------|-------------|
| `minikube start --driver=docker` | Start a local single-node Kubernetes cluster using Docker |
| `minikube version` | Verify minikube installation |
| `kubectl version --client` | Verify kubectl installation |
| `kubectl cluster-info` | Show cluster API server endpoint |
| `kubectl get nodes` | List cluster nodes (should show one minikube node) |
| `@FOR /f "tokens=*" %i IN ('minikube -p minikube docker-env --shell cmd') DO @%i` | Point Docker CLI to minikube's daemon (CMD) |
| `& minikube -p minikube docker-env --shell powershell \| Invoke-Expression` | Point Docker CLI to minikube's daemon (PowerShell) |
| `eval $(minikube docker-env)` | Point Docker CLI to minikube's daemon (Linux/macOS) |
| `docker build -t task-tracker:v1 ./app` | Build the Flask app image inside minikube's Docker |
| `docker images \| findstr task-tracker` | Verify the image exists in minikube's Docker (Windows) |
| `docker images \| grep task-tracker` | Verify the image exists in minikube's Docker (Linux/macOS) |

### Phase 2: Bare Pod

| Command | Description |
|---------|-------------|
| `kubectl apply -f k8s/pod.yaml` | Create the bare pod from manifest |
| `kubectl get pods` | List pods in default namespace |
| `kubectl describe pod task-tracker` | Inspect pod details (status, events, containers) |
| `kubectl port-forward pod/task-tracker 5000:5000` | Forward local port 5000 to pod port 5000 |
| `curl http://localhost:5000/health` | Test the health endpoint via port-forward |
| `kubectl logs task-tracker` | View pod stdout/stderr logs |
| `kubectl delete -f k8s/pod.yaml` | Delete the bare pod |

### Phase 3: Namespace

| Command | Description |
|---------|-------------|
| `kubectl apply -f k8s/namespace.yaml` | Create the "practice" namespace |
| `kubectl get namespaces` | List all namespaces |
| `kubectl get pods -n practice` | List pods in the "practice" namespace |

### Phase 4: ConfigMap & Secret

| Command | Description |
|---------|-------------|
| `kubectl apply -f k8s/configmap.yaml -f k8s/secret.yaml -n practice` | Apply ConfigMap and Secret in the practice namespace |
| `kubectl get configmaps -n practice` | List ConfigMaps in practice namespace |
| `kubectl get secrets -n practice` | List Secrets in practice namespace |
| `kubectl describe configmap task-tracker-config -n practice` | View ConfigMap contents |
| `kubectl describe secret task-tracker-secrets -n practice` | View Secret metadata |
| `[Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes("my-value"))` | Base64-encode a value for Secrets (PowerShell) |
| `echo -n "my-value" \| base64` | Base64-encode a value for Secrets (Linux/macOS) |

### Phase 5: Deployment

| Command | Description |
|---------|-------------|
| `kubectl apply -f k8s/deployment.yaml -n practice` | Create the Deployment (3 replicas) |
| `kubectl get deployments -n practice` | List Deployments |
| `kubectl get rs -n practice` | List ReplicaSets managed by the Deployment |
| `kubectl get pods -n practice` | List pods created by the ReplicaSet |
| `kubectl scale deployment task-tracker --replicas=5 -n practice` | Scale the Deployment to 5 replicas |
| `kubectl delete pod <pod-name> -n practice` | Delete a pod to observe self-healing |
| `kubectl port-forward deployment/task-tracker 5000:5000 -n practice` | Port-forward to any pod in the Deployment |
| `kubectl describe deployment task-tracker -n practice` | Inspect Deployment details |

### Phase 6: Rolling Update & Rollback

| Command | Description |
|---------|-------------|
| `docker build -t task-tracker:v2 ./app` | Build v2 image after modifying the app |
| `kubectl set image deployment/task-tracker task-tracker=task-tracker:v2 -n practice` | Update Deployment to use v2 image |
| `kubectl rollout status deployment/task-tracker -n practice` | Watch the rolling update progress |
| `kubectl rollout undo deployment/task-tracker -n practice` | Rollback to previous revision |
| `kubectl rollout history deployment/task-tracker -n practice` | View rollout revision history |

### Phase 7: Labels & Selectors

| Command | Description |
|---------|-------------|
| `kubectl get pods -n practice --show-labels` | List pods with their labels |
| `kubectl get pods -n practice -l app=task-tracker` | Filter pods by label selector |
| `kubectl label pod <pod-name> env=debug -n practice` | Add a custom label to a pod |
| `kubectl get pods -n practice -l env=debug` | Filter pods by the custom label |
| `kubectl label pod <pod-name> env- -n practice` | Remove a label from a pod |
| `kubectl get pods -l 'app=task-tracker,env=debug' -n practice` | Filter by multiple labels (AND) |

### Phase 8: Cleanup

| Command | Description |
|---------|-------------|
| `kubectl delete namespace practice` | Delete namespace and all resources inside it |
| `minikube stop` | Stop the minikube cluster (preserves state) |
| `minikube delete` | Delete the minikube cluster entirely |

### Debugging

| Command | Description |
|---------|-------------|
| `kubectl get events -n practice` | View cluster events (scheduling, errors) |
| `kubectl get events --sort-by='.lastTimestamp' -n practice` | View events sorted by time |
| `kubectl logs <pod-name> -n practice` | View pod logs |
| `kubectl logs <pod-name> -n practice -f` | Follow pod logs (live stream) |
| `kubectl logs <pod-name> -n practice --previous` | View logs from previous crash |
| `kubectl exec -it <pod-name> -n practice -- sh` | Shell into a running container |
| `kubectl get pods -o wide -n practice` | Show pods with node/IP details |
| `kubectl apply -f k8s/ -n practice` | Apply all manifests in the k8s/ directory |
| `kubectl delete -f k8s/ -n practice` | Delete all resources defined in k8s/ directory |

## State

`not-started`
