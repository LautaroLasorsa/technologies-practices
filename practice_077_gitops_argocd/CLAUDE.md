# Practice 077: GitOps with ArgoCD — Declarative Continuous Delivery

## Technologies

- **ArgoCD** -- Declarative, GitOps-based continuous delivery tool for Kubernetes
- **kind** (Kubernetes in Docker) -- Lightweight local Kubernetes cluster for development
- **Gitea** -- Lightweight self-hosted Git server (runs inside the kind cluster)
- **kubectl** -- CLI tool to interact with the Kubernetes API server
- **Docker** -- Container runtime used by kind

## Stack

- Python 3.12+ (FastAPI -- minimal HTTP app for deployment targets)
- Docker (container images, kind cluster)
- kind + kubectl (local Kubernetes cluster)
- ArgoCD (GitOps controller)
- Gitea (in-cluster Git server)

## Theoretical Context

### What is GitOps?

GitOps is an operational framework that applies DevOps best practices (version control, collaboration, CI/CD) to infrastructure automation. It was coined by Weaveworks in 2017 and is built on four core principles:

| Principle | Description |
|-----------|-------------|
| **Declarative** | The entire system's desired state is described declaratively (e.g., Kubernetes YAML manifests), not imperatively |
| **Versioned & Immutable** | The desired state is stored in Git, providing a complete audit trail of every change, who made it, and when |
| **Pulled Automatically** | Approved changes to the desired state are automatically applied to the system by software agents (not pushed by CI pipelines) |
| **Continuously Reconciled** | Software agents continuously observe the actual system state and attempt to match it to the desired state, self-healing drift |

The key insight of GitOps is that **Git becomes the single source of truth** for both application code AND infrastructure state. Instead of running `kubectl apply` from a CI pipeline (push-based), a GitOps agent running *inside* the cluster watches a Git repo and *pulls* changes into the cluster. This inverts the deployment model: the cluster pulls its own desired state rather than having external systems push to it.

**Why this matters:** Push-based pipelines require the CI system to have cluster credentials (security risk). Pull-based GitOps keeps credentials inside the cluster -- the CI system only needs to push to Git. Additionally, any manual `kubectl` changes (drift) are automatically reverted by the agent, ensuring the cluster always matches Git.

### ArgoCD Architecture

ArgoCD is the most widely adopted GitOps tool for Kubernetes. It runs as a set of controllers inside the Kubernetes cluster and continuously monitors Git repositories for changes to Kubernetes manifests.

**Core Components:**

```
                    +-------------------+
                    |   ArgoCD Web UI   |  (React frontend)
                    +--------+----------+
                             |
                    +--------v----------+
                    |   API Server      |  (gRPC + REST)
                    |  - Authentication |  (SSO, RBAC, JWT)
                    |  - Authorization  |  (project-level RBAC)
                    |  - Webhook recv   |  (Git webhooks for fast sync)
                    +--------+----------+
                             |
              +--------------+--------------+
              |                             |
    +---------v---------+      +-----------v-----------+
    | Application       |      |    Repo Server        |
    | Controller        |      |  - Git clone/fetch    |
    | - Watch loop      |      |  - Manifest gen       |
    | - Diff engine     |      |    (Helm, Kustomize,  |
    | - Sync executor   |      |     plain YAML)       |
    | - Health assess   |      |  - Caching            |
    +-------------------+      +-----------------------+
              |
              v
    +-------------------+
    |   Kubernetes API  |  (target cluster)
    +-------------------+
```

| Component | Role |
|-----------|------|
| **API Server** | Frontend gateway -- serves the Web UI and CLI, handles authentication (SSO, local accounts), authorization (RBAC per project), and receives Git webhooks to trigger immediate syncs instead of waiting for the polling interval |
| **Application Controller** | The brain -- a Kubernetes controller that continuously watches Application CRDs, compares the desired state (from Git, via Repo Server) against the live state (from Kubernetes API), computes diffs, and executes sync operations. Runs the reconciliation loop (default: every 3 minutes) |
| **Repo Server** | Stateless service that clones Git repositories, generates manifests (supports plain YAML, Helm charts, Kustomize overlays, Jsonnet), and returns the rendered manifests to the Application Controller. Caches repos to avoid redundant clones |
| **Redis** | In-memory cache used by the API Server and Application Controller for caching app state and reducing load on the Kubernetes API |
| **Dex** | Optional OpenID Connect provider for SSO integration (LDAP, SAML, GitHub, Google) |

### The Reconciliation Loop

ArgoCD's core behavior is a continuous reconciliation loop:

1. **Application Controller** polls Repo Server for the latest manifests from Git (or receives a webhook notification)
2. **Repo Server** clones/fetches the Git repo, renders manifests (Helm template, Kustomize build, or plain YAML), returns them
3. **Application Controller** fetches the live state of those same resources from the Kubernetes API
4. **Diff engine** compares desired (Git) vs. live (cluster) state
5. If the states differ, the Application is marked **OutOfSync**
6. Depending on sync policy:
   - **Manual sync**: User clicks "Sync" in the UI or runs `argocd app sync`
   - **Auto-sync**: Controller automatically applies the Git manifests to the cluster
7. After sync, **health assessment** checks if resources are actually healthy (Pods running, Deployments rolled out, etc.)
8. Loop repeats every ~3 minutes (configurable via `timeout.reconciliation`)

### The Application CRD

The `Application` is ArgoCD's primary Custom Resource Definition (CRD). It declares:

```yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: my-app
  namespace: argocd
spec:
  project: default               # ArgoCD project (RBAC boundary)
  source:
    repoURL: https://git.example.com/org/repo.git
    targetRevision: HEAD          # Branch, tag, or commit SHA
    path: k8s/                    # Directory within the repo containing manifests
  destination:
    server: https://kubernetes.default.svc  # Target cluster API
    namespace: my-app-ns          # Target namespace
  syncPolicy:
    automated:                    # Enable auto-sync (omit for manual)
      prune: true                 # Delete resources removed from Git
      selfHeal: true              # Revert manual kubectl changes
      allowEmpty: false           # Prevent sync if Git yields zero resources
    syncOptions:
      - CreateNamespace=true      # Create target namespace if missing
```

**Key fields:**

| Field | Purpose |
|-------|---------|
| `spec.source.repoURL` | Git repository URL containing the manifests |
| `spec.source.path` | Directory within the repo (ArgoCD only watches this path) |
| `spec.source.targetRevision` | Git ref to track -- `HEAD` (default branch), a branch name, tag, or commit SHA |
| `spec.destination.server` | Kubernetes API URL of the target cluster (use `https://kubernetes.default.svc` for in-cluster) |
| `spec.destination.namespace` | Target namespace for the deployed resources |
| `spec.syncPolicy.automated.prune` | If `true`, ArgoCD deletes Kubernetes resources that no longer exist in Git. Default `false` (safety: resources removed from Git survive in the cluster) |
| `spec.syncPolicy.automated.selfHeal` | If `true`, ArgoCD reverts manual changes made via `kubectl edit` or `kubectl apply` back to the Git state within ~5 seconds. Default `false` |
| `spec.project` | ArgoCD Project -- an RBAC and policy boundary (which repos, clusters, and namespaces an Application can access) |

### Sync Statuses & Health

ArgoCD tracks two orthogonal dimensions for every Application:

**Sync Status** (does Git match the cluster?):
- **Synced** -- live state matches the desired state in Git
- **OutOfSync** -- live state differs from Git (drift detected)
- **Unknown** -- unable to determine (e.g., Repo Server unreachable)

**Health Status** (are resources actually working?):
- **Healthy** -- all resources are running and ready (Deployment has desired replicas, Pods are Running, Services have endpoints)
- **Progressing** -- resources are being created/updated (e.g., Deployment rollout in progress)
- **Degraded** -- resources exist but are unhealthy (e.g., Pod in CrashLoopBackOff, failing readiness probes)
- **Suspended** -- resources are paused (e.g., suspended CronJob)
- **Missing** -- resources defined in Git don't exist in the cluster yet

ArgoCD has built-in health checks for standard Kubernetes resources (Deployments, StatefulSets, DaemonSets, Services, Ingresses, PVCs). For CRDs, you can define custom health checks using Lua scripts in the `argocd-cm` ConfigMap.

### ApplicationSet

`ApplicationSet` is a CRD that generates multiple `Application` objects from templates + generators. Useful for:
- Deploying the same app to multiple clusters
- Creating Applications for every directory in a monorepo
- Generating Applications from a list of environments

Generators include: List, Cluster, Git Directory, Git File, Matrix, Merge, Pull Request.

### Rollback via Git

In GitOps, rollback is simply a `git revert` of the offending commit. ArgoCD detects the new commit (which restores the previous manifest state) and syncs the cluster back. This provides:
- **Audit trail**: every deploy and rollback is a Git commit with author, timestamp, and message
- **Reproducibility**: any historical state can be recreated by checking out that commit
- **No special rollback mechanism**: the same sync process handles both deploys and rollbacks

ArgoCD also supports rollback via its UI/CLI (`argocd app rollback`) which uses its internal history, but the GitOps-native approach is `git revert`.

### ArgoCD vs Flux

| Dimension | ArgoCD | Flux |
|-----------|--------|------|
| **UI** | Built-in Web UI with real-time sync visualization | No UI (CLI + optional Weave GitOps dashboard) |
| **Architecture** | Centralized server (API + Controller + Repo Server) | Decentralized controllers (source-controller, kustomize-controller, helm-controller) |
| **CRD Model** | Single `Application` CRD | Multiple CRDs: `GitRepository`, `Kustomization`, `HelmRelease`, etc. |
| **Multi-tenancy** | Projects with RBAC per repo/cluster/namespace | Kubernetes RBAC + namespace isolation |
| **Manifest rendering** | Helm, Kustomize, Jsonnet, plain YAML, custom plugins | Helm, Kustomize (via dedicated controllers) |
| **Drift detection** | Visual diff in UI | Event-based, logs only |
| **Community (2025)** | Larger community, more stars, CNCF graduated | CNCF graduated, but Weaveworks (original maintainer) shut down in 2024 |
| **Best for** | Teams wanting visual observability, multi-cluster RBAC | Platform engineers wanting lightweight, Kubernetes-native infra automation |

For new projects, ArgoCD is the safer bet due to its larger community, active development, and the uncertainty around Flux's future after Weaveworks' closure.

## Description

Set up a complete **local GitOps pipeline** using ArgoCD, kind, and Gitea:

```
Developer --> git push --> Gitea (in-cluster Git) --> ArgoCD detects change
                                                          |
                                                          v
                                                    ArgoCD syncs K8s
                                                          |
                                                          v
                                                    App updated in cluster
```

A minimal FastAPI application and its Dockerfile are fully implemented. **You write the Kubernetes manifests, ArgoCD Application definitions, and sync policies** -- the parts that teach GitOps concepts.

### What you'll learn

1. **kind cluster setup** -- Creating a local Kubernetes cluster with port mappings
2. **ArgoCD installation** -- Deploying ArgoCD into the cluster and accessing its UI
3. **Gitea setup** -- In-cluster Git server to simulate a real GitOps workflow
4. **Application CRD** -- Declaring an ArgoCD Application pointing to a Git repo
5. **Manual sync** -- Triggering a sync and observing resource creation
6. **Auto-sync** -- Enabling automated sync with self-heal and prune
7. **Drift detection** -- Making manual `kubectl` changes and watching ArgoCD revert them
8. **GitOps delivery** -- Pushing manifest changes to Git and observing reconciliation
9. **Failed deploy & health** -- Introducing a broken image and observing Degraded health status
10. **Rollback via Git** -- Using `git revert` to roll back a failed deploy

## Instructions

### Phase 1: Cluster & Infrastructure Setup (~15 min)

1. Install prerequisites: `kind`, `kubectl`, `argocd` CLI
2. Create a kind cluster with port mappings: `kind create cluster --config scripts/kind-config.yaml --name gitops-lab`
3. Verify: `kubectl cluster-info --context kind-gitops-lab`
4. Install ArgoCD into the cluster (see Commands table for the exact `kubectl apply` command)
5. Wait for ArgoCD pods to be ready: `kubectl wait --for=condition=available deployment -l app.kubernetes.io/part-of=argocd -n argocd --timeout=120s`
6. Retrieve the ArgoCD admin password
7. Port-forward the ArgoCD UI: `kubectl port-forward svc/argocd-server -n argocd 8080:443`
8. Open `https://localhost:8080` in your browser, login with `admin` and the retrieved password
9. Key question: Why does ArgoCD run *inside* the cluster rather than as an external CI/CD system?

### Phase 2: Gitea (In-Cluster Git Server) (~15 min)

1. Deploy Gitea into the cluster: `kubectl apply -f gitea/`
2. Wait for Gitea to be ready: `kubectl wait --for=condition=available deployment/gitea -n gitea --timeout=120s`
3. Port-forward Gitea: `kubectl port-forward svc/gitea -n gitea 3000:3000`
4. Open `http://localhost:3000`, create an admin account (user: `gitops-admin`, password: `gitops-admin`)
5. Create a repository named `demo-app` in Gitea
6. Push the app manifests from `k8s/` to the Gitea repo using the `scripts/push_to_gitea.sh` script
7. Key question: In production, you would use GitHub/GitLab. Why do we use Gitea here?

### Phase 3: Build & Load Application Image (~5 min)

1. Build the app image: `docker build -t demo-app:v1 ./app`
2. Load it into the kind cluster: `kind load docker-image demo-app:v1 --name gitops-lab`
3. Verify: `docker exec gitops-lab-control-plane crictl images | grep demo-app`
4. Key question: Why do we need `kind load` instead of just `docker build`? (Hint: kind runs its own containerd)

### Phase 4: ArgoCD Application -- Manual Sync (~20 min)

1. Open `argocd/application.yaml` -- fill in the `TODO(human)` sections to define an ArgoCD Application
2. Register the Gitea repo with ArgoCD: use the `argocd repo add` command (see Commands table)
3. Apply the Application: `kubectl apply -f argocd/application.yaml`
4. Observe in the ArgoCD UI: the Application should appear as **OutOfSync**
5. Trigger a manual sync: `argocd app sync demo-app` (or click "Sync" in the UI)
6. Verify: `kubectl get pods -n demo-app` -- the app pods should be running
7. Port-forward the app: `kubectl port-forward svc/demo-app -n demo-app 5000:80`
8. Test: `curl http://localhost:5000/health`
9. Key question: After sync, what does the ArgoCD UI show for sync status and health status?

### Phase 5: Auto-Sync, Self-Heal & Prune (~20 min)

1. Open `argocd/application-autosync.yaml` -- fill in the `TODO(human)` sections to enable auto-sync with self-heal and prune
2. Apply: `kubectl apply -f argocd/application-autosync.yaml`
3. **Test self-heal**: manually scale the deployment via kubectl and watch ArgoCD revert it:
   ```
   kubectl scale deployment demo-app --replicas=5 -n demo-app
   # Wait ~5 seconds, then check:
   kubectl get pods -n demo-app
   ```
4. **Test prune**: push a manifest change to Gitea that removes a resource (e.g., the ConfigMap). Observe ArgoCD delete it from the cluster
5. **Test GitOps delivery**: modify the ConfigMap in Git (change `APP_VERSION`), push to Gitea, and observe ArgoCD update the cluster
6. Key question: What is the difference between self-heal and auto-sync? When would you disable self-heal?

### Phase 6: Failed Deploy & Rollback (~15 min)

1. Update `k8s/deployment.yaml` in Gitea to reference a non-existent image (e.g., `demo-app:v999`)
2. Push the change to Gitea
3. Observe ArgoCD sync the bad manifest -- the Application health should become **Degraded** (Pods in `ImagePullBackOff`)
4. Check the ArgoCD UI for the health status and error details
5. Rollback by reverting the bad commit in Git:
   ```
   git revert HEAD
   git push origin main
   ```
6. Observe ArgoCD detect the revert and sync back to the healthy state
7. Key question: Why is `git revert` preferred over `argocd app rollback` in a GitOps workflow?

### Phase 7: ApplicationSet (~15 min)

1. Open `argocd/applicationset.yaml` -- fill in the `TODO(human)` sections to generate Applications for multiple environments
2. Apply: `kubectl apply -f argocd/applicationset.yaml`
3. Observe: ArgoCD should create two Applications (staging, production) from the same template
4. Verify: `argocd app list`
5. Key question: What generator types does ApplicationSet support? When would you use Git Directory vs List generator?

### Phase 8: Cleanup (~5 min)

1. Delete ArgoCD Applications: `argocd app delete demo-app --yes`
2. Delete the kind cluster: `kind delete cluster --name gitops-lab`
3. Discussion: How does GitOps change the role of CI/CD? What does CI do in a GitOps world vs. what CD does?

## Motivation

- **Industry standard for K8s delivery**: ArgoCD is the de facto GitOps tool -- appears in most DevOps/SRE/Platform Engineer job descriptions
- **Complements 006a/006b**: Moves from "how to deploy to K8s" to "how to manage K8s deployments at scale declaratively"
- **Security model**: Pull-based deployment eliminates the need for CI pipelines to hold cluster credentials
- **Audit & compliance**: Every deployment is a Git commit with author, timestamp, and review trail
- **Self-healing**: Drift detection and automatic remediation reduce operational toil
- **Foundation for platform engineering**: ArgoCD + ApplicationSets is the basis for internal developer platforms

## References

- [ArgoCD Official Docs](https://argo-cd.readthedocs.io/en/stable/)
- [ArgoCD Architecture Overview](https://argo-cd.readthedocs.io/en/stable/operator-manual/architecture/)
- [ArgoCD Getting Started](https://argo-cd.readthedocs.io/en/stable/getting_started/)
- [ArgoCD Application CRD Spec](https://argo-cd.readthedocs.io/en/stable/user-guide/application-specification/)
- [ArgoCD Automated Sync Policy](https://argo-cd.readthedocs.io/en/stable/user-guide/auto_sync/)
- [ArgoCD Sync Options](https://argo-cd.readthedocs.io/en/latest/user-guide/sync-options/)
- [ArgoCD Resource Health](https://argo-cd.readthedocs.io/en/latest/operator-manual/health/)
- [ArgoCD ApplicationSet Generators](https://argo-cd.readthedocs.io/en/stable/operator-manual/applicationset/)
- [GitOps Principles -- OpenGitOps](https://opengitops.dev/)
- [kind Quick Start](https://kind.sigs.k8s.io/docs/user/quick-start/)
- [kind Configuration](https://kind.sigs.k8s.io/docs/user/configuration/)
- [Gitea Documentation](https://docs.gitea.com/)
- [ArgoCD vs Flux Comparison (2025)](https://www.zignuts.com/blog/argo-cd-vs-flux-cd--comparison)
- [Codefresh -- ArgoCD Architecture Overview](https://codefresh.io/learn/argo-cd/a-comprehensive-overview-of-argo-cd-architectures-2025/)
- [ArgoCD Sync Policies Practical Guide](https://codefresh.io/learn/argo-cd/argocd-sync-policies-a-practical-guide/)

## Commands

All commands are run from the `practice_077_gitops_argocd/` folder root.

### Phase 1: Cluster & ArgoCD Setup

| Command | Description |
|---------|-------------|
| `kind create cluster --config scripts/kind-config.yaml --name gitops-lab` | Create a kind cluster with port mappings for ArgoCD and app access |
| `kubectl cluster-info --context kind-gitops-lab` | Verify kubectl is connected to the kind cluster |
| `kubectl create namespace argocd` | Create the argocd namespace |
| `kubectl apply -n argocd -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml` | Install ArgoCD (all components) into the argocd namespace |
| `kubectl wait --for=condition=available deployment -l app.kubernetes.io/part-of=argocd -n argocd --timeout=120s` | Wait for all ArgoCD deployments to be ready |
| `kubectl -n argocd get secret argocd-initial-admin-secret -o jsonpath="{.data.password}" \| base64 -d` | Retrieve the ArgoCD admin password (Linux/macOS) |
| `kubectl -n argocd get secret argocd-initial-admin-secret -o jsonpath='{.data.password}'` | Retrieve the ArgoCD admin password (Windows -- decode the base64 manually) |
| `kubectl port-forward svc/argocd-server -n argocd 8080:443` | Port-forward ArgoCD UI to https://localhost:8080 |
| `argocd login localhost:8080 --username admin --password <password> --insecure` | Login to ArgoCD via CLI (--insecure skips TLS verification for self-signed cert) |

### Phase 2: Gitea Setup

| Command | Description |
|---------|-------------|
| `kubectl apply -f gitea/` | Deploy Gitea namespace, deployment, and service into the cluster |
| `kubectl wait --for=condition=available deployment/gitea -n gitea --timeout=120s` | Wait for Gitea to be ready |
| `kubectl port-forward svc/gitea -n gitea 3000:3000` | Port-forward Gitea web UI to http://localhost:3000 |
| `bash scripts/push_to_gitea.sh` | Initialize a Git repo with k8s/ manifests and push to Gitea |

### Phase 3: Build & Load Image

| Command | Description |
|---------|-------------|
| `docker build -t demo-app:v1 ./app` | Build the FastAPI application Docker image |
| `kind load docker-image demo-app:v1 --name gitops-lab` | Load the image into the kind cluster's containerd |
| `docker exec gitops-lab-control-plane crictl images \| grep demo-app` | Verify the image is available inside the kind node (Linux/macOS) |

### Phase 4: ArgoCD Application -- Manual Sync

| Command | Description |
|---------|-------------|
| `argocd repo add http://gitea.gitea.svc.cluster.local:3000/gitops-admin/demo-app.git --username gitops-admin --password gitops-admin --insecure-skip-server-verification` | Register the Gitea repo with ArgoCD (in-cluster DNS) |
| `kubectl apply -f argocd/application.yaml` | Create the ArgoCD Application CRD |
| `argocd app get demo-app` | View Application sync status and health |
| `argocd app sync demo-app` | Trigger a manual sync |
| `kubectl get pods -n demo-app` | Verify app pods are running after sync |
| `kubectl port-forward svc/demo-app -n demo-app 5000:80` | Port-forward the app to http://localhost:5000 |
| `curl http://localhost:5000/health` | Test the app health endpoint |

### Phase 5: Auto-Sync & Self-Heal

| Command | Description |
|---------|-------------|
| `kubectl apply -f argocd/application-autosync.yaml` | Apply the Application with auto-sync, self-heal, and prune |
| `kubectl scale deployment demo-app --replicas=5 -n demo-app` | Manually scale (ArgoCD should revert this within ~5s) |
| `kubectl get pods -n demo-app -w` | Watch pods to see ArgoCD self-heal (revert to desired replica count) |
| `argocd app get demo-app` | Check sync status after self-heal |

### Phase 6: Failed Deploy & Rollback

| Command | Description |
|---------|-------------|
| `argocd app get demo-app` | Check Application health (should show Degraded after bad image push) |
| `argocd app history demo-app` | View Application sync history (revisions) |
| `kubectl get pods -n demo-app` | See pods in ImagePullBackOff/ErrImagePull |
| `kubectl describe pod -l app=demo-app -n demo-app` | Inspect pod events for image pull errors |

### Phase 7: ApplicationSet

| Command | Description |
|---------|-------------|
| `kubectl apply -f argocd/applicationset.yaml` | Create the ApplicationSet (generates multiple Applications) |
| `argocd app list` | List all Applications (should show staging + production) |
| `kubectl get applications -n argocd` | List Application CRDs via kubectl |

### Phase 8: Cleanup

| Command | Description |
|---------|-------------|
| `argocd app delete demo-app --yes` | Delete the ArgoCD Application and its managed resources |
| `argocd app delete demo-app-staging --yes` | Delete staging Application from ApplicationSet |
| `argocd app delete demo-app-production --yes` | Delete production Application from ApplicationSet |
| `kubectl delete -f argocd/applicationset.yaml` | Delete the ApplicationSet |
| `kind delete cluster --name gitops-lab` | Delete the kind cluster entirely |

### Debugging

| Command | Description |
|---------|-------------|
| `argocd app diff demo-app` | Show diff between Git desired state and live cluster state |
| `argocd app logs demo-app` | View application logs via ArgoCD |
| `kubectl logs -l app.kubernetes.io/name=argocd-application-controller -n argocd` | View ArgoCD Application Controller logs |
| `kubectl logs -l app.kubernetes.io/name=argocd-repo-server -n argocd` | View ArgoCD Repo Server logs |
| `kubectl get events -n demo-app --sort-by='.lastTimestamp'` | View events in the app namespace |
| `argocd app resources demo-app` | List all resources managed by the Application |
| `argocd app manifests demo-app` | Show the rendered manifests ArgoCD is applying |

## State

`not-started`
