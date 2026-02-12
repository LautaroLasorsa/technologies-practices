# Practice 006b: Kubernetes Services, Scaling & Ingress

## Technologies

- **Kubernetes Services** -- ClusterIP, NodePort, and LoadBalancer service types for pod networking
- **Ingress (NGINX)** -- Layer 7 HTTP routing with path-based and host-based rules
- **HPA (Horizontal Pod Autoscaler)** -- Automatic pod scaling based on CPU/memory metrics
- **Resource Requests & Limits** -- CPU/memory guarantees and ceilings per container
- **Liveness & Readiness Probes** -- Health checks that control restart and traffic routing
- **Rolling Updates** -- Zero-downtime deployments with maxSurge/maxUnavailable

## Stack

- Python 3.12+ (FastAPI + httpx)
- Docker (multi-stage builds)
- minikube with NGINX Ingress addon and metrics-server addon

## Theoretical Context

Kubernetes Services provide stable networking endpoints for ephemeral Pods, solving the problem of dynamic IP addresses that change every time a Pod restarts. Without Services, clients would need to track individual Pod IPs manually -- Services abstract away this complexity by assigning a virtual IP (ClusterIP) or exposing ports on nodes (NodePort) that route traffic to healthy Pods via label selectors. Ingress extends Services by providing Layer 7 (HTTP/HTTPS) routing rules, enabling path-based and host-based traffic management without allocating separate IP addresses per service.

Internally, **ClusterIP** (the default Service type) allocates a virtual IP from the cluster's service CIDR range and programs iptables/IPVS rules on every node to load-balance traffic across Pods matching the Service's selector. CoreDNS creates a DNS record (`<service>.<namespace>.svc.cluster.local`) pointing to this ClusterIP. When a Pod queries this DNS name, it gets the ClusterIP, and the kernel's iptables NATs the connection to a backend Pod's IP. **NodePort** builds on ClusterIP by additionally opening a static port (30000-32767) on every node -- traffic to `<node-IP>:<node-port>` gets forwarded to the ClusterIP, then to a Pod. **LoadBalancer** (cloud-only) provisions an external load balancer (AWS NLB, GCP Load Balancer) that forwards to the NodePort. **Ingress** operates differently: an Ingress Controller (e.g., NGINX, Traefik) runs as Pods inside the cluster, watches Ingress resources, and dynamically configures its internal routing rules -- it's essentially an L7 proxy managed by Kubernetes manifests.

**Horizontal Pod Autoscaler (HPA)** automatically adjusts replica counts based on observed metrics (CPU, memory, custom metrics). The HPA controller queries metrics-server every 15 seconds, computes `desiredReplicas = ceil(currentReplicas * (currentMetric / targetMetric))`, and patches the Deployment's replica field if the calculated value differs. Crucially, HPA requires **resource requests** to be defined on Pods -- CPU utilization is calculated as `(actual CPU usage) / (requested CPU)`, so without requests, the denominator is undefined. HPA enforces scaling rate limits (scale-up is fast, scale-down waits ~5 minutes by default to avoid flapping) and respects min/max replica bounds. Liveness and readiness probes interact with scaling: only Pods passing readiness checks are added to Service endpoints, so during scale-up, new Pods don't receive traffic until ready; during rolling updates, old Pods drain gracefully before termination.

| Concept | Description |
|---------|-------------|
| **ClusterIP** | Default Service type. Allocates a virtual IP accessible only inside the cluster. Used for inter-service communication. |
| **NodePort** | Exposes a Service on a static port (30000-32767) on every node. External clients hit `<node-IP>:<node-port>`. |
| **LoadBalancer** | Cloud-only. Provisions an external load balancer (AWS ALB/NLB, GCP LB) that forwards to NodePort. |
| **Ingress** | Layer 7 (HTTP) routing rules. Managed by an Ingress Controller (NGINX, Traefik). Routes traffic based on paths/hosts. |
| **Liveness Probe** | Restarts the container if it fails. Use for deadlock detection. Failure = restart, not traffic removal. |
| **Readiness Probe** | Removes Pod from Service endpoints if it fails. Use for temporary unavailability (e.g., warming up cache). |
| **Resource Requests** | Guaranteed minimum resources. Scheduler uses these to place Pods. HPA calculates utilization based on requests. |
| **Resource Limits** | Hard caps. Kubelet throttles CPU or OOM-kills containers exceeding memory limits. |
| **HPA** | Automatically scales Pods based on metrics. Requires metrics-server. Computes replicas = ceil(current * (metric/target)). |
| **Rolling Update** | Gradually replaces old Pods with new ones. `maxSurge` allows extra Pods; `maxUnavailable` sets minimum ready Pods. |

**ClusterIP** is for internal microservice communication; **NodePort** is for dev/testing external access (insecure, exposes random high ports); **LoadBalancer** is for production external access (cloud-native, assigns a real IP); **Ingress** is for HTTP/HTTPS routing with path-based rules (most production-friendly for web apps). Alternatives include **service meshes** (Istio, Linkerd) which provide advanced L7 routing, retries, circuit breaking, and mTLS but add operational complexity. For simple single-service exposure, NodePort suffices; for multi-service web apps, Ingress is standard; for non-HTTP protocols (gRPC, databases), LoadBalancer or headless Services (StatefulSets) are used. HPA is essential for unpredictable traffic patterns but requires careful tuning of scale-up/down policies to avoid cost spikes or latency during sudden load changes.

## Description

Build and deploy a **two-service microservice system** (frontend gateway + backend API) into minikube, then progressively layer Kubernetes networking, health checks, autoscaling, and ingress on top. The Python apps are fully implemented -- the learning happens entirely in the K8s YAML manifests.

### Architecture

```
          [Ingress NGINX]
            /          \
     /frontend      /backend
         |              |
   [Frontend SVC]  [Backend SVC]
    (ClusterIP)     (ClusterIP)
         |              |
   [Frontend Pods] [Backend Pods]
         |              |
         +--> httpx --->+
```

- **Backend**: FastAPI app with `/health`, `/ready`, `/api/items`, and `/api/cpu-burn` (for HPA testing)
- **Frontend**: FastAPI app that proxies requests to the backend via its ClusterIP service name

### What you'll learn

1. **Service types** -- ClusterIP (internal), NodePort (external via node port), and when LoadBalancer applies
2. **Service discovery** -- How pods resolve service names via CoreDNS (e.g., `http://backend-svc:8000`)
3. **Liveness vs readiness probes** -- What each controls (restart vs traffic) and how misconfiguration causes outages
4. **Resource requests & limits** -- How the scheduler places pods and what happens when limits are exceeded
5. **Rolling updates** -- maxSurge, maxUnavailable, and how probes interact with deployments
6. **HPA** -- Metrics-server, scaling policies, and observing scale-up/scale-down in real time
7. **Ingress** -- Path-based routing, NGINX Ingress controller, and how Ingress relates to Services

## Instructions

### Phase 1: Setup & Build Images (~10 min)

1. Start minikube and enable addons:
   ```bash
   minikube start --memory=4096 --cpus=2
   minikube addons enable ingress
   minikube addons enable metrics-server
   ```
2. Point your shell to minikube's Docker daemon:
   ```bash
   eval $(minikube docker-env)        # Linux/Mac
   minikube docker-env --shell powershell | Invoke-Expression  # PowerShell
   ```
3. Build both images:
   ```bash
   docker build -t backend:v1 apps/backend/
   docker build -t frontend:v1 apps/frontend/
   ```
4. Verify images: `docker images | grep -E "backend|frontend"`

### Phase 2: Deployments with Probes & Resources (~20 min)

1. Open `k8s/backend-deployment.yaml` -- fill in the `TODO(human)` sections:
   - Container resource requests (128Mi memory, 100m CPU) and limits (256Mi memory, 250m CPU)
   - Liveness probe hitting `/health` (period 10s, failure threshold 3)
   - Readiness probe hitting `/ready` (period 5s, initial delay 5s)
   - Rolling update strategy: maxSurge 1, maxUnavailable 0 (zero-downtime)
2. Do the same for `k8s/frontend-deployment.yaml`
3. Apply both: `kubectl apply -f k8s/backend-deployment.yaml -f k8s/frontend-deployment.yaml`
4. Verify: `kubectl get pods -w` -- watch pods reach `Running` + `1/1 READY`
5. Key question: What happens if the readiness probe fails but the liveness probe passes?

### Phase 3: Services (~15 min)

1. Open `k8s/backend-service.yaml` -- fill in the `TODO(human)` sections:
   - ClusterIP service selecting backend pods on port 8000
2. Open `k8s/frontend-service.yaml`:
   - NodePort service selecting frontend pods, exposing port 8000 on a node port
3. Apply: `kubectl apply -f k8s/backend-service.yaml -f k8s/frontend-service.yaml`
4. Test internal connectivity:
   ```bash
   kubectl exec deploy/frontend -- python -c "import httpx; print(httpx.get('http://backend-svc:8000/api/items').json())"
   ```
5. Test external access via NodePort:
   ```bash
   minikube service frontend-svc --url
   curl <URL>/proxy/items
   ```
6. Key question: Why can the frontend reach `backend-svc` by name? What DNS record does Kubernetes create?

### Phase 4: Ingress (~15 min)

1. Open `k8s/ingress.yaml` -- fill in the `TODO(human)` sections:
   - Path `/api` routes to `backend-svc:8000`
   - Path `/` routes to `frontend-svc:8000`
   - Use `pathType: Prefix` and `ingressClassName: nginx`
2. Apply: `kubectl apply -f k8s/ingress.yaml`
3. Get the ingress IP:
   ```bash
   minikube ip   # or: kubectl get ingress
   ```
4. Test routing:
   ```bash
   curl http://$(minikube ip)/api/items
   curl http://$(minikube ip)/
   ```
   (On Windows, you may need to add the IP to your hosts file or use `minikube tunnel`)
5. Key question: How does the Ingress controller differ from a NodePort? What layer does each operate at?

### Phase 5: Horizontal Pod Autoscaler (~20 min)

1. Open `k8s/backend-hpa.yaml` -- fill in the `TODO(human)` sections:
   - Target the backend deployment
   - Scale between 2 and 8 replicas
   - Target 50% average CPU utilization
2. Apply: `kubectl apply -f k8s/backend-hpa.yaml`
3. Verify baseline: `kubectl get hpa -w` (wait for metrics to appear, ~60s)
4. Generate load to trigger scale-up:
   ```bash
   kubectl run load-gen --image=busybox --restart=Never -- /bin/sh -c "while true; do wget -q -O- http://backend-svc:8000/api/cpu-burn; done"
   ```
5. Watch scaling: `kubectl get hpa -w` and `kubectl get pods -w`
6. Stop load: `kubectl delete pod load-gen`
7. Watch scale-down (~5 min cooldown)
8. Key question: Why does HPA require resource requests to be set? What happens without them?

### Phase 6: Rolling Update in Action (~10 min)

1. Rebuild backend with a visible change:
   ```bash
   docker build -t backend:v2 apps/backend/
   ```
2. Update the deployment image:
   ```bash
   kubectl set image deploy/backend backend=backend:v2
   ```
3. Watch the rollout: `kubectl rollout status deploy/backend`
4. Observe: old pods drain after new pods pass readiness probes
5. Rollback: `kubectl rollout undo deploy/backend`
6. Key question: With maxSurge=1 and maxUnavailable=0, what is the minimum number of ready pods during the rollout?

### Phase 7: Discussion (~10 min)

1. When would you use ClusterIP vs NodePort vs LoadBalancer vs Ingress?
2. How do liveness and readiness probes interact with rolling updates?
3. What are the risks of setting CPU limits too low? Too high?
4. How would you handle HTTPS termination with Ingress in production?

## Motivation

- **Production Kubernetes fluency**: Services, Ingress, and HPA are the bread and butter of any K8s deployment -- understanding them is non-negotiable for backend/infrastructure roles
- **Networking mental model**: ClusterIP/NodePort/Ingress map directly to real-world load balancing decisions in cloud environments (ALB, NLB, Cloud Load Balancer)
- **Autoscaling literacy**: HPA is the standard mechanism for elastic workloads -- understanding metrics, cooldowns, and resource requests prevents costly over/under-provisioning
- **Reliability patterns**: Probes and rolling updates are how K8s achieves zero-downtime deploys -- misconfiguring them is one of the most common production issues

## References

- [Kubernetes Services](https://kubernetes.io/docs/concepts/services-networking/service/)
- [Service Types: ClusterIP vs NodePort vs LoadBalancer](https://www.baeldung.com/ops/kubernetes-service-types)
- [Ingress on Minikube with NGINX](https://kubernetes.io/docs/tasks/access-application-cluster/ingress-minikube/)
- [Configure Liveness, Readiness and Startup Probes](https://kubernetes.io/docs/tasks/configure-pod-container/configure-liveness-readiness-startup-probes/)
- [HPA Walkthrough](https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale-walkthrough/)
- [Resource Management for Pods and Containers](https://kubernetes.io/docs/concepts/configuration/manage-resources-containers/)
- [Rolling Update Strategy](https://kubernetes.io/docs/concepts/workloads/controllers/deployment/#rolling-update-deployment)
- [Ingress Controllers](https://kubernetes.io/docs/concepts/services-networking/ingress-controllers/)

## Commands

### Phase 1: Setup & Build Images

| Command | Description |
|---------|-------------|
| `minikube start --memory=4096 --cpus=2` | Start minikube with 4 GB RAM and 2 CPUs |
| `minikube addons enable ingress` | Enable the NGINX Ingress Controller addon |
| `minikube addons enable metrics-server` | Enable the metrics-server addon (required for HPA) |
| `@FOR /f "tokens=*" %i IN ('minikube -p minikube docker-env --shell cmd') DO @%i` | Point Docker CLI to minikube's daemon (CMD) |
| `& minikube -p minikube docker-env --shell powershell \| Invoke-Expression` | Point Docker CLI to minikube's daemon (PowerShell) |
| `eval $(minikube docker-env)` | Point Docker CLI to minikube's daemon (Linux/macOS) |
| `docker build -t backend:v1 apps/backend/` | Build the backend FastAPI image (v1) |
| `docker build -t frontend:v1 apps/frontend/` | Build the frontend FastAPI image (v1) |
| `docker images \| findstr "backend frontend"` | Verify both images exist (Windows) |
| `docker images \| grep -E "backend\|frontend"` | Verify both images exist (Linux/macOS) |

### Phase 2: Deployments with Probes & Resources

| Command | Description |
|---------|-------------|
| `kubectl apply -f k8s/backend-deployment.yaml -f k8s/frontend-deployment.yaml` | Apply both Deployment manifests |
| `kubectl get pods -w` | Watch pods reach Running + 1/1 READY |
| `kubectl get deployments` | List Deployments |
| `kubectl describe deployment backend` | Inspect backend Deployment details (strategy, probes) |
| `kubectl describe deployment frontend` | Inspect frontend Deployment details |

### Phase 3: Services

| Command | Description |
|---------|-------------|
| `kubectl apply -f k8s/backend-service.yaml -f k8s/frontend-service.yaml` | Apply both Service manifests |
| `kubectl get svc` | List all Services (shows ClusterIP, NodePort, ports) |
| `kubectl describe svc backend-svc` | Inspect backend Service (check Endpoints list) |
| `kubectl describe svc frontend-svc` | Inspect frontend Service (check NodePort) |
| `kubectl exec deploy/frontend -- python -c "import httpx; print(httpx.get('http://backend-svc:8000/api/items').json())"` | Test internal service discovery from frontend pod |
| `minikube service frontend-svc --url` | Get the external URL for the frontend NodePort service |
| `curl <URL>/proxy/items` | Test external access to frontend via NodePort |

### Phase 4: Ingress

| Command | Description |
|---------|-------------|
| `kubectl apply -f k8s/ingress.yaml` | Apply the Ingress manifest |
| `kubectl get ingress` | List Ingress resources (shows IP, hosts, paths) |
| `minikube ip` | Get the minikube node IP for Ingress access |
| `curl http://$(minikube ip)/api/items` | Test Ingress routing to backend (/api path) |
| `curl http://$(minikube ip)/` | Test Ingress routing to frontend (/ path) |
| `minikube tunnel` | Create a network tunnel for Ingress access (Windows/macOS) |
| `kubectl describe ingress app-ingress` | Inspect Ingress rules and backend assignments |

### Phase 5: Horizontal Pod Autoscaler

| Command | Description |
|---------|-------------|
| `kubectl apply -f k8s/backend-hpa.yaml` | Apply the HPA targeting the backend Deployment |
| `kubectl get hpa` | Show HPA status (current vs target utilization) |
| `kubectl get hpa -w` | Watch HPA status continuously (wait ~60s for metrics) |
| `kubectl run load-gen --image=busybox --restart=Never -- /bin/sh -c "while true; do wget -q -O- http://backend-svc:8000/api/cpu-burn; done"` | Start a load generator pod to trigger HPA scale-up |
| `kubectl get pods -w` | Watch new pods appear as HPA scales up |
| `kubectl delete pod load-gen` | Stop the load generator |
| `kubectl describe hpa backend-hpa` | Inspect HPA details (events, scaling decisions) |

### Phase 6: Rolling Update in Action

| Command | Description |
|---------|-------------|
| `docker build -t backend:v2 apps/backend/` | Build backend v2 image with visible changes |
| `kubectl set image deploy/backend backend=backend:v2` | Trigger rolling update to v2 |
| `kubectl rollout status deploy/backend` | Watch the rolling update progress |
| `kubectl rollout undo deploy/backend` | Rollback to the previous revision |
| `kubectl rollout history deploy/backend` | View rollout revision history |
| `docker build -t frontend:v2 apps/frontend/` | Build frontend v2 image (if updating frontend too) |
| `kubectl set image deploy/frontend frontend=frontend:v2` | Trigger rolling update on frontend |

### Cleanup

| Command | Description |
|---------|-------------|
| `kubectl delete -f k8s/` | Delete all resources defined in k8s/ manifests |
| `minikube stop` | Stop the minikube cluster (preserves state) |
| `minikube delete` | Delete the minikube cluster entirely |

### Debugging

| Command | Description |
|---------|-------------|
| `kubectl get events --sort-by='.lastTimestamp'` | View cluster events sorted by time |
| `kubectl logs <pod-name>` | View pod logs |
| `kubectl logs <pod-name> -f` | Follow pod logs (live stream) |
| `kubectl logs <pod-name> --previous` | View logs from previous crash |
| `kubectl exec -it <pod-name> -- sh` | Shell into a running container |
| `kubectl get pods -o wide` | Show pods with node/IP details |
| `kubectl describe pod <pod-name>` | Inspect pod details (events, probe status) |
| `kubectl top pods` | Show CPU/memory usage per pod (requires metrics-server) |
| `kubectl top nodes` | Show CPU/memory usage per node |
| `kubectl get endpoints backend-svc` | Show which pod IPs are behind the backend Service |
| `kubectl get endpoints frontend-svc` | Show which pod IPs are behind the frontend Service |

## State

`not-started`
