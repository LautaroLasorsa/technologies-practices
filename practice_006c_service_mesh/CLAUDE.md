# Practice 006c: Service Mesh -- Sidecar & Envoy

## Technologies

- **Envoy Proxy** -- High-performance L7 proxy and communication bus for service mesh architectures (CNCF graduated project)
- **Sidecar Pattern** -- Deploying a helper process alongside each service to handle cross-cutting concerns (networking, observability, security)
- **Docker Compose** -- Multi-container orchestration for local service mesh simulation
- **FastAPI** -- Lightweight Python HTTP services (frontend and backend)

## Stack

- Python 3.12+ (FastAPI, httpx, uvicorn)
- Docker / Docker Compose
- Envoy Proxy (official `envoyproxy/envoy` image, v1.32)

## Theoretical Context

### What Is a Service Mesh?

A service mesh is a dedicated infrastructure layer for managing service-to-service communication in a microservices architecture. Instead of embedding networking logic (retries, timeouts, circuit breaking, TLS, load balancing, tracing) into every application, the mesh handles it transparently through **sidecar proxies** deployed alongside each service. The application sends traffic to its local sidecar (on `localhost`), and the sidecar handles routing, security, and observability before forwarding the request to the destination service's sidecar.

The core problem service meshes solve: as the number of microservices grows, implementing consistent networking policies across all services becomes unmanageable. Each team would need to implement retry logic, circuit breakers, mTLS, distributed tracing, and metrics collection -- in potentially different languages and frameworks. A service mesh centralizes these cross-cutting concerns at the infrastructure level, decoupling them from application code.

### Architecture: Data Plane vs Control Plane

A service mesh has two logical layers:

| Layer | Responsibility | Examples |
|-------|---------------|----------|
| **Data Plane** | The set of sidecar proxies deployed alongside every service instance. Handles the actual request forwarding, load balancing, health checking, TLS termination, and telemetry collection. All service-to-service traffic flows through the data plane. | Envoy Proxy, Linkerd2-proxy |
| **Control Plane** | The management layer that configures and coordinates the data plane proxies. Distributes routing rules, certificates, and service discovery information. Proxies never talk to each other for configuration -- they receive it from the control plane. | Istiod (Istio), Linkerd control plane, Consul Connect |

In production, the control plane uses APIs (like Envoy's xDS protocol) to dynamically push configuration to proxies without restarts. In this practice, we use **static configuration** (YAML files) to focus on understanding Envoy's architecture without the complexity of a control plane.

### The Sidecar Pattern

The sidecar pattern deploys a helper container alongside the main application container. In a service mesh context:

```
+-------------------------------------------------------+
|  Host / Pod                                            |
|                                                        |
|  +------------------+     +------------------------+   |
|  |  Application     |     |  Envoy Sidecar Proxy   |   |
|  |  (your code)     |---->|  (handles networking)  |   |
|  |  listens :8000   |     |  inbound  :10000       |   |
|  +------------------+     |  outbound :10001       |   |
|                           |  admin    :9901        |   |
|                           +------------------------+   |
+-------------------------------------------------------+
```

**How traffic flows in a service mesh:**

1. **Outbound**: Service A wants to call Service B. Instead of calling B directly, A sends the request to its own sidecar (localhost:10001). The sidecar resolves B's address, applies retry/timeout policies, and forwards the request to B's sidecar.
2. **Inbound**: B's sidecar receives the request on its ingress listener (port 10000), applies rate limiting/auth checks, and forwards to B's application (localhost:8000).
3. **The application never communicates directly with other services.** All inter-service traffic is intercepted by the sidecar. The application only knows about `localhost`.

**Benefits of the sidecar pattern:**
- **Language-agnostic**: Whether your service is Python, Go, Java, or Rust, the same Envoy sidecar handles networking
- **Separation of concerns**: Application code focuses on business logic; networking policies are in the sidecar config
- **Consistent policies**: Every service gets the same retry, timeout, and circuit breaking behavior
- **Zero application changes**: Adding observability or mTLS requires no code modifications

### Envoy Proxy: Architecture & Configuration

Envoy is a high-performance, open-source edge and service proxy designed for cloud-native applications. Originally built at Lyft in 2015, it became the universal data plane proxy for service meshes (used by Istio, AWS App Mesh, Consul Connect, and others). Envoy is written in C++ for performance but configured entirely through YAML/JSON.

**Envoy's core abstractions:**

| Concept | Description |
|---------|-------------|
| **Listener** | A named network location (IP + port) where Envoy accepts connections. Each listener has a filter chain that processes incoming traffic. A sidecar typically has two listeners: one for inbound traffic (from other services) and one for outbound traffic (from the local application). |
| **Filter Chain** | An ordered list of network filters applied to a connection. Filters process data at different protocol layers (L3/L4 TCP, L7 HTTP). The most important filter is the HTTP Connection Manager (HCM). |
| **HTTP Connection Manager (HCM)** | A network filter that converts raw bytes into HTTP requests. It manages HTTP/1.1 and HTTP/2 codec, header manipulation, access logging, and route resolution. Configured with `envoy.filters.network.http_connection_manager`. |
| **Route Configuration** | Inside the HCM, routes map incoming requests to clusters based on URL path, headers, or other criteria. Routes define the "if request matches X, send to cluster Y" rules. |
| **Virtual Host** | A logical grouping of routes under a domain name. Multiple virtual hosts can share a listener (like Apache/Nginx virtual hosts). |
| **Cluster** | A group of upstream hosts (endpoints) that Envoy can route traffic to. Each cluster defines connection parameters: timeout, load balancing policy, health checks, and circuit breakers. |
| **Endpoint** | A single network address (IP:port) within a cluster. Envoy load-balances across endpoints in a cluster. |
| **HTTP Filter** | Plugins that process HTTP requests/responses within the HCM. Includes the `envoy.filters.http.router` (mandatory, performs routing) and optional filters like rate limiting, CORS, and fault injection. |
| **Admin Interface** | A built-in HTTP API (typically on port 9901) for inspecting Envoy's runtime state: stats, config, clusters, listeners, and health. Essential for debugging. |

**Request flow through Envoy:**

```
Downstream             Envoy                                    Upstream
(client)    ->  [Listener :10000]                              (service)
                    |
                [Filter Chain]
                    |
                [HCM Filter]
                    |
                [HTTP Filters] -> [Router Filter]
                    |                   |
                [Route Config]    [Cluster Selection]
                    |                   |
                [Virtual Host]    [Load Balancer]
                    |                   |
                [Route Match]     [Endpoint :8000] -> upstream service
```

**Configuration approaches:**
- **Static** (this practice): All listeners, clusters, and routes defined in YAML at startup. Simple, predictable, good for learning.
- **Dynamic (xDS)**: Envoy fetches configuration from a control plane via gRPC streams. Supports runtime changes without restarts. Used in production (Istio's Pilot, Consul's Connect).

### Traffic Management Features

Envoy provides rich traffic management capabilities configured per-route or per-cluster:

**Retry policies**: Automatically retry failed requests based on configurable conditions (5xx errors, connection failures, gateway errors). Supports retry budgets to prevent retry storms.

**Timeouts**: Connection timeouts (how long to wait for TCP handshake), request timeouts (how long to wait for a complete response), and idle timeouts (how long to keep an idle connection open).

**Circuit breaking**: Per-cluster thresholds that limit the maximum number of connections, pending requests, requests, and retries. When a threshold is exceeded, Envoy short-circuits the request with a 503. Unlike application-level circuit breakers (practice 052), Envoy's circuit breakers are connection/concurrency-based, not failure-rate-based.

**Load balancing**: Supports round-robin, least-request, random, ring-hash, and Maglev algorithms. Configurable per cluster.

### Observability in Envoy

Envoy was designed with observability as a first-class feature:

- **Metrics**: Emits detailed statistics (requests/sec, latency histograms, error rates, circuit breaker trips) per listener, cluster, and route. Exposed via the admin `/stats` endpoint in Prometheus format.
- **Access logging**: Configurable per-listener access logs with request/response details, upstream info, timing, and flags. Supports stdout, file, and gRPC logging backends.
- **Distributed tracing**: Generates and propagates trace headers (B3, W3C TraceContext) for distributed tracing systems (Zipkin, Jaeger, OpenTelemetry). Envoy generates a unique request ID and propagates trace context across services.

### Ecosystem Comparison

| Feature | Envoy (Data Plane) | Istio (Mesh) | Linkerd (Mesh) |
|---------|-------------------|--------------|----------------|
| **Architecture** | Standalone proxy | Control plane + Envoy sidecars | Control plane + linkerd2-proxy (Rust) |
| **Performance** | High (C++, ~154MB/sidecar) | High (Envoy-based) | Very high (Rust proxy, ~17MB/sidecar) |
| **Configuration** | YAML/xDS API | CRDs (VirtualService, DestinationRule) | CRDs + annotations |
| **Learning curve** | Moderate | Steep | Gentle |
| **Use without K8s** | Yes (Docker, bare metal) | No (requires Kubernetes) | No (requires Kubernetes) |
| **Circuit breaking** | Connection/concurrency limits | Via DestinationRule | Failure accrual (EMA) |
| **mTLS** | Manual cert management | Automatic (SPIFFE identities) | Automatic (built-in CA) |

Envoy can be used standalone (as in this practice) or as the data plane component of a full service mesh (Istio, Consul Connect, AWS App Mesh). Understanding Envoy's configuration is fundamental regardless of which control plane you adopt.

### References

- [Envoy Documentation: What is Envoy](https://www.envoyproxy.io/docs/envoy/latest/intro/what_is_envoy)
- [Envoy Configuration Overview](https://www.envoyproxy.io/docs/envoy/latest/configuration/overview/examples)
- [Envoy Static Configuration Quick Start](https://www.envoyproxy.io/docs/envoy/latest/start/quick-start/configuration-static)
- [Envoy Docker Image](https://www.envoyproxy.io/docs/envoy/latest/start/docker)
- [Envoy Circuit Breaking](https://www.envoyproxy.io/docs/envoy/latest/intro/arch_overview/upstream/circuit_breaking)
- [Envoy Double Proxy Sandbox](https://www.envoyproxy.io/docs/envoy/latest/start/sandboxes/double-proxy)
- [Codilime: Envoy Configuration in a Nutshell](https://codilime.com/blog/envoy-configuration/)
- [Tetrate: Get Started with Envoy in 5 Minutes](https://tetrate.io/blog/get-started-with-envoy-in-5-minutes)
- [iximiuz: Sidecar Proxy Pattern](https://iximiuz.com/en/posts/service-proxy-pod-sidecar-oh-my/)
- [Microsoft: Sidecar Pattern](https://learn.microsoft.com/en-us/azure/architecture/patterns/sidecar)
- [Istio vs Linkerd Comparison](https://www.buoyant.io/linkerd-vs-istio)

## Description

Build a **service mesh simulation** using Docker Compose where two Python microservices (frontend and backend) communicate exclusively through Envoy sidecar proxies. Each service has a dedicated Envoy proxy that handles all inbound and outbound traffic. The focus is on understanding **Envoy configuration** -- listeners, clusters, routes, filters, circuit breaking, retries, and observability.

### Architecture

```
                        Docker Network: mesh
  +--------------------------------------------------------------+
  |                                                              |
  |  +-------------------+          +-------------------+        |
  |  | frontend-envoy    |--------->| backend-envoy     |        |
  |  | :10000 (inbound)  |          | :10000 (inbound)  |        |
  |  | :10001 (outbound) |          | :9901  (admin)    |        |
  |  | :9901  (admin)    |          +--------+----------+        |
  |  +--------+----------+                   |                   |
  |           |                              |                   |
  |  +--------v----------+          +--------v----------+        |
  |  | frontend app      |          | backend app       |        |
  |  | FastAPI :8000     |          | FastAPI :8000      |        |
  |  | (calls backend    |          | (handles requests, |        |
  |  |  via envoy:10001) |          |  simulates delays) |        |
  |  +-------------------+          +--------------------+        |
  +--------------------------------------------------------------+
         |
    Host :10000  (entry point)
    Host :9901   (frontend envoy admin)
    Host :9902   (backend envoy admin)
```

**Traffic flow for a request:**
1. Client sends request to `localhost:10000` (frontend Envoy inbound listener)
2. Frontend Envoy forwards to frontend app (`localhost:8000`)
3. Frontend app processes request, calls backend via its Envoy sidecar (`frontend-envoy:10001`)
4. Frontend Envoy outbound listener routes to backend Envoy inbound listener (`backend-envoy:10000`)
5. Backend Envoy forwards to backend app (`localhost:8000`)
6. Response travels the reverse path

### What you'll learn

1. **Envoy listener configuration** -- How to define inbound (ingress) and outbound (egress) listeners with filter chains
2. **HTTP Connection Manager** -- The core Envoy filter that converts TCP to HTTP and manages routing
3. **Route configuration** -- Path-based routing with virtual hosts, prefix matching, and header-based routing
4. **Cluster definitions** -- Defining upstream services with connection parameters, load balancing, and DNS discovery
5. **Circuit breaking** -- Connection and concurrency limits that prevent cascading failures
6. **Retry policies** -- Automatic retries with configurable conditions and backoff
7. **Envoy admin interface** -- Using `/stats`, `/clusters`, `/config_dump` to inspect runtime state
8. **Sidecar communication pattern** -- How app -> local-sidecar -> remote-sidecar -> app traffic flows in a mesh

## Instructions

### Phase 1: Setup & Architecture (~10 min)

1. Review the architecture diagram above. Understand the two-sidecar pattern: each service has its own Envoy.
2. Read the Python services in `services/frontend/main.py` and `services/backend/main.py` -- they are fully implemented. Notice how the frontend calls backend via its own Envoy sidecar (`http://frontend-envoy:10001/`), NOT directly.
3. Key question: Why does the frontend call `frontend-envoy:10001` instead of `backend-envoy:10000` directly? What does this indirection buy us?

### Phase 2: Backend Envoy -- Inbound Listener (~20 min)

1. Open `envoy/backend-envoy.yaml`. This is the simpler sidecar -- it only needs an inbound listener (receives requests from other services' sidecars).
2. **User implements:** The inbound listener on port 10000 with an HTTP Connection Manager filter chain. This teaches the fundamental Envoy building blocks: a listener binds to a port, a filter chain processes connections, and the HCM filter converts TCP bytes into HTTP requests. Every Envoy config starts with this pattern.
3. **User implements:** The route configuration that forwards all traffic to the local backend app cluster. Routes are how Envoy decides where to send traffic. The simplest route uses a prefix match of "/" to catch all requests and forward to a single cluster.
4. **User implements:** The `backend_app` cluster definition pointing to `127.0.0.1:8000`. Clusters define upstream connection parameters -- the backend app runs on localhost because it shares a network namespace with its sidecar. Understanding this localhost relationship is key to the sidecar pattern.
5. Build and test: `docker compose up --build backend backend-envoy` and verify `curl http://localhost:10000/health` works through Envoy.

### Phase 3: Frontend Envoy -- Inbound + Outbound (~25 min)

1. Open `envoy/frontend-envoy.yaml`. The frontend sidecar needs TWO listeners: inbound (for external clients) and outbound (for the frontend app to reach backend).
2. **User implements:** The inbound listener on port 10000 (same structure as backend). This reinforces the pattern from Phase 2 and shows that every sidecar has the same inbound structure -- only the cluster target differs.
3. **User implements:** The outbound listener on port 10001 with path-based routing to the backend cluster. This is the key service mesh concept -- the application sends requests to its own sidecar's outbound port, and the sidecar routes to the correct upstream service. Path-based routing enables a single outbound port to reach multiple backends (e.g., `/api/` -> backend, `/auth/` -> auth-service).
4. **User implements:** Two cluster definitions -- one for the local frontend app, one for the remote backend-envoy. The backend cluster uses `STRICT_DNS` discovery to resolve `backend-envoy` via Docker DNS. This teaches how Envoy discovers upstream services -- in production, xDS replaces static DNS.
5. Build and test: `docker compose up --build` all services. Verify `curl http://localhost:10000/api/data` flows through both sidecars.

### Phase 4: Docker Compose Service Mesh (~15 min)

1. Open `docker-compose.yml`. The service definitions and network configuration tie the mesh together.
2. **User implements:** Network configuration ensuring sidecars and their apps share a network namespace (using `network_mode`). In Kubernetes, pods share a network namespace automatically. In Docker Compose, we use `network_mode: "service:<sidecar>"` so the app container shares the sidecar's network stack. This means the app and its sidecar communicate over `localhost` -- exactly like in production.
3. **User implements:** Health checks for Envoy using the admin API. Envoy's admin interface (`/ready` endpoint) reports when the proxy is fully initialized. Using this as a Docker health check ensures apps don't start until their sidecar is ready.
4. Test: `docker compose up --build` and verify the full request path.

### Phase 5: Resilience -- Circuit Breaking & Retries (~20 min)

1. Return to `envoy/frontend-envoy.yaml`.
2. **User implements:** Circuit breaking thresholds on the backend cluster. Envoy's circuit breakers are concurrency-based: they limit max connections, pending requests, and active requests per cluster. When a threshold is exceeded, Envoy returns 503 immediately (fail fast). This differs from the failure-rate-based circuit breaker in practice 052 -- Envoy's approach prevents resource exhaustion rather than detecting failure patterns.
3. **User implements:** Retry policy on the outbound route to backend. Retries specify which failures to retry on (5xx, connect-failure, gateway-error), how many times, and per-retry timeout. Envoy handles retries transparently -- the application never knows a retry happened.
4. Test: Use the backend's `/admin/mode/degraded` endpoint to simulate failures, then observe Envoy's retry behavior in access logs and stats.

### Phase 6: Observability & Inspection (~15 min)

1. Explore Envoy's admin interface: `http://localhost:9901` (frontend) and `http://localhost:9902` (backend)
2. Check stats: `curl http://localhost:9901/stats | grep -E "upstream_cx|upstream_rq|retry|circuit"` -- observe request counts, retry counts, circuit breaker stats
3. Check clusters: `curl http://localhost:9901/clusters` -- see upstream health, active connections, request counts
4. View config: `curl http://localhost:9901/config_dump` -- see the full resolved Envoy config
5. Run the load test: `python scripts/test_mesh.py` to generate traffic and observe metrics
6. Key question: How would you add a third service (e.g., "auth") to this mesh? What config changes are needed?

## Motivation

- **Industry standard**: Envoy is the universal data plane proxy -- used by Istio, AWS App Mesh, Consul Connect, and as a standalone proxy at companies like Lyft, Airbnb, Stripe, and Slack
- **Complements practice 052**: Practice 052 implements resilience patterns (circuit breaker, rate limiting) in application code. This practice shows the same patterns implemented at the infrastructure level -- no code changes needed
- **Foundation for Kubernetes**: Understanding Envoy configuration is essential before working with Istio/Linkerd (practices 006a/006b cover K8s basics)
- **Separation of concerns**: Service meshes embody the sidecar pattern -- a key architectural pattern for microservices
- **Observability without instrumentation**: Envoy collects metrics, traces, and access logs without any application code changes -- critical for polyglot microservice environments

## Commands

All commands are run from the `practice_006c_service_mesh/` folder root.

### Phase 1: Build & Start

| Command | Description |
|---------|-------------|
| `docker compose build` | Build all service images (frontend, backend) |
| `docker compose up --build` | Build images and start all services (foreground, logs streamed) |
| `docker compose up -d --build` | Build and start all services in detached mode |
| `docker compose up --build backend backend-envoy` | Start only backend and its Envoy sidecar (Phase 2 testing) |
| `docker compose ps` | Check status of all containers |

### Phase 2-3: Testing Traffic Flow

| Command | Description |
|---------|-------------|
| `curl http://localhost:10000/health` | Health check through frontend Envoy (or backend Envoy in Phase 2) |
| `curl http://localhost:10000/` | Frontend root endpoint through Envoy |
| `curl http://localhost:10000/api/data` | Frontend calls backend through the mesh -- full sidecar-to-sidecar flow |
| `curl http://localhost:10000/api/data?delay=2` | Request with simulated backend delay (test timeout behavior) |
| `curl http://localhost:10000/api/items` | List items endpoint through the mesh |
| `curl -X POST http://localhost:10000/api/items -H "Content-Type: application/json" -d "{\"name\": \"test\", \"price\": 9.99}"` | Create an item through the mesh |

### Phase 4: Backend Failure Simulation

| Command | Description |
|---------|-------------|
| `curl -X POST http://localhost:10000/api/admin/mode/healthy` | Set backend to healthy mode (100% success) |
| `curl -X POST http://localhost:10000/api/admin/mode/degraded` | Set backend to degraded mode (50% errors, slow responses) |
| `curl -X POST http://localhost:10000/api/admin/mode/down` | Set backend to down mode (100% errors) |

### Phase 5: Envoy Admin Interface

| Command | Description |
|---------|-------------|
| `curl http://localhost:9901/ready` | Check frontend Envoy readiness |
| `curl http://localhost:9902/ready` | Check backend Envoy readiness |
| `curl http://localhost:9901/stats` | All frontend Envoy statistics |
| `curl http://localhost:9902/stats` | All backend Envoy statistics |
| `curl http://localhost:9901/stats?filter=cluster.backend` | Frontend Envoy stats filtered to backend cluster metrics |
| `curl "http://localhost:9901/stats?filter=upstream_rq"` | Request count statistics (total, retries, errors) |
| `curl "http://localhost:9901/stats?filter=circuit_breaker"` | Circuit breaker statistics |
| `curl http://localhost:9901/clusters` | Frontend Envoy cluster membership and health |
| `curl http://localhost:9902/clusters` | Backend Envoy cluster membership and health |
| `curl http://localhost:9901/config_dump` | Full resolved frontend Envoy configuration |
| `curl http://localhost:9901/listeners` | List active listeners on frontend Envoy |

### Phase 6: Load Test

| Command | Description |
|---------|-------------|
| `pip install httpx` | Install test script dependency (if not using uv) |
| `python scripts/test_mesh.py` | Run load test: sends requests through the mesh and reports metrics |

### Inspection & Debugging

| Command | Description |
|---------|-------------|
| `docker compose logs frontend-envoy` | View frontend Envoy access logs |
| `docker compose logs backend-envoy` | View backend Envoy access logs |
| `docker compose logs frontend` | View frontend app logs |
| `docker compose logs backend` | View backend app logs |
| `docker compose logs -f` | Follow all logs in real time |
| `docker compose exec frontend-envoy sh` | Open shell inside frontend Envoy container |

### Cleanup

| Command | Description |
|---------|-------------|
| `docker compose down` | Stop and remove all containers and networks |
| `docker compose down -v` | Stop, remove containers, and delete any volumes |
| `docker compose down --rmi local` | Stop and remove locally-built images |
| `docker compose down -v --rmi local` | Full cleanup: containers, volumes, and images |

## State

`not-started`
