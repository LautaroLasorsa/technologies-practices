# Practice 005: Docker Compose Multi-Service Orchestration & Limits

## Technologies

- **Docker Compose** -- Declarative multi-container orchestration (v2 spec)
- **Resource Limits** -- CPU and memory constraints via `deploy.resources`
- **Health Checks** -- Container readiness probes with `healthcheck`
- **Service Dependencies** -- Startup ordering with `depends_on` + `condition: service_healthy`
- **Custom Networks** -- Network isolation and service discovery by DNS name
- **Profiles** -- Optional services activated by profile name
- **Replicas** -- Horizontal scaling via `deploy.replicas`
- **Nginx** -- Reverse proxy and load balancer across replicated services
- **Redis** -- In-memory data store used as task queue and cache

## Stack

- Python 3.12+ (FastAPI, redis-py, httpx)
- Docker / Docker Compose
- Nginx (official image)
- Redis (official image)

## Theoretical Context

Docker Compose is a tool for defining and running multi-container applications through a single declarative YAML file. It solves the problem of manually orchestrating multiple containers with their networks, volumes, and dependencies -- instead of running separate `docker run` commands with dozens of flags, you declare the entire stack once and manage it with simple commands like `docker compose up` and `docker compose down`.

Internally, Docker Compose translates the `docker-compose.yml` specification into Docker API calls. When you run `docker compose up`, Compose parses the YAML, creates the declared networks and volumes first, then starts containers in dependency order (respecting `depends_on` constraints). It monitors health checks and waits for services marked with `condition: service_healthy` before starting dependents. Each service definition maps to one or more containers (when using `deploy.replicas`), and Compose assigns unique names to each replica. Services communicate via a shared bridge network where each service name becomes a resolvable DNS entry (e.g., `redis` resolves to the Redis container's IP).

Docker Compose adopts a three-layer model: **projects** (the top-level unit, typically one `docker-compose.yml` per project), **services** (logical application components like "api_gateway" or "worker"), and **containers** (runtime instances of service images). Resource constraints (`deploy.resources`) are translated into Docker's cgroup limits (CPU quotas and memory caps). Health checks run inside containers at specified intervals, and Compose tracks their exit codes -- services marked unhealthy are excluded from `depends_on` readiness gates, preventing cascading failures during startup.

| Concept | Description |
|---------|-------------|
| **Service** | A logical component of your application (e.g., "nginx", "redis"). One service can spawn multiple container replicas. |
| **Network** | An isolated virtual network segment. Services on the same network resolve each other by service name via built-in DNS. |
| **Volume** | Persistent storage mounted into containers. Named volumes survive `docker compose down`; bind mounts sync host directories. |
| **Health Check** | A command run periodically inside the container. Exit code 0 = healthy, non-zero = unhealthy. Used for startup ordering. |
| **Dependency** | `depends_on` controls startup order. `condition: service_healthy` waits for health checks before starting dependents. |
| **Profile** | Optional service groups (e.g., `debug`). Services with profiles only start when explicitly requested via `--profile`. |
| **Resource Limits** | `deploy.resources.limits` caps CPU/memory; `reservations` guarantees minimum allocation. Maps to Docker's cgroup constraints. |

Docker Compose is ideal for **local development** and **integration testing** but less suitable for production multi-node orchestration (where Kubernetes dominates). It lacks built-in features for automatic scaling across machines, rolling updates with health-aware traffic shifting, or distributed secret management. Alternatives include **Docker Swarm** (simpler than Kubernetes but less widely adopted), **Kubernetes** (production-grade but heavier for local dev), and cloud-native orchestrators like **AWS ECS** or **Google Cloud Run** (managed but vendor-locked). Compose's strength is developer productivity: spin up a realistic multi-service environment with one command, no cluster setup required.

## Description

Build a **Task Processing Pipeline** with five services orchestrated by Docker Compose:

```
Client --> Nginx (reverse proxy) --> API Gateway (FastAPI)
                                 --> Dashboard  (FastAPI)
           API Gateway --> Redis (queue) --> Worker(s)
           Dashboard   --> Redis (reads status)
```

- **API Gateway** -- Accepts task submissions via REST, pushes them to a Redis queue
- **Worker** -- Polls Redis for tasks, "processes" them (simulated delay), writes results back
- **Dashboard** -- Shows queue depth, completed tasks, and worker health
- **Nginx** -- Routes `/api/*` to the API Gateway and `/dashboard/*` to the Dashboard
- **Redis** -- Shared message broker and result store

All Python service code is **fully implemented**. The learning focus is on `docker-compose.yml` and the `Dockerfile`s, where `TODO(human)` markers guide you through Docker Compose orchestration concepts.

### What you'll learn

1. **Resource limits** -- `deploy.resources.limits` and `reservations` for CPU and memory
2. **Health checks** -- `healthcheck` with `test`, `interval`, `timeout`, `retries`, `start_period`
3. **Startup ordering** -- `depends_on` with `condition: service_healthy` vs bare `depends_on`
4. **Custom networks** -- Isolating frontend/backend traffic with named networks
5. **Service discovery** -- Containers resolving each other by service name (DNS)
6. **Volumes** -- Named volumes for Redis persistence, bind mounts for dev reload
7. **Environment variables** -- `.env` file and `environment:` block
8. **Profiles** -- Conditionally starting services (e.g., debug tools)
9. **Replicas** -- Scaling workers horizontally with `deploy.replicas`
10. **Nginx as reverse proxy** -- Routing to upstream services by Docker DNS name

## Instructions

### Phase 1: Read the Code (~10 min)

1. Explore the fully-implemented Python services in `services/`:
   - `api_gateway/main.py` -- FastAPI app that enqueues tasks into Redis
   - `worker/main.py` -- Polling loop that dequeues, processes, and stores results
   - `dashboard/main.py` -- FastAPI app that reads queue/result stats from Redis
2. Read `nginx/nginx.conf` to understand the reverse proxy routing
3. Read `scripts/test_pipeline.py` to see how the end-to-end test works

### Phase 2: Dockerfiles (~15 min)

Open each `Dockerfile` in `services/*/Dockerfile`. Each has `TODO(human)` markers:

1. **Base image and workdir** -- Choose a slim Python base image
2. **Dependency installation** -- Copy requirements and install with pip
3. **Application copy** -- Copy source code into the image
4. **Expose and CMD** -- Declare port and startup command

### Phase 3: docker-compose.yml Core Services (~20 min)

Open `docker-compose.yml`. Fill in the `TODO(human)` markers in order:

1. **Redis service** -- Official image, port mapping, named volume for persistence
2. **API Gateway service** -- Build context, port, environment variables, bind mount
3. **Dashboard service** -- Similar to API Gateway but different port/route
4. **Worker service** -- Build context, environment variables (no port needed)

### Phase 4: Health Checks & Dependencies (~15 min)

1. Add `healthcheck` to Redis (use `redis-cli ping`)
2. Add `healthcheck` to API Gateway (use `curl` or Python HTTP check)
3. Add `depends_on` with `condition: service_healthy` so services wait for Redis
4. Test: `docker compose up` and observe startup ordering in logs

### Phase 5: Custom Networks (~10 min)

1. Define two networks: `frontend` (Nginx + API + Dashboard) and `backend` (API + Worker + Redis + Dashboard)
2. Assign each service to the correct network(s)
3. Verify: Worker cannot reach Nginx, Nginx cannot reach Redis directly

### Phase 6: Resource Limits (~10 min)

1. Add `deploy.resources.limits` for CPU and memory to each service
2. Add `deploy.resources.reservations` for Redis (guaranteed minimum)
3. Verify: `docker compose up` and check limits with `docker stats`

### Phase 7: Replicas & Profiles (~15 min)

1. Scale the Worker to 3 replicas with `deploy.replicas`
2. Add a `redis-commander` debug service with `profiles: ["debug"]`
3. Test: `docker compose up` (no debug), `docker compose --profile debug up` (with debug UI)
4. Observe workers sharing the Redis queue (tasks distributed across replicas)

### Phase 8: End-to-End Test (~10 min)

1. Run `docker compose up --build`
2. Execute `python scripts/test_pipeline.py` to submit tasks and check results
3. Open `http://localhost/dashboard/` to see the live dashboard
4. Check resource usage with `docker stats`

## Motivation

- **Production essential**: Docker Compose is the standard tool for local multi-service development and CI pipelines
- **Resource awareness**: Understanding CPU/memory limits prevents container OOM kills and noisy-neighbor issues in shared environments
- **Health checks & ordering**: Critical for reliable deployments -- services that start before dependencies are ready cause cascading failures
- **Network isolation**: Mirrors production security practices (frontend/backend separation)
- **Scaling patterns**: Replicas + reverse proxy is the foundation for horizontal scaling (preparation for Kubernetes in practices 006a/006b)

## References

- [Docker Compose Deploy Specification](https://docs.docker.com/reference/compose-file/deploy/)
- [Docker Resource Constraints](https://docs.docker.com/engine/containers/resource_constraints/)
- [Docker Compose Services Reference](https://docs.docker.com/reference/compose-file/services/)
- [Docker Compose Startup Order](https://docs.docker.com/compose/how-tos/startup-order/)
- [Docker Compose Networking](https://docs.docker.com/compose/how-tos/networking/)
- [Docker Compose Profiles](https://docs.docker.com/compose/how-tos/profiles/)
- [Nginx Reverse Proxy Guide](https://docs.nginx.com/nginx/admin-guide/web-server/reverse-proxy/)
- [Docker Compose Health Checks Guide](https://last9.io/blog/docker-compose-health-checks/)

## Commands

All commands are run from the `practice_005_docker_compose/` folder root.

### Phase 2-3: Build & Start Core Services

| Command | Description |
|---------|-------------|
| `docker compose build` | Build all service images (api_gateway, worker, dashboard) from their Dockerfiles |
| `docker compose build api_gateway` | Build only the API Gateway image |
| `docker compose build worker` | Build only the Worker image |
| `docker compose build dashboard` | Build only the Dashboard image |
| `docker compose up` | Start all services (foreground, logs streamed to terminal) |
| `docker compose up -d` | Start all services in detached (background) mode |
| `docker compose up --build` | Rebuild images and start all services in one step |

### Phase 4-6: Health Checks, Networks & Resource Limits

| Command | Description |
|---------|-------------|
| `docker compose up` | Start services and observe startup ordering from health checks in logs |
| `docker compose ps` | List running containers and their health status |
| `docker stats` | Live resource usage (CPU, memory) per container -- verify resource limits |
| `docker compose logs redis` | View Redis container logs (verify health check pings) |
| `docker compose logs api_gateway` | View API Gateway container logs |
| `docker compose logs dashboard` | View Dashboard container logs |
| `docker compose logs worker` | View Worker container logs (across all replicas) |

### Phase 5: Network Isolation Verification

| Command | Description |
|---------|-------------|
| `docker compose exec worker ping -c 2 nginx` | Verify Worker CANNOT reach Nginx (should fail -- different networks) |
| `docker compose exec nginx ping -c 2 redis` | Verify Nginx CANNOT reach Redis directly (should fail -- different networks) |
| `docker compose exec api_gateway ping -c 2 redis` | Verify API Gateway CAN reach Redis (both on backend network) |
| `docker compose exec nginx ping -c 2 api_gateway` | Verify Nginx CAN reach API Gateway (both on frontend network) |
| `docker network ls` | List all Docker networks (verify frontend and backend were created) |
| `docker network inspect task_pipeline_frontend` | Inspect frontend network -- see which containers are attached |
| `docker network inspect task_pipeline_backend` | Inspect backend network -- see which containers are attached |

### Phase 7: Replicas & Profiles

| Command | Description |
|---------|-------------|
| `docker compose up --build` | Start services with worker scaled to 3 replicas (no debug profile) |
| `docker compose --profile debug up --build` | Start all services INCLUDING redis-commander debug UI |
| `docker compose --profile debug up -d` | Start all services with debug profile in detached mode |

### Phase 8: End-to-End Test

| Command | Description |
|---------|-------------|
| `docker compose up --build` | Build and start the full pipeline |
| `pip install -r scripts/requirements.txt` | Install test script dependency (httpx) |
| `python scripts/test_pipeline.py` | Run end-to-end test: submit 5 tasks, poll for completion, check stats |

### Inspection & Debugging

| Command | Description |
|---------|-------------|
| `docker compose ps` | Show running containers, ports, and health status |
| `docker compose logs` | Show aggregated logs from all services |
| `docker compose logs -f` | Follow (tail) logs from all services in real time |
| `docker compose logs -f worker` | Follow logs from worker replicas only |
| `docker stats` | Live CPU/memory/network usage per container |
| `docker compose exec redis redis-cli` | Open interactive Redis CLI inside the Redis container |
| `docker compose exec redis redis-cli LLEN tasks:pending` | Check pending queue depth directly in Redis |
| `docker compose exec redis redis-cli SMEMBERS workers:active` | List active worker IDs registered in Redis |
| `docker compose exec redis redis-cli GET stats:total_completed` | Check total completed tasks counter |
| `docker compose config` | Validate and display the resolved docker-compose.yml |

### Cleanup

| Command | Description |
|---------|-------------|
| `docker compose down` | Stop and remove all containers and default networks |
| `docker compose down -v` | Stop containers and remove named volumes (deletes Redis data) |
| `docker compose down --rmi local` | Stop containers and remove locally-built images |
| `docker compose down -v --rmi local` | Full cleanup: containers, volumes, and local images |

### Web Endpoints (accessible after `docker compose up`)

| URL | Description |
|-----|-------------|
| `http://localhost/` | Root -- redirects to Dashboard |
| `http://localhost/api/health` | API Gateway health check (via Nginx) |
| `http://localhost/api/tasks` | POST to submit a task (via Nginx) |
| `http://localhost/api/stats` | API Gateway queue statistics (via Nginx) |
| `http://localhost/dashboard/` | Dashboard HTML page with auto-refreshing stats |
| `http://localhost/dashboard/stats` | Dashboard JSON stats endpoint (via Nginx) |
| `http://localhost/dashboard/health` | Dashboard health check (via Nginx) |
| `http://localhost:8000/health` | API Gateway health check (direct, bypassing Nginx) |
| `http://localhost:8001/health` | Dashboard health check (direct, bypassing Nginx) |
| `http://localhost:8081` | Redis Commander UI (only with `--profile debug`) |

## State

`not-started`
