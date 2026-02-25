# Practice 079: Distributed Task Queues: Celery & ARQ Worker Patterns

## Technologies

- **Celery** -- Distributed task queue for Python with support for multiple brokers, canvas workflows, and scheduling
- **ARQ** -- Lightweight asyncio-native task queue backed by Redis
- **Redis** -- In-memory data store used as both message broker and result backend
- **Flower** -- Real-time web monitoring tool for Celery workers and tasks
- **FastAPI** -- HTTP API for dispatching async tasks and querying results
- **Docker Compose** -- Multi-service orchestration

## Stack

- Python 3.12+ (uv)
- Docker / Docker Compose

## Theoretical Context

### Distributed Task Queues: Offloading Work from the Request Cycle

A distributed task queue decouples time-consuming or unreliable operations from the synchronous request-response cycle. Instead of making an HTTP client wait while your server sends an email, resizes an image, or queries a slow third-party API, the server publishes a **task message** to a broker and responds immediately. A separate **worker process** picks up the message and executes the task asynchronously. This pattern is foundational to any production web application that needs to scale beyond trivial workloads.

The architecture has three core components: a **producer** (the code that creates tasks), a **broker** (the message transport -- Redis, RabbitMQ, SQS), and one or more **workers** (processes that consume and execute tasks). Optionally, a **result backend** stores return values and task metadata so producers can poll for results later.

### Celery: The Industry Standard

Celery is the dominant Python task queue, powering systems at Instagram, Mozilla, Stripe, and thousands of production deployments. It has been in active development since 2009 and reached version 5.6 in 2025 with Python 3.9-3.13 support.

**Architecture:**

```
Producer (your app)
    |
    v
Broker (Redis / RabbitMQ / SQS)
    |
    v
Worker Pool (prefork / gevent / eventlet / solo)
    |
    v
Result Backend (Redis / DB / S3)
```

**Key concepts:**

| Concept | Description |
|---------|-------------|
| **Task** | A decorated Python function that can be sent to workers for async execution. Defined with `@app.task` or `@shared_task`. |
| **Broker** | Message transport that queues tasks. Redis is fast and simple; RabbitMQ supports advanced routing. Celery abstracts the broker behind a URL. |
| **Result Backend** | Stores task return values, state (PENDING, STARTED, SUCCESS, FAILURE), and metadata. Redis doubles as both broker and backend. |
| **Worker** | An OS process running `celery -A app worker` that consumes tasks from the broker and executes them. |
| **Beat** | A scheduler process (`celery -A app beat`) that publishes periodic tasks at configured intervals (crontab, timedelta). |
| **Canvas** | Workflow primitives for composing tasks: `chain` (sequential), `group` (parallel), `chord` (parallel + callback), `starmap` (map arguments). |
| **Signature** | An immutable description of a task call (function + args + kwargs) that can be passed around, composed, and executed later. Created with `task.s()` or `task.si()`. |
| **Routing** | Directing specific tasks to specific queues so dedicated workers can consume them. Configured via `task_routes` or per-call `queue=` argument. |
| **Late Ack** | With `acks_late=True`, the broker message is acknowledged only after the task completes (not when received). Required for crash-safe at-least-once delivery. |
| **Idempotency** | Tasks must produce the same result when executed multiple times. Critical when using `acks_late` or retry policies. |

**Worker concurrency models:**

| Pool | Mechanism | Best For | Concurrency | Memory |
|------|-----------|----------|-------------|--------|
| **prefork** (default) | Multiprocessing (forks) | CPU-bound tasks | ~CPU count | High (N copies of process) |
| **gevent** | Green threads (cooperative) | I/O-bound tasks | Hundreds-thousands | Low (single process) |
| **eventlet** | Green threads (monkey-patched) | I/O-bound tasks | Hundreds-thousands | Low (single process) |
| **solo** | Single-threaded, in-process | Debugging / simple cases | 1 | Minimal |

Prefork sidesteps the GIL by forking worker processes, making it the only choice for CPU-bound work. Gevent/eventlet use cooperative multitasking within a single process, ideal for I/O-bound tasks (HTTP calls, DB queries) where you need high concurrency without memory overhead.

**Canvas primitives** allow composing tasks into workflows:

- **`chain(a.s(), b.s(), c.s())`** -- Execute a, pass result to b, pass result to c (pipeline)
- **`group(a.s(1), a.s(2), a.s(3))`** -- Execute all in parallel, collect results
- **`chord(group(...), callback.s())`** -- Execute group in parallel, when ALL complete, pass results to callback
- **`starmap(task, [(arg1a, arg1b), (arg2a, arg2b)])`** -- Map argument tuples to task calls

**Retry and failure handling:**

Celery provides `autoretry_for` (list of exception types to retry on), `max_retries`, `retry_backoff` (exponential with jitter), and `retry_backoff_max`. For tasks that exhaust retries, you can define `on_failure` handlers or route to a dead-letter queue. The `acks_late=True` + `reject_on_worker_lost=True` combination ensures tasks are re-delivered if a worker crashes mid-execution.

**Task routing** directs tasks to specific queues. Workers subscribe to queues with `--queues=queue_name`. This enables dedicated workers for different workload types (e.g., CPU-heavy image processing on powerful machines, lightweight email sending on smaller instances).

### ARQ: The Async-Native Alternative

ARQ (Asynchronous Redis Queue) is a lightweight task queue designed from the ground up for Python's `asyncio`. Where Celery is synchronous and uses process-based concurrency, ARQ is fully async and can handle hundreds of concurrent I/O-bound tasks within a single process.

**Key differences from Celery:**

| Aspect | Celery | ARQ |
|--------|--------|-----|
| **Async support** | Bolted on (eventlet/gevent) | Native asyncio |
| **Broker** | Redis, RabbitMQ, SQS | Redis only |
| **Dependencies** | Heavy (~20 transitive) | Minimal (~3 transitive) |
| **Monitoring** | Flower (rich web UI) | Basic CLI, no built-in UI |
| **Canvas workflows** | chain, group, chord, starmap | None (manual composition) |
| **Scheduling** | Beat (crontab, timedelta) | Built-in cron (simpler) |
| **Maturity** | 15+ years, battle-tested | Newer, smaller community |
| **Best for** | CPU-bound, complex workflows | I/O-bound async workloads |

ARQ defines tasks as plain `async def` functions that receive a `ctx` dict as first argument. Workers are configured via a `WorkerSettings` class that lists functions, Redis settings, and concurrency limits. Jobs are enqueued via `await redis.enqueue_job('function_name', arg1, arg2)`.

**When to choose ARQ over Celery:**
- Your entire stack is async (FastAPI, httpx, asyncpg)
- Tasks are primarily I/O-bound (API calls, DB queries, file downloads)
- You want minimal dependencies and simple configuration
- You don't need complex workflows (chain/chord) or multi-broker support

**When to stick with Celery:**
- CPU-bound tasks (image processing, data crunching) -- prefork pool
- Complex workflow composition (chain, group, chord)
- Need RabbitMQ's advanced routing, or SQS for AWS-native queuing
- Require production monitoring (Flower) and mature ecosystem
- Large team / existing Celery infrastructure

### Idempotency in Task Queues

Task idempotency means executing a task multiple times with the same arguments produces the same result and side effects. This is critical because:

1. **At-least-once delivery**: Brokers may redeliver messages after timeouts or worker crashes
2. **Retries**: Failed tasks are re-executed, potentially after partial completion
3. **Duplicate messages**: Network issues can cause duplicate task publications

Common idempotency patterns:
- **Dedup key**: Store a hash of (task_name, args) in Redis with TTL; skip if already processed
- **Database upsert**: Use `INSERT ... ON CONFLICT DO UPDATE` instead of blind writes
- **Idempotency token**: Client sends a unique token; server checks if already processed
- **Status check before action**: Check current state before performing side effects

## Description

Build a **Task Processing System** that demonstrates both Celery and ARQ patterns using Redis as broker and result backend. The system includes:

1. **Celery fundamentals** -- Define tasks, invoke them, inspect results
2. **Canvas workflows** -- Chain, group, and chord for composing task pipelines
3. **Retry policies** -- Exponential backoff with jitter for transient failures
4. **Idempotent tasks** -- Deduplication using Redis-based idempotency keys
5. **Task routing** -- Direct tasks to specific queues consumed by dedicated workers
6. **Periodic scheduling** -- Celery Beat for cron-like recurring tasks
7. **ARQ comparison** -- Same task implemented with ARQ to contrast sync vs async patterns
8. **FastAPI integration** -- HTTP endpoints that dispatch Celery/ARQ tasks and poll results

### Architecture

```
                    FastAPI (HTTP API)
                    /              \
                   v                v
            Celery Tasks       ARQ Tasks
                   \                /
                    v              v
                      Redis 7.x
                   (broker + backend)
                    /              \
                   v                v
         Celery Workers        ARQ Worker
         (prefork pool)        (asyncio)
              |
         Flower (monitoring)
              |
         Celery Beat (scheduler)
```

### What you'll learn

1. **Celery task lifecycle** -- Define, dispatch, monitor, and retrieve task results
2. **Canvas workflows** -- Compose tasks into chains, groups, and chords
3. **Retry & resilience** -- Exponential backoff, max retries, dead-letter handling
4. **Idempotency** -- Redis-based dedup keys to prevent duplicate execution
5. **Task routing** -- Dedicated queues for different workload types
6. **Worker pools** -- Understand prefork vs solo concurrency
7. **Periodic scheduling** -- Celery Beat with crontab and interval schedules
8. **ARQ async alternative** -- When and how to use ARQ instead of Celery
9. **FastAPI integration** -- Dispatching background tasks from HTTP endpoints

## Instructions

### Phase 1: Setup & Infrastructure (~5 min)

1. Start Redis, Flower, and all workers with `docker compose up -d`
2. Verify Redis is running: `docker compose exec redis redis-cli ping`
3. Open Flower at `http://localhost:5555` to see the monitoring dashboard
4. Key question: Why use Redis as both broker AND result backend? What are the trade-offs vs separate systems (e.g., RabbitMQ as broker + PostgreSQL as backend)?

### Phase 2: Basic Celery Tasks (~15 min)

1. Review the Celery app initialization in `src/celery_app.py`
2. Review the task skeleton in `src/tasks_basic.py`
3. **User implements:** `add` task -- simple synchronous task that adds two numbers
4. **User implements:** `fetch_url_content` task -- simulates fetching a URL (uses `time.sleep` to simulate I/O)
5. Run `ex01_basic_tasks.py` to dispatch tasks and inspect results
6. Watch Flower dashboard to see tasks appear, execute, and complete
7. Key question: What's the difference between `task.delay(args)` and `task.apply_async(args, kwargs)`? When would you use each?

### Phase 3: Canvas Workflows (~20 min)

1. Review the canvas task definitions in `src/tasks_canvas.py`
2. **User implements:** `chain` workflow -- pipeline that processes data through multiple stages
3. **User implements:** `group` workflow -- fan-out parallel execution
4. **User implements:** `chord` workflow -- fan-out + callback when all complete
5. Run `ex02_canvas_workflows.py` to execute each workflow and observe Flower
6. Key question: What happens if one task in a chord header fails? Does the callback still run?

### Phase 4: Retry & Idempotency (~20 min)

1. Review the retry task skeleton in `src/tasks_retry.py`
2. **User implements:** Task with `autoretry_for` + `retry_backoff` -- exponential backoff on transient errors
3. **User implements:** Idempotent task with Redis-based dedup key -- skip execution if already processed
4. Run `ex03_retry_idempotency.py` to trigger failures and observe retry behavior in Flower
5. Key question: Why must tasks with `acks_late=True` be idempotent? What could go wrong if they aren't?

### Phase 5: Task Routing (~10 min)

1. Review the routing configuration in `src/celery_app.py` and `src/tasks_routing.py`
2. **User implements:** Route tasks to specific queues (`high_priority`, `low_priority`)
3. **User implements:** Configure `task_routes` mapping in the Celery config
4. Run `ex04_task_routing.py` and observe which workers consume which tasks
5. Key question: How would you implement task priority within a single queue using Redis?

### Phase 6: Periodic Tasks with Beat (~10 min)

1. Review the Beat schedule configuration in `src/celery_app.py`
2. **User implements:** A periodic cleanup task that runs every 30 seconds
3. **User implements:** A crontab-scheduled task that runs at specific times
4. Observe Celery Beat dispatching tasks automatically in Flower
5. Key question: What happens if Beat dispatches a periodic task but no worker is available? Does it queue up?

### Phase 7: ARQ Comparison (~15 min)

1. Review the ARQ worker skeleton in `src/arq_tasks.py`
2. **User implements:** The same `add` and `fetch_url_content` tasks using ARQ's async API
3. **User implements:** ARQ retry configuration and cron scheduling
4. Run `ex05_arq_comparison.py` to dispatch ARQ tasks and compare with Celery
5. Key question: When would ARQ be a better choice than Celery? What features would you lose?

### Phase 8: FastAPI Integration & End-to-End (~15 min)

1. Review the FastAPI app in `src/api.py`
2. **User implements:** `POST /tasks/celery` endpoint -- dispatches a Celery task and returns task ID
3. **User implements:** `GET /tasks/celery/{task_id}` endpoint -- polls Celery result by task ID
4. **User implements:** `POST /tasks/arq` endpoint -- dispatches an ARQ task
5. Run the full stack via Docker Compose and test with curl
6. Open Flower to see Celery tasks flowing through the system
7. Key question: How would you implement webhooks or WebSocket notifications instead of polling for task results?

## Motivation

- **Ubiquitous pattern**: Task queues are in virtually every production web application -- email sending, report generation, image processing, data pipelines
- **Interview essential**: "How would you handle long-running operations?" is a standard system design question; Celery/Redis is the canonical Python answer
- **Production relevance**: Celery is used at Instagram (billions of tasks/day), Mozilla, Stripe, and most Django/FastAPI deployments
- **Async ecosystem**: ARQ bridges the gap between Celery's synchronous model and modern async Python (FastAPI, httpx, asyncpg)
- **Complements practice 005 (Docker Compose)**: Multi-service orchestration with workers, schedulers, and monitoring
- **Foundation for practice 076 (Redis)**: Deep use of Redis as broker, backend, and dedup store

## References

- [Celery Official Documentation](https://docs.celeryq.dev/en/stable/)
- [Celery Canvas: Designing Workflows](https://docs.celeryq.dev/en/stable/userguide/canvas.html)
- [Celery Task Routing](https://docs.celeryq.dev/en/stable/userguide/routing.html)
- [Celery Periodic Tasks](https://docs.celeryq.dev/en/stable/userguide/periodic-tasks.html)
- [Celery Concurrency Models](https://docs.celeryq.dev/en/latest/userguide/concurrency/index.html)
- [Celery Worker Pools Explained](https://celery.school/celery-worker-pools)
- [Celery Configuration Reference](https://docs.celeryq.dev/en/stable/userguide/configuration.html)
- [Flower Monitoring Documentation](https://flower.readthedocs.io/)
- [ARQ Documentation](https://arq-docs.helpmanual.io/)
- [ARQ GitHub Repository](https://github.com/python-arq/arq)
- [Celery vs ARQ Comparison](https://leapcell.io/blog/celery-versus-arq-choosing-the-right-task-queue-for-python-applications)
- [Advanced Celery: Idempotency, Retries & Error Handling](https://www.vintasoftware.com/blog/celery-wild-tips-and-tricks-run-async-tasks-real-world)
- [Celery Task Resilience: Advanced Strategies](https://blog.gitguardian.com/celery-tasks-retries-errors/)

## Commands

All commands are run from the `practice_079_distributed_task_queues/` folder root.

### Phase 1: Infrastructure Setup

| Command | Description |
|---------|-------------|
| `docker compose up -d` | Start Redis, Celery workers, Celery Beat, Flower, ARQ worker, and FastAPI app |
| `docker compose up -d redis` | Start only Redis (for local development without Docker workers) |
| `docker compose ps` | Check status of all containers |
| `docker compose logs redis` | View Redis container logs |
| `docker compose logs flower` | View Flower monitoring logs |

### Phase 1: Local Development Setup (alternative to Docker workers)

| Command | Description |
|---------|-------------|
| `uv sync` | Install all Python dependencies |
| `uv run celery -A src.celery_app worker --loglevel=info --pool=solo -Q default,high_priority,low_priority` | Start Celery worker locally (solo pool for Windows compat) consuming all queues |
| `uv run celery -A src.celery_app beat --loglevel=info` | Start Celery Beat scheduler locally |
| `uv run celery -A src.celery_app flower --port=5555` | Start Flower monitoring locally on port 5555 |
| `uv run arq src.arq_tasks.WorkerSettings` | Start ARQ worker locally |
| `uv run uvicorn src.api:app --host 0.0.0.0 --port 8000` | Start FastAPI app locally |

### Phase 2-6: Running Exercises

| Command | Description |
|---------|-------------|
| `uv run python ex01_basic_tasks.py` | Exercise 1: Basic Celery task dispatch and result retrieval |
| `uv run python ex02_canvas_workflows.py` | Exercise 2: Canvas workflows -- chain, group, chord |
| `uv run python ex03_retry_idempotency.py` | Exercise 3: Retry policies and idempotent task patterns |
| `uv run python ex04_task_routing.py` | Exercise 4: Task routing to dedicated queues |
| `uv run python ex05_arq_comparison.py` | Exercise 5: ARQ async tasks comparison with Celery |

### Phase 8: FastAPI Integration Testing

| Command | Description |
|---------|-------------|
| `curl -X POST http://localhost:8000/tasks/celery -H "Content-Type: application/json" -d "{\"x\": 4, \"y\": 5}"` | Dispatch a Celery add task via HTTP |
| `curl http://localhost:8000/tasks/celery/{task_id}` | Poll Celery task result by ID |
| `curl -X POST http://localhost:8000/tasks/arq -H "Content-Type: application/json" -d "{\"x\": 4, \"y\": 5}"` | Dispatch an ARQ add task via HTTP |
| `curl http://localhost:8000/tasks/arq/{job_id}` | Poll ARQ job result by ID |
| `curl http://localhost:8000/health` | Health check endpoint |

### Monitoring & Inspection

| Command | Description |
|---------|-------------|
| Open `http://localhost:5555` in browser | Flower dashboard -- workers, tasks, queues, graphs |
| `docker compose exec redis redis-cli` | Open interactive Redis CLI |
| `docker compose exec redis redis-cli KEYS "celery*"` | List all Celery-related keys in Redis |
| `docker compose exec redis redis-cli LLEN celery` | Check default Celery queue length |
| `docker compose exec redis redis-cli LLEN high_priority` | Check high_priority queue length |
| `uv run celery -A src.celery_app inspect active` | List active tasks across all workers |
| `uv run celery -A src.celery_app inspect reserved` | List reserved (prefetched) tasks |
| `uv run celery -A src.celery_app inspect stats` | Worker statistics (pool size, uptime, etc.) |

### Cleanup

| Command | Description |
|---------|-------------|
| `docker compose down` | Stop and remove all containers |
| `docker compose down -v` | Stop containers and remove Redis data volume |
| `python clean.py` | Full cleanup: Docker volumes, caches, virtual environments |

## State

`not-started`
