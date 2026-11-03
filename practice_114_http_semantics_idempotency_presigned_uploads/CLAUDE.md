# Practice 114: HTTP Semantics, Idempotency Keys, Conditional Caching & Presigned Uploads

## Technologies
- React 19
- TypeScript
- Vite
- FastAPI
- Boto3 / MinIO SDK
- Docker Compose (MinIO S3-compatible storage)

## Stack
- Frontend: TypeScript, React 19, Vite
- Backend: Python 3.12+ (FastAPI, Uvicorn, Boto3, Pydantic)
- Storage & Infrastructure: Docker Compose, MinIO S3 Object Storage

---

## Theoretical Context

Building robust, high-throughput distributed web systems requires deep alignment with the foundational semantics of the Hypertext Transfer Protocol (RFC 9110). A naive full-stack architecture treats every network interaction as a simple JSON `fetch`, proxying large multi-megabyte payloads through application servers and treating network operations as either 100% successful or atomic failures. In production, real networks are fundamentally unreliable, and treating HTTP as an opaque transport creates data corruption, duplicate resource creation, and severe infrastructure bottlenecks.

```
       +-------------------------------------------------------------+
       |               RESILIENT INGESTION ARCHITECTURE              |
       +-------------------------------------------------------------+

      [Phase 1: Presign]           POST /api/uploads/presign
     React Client -------------> FastAPI Backend (Port 8114)
          | <------------------- Returns S3 Presigned PUT URL + Key
          |
          | [Phase 2: Direct Binary Stream via XMLHttpRequest]
          | PUT https://minio:9000/datasets/blob.csv (Live Progress % + Speed)
          +-------------------------------------------------> MinIO S3 Storage
          <------------------------------------------------- Returns 200 OK + ETag
          |
          | [Phase 3: Idempotent Commit]
          | POST /api/datasets/commit (Header: Idempotency-Key: <UUID>)
          +----------------------------> FastAPI Backend
          |                              |-- Verifies S3 Object & ETag
          |                              |-- Acquires Idempotency Lock
          |                              \-- Saves to DB & Replays on Retry
          <----------------------------- Returns 201 Created
```

### 1. Idempotency in HTTP & Full-Stack Retries

#### Safe vs. Idempotent Methods (RFC 9110)
- **Safe methods (`GET`, `HEAD`, `OPTIONS`)**: Do not alter server state. Calling them $N$ times produces zero side-effects.
- **Idempotent methods (`PUT`, `DELETE`, `GET`)**: The intended effect of multiple identical requests on the server is the same as for a single request. `PUT /users/42` updates the entity to state $S$; executing it five times leaves the entity in state $S$.
- **Non-idempotent methods (`POST`, `PATCH`)**: Each execution is permitted to append or transition state. Calling `POST /orders` five times creates five distinct credit card charges or five database records.

#### The Distributed Network Failure Problem (Fallacy of Network Reliability)
Consider a client sending `POST /api/datasets/commit`:
1. Client initiates TCP handshake and sends HTTP POST.
2. Server executes business logic, allocates resources, writes record to the database, and commits transaction.
3. Server transmits `HTTP/1.1 201 Created`.
4. **Network Drop**: A mobile switch, router timeout, or transient TCP RST packet occurs before the client receives the response packet.
5. From the client's perspective, the request timed out or threw `TypeError: Failed to fetch`.
6. Without idempotency protection, an automated retry or user re-click issues a second `POST`, producing duplicate dataset ingestion jobs, duplicate billing transactions, or split-brain inconsistencies.

```
 Client                                                   Server
   |                                                        |
   | --- POST /api/datasets/commit -----------------------> | (Executes DB commit)
   |                                                        | (Job #42 created)
   | <XX [TCP RST / Connection Reset Before 201 Reaches] XX |
   |
[Client Timeout / Failure]
   |
   | --- Retry 1: POST /api/datasets/commit ---------------> | (WITHOUT IDEMPOTENCY:
   |                                                        |  Creates Job #43! DUPLICATE!)
```

#### The `Idempotency-Key` Protocol (IETF Draft / Stripe Pattern)
To make non-idempotent endpoints safe for retries, the system introduces a full-stack contractual key protocol:

1. **Client Lifecycle**:
   - The React client generates a cryptographically random UUID v4 when the user initiates a logical mutation action (e.g., clicking "Ingest Dataset").
   - The client stores this `Idempotency-Key` in its active mutation lifecycle state.
   - If the request fails due to a network drop, timeout (504), or transient gateway error (502/503), the client **must reuse the exact same `Idempotency-Key`** during automatic exponential-backoff retries.
   - The client **only clears or rotates** the key upon receiving a terminal response (2xx Success or 4xx Client Validation Error where the payload must be modified).

2. **Backend Atomic Idempotency Lifecycle & Fingerprinting**:
   - The backend exposes an atomic storage mechanism (in-memory lock map or Redis).
   - Upon receiving a request with header `Idempotency-Key: <UUID>`:
     - **Compute Payload Fingerprint**: The backend calculates a cryptographic SHA-256 hash of the canonicalized request method, URI, and JSON body.
     - **Case A (New Key)**: The backend atomically acquires an idempotency lock with status `"started"` and saves the payload fingerprint. It then processes the business logic. Once completed, it stores the HTTP response status code, headers, and body, transitioning status to `"completed"`.
     - **Case B (In-Flight Request)**: If the key already exists with status `"started"`, another worker or concurrent retry is actively executing. The server responds with `409 Conflict` (or `425 Too Early`), instructing the client to wait.
     - **Case C (Completed Request - Payload Match)**: If the key exists with status `"completed"` and the payload SHA-256 matches, the server **bypasses business logic completely** and replays the cached HTTP response (`201 Created`) with an added header `Idempotent-Replay: true`.
     - **Case D (Payload Mismatch / Tampering)**: If the key exists but the payload SHA-256 differs, a client bug or malicious actor reused a key for an altered payload. The server rejects the request with `422 Unprocessable Entity` (`"Idempotency-Key reused with modified payload"`).

---

### 2. HTTP Conditional Requests & Caching (`ETag`, `If-None-Match`, `If-Match`)

#### Strong vs. Weak Entity Tags (ETags)
An `ETag` is an opaque validator token assigned by a web server to a specific version of a resource:
- **Strong ETag (`"33a64df551425fcc"`)**: Guarantees byte-for-byte identity of the representation. If the ETag matches, the binary content is guaranteed identical.
- **Weak ETag (`W/"0815"` - RFC 9110 §8.8.1)**: Indicates semantic equivalence. The resources represent the same logical entity even if whitespace, ordering, or internal formatting differs.

#### Bandwidth Savings via `If-None-Match` (`304 Not Modified`)
In high-frequency dashboard queries:
1. Client requests `GET /api/datasets/123`. Server returns `200 OK` with `ETag: "v1-c7a8b0"` and a 5MB JSON metadata body.
2. React client stores both the data and the ETag.
3. On background polling or re-focus, the client issues `GET /api/datasets/123` with header `If-None-Match: "v1-c7a8b0"`.
4. Server computes the current resource ETag. Because it matches, the server returns `304 Not Modified` with an **empty body**.
5. Result: Network latency drops to single-digit milliseconds, zero CPU spent serializing JSON, and massive bandwidth savings.

#### Preventing the Lost Update Problem via `If-Match` (`412 Precondition Failed`)
In collaborative or distributed environments, concurrent mutations cause silent data loss:
1. Alice reads Dataset 101 at version `"v1"`.
2. Bob reads Dataset 101 at version `"v1"`.
3. Alice modifies the dataset tags and saves: `PUT /api/datasets/101`. Her change updates the entity to version `"v2"`.
4. Bob modifies the dataset description and saves: `PUT /api/datasets/101`. Bob's save overwrites Alice's tags because Bob was unaware of Alice's mutation (the *Lost Update Problem*).
5. **The HTTP Solution**: Bob's client sends `If-Match: "v1"`. The server checks the current ETag (`"v2"`). Because `"v1" != "v2"`, the server immediately halts execution and returns `412 Precondition Failed`. Bob's client is notified to fetch fresh state and resolve differences.

---

### 3. The Direct-to-Object-Storage Pattern (Presigned URLs)

#### Why Server-Proxied Uploads Exhaust Worker Threads
In naive web architectures, files are uploaded via `multipart/form-data` directly to an application server (e.g., FastAPI, Express, or Django):
- **Memory Spikes**: Buffering 500MB video or dataset files in server RAM causes Python process memory to balloon.
- **Worker Starvation**: Web workers (e.g., Uvicorn/Gunicorn async event loops or thread pools) stay bound to slow client connections (e.g., mobile 4G uploading at 200 KB/s for 15 minutes), exhausting worker concurrency.
- **Double Bandwidth**: The payload traverses the network twice: Client $\rightarrow$ FastAPI $\rightarrow$ S3.

#### The 3-Phase Direct Upload Flow
Modern cloud architectures decouple metadata negotiation from binary payload ingestion:

1. **Phase 1: Presigned URL Negotiation (`POST /api/uploads/presign`)**:
   - The React client sends file metadata (name, content type, size) to FastAPI.
   - FastAPI authenticates the user, validates permissions, checks quota, and calls S3/MinIO (`generate_presigned_url('put_object')`) to create a cryptographically signed URL valid for a strict window (e.g., 15 minutes).
   - FastAPI returns the signed URL and generated object key.

2. **Phase 2: Direct Binary Streaming via `XMLHttpRequest`**:
   - The browser streams the binary `File` directly to MinIO/S3 using HTTP `PUT`.
   - **Why `XMLHttpRequest` over `fetch`?** The modern Fetch API (`window.fetch`) does not expose standard, cross-browser upload progress events on standard request streams. `XMLHttpRequest` provides the `xhr.upload.onprogress` event, enabling accurate byte calculation (`loaded` / `total`), instantaneous upload speed (KB/s), time-remaining estimates, and instant cancellation via `xhr.abort()`.
   - MinIO receives the payload, verifies the S3 signature, stores the object, and returns `200 OK` with an `ETag` response header (the S3 MD5 checksum of the uploaded binary).

3. **Phase 3: Idempotent Commit (`POST /api/datasets/commit`)**:
   - React sends a final commit request to FastAPI containing the object key, file metadata, and the S3-returned `ETag`.
   - React attaches an `Idempotency-Key` header.
   - FastAPI calls S3 `HeadObject` to cryptographically verify that the object exists in MinIO, matches the client-reported size and `ETag`, and commits the record to the application database.
   - If the network drops during commit, the frontend retry uses the same `Idempotency-Key` and safely recovers without re-uploading or duplicating records!

---

## Description

In this practice, you will build a **Resilient Dataset Ingestion Portal** featuring:
1. A **FastAPI Backend (port 8114)** with custom HTTP middleware/handlers for full-stack idempotency keys, conditional caching (`ETag`, `304 Not Modified`, `If-Match` 412 optimistic locking), presigned S3 generation, and a configurable network chaos simulator.
2. A **MinIO S3 Storage Container (Docker Compose)** configured with pre-provisioned buckets and CORS headers exposing `ETag` to web clients.
3. A **React 19 Frontend (Vite, port 5173)** featuring custom hooks:
   - `useIdempotentMutation`: Manages deterministic UUID key persistence across network retries and resets on terminal completion.
   - `usePresignedUpload`: Implements an `XMLHttpRequest` upload engine with real-time percentage, transfer speed, and abort controls.
   - `conditionalFetch`: Implements caching with `If-None-Match` and concurrency checks with `If-Match`.
   - `IngestionPipeline`: An interactive dashboard with simulated network drop switches, concurrency conflict simulators, and live HTTP protocol telemetry.

---

## Instructions

### Step 1: Start MinIO Object Storage
Launch the local MinIO container and initialization service via Docker Compose:
```bash
docker compose up -d
```
Verify that MinIO API is accessible at `http://localhost:9000` and the web console at `http://localhost:9001` (user: `minioadmin`, pass: `minioadmin`).

### Step 2: Set Up Backend Environment
Navigate to `backend/`, create a virtual environment, install dependencies, and launch FastAPI:
```bash
cd backend
uv venv
# On Windows: .venv\Scripts\activate | On Unix: source .venv/bin/activate
uv pip install fastapi uvicorn boto3 pydantic
uvicorn main:app --reload --port 8114
```

### Step 3: Set Up Frontend Environment
Open a second terminal, navigate to `frontend/`, install npm dependencies, and start the Vite dev server:
```bash
cd frontend
npm install
npm run dev
```
Open `http://localhost:5173` in your browser.

---

### Step 4: Exercises (`TODO(human)`)

#### Exercise 1: Full-Stack Idempotency-Key Protocol
- **Files**: `backend/idempotency.py` & `frontend/src/hooks/useIdempotentMutation.ts`
- **Context**: When networks drop after the server writes to the database, clients retry. Without idempotency handling, this duplicates records.
- **Task**:
  1. In `backend/idempotency.py`, complete `evaluate_idempotency(key, payload_hash)`. Check if key exists. If status is `"started"`, return in-flight conflict. If status is `"completed"`, verify `payload_hash` matches stored hash (reject if mismatched with `422`, replay cached response if matched).
  2. In `frontend/src/hooks/useIdempotentMutation.ts`, implement the key lifecycle: generate a UUID v4 for new user actions, retain the exact same key during retries when network or 5xx errors occur, and clear/rotate the key only on 2xx success or 4xx client validation errors.

#### Exercise 2: HTTP Conditional Requests (`ETag`, `If-None-Match`, `If-Match`)
- **Files**: `backend/main.py` & `frontend/src/api/conditionalFetch.ts`
- **Context**: Prevents unnecessary bandwidth consumption for unchanged resources (304) and prevents the Lost Update Problem during concurrent edits (412).
- **Task**:
  1. In `backend/main.py`:
     - In `GET /api/datasets/{dataset_id}`: Check the incoming `If-None-Match` header. If it matches the dataset's current computed ETag, return HTTP `304 Not Modified` with empty body.
     - In `PUT /api/datasets/{dataset_id}`: Check incoming `If-Match` header. If missing or different from current ETag, raise `HTTPException(status_code=412, detail="Precondition Failed: Resource modified")`.
  2. In `frontend/src/api/conditionalFetch.ts`:
     - Implement `fetchWithETag`: pass `If-None-Match: cachedETag`. If status is 304, reuse cached data; if 200, update cache and stored ETag.
     - Implement `updateWithIfMatch`: pass `If-Match: currentETag`. Handle 412 status by catching and reporting concurrency conflicts.

#### Exercise 3: Presigned S3 Generation & Cancelable XHR Uploader
- **Files**: `backend/storage.py` & `frontend/src/hooks/usePresignedUpload.ts`
- **Context**: Large binary streams must bypass the application server. Browsers need upload progress reporting and abort capabilities.
- **Task**:
  1. In `backend/storage.py`, implement `generate_presigned_put_url(s3_key, content_type, expires_in)` using `boto3` client `generate_presigned_url('put_object', ...)`.
  2. In `frontend/src/hooks/usePresignedUpload.ts`, implement `uploadFile(uploadUrl, file)` using `XMLHttpRequest`. Attach listeners to `xhr.upload.onprogress` to calculate real-time percentage and transfer rate. Extract `ETag` from `xhr.getResponseHeader("ETag")` upon 200 completion. Implement `abort()` to cleanly terminate the stream.

#### Exercise 4: End-to-End Resilient Ingestion & Chaos Testing
- **File**: `frontend/src/components/IngestionPipeline.tsx`
- **Context**: Combine all 3 phases into a bulletproof ingestion workflow that survives deliberate network drops.
- **Task**: Wire the 3-phase flow:
  1. Request Presigned URL from FastAPI.
  2. Upload directly to MinIO via `usePresignedUpload`.
  3. Commit metadata + ETag via `useIdempotentMutation`.
  4. Test with the UI Chaos toggle ("Simulate Post-Commit Network Drop"): observe the server commit the dataset, drop the response, and see the frontend retry with the same `Idempotency-Key` and successfully receive the replayed `201 Created` without duplicating records!

---

## Motivation

In modern engineering architectures, distributed systems fail constantly. Cloud networks drop packets, users browse on spotty mobile connections, and object storage dwarfs traditional database capacities. Engineers who rely solely on default `POST` operations create silent data duplication, duplicate payment charges, and race conditions. Furthermore, streaming 500MB+ datasets through application worker threads degrades backend scalability. Mastering HTTP semantics—idempotency tokens, conditional validation headers, and presigned direct-to-S3 pipelines—is the hallmark of a senior full-stack engineer capable of building resilient, cost-efficient, production-grade cloud systems.

---

## Commands

| Phase | Command | Description |
|---|---|---|
| Infrastructure | `docker compose up -d` | Launch MinIO S3 container and init container with CORS rules |
| Infrastructure | `docker compose ps` | Verify MinIO service health and bucket initialization |
| Infrastructure | `docker compose down -v` | Stop MinIO and delete all storage volumes |
| Backend | `cd backend && uv venv` | Create Python 3.12+ virtual environment |
| Backend | `cd backend && uv pip install fastapi uvicorn boto3 pydantic` | Install FastAPI backend dependencies |
| Backend | `cd backend && uvicorn main:app --reload --port 8114` | Start FastAPI server on port 8114 with hot reload |
| Frontend | `cd frontend && npm install` | Install React 19, TypeScript, and Vite dependencies |
| Frontend | `cd frontend && npm run dev` | Start Vite development server on `http://localhost:5173` |
| Frontend | `cd frontend && npm run build` | Validate TypeScript types and compile production bundle |
| Cleanup | `python clean.py` | Standalone cleanup of Docker volumes, node_modules, caches, and venvs |

---

## Notes
*(Add personal observations, architectural reflections, and insights here as you work through the exercises.)*
