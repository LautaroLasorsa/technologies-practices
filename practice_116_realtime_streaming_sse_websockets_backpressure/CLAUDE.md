# Practice 116: Real-Time & Streaming UI — SSE, WebSockets, `useSyncExternalStore` & UI Backpressure

## Technologies
- **Frontend:** React 19, TypeScript, Vite, `useSyncExternalStore`, Web Streams API (`ReadableStream`, `TextDecoder`)
- **Backend:** Python 3.12+, FastAPI, Uvicorn, WebSockets, `asyncio`

## Stack
- TypeScript (Frontend Browser Application)
- Python 3.12+ (FastAPI High-Concurrency Backend on Port 8116)

---

## Theoretical Context

Building high-throughput real-time user interfaces requires bridging two fundamentally opposing paradigms:
1. **Network Ingestion Rate:** High-frequency, unpredictable packet arrival rates (e.g., 200–1000 messages/second from financial orderbooks, simulation engines, or rapid LLM token emission).
2. **UI Render Cadence:** Fixed hardware display refresh rates (typically 60 Hz or 120 Hz, meaning 16.6ms or 8.3ms per frame budget), governed by the browser main thread and React reconciliation loop.

Failing to architect this boundary results in thread starvation, dropped frames, micro-stutter, and interface unresponsiveness.

### 1. Protocol Trade-Offs: SSE vs. WebSockets vs. Long Polling

| Feature | Server-Sent Events (SSE) | WebSockets (WS) | Long Polling |
| :--- | :--- | :--- | :--- |
| **Directionality** | Unidirectional (Server $\to$ Client) | Full-Duplex Bidirectional | Half-Duplex (Repeated HTTP) |
| **Transport** | Standard HTTP/1.1 or HTTP/2 (`text/event-stream`) | Upgraded TCP frame connection (`ws://`, `wss://`) | Standard HTTP request/response |
| **Multiplexing** | Native with HTTP/2 (multiple streams over 1 TCP connection) | Single TCP connection per socket (unless framed over HTTP/2) | Multiple concurrent HTTP connections |
| **Proxy & Firewall Traversal** | Excellent; traverses corporate proxies, firewalls, and CDNs seamlessly | Can be blocked by restrictive proxies or stateful inspection firewalls | Good, but incurs massive request header overhead |
| **Connection Recovery** | Built-in browser reconnection & `Last-Event-ID` resumption | Application-level logic required (heartbeats, backoff, sequence tracking) | Inherent in request loop |
| **Payload Format** | UTF-8 plain text with `event:`, `data:`, `id:` framing | Binary (ArrayBuffer, Blob) or UTF-8 text frames | Any HTTP response body (JSON, binary) |
| **Primary Use Cases** | LLM token streaming, notification feeds, progress updates, audit trails | Collaborative canvas, live gaming, low-latency control, financial feeds | Legacy fallback when WS/SSE are blocked |

#### Deep-Dive: SSE and the Limitation of Native `EventSource`
Native browser `EventSource` (`new EventSource('/stream')`) provides native auto-reconnection and event parsing. However, the W3C `EventSource` specification has two severe limitations in production architectures:
1. **HTTP Method Restriction:** It supports only HTTP `GET`. LLM prompts, complex search queries, or solver parameter payloads often exceed URL length limits (around 2 KB) and require request bodies.
2. **Headers Restriction:** It cannot set custom request headers, preventing standard Bearer token authentication (`Authorization: Bearer <token>`) without falling back to insecure query parameters or cookies.

**The Solution:** Use standard `fetch()` with an `AbortSignal`, obtain a `ReadableStream` via `response.body.getReader()`, pipe chunks through `TextDecoder`, and parse the `text/event-stream` wire format manually. This gives full control over HTTP `POST`, arbitrary authentication headers, and instant client-side cancellation.

#### Deep-Dive: WebSockets and Transport Robustness
WebSockets operate over a raw TCP socket after an initial HTTP handshake. Because intermediaries (proxies, NAT gateways, load balancers) may silently drop idle TCP connections without sending a `FIN` or `RST` packet (a "black hole" disconnect), a production WebSocket client must implement:
- **Application Heartbeats:** Periodic ping messages (`ping`/`pong`). If no response is received within a threshold (e.g., 5 seconds), the client proactively terminates the zombie socket.
- **Decorrelated & Full-Jitter Exponential Backoff:** Reconnecting immediately or on a deterministic timer causes the "Thundering Herd" problem when a server or load balancer restarts. Full jitter randomizes delays:
  $$\text{delay} = \text{random}(0, \min(t_{\max}, t_{\text{base}} \times 2^{\text{attempt}}))$$
- **Outbound Command Queueing:** In-flight user interactions while offline should be buffered in an offline queue (with bounded eviction) and flushed upon re-establishment of the `OPEN` state.

---

### 2. Frontend Backpressure & The Render Loop Bottleneck

#### Why Naive `useState` Chokes Under High-Frequency Streams
Consider a WebSocket feed transmitting 250 solver telemetry updates per second. If a React component implements naive state updates:
```typescript
// ANTI-PATTERN: Direct state update on high-frequency stream
ws.onmessage = (event) => {
  const telemetry = JSON.parse(event.data);
  setTelemetry(prev => [...prev, telemetry]);
};
```
Each `setTelemetry` call notifies React that a state slice is dirty. React schedules a re-render. Even with automatic batching in modern React, incoming microtasks from network callbacks frequently force up to 100–250 reconciliation passes per second.
- The virtual DOM diffing and component execution monopolize the JavaScript single thread.
- Long tasks exceed the 16.6ms frame budget ($1000\text{ ms} / 60\text{ fps}$).
- The browser cannot perform style calculation, layout, or composite layers.
- **Result:** Animation frames drop to 5–10 FPS, input lag exceeds several hundred milliseconds, and memory balloons due to repeated immutable array allocations ($O(N)$ copies).

#### Decoupling Network Ingestion from Render Cadence
To maintain a locked 60 FPS while consuming thousands of messages per second, we decouple ingestion from rendering:
1. **Network Layer (High-Frequency Ingestion):** Incoming packets are pushed into an external, mutable, pre-allocated **Ring Buffer** (circular array) in $O(1)$ time without touching React or allocating new arrays.
2. **Scheduling Layer (Coalescing & Backpressure):** An animation frame scheduler (`requestAnimationFrame`) or throttled interval (e.g. 50ms) tracks whether a render notification is pending. Multiple network packets arriving within the same 16.6ms window are coalesced into a single flush notification.
3. **React Layer (`useSyncExternalStore`):** React subscribes to the external store. On scheduled flush, `getSnapshot()` produces an immutable snapshot. React renders exactly once per frame, unaffected by whether 5 or 500 packets arrived during that frame.

```
+-------------------+      250 msgs/sec       +------------------------+
|  WebSocket / SSE  | ----------------------> |  External Ring Buffer  |
|  Network Ingest   |                         |  (O(1) in-place push)  |
+-------------------+                         +------------------------+
                                                          |
                                           requestAnimationFrame (60 Hz)
                                           coalesced dirty trigger
                                                          v
                                              +-----------------------+
                                              | useSyncExternalStore  |
                                              | getSnapshot() (cached)|
                                              +-----------------------+
                                                          |
                                                    60 FPS locked
                                                          v
                                              +-----------------------+
                                              |   React UI Component  |
                                              +-----------------------+
```

#### Snapshot Identity & The `useSyncExternalStore` Contract
React's `useSyncExternalStore(subscribe, getSnapshot)` enforces strict referential stability:
- If `getSnapshot()` returns a new object or array reference on every call when state has not flushed, React assumes store volatility and falls into an **infinite re-render loop**.
- Therefore, the external store must cache its snapshot reference and only update the snapshot reference when a coalesced flush actually takes place.

---

## Description

In this practice, you will build a complete, high-performance **Live Optimization Solver & AI Reasoning Console**:
- A **FastAPI Backend (port 8116)** that:
  - Streams AI solver reasoning tokens via HTTP `POST` Server-Sent Events (`text/event-stream`), deliberately splitting chunks across TCP packets to challenge the client parser, and detecting client abort via `request.is_disconnected()`.
  - Streams 10 Hz – 500 Hz optimization telemetry metrics (iterations, objective values, duality gap, cooling temperature) over WebSockets with heartbeat ping/pong support.
- A **React 19 + TypeScript Frontend (Vite)** featuring:
  - A custom cancelable `useFetchSseStream` hook using the Web Streams API.
  - A zero-allocation circular `RingBuffer` external store hooked to React via `useSyncExternalStore`.
  - A resilient `useResilientWebSocket` hook with full-jitter backoff and heartbeat recovery.
  - A live side-by-side **FPS and performance meter** comparing naive React state updates vs. backpressured ring buffer rendering under 250–500 msg/s load.

---

## Instructions

### Step 1: Environment & Dependency Setup
1. **Backend:** In `backend/`, initialize the Python environment and install dependencies:
   ```bash
   cd backend
   uv sync  # or pip install -e .
   ```
2. **Frontend:** In `frontend/`, install npm dependencies:
   ```bash
   cd frontend
   npm install
   ```

### Step 2: Implement Exercise 1 — Cancelable Fetch SSE Stream Parser
- **File:** `frontend/src/streaming/useFetchSseStream.ts`
- **Context & Educational Value:** Native `EventSource` cannot send `POST` request bodies or custom headers. You will build a stream parser on top of `fetch()` and `ReadableStream`.
- **Key Mechanics:**
  - Consume chunks via `response.body.getReader()`.
  - Decode binary chunks into strings with `TextDecoder({ fatal: false, stream: true })`.
  - Handle TCP chunk fragmentation: a single SSE event or line may be split across multiple network read calls. Maintain an internal buffer, split complete lines by `\n`, and retain incomplete line fragments for the next chunk.
  - Parse `event:`, `data:`, `id:`, and dispatch typed events.
  - Hook an `AbortController` signal to cleanly cancel network transfer and backend processing.

### Step 3: Implement Exercise 2 — UI Backpressure via RingBuffer & `useSyncExternalStore`
- **File:** `frontend/src/streaming/ringBufferStore.ts`
- **Context & Educational Value:** Discover why high-frequency `useState` causes frame-rate collapse, and how to build a zero-allocation circular buffer external store.
- **Key Mechanics:**
  - Build `RingBuffer<T>` with fixed capacity, circular head/tail pointers, and $O(1)$ push.
  - Build `createRingBufferStore<T>(capacity, maxFlushIntervalMs)`:
    - Ingest telemetry in $O(1)$ without allocating intermediate arrays.
    - Coalesce re-render notifications using `requestAnimationFrame`. If a rAF tick is already scheduled, subsequent pushes during the current frame update the buffer in-place without queueing extra render tasks.
    - Provide referentially stable `getSnapshot()` complying with `useSyncExternalStore` rules.

### Step 4: Implement Exercise 3 — Resilient WebSocket Connection Manager
- **File:** `frontend/src/streaming/useResilientWebSocket.ts`
- **Context & Educational Value:** Real-world networks drop connections, restart servers, and silence idle sockets.
- **Key Mechanics:**
  - Implement connection state machine: `CONNECTING`, `OPEN`, `CLOSING`, `CLOSED`, `RECONNECTING`.
  - Implement full-jitter exponential backoff algorithm to prevent reconnect storms.
  - Implement heartbeat watchdog: send `PING` every 5 seconds; if no `PONG` returns within 3 seconds, forcefully close and reconnect.
  - Implement offline message queue: buffer outgoing steering commands when disconnected, and flush automatically on reconnection.

### Step 5: Implement Exercise 4 — FastAPI SSE with Client Disconnect & WS Broadcast
- **File:** `backend/main.py`
- **Context & Educational Value:** Prevent server-side compute leakage when clients abandon streams, and handle high-speed bidirectional WebSocket communication.
- **Key Mechanics:**
  - In `/api/solver/stream-reasoning`: monitor `await request.is_disconnected()`. If the client clicks "Abort", immediately terminate token generation and log cancellation.
  - In `/ws/telemetry`: read client frequency commands (`SET_RATE`, 10–500 Hz), answer heartbeats (`PING` $\to$ `PONG`), and stream telemetry frames at the requested cadence.

---

## Motivation

Modern AI applications and computational tools demand real-time interactivity:
- **LLM Token Streaming:** ChatGPT, Claude, and internal reasoning models emit tokens incrementally. Applications must handle partial stream parsing, cancellation, and markdown formatting without UI stutter.
- **High-Frequency Engineering Telemetry:** Live optimization solvers, financial execution engines, robotics logs, and simulation dashboards broadcast at 100–1000 Hz. Naive React state patterns break down completely under this volume.
- **Production Full-Stack Architecture:** Mastering the duality of `POST` SSE streams and resilient WebSockets equipped with `useSyncExternalStore` backpressure separates junior frontend developers from senior full-stack systems engineers.

---

## Commands

| Phase | Command | Description |
| :--- | :--- | :--- |
| **Setup (Backend)** | `cd backend && uv sync` | Install backend dependencies (`fastapi`, `uvicorn`, `websockets`) |
| **Setup (Frontend)** | `cd frontend && npm install` | Install frontend React & Vite dependencies |
| **Run Backend** | `cd backend && uvicorn main:app --port 8116 --reload` | Run FastAPI backend on port 8116 with live reload |
| **Run Frontend** | `cd frontend && npm run dev` | Launch Vite frontend development server (http://localhost:5173) |
| **Type Check** | `cd frontend && npx tsc --noEmit` | Validate TypeScript types and interfaces |
| **Build Frontend** | `cd frontend && npm run build` | Verify production build bundle creation |
| **Clean** | `python clean.py` | Clean virtual environments, caches, node_modules, and build artifacts |

---

## Notes
*(To be populated with personal observations, benchmarking metrics, and lessons learned during practice)*
