# Practice 118: Performance Engineering — Virtualization, Concurrent Transitions, Web Workers & Network-Boundary Testing (MSW)

## Technologies

- **React 19 (`useTransition`, `useDeferredValue`)** — concurrent primitives for interruptible render-bound updates and deferred priority state synchronization.
- **TypeScript & Vite** — strict typing for Web Worker message protocols, mock definitions, and fast Hot Module Replacement (HMR).
- **TanStack Virtual (`@tanstack/react-virtual`)** — headless, DOM-virtualization engine maintaining constant DOM tree size regardless of dataset scale.
- **Web Workers** — browser multi-threading API for offloading CPU-intensive multi-field regex filtering and anomaly scoring off the main UI thread.
- **Vitest & React Testing Library** — fast, jsdom-based unit and integration test runner adhering to user-centric interaction paradigms.
- **Mock Service Worker (MSW v2)** — declarative network-level mocking using Service Worker and Node interceptors (`http`, `HttpResponse`, `delay`).

## Stack

TypeScript (Frontend + Web Worker + Vitest Integration Test Suite).

## Theoretical Context

### The Browser Main-Thread Budget (16.6ms at 60Hz) & DOM Virtualization

Modern web browsers target a $60\text{ Hz}$ refresh rate (a frame budget of $16.6\text{ ms}$) or $120\text{ Hz}$ ($8.3\text{ ms}$). Every frame must accommodate:
1. JavaScript execution (event handlers, state reconciliations, scheduler turns),
2. Style recalculation,
3. Layout (geometry computation, reflow),
4. Painting (rasterization of vector primitives into pixels), and
5. Compositing (layer GPU upload and assembly).

When an application mounts $50,000$ table rows into the DOM, the browser's layout engine must construct and traverse a tree of over $250,000$ DOM elements (cells, text nodes, wrappers). Even if React memoization (`React.memo`, `useMemo`) prevents component re-renders, the native browser engine still experiences catastrophic layout thrashing and massive V8 heap bloat whenever the user scrolls or resizes the viewport.

**DOM Virtualization (Windowing)** solves this fundamental limitation by decoupling dataset size from DOM node count:
- The virtualizer monitors the scroll container geometry (scroll offset $S$, container height $H$).
- It computes the subset of visible row indices: $[\lfloor S / h_{\text{row}} \rfloor - \text{overscan}, \lceil (S + H) / h_{\text{row}} \rceil + \text{overscan}]$.
- It maintains a phantom scrollbar using a spacer container sized to $\text{totalItems} \times h_{\text{row}}$.
- Only the currently visible elements ($\sim 20\text{--}30$ DOM nodes) are rendered. Rows are absolutely positioned using GPU-accelerated CSS `transform: translateY(...)` rather than layout-triggering properties like `top` or `margin-top`.
- As the user scrolls through $50,000$ items, memory usage and layout calculations remain $O(1)$ relative to the dataset length.

### CPU-Bound Work vs Render-Bound Work

A common misconception in React performance tuning is conflating **render-bound work** with **CPU-bound work**:

1. **Render-Bound Work (Concurrent React: `useTransition` / `useDeferredValue`):**
   - Occurs when React must reconcile and commit a large, deeply nested virtual DOM tree.
   - `useTransition` and `useDeferredValue` instruct React's fiber reconciler to mark these updates as non-urgent. If high-priority input arrives (e.g. typing into a search input), React yields control back to the browser's main thread and pauses rendering the deferred tree.
   - *Crucial caveat:* Concurrent React does **not** run code on a background thread. All JavaScript evaluation remains on the browser's single main thread.

2. **CPU-Bound Work (Web Workers):**
   - If an event handler executes a synchronous $300\text{ ms}$ algorithmic loop (such as regex pattern matching, Levenshtein distance scoring, or anomaly weighting across $50,000$ records), `useTransition` cannot save the application. The single thread is completely starved; user keystrokes, CSS animations, and scroll events freeze until the loop terminates.
   - Offloading CPU-bound tasks to a dedicated **Web Worker** executes the computation in an independent OS thread.
   - Communication occurs asynchronously via structured cloning (`postMessage`).
   - By combining Web Workers for heavy filtering with `useDeferredValue` and virtualization for row rendering, the main thread remains permanently at $60\text{ FPS}$.

### Why Mocking `fetch` or API Modules Lies in Tests & How MSW Works

Traditional frontend testing frequently relies on function-level mocking:
```ts
vi.mock('./apiClient', () => ({ fetchLogs: vi.fn().mockResolvedValue([...]) }));
// or
global.fetch = vi.fn().mockResolvedValue({ json: () => [...] });
```

This pattern creates brittle, untrustworthy tests:
- **It bypasses the network boundary:** URL parsing, query string serialization, request headers (e.g., `Authorization`, `Content-Type`), and body formatting are completely bypassed and untested.
- **It masks transport-level bugs:** It fails to detect when an endpoint expects query parameters formatted as `?status=WARN&status=ERROR` instead of `?status=WARN,ERROR`.
- **It ignores browser runtime semantics:** `AbortController` cancellation, `ReadableStream` error bubbling, network timeouts, and HTTP status code handling (`429 Too Many Requests`, `503 Service Unavailable`, `Retry-After` headers) are simulated inaccurately or forgotten entirely.

**Mock Service Worker (MSW v2)** solves this by operating at the lowest boundary of the network stack:
- In Node/Vitest environments, MSW intercepts native socket/http requests via Node's `undici`/`http` interceptors.
- In browser environments, MSW intercepts requests using the native Service Worker API.
- Your application code runs unmodified: React Component $\rightarrow$ Hook / TanStack Query $\rightarrow$ native `fetch` $\rightarrow$ serialization $\rightarrow$ MSW Handler.
- MSW tests exercise the entire application layer, allowing deterministic verification of retry backoff, race conditions, stale response discards, and HTTP error boundaries.

### Key Concepts

| Concept | Definition |
|---|---|
| **DOM Virtualization** | Windowing technique that renders only items within the visible viewport, maintaining $O(\text{viewport})$ DOM nodes. |
| **Overscan** | Buffer of unrendered items rendered just outside viewport edges to prevent blank flashes during rapid scrolling. |
| **Render-Bound Work** | React fiber tree reconciliation overhead, mitigated by `useTransition` and `useDeferredValue`. |
| **CPU-Bound Work** | Synchronous JavaScript computation blocking the main event loop, requiring Web Worker offloading. |
| **MSW v2** | Declarative API mocking library intercepting network requests at the transport boundary without touching `fetch` bindings. |
| **Request Race Condition** | In-flight asynchronous requests resolving out of order, leading to stale state overwrites if unmitigated. |

### Where This Fits

In modern high-performance web applications (e.g. data observability dashboards, supply chain execution grids at AutoScheduler.AI, and real-time telemetry viewers), rendering large datasets without UI lag is mandatory. Virtualization and Web Workers prevent frontend execution bottlenecks, while MSW provides production-grade network boundary testing without requiring a live backend daemon during CI.

### References

- TanStack Virtual Documentation: <https://tanstack.com/virtual/latest>
- React 19 Concurrent Primitives (`useDeferredValue`, `useTransition`): <https://react.dev/reference/react>
- Web Workers API (MDN): <https://developer.mozilla.org/en-US/docs/Web/API/Web_Workers_API>
- Mock Service Worker (MSW v2) Docs: <https://mswjs.io/docs/>
- Testing Library Guiding Principles: <https://testing-library.com/docs/guiding-principles>

---

## Description

Engineer a high-volume **50,000-row Log & Schedule Inspector** in React 19 and TypeScript:
1. Procedurally generate 50,000 realistic telemetry log records with anomaly scores, severity levels, timestamps, and service tags.
2. Offload heavy multi-field filtering, regex evaluation, and anomaly scoring to a dedicated **Web Worker** with request ID cancellation.
3. Render the filtered dataset at 60 FPS using **`@tanstack/react-virtual`** (`useVirtualizer`) and **`useDeferredValue`**.
4. Construct a network-boundary integration test suite using **Vitest**, **React Testing Library**, and **MSW v2** verifying wire deserialization, `429`/`500` error recovery, and race-condition cancellation under rapid user input.

### What you'll learn

1. How to identify main-thread layout bottlenecks and measure DOM node rendering overhead.
2. How to implement viewport windowing with `@tanstack/react-virtual` using absolute translateY offsets and overscan buffers.
3. How to decouple heavy CPU filtering from React's render loop using Web Workers and request cancellation IDs.
4. How to configure MSW v2 to intercept requests at the transport layer, asserting real `fetch` calls, query params, and status codes.
5. How to write deterministic tests for race conditions when asynchronous network responses arrive out of order.

---

## Instructions

### Phase 0: Setup

1. Navigate to the frontend directory: `cd frontend`
2. Install dependencies: `npm install`
3. Launch development server: `npm run dev`
4. Run the test suite: `npm test`

---

### Phase 1: CPU Offloading via Web Worker (~25 min)
**Files:** `frontend/src/workers/datasetWorker.ts` & `frontend/src/hooks/useWorkerFilter.ts`

1. **TODO #1 — Web Worker Execution (`datasetWorker.ts`)**:
   Implement message handling for incoming filter requests. Perform case-insensitive regex/substring filtering and anomaly score filtering across 50,000 records. Calculate execution time in milliseconds and post the structured result back with the matching `requestId`.

2. **TODO #2 — Worker Hook & Stale Reply Invalidation (`useWorkerFilter.ts`)**:
   Manage worker lifecycle and request tracking. Dispatch messages on filter criteria changes and ensure responses with outdated `requestId` values are ignored so fast keystrokes never produce stale UI states.

---

### Phase 2: High-Performance Virtual Grid (~25 min)
**File:** `frontend/src/components/VirtualizedLogGrid.tsx`

3. **TODO #3 — Virtualizer Integration (`VirtualizedLogGrid.tsx`)**:
   Configure `@tanstack/react-virtual`'s `useVirtualizer` with `count`, `getScrollElement`, `estimateSize: () => 40`, and `overscan: 10`.
   Map the virtual items inside an absolute translateY positioned container inside a phantom scroll spacer. Add a toggle to contrast performance against naive unvirtualized rendering.

---

### Phase 3: MSW v2 Network Boundary Testing (~30 min)
**Files:** `frontend/src/mocks/handlers.ts` & `frontend/src/__tests__/LogInspector.integration.test.tsx`

4. **TODO #4 — MSW v2 Request Interceptors (`handlers.ts`)**:
   Implement MSW v2 `http.get('/api/logs', ...)` handling query parameters (`partition`, `minSeverity`), simulated delays, `500 Internal Server Error`, and `429 Too Many Requests` responses with `Retry-After` headers.

5. **TODO #5 — Real-Wire Integration Tests (`LogInspector.integration.test.tsx`)**:
   Write integration tests without mocking `fetch` or React hooks:
   - Verify logs load over the wire and render into the virtualized grid.
   - Verify query-parameter filtering over HTTP.
   - Verify error boundary banner display and retry recovery when encountering status 500.
   - Verify rate-limiting notice on status 429.

---

### Phase 4: Network Race Conditions & Stale State Prevention (~20 min)
**File:** `frontend/src/__tests__/raceCondition.integration.test.tsx`

6. **TODO #6 — Deterministic Out-of-Order Network Test (`raceCondition.integration.test.tsx`)**:
   Simulate rapid partition switching:
   - Request 1 (Partition A) has a slow network delay (e.g. 150ms).
   - Request 2 (Partition B) has an instantaneous network delay (e.g. 10ms).
   - Verify that Partition B's data renders and is NOT overwritten when Partition A's delayed response finally arrives.

---

## Motivation

- **AutoScheduler.AI Relevance:** Core warehouse optimization suites require inspecting millions of scheduler allocations, execution logs, and live dispatch events. Engineers must know how to render massive datasets smoothly without freezing the operator's browser.
- **Senior $\rightarrow$ Staff Differentiator:** Novice engineers mock everything down to the hook or function level, missing catastrophic transport-level bugs and race conditions. Senior engineers build robust virtualized interfaces and verify them with real-wire network mocks.
- **Zero-Friction CI Testing:** MSW allows comprehensive integration testing in headless CI environments without running heavy backend containers or flaky external services.

---

## Commands

| Phase | Command | Description |
|---|---|---|
| **Setup** | `npm install` | Install frontend dependencies in `frontend/`. |
| **Dev** | `npm run dev` | Launch Vite development server with HMR. |
| **Test** | `npm test` | Run Vitest integration and unit tests once. |
| **Test (Watch)** | `npm run test:watch` | Run Vitest in interactive watch mode. |
| **Typecheck** | `npm run typecheck` | Run TypeScript compiler verification without emitting code. |
| **Clean** | `python clean.py` | Clean build artifacts, node_modules, and cache files. |

---

## Notes

_(populated during the practice.)_

## State

`not-started`
