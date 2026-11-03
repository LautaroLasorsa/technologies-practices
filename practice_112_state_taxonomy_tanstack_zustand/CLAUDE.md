# Practice 112: State Taxonomy — Server Cache (TanStack Query) vs Client State (Zustand) vs URL State

## Technologies
- **Frontend:** React 19, TypeScript 5.7, Vite 6, `@tanstack/react-query` v5, `zustand` v5, `lucide-react`
- **Backend:** FastAPI, Uvicorn, Pydantic v2, Python 3.12+

## Stack
- TypeScript (Frontend SPA)
- Python 3.12+ (Backend REST API with Optimistic Concurrency Control)

---

## Theoretical Context

### 1. The Redux/Zustand Anti-Pattern: Confusing Server Cache with Client State
For years, the standard approach in React applications was to fetch remote API data in `useEffect` and dispatch actions to store it in a global state store (Redux, Zustand, or large `useState` trees). This is now recognized as the **#1 architectural anti-pattern in modern frontend engineering**:

1. **Treating Asynchronous Remote Data as Synchronous Local State:**
   Data from an API does not belong to the browser. The browser only holds a **temporary, potentially stale snapshot** of database records that can be modified concurrently by other users, background cron jobs, or database triggers at any second.
2. **Missing Deduplication & Background Revalidation:**
   Hand-rolled stores lack request deduplication (preventing multiple identical inflight requests), automatic garbage collection of unused screens, window-focus revalidation, exponential backoff retries, and network reconnection refetching.
3. **Synchronous Serialization & Re-render Cascades:**
   Putting high-frequency UI interactions (card selections, modal toggles, drag state) inside the same state structures as asynchronous server payloads forces frequent unnecessary re-render passes across large component subtrees.

---

### 2. The 4-Axis State Taxonomy

Modern scalable frontends categorize every piece of state into exactly one of four distinct axes based on **ownership, lifetime, and access patterns**:

```
+-----------------------------------------------------------------------------------+
|                                4-AXIS STATE TAXONOMY                              |
+-------------------+--------------------+--------------------+---------------------+
| Axis              | Tool / Mechanism   | Ownership & Scope  | Typical Examples    |
+-------------------+--------------------+--------------------+---------------------+
| 1. Server State   | TanStack Query     | Remote Database    | Task records, user  |
|    (Cache)        | (v5 Cache Engine)  | Shared, Asynch,    | profiles, analytics |
|                   |                    | potentially stale  | metrics, comments   |
+-------------------+--------------------+--------------------+---------------------+
| 2. Global Client/ | Zustand v5         | Browser Session    | Multi-select IDs,   |
|    UI State       | (useSyncExternal-  | Synchronous,       | modal/drawer open,  |
|                   |  Store selectors)  | ephemeral, local   | theme, toast queue  |
+-------------------+--------------------+--------------------+---------------------+
| 3. URL State      | URLSearchParams /  | Browser Address    | Search queries, tab |
|    (Location)     | History API        | Shareable, reload- | filters, sort order,|
|                   |                    | resilient, history | pagination cursor   |
+-------------------+--------------------+--------------------+---------------------+
| 4. Local UI State | useState / useRef  | Single Component   | Hover state, draft  |
|    (Ephemeral)    |                    | Private, isolated  | input, dropdown open|
+-------------------+--------------------+--------------------+---------------------+
```

#### Axis 1: Server State & TanStack Query v5
- **Query Key Factories:** Query keys are serialized tuples that uniquely identify cached data queries (e.g. `['tasks', 'list', { status: 'todo' }]`). Hierarchical factory functions (`taskKeys.list(filters)`) prevent accidental cache misses and allow surgical cache invalidation.
- **`staleTime` vs `gcTime`:**
  - `staleTime` determines how long data is considered *fresh*. While fresh, components mounting or focusing the window will immediately consume memory cache without issuing background network requests.
  - `gcTime` (Garbage Collection Time, formerly `cacheTime` in v4) dictates how long an *inactive* query (when no mounted components observe it) remains in memory before being evicted.
- **Stale-While-Revalidate (`placeholderData: keepPreviousData`):** Keeps existing list data on screen during cursor pagination or filter switches while new data fetches in the background, eliminating layout shifts.

#### Axis 2: Global Client / UI State & Zustand v5
- **Atomic Selectors via `useSyncExternalStore`:** Zustand does not wrap components in React Context providers. Instead, stores live outside React's lifecycle as external stores. React components subscribe via fine-grained selectors:
  ```ts
  const isSelected = useBoardUiStore((s) => s.selectedTaskIds.includes(taskId));
  ```
  Only the specific card whose selection boolean flips will re-render; other cards in the 100-item grid remain completely untouched.

#### Axis 3: URL State
- Any view configuration that a user might want to bookmark, refresh, or copy-paste to a teammate belongs in the URL (`URLSearchParams`).
- URL state acts as the reactive driver for TanStack Query keys: when URL parameters change, TanStack Query observes the key change and refetches or serves matching cached records automatically.

#### Axis 4: Local Ephemeral State
- Reserved for state that no other component cares about, such as whether a specific tooltip is open or raw keyboard navigation focus within an input field.

---

### 3. Distributed Consistency: Optimistic Updates & 409 Concurrency Control

When users mutate state in a collaborative system, waiting for server round-trips creates perceptible lag. **Optimistic Updates** patch the TanStack Query cache immediately, assuming success:

```
[User Action]
     │
     ▼
[onMutate] ──► 1. Cancel inflight queries (cancelQueries)
               2. Snapshot previous cache (getQueriesData)
               3. Patch cache optimistically (setQueriesData)
               4. Return snapshot context
     │
     ├──► [Server Responds 200 OK] ───────────► [onSettled] Invalidate & Sync
     │
     └──► [Server Responds 409 Conflict] ─────► [onError] Rollback from Snapshot
                                                          & Display Conflict Toast
```

- **Optimistic Concurrency Control (OCC):** The FastAPI backend enforces a record `version: int`. If a client sends an update with `version = 1`, but another session already updated the record to `version = 2`, the backend rejects the request with `HTTP 409 Conflict`.
- **Deterministic Rollback:** The `onError` handler reads the snapshot saved during `onMutate` and restores the exact previous state into `queryClient.setQueryData`, ensuring zero data corruption.

---

## Description

In this practice, you will build an **Operations Task Board** demonstrating the 4-axis state taxonomy:
1. **URL State:** Filter tasks by search query, status, and priority stored directly in `URLSearchParams`.
2. **Server Cache (TanStack Query):** Hierarchically cached task list with `staleTime`, `gcTime`, cursor pagination, and background revalidation.
3. **Client UI State (Zustand):** Multi-task batch selection, inspector drawer state, and conflict toast queue with atomic selector subscriptions.
4. **Optimistic Mutations with 409 Rollback:** Immediate visual status transitions with automatic rollback when the FastAPI backend returns `409 Conflict`.

---

## Instructions

### Step 1: Environment Setup
1. **Backend:**
   Create a virtual environment, install dependencies, and start the FastAPI server on port 8112:
   ```bash
   cd backend
   uv venv
   uv pip install fastapi uvicorn pydantic
   uv run uvicorn main:app --port 8112 --reload
   ```
2. **Frontend:**
   In another terminal, install frontend dependencies and start Vite:
   ```bash
   cd frontend
   npm install
   npm run dev
   ```
   Open `http://localhost:5173`.

---

### Step 2: Implement the 4 Core Exercises

#### Exercise 1: URL State & TanStack Query Hierarchical Key Factory
- **Files:** `frontend/src/hooks/useUrlFilters.ts` & `frontend/src/hooks/useTasksQuery.ts`
- **Context & Educational Value:**
  URL parameters must drive server queries without storing redundant copies in React component state.
- **Tasks:**
  - In `useUrlFilters.ts`: Implement `setFilters` to synchronize status/priority/search updates into `window.location.search` and browser history (`window.history.pushState`).
  - In `useTasksQuery.ts`: Configure `useQuery` using `taskKeys.list(filters)`, `staleTime: 10_000`, `gcTime: 300_000`, and `placeholderData: keepPreviousData`.

#### Exercise 2: Optimistic Mutation & 409 Conflict Rollback
- **File:** `frontend/src/hooks/useOptimisticTaskMutation.ts`
- **Context & Educational Value:**
  Teaches the complete optimistic update lifecycle in TanStack Query: snapshotting cache data, patching ahead of network completion, catching `409 Conflict` version mismatches, restoring the snapshot, and invalidating on settle.
- **Tasks:**
  - Implement `onMutate`: cancel outgoing queries, snapshot cache with `getQueriesData`, apply optimistic updates to task cards, and return `{ previousQueries }`.
  - Implement `onError`: restore cache from snapshot on error, and log `ConflictEvent` into Zustand store.
  - Implement `onSettled`: call `queryClient.invalidateQueries({ queryKey: taskKeys.all })`.

#### Exercise 3: Zustand Ephemeral UI State & Atomic Selectors
- **File:** `frontend/src/store/useBoardUiStore.ts`
- **Context & Educational Value:**
  Teaches how to structure ephemeral client state (multi-select IDs, inspector drawer, toast queue) using Zustand v5 without triggering Context Provider re-renders.
- **Tasks:**
  - Implement `toggleTaskSelection`, `selectAllTasks`, `clearSelection`, `setInspectorTaskId`, `addConflictEvent`, and `dismissConflictEvent`.
  - Notice the atomic selector `useIsTaskSelected(taskId)`: test selecting cards and observe that only the clicked card's render count (`R:n`) increments!

#### Exercise 4: Combining Server State & Zustand Client State in Batch Mutations
- **File:** `frontend/src/components/BoardWorkspace.tsx`
- **Context & Educational Value:**
  Demonstrates coordinating client selection state with server mutations: triggering a batch mutation for selected task IDs, invalidating the TanStack Query cache, and clearing the selection upon success.
- **Tasks:**
  - In `BoardWorkspace.tsx`: Implement `batchMutation` to call `batchUpdateTasks(selectedTaskIds, targetStatus)`, invalidate `taskKeys.all`, and invoke `clearSelection()`.

---

## Motivation
Modern enterprise web applications regularly collapse into unmaintainable re-render spirals and desynchronized caches when developers dump API responses into Redux or Zustand stores. Mastering the **4-Axis State Taxonomy** (TanStack Query for server cache, Zustand for atomic UI state, URL for location, and useState for local ephemerals) is a fundamental engineering competency required for high-performance React architectures.

---

## Commands

| Phase | Command | Description |
|---|---|---|
| **Backend Setup** | `cd backend && uv venv` | Create Python virtual environment for backend |
| **Backend Dependencies** | `cd backend && uv pip install -e .` | Install FastAPI and dependencies |
| **Backend Run** | `cd backend && uv run uvicorn main:app --port 8112 --reload` | Run FastAPI server with OCC and conflict simulator on port 8112 |
| **Frontend Setup** | `cd frontend && npm install` | Install React 19, TanStack Query v5, Zustand v5 |
| **Frontend Run** | `cd frontend && npm run dev` | Start Vite dev server on `http://localhost:5173` |
| **Frontend Build Check** | `cd frontend && npm run build` | Validate TypeScript types and bundle production assets |
| **Test Concurrency Conflict** | Check "Simulate 409 Conflict" in UI | Forces backend 409 error to observe optimistic rollback |
| **Reset Database** | Click "Reset Backend Seed Data" | Resets in-memory task database via `POST /api/reset` |
| **Clean** | `python clean.py` | Remove build artifacts, `node_modules`, and caches |

---

## Notes
*(Add your notes and observations here as you complete the exercises)*
