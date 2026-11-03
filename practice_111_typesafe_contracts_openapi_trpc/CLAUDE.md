# Practice 111: Type-Safe Full-Stack Contracts — OpenAPI (Python) vs tRPC (Node)

## Technologies

- **React 19** — Modern declarative UI library with Server Components & enhanced hook semantics
- **TypeScript 5.5+** — Static type system with advanced template literals, indexed access types, and discriminated unions
- **Vite 6** — Next-generation frontend tooling and bundler
- **FastAPI (v0.115+)** — High-performance Python async web framework with automatic OpenAPI 3.1 generation
- **Pydantic v2** — Blazing-fast Rust-backed data validation and schema serialization library
- **openapi-typescript (v7+) & openapi-fetch (v0.13+)** — Lightweight zero-runtime type generation and type-safe fetch client for OpenAPI schemas
- **Node.js (v22+)** — Modern JavaScript server runtime
- **tRPC (v11)** — End-to-end type-safe RPC framework leveraging TypeScript inference without code generation
- **Zod (v3.23+)** — TypeScript-first schema declaration and runtime validation library

## Stack

- **Frontend & BFF Service:** TypeScript 5.5+, Node.js 22, React 19, Vite
- **Optimization & Core Service:** Python 3.11+ / 3.12+, FastAPI, Pydantic v2, uv

---

## Theoretical Context

### The Schema Drift Problem in Distributed Systems

In modern full-stack web architectures, the single most frequent cause of production regressions is **Schema Drift**: the gradual, silent divergence between what a backend service serializes across the network and what the frontend consumer expects to receive.

Historically, frontend engineers maintain manual TypeScript interfaces:

```typescript
// frontend/src/types/job.ts (Manually maintained - FRAGILE)
interface OptimizationJob {
  id: string;
  assignedShifts: number;
  totalCost: number;
}
```

Meanwhile, a backend engineer working on a Python FastAPI or Node service modifies a model:

```python
# services/python-api/models.py
class OptimizationJob(BaseModel):
    id: UUID
    assigned_shifts_count: int  # Renamed from assigned_shifts!
    total_cost: float
    overtime_penalty: float     # Newly added required field!
```

This manual synchronization approach suffers from critical structural flaws:
1. **Silent Runtime Crashes:** The TypeScript compiler (`tsc`) has no visibility into the Python backend. The frontend compiles without errors, but in production, accessing `job.assignedShifts` evaluates to `undefined`, silently corrupting charts, forms, or crashing with `TypeError: Cannot read properties of undefined`.
2. **Untracked Nullability and Enums:** A backend field changed from `string` to `Optional[str]` or an enum value added to a status enum will go unnoticed by the frontend until runtime exceptions emerge.
3. **Loss of Error Contract Visibility:** HTTP error payloads (`400 Bad Request`, `409 Conflict`, `422 Unprocessable Entity`) are frequently untyped or cast to `any`, causing UI error handlers to miss specific domain error properties.
4. **Organizational Friction:** As teams scale, verbal coordination, Slack messages, or out-of-date Confluence pages fail to prevent contract divergence.

To achieve true resilience, the contract across the network boundary must be **machine-verifiable, statically enforced, and impossible to break without failing the build pipeline**.

---

### Architectural Paradigms: Schema-First vs Code-First Inference

To eliminate schema drift, modern engineering has converged on two primary paradigms:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                               ARCHITECTURAL PARADIGMS                                 │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 1. Schema-First / Codegen Boundary (Cross-Language: Python -> OpenAPI -> TypeScript)  │
│                                                                                        │
│   FastAPI (Pydantic v2) ──> OpenAPI 3.1 JSON AST ──> openapi-typescript ──> schema.d.ts│
│                                                            │                           │
│                                                            ▼                           │
│                                             openapi-fetch (Typed Client)               │
│                                                                                        │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 2. Code-First / Type Inference Boundary (Full TypeScript Monorepo: Node -> React)      │
│                                                                                        │
│   tRPC Router (Zod) ──> export type AppRouter ──> import type { AppRouter } ──> Client │
│   (No code generation, zero bundle size, direct generic type propagation)              │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

#### Paradigm 1: Schema-First / Codegen Boundary (Python FastAPI → OpenAPI → TypeScript)

When backends and frontends are polyglot (e.g., Python for optimization/machine learning and TypeScript for user interfaces), language boundaries prevent direct type sharing.

1. **AST Extraction via Pydantic v2:** Pydantic v2 inspects Python type hints, constraints (e.g., `Field(ge=0)`), docstrings, and response models, compiling them into a standardized OpenAPI 3.1 JSON Abstract Syntax Tree (AST).
2. **Zero-Runtime Codegen via `openapi-typescript`:** Unlike legacy Java-based tools (such as `openapi-generator-cli`) that generated thousands of lines of bloated object-oriented boilerplate classes and Axios wrappers, `openapi-typescript` generates pure, compile-time TypeScript type definitions (`schema.d.ts`). It produces **zero JavaScript runtime code**, meaning zero bundle weight overhead.
3. **Indexed Access Types via `openapi-fetch`:** `openapi-fetch` leverages TypeScript's mapped and indexed access types:
   ```typescript
   type JobResponse = paths["/api/v1/jobs/{job_id}"]["get"]["responses"][200]["content"]["application/json"];
   ```
   The client function `client.GET("/api/v1/jobs/{job_id}", { params: { path: { job_id } } })` validates paths, query parameters, headers, and request bodies at compile time. If a backend engineer changes `/api/v1/jobs/{job_id}` to `/api/v1/optimization-jobs/{id}`, `tsc` fails immediately with an invalid route literal error.
4. **The Critical Role of `operationId`:** OpenAPI operations can be indexed by route path or by `operationId`. Configuring clear, explicit `operation_id` attributes (e.g., `create_optimization_job`) ensures predictable, stable naming when consuming endpoints.

#### Paradigm 2: Code-First / Type Inference Boundary (Node + tRPC + Zod)

When both backend (or Backend-For-Frontend) and frontend reside within a TypeScript ecosystem, code generation is entirely unnecessary:

1. **Generic Type Inference:** tRPC uses TypeScript's advanced conditional and mapped types to infer the entire API contract directly from the implementation:
   ```typescript
   // services/node-trpc/src/router.ts
   export const appRouter = router({
     workspace: router({
       getById: publicProcedure
         .input(z.object({ id: z.string().uuid() }))
         .query(async ({ input }) => { /* ... */ }),
     }),
   });
   export type AppRouter = typeof appRouter;
   ```
2. **Zero Bundle Leakage via `import type`:** The frontend imports **only the type declaration**:
   ```typescript
   // frontend/src/api/trpcClient.ts
   import type { AppRouter } from '../../../services/node-trpc/src/router';
   ```
   Because `import type` is used, the TypeScript compiler erases the import completely during compilation. None of the server code, database queries, or server-side dependencies leak into the client bundle.
3. **RPC Semantics over HTTP:** Rather than manually composing REST URLs and HTTP verbs, the developer calls methods via a proxy client: `trpc.workspace.getById.query({ id })`. Rename a procedure on the server, and every client reference across the workspace turns red in the IDE before you even save the file.

---

### Compile-Time vs Runtime Type Safety

A common misconception is that TypeScript types guarantee runtime data integrity. In reality:

> **TypeScript types are completely erased at compile time.** At runtime, JavaScript is untyped. The network is an untrusted boundary.

```
       Compile Time (Static Analysis)            Runtime (JavaScript Execution)
   ┌─────────────────────────────────────┐    ┌───────────────────────────────────┐
   │ Interface / Type definitions exist  │    │ Types are ERASED (`tsc` emit)     │
   │ `tsc` checks assignments & calls    │ ──>│ Untrusted JSON arrives over wire  │
   │ Refactoring triggers IDE errors     │    │ Payload might be corrupted or old │
   └─────────────────────────────────────┘    └─────────────────┬─────────────────┘
                                                                │
                                              Defensive Runtime Validation
                                              (Zod / Pydantic parse boundary)
                                                                │
                                                                ▼
                                                Valid State OR Handled Failure
```

If an external service sends unexpected JSON (or an attacker sends a malicious payload), static TypeScript types cannot prevent runtime errors. Therefore:
- **Zod & Pydantic provide runtime parsing at boundaries:** They validate incoming JSON against schemas, parsing valid values into known shapes and rejecting invalid values with clear diagnostics.
- **Discriminated Unions model exhaustive error states:** By tagging responses with a discriminator property (e.g., `{ success: true, data: T } | { success: false, error: E }`), TypeScript forces developers to check the status before accessing payload properties:
  ```typescript
  if (result.status === "error") {
    // TypeScript knows `result.data` DOES NOT exist here!
    // TypeScript knows `result.errorDetails` DOES exist here!
    console.error(result.errorDetails.reason);
  }
  ```

---

## Description

In this practice, you will build and analyze a hybrid, polyglot full-stack architecture that demonstrates how to achieve 100% type-safe contracts across both schema-first and code-first boundaries:

1. **Python FastAPI Optimization Service (Port 8111):** Simulates a mathematical scheduling optimization backend. Exposes endpoints for creating optimization jobs, checking status, and handling domain conflicts (`SCHEDULE_CONFLICT`, `RESOURCE_EXHAUSTED`). Generates an OpenAPI 3.1 specification.
2. **Node.js tRPC Workspace BFF (Port 4111):** Simulates a collaboration & workspace configuration backend. Implements procedures with Zod validation, context middleware, and structured `TRPCError` exceptions. Exports its inferred `AppRouter` type.
3. **React 19 + TypeScript Frontend (Port 5173):** Connects simultaneously to both services:
   - Consumes the Python service using `openapi-fetch` and types generated by `openapi-typescript`.
   - Consumes the Node service using `@trpc/client` and inferred types from `AppRouter`.
   - Features a **Contract Drift Simulator** that lets you simulate field renames and schema violations, observing the stark difference between compile-time prevention and runtime safety guards.

---

## Instructions

### Step 1: Environment Setup

1. Setup the Python service:
   ```bash
   cd services/python-api
   uv sync
   ```
2. Setup the Node tRPC service:
   ```bash
   cd services/node-trpc
   npm install
   ```
3. Setup the Frontend:
   ```bash
   cd frontend
   npm install
   ```

---

### Step 2: Exercises & Core Learning Logic

#### Exercise 1: Python OpenAPI Contract Definition & Typed Fetch Client
- **Files:** `services/python-api/main.py` & `frontend/src/api/openapiClient.ts`
- **Context & Learning Objective:** Understand how Pydantic v2 models map to OpenAPI 3.1 schemas, how explicit `operation_id`s structure client codegen, and how `openapi-fetch` enforces type-safe requests and responses based on HTTP status codes.
- **Tasks (`TODO(human)`):**
  1. In `services/python-api/main.py`, define the strict Pydantic models for `OptimizationRequest`, `OptimizationSuccessResponse`, and a discriminated error model `OptimizationConflictError` with code `"SCHEDULE_CONFLICT"`. Ensure the endpoint specifies `responses={409: {"model": OptimizationConflictError}}`.
  2. Run the OpenAPI export script (`python export_openapi.py`) and regenerate the TypeScript types using `npm run generate:openapi` in `frontend/`.
  3. In `frontend/src/api/openapiClient.ts`, complete the typed wrapper function `triggerOptimization` to handle both `200` success and `409` conflict responses with compile-time checked property access.

#### Exercise 2: Node.js tRPC Router, Zod Validation & Context Middleware
- **Files:** `services/node-trpc/src/router.ts` & `services/node-trpc/src/trpc.ts`
- **Context & Learning Objective:** Understand how TypeScript generic inference propagates Zod schemas from backend procedures directly into client types without intermediary files, and how domain errors are structured using `TRPCError`.
- **Tasks (`TODO(human)`):**
  1. In `services/node-trpc/src/router.ts`, implement the `workspace.updateSchedulePolicy` mutation.
  2. Define the Zod input schema enforcing valid constraints (e.g. `maxOvertimeHours` between 0 and 40, `allowWeekendShifts` boolean).
  3. Implement business logic validation that throws a `TRPCError` with code `'CONFLICT'` if overtime policy violates workspace tier limits.
  4. Verify that `export type AppRouter = typeof appRouter;` automatically exposes this procedure to consumers.

#### Exercise 3: React Contract Consumer & Exhaustive Discriminated Union Narrowing
- **Files:** `frontend/src/components/ContractExplorer.tsx`
- **Context & Learning Objective:** Learn how to consume both API contracts in a React UI using exhaustive pattern matching without resorting to `any`, `unknown` bypasses, or unsafe type assertions (`as Type`).
- **Tasks (`TODO(human)`):**
  1. In `ContractExplorer.tsx`, wire the state handlers for calling both the Python OpenAPI endpoint and the Node tRPC mutation.
  2. Implement an exhaustive `switch` / discriminated union narrowing block for the optimization job response, handling `status === 'COMPLETED'`, `status === 'QUEUED'`, and `status === 'FAILED'` without unhandled edge cases.

#### Exercise 4: Defensive Runtime Boundary Guard & Drift Resilience
- **Files:** `frontend/src/utils/contractBoundaryGuard.ts`
- **Context & Learning Objective:** Contrast compile-time type safety with runtime safety. Discover why third-party or version-lagged APIs require defensive parsing, and implement a Zod-powered safe parser that catches schema drift at runtime before data touches the UI components.
- **Tasks (`TODO(human)`):**
  1. Implement `safeParseContractPayload` using a versioned Zod schema.
  2. Simulate a contract drift scenario (e.g., backend renames a field or returns unexpected nulls) and verify that the defensive boundary guard catches the error and surfaces human-readable diagnostics.

---

## Motivation

- **High-Demand Full-Stack Architecture:** Modern enterprise stacks overwhelmingly pair TypeScript frontends (Next.js, Vite, React) with either Python backends (FastAPI, PyTorch, Celery) or Node/TypeScript BFFs. Mastering both contract paradigms is essential for senior full-stack roles.
- **Elimination of Class of Production Bugs:** Eliminating schema drift prevents 80%+ of cross-team integration bugs before code is even committed.
- **Zero-Cost Abstractions:** Understanding how `openapi-typescript` and `tRPC` achieve type safety with zero bundle bloat and zero runtime performance penalty separates novice developers from experienced systems architects.

---

## Commands

All commands can be executed from the respective directories or from the root of `practice_111_typesafe_contracts_openapi_trpc/`.

### Python Service (`services/python-api/`)

| Command | Description |
|---------|-------------|
| `uv sync` | Install Python dependencies into virtual environment |
| `uv run uvicorn main:app --reload --port 8111` | Start FastAPI service on port 8111 with live reload |
| `uv run python export_openapi.py` | Export OpenAPI 3.1 JSON AST to `openapi.json` |

### Node tRPC Service (`services/node-trpc/`)

| Command | Description |
|---------|-------------|
| `npm install` | Install Node dependencies for tRPC backend |
| `npm run dev` | Start tRPC HTTP server on port 4111 with hot reload (`tsx`) |
| `npm run typecheck` | Run `tsc --noEmit` to verify backend types |

### Frontend (`frontend/`)

| Command | Description |
|---------|-------------|
| `npm install` | Install frontend React & tooling dependencies |
| `npm run generate:openapi` | Generate TypeScript contract (`schema.d.ts`) from Python `openapi.json` |
| `npm run dev` | Start Vite dev server on port 5173 |
| `npm run typecheck` | Statically typecheck all frontend components, OpenAPI client, and tRPC client |
| `npm run build` | Compile frontend for production |

### Maintenance & Cleanup

| Command | Description |
|---------|-------------|
| `python clean.py` | Remove all temporary build files, caches, `.venv`, `node_modules`, and generated artifacts |

---

## Notes

*(To be populated during practice execution with observations, design trade-offs, and learnings.)*

---

## State

`not-started`
