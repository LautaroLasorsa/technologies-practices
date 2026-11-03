# Practice 113: Complex Forms, Action Workflows & Multi-Layer Validation

## Technologies
- **React 19** (`useActionState`, `useFormStatus`)
- **React Hook Form** (`useForm`, `useFieldArray`, `useWatch`, `setError`)
- **Zod** (Schema definition, discriminated unions, `.superRefine()` cross-field invariants)
- **Fastify** (High-performance Node.js HTTP framework with CORS)
- **TypeScript** (End-to-end type safety between client and server)
- **Vite** (Next-generation frontend tooling and HMR)

## Stack
- **Frontend:** React 19, TypeScript, Vite, React Hook Form, `@hookform/resolvers/zod`
- **Backend:** Node.js, Fastify, TypeScript (`tsx`), Zod
- **Shared:** Isomorphic TypeScript schemas module (`shared/schemas.ts`)

---

## Theoretical Context

Enterprise web applications spend over half their lifecycle managing complex state-mutating forms: multi-step wizards, dynamic tabular rosters, financial ledgers, and operational schedules. Managing these forms naively leads to severe performance degradation ($O(N)$ re-render waterfalls) and architectural validation drift between frontend and backend.

### 1. Controlled vs. Uncontrolled Inputs & Render Performance
In standard React controlled form patterns, every input binds its state directly to component state:
```tsx
// NAIVE CONTROLLED PATTERN
const [formState, setFormState] = useState({ ... });
<input
  value={formState.allocations[i].resourceId}
  onChange={e => setFormState(prev => update(prev, i, e.target.value))}
/>
```
- **The $O(N)$ Waterfall:** When a form grows to dozens or hundreds of fields (e.g., dynamic resource allocations, time pickers, selects), every keystroke updates the top-level state. This forces React to re-render the entire component tree, reconcile all DOM nodes, and re-execute all child render functions. Frame rates plummet below 60 FPS, causing input lag.
- **React Hook Form's Inversion:** React Hook Form treats inputs as **uncontrolled DOM nodes by default**, registering native element references via `ref={register(...).ref}`. Keystrokes mutate native DOM value properties without triggering React component renders.
- **Micro-Subscriptions with `useWatch`:** When derived state (such as an aggregate load badge or dynamic summary) requires live values, calling `watch()` at the root would destroy the performance isolation. Instead, React Hook Form employs an internal event emitter (`Subject`). By placing `useWatch` inside an **isolated child component**, only that specific badge re-renders when inputs mutate; the surrounding form tree remains untouched.

### 2. React 19 Actions: Native Transitions vs. Complex Form Management
React 19 introduces native form handling integrated directly into the React Concurrent Reconciler:
- **`useActionState`**: Wraps an asynchronous action function `(prevState, formData) => Promise<nextState>`, automatically managing pending state (`isPending`), errors, and returned values within a React Transition (`startTransition`).
- **`useFormStatus`**: Consumes parent `<form>` context to read pending status inside child components without prop drilling.
- **Architectural Trade-Off Analysis:**
  - **Native React 19 Actions** shine for *atomic, transactional, RPC-style operations* (e.g., "Sign and Approve Shift", "Delete Item", "Toggle Status"). They require minimal boilerplate, use native browser `FormData`, and integrate seamlessly with React Server Components (RSC) and server actions.
  - **React Hook Form + Zod** remains indispensable for *complex multi-step client workflows*: dynamic arrays (`useFieldArray`), client-side caching of wizard steps before submission, fine-grained input masking, cross-field live validation feedback, and nested tabular error hydration.
  - Production architectures combine both: complex multi-step data collection uses React Hook Form + Zod, while downstream workflow state transitions use React 19 Actions.

### 3. Two-Layer Validation Architecture (Symmetric vs. Asymmetric)
Robust full-stack systems separate validation into two distinct structural layers:

```
[User Input in Browser]
          │
          ▼
┌────────────────────────────────────────────────────────┐
│ LAYER 1: Symmetric Domain Invariants (Shared Zod)       │
│ - Instant client feedback via @hookform/resolvers      │
│ - Re-executed on Fastify backend (Never trust client)  │
│ - Invariants: regex, types, endTime > startTime,       │
│   allocations.length >= 1, no duplicate resource IDs   │
└──────────────────────────┬─────────────────────────────┘
                           │ Valid? (HTTP POST)
                           ▼
┌────────────────────────────────────────────────────────┐
│ LAYER 2: Asymmetric Business & Persistence Validation   │
│ - Evaluated ONLY by Fastify backend (DB queries)       │
│ - Stateful rules: Shift code uniqueness in DB,         │
│   resource scheduling collision in booked windows       │
│ - Generates structured field-path error map:           │
│   { "allocations.1.resourceId": "Already booked" }     │
└──────────────────────────┬─────────────────────────────┘
                           │ HTTP 422 Unprocessable Entity
                           ▼
┌────────────────────────────────────────────────────────┐
│ Frontend Server-Error Hydration                        │
│ - Loop over fieldErrors:                               │
│     setError(path, { type: 'server', message })        │
│ - Preserves 100% of user input; highlights input in red │
└────────────────────────────────────────────────────────┘
```

#### Layer 1: Symmetric Domain Invariants (Shared Zod Schema)
- **Concept:** Rules that depend purely on the payload itself (data types, regex formatting, chronological order like `endTime > startTime`, minimum array lengths, internal duplicate detection).
- **Zero-Drift Architecture:** A single module (`shared/schemas.ts`) defines the Zod schema. The frontend binds it via `@hookform/resolvers/zod` for immediate client feedback on blur/submit. Fastify imports the exact same schema to parse incoming request bodies. If a malicious user bypasses the UI with `curl`, Fastify rejects the payload with HTTP 400.

#### Layer 2: Asymmetric Business & Persistence Validation
- **Concept:** Rules that depend on state stored in databases or external services (e.g., checking if `RES-ROBOT-01` is already booked during `08:00–16:00` by another team, or verifying shift code uniqueness). The browser cannot evaluate these synchronously.
- **Field-Path Error Hydration:** When a business rule fails, the backend must not return a generic string. Instead, it responds with HTTP 422 and a structured map pointing to specific dot-notation field paths:
  ```json
  {
    "success": false,
    "message": "Business rule validation failed",
    "fieldErrors": {
      "allocations.1.resourceId": "Resource 'RES-ROBOT-01' is already booked in shift SHIFT-2026-001 (08:00-16:00)"
    }
  }
  ```
- **Client Hydration:** The client receives this map and calls React Hook Form's `setError(path, { type: 'server', message })`. The UI immediately illuminates the exact row input in red and displays the server message, while preserving all existing user-entered values.

---

## Description
In this practice, you build a **Multi-Step Production Shift & Resource Allocation Wizard** for an advanced manufacturing plant:
1. **Step 1 (Shift Header):** Shift code, department, shift type (standard vs. overnight), start/end times, and conditional overnight supervisor.
2. **Step 2 (Dynamic Allocations):** Dynamic field array of robotics and human operators with real-time aggregate workload telemetry.
3. **Multi-Layer Validation Pipeline:** Shared Zod invariants (Layer 1) and Fastify database collision checks (Layer 2) with seamless field-error hydration.
4. **React 19 Action Workflow:** An atomic supervisor sign-off action built with `useActionState` and `useFormStatus`.

---

## Instructions

### Step 1: Install Dependencies
Open two terminal windows or run the install commands from the practice root:
```bash
# Install backend dependencies
cd backend && npm install

# Install frontend dependencies
cd ../frontend && npm install
```

### Step 2: Implement Exercises

#### Exercise 1: Shared Zod Schema with Cross-Field & Array Invariants
- **File:** `shared/schemas.ts`
- **Context:** Forms must enforce domain invariants before touching the network. Using Zod's `.superRefine()`, you will build cross-field validation logic shared across frontend and backend.
- **Tasks (`TODO(human)`):**
  1. For standard shifts (`shiftType === 'standard'`), verify `endTime` is strictly greater than `startTime` on the same calendar day. If not, add an issue to path `['endTime']`.
  2. For overnight shifts (`shiftType === 'overnight'`), verify that a `handoverSupervisor` of at least 3 characters is specified, and that `endTime` crosses midnight (numerically earlier than `startTime`).
  3. For the `allocations` array, verify that `allocations.length >= 1`.
  4. Iterate through `allocations` to detect duplicate `resourceId` entries, adding an issue targeting `['allocations', i, 'resourceId']`.

#### Exercise 2: Dynamic Field Array with Isolated Telemetry
- **File:** `frontend/src/components/AllocationArrayFields.tsx`
- **Context:** Managing dynamic arrays without causing performance degradation requires understanding React Hook Form's uncontrolled array model.
- **Tasks (`TODO(human)`):**
  1. Configure `useFieldArray({ control, name: 'allocations' })`.
  2. Render dynamic allocation cards using `key={field.id}` (never use array index as key).
  3. Register nested inputs (`allocations.${index}.resourceId`, `role`, `allocatedHours`, `loadPercentage`).
  4. Observe how `AllocationMetricsBadge` uses `useWatch` to compute total hours and aggregate workload in an isolated component without re-rendering the rows.

#### Exercise 3: Stateful Business Validation & Field-Error Hydration
- **Files:** `backend/src/server.ts` & `frontend/src/hooks/useWizardSubmit.ts`
- **Context:** Client schemas cannot check database collisions. You will implement backend persistence checks and client error hydration.
- **Tasks (`TODO(human)`):**
  1. In `backend/src/server.ts`:
     - Check if `ACTIVE_SHIFT_CODES.has(shiftData.shiftCode)`. If taken, attach `fieldErrors['shiftCode']`.
     - Check each allocation in `shiftData.allocations` against `BOOKED_RESOURCES`. If `resourceId` matches and time intervals overlap (`Math.max(s1, s2) < Math.min(e1, e2)`), attach `fieldErrors[\`allocations.${i}.resourceId\`]`.
     - Return HTTP 422 with the structured `fieldErrors` map if collisions exist.
  2. In `frontend/src/hooks/useWizardSubmit.ts`:
     - When receiving HTTP 422 / 400, parse JSON and iterate over `Object.entries(errorData.fieldErrors)`.
     - Call `setError(fieldPath as any, { type: 'server', message })` to illuminate the exact form inputs.

#### Exercise 4: React 19 Action Workflow
- **File:** `frontend/src/components/QuickApprovalAction.tsx`
- **Context:** Atomic mutations should not require heavy form state. React 19 Actions provide transition-managed server mutations.
- **Tasks (`TODO(human)`):**
  1. Author the `approvalAction(prevState, formData)` function handling `POST /api/shifts/:id/quick-approval`.
  2. Bind the action with `useActionState(approvalAction, initialState)`.
  3. Create a child submit button consuming `useFormStatus()` to automatically reflect pending state and disable submission.

---

## Motivation
Modern enterprise web development demands mastery over both high-throughput client form UX and bulletproof backend data integrity. Learning how to decouple renders with uncontrolled forms, share Zod contracts end-to-end, hydrate server business errors into nested UI inputs, and leverage React 19 Actions equips you to build production-grade, high-performance web applications.

---

## Commands

| Phase | Command | Description |
|---|---|---|
| **Setup** | `cd backend && npm install` | Install Fastify backend dependencies |
| **Setup** | `cd frontend && npm install` | Install React 19 & Vite frontend dependencies |
| **Execution** | `cd backend && npm run dev` | Run Fastify backend with hot-reload on port 4113 |
| **Execution** | `cd frontend && npm run dev` | Run Vite dev server on port 5113 |
| **Build** | `cd frontend && npm run build` | Typecheck and compile Vite production bundle |
| **Maintenance** | `python clean.py` | Remove `node_modules`, `dist`, caches, and build artifacts |

---

## Notes
*(Add your notes and observations here during practice)*
