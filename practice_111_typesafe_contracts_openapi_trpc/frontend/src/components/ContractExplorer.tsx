import { useState, useEffect, type FC } from "react";
import {
  triggerOptimization,
  type OptimizationApiResult,
  type OptimizationRequest,
} from "../api/openapiClient.js";
import { trpc } from "../api/trpcClient.js";
import {
  validateContractBoundary,
  type BoundaryValidationResult,
} from "../utils/contractBoundaryGuard.js";

interface WorkspaceItem {
  id: string;
  name: string;
  tier: "STARTER" | "GROWTH" | "ENTERPRISE";
  maxOvertimeHours: number;
  allowWeekendShifts: boolean;
  updatedAt: string;
}

export const ContractExplorer: FC = () => {
  // ---------------------------------------------------------------------------
  // Python FastAPI / OpenAPI State
  // ---------------------------------------------------------------------------
  const [pythonParams, setPythonParams] = useState<OptimizationRequest>({
    horizon_days: 14,
    workforce_capacity: 12,
    cost_weight: 0.7,
    allow_consecutive_night_shifts: false,
  });
  const [pythonLoading, setPythonLoading] = useState(false);
  const [pythonResult, setPythonResult] = useState<OptimizationApiResult | null>(
    null
  );

  // ---------------------------------------------------------------------------
  // Node.js / tRPC State
  // ---------------------------------------------------------------------------
  const [workspaces, setWorkspaces] = useState<WorkspaceItem[]>([]);
  const [selectedWsId, setSelectedWsId] = useState<string>("ws-alpha");
  const [overtimeHours, setOvertimeHours] = useState<number>(15);
  const [allowWeekends, setAllowWeekends] = useState<boolean>(true);
  const [wsTier, setWsTier] = useState<"STARTER" | "GROWTH" | "ENTERPRISE">(
    "GROWTH"
  );
  const [trpcLoading, setTrpcLoading] = useState(false);
  const [trpcMessage, setTrpcMessage] = useState<string | null>(null);
  const [trpcError, setTrpcError] = useState<string | null>(null);

  // ---------------------------------------------------------------------------
  // Runtime Boundary & Schema Drift Simulation State
  // ---------------------------------------------------------------------------
  const [driftSimulationMode, setDriftSimulationMode] = useState<
    "VALID" | "RENAMED_FIELD" | "TYPE_MISMATCH" | "MISSING_NESTED"
  >("VALID");
  const [boundaryCheckResult, setBoundaryCheckResult] =
    useState<BoundaryValidationResult | null>(null);

  // Load workspaces on mount via tRPC query
  useEffect(() => {
    async function loadWorkspaces() {
      try {
        const list = await trpc.workspace.list.query();
        setWorkspaces(list);
        if (list.length > 0) {
          const first = list[0];
          setSelectedWsId(first.id);
          setOvertimeHours(first.maxOvertimeHours);
          setAllowWeekends(first.allowWeekendShifts);
          setWsTier(first.tier);
        }
      } catch (err: unknown) {
        console.warn("Could not connect to tRPC backend yet:", err);
      }
    }
    loadWorkspaces();
  }, []);

  // Update form fields when workspace selection changes
  const handleWorkspaceSelect = (id: string) => {
    setSelectedWsId(id);
    const found = workspaces.find((w) => w.id === id);
    if (found) {
      setOvertimeHours(found.maxOvertimeHours);
      setAllowWeekends(found.allowWeekendShifts);
      setWsTier(found.tier);
      setTrpcMessage(null);
      setTrpcError(null);
    }
  };

  // Trigger Python API call
  const handleRunPythonOptimization = async () => {
    setPythonLoading(true);
    setPythonResult(null);
    try {
      const result = await triggerOptimization(pythonParams);
      setPythonResult(result);
    } catch (err: unknown) {
      setPythonResult({
        kind: "UNKNOWN_ERROR",
        statusCode: 0,
        message: err instanceof Error ? err.message : "Network failure",
      });
    } finally {
      setPythonLoading(false);
    }
  };

  // Trigger Node tRPC mutation
  const handleUpdatePolicy = async () => {
    setTrpcLoading(true);
    setTrpcMessage(null);
    setTrpcError(null);

    try {
      // Inferred tRPC client call with compile-time type verification!
      const res = await trpc.workspace.updateSchedulePolicy.mutate({
        workspaceId: selectedWsId,
        maxOvertimeHours: overtimeHours,
        allowWeekendShifts: allowWeekends,
        tier: wsTier,
      });

      setTrpcMessage(
        `Policy successfully updated for ${res.workspace.name} (Request ID: ${res.requestId})`
      );

      // Refresh list
      const updatedList = await trpc.workspace.list.query();
      setWorkspaces(updatedList);
    } catch (err: unknown) {
      if (typeof err === "object" && err !== null && "message" in err) {
        setTrpcError(String((err as { message: string }).message));
      } else {
        setTrpcError("Unknown tRPC mutation error.");
      }
    } finally {
      setTrpcLoading(false);
    }
  };

  // Execute Boundary Validation Test
  const runDriftSimulation = () => {
    let mockPayload: unknown;

    if (driftSimulationMode === "VALID") {
      mockPayload = {
        status: "COMPLETED",
        job_id: "opt-sim-999",
        horizon_days: 14,
        total_cost: 1680.5,
        solver_status: "OPTIMAL",
        assignments: [
          {
            slot_id: "slot-01",
            worker_id: "worker-04",
            shift_name: "Morning",
            day_offset: 0,
            estimated_cost: 150.0,
          },
        ],
        created_at: new Date().toISOString(),
      };
    } else if (driftSimulationMode === "RENAMED_FIELD") {
      // Backend engineer renamed `total_cost` -> `overall_expense`
      mockPayload = {
        status: "COMPLETED",
        job_id: "opt-drift-101",
        horizon_days: 14,
        overall_expense: 1680.5, // DRIFT!
        solver_status: "OPTIMAL",
        assignments: [],
        created_at: new Date().toISOString(),
      };
    } else if (driftSimulationMode === "TYPE_MISMATCH") {
      // Backend engineer serialized number as string
      mockPayload = {
        status: "COMPLETED",
        job_id: "opt-drift-102",
        horizon_days: "fourteen_days", // DRIFT! Expected positive integer
        total_cost: 1680.5,
        solver_status: "OPTIMAL",
        assignments: [],
        created_at: new Date().toISOString(),
      };
    } else {
      // Missing nested required array
      mockPayload = {
        status: "COMPLETED",
        job_id: "opt-drift-103",
        horizon_days: 14,
        total_cost: 1680.5,
        solver_status: "OPTIMAL",
        // assignments missing!
        created_at: "not-a-valid-iso-datetime",
      };
    }

    const check = validateContractBoundary(mockPayload);
    setBoundaryCheckResult(check);
  };

  // ===========================================================================
  // TODO(human): Exercise 3
  // Notice this exhaustive pattern matcher on `pythonResult.kind`.
  // Because `OptimizationApiResult` is a Discriminated Union, TypeScript
  // guarantees that you cannot accidentally read `result.data` inside the
  // `CONFLICT` branch or `result.error` inside the `SUCCESS` branch.
  // ===========================================================================
  const renderPythonOutcome = () => {
    if (!pythonResult) return null;

    switch (pythonResult.kind) {
      case "SUCCESS": {
        const { data } = pythonResult;
        return (
          <div className="result-card success-card">
            <div className="badge-row">
              <span className="badge badge-success">HTTP 200 OK</span>
              <span className="badge badge-info">{data.solver_status}</span>
              <span className="badge badge-neutral">Job ID: {data.job_id}</span>
            </div>
            <p>
              <strong>Total Minimized Cost:</strong> ${data.total_cost.toFixed(2)} |{" "}
              <strong>Horizon:</strong> {data.horizon_days} days
            </p>
            <h4>Assigned Shift Schedule ({data.assignments.length} slots):</h4>
            <div className="table-wrapper">
              <table className="schedule-table">
                <thead>
                  <tr>
                    <th>Day</th>
                    <th>Worker</th>
                    <th>Shift</th>
                    <th>Est. Cost</th>
                  </tr>
                </thead>
                <tbody>
                  {data.assignments.map((slot) => (
                    <tr key={slot.slot_id}>
                      <td>Day +{slot.day_offset}</td>
                      <td><code>{slot.worker_id}</code></td>
                      <td>{slot.shift_name}</td>
                      <td>${slot.estimated_cost.toFixed(2)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        );
      }

      case "CONFLICT": {
        const { error } = pythonResult;
        return (
          <div className="result-card conflict-card">
            <div className="badge-row">
              <span className="badge badge-danger">HTTP 409 CONFLICT</span>
              <span className="badge badge-warning">{error.code}</span>
            </div>
            <p><strong>Message:</strong> {error.message}</p>
            <p><strong>Recommended Action:</strong> {error.recommended_action}</p>
            {error.conflicting_slots && error.conflicting_slots.length > 0 && (
              <p>
                <strong>Conflicting Slots:</strong>{" "}
                {error.conflicting_slots.map((s) => (
                  <code key={s} className="tag">{s}</code>
                ))}
              </p>
            )}
          </div>
        );
      }

      case "EXHAUSTED": {
        const { error } = pythonResult;
        return (
          <div className="result-card error-card">
            <div className="badge-row">
              <span className="badge badge-danger">HTTP 422 UNPROCESSABLE</span>
              <span className="badge badge-warning">{error.code}</span>
            </div>
            <p><strong>Error:</strong> {error.message}</p>
            <p>
              <strong>Available Workers:</strong> {error.available_workers} |{" "}
              <strong>Required Minimum:</strong> {error.required_workers}
            </p>
          </div>
        );
      }

      case "UNKNOWN_ERROR": {
        return (
          <div className="result-card error-card">
            <span className="badge badge-danger">
              Network / HTTP {pythonResult.statusCode}
            </span>
            <p>{pythonResult.message}</p>
          </div>
        );
      }
    }
  };

  return (
    <div className="contract-explorer-container">
      <header className="hero-header">
        <h1>Practice 111: Type-Safe Full-Stack Contracts</h1>
        <p className="subtitle">
          Comparing <strong>Schema-First / Codegen (Python FastAPI + OpenAPI)</strong> vs{" "}
          <strong>Code-First / Inference (Node.js + tRPC)</strong>
        </p>
      </header>

      <div className="grid-two-col">
        {/* ================================================================= */}
        {/* PANEL 1: Python FastAPI + OpenAPI */}
        {/* ================================================================= */}
        <section className="panel-card">
          <div className="panel-header">
            <span className="panel-icon">🐍</span>
            <div>
              <h2>Python Optimization Service</h2>
              <span className="tag-sub">FastAPI • Pydantic v2 • openapi-fetch (Port 8111)</span>
            </div>
          </div>

          <div className="form-group">
            <label>Planning Horizon (Days): {pythonParams.horizon_days}</label>
            <input
              type="range"
              min={1}
              max={60}
              value={pythonParams.horizon_days}
              onChange={(e) =>
                setPythonParams({
                  ...pythonParams,
                  horizon_days: Number(e.target.value),
                })
              }
            />
            <span className="hint">Tip: Set &gt;30 with capacity &lt;10 to trigger HTTP 409 Conflict!</span>
          </div>

          <div className="form-group">
            <label>Workforce Capacity (Workers): {pythonParams.workforce_capacity}</label>
            <input
              type="range"
              min={1}
              max={50}
              value={pythonParams.workforce_capacity}
              onChange={(e) =>
                setPythonParams({
                  ...pythonParams,
                  workforce_capacity: Number(e.target.value),
                })
              }
            />
            <span className="hint">Tip: Set &lt;3 to trigger HTTP 422 Resource Exhausted!</span>
          </div>

          <div className="form-group">
            <label>Cost Reduction Weight: {pythonParams.cost_weight}</label>
            <input
              type="range"
              min={0.1}
              max={1.0}
              step={0.1}
              value={pythonParams.cost_weight}
              onChange={(e) =>
                setPythonParams({
                  ...pythonParams,
                  cost_weight: Number(e.target.value),
                })
              }
            />
          </div>

          <div className="checkbox-row">
            <label>
              <input
                type="checkbox"
                checked={pythonParams.allow_consecutive_night_shifts}
                onChange={(e) =>
                  setPythonParams({
                    ...pythonParams,
                    allow_consecutive_night_shifts: e.target.checked,
                  })
                }
              />
              Allow Consecutive Night Shifts
            </label>
          </div>

          <button
            className="btn btn-primary"
            onClick={handleRunPythonOptimization}
            disabled={pythonLoading}
          >
            {pythonLoading ? "Solving Optimization..." : "Execute Python Solver (POST /optimize)"}
          </button>

          <div className="results-container">{renderPythonOutcome()}</div>
        </section>

        {/* ================================================================= */}
        {/* PANEL 2: Node.js + tRPC */}
        {/* ================================================================= */}
        <section className="panel-card">
          <div className="panel-header">
            <span className="panel-icon">⚡</span>
            <div>
              <h2>Node Workspace BFF</h2>
              <span className="tag-sub">tRPC v11 • Zod • AppRouter Inference (Port 4111)</span>
            </div>
          </div>

          <div className="form-group">
            <label>Select Workspace Target:</label>
            <select
              value={selectedWsId}
              onChange={(e) => handleWorkspaceSelect(e.target.value)}
              className="select-input"
            >
              {workspaces.map((w) => (
                <option key={w.id} value={w.id}>
                  {w.name} ({w.tier})
                </option>
              ))}
            </select>
          </div>

          <div className="form-group">
            <label>Workspace Tier:</label>
            <select
              value={wsTier}
              onChange={(e) =>
                setWsTier(e.target.value as "STARTER" | "GROWTH" | "ENTERPRISE")
              }
              className="select-input"
            >
              <option value="STARTER">STARTER (Max 10h overtime)</option>
              <option value="GROWTH">GROWTH (Standard)</option>
              <option value="ENTERPRISE">ENTERPRISE (Unlimited)</option>
            </select>
          </div>

          <div className="form-group">
            <label>Max Overtime Hours / Cycle: {overtimeHours}h</label>
            <input
              type="number"
              min={0}
              max={40}
              value={overtimeHours}
              onChange={(e) => setOvertimeHours(Number(e.target.value))}
              className="number-input"
            />
            <span className="hint">
              Tip: Set tier to STARTER and overtime &gt;10h to trigger tRPC CONFLICT exception!
            </span>
          </div>

          <div className="checkbox-row">
            <label>
              <input
                type="checkbox"
                checked={allowWeekends}
                onChange={(e) => setAllowWeekends(e.target.checked)}
              />
              Allow Weekend Shift Scheduling
            </label>
          </div>

          <button
            className="btn btn-secondary"
            onClick={handleUpdatePolicy}
            disabled={trpcLoading}
          >
            {trpcLoading ? "Updating Policy..." : "Mutate Workspace Policy (tRPC)"}
          </button>

          {trpcMessage && (
            <div className="result-card success-card">
              <span className="badge badge-success">tRPC Mutation Success</span>
              <p>{trpcMessage}</p>
            </div>
          )}

          {trpcError && (
            <div className="result-card conflict-card">
              <span className="badge badge-danger">tRPC Domain Error</span>
              <p><strong>Error:</strong> {trpcError}</p>
            </div>
          )}
        </section>
      </div>

      {/* =================================================================== */}
      {/* PANEL 3: Contract Drift & Runtime Boundary Simulator */}
      {/* =================================================================== */}
      <section className="panel-card drift-panel">
        <div className="panel-header">
          <span className="panel-icon">🛡️</span>
          <div>
            <h2>Contract Drift &amp; Defensive Runtime Boundary Guard</h2>
            <span className="tag-sub">Exercise 4 • Zod Boundary Parsing vs Compile-Time Types</span>
          </div>
        </div>

        <p className="drift-desc">
          TypeScript static types disappear after compilation. If a microservice field is renamed,
          untyped network boundaries crash the browser unless a defensive parser guards the perimeter.
        </p>

        <div className="drift-controls">
          <label><strong>Select Payload Scenario:</strong></label>
          <div className="radio-group">
            <label>
              <input
                type="radio"
                name="driftScenario"
                checked={driftSimulationMode === "VALID"}
                onChange={() => setDriftSimulationMode("VALID")}
              />
              Normal Valid Response
            </label>
            <label>
              <input
                type="radio"
                name="driftScenario"
                checked={driftSimulationMode === "RENAMED_FIELD"}
                onChange={() => setDriftSimulationMode("RENAMED_FIELD")}
              />
              Schema Drift: <code>total_cost</code> renamed to <code>overall_expense</code>
            </label>
            <label>
              <input
                type="radio"
                name="driftScenario"
                checked={driftSimulationMode === "TYPE_MISMATCH"}
                onChange={() => setDriftSimulationMode("TYPE_MISMATCH")}
              />
              Type Corruption: <code>horizon_days</code> sent as string
            </label>
            <label>
              <input
                type="radio"
                name="driftScenario"
                checked={driftSimulationMode === "MISSING_NESTED"}
                onChange={() => setDriftSimulationMode("MISSING_NESTED")}
              />
              Missing Array &amp; Bad Date: <code>assignments</code> omitted
            </label>
          </div>

          <button className="btn btn-warning" onClick={runDriftSimulation}>
            Run Defensive Boundary Guard Check
          </button>
        </div>

        {boundaryCheckResult && (
          <div
            className={`result-card ${
              boundaryCheckResult.success ? "success-card" : "error-card"
            }`}
          >
            {boundaryCheckResult.success ? (
              <div>
                <span className="badge badge-success">Boundary Guard PASSED</span>
                <p>
                  Payload validated in {boundaryCheckResult.validationDurationMs}ms. Safe to pass to React components.
                </p>
              </div>
            ) : (
              <div>
                <span className="badge badge-danger">
                  {boundaryCheckResult.driftDetected
                    ? "🚨 SCHEMA DRIFT DETECTED"
                    : "⚠️ VALIDATION ERROR"}
                </span>
                <p>The runtime guard intercepted an unannounced contract violation before it could crash the UI:</p>
                <ul>
                  {boundaryCheckResult.errorMessages.map((msg, i) => (
                    <li key={i}><code>{msg}</code></li>
                  ))}
                </ul>
                <details>
                  <summary>View Intercepted Raw JSON</summary>
                  <pre className="code-block">{boundaryCheckResult.rawPayloadSample}</pre>
                </details>
              </div>
            )}
          </div>
        )}
      </section>
    </div>
  );
};
