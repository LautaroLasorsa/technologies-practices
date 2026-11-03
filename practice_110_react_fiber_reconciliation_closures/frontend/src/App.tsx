import React, { useState, useEffect } from "react";
import { TaskItem } from "./types";
import { TaskQueueList } from "./components/TaskQueueList";
import { useTelemetryPoller } from "./hooks/useTelemetryPoller";
import { useSearchEndpoint } from "./hooks/useSearchEndpoint";
import { RenderCascadeInspector } from "./components/RenderCascadeInspector";

export const App: React.FC = () => {
  const [tasks, setTasks] = useState<TaskItem[]>([]);
  const [backendAvailable, setBackendAvailable] = useState<boolean | null>(null);
  const [alertThreshold, setAlertThreshold] = useState<number>(65);

  // Exercise 2: Telemetry Poller Hook
  const {
    currentMetric,
    history,
    alertTriggered,
    tickCount,
    clearAlert,
  } = useTelemetryPoller({
    pollingIntervalMs: 1200,
    alertThreshold,
  });

  // Exercise 3: Search Endpoint Hook
  const {
    query,
    setQuery,
    results: searchResults,
    isSearching,
    lastResolvedQuery,
    resolvedLatencyMs,
    hasRaceConditionOccurred,
    error: searchError,
  } = useSearchEndpoint("");

  // Fetch initial tasks from backend
  const fetchTasks = async () => {
    try {
      const res = await fetch("http://localhost:8110/api/tasks");
      if (res.ok) {
        const data = await res.json();
        setTasks(data);
        setBackendAvailable(true);
      } else {
        setBackendAvailable(false);
      }
    } catch {
      setBackendAvailable(false);
    }
  };

  useEffect(() => {
    fetchTasks();
  }, []);

  const handleResetBackend = async () => {
    try {
      await fetch("http://localhost:8110/api/reset", { method: "POST" });
      await fetchTasks();
    } catch (e) {
      console.error(e);
    }
  };

  return (
    <div style={{ maxWidth: "1200px", margin: "0 auto", padding: "24px 16px" }}>
      {/* Header */}
      <header
        style={{
          borderBottom: "1px solid #334155",
          paddingBottom: "16px",
          marginBottom: "24px",
          display: "flex",
          justifyContent: "space-between",
          alignItems: "flex-start",
        }}
      >
        <div>
          <h1 style={{ margin: 0, fontSize: "1.75rem", color: "#f8fafc" }}>
            Practice 110: React Mental Model
          </h1>
          <p style={{ margin: "6px 0 0 0", color: "#94a3b8", fontSize: "0.95rem" }}>
            Fiber Reconciliation, Stale Closure Traps & Async Boundary Invariants
          </p>
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
          <div
            style={{
              padding: "6px 12px",
              borderRadius: "6px",
              fontSize: "0.85rem",
              fontWeight: 600,
              backgroundColor: backendAvailable ? "#065f46" : "#7f1d1d",
              color: backendAvailable ? "#a7f3d0" : "#fecaca",
            }}
          >
            {backendAvailable === null
              ? "Connecting..."
              : backendAvailable
              ? "● Backend Connected (Port 8110)"
              : "○ Backend Offline (Run uv run uvicorn main:app)"}
          </div>
          <button
            onClick={handleResetBackend}
            style={{
              padding: "6px 14px",
              backgroundColor: "#334155",
              color: "#f1f5f9",
              border: "none",
              borderRadius: "6px",
              cursor: "pointer",
              fontSize: "0.85rem",
            }}
          >
            Reset Backend Data
          </button>
        </div>
      </header>

      {/* Grid Layout for the 4 Exercise Modules */}
      <div style={{ display: "flex", flexDirection: "column", gap: "28px" }}>
        {/* Module 1: Telemetry Poller (Exercise 2) */}
        <section
          style={{
            backgroundColor: "#1e293b",
            borderRadius: "10px",
            padding: "20px",
            border: "1px solid #334155",
          }}
        >
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "16px" }}>
            <div>
              <h2 style={{ margin: 0, fontSize: "1.2rem", color: "#f8fafc" }}>
                1. Fleet Telemetry Poller (Exercise 2: Stale Closures)
              </h2>
              <p style={{ margin: "4px 0 0 0", fontSize: "0.85rem", color: "#94a3b8" }}>
                Tests interval state accumulation and dynamic prop/state synchronization inside effects.
              </p>
            </div>
            <div style={{ fontSize: "0.85rem", color: "#94a3b8" }}>
              Total Ticks: <strong>{tickCount}</strong> | Rolling History Length:{" "}
              <strong style={{ color: history.length < 2 ? "#f87171" : "#4ade80" }}>
                {history.length} / 10
              </strong>
            </div>
          </div>

          {alertTriggered && (
            <div
              style={{
                backgroundColor: "#7f1d1d",
                border: "1px solid #dc2626",
                color: "#fecaca",
                padding: "10px 16px",
                borderRadius: "6px",
                marginBottom: "16px",
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
              }}
            >
              <span>
                ⚠️ <strong>HIGH LOAD ALERT:</strong> CPU exceeded alert threshold ({alertThreshold}%)!
              </span>
              <button
                onClick={clearAlert}
                style={{
                  backgroundColor: "#991b1b",
                  border: "none",
                  color: "#fff",
                  padding: "4px 10px",
                  borderRadius: "4px",
                  cursor: "pointer",
                }}
              >
                Dismiss
              </button>
            </div>
          )}

          {/* Metric Cards */}
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", gap: "12px", marginBottom: "16px" }}>
            <div style={{ backgroundColor: "#0f172a", padding: "12px", borderRadius: "8px" }}>
              <div style={{ fontSize: "0.75rem", color: "#94a3b8" }}>CPU USAGE</div>
              <div style={{ fontSize: "1.5rem", fontWeight: 700, color: (currentMetric?.cpu_usage ?? 0) > alertThreshold ? "#f87171" : "#38bdf8" }}>
                {currentMetric ? `${currentMetric.cpu_usage}%` : "--"}
              </div>
            </div>

            <div style={{ backgroundColor: "#0f172a", padding: "12px", borderRadius: "8px" }}>
              <div style={{ fontSize: "0.75rem", color: "#94a3b8" }}>MEMORY USAGE</div>
              <div style={{ fontSize: "1.5rem", fontWeight: 700, color: "#38bdf8" }}>
                {currentMetric ? `${currentMetric.memory_usage}%` : "--"}
              </div>
            </div>

            <div style={{ backgroundColor: "#0f172a", padding: "12px", borderRadius: "8px" }}>
              <div style={{ fontSize: "0.75rem", color: "#94a3b8" }}>ACTIVE WORKERS</div>
              <div style={{ fontSize: "1.5rem", fontWeight: 700, color: "#4ade80" }}>
                {currentMetric ? currentMetric.active_workers : "--"}
              </div>
            </div>

            <div style={{ backgroundColor: "#0f172a", padding: "12px", borderRadius: "8px" }}>
              <div style={{ fontSize: "0.75rem", color: "#94a3b8" }}>THROUGHPUT</div>
              <div style={{ fontSize: "1.5rem", fontWeight: 700, color: "#c084fc" }}>
                {currentMetric ? `${currentMetric.throughput_req_sec} req/s` : "--"}
              </div>
            </div>
          </div>

          {/* Slider for Alert Threshold */}
          <div style={{ backgroundColor: "#0f172a", padding: "12px 16px", borderRadius: "8px", display: "flex", alignItems: "center", gap: "16px" }}>
            <label style={{ fontSize: "0.85rem", color: "#e2e8f0", minWidth: "180px" }}>
              CPU Alert Threshold: <strong>{alertThreshold}%</strong>
            </label>
            <input
              type="range"
              min="40"
              max="95"
              value={alertThreshold}
              onChange={(e) => setAlertThreshold(Number(e.target.value))}
              style={{ flex: 1 }}
            />
            <span style={{ fontSize: "0.75rem", color: "#94a3b8" }}>
              (Move slider to verify whether interval callback reacts to new values or stays stale)
            </span>
          </div>
        </section>

        {/* Module 2: Priority Queue & Fiber Reconciliation (Exercise 1) */}
        <section
          style={{
            backgroundColor: "#1e293b",
            borderRadius: "10px",
            padding: "20px",
            border: "1px solid #334155",
          }}
        >
          <h2 style={{ margin: "0 0 12px 0", fontSize: "1.2rem", color: "#f8fafc" }}>
            2. Priority Queue (Exercise 1: Reconciliation & Key Semantics)
          </h2>
          <TaskQueueList tasks={tasks} onTasksChange={setTasks} />
        </section>

        {/* Module 3: Jittered Search & Async Race Conditions (Exercise 3) */}
        <section
          style={{
            backgroundColor: "#1e293b",
            borderRadius: "10px",
            padding: "20px",
            border: "1px solid #334155",
          }}
        >
          <div style={{ marginBottom: "16px" }}>
            <h2 style={{ margin: 0, fontSize: "1.2rem", color: "#f8fafc" }}>
              3. Fleet Task Search (Exercise 3: Async Race Conditions)
            </h2>
            <p style={{ margin: "4px 0 0 0", fontSize: "0.85rem", color: "#94a3b8" }}>
              Backend delays short queries (&le;2 chars: 1200ms) longer than long queries (&gt;2 chars: 80ms).
              Type quickly (e.g. &ldquo;drone&rdquo;) and observe if stale responses overwrite newer queries!
            </p>
          </div>

          {/* Search Input */}
          <div style={{ display: "flex", gap: "12px", alignItems: "center", marginBottom: "16px" }}>
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search tasks by title, priority, worker (e.g. 'drone')..."
              style={{
                flex: 1,
                padding: "10px 14px",
                fontSize: "1rem",
                borderRadius: "6px",
                border: "1px solid #475569",
                backgroundColor: "#0f172a",
                color: "#f8fafc",
              }}
            />
            {isSearching && (
              <span style={{ fontSize: "0.85rem", color: "#38bdf8" }}>Searching...</span>
            )}
          </div>

          {/* Race Condition Warning Banner */}
          {hasRaceConditionOccurred && (
            <div
              style={{
                backgroundColor: "#7f1d1d",
                border: "1px solid #ef4444",
                color: "#fecaca",
                padding: "12px 16px",
                borderRadius: "6px",
                marginBottom: "16px",
              }}
            >
              <strong>🚨 RACE CONDITION DETECTED!</strong>
              <div style={{ marginTop: "4px", fontSize: "0.85rem" }}>
                Current input query is <code>&quot;{query}&quot;</code>, but the active results
                belong to delayed query <code>&quot;{lastResolvedQuery}&quot;</code> (latency:{" "}
                {resolvedLatencyMs}ms). Stale HTTP response overwrote newer state!
              </div>
            </div>
          )}

          {searchError && (
            <div style={{ color: "#f87171", fontSize: "0.85rem", marginBottom: "12px" }}>
              Error: {searchError}
            </div>
          )}

          {/* Results Summary */}
          <div style={{ fontSize: "0.85rem", color: "#94a3b8", marginBottom: "12px" }}>
            Resolved Query: <code>&quot;{lastResolvedQuery || "(none)"}&quot;</code> | Matches:{" "}
            <strong>{searchResults.length}</strong> | Latency:{" "}
            {resolvedLatencyMs !== null ? `${resolvedLatencyMs}ms` : "--"}
          </div>

          {/* Results List */}
          <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
            {searchResults.map((item) => (
              <div
                key={item.id}
                style={{
                  padding: "10px 14px",
                  backgroundColor: "#0f172a",
                  borderRadius: "6px",
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "center",
                }}
              >
                <div>
                  <span style={{ fontWeight: 600, color: "#f1f5f9" }}>{item.title}</span>
                  <div style={{ fontSize: "0.8rem", color: "#64748b" }}>
                    Worker: {item.assigned_worker} | Priority: {item.priority}
                  </div>
                </div>
                <span
                  style={{
                    fontSize: "0.75rem",
                    padding: "3px 8px",
                    borderRadius: "4px",
                    backgroundColor: "#334155",
                    color: "#e2e8f0",
                  }}
                >
                  {item.status}
                </span>
              </div>
            ))}
            {query.trim() && !isSearching && searchResults.length === 0 && (
              <div style={{ color: "#64748b", fontSize: "0.9rem", padding: "8px" }}>
                No tasks match &quot;{query}&quot;.
              </div>
            )}
          </div>
        </section>

        {/* Module 4: Referential Identity & Render Cascades (Exercise 4) */}
        <section
          style={{
            backgroundColor: "#1e293b",
            borderRadius: "10px",
            padding: "20px",
            border: "1px solid #334155",
          }}
        >
          <h2 style={{ margin: "0 0 12px 0", fontSize: "1.2rem", color: "#f8fafc" }}>
            4. System Audit & Memoization (Exercise 4: Referential Identity)
          </h2>
          <RenderCascadeInspector parentHeartbeatTick={tickCount} />
        </section>
      </div>
    </div>
  );
};
