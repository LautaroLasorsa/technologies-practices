import React, { useState, useRef } from "react";
import { AuditLogEntry, ThemeConfig } from "../types";

const INITIAL_LOGS: AuditLogEntry[] = [
  { id: "log-1", timestamp: "12:00:01", event: "Worker unit-sky-01 dispatched to Zone A", severity: "info" },
  { id: "log-2", timestamp: "12:00:15", event: "Battery threshold warning for unit-depot-09 (19%)", severity: "warn" },
  { id: "log-3", timestamp: "12:00:32", event: "Station Epsilon mesh synchronization completed", severity: "info" },
  { id: "log-4", timestamp: "12:00:58", event: "Sensor ping dropped on repeater node 3", severity: "error" },
];

interface SystemAuditPanelProps {
  themeConfig: ThemeConfig;
  logs: AuditLogEntry[];
  onExport: (format: "json" | "csv") => void;
}

/**
 * SystemAuditPanel is wrapped with React.memo to prevent re-renders when props do not change.
 * It tracks and displays its own render count.
 */
const SystemAuditPanel: React.FC<SystemAuditPanelProps> = React.memo(
  ({ themeConfig, logs, onExport }) => {
    const renderCounter = useRef(0);
    renderCounter.current += 1;

    return (
      <div
        style={{
          marginTop: "16px",
          padding: "16px",
          borderRadius: "8px",
          backgroundColor: themeConfig.mode === "dark" ? "#1e293b" : "#f1f5f9",
          border: "1px solid #334155",
        }}
      >
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <div>
            <h4 style={{ margin: 0, color: "#f8fafc" }}>Memoized System Audit Panel</h4>
            <span style={{ fontSize: "0.8rem", color: "#94a3b8" }}>
              Theme: {themeConfig.mode} | Density: {themeConfig.density}
            </span>
          </div>

          <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
            <span
              style={{
                padding: "4px 8px",
                borderRadius: "9999px",
                fontSize: "0.8rem",
                fontWeight: 700,
                backgroundColor: renderCounter.current > 3 ? "#ef4444" : "#10b981",
                color: "#ffffff",
              }}
            >
              Child Renders: {renderCounter.current}
            </span>

            <button
              onClick={() => onExport("json")}
              style={{
                padding: "6px 12px",
                backgroundColor: "#475569",
                color: "#ffffff",
                border: "none",
                borderRadius: "4px",
                cursor: "pointer",
                fontSize: "0.8rem",
              }}
            >
              Export JSON
            </button>
          </div>
        </div>

        <ul style={{ listStyle: "none", padding: 0, marginTop: "12px", fontSize: "0.85rem" }}>
          {logs.map((log) => (
            <li
              key={log.id}
              style={{
                padding: "6px 8px",
                borderBottom: "1px solid #334155",
                display: "flex",
                gap: "8px",
              }}
            >
              <code style={{ color: "#38bdf8" }}>[{log.timestamp}]</code>
              <span style={{ color: log.severity === "error" ? "#f87171" : log.severity === "warn" ? "#fde047" : "#e2e8f0" }}>
                {log.event}
              </span>
            </li>
          ))}
        </ul>
      </div>
    );
  }
);

interface RenderCascadeInspectorProps {
  parentHeartbeatTick: number; // Ticks every 1000ms from telemetry
}

export const RenderCascadeInspector: React.FC<RenderCascadeInspectorProps> = ({
  parentHeartbeatTick,
}) => {
  const [logs] = useState<AuditLogEntry[]>(INITIAL_LOGS);
  const [exportNotice, setExportNotice] = useState<string | null>(null);

  /**
   * FLAGGED BEHAVIOR (Exercise 4 - Referential Identity Breakdown):
   *
   * Every time `parentHeartbeatTick` changes in the parent component, this component re-renders.
   *
   * Look at `themeConfig` and `onExport` passed to `<SystemAuditPanel />` below:
   * 1. `themeConfig` is created as an inline object literal: `{ mode: "dark", density: "compact" }`.
   *    In JavaScript: `{} !== {}`. A brand-new reference is created on every render!
   * 2. `onExport` is an inline arrow function: `(fmt) => { ... }`.
   *    A brand-new function pointer is created on every render!
   *
   * Even though `SystemAuditPanel` is wrapped in `React.memo`, `Object.is` comparison fails
   * because `prevProps.themeConfig !== nextProps.themeConfig` and `prevProps.onExport !== nextProps.onExport`.
   * Result: The child component renders on EVERY SINGLE parent tick!
   */

  // TODO(human): Exercise 4
  //
  // 1. Stabilize `themeConfig` so its reference does not change across renders:
  //    const themeConfig = useMemo(() => ({ mode: "dark" as const, density: "compact" as const }), []);
  //    (or extract it as a static module-level constant if it never depends on props).
  //
  // 2. Stabilize `handleExport` using `useCallback`:
  //    const handleExport = useCallback((format: "json" | "csv") => {
  //      setExportNotice(`Exported ${logs.length} logs in ${format.toUpperCase()} format.`);
  //    }, [logs.length]);
  //
  // 3. Pass the stabilized references to `<SystemAuditPanel themeConfig={themeConfig} onExport={handleExport} />`.
  //
  // Verification:
  // Once stabilized, watch the "Child Renders" badge in the child panel: it should STAY AT 1
  // even while the parent heartbeat tick continues to increment!

  // --- BUGGY REFERENCES (BREAKS REACT.MEMO) ---
  const buggyThemeConfig: ThemeConfig = { mode: "dark", density: "compact" };
  const buggyExportHandler = (format: "json" | "csv") => {
    setExportNotice(`Exported ${logs.length} logs in ${format.toUpperCase()} format.`);
  };
  // --------------------------------------------

  return (
    <div style={{ backgroundColor: "#0f172a", padding: "16px", borderRadius: "8px" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <div>
          <h3 style={{ margin: 0, color: "#f8fafc" }}>Referential Identity & Render Cascade</h3>
          <p style={{ margin: "4px 0 0 0", fontSize: "0.85rem", color: "#94a3b8" }}>
            Parent re-renders on telemetry ticks. Observe whether the memoized child can bail out.
          </p>
        </div>
        <div style={{ textAlign: "right" }}>
          <span style={{ fontSize: "0.85rem", color: "#e2e8f0" }}>
            Parent Heartbeat: <strong>#{parentHeartbeatTick}</strong>
          </span>
        </div>
      </div>

      {exportNotice && (
        <div
          style={{
            marginTop: "8px",
            padding: "8px 12px",
            backgroundColor: "#065f46",
            color: "#a7f3d0",
            borderRadius: "4px",
            fontSize: "0.85rem",
          }}
        >
          {exportNotice}
        </div>
      )}

      {/* Memoized child receiving props */}
      <SystemAuditPanel
        themeConfig={buggyThemeConfig}
        logs={logs}
        onExport={buggyExportHandler}
      />
    </div>
  );
};
