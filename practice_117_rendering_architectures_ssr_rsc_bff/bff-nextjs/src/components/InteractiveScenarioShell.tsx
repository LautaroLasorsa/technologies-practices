"use client";

import { useState } from "react";

interface InteractiveScenarioShellProps {
  /**
   * EXERCISE 2: RSC Slot / Children Composition Pattern
   *
   * slotBreakdown and slotScenarios are React Server Components passed as JSX slots.
   * They execute exclusively on the server and stream serialized RSC Flight markup.
   */
  slotBreakdown: React.ReactNode;
  slotScenarios: React.ReactNode;
  slotForm: React.ReactNode;
}

type TabKey = "breakdown" | "scenarios" | "new";

/**
 * EXERCISE 2: RSC Slot / Children Composition Pattern
 *
 * THE PROBLEM:
 * Interactive UI components (tabs, accordions, modals, dropdowns) require React state (`useState`),
 * so they MUST be marked with `'use client'`.
 *
 * HOWEVER:
 * If a Client Component directly imports a Server Component (e.g. `import HeavyBreakdownTable from './HeavyBreakdownTable'`),
 * Next.js and React treat the imported component as a Client Component too!
 * This forces all data-fetching code, data schemas, and helper libraries into the client bundle, bloating JS size
 * and losing the 0-KB benefit of React Server Components.
 *
 * THE SOLUTION:
 * Use the **Slot / Children Composition Pattern**.
 * The parent Server Component (`dashboard/page.tsx`) renders `<HeavyBreakdownTable />` and passes it as a prop
 * (`children` or named slots like `slotBreakdown`) to this Client Component shell.
 *
 * TODO(human) [Exercise 2]:
 * 1. Maintain the interactive tab selection state using `useState<TabKey>('breakdown')`.
 * 2. Conditionally project the corresponding server-rendered slot (`slotBreakdown`, `slotScenarios`, or `slotForm`)
 *    into the main view container based on the active tab.
 * 3. Add an interactive toggle (e.g. collapsed / expanded or refresh indicator) without causing child server
 *    components to re-fetch on the server or leak into the client JS bundle.
 */
export default function InteractiveScenarioShell({
  slotBreakdown,
  slotScenarios,
  slotForm,
}: InteractiveScenarioShellProps) {
  const [activeTab, setActiveTab] = useState<TabKey>("breakdown");
  const [isExpanded, setIsExpanded] = useState<boolean>(true);

  return (
    <div className="card">
      <div className="card-title">
        <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
          <span>Capacity Planning & Cluster Subsystems</span>
          <span className="badge badge-cyan">Client Shell + RSC Slot Pattern</span>
        </div>
        <button
          onClick={() => setIsExpanded(!isExpanded)}
          className="btn btn-secondary"
          style={{ padding: "0.3rem 0.75rem", fontSize: "0.8rem" }}
        >
          {isExpanded ? "Collapse Section ▲" : "Expand Section ▼"}
        </button>
      </div>

      {isExpanded && (
        <>
          <div className="tabs-header">
            <button
              className={`tab-btn ${activeTab === "breakdown" ? "active" : ""}`}
              onClick={() => setActiveTab("breakdown")}
            >
              Microservice Breakdown (Server Slot)
            </button>
            <button
              className={`tab-btn ${activeTab === "scenarios" ? "active" : ""}`}
              onClick={() => setActiveTab("scenarios")}
            >
              Registered Scenarios (Server Slot)
            </button>
            <button
              className={`tab-btn ${activeTab === "new" ? "active" : ""}`}
              onClick={() => setActiveTab("new")}
            >
              + Propose New Scenario (Client Form + Server Action)
            </button>
          </div>

          <div style={{ minHeight: "220px", marginTop: "1rem" }}>
            {/*
              TODO(human) [Exercise 2]:
              Render the active slot according to `activeTab`:
              - If activeTab === 'breakdown', render {slotBreakdown}
              - If activeTab === 'scenarios', render {slotScenarios}
              - If activeTab === 'new', render {slotForm}

              Observe how slotBreakdown and slotScenarios are server-rendered without
              being bundled into the client JS code!
            */}
            {activeTab === "breakdown" && slotBreakdown}
            {activeTab === "scenarios" && slotScenarios}
            {activeTab === "new" && slotForm}
          </div>
        </>
      )}
    </div>
  );
}
