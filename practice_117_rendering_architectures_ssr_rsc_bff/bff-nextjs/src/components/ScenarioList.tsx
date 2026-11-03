import { getInternalScenarios } from "@/lib/api";

/**
 * Server Component listing active capacity planning scenarios.
 */
export default async function ScenarioList() {
  const scenarios = await getInternalScenarios();

  if (scenarios.length === 0) {
    return (
      <div style={{ padding: "2rem", textAlign: "center", color: "var(--text-secondary)" }}>
        No scenarios registered yet. Use the simulation form to propose one.
      </div>
    );
  }

  return (
    <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))", gap: "1rem" }}>
      {scenarios.map((scen) => (
        <div key={scen.id} className="card" style={{ background: "#131c2e", margin: 0 }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: "0.5rem" }}>
            <span style={{ fontWeight: 600, color: "#fff", fontSize: "1rem" }}>{scen.name}</span>
            <span className="badge badge-cyan">{scen.id}</span>
          </div>
          <div style={{ fontSize: "0.85rem", color: "var(--text-secondary)", marginBottom: "0.75rem" }}>
            {scen.notes || "No additional parameters provided."}
          </div>
          <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.85rem", borderTop: "1px solid var(--border-subtle)", paddingTop: "0.5rem" }}>
            <div>
              <span style={{ color: "var(--text-muted)" }}>Growth Target: </span>
              <strong style={{ color: "var(--accent-amber)" }}>+{scen.target_growth_pct}%</strong>
            </div>
            <div>
              <span style={{ color: "var(--text-muted)" }}>Budget: </span>
              <strong style={{ color: "var(--accent-green)" }}>${scen.allocated_budget_usd.toLocaleString()}</strong>
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}
