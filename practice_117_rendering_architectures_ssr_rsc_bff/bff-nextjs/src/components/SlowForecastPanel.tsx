import { getInternalForecast } from "@/lib/api";

export default async function SlowForecastPanel() {
  // This call takes ~1200ms on the internal Python service to simulate ML capacity prediction
  const forecast = await getInternalForecast();

  const isModerateRisk = forecast.risk_level === "MODERATE_SPIKE_RISK";

  return (
    <div className="card" style={{ borderColor: isModerateRisk ? "rgba(245, 158, 11, 0.4)" : "var(--border-subtle)" }}>
      <div className="card-title">
        <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
          <span>AI / ML Capacity Forecast (90-Day Projection)</span>
          <span className="badge badge-cyan">Streamed via RSC &lt;Suspense&gt;</span>
        </div>
        <span className={isModerateRisk ? "badge badge-amber" : "badge badge-green"}>
          {forecast.risk_level}
        </span>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", gap: "1rem", marginBottom: "1rem" }}>
        <div style={{ background: "#131c2e", padding: "0.85rem", borderRadius: "0.375rem" }}>
          <div className="kpi-label">Projected Growth</div>
          <div style={{ fontSize: "1.35rem", fontWeight: 700, color: "var(--accent-amber)" }}>
            +{forecast.projected_traffic_growth_pct}%
          </div>
          <div className="kpi-subtext">95% CI: [{forecast.confidence_interval.join(", ")}%]</div>
        </div>

        <div style={{ background: "#131c2e", padding: "0.85rem", borderRadius: "0.375rem" }}>
          <div className="kpi-label">Recommended Capacity</div>
          <div style={{ fontSize: "1.35rem", fontWeight: 700, color: "#fff" }}>
            {forecast.recommended_node_capacity} Nodes
          </div>
          <div className="kpi-subtext">Requires +{forecast.additional_nodes_required} workers</div>
        </div>

        <div style={{ background: "#131c2e", padding: "0.85rem", borderRadius: "0.375rem" }}>
          <div className="kpi-label">Est. Monthly Delta</div>
          <div style={{ fontSize: "1.35rem", fontWeight: 700, color: "#fff" }}>
            ${forecast.estimated_monthly_cost_usd.toLocaleString()}
          </div>
          <div className="kpi-subtext">Compute & egress budget</div>
        </div>

        <div style={{ background: "#131c2e", padding: "0.85rem", borderRadius: "0.375rem" }}>
          <div className="kpi-label">Primary Bottleneck</div>
          <div style={{ fontSize: "1.1rem", fontWeight: 600, color: "var(--accent-rose)", marginTop: "0.25rem" }}>
            {forecast.bottleneck_microservice}
          </div>
          <div className="kpi-subtext">Model: {forecast.model_name}</div>
        </div>
      </div>

      <div className="callout" style={{ margin: 0, borderLeftColor: "var(--accent-amber)", background: "rgba(245, 158, 11, 0.08)" }}>
        <strong>Model Recommendation:</strong> {forecast.recommended_action}
      </div>
    </div>
  );
}

export function ForecastSkeleton() {
  return (
    <div className="card skeleton" style={{ minHeight: "220px", display: "flex", flexDirection: "column", justifyContent: "space-between" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <div style={{ width: "280px", height: "24px", background: "#475569", borderRadius: "4px" }} />
        <div style={{ width: "120px", height: "20px", background: "#475569", borderRadius: "4px" }} />
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", gap: "1rem", margin: "1.5rem 0" }}>
        {[1, 2, 3, 4].map((i) => (
          <div key={i} style={{ height: "65px", background: "#475569", borderRadius: "6px" }} />
        ))}
      </div>
      <div style={{ height: "38px", background: "#475569", borderRadius: "4px" }} />
    </div>
  );
}
