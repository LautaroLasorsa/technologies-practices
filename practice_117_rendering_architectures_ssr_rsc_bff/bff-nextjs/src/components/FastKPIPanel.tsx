import { getInternalKPIs } from "@/lib/api";

export default async function FastKPIPanel() {
  const kpis = await getInternalKPIs();

  return (
    <div className="grid-cols-4">
      <div className="kpi-tile">
        <div className="kpi-label">Active Cluster Nodes</div>
        <div className="kpi-value">{kpis.active_nodes} / {kpis.healthy_nodes}</div>
        <div className="kpi-subtext">
          <span className="badge badge-green">● Healthy</span>
        </div>
      </div>

      <div className="kpi-tile">
        <div className="kpi-label">Total Throughput</div>
        <div className="kpi-value">{(kpis.throughput_rps / 1000).toFixed(1)}k RPS</div>
        <div className="kpi-subtext">Inbound HTTP / gRPC traffic</div>
      </div>

      <div className="kpi-tile">
        <div className="kpi-label">CPU Utilization</div>
        <div className="kpi-value">{kpis.cpu_utilization_pct.toFixed(1)}%</div>
        <div className="kpi-subtext">Across {kpis.active_nodes} compute workers</div>
      </div>

      <div className="kpi-tile">
        <div className="kpi-label">p99 Gateway Latency</div>
        <div className="kpi-value">{kpis.p99_latency_ms.toFixed(1)} ms</div>
        <div className="kpi-subtext">SLA Target &lt; 25.0 ms</div>
      </div>
    </div>
  );
}

export function FastKPISkeleton() {
  return (
    <div className="grid-cols-4">
      {[1, 2, 3, 4].map((i) => (
        <div key={i} className="kpi-tile skeleton" style={{ height: "105px" }} />
      ))}
    </div>
  );
}
