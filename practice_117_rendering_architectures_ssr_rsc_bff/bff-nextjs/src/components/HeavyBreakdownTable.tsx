import { getServiceBreakdown } from "@/lib/api";

/**
 * Server Component representing a detailed data table.
 * Executes strictly on the server and contributes 0 KB of data-fetching or parsing code to the client bundle.
 */
export default async function HeavyBreakdownTable() {
  const services = await getServiceBreakdown();

  return (
    <div className="table-container">
      <table>
        <thead>
          <tr>
            <th>Microservice</th>
            <th>Replicas</th>
            <th>CPU Allocated</th>
            <th>Memory Allocated</th>
            <th>Throughput (RPS)</th>
            <th>p99 Latency</th>
            <th>Status</th>
          </tr>
        </thead>
        <tbody>
          {services.map((svc) => (
            <tr key={svc.name}>
              <td style={{ fontWeight: 600, color: "#fff" }}>{svc.name}</td>
              <td>{svc.replicas} pods</td>
              <td>{svc.cpu_cores} cores</td>
              <td>{svc.memory_gb} GB</td>
              <td>{svc.rps.toLocaleString()} rps</td>
              <td>{svc.p99_ms} ms</td>
              <td>
                <span className={svc.status.includes("WARNING") ? "badge badge-amber" : "badge badge-green"}>
                  {svc.status}
                </span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
