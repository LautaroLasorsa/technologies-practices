export interface TaskItem {
  id: string;
  title: string;
  priority: "low" | "medium" | "high" | "critical";
  status: "queued" | "in-progress" | "completed";
  assigned_worker: string;
  battery_level: number;
}

export interface TelemetryMetric {
  timestamp: number;
  cpu_usage: number;
  memory_usage: number;
  active_workers: number;
  throughput_req_sec: number;
}

export interface SearchResponse {
  query: string;
  results: TaskItem[];
  total: number;
  latency_ms: number;
  server_timestamp: number;
}

export interface AuditLogEntry {
  id: string;
  timestamp: string;
  event: string;
  severity: "info" | "warn" | "error";
}

export interface ThemeConfig {
  mode: "dark" | "light";
  density: "compact" | "normal";
}
