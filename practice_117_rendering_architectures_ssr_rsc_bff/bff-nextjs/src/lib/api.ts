/**
 * Internal API Client used exclusively on the Next.js server runtime (BFF).
 * Holds and transmits the private INTERNAL_SERVICE_KEY to the internal Python microservice.
 * This file is NEVER executed on or bundled to the browser.
 */

export interface ClusterKPIs {
  cluster_id: string;
  status: "HEALTHY" | "DEGRADED" | "CRITICAL";
  active_nodes: number;
  healthy_nodes: number;
  cpu_utilization_pct: number;
  memory_utilization_pct: number;
  throughput_rps: number;
  p99_latency_ms: number;
  active_alerts_count: number;
  timestamp: string;
}

export interface ForecastResult {
  model_name: string;
  forecast_horizon_days: number;
  projected_traffic_growth_pct: number;
  recommended_node_capacity: number;
  additional_nodes_required: number;
  estimated_monthly_cost_usd: number;
  risk_level: "LOW" | "MODERATE_SPIKE_RISK" | "HIGH_SATURATION_RISK";
  confidence_interval: [number, number];
  bottleneck_microservice: string;
  recommended_action: string;
  generated_at: string;
}

export interface ServiceMetric {
  name: string;
  replicas: number;
  cpu_cores: number;
  memory_gb: number;
  rps: number;
  p99_ms: number;
  status: string;
}

export interface Scenario {
  id: string;
  name: string;
  target_growth_pct: number;
  allocated_budget_usd: number;
  notes?: string;
  created_at: string;
}

const PYTHON_SERVICE_URL = process.env.PYTHON_SERVICE_URL || "http://127.0.0.1:8117";
const INTERNAL_SERVICE_KEY = process.env.INTERNAL_SERVICE_KEY || "super-secret-internal-key";

function getInternalHeaders(): HeadersInit {
  return {
    "Content-Type": "application/json",
    "X-Internal-Service-Key": INTERNAL_SERVICE_KEY,
  };
}

/**
 * Fetch operational telemetry from internal Python microservice (~40ms).
 */
export async function getInternalKPIs(): Promise<ClusterKPIs> {
  const res = await fetch(`${PYTHON_SERVICE_URL}/internal/kpis`, {
    headers: getInternalHeaders(),
    cache: "no-store", // Telemetry must always reflect current state
  });

  if (!res.ok) {
    throw new Error(`Failed to fetch internal KPIs: HTTP ${res.status} ${res.statusText}`);
  }
  return res.json();
}

/**
 * Fetch ML capacity projection from internal Python microservice (~1200ms latency).
 */
export async function getInternalForecast(): Promise<ForecastResult> {
  const res = await fetch(`${PYTHON_SERVICE_URL}/internal/forecast`, {
    headers: getInternalHeaders(),
    cache: "no-store", // Intensive simulation compute
  });

  if (!res.ok) {
    throw new Error(`Failed to fetch internal forecast: HTTP ${res.status} ${res.statusText}`);
  }
  return res.json();
}

/**
 * Fetch detailed microservice resource breakdown (~100ms).
 */
export async function getServiceBreakdown(): Promise<ServiceMetric[]> {
  const res = await fetch(`${PYTHON_SERVICE_URL}/internal/breakdown`, {
    headers: getInternalHeaders(),
    cache: "no-store",
  });

  if (!res.ok) {
    throw new Error(`Failed to fetch breakdown: HTTP ${res.status}`);
  }
  return res.json();
}

/**
 * Fetch registered capacity planning scenarios.
 */
export async function getInternalScenarios(): Promise<Scenario[]> {
  const res = await fetch(`${PYTHON_SERVICE_URL}/internal/scenarios`, {
    headers: getInternalHeaders(),
    next: { tags: ["scenarios"] },
  });

  if (!res.ok) {
    throw new Error(`Failed to fetch scenarios: HTTP ${res.status}`);
  }
  return res.json();
}
