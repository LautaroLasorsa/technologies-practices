import asyncio
from datetime import datetime, timezone
import os
from typing import List, Optional
from fastapi import FastAPI, Header, HTTPException, status
from pydantic import BaseModel, Field

EXPECTED_SERVICE_KEY = os.getenv("INTERNAL_SERVICE_KEY", "super-secret-internal-key")

app = FastAPI(
    title="Internal Capacity Planning & Analytics Service",
    description="Backend microservice serving cluster KPIs, heavy forecasting models, and capacity scenarios. Accessible only via BFF.",
    version="1.0.0",
)


def verify_internal_key(x_internal_service_key: Optional[str] = Header(None)) -> str:
    """Validate internal service key to prevent direct browser exposure."""
    if not x_internal_service_key or x_internal_service_key != EXPECTED_SERVICE_KEY:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Forbidden: Missing or invalid X-Internal-Service-Key. Direct client access is not permitted; requests must traverse the Next.js BFF layer.",
        )
    return x_internal_service_key


class Scenario(BaseModel):
    id: str
    name: str
    target_growth_pct: float
    allocated_budget_usd: float
    notes: Optional[str] = None
    created_at: str


class ScenarioCreate(BaseModel):
    name: str = Field(..., min_length=3, max_length=100)
    target_growth_pct: float = Field(..., ge=0, le=500)
    allocated_budget_usd: float = Field(..., ge=100, le=1000000)
    notes: Optional[str] = None


# In-memory storage for capacity scenarios
_SCENARIO_DB: List[Scenario] = [
    Scenario(
        id="scen-01",
        name="Q3 Traffic Spike Anticipation",
        target_growth_pct=35.0,
        allocated_budget_usd=15000.0,
        notes="Scale out Redis caching cluster and add 12 worker replicas.",
        created_at="2026-10-15T12:00:00Z",
    ),
    Scenario(
        id="scen-02",
        name="EU Region Redundancy Expansion",
        target_growth_pct=50.0,
        allocated_budget_usd=28000.0,
        notes="Provision multi-region failover pool in eu-west-1.",
        created_at="2026-10-28T09:30:00Z",
    ),
]


@app.get("/health")
async def health_check():
    """Unauthenticated health probe."""
    return {"status": "ok", "service": "python-service"}


@app.get("/internal/kpis")
async def get_internal_kpis(x_key: str = Header(..., alias="X-Internal-Service-Key")):
    """Fast operational cluster telemetry (~40ms). Requires internal key."""
    verify_internal_key(x_key)
    # Simulate lightweight database / cache lookup
    await asyncio.sleep(0.04)

    return {
        "cluster_id": "k8s-prod-us-east-1",
        "status": "HEALTHY",
        "active_nodes": 48,
        "healthy_nodes": 48,
        "cpu_utilization_pct": 64.2,
        "memory_utilization_pct": 71.8,
        "throughput_rps": 18450,
        "p99_latency_ms": 14.2,
        "active_alerts_count": 0,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/internal/forecast")
async def get_internal_forecast(x_key: str = Header(..., alias="X-Internal-Service-Key")):
    """Intensive capacity forecasting model (~1200ms latency). Requires internal key."""
    verify_internal_key(x_key)
    # Simulate intensive ML model inference or simulation calculation
    await asyncio.sleep(1.2)

    return {
        "model_name": "Prophet-Capacity-Forecaster-v4",
        "forecast_horizon_days": 90,
        "projected_traffic_growth_pct": 38.5,
        "recommended_node_capacity": 64,
        "additional_nodes_required": 16,
        "estimated_monthly_cost_usd": 13440.0,
        "risk_level": "MODERATE_SPIKE_RISK",
        "confidence_interval": [33.8, 43.2],
        "bottleneck_microservice": "payment-checkout-worker",
        "recommended_action": "Provision 16 c6i.4xlarge compute instances before week 38 to absorb projected holiday volume.",
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/internal/breakdown")
async def get_service_breakdown(x_key: str = Header(..., alias="X-Internal-Service-Key")):
    """Detailed microservice resource consumption breakdown (~100ms). Requires internal key."""
    verify_internal_key(x_key)
    await asyncio.sleep(0.1)

    return [
        {
            "name": "gateway-api",
            "replicas": 12,
            "cpu_cores": 24,
            "memory_gb": 48,
            "rps": 18450,
            "p99_ms": 4.1,
            "status": "HEALTHY",
        },
        {
            "name": "auth-service",
            "replicas": 8,
            "cpu_cores": 16,
            "memory_gb": 32,
            "rps": 14200,
            "p99_ms": 6.8,
            "status": "HEALTHY",
        },
        {
            "name": "payment-checkout-worker",
            "replicas": 16,
            "cpu_cores": 64,
            "memory_gb": 128,
            "rps": 3800,
            "p99_ms": 28.5,
            "status": "WARNING_HIGH_LOAD",
        },
        {
            "name": "analytics-ingestion",
            "replicas": 12,
            "cpu_cores": 48,
            "memory_gb": 96,
            "rps": 9200,
            "p99_ms": 11.2,
            "status": "HEALTHY",
        },
    ]


@app.get("/internal/scenarios")
async def list_scenarios(x_key: str = Header(..., alias="X-Internal-Service-Key")):
    """List all registered capacity planning scenarios. Requires internal key."""
    verify_internal_key(x_key)
    await asyncio.sleep(0.05)
    return _SCENARIO_DB


@app.post("/internal/scenarios", status_code=status.HTTP_201_CREATED)
async def create_scenario(
    payload: ScenarioCreate,
    x_key: str = Header(..., alias="X-Internal-Service-Key"),
):
    """Mutate capacity planning scenario database. Requires internal key."""
    verify_internal_key(x_key)
    await asyncio.sleep(0.08)

    new_scenario = Scenario(
        id=f"scen-0{len(_SCENARIO_DB) + 1}",
        name=payload.name,
        target_growth_pct=payload.target_growth_pct,
        allocated_budget_usd=payload.allocated_budget_usd,
        notes=payload.notes,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    _SCENARIO_DB.append(new_scenario)
    return new_scenario


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8117, reload=True)
