import { Suspense } from "react";
import FastKPIPanel, { FastKPISkeleton } from "@/components/FastKPIPanel";
import SlowForecastPanel, { ForecastSkeleton } from "@/components/SlowForecastPanel";
import HeavyBreakdownTable from "@/components/HeavyBreakdownTable";
import ScenarioList from "@/components/ScenarioList";
import ScenarioForm from "@/components/ScenarioForm";
import InteractiveScenarioShell from "@/components/InteractiveScenarioShell";
import LocalTimestampBadge from "@/components/LocalTimestampBadge";

// Always revalidate dynamically on request to inspect real-time streaming
export const dynamic = "force-dynamic";

/**
 * EXERCISE 1: Eliminate CSR Waterfall via Parallel RSC & Granular Streaming
 *
 * THE PROBLEM (Client-Side Rendering Waterfall):
 * In a traditional CSR single-page app, the browser undergoes a 3-step waterfall:
 * 1. Download HTML & JS bundle (~300ms)
 * 2. Mount root component -> trigger fetch('/api/user') & fetch('/api/kpis') (~200ms)
 * 3. Mount child components -> trigger fetch('/api/ml-forecast') (~1200ms)
 * Result: User sees a blank screen or spinner for ~1700ms before seeing any data!
 *
 * THE SOLUTION (Streaming Server Components with React <Suspense>):
 * In Next.js App Router:
 * 1. Fast Server Components (<FastKPIPanel />) execute on the server (~40ms) and are flushed
 *    to the browser immediately in the initial HTML stream.
 * 2. Slow Server Components (<SlowForecastPanel /> ~1200ms) are wrapped in <Suspense fallback={<ForecastSkeleton />}>.
 *    The server flushes the skeleton immediately with the initial HTML.
 * 3. Once the Python ML model finishes computing, Next.js streams the RSC Flight chunk over the open HTTP connection,
 *    and React swaps the skeleton with the real forecast without any client page reload!
 *
 * TODO(human) [Exercise 1]:
 * 1. Ensure <FastKPIPanel /> executes concurrently on the server.
 * 2. Wrap <SlowForecastPanel /> in a granular <Suspense fallback={<ForecastSkeleton />}> boundary.
 * 3. Observe the network tab in the browser (DevTools): Note how the document HTTP response remains 'pending'
 *    while chunk 1 (Fast KPIs) is rendered in ~50ms, followed by chunk 2 (Forecast) at ~1250ms.
 */
export default async function DashboardPage() {
  const currentTimestamp = new Date().toISOString();

  return (
    <div>
      {/* Dashboard Top Header */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "1.5rem" }}>
        <div>
          <h1 style={{ fontSize: "1.85rem", fontWeight: 700, color: "#fff" }}>
            Cluster Capacity & Operations
          </h1>
          <p style={{ color: "var(--text-secondary)", fontSize: "0.95rem" }}>
            Region: <code>us-east-1</code> | Primary Cluster: <code>k8s-prod-us-east-1</code>
          </p>
        </div>

        {/* EXERCISE 4: SSR Hydration Mismatch Exercise */}
        <LocalTimestampBadge utcIsoString={currentTimestamp} />
      </div>

      {/* SECTION 1: Fast Operational KPIs (RSC ~40ms) */}
      <section style={{ marginBottom: "1.5rem" }}>
        <h2 style={{ fontSize: "1rem", color: "var(--text-secondary)", marginBottom: "0.5rem", fontWeight: 600 }}>
          Live Operational Telemetry (Fast RSC)
        </h2>
        {/* FastKPIPanel resolves in ~40ms on the internal network */}
        <Suspense fallback={<FastKPISkeleton />}>
          <FastKPIPanel />
        </Suspense>
      </section>

      {/* SECTION 2: Slow ML Capacity Forecasting (RSC ~1200ms Streaming) */}
      <section style={{ marginBottom: "1.5rem" }}>
        {/*
          TODO(human) [Exercise 1]:
          Wrap <SlowForecastPanel /> in <Suspense fallback={<ForecastSkeleton />}>.
          Verify that the Fast KPIs render within 50ms while the Forecast skeleton pulses,
          followed by the streamed forecast payload ~1200ms later.
        */}
        <Suspense fallback={<ForecastSkeleton />}>
          <SlowForecastPanel />
        </Suspense>
      </section>

      {/* SECTION 3: Interactive Client Shell + RSC Slot Composition Pattern (Exercise 2) */}
      <section>
        <InteractiveScenarioShell
          slotBreakdown={<HeavyBreakdownTable />}
          slotScenarios={<ScenarioList />}
          slotForm={<ScenarioForm />}
        />
      </section>
    </div>
  );
}
