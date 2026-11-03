import Link from "next/link";

export default function HomePage() {
  return (
    <div>
      <div style={{ marginBottom: "2rem" }}>
        <h1 style={{ fontSize: "2.25rem", fontWeight: 800, color: "#fff", marginBottom: "0.5rem" }}>
          Next.js App Router as a BFF over Python
        </h1>
        <p style={{ color: "var(--text-secondary)", fontSize: "1.1rem" }}>
          Practice 117 — Architectural Patterns: CSR vs SSR, React Server Components (RSC), Suspense Streaming & Server Actions.
        </p>
      </div>

      <div className="card" style={{ borderLeft: "4px solid var(--accent-blue)" }}>
        <h2 className="card-title">System Architecture Overview</h2>
        <p style={{ color: "var(--text-secondary)", marginBottom: "1rem" }}>
          In modern enterprise web development, Next.js App Router functions as a high-performance <strong>Backend-for-Frontend (BFF)</strong>.
          It interfaces directly with the browser via React 19 Server Components and connects over a private internal network to heavy computational services written in Python (FastAPI, PyTorch, Celery).
        </p>
        <div style={{ background: "#090d16", padding: "1.25rem", borderRadius: "0.5rem", overflowX: "auto" }}>
          <pre style={{ color: "#38bdf8", fontFamily: "var(--font-mono)", fontSize: "0.85rem", lineHeight: "1.4" }}>
{`+-------------------------------------------------------------------------------+
| BROWSER (Client)                                                              |
| Instant HTML Stream -> Initial Paint -> RSC Flight Payload -> Hydration       |
+-------------------------------------------------------------------------------+
        |                                       ^
        | Public HTTPS (Port 3000)              | Next.js 15 App Router
        v                                       |
+-------------------------------------------------------------------------------+
| NEXT.JS BFF LAYER (Node.js Server Runtime)                                    |
| - Securely holds INTERNAL_SERVICE_KEY (never leaked to browser bundle)        |
| - Fast KPI Fetch (40ms) -> Flushes immediate HTML                             |
| - Slow ML Forecast (1200ms) -> Streamed asynchronously via <Suspense>         |
| - Server Actions ('use server') -> Authenticated mutations & revalidation     |
+-------------------------------------------------------------------------------+
        |                                       ^
        | Internal Private HTTP (Port 8117)     | Header: X-Internal-Service-Key
        v                                       |
+-------------------------------------------------------------------------------+
| FASTAPI INTERNAL MICROSERVICE (Python Service)                                |
| - Microservice on port 8117 (rejects unauthenticated browser requests)        |
| - Endpoints: /internal/kpis, /internal/forecast, /internal/scenarios           |
+-------------------------------------------------------------------------------+`}
          </pre>
        </div>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))", gap: "1.25rem", marginBottom: "2rem" }}>
        <div className="card">
          <h3 style={{ fontSize: "1.1rem", fontWeight: 600, color: "var(--accent-cyan)", marginBottom: "0.5rem" }}>
            1. Zero-Bundle RSC
          </h3>
          <p style={{ fontSize: "0.9rem", color: "var(--text-secondary)" }}>
            Server components execute strictly on the server and add <code>0 KB</code> to the client bundle. Dependencies like heavy database drivers or API clients are never downloaded by the browser.
          </p>
        </div>

        <div className="card">
          <h3 style={{ fontSize: "1.1rem", fontWeight: 600, color: "var(--accent-green)", marginBottom: "0.5rem" }}>
            2. Progressive Streaming
          </h3>
          <p style={{ fontSize: "0.9rem", color: "var(--text-secondary)" }}>
            Using React <code>&lt;Suspense&gt;</code>, the BFF sends the page shell and fast telemetry in 50ms, while long-running Python ML capacity forecasts stream in at 1200ms without blocking initial paint.
          </p>
        </div>

        <div className="card">
          <h3 style={{ fontSize: "1.1rem", fontWeight: 600, color: "var(--accent-amber)", marginBottom: "0.5rem" }}>
            3. Slot Composition Pattern
          </h3>
          <p style={{ fontSize: "0.9rem", color: "var(--text-secondary)" }}>
            Interactive Client shells (tabs, accordions) manage local state with <code>'use client'</code>, but receive server components as <code>children</code> so child data never gets compiled into client JS.
          </p>
        </div>

        <div className="card">
          <h3 style={{ fontSize: "1.1rem", fontWeight: 600, color: "var(--accent-rose)", marginBottom: "0.5rem" }}>
            4. Server Actions
          </h3>
          <p style={{ fontSize: "0.9rem", color: "var(--text-secondary)" }}>
            Declared with <code>'use server'</code>, Server Actions handle mutations directly from forms, validate data, talk to internal Python APIs with private keys, and trigger instant cache revalidation.
          </p>
        </div>
      </div>

      <div style={{ textAlign: "center", padding: "1.5rem" }}>
        <Link
          href="/dashboard"
          className="btn btn-primary"
          style={{ fontSize: "1.1rem", padding: "0.85rem 2rem" }}
        >
          Launch Cluster Capacity Dashboard →
        </Link>
      </div>
    </div>
  );
}
