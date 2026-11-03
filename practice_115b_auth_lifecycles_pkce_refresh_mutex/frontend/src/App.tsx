import React, { useState, useEffect } from 'react';
import { useAuthSession } from './auth/useAuthSession.js';
import {
  authenticatedFetch,
  subscribeToTrace,
  AuthTraceEvent,
} from './auth/authClient.js';

interface WidgetState {
  name: string;
  endpoint: string;
  status: 'idle' | 'loading' | 'success' | 'error';
  statusCode: number | null;
  data: any;
  durationMs: number | null;
}

const INITIAL_WIDGETS: WidgetState[] = [
  { name: 'Analytics', endpoint: '/api/widgets/analytics', status: 'idle', statusCode: null, data: null, durationMs: null },
  { name: 'User Profile', endpoint: '/api/widgets/profile', status: 'idle', statusCode: null, data: null, durationMs: null },
  { name: 'Notifications', endpoint: '/api/widgets/notifications', status: 'idle', statusCode: null, data: null, durationMs: null },
  { name: 'Recent Activity', endpoint: '/api/widgets/activity', status: 'idle', statusCode: null, data: null, durationMs: null },
  { name: 'Billing & Plan', endpoint: '/api/widgets/billing', status: 'idle', statusCode: null, data: null, durationMs: null },
];

export default function App() {
  const { session, loginWithPkce, logout, triggerSimulatedReplay } = useAuthSession();
  const [widgets, setWidgets] = useState<WidgetState[]>(INITIAL_WIDGETS);
  const [traces, setTraces] = useState<AuthTraceEvent[]>([]);
  const [isFiring, setIsFiring] = useState(false);

  // Subscribe to real-time auth traces
  useEffect(() => {
    return subscribeToTrace((event) => {
      setTraces((prev) => [event, ...prev.slice(0, 49)]); // keep latest 50
    });
  }, []);

  // Fire all 5 widgets concurrently to test the Single-Flight Mutex
  const fireConcurrentRequests = async () => {
    setIsFiring(true);
    const startTime = performance.now();

    // Mark all as loading
    setWidgets((prev) =>
      prev.map((w) => ({ ...w, status: 'loading', statusCode: null, durationMs: null }))
    );

    const promises = INITIAL_WIDGETS.map(async (widget, index) => {
      const widgetStart = performance.now();
      try {
        const res = await authenticatedFetch(widget.endpoint);
        const duration = Math.round(performance.now() - widgetStart);
        const data = await res.json().catch(() => null);

        setWidgets((prev) => {
          const updated = [...prev];
          updated[index] = {
            ...updated[index],
            status: res.ok ? 'success' : 'error',
            statusCode: res.status,
            data,
            durationMs: duration,
          };
          return updated;
        });
      } catch (err: any) {
        setWidgets((prev) => {
          const updated = [...prev];
          updated[index] = {
            ...updated[index],
            status: 'error',
            statusCode: 500,
            data: { error: err.message },
            durationMs: Math.round(performance.now() - widgetStart),
          };
          return updated;
        });
      }
    });

    await Promise.all(promises);
    setIsFiring(false);
  };

  const getSourceBadgeColor = (source: AuthTraceEvent['source']) => {
    switch (source) {
      case 'PKCE': return '#58a6ff';
      case 'MUTEX': return '#d29922';
      case 'REFRESH': return '#bc8cff';
      case 'RETRY': return '#3fb950';
      case 'REVOKED': return '#f85149';
      default: return '#8b949e';
    }
  };

  return (
    <div style={{ maxWidth: '1200px', margin: '0 auto', padding: '24px' }}>
      {/* Header */}
      <header style={{ marginBottom: '24px', borderBottom: '1px solid var(--border-color)', paddingBottom: '16px' }}>
        <h1 style={{ margin: '0 0 8px 0', color: 'var(--text-heading)' }}>
          🔐 Practice 115b: Auth Lifecycles & Session Mutex
        </h1>
        <p style={{ margin: 0, color: '#8b949e', fontSize: '15px' }}>
          RFC 7636 PKCE · Rotating Refresh Tokens · In-Memory Token Closures · Single-Flight Concurrent 401 Mutex
        </p>
      </header>

      {/* Auth Status Banner */}
      <div
        style={{
          display: 'flex',
          flexWrap: 'wrap',
          alignItems: 'center',
          justifyContent: 'space-between',
          gap: '16px',
          padding: '16px 20px',
          backgroundColor: 'var(--bg-card)',
          borderRadius: '8px',
          border: '1px solid var(--border-color)',
          marginBottom: '24px',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
          <div>
            <span style={{ fontSize: '12px', color: '#8b949e', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
              Session State
            </span>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginTop: '4px' }}>
              <span
                style={{
                  display: 'inline-block',
                  width: '10px',
                  height: '10px',
                  borderRadius: '50%',
                  backgroundColor:
                    session.status === 'authenticated'
                      ? 'var(--accent-green)'
                      : session.status === 'revoked'
                      ? 'var(--accent-red)'
                      : '#8b949e',
                }}
              />
              <strong style={{ fontSize: '16px', textTransform: 'capitalize' }}>
                {session.status === 'authenticated'
                  ? `Authenticated (${session.userId})`
                  : session.status === 'revoked'
                  ? 'Revoked (Security Breach)'
                  : 'Logged Out'}
              </strong>
            </div>
          </div>

          {session.isAuthenticated && (
            <div style={{ borderLeft: '1px solid var(--border-color)', paddingLeft: '16px' }}>
              <span style={{ fontSize: '12px', color: '#8b949e', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                In-Memory Access Token Expiry
              </span>
              <div style={{ marginTop: '4px' }}>
                <span
                  style={{
                    fontWeight: 'bold',
                    fontSize: '16px',
                    color: session.tokenExpiresIn <= 3 ? 'var(--accent-red)' : 'var(--accent-green)',
                  }}
                >
                  {session.tokenExpiresIn}s remaining
                </span>
                <span style={{ fontSize: '12px', color: '#8b949e', marginLeft: '6px' }}>
                  (Short-lived for lab demo)
                </span>
              </div>
            </div>
          )}
        </div>

        {/* Auth Actions */}
        <div style={{ display: 'flex', gap: '10px' }}>
          {!session.isAuthenticated ? (
            <button
              onClick={loginWithPkce}
              style={{
                backgroundColor: 'var(--accent-green)',
                color: '#fff',
                border: 'none',
                padding: '10px 18px',
                borderRadius: '6px',
                fontWeight: 600,
                cursor: 'pointer',
              }}
            >
              🔑 Login via PKCE (RFC 7636)
            </button>
          ) : (
            <button
              onClick={logout}
              style={{
                backgroundColor: '#21262d',
                color: 'var(--text-primary)',
                border: '1px solid var(--border-color)',
                padding: '10px 18px',
                borderRadius: '6px',
                fontWeight: 600,
                cursor: 'pointer',
              }}
            >
              Logout (Cross-Tab Sync)
            </button>
          )}

          <button
            onClick={() => window.open(window.location.href, '_blank')}
            style={{
              backgroundColor: '#21262d',
              color: 'var(--text-primary)',
              border: '1px solid var(--border-color)',
              padding: '10px 14px',
              borderRadius: '6px',
              fontSize: '13px',
              cursor: 'pointer',
            }}
          >
            🪟 Open Duplicate Tab (Test BroadcastChannel)
          </button>
        </div>
      </div>

      {/* Control Actions & Simulation */}
      <div style={{ display: 'flex', gap: '12px', marginBottom: '24px', flexWrap: 'wrap' }}>
        <button
          onClick={fireConcurrentRequests}
          disabled={!session.isAuthenticated || isFiring}
          style={{
            backgroundColor: 'var(--accent-blue)',
            color: '#fff',
            border: 'none',
            padding: '12px 20px',
            borderRadius: '6px',
            fontWeight: 600,
            cursor: (!session.isAuthenticated || isFiring) ? 'not-allowed' : 'pointer',
            opacity: (!session.isAuthenticated || isFiring) ? 0.6 : 1,
          }}
        >
          {isFiring ? '⏳ Firing Concurrent Requests...' : '⚡ Fire 5 Concurrent Widget Calls (Test Mutex)'}
        </button>

        <button
          onClick={triggerSimulatedReplay}
          style={{
            backgroundColor: '#21262d',
            color: 'var(--accent-red)',
            border: '1px solid var(--accent-red)',
            padding: '12px 18px',
            borderRadius: '6px',
            fontWeight: 600,
            cursor: 'pointer',
          }}
        >
          🦹 Simulate Stolen Refresh Token Replay (Attack)
        </button>

        <button
          onClick={() => setTraces([])}
          style={{
            backgroundColor: 'transparent',
            color: '#8b949e',
            border: '1px solid var(--border-color)',
            padding: '12px 14px',
            borderRadius: '6px',
            cursor: 'pointer',
            marginLeft: 'auto',
          }}
        >
          Clear Trace Log
        </button>
      </div>

      {/* Main Grid: Widgets on Left, Trace Log on Right */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>
        {/* Protected Widgets Grid */}
        <div>
          <h2 style={{ fontSize: '18px', margin: '0 0 12px 0', color: 'var(--text-heading)' }}>
            📊 Protected Dashboard Widgets
          </h2>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
            {widgets.map((w) => (
              <div
                key={w.name}
                style={{
                  backgroundColor: 'var(--bg-card)',
                  borderRadius: '6px',
                  border: '1px solid var(--border-color)',
                  padding: '14px 16px',
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <strong style={{ color: 'var(--text-heading)' }}>{w.name}</strong>
                  <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
                    {w.durationMs !== null && (
                      <span style={{ fontSize: '11px', color: '#8b949e' }}>
                        {w.durationMs}ms
                      </span>
                    )}
                    <span
                      style={{
                        fontSize: '12px',
                        padding: '2px 8px',
                        borderRadius: '4px',
                        fontWeight: 600,
                        backgroundColor:
                          w.status === 'success'
                            ? 'rgba(63, 185, 80, 0.2)'
                            : w.status === 'error'
                            ? 'rgba(248, 81, 73, 0.2)'
                            : '#21262d',
                        color:
                          w.status === 'success'
                            ? 'var(--accent-green)'
                            : w.status === 'error'
                            ? 'var(--accent-red)'
                            : '#8b949e',
                      }}
                    >
                      {w.status === 'loading'
                        ? 'Fetching...'
                        : w.statusCode
                        ? `${w.statusCode}`
                        : 'Idle'}
                    </span>
                  </div>
                </div>

                <div style={{ marginTop: '8px', fontSize: '12px', fontFamily: 'monospace', color: '#8b949e' }}>
                  Endpoint: {w.endpoint}
                </div>

                {w.data && (
                  <pre
                    style={{
                      margin: '8px 0 0 0',
                      padding: '8px',
                      backgroundColor: '#0d1117',
                      borderRadius: '4px',
                      fontSize: '11px',
                      overflowX: 'auto',
                    }}
                  >
                    {JSON.stringify(w.data, null, 2)}
                  </pre>
                )}
              </div>
            ))}
          </div>
        </div>

        {/* Live Trace & Security Audit Log */}
        <div>
          <h2 style={{ fontSize: '18px', margin: '0 0 12px 0', color: 'var(--text-heading)' }}>
            📡 Live Auth & Mutex Trace Stream
          </h2>
          <div
            style={{
              backgroundColor: '#090d13',
              borderRadius: '6px',
              border: '1px solid var(--border-color)',
              height: '580px',
              overflowY: 'auto',
              padding: '12px',
              fontFamily: 'monospace',
              fontSize: '12px',
            }}
          >
            {traces.length === 0 ? (
              <div style={{ color: '#8b949e', textAlign: 'center', marginTop: '40px' }}>
                No events recorded yet.<br />
                Click <strong>"Login via PKCE"</strong> or <strong>"Fire 5 Concurrent Widget Calls"</strong> to observe the mutex.
              </div>
            ) : (
              traces.map((t) => (
                <div
                  key={t.id}
                  style={{
                    padding: '6px 0',
                    borderBottom: '1px solid #1f242c',
                    display: 'flex',
                    flexDirection: 'column',
                    gap: '2px',
                  }}
                >
                  <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
                    <span style={{ color: '#8b949e', fontSize: '10px' }}>{t.timestamp}</span>
                    <span
                      style={{
                        padding: '1px 6px',
                        borderRadius: '3px',
                        fontSize: '10px',
                        fontWeight: 700,
                        backgroundColor: `${getSourceBadgeColor(t.source)}22`,
                        color: getSourceBadgeColor(t.source),
                      }}
                    >
                      {t.source}
                    </span>
                    <span style={{ color: 'var(--text-primary)' }}>{t.message}</span>
                  </div>
                  {t.details && (
                    <pre
                      style={{
                        margin: '4px 0 0 20px',
                        padding: '4px',
                        backgroundColor: '#161b22',
                        borderRadius: '4px',
                        fontSize: '10px',
                        color: '#8b949e',
                        overflowX: 'auto',
                      }}
                    >
                      {JSON.stringify(t.details, null, 2)}
                    </pre>
                  )}
                </div>
              ))
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
