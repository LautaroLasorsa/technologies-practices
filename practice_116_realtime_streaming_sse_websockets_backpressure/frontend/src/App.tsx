import { useState, useEffect, useRef, useMemo, useCallback } from 'react';
import type { TelemetryPacket, IngestionMode } from './types';
import { useFetchSseStream } from './streaming/useFetchSseStream';
import { createRingBufferStore, useRingBuffer } from './streaming/ringBufferStore';
import { useResilientWebSocket } from './streaming/useResilientWebSocket';

export default function App() {
  // Ingestion Mode: Naive (triggers render per packet) vs Backpressure (useSyncExternalStore + RingBuffer)
  const [ingestionMode, setIngestionMode] = useState<IngestionMode>('backpressure');

  // Solver telemetry frequency (10Hz to 500Hz)
  const [targetHz, setTargetHz] = useState<number>(200);

  // Live FPS tracking
  const [fps, setFps] = useState<number>(60);
  const frameTimesRef = useRef<number[]>([]);
  const lastRafTimeRef = useRef<number>(performance.now());

  // Message arrival rate meter (calculated over 1-second rolling windows)
  const [arrivalRate, setArrivalRate] = useState<number>(0);
  const arrivalCountRef = useRef<number>(0);

  // --- External Ring Buffer Store (Backpressure Mode) ---
  const ringBufferStore = useMemo(() => createRingBufferStore<TelemetryPacket>(300), []);
  const backpressurePackets = useRingBuffer(ringBufferStore);

  // --- Naive React State (Naive Mode - Deliberate Anti-Pattern for Benchmark) ---
  const [naivePackets, setNaivePackets] = useState<TelemetryPacket[]>([]);
  const naiveTotalIngestedRef = useRef<number>(0);

  // Prompt state for AI Reasoning Console
  const [promptText, setPromptText] = useState<string>(
    'Solve TSP-100 instance with branch-and-cut and simulated annealing refinement'
  );

  // --- FPS Measurement Loop ---
  useEffect(() => {
    let animId: number;
    const tick = (now: number) => {
      const delta = now - lastRafTimeRef.current;
      lastRafTimeRef.current = now;

      if (delta > 0) {
        const instantFps = 1000 / delta;
        frameTimesRef.current.push(instantFps);
        if (frameTimesRef.current.length > 30) {
          frameTimesRef.current.shift();
        }
        const avgFps =
          frameTimesRef.current.reduce((a, b) => a + b, 0) / frameTimesRef.current.length;
        setFps(Math.round(avgFps));
      }
      animId = requestAnimationFrame(tick);
    };

    animId = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(animId);
  }, []);

  // --- Arrival Rate Counter (1 Hz interval) ---
  useEffect(() => {
    const interval = window.setInterval(() => {
      setArrivalRate(arrivalCountRef.current);
      arrivalCountRef.current = 0;
    }, 1000);
    return () => clearInterval(interval);
  }, []);

  // --- WebSocket Telemetry Ingestion ---
  const handleWsMessage = useCallback(
    (msg: unknown) => {
      if (typeof msg === 'object' && msg !== null && (msg as TelemetryPacket).type === 'TELEMETRY') {
        const pkt = msg as TelemetryPacket;
        arrivalCountRef.current++;

        if (ingestionMode === 'backpressure') {
          // O(1) in-place push into RingBuffer; coalesced via requestAnimationFrame
          ringBufferStore.push(pkt);
        } else {
          // ANTI-PATTERN: Naive state update per packet (200-500 times/sec!)
          naiveTotalIngestedRef.current++;
          setNaivePackets(prev => [pkt, ...prev.slice(0, 100)]);
        }
      }
    },
    [ingestionMode, ringBufferStore]
  );

  const {
    state: wsState,
    send: wsSend,
    latencyMs,
    reconnectAttempts,
    connect: wsConnect,
    disconnect: wsDisconnect,
  } = useResilientWebSocket({
    url: '/ws/telemetry',
    autoConnect: true,
    onMessage: handleWsMessage,
  });

  // Send rate changes to backend
  const handleRateChange = (newHz: number) => {
    setTargetHz(newHz);
    wsSend({ type: 'SET_RATE', hz: newHz });
  };

  // Switch modes and clear prior buffers
  const handleModeSwitch = (mode: IngestionMode) => {
    setIngestionMode(mode);
    ringBufferStore.clear();
    setNaivePackets([]);
    naiveTotalIngestedRef.current = 0;
  };

  // --- SSE Reasoning Stream ---
  const {
    startStream,
    abortStream,
    accumulatedText,
    isStreaming,
    eventCount,
    error: sseError,
  } = useFetchSseStream();

  const handleStartReasoning = () => {
    startStream('/api/solver/stream-reasoning', {
      prompt: promptText,
      problem_type: 'TSP-100',
      simulated_delay_ms: 35,
    });
  };

  // Selected telemetry items based on active mode
  const activePackets =
    ingestionMode === 'backpressure'
      ? [...backpressurePackets].reverse().slice(0, 25)
      : naivePackets.slice(0, 25);

  const latestTelemetry =
    ingestionMode === 'backpressure'
      ? backpressurePackets[backpressurePackets.length - 1]
      : naivePackets[0];

  const totalIngested =
    ingestionMode === 'backpressure'
      ? ringBufferStore.getStats().totalIngested
      : naiveTotalIngestedRef.current;

  return (
    <div style={{ maxWidth: '1400px', margin: '0 auto', padding: '1.5rem', display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
      {/* Header */}
      <header style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '1rem', borderBottom: '1px solid #1e293b', paddingBottom: '1rem' }}>
        <div>
          <h1 style={{ fontSize: '1.6rem', fontWeight: 700, color: '#f8fafc', display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
            <span>⚡ Practice 116: Real-Time & Streaming UI</span>
          </h1>
          <p style={{ fontSize: '0.875rem', color: '#94a3b8', marginTop: '0.2rem' }}>
            Live SSE Token Streaming &bull; WebSockets &bull; <code style={{ color: '#818cf8' }}>useSyncExternalStore</code> Backpressure
          </p>
        </div>

        {/* Global Performance Badges */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', flexWrap: 'wrap' }}>
          {/* FPS Badge */}
          <div
            style={{
              padding: '0.4rem 0.8rem',
              borderRadius: '8px',
              backgroundColor: fps >= 55 ? '#064e3b' : fps >= 30 ? '#78350f' : '#7f1d1d',
              border: `1px solid ${fps >= 55 ? '#059669' : fps >= 30 ? '#d97706' : '#dc2626'}`,
              color: '#f8fafc',
              fontSize: '0.85rem',
              fontWeight: 700,
              display: 'flex',
              alignItems: 'center',
              gap: '0.4rem',
            }}
          >
            <span>{fps} FPS</span>
            <span style={{ fontSize: '0.75rem', opacity: 0.8 }}>
              {fps >= 55 ? 'Smooth' : fps >= 30 ? 'Stuttering' : 'STARVED'}
            </span>
          </div>

          {/* Ingestion Rate Badge */}
          <div style={{ padding: '0.4rem 0.8rem', borderRadius: '8px', backgroundColor: '#1e293b', border: '1px solid #334155', fontSize: '0.85rem', color: '#cbd5e1' }}>
            <strong>{arrivalRate}</strong> msgs/sec
          </div>

          {/* WebSocket Status */}
          <div style={{ padding: '0.4rem 0.8rem', borderRadius: '8px', backgroundColor: wsState === 'OPEN' ? '#0f172a' : '#450a0a', border: `1px solid ${wsState === 'OPEN' ? '#38bdf8' : '#f87171'}`, fontSize: '0.85rem', color: wsState === 'OPEN' ? '#38bdf8' : '#f87171', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
            <span style={{ width: '8px', height: '8px', borderRadius: '50%', backgroundColor: wsState === 'OPEN' ? '#38bdf8' : '#f87171' }} />
            <span>WS: {wsState}</span>
            {latencyMs !== null && <span style={{ opacity: 0.7 }}>({latencyMs}ms RTT)</span>}
            {reconnectAttempts > 0 && <span style={{ color: '#fbbf24' }}>#{reconnectAttempts}</span>}
          </div>
        </div>
      </header>

      {/* Benchmark Control Bar: Naive vs Backpressure */}
      <section style={{ backgroundColor: '#0f172a', border: '1px solid #1e293b', borderRadius: '12px', padding: '1.25rem', display: 'flex', flexDirection: 'column', gap: '1rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '1rem' }}>
          <div>
            <h2 style={{ fontSize: '1.1rem', fontWeight: 600, color: '#f1f5f9' }}>Render Backpressure Benchmark</h2>
            <p style={{ fontSize: '0.825rem', color: '#94a3b8' }}>
              Compare un-throttled React state reconciliation against external RingBuffer store coalesced with <code style={{ color: '#c084fc' }}>requestAnimationFrame</code>.
            </p>
          </div>

          {/* Mode Switcher Buttons */}
          <div style={{ display: 'flex', gap: '0.5rem', backgroundColor: '#1e293b', padding: '0.25rem', borderRadius: '8px' }}>
            <button
              onClick={() => handleModeSwitch('naive')}
              style={{
                backgroundColor: ingestionMode === 'naive' ? '#dc2626' : 'transparent',
                color: ingestionMode === 'naive' ? '#ffffff' : '#94a3b8',
                border: 'none',
              }}
            >
              ⚠ Naive useState (Drop FPS)
            </button>
            <button
              onClick={() => handleModeSwitch('backpressure')}
              style={{
                backgroundColor: ingestionMode === 'backpressure' ? '#4f46e5' : 'transparent',
                color: ingestionMode === 'backpressure' ? '#ffffff' : '#94a3b8',
                border: 'none',
              }}
            >
              ✓ RingBuffer + useSyncExternalStore (60 FPS)
            </button>
          </div>
        </div>

        {/* Telemetry Stream Frequency Slider */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', flexWrap: 'wrap', borderTop: '1px solid #1e293b', paddingTop: '0.75rem' }}>
          <span style={{ fontSize: '0.875rem', color: '#cbd5e1', fontWeight: 500 }}>
            WebSocket Broadcast Cadence: <strong style={{ color: '#38bdf8' }}>{targetHz} Hz</strong> ({targetHz} packets/sec)
          </span>
          <input
            type="range"
            min="10"
            max="500"
            step="10"
            value={targetHz}
            onChange={(e) => handleRateChange(Number(e.target.value))}
            style={{ flex: 1, minWidth: '180px', accentColor: '#6366f1' }}
          />
          <div style={{ display: 'flex', gap: '0.5rem' }}>
            {[20, 100, 250, 500].map((hz) => (
              <button
                key={hz}
                onClick={() => handleRateChange(hz)}
                style={{
                  padding: '0.2rem 0.6rem',
                  fontSize: '0.75rem',
                  backgroundColor: targetHz === hz ? '#312e81' : '#1e293b',
                  color: targetHz === hz ? '#c7d2fe' : '#94a3b8',
                  border: '1px solid #334155',
                }}
              >
                {hz}Hz
              </button>
            ))}
          </div>
          {wsState === 'OPEN' ? (
            <button onClick={wsDisconnect} style={{ backgroundColor: '#7f1d1d', color: '#fecaca', padding: '0.3rem 0.75rem', fontSize: '0.75rem' }}>
              Disconnect WS
            </button>
          ) : (
            <button onClick={wsConnect} style={{ backgroundColor: '#065f46', color: '#a7f3d0', padding: '0.3rem 0.75rem', fontSize: '0.75rem' }}>
              Reconnect WS
            </button>
          )}
        </div>
      </section>

      {/* Main Grid: SSE AI Reasoning vs WebSocket Solver Telemetry */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(420px, 1fr))', gap: '1.5rem' }}>
        {/* Panel 1: Server-Sent Events (SSE) AI Reasoning Console */}
        <section style={{ backgroundColor: '#0f172a', border: '1px solid #1e293b', borderRadius: '12px', padding: '1.25rem', display: 'flex', flexDirection: 'column', gap: '1rem' }}>
          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <h2 style={{ fontSize: '1.1rem', fontWeight: 600, color: '#f1f5f9' }}>AI Reasoning Stream (POST SSE)</h2>
              <span style={{ fontSize: '0.75rem', padding: '0.2rem 0.5rem', borderRadius: '4px', backgroundColor: isStreaming ? '#064e3b' : '#1e293b', color: isStreaming ? '#34d399' : '#94a3b8' }}>
                {isStreaming ? `Streaming (${eventCount} events)` : 'Idle'}
              </span>
            </div>
            <p style={{ fontSize: '0.8rem', color: '#94a3b8', marginTop: '0.25rem' }}>
              POST request via <code style={{ color: '#38bdf8' }}>ReadableStream</code> + <code style={{ color: '#38bdf8' }}>TextDecoder</code> with chunk-splitting resilience.
            </p>
          </div>

          <div style={{ display: 'flex', gap: '0.5rem' }}>
            <input
              type="text"
              value={promptText}
              onChange={(e) => setPromptText(e.target.value)}
              disabled={isStreaming}
              style={{
                flex: 1,
                padding: '0.5rem 0.75rem',
                backgroundColor: '#1e293b',
                border: '1px solid #334155',
                borderRadius: '6px',
                color: '#f8fafc',
                fontSize: '0.85rem',
              }}
            />
            {isStreaming ? (
              <button
                onClick={abortStream}
                style={{ backgroundColor: '#b91c1c', color: '#ffffff', border: 'none', padding: '0.5rem 1rem' }}
              >
                Abort
              </button>
            ) : (
              <button
                onClick={handleStartReasoning}
                style={{ backgroundColor: '#4f46e5', color: '#ffffff', border: 'none', padding: '0.5rem 1rem' }}
              >
                Stream
              </button>
            )}
          </div>

          {/* SSE Token Console Box */}
          <div
            style={{
              flex: 1,
              minHeight: '260px',
              maxHeight: '340px',
              overflowY: 'auto',
              backgroundColor: '#05070e',
              border: '1px solid #1e293b',
              borderRadius: '8px',
              padding: '1rem',
              fontFamily: 'monospace',
              fontSize: '0.825rem',
              lineHeight: 1.6,
              color: '#a5f3fc',
              whiteSpace: 'pre-wrap',
            }}
          >
            {accumulatedText ? (
              <span>{accumulatedText}</span>
            ) : (
              <span style={{ color: '#475569', fontStyle: 'italic' }}>
                Click "Stream" to dispatch POST SSE reasoning request to FastAPI...
              </span>
            )}
            {isStreaming && <span style={{ display: 'inline-block', width: '8px', height: '14px', backgroundColor: '#38bdf8', marginLeft: '4px', verticalAlign: 'middle', animation: 'blink 1s infinite' }} />}
            {sseError && <div style={{ color: '#f87171', marginTop: '0.5rem' }}>Error: {sseError.message}</div>}
          </div>
        </section>

        {/* Panel 2: Live Solver Telemetry (WebSockets + Backpressure) */}
        <section style={{ backgroundColor: '#0f172a', border: '1px solid #1e293b', borderRadius: '12px', padding: '1.25rem', display: 'flex', flexDirection: 'column', gap: '1rem' }}>
          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <h2 style={{ fontSize: '1.1rem', fontWeight: 600, color: '#f1f5f9' }}>Solver Telemetry HUD (WebSockets)</h2>
              <span style={{ fontSize: '0.75rem', padding: '0.2rem 0.5rem', borderRadius: '4px', backgroundColor: '#1e293b', color: '#cbd5e1' }}>
                Total Ingested: {totalIngested.toLocaleString()}
              </span>
            </div>
            <p style={{ fontSize: '0.8rem', color: '#94a3b8', marginTop: '0.25rem' }}>
              {ingestionMode === 'backpressure'
                ? `RingBuffer capacity: 300 | Snapshots flushed at 60Hz via rAF`
                : `Naive useState reconciliation on all ${targetHz} msgs/sec`}
            </p>
          </div>

          {/* Metric Cards */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '0.5rem' }}>
            <div style={{ backgroundColor: '#1e293b', padding: '0.6rem', borderRadius: '6px' }}>
              <div style={{ fontSize: '0.7rem', color: '#94a3b8' }}>Iteration</div>
              <div style={{ fontSize: '1rem', fontWeight: 700, color: '#f8fafc' }}>
                {latestTelemetry ? `#${latestTelemetry.iteration}` : '—'}
              </div>
            </div>
            <div style={{ backgroundColor: '#1e293b', padding: '0.6rem', borderRadius: '6px' }}>
              <div style={{ fontSize: '0.7rem', color: '#94a3b8' }}>Objective</div>
              <div style={{ fontSize: '1rem', fontWeight: 700, color: '#34d399' }}>
                {latestTelemetry ? `${latestTelemetry.objective.toFixed(1)} km` : '—'}
              </div>
            </div>
            <div style={{ backgroundColor: '#1e293b', padding: '0.6rem', borderRadius: '6px' }}>
              <div style={{ fontSize: '0.7rem', color: '#94a3b8' }}>Duality Gap</div>
              <div style={{ fontSize: '1rem', fontWeight: 700, color: '#f59e0b' }}>
                {latestTelemetry ? `${latestTelemetry.duality_gap.toFixed(2)}%` : '—'}
              </div>
            </div>
            <div style={{ backgroundColor: '#1e293b', padding: '0.6rem', borderRadius: '6px' }}>
              <div style={{ fontSize: '0.7rem', color: '#94a3b8' }}>Temperature</div>
              <div style={{ fontSize: '1rem', fontWeight: 700, color: '#818cf8' }}>
                {latestTelemetry ? `${latestTelemetry.temperature.toFixed(2)}°` : '—'}
              </div>
            </div>
          </div>

          {/* Telemetry Stream Feed Table */}
          <div
            style={{
              flex: 1,
              minHeight: '200px',
              maxHeight: '260px',
              overflowY: 'auto',
              backgroundColor: '#05070e',
              border: '1px solid #1e293b',
              borderRadius: '8px',
              fontFamily: 'monospace',
              fontSize: '0.75rem',
            }}
          >
            <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid #1e293b', color: '#64748b' }}>
                  <th style={{ padding: '0.4rem 0.6rem' }}>Iter</th>
                  <th style={{ padding: '0.4rem 0.6rem' }}>Objective</th>
                  <th style={{ padding: '0.4rem 0.6rem' }}>Gap</th>
                  <th style={{ padding: '0.4rem 0.6rem' }}>Temp</th>
                  <th style={{ padding: '0.4rem 0.6rem' }}>Timestamp</th>
                </tr>
              </thead>
              <tbody>
                {activePackets.length > 0 ? (
                  activePackets.map((pkt) => (
                    <tr key={`${pkt.iteration}-${pkt.timestamp}`} style={{ borderBottom: '1px solid #0f172a', color: '#cbd5e1' }}>
                      <td style={{ padding: '0.35rem 0.6rem', color: '#60a5fa' }}>#{pkt.iteration}</td>
                      <td style={{ padding: '0.35rem 0.6rem', color: '#34d399' }}>{pkt.objective} km</td>
                      <td style={{ padding: '0.35rem 0.6rem', color: '#fbbf24' }}>{pkt.duality_gap}%</td>
                      <td style={{ padding: '0.35rem 0.6rem', color: '#c084fc' }}>{pkt.temperature}</td>
                      <td style={{ padding: '0.35rem 0.6rem', color: '#64748b' }}>
                        {new Date(pkt.timestamp * 1000).toISOString().slice(14, 23)}
                      </td>
                    </tr>
                  ))
                ) : (
                  <tr>
                    <td colSpan={5} style={{ padding: '1rem', textAlign: 'center', color: '#475569' }}>
                      Connecting to WebSocket telemetry feed at {targetHz} Hz...
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </section>
      </div>

      {/* Footer / Architecture Summary */}
      <footer style={{ borderTop: '1px solid #1e293b', paddingTop: '1rem', color: '#64748b', fontSize: '0.8rem', display: 'flex', justifyContent: 'space-between', flexWrap: 'wrap', gap: '0.5rem' }}>
        <span>Practice 116 — React 19 + TypeScript + FastAPI</span>
        <span>Exercises 1–4: `useFetchSseStream`, `ringBufferStore`, `useResilientWebSocket`, `main.py`</span>
      </footer>
    </div>
  );
}
