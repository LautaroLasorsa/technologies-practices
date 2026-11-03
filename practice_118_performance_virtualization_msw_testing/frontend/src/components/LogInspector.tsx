import React, { useState, useEffect, useCallback, useTransition } from 'react';
import { LogRecord, LogLevel, ServiceTag, PartitionId, FilterCriteria } from '../types';
import { useWorkerFilter } from '../hooks/useWorkerFilter';
import { VirtualizedLogGrid } from './VirtualizedLogGrid';

export interface LogInspectorProps {
  initialPartition?: PartitionId;
  defaultSimulateError?: string;
  defaultSimulateRateLimit?: string;
}

export const LogInspector: React.FC<LogInspectorProps> = ({
  initialPartition = 'warehouse-us-east',
  defaultSimulateError,
  defaultSimulateRateLimit,
}) => {
  const [partition, setPartition] = useState<PartitionId>(initialPartition);
  const [rawLogs, setRawLogs] = useState<LogRecord[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [errorBanner, setErrorBanner] = useState<string | null>(null);
  const [retryAfter, setRetryAfter] = useState<number | null>(null);
  const [naiveMode, setNaiveMode] = useState<boolean>(false);

  // Filter criteria state
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [minSeverity, setMinSeverity] = useState<LogLevel>('INFO');
  const [serviceTag, setServiceTag] = useState<ServiceTag | 'ALL'>('ALL');
  const [minAnomalyScore, setMinAnomalyScore] = useState<number>(0);

  // Concurrent transition for non-urgent filter criteria updates
  const [, startTransition] = useTransition();

  const [criteria, setCriteria] = useState<FilterCriteria>({
    query: '',
    minSeverity: 'INFO',
    service: 'ALL',
    minAnomalyScore: 0,
  });

  const handleQueryChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const val = e.target.value;
    setSearchQuery(val);
    startTransition(() => {
      setCriteria((prev) => ({ ...prev, query: val }));
    });
  };

  const handleSeverityChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const val = e.target.value as LogLevel;
    setMinSeverity(val);
    startTransition(() => {
      setCriteria((prev) => ({ ...prev, minSeverity: val }));
    });
  };

  const handleServiceChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const val = e.target.value as ServiceTag | 'ALL';
    setServiceTag(val);
    startTransition(() => {
      setCriteria((prev) => ({ ...prev, service: val }));
    });
  };

  const handleScoreChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const val = parseFloat(e.target.value);
    setMinAnomalyScore(val);
    startTransition(() => {
      setCriteria((prev) => ({ ...prev, minAnomalyScore: val }));
    });
  };

  const fetchLogs = useCallback(
    async (targetPartition: PartitionId, signal?: AbortSignal) => {
      setIsLoading(true);
      setErrorBanner(null);
      setRetryAfter(null);

      const params = new URLSearchParams({ partition: targetPartition });
      if (defaultSimulateError) params.set('simulateError', defaultSimulateError);
      if (defaultSimulateRateLimit) params.set('simulateRateLimit', defaultSimulateRateLimit);

      try {
        const response = await fetch(`/api/logs?${params.toString()}`, { signal });

        if (response.status === 429) {
          const retryHeader = response.headers.get('Retry-After');
          const delaySec = retryHeader ? parseInt(retryHeader, 10) : 5;
          setRetryAfter(delaySec);
          setErrorBanner(`HTTP 429: Rate limit exceeded. Retry allowed in ${delaySec}s.`);
          setIsLoading(false);
          return;
        }

        if (!response.ok) {
          const body = await response.json().catch(() => ({ error: 'Unknown server error' }));
          setErrorBanner(`HTTP ${response.status}: ${body.error || 'Failed to fetch logs'}`);
          setIsLoading(false);
          return;
        }

        const data = await response.json();
        setRawLogs(data.logs);
      } catch (err: unknown) {
        if (err instanceof DOMException && err.name === 'AbortError') {
          // Request was intentionally aborted by switching partitions; do not show error
          return;
        }
        setErrorBanner(`Network Error: ${err instanceof Error ? err.message : String(err)}`);
      } finally {
        setIsLoading(false);
      }
    },
    [defaultSimulateError, defaultSimulateRateLimit]
  );

  useEffect(() => {
    const controller = new AbortController();
    fetchLogs(partition, controller.signal);

    return () => {
      controller.abort();
    };
  }, [partition, fetchLogs]);

  // Offload heavy multi-criteria filtering and anomaly scoring to Web Worker
  const { filteredLogs, isComputing, computeTimeMs, totalMatched } = useWorkerFilter(
    rawLogs,
    criteria
  );

  return (
    <div style={{ maxWidth: '1200px', margin: '0 auto', padding: '24px' }}>
      <header style={{ marginBottom: '20px' }}>
        <h1 style={{ margin: '0 0 8px 0', fontSize: '24px', color: '#f8fafc' }}>
          Real-Time Log & Schedule Inspector
        </h1>
        <p style={{ margin: 0, color: '#94a3b8', fontSize: '14px' }}>
          50,000-row telemetry dataset with Web Worker offloaded filtering, concurrent transitions, and TanStack DOM virtualization.
        </p>
      </header>

      {/* Error / Rate Limit Banner */}
      {errorBanner && (
        <div
          data-testid="error-banner"
          style={{
            padding: '12px 16px',
            marginBottom: '16px',
            borderRadius: '6px',
            backgroundColor: retryAfter ? '#78350f' : '#7f1d1d',
            border: `1px solid ${retryAfter ? '#d97706' : '#dc2626'}`,
            color: '#fef2f2',
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
          }}
        >
          <span>{errorBanner}</span>
          <button
            data-testid="retry-button"
            onClick={() => fetchLogs(partition)}
            style={{
              padding: '6px 12px',
              backgroundColor: '#b91c1c',
              color: 'white',
              border: 'none',
              borderRadius: '4px',
              cursor: 'pointer',
              fontWeight: 'bold',
            }}
          >
            Retry Request
          </button>
        </div>
      )}

      {/* Control Toolbar */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
          gap: '12px',
          marginBottom: '16px',
          padding: '16px',
          backgroundColor: '#1e293b',
          borderRadius: '8px',
        }}
      >
        <div>
          <label style={{ display: 'block', fontSize: '12px', color: '#94a3b8', marginBottom: '4px' }}>
            Warehouse Partition:
          </label>
          <select
            data-testid="partition-select"
            value={partition}
            onChange={(e) => setPartition(e.target.value as PartitionId)}
            style={{ width: '100%', padding: '6px 8px', borderRadius: '4px', backgroundColor: '#0f172a', color: '#f8fafc', border: '1px solid #475569' }}
          >
            <option value="warehouse-us-east">warehouse-us-east</option>
            <option value="warehouse-eu-central">warehouse-eu-central</option>
            <option value="warehouse-apac">warehouse-apac</option>
          </select>
        </div>

        <div>
          <label style={{ display: 'block', fontSize: '12px', color: '#94a3b8', marginBottom: '4px' }}>
            Filter Message / Service / ID:
          </label>
          <input
            data-testid="search-input"
            type="text"
            placeholder="Type regex or keyword..."
            value={searchQuery}
            onChange={handleQueryChange}
            style={{ width: '100%', padding: '6px 8px', borderRadius: '4px', backgroundColor: '#0f172a', color: '#f8fafc', border: '1px solid #475569' }}
          />
        </div>

        <div>
          <label style={{ display: 'block', fontSize: '12px', color: '#94a3b8', marginBottom: '4px' }}>
            Min Severity:
          </label>
          <select
            data-testid="severity-select"
            value={minSeverity}
            onChange={handleSeverityChange}
            style={{ width: '100%', padding: '6px 8px', borderRadius: '4px', backgroundColor: '#0f172a', color: '#f8fafc', border: '1px solid #475569' }}
          >
            <option value="INFO">INFO (All)</option>
            <option value="WARN">WARN + Higher</option>
            <option value="ERROR">ERROR + CRITICAL</option>
            <option value="CRITICAL">CRITICAL Only</option>
          </select>
        </div>

        <div>
          <label style={{ display: 'block', fontSize: '12px', color: '#94a3b8', marginBottom: '4px' }}>
            Service:
          </label>
          <select
            data-testid="service-select"
            value={serviceTag}
            onChange={handleServiceChange}
            style={{ width: '100%', padding: '6px 8px', borderRadius: '4px', backgroundColor: '#0f172a', color: '#f8fafc', border: '1px solid #475569' }}
          >
            <option value="ALL">All Services</option>
            <option value="scheduler-engine">scheduler-engine</option>
            <option value="order-router">order-router</option>
            <option value="inventory-sync">inventory-sync</option>
            <option value="dispatch-gateway">dispatch-gateway</option>
          </select>
        </div>

        <div>
          <label style={{ display: 'block', fontSize: '12px', color: '#94a3b8', marginBottom: '4px' }}>
            Min Anomaly Score: {minAnomalyScore.toFixed(2)}
          </label>
          <input
            data-testid="anomaly-slider"
            type="range"
            min="0"
            max="1"
            step="0.05"
            value={minAnomalyScore}
            onChange={handleScoreChange}
            style={{ width: '100%' }}
          />
        </div>

        <div style={{ display: 'flex', alignItems: 'center', marginTop: '16px' }}>
          <label style={{ fontSize: '13px', color: '#cbd5e1', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <input
              data-testid="naive-toggle"
              type="checkbox"
              checked={naiveMode}
              onChange={(e) => setNaiveMode(e.target.checked)}
            />
            Benchmark Naive Mode (No Virtualization)
          </label>
        </div>
      </div>

      {/* Performance & Metrics HUD */}
      <div
        data-testid="performance-hud"
        style={{
          display: 'flex',
          gap: '20px',
          alignItems: 'center',
          padding: '8px 16px',
          marginBottom: '12px',
          backgroundColor: '#090d16',
          borderRadius: '6px',
          fontSize: '12px',
          fontFamily: 'monospace',
          color: '#38bdf8',
          border: '1px solid #1e293b',
        }}
      >
        <span>Loaded: <strong>{rawLogs.length.toLocaleString()}</strong></span>
        <span>Matched: <strong>{totalMatched.toLocaleString()}</strong></span>
        <span>
          Worker CPU Time:{' '}
          <strong style={{ color: computeTimeMs > 20 ? '#f87171' : '#4ade80' }}>
            {computeTimeMs.toFixed(1)}ms
          </strong>
        </span>
        <span>
          Worker Status:{' '}
          <strong style={{ color: isComputing ? '#fbbf24' : '#94a3b8' }}>
            {isComputing ? 'COMPUTING...' : 'IDLE'}
          </strong>
        </span>
        <span>
          Mode: <strong style={{ color: naiveMode ? '#f87171' : '#38bdf8' }}>{naiveMode ? 'NAIVE DOM' : 'VIRTUALIZED'}</strong>
        </span>
        {isLoading && <span style={{ color: '#38bdf8' }}>[NETWORK FETCHING...]</span>}
      </div>

      {/* Table Header */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: '90px 140px 180px 1fr 80px 70px',
          padding: '8px 12px',
          backgroundColor: '#1e293b',
          color: '#94a3b8',
          fontSize: '12px',
          fontWeight: 'bold',
          borderTopLeftRadius: '8px',
          borderTopRightRadius: '8px',
        }}
      >
        <span>SEVERITY</span>
        <span>TIMESTAMP</span>
        <span>SERVICE</span>
        <span>LOG MESSAGE</span>
        <span style={{ textAlign: 'right' }}>LATENCY</span>
        <span style={{ textAlign: 'right' }}>ANOMALY</span>
      </div>

      {/* Virtualized / Naive Grid */}
      <VirtualizedLogGrid logs={filteredLogs} naiveMode={naiveMode} />
    </div>
  );
};
