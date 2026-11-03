import { useEffect, useState, useCallback } from 'react';
import {
  Layers,
  Activity,
  Database,
  RefreshCw,
  Zap,
  Edit3,
  Server,
  Network,
} from 'lucide-react';
import { IngestionPipeline } from './components/IngestionPipeline';
import {
  fetchWithETag,
  updateWithIfMatch,
  type Dataset,
} from './api/conditionalFetch';

export function App() {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [selectedDataset, setSelectedDataset] = useState<Dataset | null>(null);
  const [editName, setEditName] = useState('');
  const [editDesc, setEditDesc] = useState('');
  const [etagCache, setEtagCache] = useState<Record<string, { etag: string; data: Dataset }>>({});
  const [logs, setLogs] = useState<string[]>([]);
  const [chaosDropRate, setChaosDropRate] = useState<number>(0.0);
  const [chaosDelay, setChaosDelay] = useState<number>(0);
  const [lastCacheHit, setLastCacheHit] = useState<string | null>(null);

  const addLog = useCallback((msg: string) => {
    const timestamp = new Date().toLocaleTimeString();
    setLogs((prev) => [`[${timestamp}] ${msg}`, ...prev.slice(0, 49)]);
  }, []);

  // Fetch all registered datasets
  const loadDatasets = useCallback(async () => {
    try {
      const res = await fetch('/api/datasets');
      if (res.ok) {
        const data: Dataset[] = await res.json();
        setDatasets(data);
        addLog(`Loaded ${data.length} registered datasets from FastAPI`);
      }
    } catch (err: unknown) {
      addLog(`Error loading datasets: ${err instanceof Error ? err.message : String(err)}`);
    }
  }, [addLog]);

  useEffect(() => {
    loadDatasets();
  }, [loadDatasets]);

  // Exercise 2 Demo: Test 304 Not Modified
  const handleTestETagCaching = async (datasetId: string) => {
    const cached = etagCache[datasetId];
    addLog(`GET /api/datasets/${datasetId} with If-None-Match: ${cached ? cached.etag : '(none)'}`);
    try {
      const result = await fetchWithETag<Dataset>(
        `/api/datasets/${datasetId}`,
        cached?.etag,
        cached?.data
      );

      if (result.fromCache) {
        setLastCacheHit(`304 Not Modified for ${datasetId} (Reused client cache, 0 body bytes)`);
        addLog(`Response 304 Not Modified! Server validated ETag: ${result.etag}`);
      } else {
        setLastCacheHit(`200 OK for ${datasetId} (Fresh payload downloaded)`);
        setEtagCache((prev) => ({
          ...prev,
          [datasetId]: { etag: result.etag || '', data: result.data },
        }));
        addLog(`Response 200 OK! New ETag: ${result.etag}`);
      }
    } catch (err: unknown) {
      addLog(`Conditional fetch error: ${err instanceof Error ? err.message : String(err)}`);
    }
  };

  // Exercise 2 Demo: Test If-Match Optimistic Locking (412)
  const handleSaveEdit = async (staleETag = false) => {
    if (!selectedDataset) return;
    const targetETag = staleETag ? '"stale-etag-simulation-v0"' : selectedDataset.etag;
    addLog(`PUT /api/datasets/${selectedDataset.id} with If-Match: ${targetETag}`);

    try {
      const result = await updateWithIfMatch<Dataset>(
        `/api/datasets/${selectedDataset.id}`,
        { name: editName, description: editDesc },
        targetETag
      );
      addLog(`Update successful! 200 OK. New ETag: ${result.etag}`);
      setSelectedDataset(result.data);
      loadDatasets();
    } catch (err: unknown) {
      addLog(`Update rejected: ${err instanceof Error ? err.message : String(err)}`);
    }
  };

  // Update Chaos configuration
  const handleApplyChaos = async (dropRate: number, delayMs: number) => {
    setChaosDropRate(dropRate);
    setChaosDelay(delayMs);
    try {
      await fetch('/api/chaos/config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ drop_post_commit_rate: dropRate, delay_ms: delayMs }),
      });
      addLog(`Chaos updated: drop_post_commit_rate=${dropRate}, delay_ms=${delayMs}`);
    } catch (err: unknown) {
      addLog(`Failed to update chaos: ${err instanceof Error ? err.message : String(err)}`);
    }
  };

  const handleResetState = async () => {
    try {
      await fetch('/api/chaos/reset', { method: 'POST' });
      setEtagCache({});
      setLastCacheHit(null);
      setSelectedDataset(null);
      setChaosDropRate(0);
      setChaosDelay(0);
      addLog('Reset backend storage, seeded datasets, and flushed idempotency store.');
      loadDatasets();
    } catch (err: unknown) {
      addLog(`Reset failed: ${err instanceof Error ? err.message : String(err)}`);
    }
  };

  return (
    <div className="container">
      {/* Header */}
      <header className="app-header">
        <div>
          <h1 className="app-title">
            <Layers className="text-accent-blue" size={28} />
            Practice 114: HTTP Semantics & Resilient Ingestion
          </h1>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.875rem', marginTop: '0.25rem' }}>
            Idempotency-Key Protocol · Conditional Requests (ETag, 304, 412) · Direct S3 Streaming
          </p>
        </div>
        <div style={{ display: 'flex', gap: '0.5rem' }}>
          <span className="badge badge-blue">React 19</span>
          <span className="badge badge-green">FastAPI</span>
          <span className="badge badge-yellow">MinIO S3</span>
        </div>
      </header>

      {/* Main Grid */}
      <div className="dashboard-grid">
        {/* Left Column: Ingestion Pipeline & Chaos Controls */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
          <IngestionPipeline
            onDatasetCreated={(newDs) => {
              addLog(`Dataset committed: ${newDs.id} (${newDs.name})`);
              loadDatasets();
            }}
            onLogEvent={addLog}
          />

          {/* Chaos Simulator Card */}
          <div className="card">
            <h3 className="card-title">
              <Zap className="text-accent-amber" size={20} />
              Network Chaos & Failure Simulator
            </h3>
            <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: '1rem' }}>
              Simulates production network instability to test idempotency retries and key persistence.
            </p>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
              <div>
                <label className="form-label">
                  Post-Commit Drop Rate: <strong>{Math.round(chaosDropRate * 100)}%</strong>
                </label>
                <div style={{ display: 'flex', gap: '0.5rem' }}>
                  {[0.0, 0.5, 1.0].map((rate) => (
                    <button
                      key={rate}
                      type="button"
                      className={`btn ${chaosDropRate === rate ? 'btn-danger' : 'btn-secondary'}`}
                      style={{ flex: 1, padding: '0.4rem', fontSize: '0.75rem' }}
                      onClick={() => handleApplyChaos(rate, chaosDelay)}
                    >
                      {rate === 0 ? '0% (Normal)' : rate === 0.5 ? '50% Flaky' : '100% Drop Always'}
                    </button>
                  ))}
                </div>
              </div>

              <div>
                <label className="form-label">
                  Simulated Network Delay: <strong>{chaosDelay} ms</strong>
                </label>
                <div style={{ display: 'flex', gap: '0.5rem' }}>
                  {[0, 500, 1500].map((d) => (
                    <button
                      key={d}
                      type="button"
                      className={`btn ${chaosDelay === d ? 'btn-primary' : 'btn-secondary'}`}
                      style={{ flex: 1, padding: '0.4rem', fontSize: '0.75rem' }}
                      onClick={() => handleApplyChaos(chaosDropRate, d)}
                    >
                      {d} ms
                    </button>
                  ))}
                </div>
              </div>

              <button
                type="button"
                className="btn btn-secondary"
                onClick={handleResetState}
                style={{ marginTop: '0.5rem' }}
              >
                <RefreshCw size={14} /> Reset State & Clear Idempotency Cache
              </button>
            </div>
          </div>
        </div>

        {/* Right Column: Datasets, Conditional Caching & Telemetry */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
          {/* Datasets Table */}
          <div className="card">
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
              <h3 className="card-title" style={{ margin: 0 }}>
                <Database className="text-accent-emerald" size={20} />
                Registered Datasets ({datasets.length})
              </h3>
              <button
                type="button"
                className="btn btn-secondary"
                style={{ padding: '0.35rem 0.65rem', fontSize: '0.75rem' }}
                onClick={loadDatasets}
              >
                <RefreshCw size={12} /> Refresh
              </button>
            </div>

            {lastCacheHit && (
              <div style={{ padding: '0.5rem', background: 'rgba(52, 211, 153, 0.1)', border: '1px solid var(--accent-emerald)', borderRadius: '0.4rem', fontSize: '0.75rem', marginBottom: '0.75rem', color: 'var(--accent-emerald)' }}>
                {lastCacheHit}
              </div>
            )}

            <div className="table-container">
              <table className="custom-table">
                <thead>
                  <tr>
                    <th>ID / Name</th>
                    <th>Version & ETag</th>
                    <th>Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {datasets.map((d) => (
                    <tr key={d.id}>
                      <td>
                        <strong>{d.name}</strong>
                        <div style={{ fontSize: '0.7rem', color: 'var(--text-secondary)' }}>
                          {d.id} · {(d.size_bytes / 1024 / 1024).toFixed(2)} MB
                        </div>
                      </td>
                      <td>
                        <span className="badge badge-blue">v{d.version}</span>
                        <div style={{ fontSize: '0.7rem', color: 'var(--text-secondary)', fontFamily: 'var(--font-mono)' }}>
                          {d.etag || 'N/A'}
                        </div>
                      </td>
                      <td>
                        <div style={{ display: 'flex', gap: '0.35rem' }}>
                          <button
                            type="button"
                            className="btn btn-secondary"
                            style={{ padding: '0.25rem 0.5rem', fontSize: '0.7rem' }}
                            title="Test 304 Not Modified conditional GET"
                            onClick={() => handleTestETagCaching(d.id)}
                          >
                            Test 304
                          </button>
                          <button
                            type="button"
                            className="btn btn-primary"
                            style={{ padding: '0.25rem 0.5rem', fontSize: '0.7rem' }}
                            onClick={() => {
                              setSelectedDataset(d);
                              setEditName(d.name);
                              setEditDesc(d.description);
                            }}
                          >
                            <Edit3 size={12} /> Edit
                          </button>
                        </div>
                      </td>
                    </tr>
                  ))}
                  {datasets.length === 0 && (
                    <tr>
                      <td colSpan={3} style={{ textAlign: 'center', color: 'var(--text-secondary)' }}>
                        No datasets registered yet.
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>

            {/* Optimistic Concurrency Editor Modal/Section */}
            {selectedDataset && (
              <div style={{ marginTop: '1rem', padding: '1rem', background: '#090d16', border: '1px solid var(--border-color)', borderRadius: '0.5rem' }}>
                <h4 style={{ fontSize: '0.9rem', marginBottom: '0.5rem', color: 'var(--accent-blue)' }}>
                  Optimistic Concurrency Editor: {selectedDataset.id} (v{selectedDataset.version})
                </h4>
                <div className="form-group">
                  <label className="form-label">Name</label>
                  <input
                    type="text"
                    className="form-input"
                    value={editName}
                    onChange={(e) => setEditName(e.target.value)}
                  />
                </div>
                <div className="form-group">
                  <label className="form-label">Description</label>
                  <input
                    type="text"
                    className="form-input"
                    value={editDesc}
                    onChange={(e) => setEditDesc(e.target.value)}
                  />
                </div>
                <div style={{ display: 'flex', gap: '0.5rem', marginTop: '0.75rem' }}>
                  <button
                    type="button"
                    className="btn btn-success"
                    onClick={() => handleSaveEdit(false)}
                    style={{ fontSize: '0.75rem' }}
                  >
                    Save with Valid ETag (If-Match)
                  </button>
                  <button
                    type="button"
                    className="btn btn-danger"
                    onClick={() => handleSaveEdit(true)}
                    style={{ fontSize: '0.75rem' }}
                    title="Simulates concurrent update conflict by sending stale ETag"
                  >
                    Simulate 412 Conflict (Stale ETag)
                  </button>
                  <button
                    type="button"
                    className="btn btn-secondary"
                    onClick={() => setSelectedDataset(null)}
                    style={{ fontSize: '0.75rem' }}
                  >
                    Cancel
                  </button>
                </div>
              </div>
            )}
          </div>

          {/* Protocol Telemetry Log */}
          <div className="card">
            <h3 className="card-title">
              <Activity className="text-accent-blue" size={20} />
              HTTP Protocol & Telemetry Log
            </h3>
            <div className="telemetry-log">
              {logs.map((log, idx) => (
                <div key={idx} className="telemetry-entry">
                  {log}
                </div>
              ))}
              {logs.length === 0 && <div>Awaiting HTTP network events...</div>}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
