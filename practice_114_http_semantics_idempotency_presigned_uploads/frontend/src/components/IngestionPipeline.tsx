/**
 * IngestionPipeline Component.
 *
 * Orchestrates the full 3-phase resilient ingestion workflow:
 * Phase 1: Request S3 Presigned URL from FastAPI.
 * Phase 2: Direct-to-MinIO streaming via XMLHttpRequest with real-time progress.
 * Phase 3: Idempotent Commit with Idempotency-Key and S3 ETag verification.
 */

import React, { useState } from 'react';
import {
  UploadCloud,
  FileCheck,
  ShieldCheck,
  AlertTriangle,
  RotateCw,
  XCircle,
  FileCode2,
} from 'lucide-react';
import { useIdempotentMutation } from '../hooks/useIdempotentMutation';
import { usePresignedUpload } from '../hooks/usePresignedUpload';
import type { Dataset } from '../api/conditionalFetch';

interface IngestionPipelineProps {
  onDatasetCreated: (dataset: Dataset) => void;
  onLogEvent: (message: string) => void;
}

export const IngestionPipeline: React.FC<IngestionPipelineProps> = ({
  onDatasetCreated,
  onLogEvent,
}) => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [datasetName, setDatasetName] = useState('Production Metrics 2026');
  const [datasetDesc, setDatasetDesc] = useState('Real-time server transaction log telemetry');
  const [currentStep, setCurrentStep] = useState<1 | 2 | 3 | 4>(1);
  const [pipelineError, setPipelineError] = useState<string | null>(null);

  // Hooks
  const uploader = usePresignedUpload();
  const commitMutation = useIdempotentMutation<Dataset, {
    name: string;
    description: string;
    s3_key: string;
    etag: string;
    size_bytes: number;
  }>();

  // Create a synthetic CSV file on demand for instant testing
  const generateSyntheticFile = (sizeMB = 2) => {
    onLogEvent(`Generating ${sizeMB}MB synthetic CSV dataset in memory...`);
    const line = '2026-11-03T16:00:00Z,server-us-east-1,CPU_USAGE,42.5,NORMAL\n';
    const linesNeeded = Math.ceil((sizeMB * 1024 * 1024) / line.length);
    const content = line.repeat(linesNeeded);
    const blob = new Blob([content], { type: 'text/csv' });
    const file = new File([blob], `synthetic_telemetry_${Date.now()}.csv`, { type: 'text/csv' });
    setSelectedFile(file);
    onLogEvent(`Synthetic file created: ${file.name} (${(file.size / 1024 / 1024).toFixed(2)} MB)`);
  };

  /**
   * Run the 3-Phase Ingestion Workflow.
   *
   * ========================================================================
   * TODO(human): Exercise 4 - Orchestrate Resilient 3-Phase Ingestion Flow
   * ========================================================================
   * Context:
   * Combines presigned S3 URL negotiation, direct binary streaming with
   * progress tracking, and idempotent commit.
   *
   * Instructions:
   * 1. Validate `selectedFile`. If missing, set error and exit.
   * 2. Phase 1: Request Presigned URL from FastAPI:
   *    const presignRes = await fetch('/api/uploads/presign', {
   *      method: 'POST',
   *      headers: { 'Content-Type': 'application/json' },
   *      body: JSON.stringify({
   *        filename: selectedFile.name,
   *        content_type: selectedFile.type || 'text/csv',
   *        size_bytes: selectedFile.size,
   *      }),
   *    });
   *    const { upload_url, s3_key } = await presignRes.json();
   *    Update step to 2.
   * 3. Phase 2: Stream binary directly to MinIO using `uploader.upload(upload_url, selectedFile)`:
   *    const { etag } = await uploader.upload(upload_url, selectedFile);
   *    Update step to 3.
   * 4. Phase 3: Idempotent Commit to FastAPI:
   *    const committedDataset = await commitMutation.mutate('/api/datasets/commit', {
   *      name: datasetName,
   *      description: datasetDesc,
   *      s3_key,
   *      etag,
   *      size_bytes: selectedFile.size,
   *    });
   * 5. Finalize: Call `onDatasetCreated(committedDataset)`, set step to 4.
   * ========================================================================
   */
  const handleStartIngestion = async () => {
    if (!selectedFile) {
      setPipelineError('Please select or generate a file first.');
      return;
    }
    setPipelineError(null);
    setCurrentStep(1);

    // --- IMPLEMENTATION GOES HERE ---
    throw new Error(
      'Exercise 4: Implement handleStartIngestion in frontend/src/components/IngestionPipeline.tsx'
    );
  };

  const handleRetryCommit = async () => {
    onLogEvent(`Retrying commit with persistent Idempotency-Key: ${commitMutation.idempotencyKey}...`);
    try {
      const result = await commitMutation.retry();
      if (result) {
        onDatasetCreated(result);
        setCurrentStep(4);
        onLogEvent(`Commit recovered successfully! Replay status: ${commitMutation.isReplay}`);
      }
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      onLogEvent(`Retry attempt failed: ${msg}`);
    }
  };

  return (
    <div className="card">
      <h2 className="card-title">
        <UploadCloud className="text-accent-blue" size={22} />
        Resilient Ingestion Pipeline (3-Phase Flow)
      </h2>

      {/* Pipeline Step Indicators */}
      <div style={{ display: 'flex', gap: '0.5rem', marginBottom: '1.5rem' }}>
        {[
          { num: 1, label: '1. Presign S3 URL' },
          { num: 2, label: '2. Direct MinIO Stream' },
          { num: 3, label: '3. Idempotent Commit' },
        ].map((s) => (
          <div
            key={s.num}
            style={{
              flex: 1,
              padding: '0.5rem',
              borderRadius: '0.35rem',
              background: currentStep >= s.num ? 'rgba(56, 189, 248, 0.15)' : 'var(--bg-tertiary)',
              border: `1px solid ${currentStep >= s.num ? 'var(--accent-blue)' : 'var(--border-color)'}`,
              fontSize: '0.8rem',
              fontWeight: 600,
              color: currentStep >= s.num ? 'var(--accent-blue)' : 'var(--text-secondary)',
              textAlign: 'center',
            }}
          >
            {s.label}
          </div>
        ))}
      </div>

      {/* Dataset Metadata Inputs */}
      <div className="form-group">
        <label className="form-label">Dataset Name</label>
        <input
          type="text"
          className="form-input"
          value={datasetName}
          onChange={(e) => setDatasetName(e.target.value)}
        />
      </div>

      <div className="form-group">
        <label className="form-label">Description</label>
        <input
          type="text"
          className="form-input"
          value={datasetDesc}
          onChange={(e) => setDatasetDesc(e.target.value)}
        />
      </div>

      {/* File Selection / Generator */}
      <div className="form-group">
        <label className="form-label">Target Binary / CSV File</label>
        <div style={{ display: 'flex', gap: '0.75rem', alignItems: 'center' }}>
          <input
            type="file"
            onChange={(e) => setSelectedFile(e.target.files?.[0] || null)}
            style={{ flex: 1, color: 'var(--text-secondary)', fontSize: '0.85rem' }}
          />
          <button
            type="button"
            className="btn btn-secondary"
            onClick={() => generateSyntheticFile(4)}
          >
            <FileCode2 size={16} />
            Generate 4MB Test CSV
          </button>
        </div>
        {selectedFile && (
          <div style={{ marginTop: '0.4rem', fontSize: '0.8rem', color: 'var(--accent-emerald)' }}>
            Selected: <strong>{selectedFile.name}</strong> ({(selectedFile.size / 1024 / 1024).toFixed(2)} MB)
          </div>
        )}
      </div>

      {/* Progress & Speed Display (Phase 2) */}
      {uploader.isUploading && (
        <div style={{ margin: '1rem 0' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', marginBottom: '0.25rem' }}>
            <span>Streaming directly to MinIO (port 9000)...</span>
            <span style={{ fontWeight: 600 }}>{uploader.progress}%</span>
          </div>
          <div className="progress-container">
            <div className="progress-fill" style={{ width: `${uploader.progress}%` }} />
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', color: 'var(--text-secondary)', marginTop: '0.35rem' }}>
            <span>
              Transferred: {(uploader.loadedBytes / 1024 / 1024).toFixed(2)} MB / {(uploader.totalBytes / 1024 / 1024).toFixed(2)} MB
            </span>
            <span>Speed: {(uploader.speedBytesPerSec / 1024).toFixed(1)} KB/s</span>
          </div>
          <button
            type="button"
            className="btn btn-danger"
            onClick={uploader.abort}
            style={{ marginTop: '0.5rem', padding: '0.4rem 0.8rem', fontSize: '0.75rem' }}
          >
            <XCircle size={14} /> Abort Stream
          </button>
        </div>
      )}

      {/* Phase 3 Idempotency State Visualizer */}
      {commitMutation.idempotencyKey && (
        <div style={{ marginTop: '1rem', padding: '0.75rem', background: '#090d16', borderRadius: '0.5rem', border: '1px solid var(--border-color)', fontSize: '0.8rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.25rem' }}>
            <ShieldCheck size={16} className="text-accent-blue" />
            <strong style={{ color: 'var(--accent-blue)' }}>Idempotency-Key Active:</strong>
            <code style={{ color: 'var(--accent-amber)' }}>{commitMutation.idempotencyKey}</code>
          </div>
          <div style={{ color: 'var(--text-secondary)', fontSize: '0.75rem' }}>
            Retries: {commitMutation.retryCount} | Replay: {commitMutation.isReplay ? 'YES (Cached)' : 'NO (Fresh)'}
          </div>
        </div>
      )}

      {/* Pipeline Error & Network Drop Recovery */}
      {(pipelineError || commitMutation.error) && (
        <div style={{ marginTop: '1rem', padding: '0.75rem', background: 'rgba(244, 63, 94, 0.1)', border: '1px solid var(--accent-rose)', borderRadius: '0.5rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: 'var(--accent-rose)', fontWeight: 600, fontSize: '0.85rem' }}>
            <AlertTriangle size={18} />
            Network Failure / Post-Commit Drop Detected
          </div>
          <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginTop: '0.25rem' }}>
            {pipelineError || commitMutation.error}
          </p>
          {commitMutation.idempotencyKey && (
            <button
              type="button"
              className="btn btn-success"
              onClick={handleRetryCommit}
              style={{ marginTop: '0.75rem' }}
            >
              <RotateCw size={16} />
              Retry Commit with Same Key ({commitMutation.idempotencyKey.slice(0, 8)}...)
            </button>
          )}
        </div>
      )}

      {/* Action Button */}
      <div style={{ marginTop: '1.5rem', display: 'flex', gap: '0.75rem' }}>
        <button
          type="button"
          className="btn btn-primary"
          onClick={handleStartIngestion}
          disabled={!selectedFile || uploader.isUploading || commitMutation.isLoading}
        >
          <FileCheck size={18} />
          {uploader.isUploading
            ? 'Streaming to MinIO...'
            : commitMutation.isLoading
            ? 'Committing with Idempotency-Key...'
            : 'Start 3-Phase Ingestion'}
        </button>
      </div>
    </div>
  );
};
