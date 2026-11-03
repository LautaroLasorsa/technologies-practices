/**
 * React Hook for Direct S3/MinIO Binary Streaming via XMLHttpRequest.
 *
 * Why XMLHttpRequest over fetch?
 * The browser Fetch API (`window.fetch`) does not expose standard upload
 * progress events (`onprogress`) for request bodies in stable web specs.
 * `XMLHttpRequest.upload.onprogress` provides real-time byte tracking,
 * transfer throughput calculation (KB/s), and instant client-side cancellation.
 */

import { useCallback, useRef, useState } from 'react';

export interface UploadProgress {
  progress: number; // 0 to 100
  loadedBytes: number;
  totalBytes: number;
  speedBytesPerSec: number;
}

export interface UsePresignedUploadReturn {
  upload: (uploadUrl: string, file: File, contentType?: string) => Promise<{ etag: string }>;
  abort: () => void;
  reset: () => void;
  isUploading: boolean;
  progress: number;
  loadedBytes: number;
  totalBytes: number;
  speedBytesPerSec: number;
  etag: string | null;
  error: string | null;
}

export function usePresignedUpload(): UsePresignedUploadReturn {
  const [isUploading, setIsUploading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [loadedBytes, setLoadedBytes] = useState(0);
  const [totalBytes, setTotalBytes] = useState(0);
  const [speedBytesPerSec, setSpeedBytesPerSec] = useState(0);
  const [etag, setEtag] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const xhrRef = useRef<XMLHttpRequest | null>(null);

  const abort = useCallback(() => {
    if (xhrRef.current) {
      xhrRef.current.abort();
      xhrRef.current = null;
      setIsUploading(false);
      setError('Upload aborted by user.');
    }
  }, []);

  const reset = useCallback(() => {
    abort();
    setIsUploading(false);
    setProgress(0);
    setLoadedBytes(0);
    setTotalBytes(0);
    setSpeedBytesPerSec(0);
    setEtag(null);
    setError(null);
  }, [abort]);

  /**
   * Stream a binary file directly to S3/MinIO using XMLHttpRequest.
   *
   * ========================================================================
   * TODO(human): Exercise 3 (Frontend) - Direct XHR Binary Uploader
   * ========================================================================
   * Context:
   * By streaming directly to MinIO, our backend avoids proxying heavy binaries.
   * We use XHR to capture byte progress and extract MinIO's returned ETag.
   *
   * Instructions:
   * 1. Initialize `const xhr = new XMLHttpRequest()`. Store in `xhrRef.current = xhr`.
   * 2. Track upload speed:
   *    Let `lastTime = performance.now()`, `lastLoaded = 0`.
   * 3. Attach progress handler:
   *    `xhr.upload.onprogress = (event) => {`
   *       if (event.lengthComputable) {
   *         const pct = Math.round((event.loaded / event.total) * 100);
   *         setProgress(pct);
   *         setLoadedBytes(event.loaded);
   *         setTotalBytes(event.total);
   *         // Compute speed: bytes / elapsed seconds
   *         const now = performance.now();
   *         const elapsedSec = (now - lastTime) / 1000;
   *         if (elapsedSec > 0.2) {
   *           setSpeedBytesPerSec(Math.round((event.loaded - lastLoaded) / elapsedSec));
   *           lastTime = now;
   *           lastLoaded = event.loaded;
   *         }
   *       }
   *    `};`
   * 4. Wrap the execution in a Promise:
   *    - In `xhr.onload`:
   *        - If `xhr.status >= 200 && xhr.status < 300`:
   *            Extract ETag: `const rawEtag = xhr.getResponseHeader('ETag') || '';`
   *            Clean quotes: `const cleanEtag = rawEtag.replace(/"/g, '');`
   *            Set state, resolve `{ etag: cleanEtag }`.
   *        - Else: reject with Error including `xhr.status` and `xhr.statusText`.
   *    - In `xhr.onerror`: reject with Error('Direct upload network error').
   *    - In `xhr.onabort`: reject with Error('Upload aborted').
   * 5. Open and send:
   *    `xhr.open('PUT', uploadUrl, true);`
   *    `xhr.setRequestHeader('Content-Type', contentType || file.type || 'application/octet-stream');`
   *    `xhr.send(file);`
   * ========================================================================
   */
  const upload = useCallback(
    async (
      uploadUrl: string,
      file: File,
      contentType?: string
    ): Promise<{ etag: string }> => {
      reset();
      setIsUploading(true);
      setError(null);

      // --- IMPLEMENTATION GOES HERE ---
      throw new Error(
        'Exercise 3: Implement direct XHR upload in frontend/src/hooks/usePresignedUpload.ts'
      );
    },
    [reset]
  );

  return {
    upload,
    abort,
    reset,
    isUploading,
    progress,
    loadedBytes,
    totalBytes,
    speedBytesPerSec,
    etag,
    error,
  };
}
