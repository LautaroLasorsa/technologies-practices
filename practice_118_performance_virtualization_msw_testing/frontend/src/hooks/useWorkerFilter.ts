import { useState, useEffect, useRef } from 'react';
import {
  LogRecord,
  FilterCriteria,
  WorkerFilterRequest,
  WorkerFilterResponse,
} from '../types';
import { executeFilter } from '../workers/datasetWorker';

export interface UseWorkerFilterResult {
  filteredLogs: LogRecord[];
  isComputing: boolean;
  computeTimeMs: number;
  totalMatched: number;
}

export function useWorkerFilter(
  logs: LogRecord[],
  criteria: FilterCriteria
): UseWorkerFilterResult {
  const [filteredLogs, setFilteredLogs] = useState<LogRecord[]>(logs);
  const [isComputing, setIsComputing] = useState<boolean>(false);
  const [computeTimeMs, setComputeTimeMs] = useState<number>(0);

  // Tracks the ID of the most recently initiated filter request
  const latestRequestIdRef = useRef<number>(0);
  const workerRef = useRef<Worker | null>(null);

  useEffect(() => {
    // Instantiate the Web Worker if supported in the current environment
    if (typeof Worker !== 'undefined' && !workerRef.current) {
      try {
        workerRef.current = new Worker(
          new URL('../workers/datasetWorker.ts', import.meta.url),
          { type: 'module' }
        );

        workerRef.current.onmessage = (
          event: MessageEvent<WorkerFilterResponse>
        ) => {
          const { requestId, filteredLogs, computeTimeMs } = event.data;

          // =========================================================================
          // TODO(human): Exercise 1 - Invalidate stale worker responses
          // =========================================================================
          // Instructions:
          // If the requestId does not match latestRequestIdRef.current, the response is
          // stale (e.g. from an earlier keystroke). Drop it without updating state.
          // Otherwise, update filteredLogs, computeTimeMs, and set isComputing to false.
          if (requestId === latestRequestIdRef.current) {
            setFilteredLogs(filteredLogs);
            setComputeTimeMs(computeTimeMs);
            setIsComputing(false);
          }
        };
      } catch (err) {
        console.warn('Web Worker creation failed; falling back to asynchronous execution', err);
        workerRef.current = null;
      }
    }

    return () => {
      workerRef.current?.terminate();
      workerRef.current = null;
    };
  }, []);

  useEffect(() => {
    // =========================================================================
    // TODO(human): Exercise 1 - Dispatch filter request with incremented requestId
    // =========================================================================
    // Instructions:
    // 1. Increment latestRequestIdRef.current.
    // 2. Set isComputing to true.
    // 3. If workerRef.current is available, postMessage a WorkerFilterRequest.
    // 4. If no worker is available (e.g. inside JSDOM/Vitest), execute asynchronously
    //    via setTimeout/queueMicrotask, still verifying requestId before updating state.
    const requestId = ++latestRequestIdRef.current;
    setIsComputing(true);

    if (workerRef.current) {
      const request: WorkerFilterRequest = {
        type: 'FILTER_DATASET',
        requestId,
        logs,
        criteria,
      };
      workerRef.current.postMessage(request);
    } else {
      // Node/JSDOM fallback environment
      const timer = setTimeout(() => {
        const start = performance.now();
        const result = executeFilter(logs, criteria);
        const duration = performance.now() - start;

        if (requestId === latestRequestIdRef.current) {
          setFilteredLogs(result);
          setComputeTimeMs(duration);
          setIsComputing(false);
        }
      }, 0);

      return () => clearTimeout(timer);
    }
  }, [logs, criteria]);

  return {
    filteredLogs,
    isComputing,
    computeTimeMs,
    totalMatched: filteredLogs.length,
  };
}
