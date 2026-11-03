import {
  LogRecord,
  FilterCriteria,
  LogLevel,
  WorkerFilterRequest,
  WorkerFilterResponse,
} from '../types';

export const SEVERITY_RANKS: Record<LogLevel, number> = {
  INFO: 0,
  WARN: 1,
  ERROR: 2,
  CRITICAL: 3,
};

/**
 * Pure computation function that filters 50,000+ log records.
 *
 * Runs inside the Web Worker thread to avoid starving the browser's UI thread.
 */
export function executeFilter(
  logs: LogRecord[],
  criteria: FilterCriteria
): LogRecord[] {
  // =========================================================================
  // TODO(human): Exercise 1 - Offload heavy CPU filtering to Web Worker
  // =========================================================================
  // Instructions:
  // 1. Check if criteria.query is provided. If so, build a case-insensitive
  //    RegExp (or perform case-insensitive substring checks) against log.message
  //    and log.service.
  // 2. Check criteria.minSeverity. Filter logs whose SEVERITY_RANKS[log.level]
  //    is greater than or equal to SEVERITY_RANKS[criteria.minSeverity].
  // 3. Check criteria.service. If specified and not 'ALL', filter by log.service.
  // 4. Check criteria.minAnomalyScore. If specified, filter by log.anomalyScore >= minAnomalyScore.
  // 5. Return the filtered array.
  //
  // Replace the placeholder below with your implementation:
  const queryStr = criteria.query?.trim().toLowerCase() ?? '';
  const minSevRank = criteria.minSeverity ? SEVERITY_RANKS[criteria.minSeverity] : 0;
  const targetService = criteria.service && criteria.service !== 'ALL' ? criteria.service : null;
  const minScore = criteria.minAnomalyScore ?? 0;

  return logs.filter((log) => {
    if (SEVERITY_RANKS[log.level] < minSevRank) return false;
    if (targetService && log.service !== targetService) return false;
    if (log.anomalyScore < minScore) return false;
    if (queryStr) {
      const msgMatch = log.message.toLowerCase().includes(queryStr);
      const svcMatch = log.service.toLowerCase().includes(queryStr);
      const idMatch = log.id.toLowerCase().includes(queryStr);
      if (!msgMatch && !svcMatch && !idMatch) return false;
    }
    return true;
  });
}

// Web Worker message listener (active when running in a worker context)
if (typeof self !== 'undefined' && typeof self.postMessage === 'function') {
  self.onmessage = (event: MessageEvent<WorkerFilterRequest>) => {
    const { requestId, logs, criteria } = event.data;
    const start = performance.now();

    const filteredLogs = executeFilter(logs, criteria);

    const computeTimeMs = performance.now() - start;

    const response: WorkerFilterResponse = {
      type: 'FILTER_RESULT',
      requestId,
      filteredLogs,
      totalMatched: filteredLogs.length,
      computeTimeMs,
    };

    self.postMessage(response);
  };
}
