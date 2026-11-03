import { http, HttpResponse, delay } from 'msw';
import { generateSyntheticLogs } from '../data/generateSyntheticLogs';
import { PartitionId, LogLevel, SEVERITY_RANKS } from '../workers/datasetWorker';

// In-memory cache of generated logs per partition for test determinism
const partitionCache: Record<string, ReturnType<typeof generateSyntheticLogs>> = {};

function getLogsForPartition(partition: PartitionId) {
  if (!partitionCache[partition]) {
    // Generate 1,000 logs for fast, predictable test execution
    partitionCache[partition] = generateSyntheticLogs(1000, partition);
  }
  return partitionCache[partition];
}

export const handlers = [
  // =========================================================================
  // TODO(human): Exercise 3 - Define MSW v2 Network Interceptors
  // =========================================================================
  // Instructions:
  // 1. Intercept GET requests matching '*/api/logs'.
  // 2. Parse query parameters from request.url:
  //    - 'partition' (default to 'warehouse-us-east')
  //    - 'minSeverity' (optional LogLevel)
  //    - 'delayMs' (optional artificial network latency)
  //    - 'simulateError' ('500' returns internal error)
  //    - 'simulateRateLimit' ('429' returns rate limit with Retry-After header)
  // 3. Return real HTTP responses using HttpResponse.json(body, init).
  http.get('*/api/logs', async ({ request }) => {
    const url = new URL(request.url);
    const partition = (url.searchParams.get('partition') as PartitionId) || 'warehouse-us-east';
    const minSeverity = url.searchParams.get('minSeverity') as LogLevel | null;
    const delayMs = parseInt(url.searchParams.get('delayMs') || '0', 10);
    const simulateError = url.searchParams.get('simulateError');
    const simulateRateLimit = url.searchParams.get('simulateRateLimit');

    if (delayMs > 0) {
      await delay(delayMs);
    }

    if (simulateError === '500') {
      return HttpResponse.json(
        { error: 'Internal Server Error: Database cluster unreachable' },
        { status: 500 }
      );
    }

    if (simulateRateLimit === '429') {
      return HttpResponse.json(
        { error: 'Rate limit exceeded: Too Many Requests', retryAfterSeconds: 5 },
        {
          status: 429,
          headers: {
            'Retry-After': '5',
            'Content-Type': 'application/json',
          },
        }
      );
    }

    let records = getLogsForPartition(partition);

    if (minSeverity && SEVERITY_RANKS[minSeverity] !== undefined) {
      const minRank = SEVERITY_RANKS[minSeverity];
      records = records.filter((r) => SEVERITY_RANKS[r.level] >= minRank);
    }

    return HttpResponse.json({
      partition,
      total: records.length,
      logs: records,
    });
  }),

  http.post('*/api/logs/annotate', async ({ request }) => {
    const body = (await request.json()) as { logId: string; annotation: string };
    return HttpResponse.json({
      success: true,
      logId: body.logId,
      annotatedAt: new Date().toISOString(),
    });
  }),
];
