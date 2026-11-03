import { LogRecord, LogLevel, ServiceTag, PartitionId } from '../types';

/**
 * Fast Mulberry32 PRNG for deterministic, reproducible synthetic datasets.
 */
function createPrng(seed: number) {
  let s = seed | 0;
  return function next(): number {
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const SERVICES: ServiceTag[] = [
  'scheduler-engine',
  'order-router',
  'inventory-sync',
  'dispatch-gateway',
];

const LOG_TEMPLATES: Record<ServiceTag, string[]> = {
  'scheduler-engine': [
    'Optimization solver converged in {latency}ms for wave #{num}',
    'Dock door allocation conflict detected between trailer #{num} and #{num2}',
    'Heuristic dispatch recalculation scheduled for zone #{num}',
    'MILP constraint relaxed: buffer capacity exceeded in bay #{num}',
  ],
  'order-router': [
    'Order #{num} routing plan evaluated across {num2} carriers',
    'High priority fulfillment path assigned to shipment #{num}',
    'Order validation retry triggered: stock reservation timeout on SKU #{num}',
    'Batch picking manifest generated for cart #{num}',
  ],
  'inventory-sync': [
    'Cycle count reconciliation complete for aisle #{num}',
    'Discrepancy detected in automated storage buffer bin #{num}',
    'RFID gate sensor telemetry packet ingested: tag #{num}',
    'Cross-dock pallet inventory updated in storage rack #{num}',
  ],
  'dispatch-gateway': [
    'Autonomous forklift telemetry beacon received at waypoint #{num}',
    'Outbound yard slot assignment verified for driver ID #{num}',
    'Conveyor divert motor temperature warning on scanner array #{num}',
    'Gatekeeper webhook delivered payload for bill of lading #{num}',
  ],
};

const SEVERITIES: LogLevel[] = ['INFO', 'WARN', 'ERROR', 'CRITICAL'];

/**
 * Procedurally generates an array of LogRecord items.
 *
 * @param count - Number of logs to generate (e.g. 50,000)
 * @param partition - Target warehouse partition
 * @param seed - PRNG seed for deterministic test stability
 */
export function generateSyntheticLogs(
  count: number = 50000,
  partition: PartitionId = 'warehouse-us-east',
  seed: number = 42
): LogRecord[] {
  const prng = createPrng(seed);
  const logs: LogRecord[] = new Array(count);
  const baseTime = Date.now() - 24 * 3600 * 1000;

  for (let i = 0; i < count; i++) {
    const serviceIndex = Math.floor(prng() * SERVICES.length);
    const service = SERVICES[serviceIndex];

    const sevRoll = prng();
    let level: LogLevel = 'INFO';
    if (sevRoll > 0.97) level = 'CRITICAL';
    else if (sevRoll > 0.88) level = 'ERROR';
    else if (sevRoll > 0.70) level = 'WARN';

    const templates = LOG_TEMPLATES[service];
    const template = templates[Math.floor(prng() * templates.length)];
    const num1 = Math.floor(prng() * 9000 + 1000);
    const num2 = Math.floor(prng() * 90 + 10);
    const latency = Math.floor(prng() * 450 + 5);

    const message = template
      .replace('{latency}', String(latency))
      .replace('{num}', String(num1))
      .replace('{num2}', String(num2));

    const anomalyScore = parseFloat((prng() * (level === 'CRITICAL' || level === 'ERROR' ? 0.6 : 0.2) + (level === 'CRITICAL' ? 0.4 : 0.0)).toFixed(3));

    logs[i] = {
      id: `log-${partition}-${i + 1}`,
      timestamp: new Date(baseTime + i * 1500).toISOString(),
      level,
      service,
      message,
      latencyMs: latency,
      anomalyScore: Math.min(1.0, anomalyScore),
      partition,
    };
  }

  return logs;
}
