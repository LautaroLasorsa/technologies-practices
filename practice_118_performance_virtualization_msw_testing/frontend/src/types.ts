export type LogLevel = 'INFO' | 'WARN' | 'ERROR' | 'CRITICAL';

export type ServiceTag =
  | 'scheduler-engine'
  | 'order-router'
  | 'inventory-sync'
  | 'dispatch-gateway';

export type PartitionId =
  | 'warehouse-us-east'
  | 'warehouse-eu-central'
  | 'warehouse-apac';

export interface LogRecord {
  id: string;
  timestamp: string;
  level: LogLevel;
  service: ServiceTag;
  message: string;
  latencyMs: number;
  anomalyScore: number; // 0.00 to 1.00
  partition: PartitionId;
}

export interface FilterCriteria {
  query: string;
  minSeverity?: LogLevel;
  service?: ServiceTag | 'ALL';
  minAnomalyScore?: number;
}

export interface WorkerFilterRequest {
  type: 'FILTER_DATASET';
  requestId: number;
  logs: LogRecord[];
  criteria: FilterCriteria;
}

export interface WorkerFilterResponse {
  type: 'FILTER_RESULT';
  requestId: number;
  filteredLogs: LogRecord[];
  totalMatched: number;
  computeTimeMs: number;
}

export interface GetLogsResponse {
  partition: PartitionId;
  total: number;
  logs: LogRecord[];
}

export interface ApiErrorResponse {
  error: string;
  retryAfterSeconds?: number;
}
