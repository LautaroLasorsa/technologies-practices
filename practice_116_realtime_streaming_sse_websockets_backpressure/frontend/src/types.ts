/**
 * Shared Type Definitions for Practice 116.
 */

export interface TelemetryPacket {
  type: 'TELEMETRY';
  iteration: number;
  objective: number;
  duality_gap: number;
  temperature: number;
  hz: number;
  timestamp: number;
}

export interface ReasoningToken {
  token: string;
  step: number;
  id: number;
  timestamp: number;
}

export interface SseRawEvent {
  id?: string;
  event: string;
  data: string;
}

export type WebSocketState = 'CONNECTING' | 'OPEN' | 'CLOSING' | 'CLOSED' | 'RECONNECTING';

export type IngestionMode = 'naive' | 'backpressure';

export interface PingMessage {
  type: 'PING';
  timestamp: number;
  seq: number;
}

export interface PongMessage {
  type: 'PONG';
  client_timestamp: number;
  server_timestamp: number;
  seq: number;
}

export interface SetRateMessage {
  type: 'SET_RATE';
  hz: number;
}
