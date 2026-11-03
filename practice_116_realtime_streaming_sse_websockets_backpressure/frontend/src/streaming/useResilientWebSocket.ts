/**
 * Exercise 3: Resilient WebSocket Manager with Heartbeats & Full-Jitter Backoff
 *
 * Implements a production-grade WebSocket client featuring:
 * - Connection state machine
 * - Full-jitter exponential backoff reconnection
 * - Heartbeat ping/pong watchdog to detect dead TCP sockets
 * - Bounded outbound message queue for offline buffering
 */

import { useState, useRef, useEffect, useCallback } from 'react';
import type { WebSocketState, PingMessage, PongMessage } from '../types';

export interface UseResilientWebSocketOptions {
  url: string;
  autoConnect?: boolean;
  baseBackoffMs?: number;
  maxBackoffMs?: number;
  heartbeatIntervalMs?: number;
  heartbeatTimeoutMs?: number;
  maxOfflineQueueSize?: number;
  onMessage?: (data: unknown) => void;
  onStateChange?: (state: WebSocketState) => void;
}

export interface UseResilientWebSocketReturn {
  state: WebSocketState;
  send: (data: unknown) => boolean;
  latencyMs: number | null;
  reconnectAttempts: number;
  connect: () => void;
  disconnect: () => void;
}

export function useResilientWebSocket({
  url,
  autoConnect = true,
  baseBackoffMs = 500,
  maxBackoffMs = 10000,
  heartbeatIntervalMs = 5000,
  heartbeatTimeoutMs = 3500,
  maxOfflineQueueSize = 100,
  onMessage,
  onStateChange,
}: UseResilientWebSocketOptions): UseResilientWebSocketReturn {
  const [state, setState] = useState<WebSocketState>('CLOSED');
  const [latencyMs, setLatencyMs] = useState<number | null>(null);
  const [reconnectAttempts, setReconnectAttempts] = useState<number>(0);

  const socketRef = useRef<WebSocket | null>(null);
  const reconnectTimerRef = useRef<number | null>(null);
  const heartbeatTimerRef = useRef<number | null>(null);
  const heartbeatTimeoutTimerRef = useRef<number | null>(null);
  const pingSeqRef = useRef<number>(0);
  const offlineQueueRef = useRef<string[]>([]);
  const isManuallyClosedRef = useRef<boolean>(false);

  const updateState = useCallback((newState: WebSocketState) => {
    setState(newState);
    onStateChange?.(newState);
  }, [onStateChange]);

  const clearTimers = useCallback(() => {
    if (reconnectTimerRef.current !== null) {
      clearTimeout(reconnectTimerRef.current);
      reconnectTimerRef.current = null;
    }
    if (heartbeatTimerRef.current !== null) {
      clearInterval(heartbeatTimerRef.current);
      heartbeatTimerRef.current = null;
    }
    if (heartbeatTimeoutTimerRef.current !== null) {
      clearTimeout(heartbeatTimeoutTimerRef.current);
      heartbeatTimeoutTimerRef.current = null;
    }
  }, []);

  const calculateFullJitterBackoff = useCallback((attempt: number): number => {
    // --- TODO(human): Exercise 3 (Full Jitter Exponential Backoff) ---
    // Task:
    // Calculate exponential backoff delay with full randomization (jitter):
    //   ceiling = Math.min(maxBackoffMs, baseBackoffMs * Math.pow(2, attempt))
    //   delay = Math.random() * ceiling
    // Returns delay in milliseconds.
    const ceiling = Math.min(maxBackoffMs, baseBackoffMs * Math.pow(2, attempt));
    return Math.floor(Math.random() * ceiling);
  }, [baseBackoffMs, maxBackoffMs]);

  const send = useCallback((data: unknown): boolean => {
    const raw = typeof data === 'string' ? data : JSON.stringify(data);
    if (socketRef.current && socketRef.current.readyState === WebSocket.OPEN) {
      socketRef.current.send(raw);
      return true;
    }

    // Queue for when connection recovers
    if (offlineQueueRef.current.length < maxOfflineQueueSize) {
      offlineQueueRef.current.push(raw);
      return true;
    } else {
      console.warn('[useResilientWebSocket] Offline queue full. Dropping message.');
      return false;
    }
  }, [maxOfflineQueueSize]);

  const startHeartbeat = useCallback(() => {
    if (heartbeatTimerRef.current) clearInterval(heartbeatTimerRef.current);

    heartbeatTimerRef.current = window.setInterval(() => {
      if (socketRef.current && socketRef.current.readyState === WebSocket.OPEN) {
        pingSeqRef.current++;
        const ping: PingMessage = {
          type: 'PING',
          timestamp: Date.now(),
          seq: pingSeqRef.current,
        };

        socketRef.current.send(JSON.stringify(ping));

        // Start watchdog timer: if no pong within heartbeatTimeoutMs, socket is dead
        if (heartbeatTimeoutTimerRef.current) clearTimeout(heartbeatTimeoutTimerRef.current);
        heartbeatTimeoutTimerRef.current = window.setTimeout(() => {
          console.warn('[useResilientWebSocket] Heartbeat timeout! Zombie socket detected. Closing...');
          if (socketRef.current) {
            socketRef.current.close(4001, 'Heartbeat Timeout');
          }
        }, heartbeatTimeoutMs);
      }
    }, heartbeatIntervalMs);
  }, [heartbeatIntervalMs, heartbeatTimeoutMs]);

  const connect = useCallback(() => {
    clearTimers();
    isManuallyClosedRef.current = false;

    // Resolve ws/wss protocol and relative URL
    let fullUrl = url;
    if (url.startsWith('/')) {
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      fullUrl = `${protocol}//${window.location.host}${url}`;
    }

    updateState('CONNECTING');

    try {
      const ws = new WebSocket(fullUrl);
      socketRef.current = ws;

      ws.onopen = () => {
        updateState('OPEN');
        setReconnectAttempts(0);
        startHeartbeat();

        // Drain offline queue
        while (offlineQueueRef.current.length > 0 && ws.readyState === WebSocket.OPEN) {
          const queued = offlineQueueRef.current.shift();
          if (queued) ws.send(queued);
        }
      };

      ws.onmessage = (event: MessageEvent) => {
        try {
          const parsed = JSON.parse(event.data);
          // Check for heartbeat PONG
          if (parsed && parsed.type === 'PONG') {
            if (heartbeatTimeoutTimerRef.current) {
              clearTimeout(heartbeatTimeoutTimerRef.current);
              heartbeatTimeoutTimerRef.current = null;
            }
            const pong = parsed as PongMessage;
            if (pong.client_timestamp) {
              setLatencyMs(Date.now() - pong.client_timestamp);
            }
            return;
          }

          onMessage?.(parsed);
        } catch {
          onMessage?.(event.data);
        }
      };

      ws.onclose = (event: CloseEvent) => {
        clearTimers();
        socketRef.current = null;

        if (isManuallyClosedRef.current) {
          updateState('CLOSED');
          return;
        }

        updateState('RECONNECTING');
        setReconnectAttempts(prev => {
          const nextAttempt = prev + 1;
          const delay = calculateFullJitterBackoff(nextAttempt);
          console.log(`[useResilientWebSocket] Disconnected (${event.code}). Reconnecting attempt #${nextAttempt} in ${delay}ms...`);

          reconnectTimerRef.current = window.setTimeout(() => {
            connect();
          }, delay);

          return nextAttempt;
        });
      };

      ws.onerror = (err) => {
        console.error('[useResilientWebSocket] Error:', err);
      };
    } catch (err) {
      console.error('[useResilientWebSocket] Connection initialization failed:', err);
      updateState('CLOSED');
    }
  }, [url, updateState, startHeartbeat, clearTimers, calculateFullJitterBackoff, onMessage]);

  const disconnect = useCallback(() => {
    isManuallyClosedRef.current = true;
    clearTimers();
    if (socketRef.current) {
      updateState('CLOSING');
      socketRef.current.close(1000, 'User Disconnect');
      socketRef.current = null;
    }
    updateState('CLOSED');
    setLatencyMs(null);
  }, [clearTimers, updateState]);

  useEffect(() => {
    if (autoConnect) {
      connect();
    }
    return () => {
      disconnect();
    };
  }, [autoConnect, connect, disconnect]);

  return {
    state,
    send,
    latencyMs,
    reconnectAttempts,
    connect,
    disconnect,
  };
}
