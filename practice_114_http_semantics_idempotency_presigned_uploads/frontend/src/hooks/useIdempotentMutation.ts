/**
 * React Hook for Idempotent HTTP Mutations.
 *
 * Implements the client-side lifecycle of the Idempotency-Key protocol:
 * 1. Generates and persists a deterministic UUID per logical user action.
 * 2. Injects the `Idempotency-Key` HTTP header.
 * 3. Preserves the key across retries when network or 5xx server errors occur.
 * 4. Rotates / clears the key ONLY upon terminal 2xx success or 4xx client errors.
 */

import { useCallback, useRef, useState } from 'react';

export interface UseIdempotentMutationReturn<TData, TPayload> {
  mutate: (url: string, payload: TPayload) => Promise<TData>;
  retry: () => Promise<TData | undefined>;
  reset: () => void;
  idempotencyKey: string | null;
  isLoading: boolean;
  isSuccess: boolean;
  isReplay: boolean;
  retryCount: number;
  error: string | null;
  data: TData | null;
}

export function useIdempotentMutation<TData = unknown, TPayload = unknown>(): UseIdempotentMutationReturn<
  TData,
  TPayload
> {
  const [idempotencyKey, setIdempotencyKey] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isSuccess, setIsSuccess] = useState(false);
  const [isReplay, setIsReplay] = useState(false);
  const [retryCount, setRetryCount] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<TData | null>(null);

  // Store last invocation parameters to support manual retry
  const lastCallRef = useRef<{ url: string; payload: TPayload } | null>(null);
  const keyRef = useRef<string | null>(null);

  /**
   * Helper to ensure a valid UUID v4 exists for the current logical mutation.
   */
  const getOrCreateKey = useCallback((): string => {
    if (!keyRef.current) {
      const newKey = crypto.randomUUID();
      keyRef.current = newKey;
      setIdempotencyKey(newKey);
      return newKey;
    }
    return keyRef.current;
  }, []);

  const clearKey = useCallback(() => {
    keyRef.current = null;
    setIdempotencyKey(null);
  }, []);

  /**
   * Execute the idempotent mutation against the target URL.
   *
   * ========================================================================
   * TODO(human): Exercise 1 (Frontend) - Client-side Idempotency Lifecycle
   * ========================================================================
   * Context:
   * A client must guarantee that retried requests (due to timeouts, 503s,
   * or dropped connections) send the EXACT SAME `Idempotency-Key`, while
   * fresh user actions generate a new UUID.
   *
   * Instructions:
   * 1. Obtain active key using `const activeKey = getOrCreateKey()`.
   * 2. Send fetch request:
   *    const response = await fetch(url, {
   *      method: 'POST',
   *      headers: {
   *        'Content-Type': 'application/json',
   *        'Idempotency-Key': activeKey,
   *      },
   *      body: JSON.stringify(payload),
   *    });
   * 3. Evaluate response status:
   *    - If `response.ok` (2xx):
   *        - Check header `response.headers.get('Idempotent-Replay') === 'true'`.
   *        - Parse JSON body.
   *        - Terminal success reached: call `clearKey()`, reset `retryCount`,
   *          set `isSuccess = true`, set `data`.
   *        - Return parsed data.
   *    - If `response.status >= 400 && response.status < 500`:
   *        - Client-side error (e.g. 400 bad request, 422 unprocessable).
   *        - Terminal client failure: call `clearKey()`, throw Error.
   *    - If `response.status >= 500` or if `fetch` throws a Network Error:
   *        - Transient server error or connection drop.
   *        - DO NOT CALL clearKey()! Leave keyRef.current intact so retries
   *          will reuse the exact same key!
   *        - Increment `retryCount`, update `error`, and rethrow Error.
   * ========================================================================
   */
  const mutate = useCallback(
    async (url: string, payload: TPayload): Promise<TData> => {
      lastCallRef.current = { url, payload };
      setIsLoading(true);
      setError(null);

      // --- IMPLEMENTATION GOES HERE ---
      throw new Error(
        'Exercise 1: Implement idempotent mutate lifecycle in frontend/src/hooks/useIdempotentMutation.ts'
      );
    },
    [getOrCreateKey, clearKey]
  );

  const retry = useCallback(async (): Promise<TData | undefined> => {
    if (!lastCallRef.current) return;
    setRetryCount((prev) => prev + 1);
    return mutate(lastCallRef.current.url, lastCallRef.current.payload);
  }, [mutate]);

  const reset = useCallback(() => {
    clearKey();
    setIsLoading(false);
    setIsSuccess(false);
    setIsReplay(false);
    setRetryCount(0);
    setError(null);
    setData(null);
    lastCallRef.current = null;
  }, [clearKey]);

  return {
    mutate,
    retry,
    reset,
    idempotencyKey,
    isLoading,
    isSuccess,
    isReplay,
    retryCount,
    error,
    data,
  };
}
