/**
 * Single-Flight Refresh Mutex & In-Memory Token Client
 *
 * Solves the Concurrent 401 Thundering Herd problem in SPAs:
 * Multiple parallel requests receiving 401 must queue behind a SINGLE
 * refresh request. Otherwise, concurrent /api/auth/refresh calls violate
 * Refresh Token Rotation and trigger a false-positive security lockout!
 */

export interface AuthTraceEvent {
  id: string;
  timestamp: string;
  source: 'REQUEST' | 'MUTEX' | 'REFRESH' | 'RETRY' | 'REVOKED' | 'PKCE';
  message: string;
  details?: any;
}

export type TraceListener = (event: AuthTraceEvent) => void;

// ----------------------------------------------------------------------------
// In-Memory Token Storage Closure (Immune to XSS token exfiltration)
// ----------------------------------------------------------------------------
let currentAccessToken: string | null = null;
let tokenExpiresAt: number = 0;
const traceListeners: Set<TraceListener> = new Set();

export function emitTrace(source: AuthTraceEvent['source'], message: string, details?: any) {
  const event: AuthTraceEvent = {
    id: `trace_${Math.random().toString(36).slice(2, 9)}`,
    timestamp: new Date().toLocaleTimeString(),
    source,
    message,
    details,
  };
  traceListeners.forEach((listener) => listener(event));
  console.log(`[${event.source}] ${event.message}`, details ?? '');
}

export function subscribeToTrace(listener: TraceListener): () => void {
  traceListeners.add(listener);
  return () => traceListeners.delete(listener);
}

export function getAccessToken(): string | null {
  return currentAccessToken;
}

export function getTokenExpiresAt(): number {
  return tokenExpiresAt;
}

export function setAccessToken(token: string, expiresInSeconds: number): void {
  currentAccessToken = token;
  tokenExpiresAt = Date.now() + expiresInSeconds * 1000;
  emitTrace('REQUEST', `In-memory access token updated (expires in ${expiresInSeconds}s)`);
}

export function clearAccessToken(): void {
  currentAccessToken = null;
  tokenExpiresAt = 0;
  emitTrace('REQUEST', 'In-memory access token cleared');
}

// ----------------------------------------------------------------------------
// Refresh Mutex State
// ----------------------------------------------------------------------------

/**
 * When a refresh is in-flight, this holds the single active Promise.
 * All parallel 401 responses wait on this exact promise instead of firing
 * duplicate /api/auth/refresh calls.
 */
let refreshPromise: Promise<string | null> | null = null;

/**
 * Low-level call to backend /api/auth/refresh using HttpOnly cookie.
 */
export async function performTokenRefresh(simulatedStolenToken?: string): Promise<string | null> {
  emitTrace('REFRESH', 'Sending single /api/auth/refresh call to server...');

  try {
    const res = await fetch('/api/auth/refresh', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      credentials: 'include', // Includes HttpOnly cookie
      body: simulatedStolenToken ? JSON.stringify({ simulatedStolenToken }) : undefined,
    });

    if (!res.ok) {
      const errData = await res.json().catch(() => ({}));
      if (res.status === 401 && errData.error === 'token_reuse_detected') {
        emitTrace('REVOKED', '🚨 TOKEN REUSE DETECTED! Family revoked by server.', errData);
        clearAccessToken();
        // Dispatch custom event for UI reaction
        window.dispatchEvent(new CustomEvent('auth:reuse_detected', { detail: errData }));
        return null;
      }

      emitTrace('REFRESH', `Refresh failed with status ${res.status}`, errData);
      clearAccessToken();
      window.dispatchEvent(new CustomEvent('auth:session_expired'));
      return null;
    }

    const data = await res.json();
    setAccessToken(data.access_token, data.expires_in);
    emitTrace('REFRESH', '✅ Refresh succeeded! New access token stored in memory.');
    return data.access_token;
  } catch (err: any) {
    emitTrace('REFRESH', `Network error during refresh: ${err.message}`);
    clearAccessToken();
    return null;
  }
}

// ----------------------------------------------------------------------------
// Exercise 3: Authenticated Fetch with Single-Flight Refresh Mutex
// ----------------------------------------------------------------------------

/**
 * Exercise 3: TODO(human)
 *
 * Implement `authenticatedFetch` with a Single-Flight Mutex for 401 handling.
 *
 * Problem:
 * When a dashboard loads, 5 widgets may fire parallel fetch requests simultaneously.
 * If the access token expired 2 seconds ago, ALL 5 requests will return 401 Unauthorized.
 * If each request naively calls /api/auth/refresh:
 *   - 5 requests call the refresh endpoint concurrently.
 *   - The 1st refresh request consumes the rotating refresh token.
 *   - The remaining 4 requests present an ALREADY USED refresh token!
 *   - The server flags this as a replay attack and REVOKES the user's session!
 *
 * Solution (The Mutex Interceptor Pattern):
 * 1. Attach `Authorization: Bearer <currentAccessToken>` to the outgoing request headers.
 * 2. Execute `let res = await fetch(url, options)`.
 * 3. If `res.status !== 401`, return `res` immediately.
 * 4. If `res.status === 401`:
 *    - You must ensure ONLY ONE `performTokenRefresh()` runs at a time.
 *    - If `refreshPromise` is null, create it:
 *        refreshPromise = performTokenRefresh().finally(() => {
 *          refreshPromise = null; // release the mutex lock
 *        });
 *    - Emit a trace message: `emitTrace('MUTEX', ...)` so the UI can show the queue.
 *    - Await `const newAccessToken = await refreshPromise;`
 *    - If `newAccessToken` is null (refresh failed or revoked), return the original 401 response `res`.
 *    - If `newAccessToken` is present:
 *        - Clone/update the original request headers with `Authorization: Bearer <newAccessToken>`.
 *        - Retry the original fetch: `return await fetch(url, retriedOptions)`.
 */
export async function authenticatedFetch(
  url: string,
  options: RequestInit = {}
): Promise<Response> {
  // TODO(human): Exercise 3 - Implement authenticatedFetch with Single-Flight Refresh Mutex
  throw new Error('TODO(human): Exercise 3 - Implement authenticatedFetch in frontend/src/auth/authClient.ts');
}
