/**
 * Custom HTTP Client for Practice 115a.
 * Demonstrates:
 * - Ambient Cookie Transmission via credentials: 'include'
 * - Exercise 2: Double-Submit Anti-CSRF Token Injection (X-CSRF-Token header)
 */

export const API_BASE_URL = 'http://localhost:8115';
export const CSRF_COOKIE_NAME = 'csrf_token';
export const CSRF_HEADER_NAME = 'X-CSRF-Token';

const MUTATING_METHODS = new Set(['POST', 'PUT', 'DELETE', 'PATCH']);

/**
 * Extracts a cookie value by name from document.cookie.
 */
export function getCookieValue(cookieName: string): string | null {
  // ---------------------------------------------------------------------------
  // Exercise 2 - Task A (Client): Read Anti-CSRF Cookie
  // ---------------------------------------------------------------------------
  // TODO(human):
  // 1. Split document.cookie on ';'
  // 2. Trim whitespace and find the segment starting with `${cookieName}=`
  // 3. Return decodeURIComponent(value) or null if not found.
  //
  // NAIVE / UNHARDENED STARTER (Always returns null or dummy):
  const match = document.cookie
    .split('; ')
    .find((row) => row.startsWith(`${cookieName}=`));
  return match ? decodeURIComponent(match.split('=')[1]) : null;
}

/**
 * Performs an authenticated HTTP request with ambient credentials and Anti-CSRF tokens.
 */
export async function apiClient<T>(
  endpoint: string,
  options: RequestInit = {}
): Promise<T> {
  const url = `${API_BASE_URL}${endpoint}`;
  const method = (options.method || 'GET').toUpperCase();
  const headers = new Headers(options.headers || {});

  // Ensure JSON Content-Type for request bodies
  if (options.body && !headers.has('Content-Type')) {
    headers.set('Content-Type', 'application/json');
  }

  // ---------------------------------------------------------------------------
  // Exercise 2 - Task B (Client): Inject X-CSRF-Token Header on Mutating Requests
  // ---------------------------------------------------------------------------
  // When making state-mutating calls (POST/PUT/DELETE), the SPA must extract
  // the CSRF token from the cookie and attach it in the X-CSRF-Token header.
  // Cross-origin attackers cannot read document.cookie due to the Same-Origin Policy!
  //
  // TODO(human):
  // 1. Check if method is in MUTATING_METHODS.
  // 2. If so, read the token: const csrfToken = getCookieValue(CSRF_COOKIE_NAME);
  // 3. If csrfToken is present, set headers.set(CSRF_HEADER_NAME, csrfToken);
  //
  // NAIVE / UNHARDENED STARTER (Does not inject X-CSRF-Token header):
  // [TODO(human): Attach X-CSRF-Token header here]

  const response = await fetch(url, {
    ...options,
    method,
    headers,
    credentials: 'include', // Crucial: includes ambient session_id cookie cross-origin
  });

  if (!response.ok) {
    let errorDetail = `HTTP Error ${response.status} ${response.statusText}`;
    try {
      const errJson = await response.json();
      errorDetail = errJson.detail || errJson.error || errorDetail;
    } catch {
      // Non-JSON error
    }
    throw new Error(errorDetail);
  }

  return response.json() as Promise<T>;
}
