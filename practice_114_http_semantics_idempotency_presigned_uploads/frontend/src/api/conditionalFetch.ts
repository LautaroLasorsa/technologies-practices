/**
 * HTTP Conditional Request Client Utilities.
 *
 * Implements:
 * 1. Safe bandwidth savings using `ETag` + `If-None-Match` (304 Not Modified).
 * 2. Optimistic concurrency control using `If-Match` (412 Precondition Failed)
 *    to prevent the Lost Update Problem.
 */

export interface Dataset {
  id: string;
  name: string;
  description: string;
  s3_key: string;
  size_bytes: number;
  etag: string;
  version: number;
  created_at: string;
  updated_at: string;
}

export interface ConditionalFetchResult<T> {
  data: T;
  etag: string | null;
  fromCache: boolean;
  status: number;
}

export class PreconditionFailedError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'PreconditionFailedError';
  }
}

/**
 * Perform a GET request using If-None-Match conditional caching.
 *
 * ========================================================================
 * TODO(human): Exercise 2 (Frontend part A) - If-None-Match 304 Handler
 * ========================================================================
 * Context:
 * When polling or re-fetching resources, sending the cached ETag allows the
 * server to respond with 304 Not Modified without sending any body bytes.
 *
 * Instructions:
 * 1. Build headers object. If `cachedETag` is provided, add `'If-None-Match': cachedETag`.
 * 2. Execute `fetch(url, { headers })`.
 * 3. If `response.status === 304`:
 *      Return `{ data: cachedData as T, etag: cachedETag, fromCache: true, status: 304 }`.
 * 4. If `response.ok`:
 *      Extract `const newEtag = response.headers.get('ETag')`.
 *      Parse `const data = await response.json()`.
 *      Return `{ data, etag: newEtag, fromCache: false, status: response.status }`.
 * 5. Otherwise, throw an Error with response status and text.
 * ========================================================================
 */
export async function fetchWithETag<T>(
  url: string,
  cachedETag?: string | null,
  cachedData?: T | null
): Promise<ConditionalFetchResult<T>> {
  // --- IMPLEMENTATION GOES HERE ---
  throw new Error(
    'Exercise 2: Implement fetchWithETag in frontend/src/api/conditionalFetch.ts'
  );
}

/**
 * Perform a PUT request protected by If-Match optimistic locking.
 *
 * ========================================================================
 * TODO(human): Exercise 2 (Frontend part B) - If-Match Optimistic Locking
 * ========================================================================
 * Context:
 * Ensures the resource has not been updated by another user or session
 * between the time it was fetched and the time this update is sent.
 *
 * Instructions:
 * 1. Execute `fetch(url, {
 *      method: 'PUT',
 *      headers: {
 *        'Content-Type': 'application/json',
 *        'If-Match': currentETag,
 *      },
 *      body: JSON.stringify(payload),
 *    })`.
 * 2. If `response.status === 412`:
 *      Throw new PreconditionFailedError('Resource has been modified by another client. Please refresh.');
 * 3. If `response.ok`:
 *      Extract `const newEtag = response.headers.get('ETag')`.
 *      Parse `const data = await response.json()`.
 *      Return `{ data, etag: newEtag }`.
 * 4. Otherwise, throw an Error with response status and text.
 * ========================================================================
 */
export async function updateWithIfMatch<T>(
  url: string,
  payload: unknown,
  currentETag: string
): Promise<{ data: T; etag: string | null }> {
  // --- IMPLEMENTATION GOES HERE ---
  throw new Error(
    'Exercise 2: Implement updateWithIfMatch in frontend/src/api/conditionalFetch.ts'
  );
}
