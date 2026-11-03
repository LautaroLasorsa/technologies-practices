/**
 * RFC 7636: Proof Key for Code Exchange by OAuth Public Clients (PKCE)
 *
 * PKCE binds the authorization code exchange to the specific client instance
 * that initiated the request, eliminating authorization code interception attacks
 * in public clients (SPAs, mobile apps) that cannot safely protect a static client secret.
 */

/**
 * Characters permitted by RFC 7636 Section 4.1:
 * unreserved = ALPHA / DIGIT / "-" / "." / "_" / "~"
 */
const PKCE_CHARSET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~';

/**
 * Helper: Converts an ArrayBuffer to a Base64URL string (RFC 7636 Section 3):
 * - Standard Base64
 * - Replace '+' with '-'
 * - Replace '/' with '_'
 * - Strip trailing '=' padding
 */
export function bufferToBase64Url(buffer: ArrayBuffer): string {
  const bytes = new Uint8Array(buffer);
  let binary = '';
  for (let i = 0; i < bytes.byteLength; i++) {
    binary += String.fromCharCode(bytes[i]);
  }
  return btoa(binary)
    .replace(/\+/g, '-')
    .replace(/\//g, '_')
    .replace(/=+$/, '');
}

/**
 * Exercise 1: TODO(human) - Part A
 *
 * Generate a cryptographically random code_verifier string.
 *
 * Specification (RFC 7636 Section 4.1):
 * - Minimum length: 43 characters
 * - Maximum length: 128 characters (default to 64 or 128)
 * - Allowed characters: A-Z, a-z, 0-9, "-", ".", "_", "~"
 *
 * Hints:
 * 1. Validate that length is between 43 and 128.
 * 2. Use `crypto.getRandomValues(new Uint8Array(length))` to get cryptographically random bytes.
 * 3. Map each byte to an index in `PKCE_CHARSET` using modulo: `PKCE_CHARSET[byte % PKCE_CHARSET.length]`.
 * 4. Join and return the string.
 */
export function generateCodeVerifier(length: number = 64): string {
  // TODO(human): Exercise 1 - Implement generateCodeVerifier
  throw new Error('TODO(human): Exercise 1 - Implement generateCodeVerifier in frontend/src/auth/pkce.ts');
}

/**
 * Exercise 1: TODO(human) - Part B
 *
 * Derive the S256 code_challenge from a code_verifier string.
 *
 * Specification (RFC 7636 Section 4.2):
 * code_challenge = BASE64URL-ENCODE(SHA256(ASCII(code_verifier)))
 *
 * Hints:
 * 1. Convert the verifier string into a Uint8Array:
 *    `const data = new TextEncoder().encode(verifier);`
 * 2. Hash it with SHA-256 using the browser Web Crypto API:
 *    `const digestBuffer = await crypto.subtle.digest('SHA-256', data);`
 * 3. Convert digestBuffer to Base64URL using the provided `bufferToBase64Url(digestBuffer)`.
 */
export async function deriveCodeChallenge(verifier: string): Promise<string> {
  // TODO(human): Exercise 1 - Implement deriveCodeChallenge
  throw new Error('TODO(human): Exercise 1 - Implement deriveCodeChallenge in frontend/src/auth/pkce.ts');
}
