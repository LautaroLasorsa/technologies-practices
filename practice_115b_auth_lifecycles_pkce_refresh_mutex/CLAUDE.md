# Practice 115b: Auth Lifecycles — OAuth2 PKCE, Refresh Token Rotation & Concurrent 401 Mutex

## Technologies
- **Frontend:** React 19, TypeScript, Vite, Web Crypto API (`crypto.subtle`), BroadcastChannel API
- **Backend:** Node.js, Fastify, `@fastify/cors`, `@fastify/cookie`, `node:crypto`
- **Tooling:** `tsx`, `npm`

## Stack
TypeScript (React Frontend + Node/Fastify Auth & Resource Server)

---

## Theoretical Context

Authentication and session lifecycle management in modern Single Page Applications (SPAs) presents unique security and concurrency challenges that do not exist in traditional server-rendered web applications.

### 1. Why OAuth2 Implicit Flow is Dead & How PKCE (RFC 7636) Works
Historically, OAuth 2.0 specified the **Implicit Grant** for browser-based clients. In the Implicit Flow, the authorization server redirects the user directly back to the SPA with the access token appended in the URL fragment (`#access_token=...`).

The Implicit Flow is now officially deprecated (OAuth 2.0 Security Best Current Practice) due to fatal security vulnerabilities:
- **Token Leakage via Browser History & Logs:** URL fragments persist in browser history, browser extensions, and server access logs.
- **Referrer Header Leaks:** If the redirected landing page makes outbound requests or includes external scripts, the `#access_token` fragment can leak via the HTTP `Referer` header.
- **Open Redirector Interception:** Malicious redirectors can hijack the token in-flight.

**Proof Key for Code Exchange (PKCE, RFC 7636)** was originally invented for mobile native apps and has become the mandatory standard for SPAs:
1. **`code_verifier` Generation:** The SPA dynamically generates a cryptographically random, high-entropy secret string (43–128 characters from unreserved ASCII).
2. **`code_challenge` Derivation:** The SPA hashes the verifier using SHA-256 and Base64URL-encodes the digest:
   $$\text{code\_challenge} = \text{Base64URL}(\text{SHA-256}(\text{code\_verifier}))$$
3. **Authorization Request:** The SPA sends the `code_challenge` (and `code_challenge_method=S256`) to the authorization endpoint `/api/auth/authorize`. The authorization server stores this challenge with the issued authorization code.
4. **Token Exchange:** The SPA exchanges the authorization code for tokens at `/api/auth/token`, sending the plain `code_verifier`.
5. **Cryptographic Proof:** The authorization server hashes the received `code_verifier` and verifies it matches the pre-registered `code_challenge`. Even if an attacker intercepts the authorization code in the browser URL, they cannot exchange it without the secret `code_verifier` that never left the originating SPA instance!

### 2. Where to Store Credentials in an SPA (The Hybrid Architecture)
A major security controversy in SPAs is where to hold session tokens:
- **Why NOT `localStorage` / `sessionStorage`:** Any third-party JavaScript running on the domain (compromised NPM dependencies, analytics, ad networks, or Cross-Site Scripting (XSS)) has unmitigated read access to Web Storage. Storing long-lived refresh tokens or access tokens here means one XSS vulnerability leads to permanent account takeover.
- **Why pure HttpOnly cookies struggle in distributed APIs:** HttpOnly cookies protect against JavaScript reading, but they are subject to Cross-Site Request Forgery (CSRF), cookie scoping constraints, and do not fit modern mobile or multi-service microservice architectures easily.

**The Hybrid Gold Standard:**
- **Access Token:** Short-lived (e.g. 15 seconds in this lab demo, 5–15 minutes in production). Stored **strictly in an in-memory JavaScript closure**. It is never written to `localStorage`, `sessionStorage`, or IndexedDB. If an attacker manages XSS, they can only access the current token for seconds, and cannot exfiltrate persistent credentials.
- **Refresh Token:** Long-lived (days/weeks). Stored in a strict **`HttpOnly; Secure; SameSite=Strict; Path=/api/auth/refresh`** cookie. JavaScript cannot read or modify this cookie. It is automatically transmitted by the browser *only* when hitting the token refresh endpoint.

### 3. Refresh Token Rotation (RTR) & Family Reuse Detection
To prevent stolen refresh tokens from providing perpetual access, RFC 6749 BCP mandates **Refresh Token Rotation (RTR)**:
- Every time `/api/auth/refresh` is called, the presented refresh token is immediately invalidated (`isUsed = true`), and a new refresh token is issued within the same **Token Family** (`familyId`).
- **Reuse Detection:** If an attacker intercepts a refresh token and attempts to replay it *after* the legitimate client has already rotated it, the backend identifies that `isUsed === true`.
- **Compromise Mitigation:** Because the server cannot know whether the attacker or the victim made the first call, it must treat the family as breached: **it immediately revokes the entire `familyId`**, invalidating all outstanding access and refresh tokens. Both the victim and attacker are locked out, forcing full re-authentication and neutralizing ongoing unauthorized access.

### 4. The Concurrent 401 Thundering Herd Problem (Client Refresh Mutex)
While Refresh Token Rotation provides elite security, it introduces a critical concurrency race condition on the client:
- Consider a dashboard with 5 widgets (Analytics, Profile, Notifications, Activity, Billing).
- When the user returns to the tab after the 15-second access token expires, all 5 widgets fire parallel HTTP requests simultaneously.
- All 5 requests receive `401 Unauthorized`.
- **The Catastrophe:** If each widget independently reacts to `401` by calling `/api/auth/refresh`:
  1. Widget 1 sends Refresh Token $R_1$. The server accepts it, invalidates $R_1$, and issues $R_2$.
  2. Widgets 2, 3, 4, and 5 send concurrent refresh requests with the now-invalidated $R_1$.
  3. The server's reuse detection triggers! It concludes that a replay attack is underway and **revokes the user's entire session**!
  4. The user is logged out simply because their dashboard loaded 5 widgets in parallel!

**The Solution — Single-Flight Refresh Mutex:**
An interceptor in the HTTP client (e.g., `authenticatedFetch`) implements a promise mutex / request queue:
1. When a `401` is received, the client checks if a refresh operation is already in-flight (`refreshPromise !== null`).
2. If in-flight, subsequent failing requests queue behind the existing `refreshPromise`.
3. If no refresh is in-flight, the first request acquires the lock and initiates `performTokenRefresh()`.
4. Once the single refresh completes and stores the new access token in memory, all queued requests resolve and transparently retry with the new token.

```
Without Mutex (Race Condition & Revocation):
Widget 1 (401) ───► POST /api/auth/refresh (R1) ───► 200 OK (Issues R2)
Widget 2 (401) ──────► POST /api/auth/refresh (R1) ───► 🚨 401 REUSE DETECTED! Family Revoked!
Widget 3 (401) ─────────► POST /api/auth/refresh (R1) ───► 🚨 401 REUSE DETECTED! Family Revoked!

With Single-Flight Mutex:
Widget 1 (401) ───► Acquires Mutex ───► POST /api/auth/refresh (R1) ───► 200 OK (Issues R2)
Widget 2 (401) ───► Queued on Mutex ─── (Awaits Promise) ──────────────► Retried with R2 (200 OK)
Widget 3 (401) ───► Queued on Mutex ─── (Awaits Promise) ──────────────► Retried with R2 (200 OK)
```

### 5. Cross-Tab Session Synchronization via BroadcastChannel
Because access tokens are stored in-memory, tab states can drift:
- If a user logs out in Tab A, Tab B's in-memory token remains active until its next request.
- If a token replay attack is detected in Tab A, Tab B needs immediate notification to lock the screen.
- Using the modern `BroadcastChannel` API, tabs on the same origin synchronize lifecycle events (`LOGOUT`, `REVOKED`, `LOGIN`) in real time without polling.

---

## Description
In this practice, you will implement a complete, enterprise-grade OAuth2 PKCE and rotating refresh token architecture between a React 19 SPA and a Node/Fastify API server:
1. Generate cryptographic PKCE verifiers and SHA-256 challenges in the browser.
2. Implement backend Refresh Token Rotation with token family tracking and replay detection.
3. Build the client-side Single-Flight Refresh Mutex interceptor to eliminate 401 thundering herds.
4. Synchronize multi-tab auth lifecycles using `BroadcastChannel`.

---

## Instructions

### Prerequisites
- Node.js 20+ (v22.x recommended)
- Fastify 5, React 19, Vite

### Step-by-Step Implementation

#### Exercise 1: RFC 7636 PKCE Generator (`frontend/src/auth/pkce.ts`)
- **What it teaches:** How SPAs protect public client authorization flows using browser-native Web Crypto without client secrets.
- **Task:** Implement `generateCodeVerifier(length)` using `crypto.getRandomValues()` with unreserved URI characters, and `deriveCodeChallenge(verifier)` using `crypto.subtle.digest('SHA-256', ...)` with Base64URL encoding.
- **Verification:** Run `npm test` inside `frontend/` to test against the official RFC 7636 Appendix B test vector.

#### Exercise 2: Refresh Token Rotation & Reuse Detection (`backend/src/tokenStore.ts`)
- **What it teaches:** How backend auth servers prevent token theft replay attacks using token families and immediate compromise revocation.
- **Task:** Implement `rotateRefreshToken(oldRefreshToken)` in `backend/src/tokenStore.ts`:
  1. Validate token existence, expiration, and family revocation state.
  2. Detect token reuse if `record.isUsed === true`, immediately revoke `family.isRevoked = true` and purge access tokens.
  3. On valid rotation, mark the old token used, issue new tokens, and maintain the family link.
- **Verification:** Run `npm test` inside `backend/` to run the token family reuse test suite.

#### Exercise 3: Single-Flight Refresh Mutex Interceptor (`frontend/src/auth/authClient.ts`)
- **What it teaches:** How to prevent 401 thundering herd race conditions from triggering false-positive rotation lockdowns.
- **Task:** Implement `authenticatedFetch(url, options)` in `frontend/src/auth/authClient.ts`:
  1. Attach in-memory `Authorization: Bearer <token>` header.
  2. If the response returns `401`, intercept it.
  3. Ensure only a single `performTokenRefresh()` runs by maintaining a singleton `refreshPromise`.
  4. Await the refresh promise, retry the original failed request with the new access token, and return the retried response.
- **Verification:** Start both servers, click **"Fire 5 Concurrent Widget Calls"** in the UI, and observe the live trace log proving exactly 1 refresh call handled all 5 requests.

#### Exercise 4: Cross-Tab Session Synchronization (`frontend/src/auth/useAuthSession.ts`)
- **What it teaches:** Multi-window session consistency in in-memory SPAs using the `BroadcastChannel` API.
- **Task:** Implement the `BroadcastChannel` subscriber in `useAuthSession.ts` to listen for `LOGOUT` and `REVOKED` events, clearing memory and locking the UI when another tab revokes the session.
- **Verification:** Open two tabs with the **"Open Duplicate Tab"** button. Log out or trigger a simulated attack in Tab A, and watch Tab B instantly update.

---

## Motivation
Token lifecycle races and security flaws are among the most pervasive bugs in modern web applications:
- Storing tokens in `localStorage` leaves high-value enterprise apps vulnerable to XSS credential harvesting.
- Implementing token rotation without a client-side mutex causes random session dropouts and user complaints whenever a dashboard executes parallel requests.
- Mastering RFC 7636 PKCE, in-memory closures, and mutex interceptors is a fundamental senior engineering skill required in financial, healthcare, and enterprise SaaS platforms.

---

## Commands

| Phase | Command | Description |
|---|---|---|
| **Setup** | `cd backend && npm install` | Install backend Fastify & TypeScript dependencies |
| **Setup** | `cd frontend && npm install` | Install frontend React 19 & Vite dependencies |
| **Test** | `cd frontend && npm test` | Run RFC 7636 Appendix B unit tests for Exercise 1 |
| **Test** | `cd backend && npm test` | Run Refresh Token Rotation & Family Reuse tests for Exercise 2 |
| **Run** | `cd backend && npm run dev` | Start Fastify Auth & Resource server on `http://localhost:4115` |
| **Run** | `cd frontend && npm run dev` | Start Vite React dev server on `http://localhost:5173` |
| **Build** | `cd frontend && npm run build` | Typecheck and build frontend production bundle |
| **Clean** | `python clean.py` | Remove all `node_modules`, `dist`, build caches, and logs |

---

## Notes
*(Add your notes and observations here as you complete the exercises)*
