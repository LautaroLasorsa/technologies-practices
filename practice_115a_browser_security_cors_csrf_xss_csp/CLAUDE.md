# Practice 115a: Browser Security Boundaries — SOP, CORS Preflight, CSRF, XSS & Content Security Policy (CSP)

## Technologies
- **React 19 & Vite** (Victim Frontend SPA on `http://localhost:5173`)
- **FastAPI & Uvicorn** (Treasury Backend API on `http://localhost:8115`)
- **Node.js Native HTTP** (Simulated Attacker Origin Server on `http://localhost:6660`)
- **DOMPurify & TypeScript** (Client-side XSS defense & type safety)

## Stack
- **Frontend & Attacker:** TypeScript, React 19, Vite, Node.js (ESM `server.mjs`)
- **Backend:** Python 3.12+, FastAPI, Uvicorn, Pydantic, Starlette

---

## Theoretical Context

Modern web application security is anchored in the security architecture of the web browser. The browser operates as an untrusted multi-tenant host executing code from multiple mutually untrusted origins (`https://bank.example.com`, `https://social.example.com`, `https://evil-hacker.com`). Understanding browser security boundaries requires analyzing the exact mechanics of origin boundaries, credential handling, and code evaluation.

```
+-----------------------------------------------------------------------------------------+
|                                    USER BROWSER                                         |
|                                                                                         |
|  +----------------------------+                          +---------------------------+  |
|  |     Victim SPA Origin      |                          |      Attacker Origin      |  |
|  |   http://localhost:5173    |                          |   http://localhost:6660   |  |
|  +--------------+-------------+                          +-------------+-------------+  |
|                 |                                                      |                |
|                 | 1. Safe API Request                                  | 2. Cross-Site  |
|                 |    (Same-Site Context)                               |    Form/Fetch  |
|                 |    + Custom Header: X-CSRF-Token                     |    (Ambient    |
|                 |    + Credentials: cookie                             |     Cookies!)  |
|                 v                                                      v                |
+-----------------|------------------------------------------------------|----------------+
                  |                                                      |
                  +---------------------------+--------------------------+
                                              |
                                              v
                              +-------------------------------+
                              |    FastAPI Treasury Backend   |
                              |     http://localhost:8115     |
                              |                               |
                              |  1. CORS Preflight & Origin   |
                              |  2. Sec-Fetch-Site Validation |
                              |  3. Double-Submit CSRF Check  |
                              |  4. Strict CSP Headers        |
                              +-------------------------------+
```

### 1. Same-Origin Policy (SOP) & Cross-Origin Resource Sharing (CORS)

- **The Origin Tuple:** An origin is strictly defined by `(Scheme, Host, Port)`. `http://localhost:5173` and `http://localhost:8115` are completely distinct origins.
- **SOP Asymmetry (Reads vs Writes vs Embeds):**
  - SOP blocks cross-origin **reads**: Origin A cannot read HTTP response bodies from Origin B using `fetch()` or `XMLHttpRequest`.
  - Historically, SOP **permitted cross-origin writes and embeds**: `<form action="http://api:8115/transfer" method="POST">` can issue a state-mutating request without reading the response. `<img src="...">`, `<script src="...">`, and `<iframe>` can be embedded across origins.
- **CORS is a Relaxation, Not Authentication:**
  - CORS (`Access-Control-Allow-*`) is a browser mechanism allowing a server to relax the browser's SOP restriction for trusted origins.
  - CORS does **not** protect a server from receiving requests; an attacker using `curl`, Python `requests`, or a legacy HTML form can still hit backend endpoints. CORS only instructs the *browser* whether to allow client-side JavaScript to read the response or whether to allow complex requests.
- **Simple Requests vs Preflighted Requests:**
  - **Simple Request:** Uses methods `GET`, `HEAD`, or `POST`, with restricted headers (`Accept`, `Accept-Language`, `Content-Language`, `Content-Type` limited to `application/x-www-form-urlencoded`, `multipart/form-data`, or `text/plain`). These requests are sent immediately with credentials if configured.
  - **Preflight (`OPTIONS`):** Triggered when using custom headers (`X-CSRF-Token`, `Authorization`), JSON payloads (`Content-Type: application/json`), or methods like `PUT`, `DELETE`, `PATCH`. The browser sends an `OPTIONS` request with `Access-Control-Request-Method` and `Access-Control-Request-Headers`.
  - **Caching Preflights:** `Access-Control-Max-Age: 86400` instructs the browser to cache preflight decisions, eliminating preflight latency for subsequent calls.
  - **The Credential Dilemma:** If `Access-Control-Allow-Credentials: true` is set, browsers strictly reject wildcards: `Access-Control-Allow-Origin: *` is invalid and rejected by the browser engine. The server must echo back the specific validated origin.

---

### 2. Cross-Site Request Forgery (CSRF) Deep Dive

- **The Root Cause — Ambient Credentials:**
  - When a browser makes an HTTP request to `http://localhost:8115`, it automatically attaches any cookies scoped to `localhost:8115` (`Cookie: session_id=...`).
  - The browser does this regardless of which origin initiated the request: whether initiated from the legitimate app `localhost:5173` or malicious site `localhost:6660`.
- **`SameSite` Cookie Attribute & Where It Fails:**
  - `SameSite=Strict`: Cookies are never sent on cross-site requests.
  - `SameSite=Lax`: Modern browser default. Cookies are withheld on cross-site subrequests (`fetch`, `<img>`, `<iframe>`, `POST` forms), but **still sent on top-level navigation** (`<a href="...">` or `GET` redirects).
  - **Why `SameSite=Lax` is not a silver bullet:**
    1. *State-mutating GETs:* If an endpoint like `/api/transfer?amount=1000` responds to `GET`, an attacker's `<a href>` or `window.location` triggers it with cookies attached.
    2. *Same-Site vs Cross-Origin Confusion:* `SameSite` operates on the registrable domain (eTLD+1), e.g., `attacker.example.com` and `bank.example.com` are *same-site*, but *cross-origin*. A compromised subdomain bypasses `SameSite=Lax`.
    3. *Lax-allowing-unsafe (2-minute window):* Certain browsers permit top-level `POST` requests without `SameSite` restrictions within 120 seconds of cookie creation.
- **Defense-in-Depth for SPAs:**
  1. **Double-Submit / Cryptographic Anti-CSRF Token (`X-CSRF-Token`):**
     - The server sets a cryptographically random token in a readable cookie (`csrf_token`) or provides it via a dedicated `/api/csrf-token` endpoint.
     - The SPA reads this value and injects it into a custom HTTP request header (`X-CSRF-Token`).
     - Because SOP prevents malicious origins from reading cookies or DOM belonging to other origins, the attacker cannot read the token value to inject the header.
     - Cross-origin custom headers require a CORS preflight (`OPTIONS`), ensuring malicious cross-origin scripts are halted before reaching the backend.
  2. **Metadata Headers (`Sec-Fetch-Site`):**
     - Modern browsers automatically send `Sec-Fetch-Site: same-origin`, `same-site`, or `cross-site`. The server can reject mutating requests where `Sec-Fetch-Site == "cross-site"`.
  3. **The `__Host-` Cookie Prefix:**
     - Cookies prefixed with `__Host-` must be set with `Secure`, `Path=/`, and without a `Domain` attribute. This guarantees the cookie cannot be overwritten by sibling subdomains.

---

### 3. Cross-Site Scripting (XSS) in React & Content Security Policy (CSP)

- **The React Escaping Model & Its Escape Hatches:**
  - React auto-escapes string content in JSX interpolation (`<div>{userComment}</div>` escapes `<` to `&lt;`), preventing basic HTML injection.
  - **Vulnerability Vectors in React:**
    1. `dangerouslySetInnerHTML={{ __html: userHtml }}`: Directly injects raw HTML into the DOM tree.
    2. URL Schemes: `<a href={userUrl}>Profile</a>`. If `userUrl` is `javascript:alert(document.cookie)` or `data:text/html,...`, clicking or hovering executes arbitrary JavaScript.
    3. React `useId` or SSR hydration payloads: Inserting raw unescaped JSON strings into `<script>` tags on the server creates hydration injection.
- **Sanitization with DOMPurify:**
  - Strips dangerous tags (`<script>`, `<iframe>`, `<object>`), cleans attributes, and disables execution vectors while preserving safe styling (`<b>`, `<em>`, `<code>`).
  - Must be paired with protocol allowlisting (`http:`, `https:`, `mailto:`) for `href` attributes.
- **Content Security Policy (CSP) as a Last Line of Defense:**
  - An HTTP response header sent by the server to instruct the browser which resource origins and execution methods are permitted.
  - `default-src 'self'`: Disallows loading any external resources unless explicitly allowed.
  - `script-src 'self'`: Disallows execution of inline scripts (`<script>alert(1)</script>`) and unsafe evals (`eval()`).
  - `connect-src 'self' http://localhost:8115`: Restricts `fetch` and WebSocket destinations, preventing attackers from exfiltrating stolen tokens to external servers.
  - `frame-ancestors 'none'`: Prevents clickjacking by forbidding embedding inside any `<iframe>`.
  - `report-uri /api/csp-report`: Instructs the browser to send JSON diagnostic payloads whenever policy violations occur.

---

## Description

In this practice, you run a **3-origin local security laboratory**:
1. **Victim Treasury React SPA** on `http://localhost:5173`: A banking transfer and transaction audit portal.
2. **FastAPI Treasury Backend** on `http://localhost:8115`: An authenticated banking service storing balances and session cookies.
3. **Attacker Portal** on `http://localhost:6660`: A malicious website hosting automated cross-site POST forms, cross-origin fetch attacks, clickjacking iframes, and XSS payload templates.

You will observe both CSRF and XSS attacks succeeding against an unhardened configuration, and then systematically implement full defense-in-depth across the React frontend and FastAPI backend.

---

## Instructions

### Step 0: Environment Setup
1. Launch the FastAPI backend:
   ```bash
   cd backend
   uv venv
   source .venv/bin/activate  # or .venv\Scripts\activate on Windows
   uv sync
   uvicorn main:app --host 127.0.0.1 --port 8115 --reload
   ```
2. Launch the Attacker origin server:
   ```bash
   cd attacker-server
   node server.mjs
   ```
3. Launch the Victim React SPA:
   ```bash
   cd frontend
   npm install
   npm run dev
   ```
4. Open `http://localhost:5173` to verify the victim application loads and logs in with session cookies.
5. Open `http://localhost:6660` to view the attacker console and observe the baseline attack vectors.

---

### Exercise 1: Explicit CORS Rules & Fetch Metadata Inspection (`backend/security_middleware.py`)
- **Context:** FastAPI's default or naive CORS middlewares often either allow all origins with `*` (which breaks credentials) or fail to correctly negotiate preflight `OPTIONS` requests for custom security headers. Additionally, servers must leverage modern browser metadata (`Sec-Fetch-Site`).
- **File:** `backend/security_middleware.py`
- **Tasks (`TODO(human)`):**
  1. Configure explicit CORS headers on responses matching trusted origins (`http://localhost:5173`).
  2. Handle `OPTIONS` preflight requests: return allowed methods (`GET, POST, PUT, DELETE, OPTIONS`), allowed headers (`Content-Type, X-CSRF-Token`), and `Access-Control-Max-Age: 86400`.
  3. Validate `Sec-Fetch-Site` header on state-mutating requests (`POST`, `PUT`, `DELETE`): reject any request where `Sec-Fetch-Site == "cross-site"` unless explicitly exempted.

---

### Exercise 2: Double-Submit HMAC Anti-CSRF Defense (`backend/csrf.py` & `frontend/src/api/httpClient.ts`)
- **Context:** To thwart cross-site requests with ambient session cookies, the server issues a cryptographically verified CSRF token. The frontend client extracts this token and sends it in the `X-CSRF-Token` header. The attacker origin cannot read this cookie due to SOP and cannot forge the header.
- **Files:** `backend/csrf.py` and `frontend/src/api/httpClient.ts`
- **Tasks (`TODO(human)`):**
  1. In `backend/csrf.py`:
     - Implement token generation combining a random nonce and an HMAC signature.
     - Set the `csrf_token` cookie with `SameSite=Lax`, `Path=/`, and `HttpOnly=False` (readable by legitimate SPA JavaScript).
     - Implement constant-time token verification (`secrets.compare_digest`) comparing the `X-CSRF-Token` header against the expected value.
  2. In `frontend/src/api/httpClient.ts`:
     - Implement a cookie reader helper that extracts `csrf_token`.
     - Add an interceptor to the custom HTTP client that injects `X-CSRF-Token` on all mutating methods (`POST`, `PUT`, `PATCH`, `DELETE`).
  3. Test with the Attacker site: observe that the automated CSRF exploit on `http://localhost:6660` now fails with `403 Forbidden: Invalid or missing CSRF token`.

---

### Exercise 3: React XSS Remediation & DOMPurify Sanitization (`frontend/src/components/AuditCommentFeed.tsx`)
- **Context:** An audit comments feed allows users to submit notes with author links. Naive rendering creates two critical XSS vulnerabilities: an unsafe `href` attribute executing `javascript:` pseudo-protocols and a rich text parser rendering unsanitized HTML.
- **File:** `frontend/src/components/AuditCommentFeed.tsx`
- **Tasks (`TODO(human)`):**
  1. Fix the link sink: create a URL validator ensuring only safe protocols (`http:`, `https:`) are allowed in `<a href={...}>`. Block `javascript:`, `data:`, and `vbscript:`.
  2. Fix the rich-text sink: sanitize the comment body with `DOMPurify.sanitize()` using strict options (`ALLOWED_TAGS: ['b', 'i', 'em', 'strong', 'code', 'pre', 'p', 'br']`, `ALLOWED_ATTR: ['class']`).
  3. In the Attacker dashboard, copy the XSS payloads and submit them into the feed. Verify that no script execution occurs and safe formatting is preserved.

---

### Exercise 4: Content Security Policy & Clickjacking Protection (`backend/csp.py`)
- **Context:** Even if an application contains unknown XSS vulnerabilities or third-party package compromises, a strict CSP header prevents inline script execution, stops data exfiltration, and eliminates clickjacking.
- **File:** `backend/csp.py`
- **Tasks (`TODO(human)`):**
  1. Construct a comprehensive `Content-Security-Policy` header:
     - `default-src 'self'`
     - `script-src 'self'`
     - `style-src 'self' 'unsafe-inline'` (for Vite dev styles)
     - `connect-src 'self' http://localhost:8115`
     - `img-src 'self' data:`
     - `frame-ancestors 'none'` (blocks iframe clickjacking)
     - `report-uri /api/csp-report`
  2. Add the CSP middleware to the FastAPI application and implement the `/api/csp-report` endpoint to log incoming browser policy violation telemetry.
  3. Test in the browser console: attempt to dynamically inject an external script or iframe the backend, and inspect the logged violation.

---

## Motivation

Browser security boundaries are the first line of defense for every web application. Modern web development relies heavily on APIs, SPAs, and microfrontends, where misunderstandings about CORS ("let's just add `*`"), CSRF ("we use JSON so we are safe"), and XSS ("React auto-escapes everything") lead directly to catastrophic account takeovers and data breaches. Mastering SOP, preflight caching, Double-Submit CSRF tokens, DOMPurify sanitization, and CSP transforms a developer from writing fragile code to architecting battle-tested, defense-in-depth web systems.

---

## Commands

| Phase | Command | Description |
|---|---|---|
| **Backend Setup** | `cd backend && uv venv && uv sync` | Create Python venv and install FastAPI dependencies |
| **Backend Run** | `cd backend && uv run uvicorn main:app --host 127.0.0.1 --port 8115 --reload` | Start FastAPI Treasury API on port 8115 |
| **Attacker Run** | `cd attacker-server && node server.mjs` | Start Attacker Origin Server on port 6660 |
| **Frontend Setup** | `cd frontend && npm install` | Install React 19, Vite, and DOMPurify dependencies |
| **Frontend Run** | `cd frontend && npm run dev` | Start Victim React application on port 5173 |
| **Test CSRF cURL** | `curl -i -X POST http://localhost:8115/api/transfer -H "Origin: http://localhost:6660" -H "Content-Type: application/json" -d "{\"recipient\":\"attacker\",\"amount\":5000}"` | Simulate cross-origin attack request |
| **Test CSP Report** | `curl -i -X POST http://localhost:8115/api/csp-report -H "Content-Type: application/json" -d "{\"csp-report\":{\"blocked-uri\":\"eval\",\"violated-directive\":\"script-src\"}}"` | Verify CSP telemetry endpoint |
| **Clean Workspace** | `python clean.py` | Clean all temporary files, node_modules, and virtual environments |

---

## Notes

*(To be populated during practice execution)*
