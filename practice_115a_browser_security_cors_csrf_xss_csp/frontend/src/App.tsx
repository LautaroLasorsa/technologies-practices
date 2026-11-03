import React, { useEffect, useState } from 'react';
import { apiClient, getCookieValue, CSRF_COOKIE_NAME } from './api/httpClient';
import { AuditCommentFeed, AuditComment } from './components/AuditCommentFeed';

interface UserProfile {
  user: string;
  balance: number;
  transactions: Array<{ id: number; recipient: string; amount: number; status: string }>;
  csrf_token: string;
}

export const App: React.FC = () => {
  const [profile, setProfile] = useState<UserProfile | null>(null);
  const [comments, setComments] = useState<AuditComment[]>([]);
  const [recipient, setRecipient] = useState('');
  const [amount, setAmount] = useState('');
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [csrfCookieVal, setCsrfCookieVal] = useState<string | null>(null);

  const refreshData = async () => {
    try {
      const userRes = await apiClient<UserProfile>('/api/me');
      setProfile(userRes);
      const commentsRes = await apiClient<AuditComment[]>('/api/comments');
      setComments(commentsRes);
      setCsrfCookieVal(getCookieValue(CSRF_COOKIE_NAME));
    } catch (err: any) {
      setError(`Failed to connect to Treasury Backend: ${err.message}`);
    }
  };

  useEffect(() => {
    refreshData();
    const interval = setInterval(() => {
      setCsrfCookieVal(getCookieValue(CSRF_COOKIE_NAME));
    }, 2000);
    return () => clearInterval(interval);
  }, []);

  const handleTransfer = async (e: React.FormEvent) => {
    e.preventDefault();
    setMessage(null);
    setError(null);
    try {
      const res = await apiClient<{ message: string; new_balance: number }>('/api/transfer', {
        method: 'POST',
        body: JSON.stringify({ recipient, amount: parseFloat(amount) }),
      });
      setMessage(res.message);
      setRecipient('');
      setAmount('');
      refreshData();
    } catch (err: any) {
      setError(err.message);
    }
  };

  const handleAddComment = async (newComment: { author: string; author_url: string; body: string }) => {
    setMessage(null);
    setError(null);
    try {
      await apiClient('/api/comments', {
        method: 'POST',
        body: JSON.stringify(newComment),
      });
      refreshData();
    } catch (err: any) {
      setError(err.message);
    }
  };

  const handleReset = async () => {
    setMessage(null);
    setError(null);
    try {
      await apiClient('/api/reset', { method: 'POST' });
      setMessage('State restored to default $10,000.00 baseline');
      refreshData();
    } catch (err: any) {
      setError(err.message);
    }
  };

  return (
    <div style={{ maxWidth: 960, margin: '0 auto', padding: '32px 20px' }}>
      {/* Header */}
      <header style={{ borderBottom: '1px solid #334155', paddingBottom: 16, marginBottom: 24, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <h1 style={{ margin: 0, fontSize: 24, color: '#f8fafc' }}>
            Treasury Banking Portal
          </h1>
          <p style={{ margin: '4px 0 0', color: '#94a3b8', fontSize: 14 }}>
            Origin: <code style={{ color: '#38bdf8' }}>http://localhost:5173</code> | Target API: <code style={{ color: '#38bdf8' }}>http://localhost:8115</code>
          </p>
        </div>
        <button
          onClick={handleReset}
          style={{ padding: '8px 14px', background: '#334155', color: '#f8fafc', border: 'none', borderRadius: 4, cursor: 'pointer', fontSize: 13 }}
        >
          Reset Lab State
        </button>
      </header>

      {/* Security Status Panel */}
      <div style={{ background: '#1e293b', padding: 14, borderRadius: 8, marginBottom: 20, border: '1px solid #334155', display: 'flex', gap: 20, flexWrap: 'wrap' }}>
        <div>
          <span style={{ color: '#94a3b8', fontSize: 12 }}>ACTIVE USER:</span>
          <div style={{ fontWeight: 600, color: '#22c55e' }}>{profile ? profile.user : 'Connecting...'}</div>
        </div>
        <div>
          <span style={{ color: '#94a3b8', fontSize: 12 }}>AVAILABLE BALANCE:</span>
          <div style={{ fontWeight: 700, fontSize: 18, color: '#f8fafc' }}>
            ${profile ? profile.balance.toLocaleString('en-US', { minimumFractionDigits: 2 }) : '0.00'}
          </div>
        </div>
        <div>
          <span style={{ color: '#94a3b8', fontSize: 12 }}>CSRF COOKIE STATUS:</span>
          <div style={{ fontSize: 13, color: csrfCookieVal ? '#22c55e' : '#ef4444' }}>
            {csrfCookieVal ? `Present (${csrfCookieVal.substring(0, 10)}...)` : 'Missing / Not Set'}
          </div>
        </div>
      </div>

      {message && (
        <div style={{ padding: 12, background: '#064e3b', color: '#a7f3d0', borderRadius: 6, marginBottom: 16 }}>
          {message}
        </div>
      )}
      {error && (
        <div style={{ padding: 12, background: '#7f1d1d', color: '#fecaca', borderRadius: 6, marginBottom: 16 }}>
          {error}
        </div>
      )}

      {/* Main Grid: Transfer Form + Transactions */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 24, marginBottom: 24 }}>
        {/* Transfer Form */}
        <div style={{ background: '#1e293b', padding: 20, borderRadius: 8, border: '1px solid #334155' }}>
          <h3 style={{ marginTop: 0, color: '#f8fafc' }}>Execute Fund Transfer</h3>
          <p style={{ color: '#94a3b8', fontSize: 13 }}>
            State-mutating <code>POST /api/transfer</code>. Requires Anti-CSRF protection to prevent unauthorized cross-origin execution!
          </p>
          <form onSubmit={handleTransfer} style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
            <div>
              <label style={{ display: 'block', marginBottom: 4, fontSize: 13, color: '#cbd5e1' }}>Recipient Name / IBAN</label>
              <input
                type="text"
                placeholder="e.g. Vendor Corp"
                value={recipient}
                onChange={(e) => setRecipient(e.target.value)}
                style={{ width: '100%', padding: 8, background: '#0f172a', border: '1px solid #475569', color: '#fff', borderRadius: 4, boxSizing: 'border-box' }}
                required
              />
            </div>
            <div>
              <label style={{ display: 'block', marginBottom: 4, fontSize: 13, color: '#cbd5e1' }}>Amount (USD)</label>
              <input
                type="number"
                step="0.01"
                placeholder="1000.00"
                value={amount}
                onChange={(e) => setAmount(e.target.value)}
                style={{ width: '100%', padding: 8, background: '#0f172a', border: '1px solid #475569', color: '#fff', borderRadius: 4, boxSizing: 'border-box' }}
                required
              />
            </div>
            <button
              type="submit"
              style={{
                padding: '10px 16px',
                background: '#2563eb',
                color: '#fff',
                border: 'none',
                borderRadius: 4,
                cursor: 'pointer',
                fontWeight: 600,
                marginTop: 6,
              }}
            >
              Transfer Funds
            </button>
          </form>
        </div>

        {/* Transaction History */}
        <div style={{ background: '#1e293b', padding: 20, borderRadius: 8, border: '1px solid #334155' }}>
          <h3 style={{ marginTop: 0, color: '#f8fafc' }}>Recent Transactions</h3>
          <p style={{ color: '#94a3b8', fontSize: 13 }}>Audit log of processed wire transfers:</p>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
            {profile?.transactions.map((tx) => (
              <div
                key={tx.id}
                style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  padding: 10,
                  background: '#0f172a',
                  border: '1px solid #334155',
                  borderRadius: 6,
                }}
              >
                <div>
                  <div style={{ fontWeight: 600, color: '#f1f5f9' }}>{tx.recipient}</div>
                  <div style={{ fontSize: 12, color: '#94a3b8' }}>Status: {tx.status}</div>
                </div>
                <div style={{ fontWeight: 700, color: '#ef4444' }}>
                  -${tx.amount.toFixed(2)}
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Audit Comments Feed (Exercise 3: XSS testing) */}
      <AuditCommentFeed comments={comments} onAddComment={handleAddComment} />
    </div>
  );
};
