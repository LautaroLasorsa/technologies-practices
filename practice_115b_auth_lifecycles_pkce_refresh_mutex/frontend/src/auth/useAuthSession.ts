import { useState, useEffect, useCallback } from 'react';
import { generateCodeVerifier, deriveCodeChallenge } from './pkce.js';
import {
  getAccessToken,
  setAccessToken,
  clearAccessToken,
  getTokenExpiresAt,
  emitTrace,
} from './authClient.js';

export type SessionStatus = 'idle' | 'authenticated' | 'refreshing' | 'revoked';

export interface AuthSessionState {
  isAuthenticated: boolean;
  userId: string | null;
  status: SessionStatus;
  tokenExpiresIn: number;
}

export type AuthSyncMessage =
  | { type: 'LOGOUT'; reason?: string }
  | { type: 'REVOKED'; message: string; familyId?: string }
  | { type: 'LOGIN'; userId: string };

const BROADCAST_CHANNEL_NAME = 'auth_session_channel';

export function useAuthSession() {
  const [session, setSession] = useState<AuthSessionState>({
    isAuthenticated: false,
    userId: null,
    status: 'idle',
    tokenExpiresIn: 0,
  });

  // Countdown timer for access token expiry
  useEffect(() => {
    if (!session.isAuthenticated) return;

    const interval = setInterval(() => {
      const remainingMs = getTokenExpiresAt() - Date.now();
      const remainingSec = Math.max(0, Math.ceil(remainingMs / 1000));

      setSession((prev) => ({
        ...prev,
        tokenExpiresIn: remainingSec,
      }));
    }, 500);

    return () => clearInterval(interval);
  }, [session.isAuthenticated]);

  // --------------------------------------------------------------------------
  // Exercise 4: TODO(human) - Cross-Tab Synchronization with BroadcastChannel
  // --------------------------------------------------------------------------
  /**
   * Exercise 4: TODO(human)
   *
   * Synchronize auth state across multiple browser tabs using the BroadcastChannel API.
   *
   * Why:
   * SPAs keep access tokens in JavaScript memory. If a user logs out in Tab A,
   * Tab B still has the old in-memory access token until it makes a request.
   * Furthermore, if Tab A triggers token reuse revocation, Tab B must immediately
   * lock its UI and notify the user that the session was revoked.
   *
   * Steps:
   * 1. Create a `new BroadcastChannel(BROADCAST_CHANNEL_NAME)`.
   * 2. Set up an `onmessage` listener:
   *    - If event.data.type === 'LOGOUT':
   *        clearAccessToken();
   *        setSession({ isAuthenticated: false, userId: null, status: 'idle', tokenExpiresIn: 0 });
   *        emitTrace('PKCE', 'Cross-tab logout received via BroadcastChannel');
   *    - If event.data.type === 'REVOKED':
   *        clearAccessToken();
   *        setSession({ isAuthenticated: false, userId: null, status: 'revoked', tokenExpiresIn: 0 });
   *        emitTrace('REVOKED', 'Cross-tab session revocation received via BroadcastChannel');
   * 3. Cleanup: In the useEffect return callback, close the channel: `channel.close()`.
   * 4. Update the `logout` function below so that upon logging out, it broadcasts:
   *    `channel.postMessage({ type: 'LOGOUT' })`.
   */
  useEffect(() => {
    // TODO(human): Exercise 4 - Set up BroadcastChannel listener and cleanup
    // Hint: const channel = new BroadcastChannel(BROADCAST_CHANNEL_NAME);
    // channel.onmessage = (event: MessageEvent<AuthSyncMessage>) => { ... };
    // return () => channel.close();
  }, []);

  // Listen for local reuse detection events dispatched from authClient
  useEffect(() => {
    const handleReuse = (e: any) => {
      setSession({
        isAuthenticated: false,
        userId: null,
        status: 'revoked',
        tokenExpiresIn: 0,
      });

      // Broadcast to other tabs
      try {
        const channel = new BroadcastChannel(BROADCAST_CHANNEL_NAME);
        channel.postMessage({
          type: 'REVOKED',
          message: e.detail?.message || 'Token family revoked',
          familyId: e.detail?.family_id,
        });
        channel.close();
      } catch (err) {
        // BroadcastChannel may fail in unsupported test runners
      }
    };

    window.addEventListener('auth:reuse_detected', handleReuse);
    return () => window.removeEventListener('auth:reuse_detected', handleReuse);
  }, []);

  // --------------------------------------------------------------------------
  // OAuth2 PKCE Login Flow
  // --------------------------------------------------------------------------
  const loginWithPkce = useCallback(async () => {
    try {
      emitTrace('PKCE', 'Starting OAuth2 PKCE login flow...');

      // 1. Generate code_verifier and derive code_challenge (Exercise 1)
      const verifier = generateCodeVerifier(64);
      const challenge = await deriveCodeChallenge(verifier);

      emitTrace('PKCE', 'Generated verifier & S256 challenge', {
        verifier: verifier.slice(0, 10) + '...',
        challenge,
      });

      // 2. Request authorization code from server
      const authRes = await fetch('/api/auth/authorize', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          clientId: 'spa_client_lab',
          codeChallenge: challenge,
          codeChallengeMethod: 'S256',
          userId: 'user_alice',
        }),
      });

      if (!authRes.ok) {
        throw new Error(`Authorize failed: ${authRes.status}`);
      }

      const { code } = await authRes.json();
      emitTrace('PKCE', 'Received Authorization Code from server', { code });

      // 3. Trade authorization code + code_verifier for tokens
      const tokenRes = await fetch('/api/auth/token', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify({
          code,
          codeVerifier: verifier,
        }),
      });

      if (!tokenRes.ok) {
        throw new Error(`Token exchange failed: ${tokenRes.status}`);
      }

      const tokenData = await tokenRes.json();
      setAccessToken(tokenData.access_token, tokenData.expires_in);

      setSession({
        isAuthenticated: true,
        userId: tokenData.user_id,
        status: 'authenticated',
        tokenExpiresIn: tokenData.expires_in,
      });

      emitTrace('PKCE', '🎉 PKCE Login Complete! Access token stored in memory closure.');
    } catch (err: any) {
      emitTrace('PKCE', `Login error: ${err.message}`);
      alert(`Login failed: ${err.message}`);
    }
  }, []);

  // --------------------------------------------------------------------------
  // Logout Flow
  // --------------------------------------------------------------------------
  const logout = useCallback(async () => {
    try {
      await fetch('/api/auth/logout', {
        method: 'POST',
        credentials: 'include',
      });
    } catch (e) {
      // Ignore network errors on logout
    }

    clearAccessToken();
    setSession({
      isAuthenticated: false,
      userId: null,
      status: 'idle',
      tokenExpiresIn: 0,
    });

    emitTrace('REQUEST', 'Logged out locally');

    // Broadcast logout to other tabs (Exercise 4)
    try {
      const channel = new BroadcastChannel(BROADCAST_CHANNEL_NAME);
      channel.postMessage({ type: 'LOGOUT' });
      channel.close();
    } catch (err) {
      // fallback
    }
  }, []);

  // --------------------------------------------------------------------------
  // Simulated Replay Attack (Educational)
  // --------------------------------------------------------------------------
  const triggerSimulatedReplay = useCallback(async () => {
    try {
      emitTrace('REVOKED', 'Fetching last rotated (used) refresh token from server...');
      const res = await fetch('/api/auth/last-rotated-token');
      const data = await res.json();

      if (!data.lastRotatedToken) {
        alert('No rotated token available yet. Please fire requests or wait for at least one refresh first!');
        return;
      }

      emitTrace('REVOKED', '🚨 Attacker replaying old refresh token:', data.lastRotatedToken);

      // Attempt to refresh using the already-used token
      const attackRes = await fetch('/api/auth/refresh', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify({ simulatedStolenToken: data.lastRotatedToken }),
      });

      const attackData = await attackRes.json();
      if (attackRes.status === 401 && attackData.error === 'token_reuse_detected') {
        emitTrace('REVOKED', '🛡️ SERVER DETECTED REUSE! Family revoked.', attackData);
        clearAccessToken();
        setSession({
          isAuthenticated: false,
          userId: null,
          status: 'revoked',
          tokenExpiresIn: 0,
        });

        // Broadcast to other tabs
        const channel = new BroadcastChannel(BROADCAST_CHANNEL_NAME);
        channel.postMessage({ type: 'REVOKED', message: attackData.message });
        channel.close();
      }
    } catch (err: any) {
      emitTrace('REVOKED', `Simulated attack failed: ${err.message}`);
    }
  }, []);

  return {
    session,
    loginWithPkce,
    logout,
    triggerSimulatedReplay,
  };
}
