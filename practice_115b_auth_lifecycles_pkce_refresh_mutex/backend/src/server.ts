import Fastify, { FastifyRequest, FastifyReply } from 'fastify';
import cors from '@fastify/cors';
import cookie from '@fastify/cookie';
import { TokenStore } from './tokenStore.js';

const fastify = Fastify({
  logger: {
    transport: {
      target: 'pino-pretty',
      options: {
        colorize: true,
        translateTime: 'HH:MM:ss Z',
        ignore: 'pid,hostname',
      },
    },
  },
});

const tokenStore = new TokenStore();
const PORT = 4115;
const REFRESH_COOKIE_NAME = 'auth_refresh_token';

// ----------------------------------------------------------------------------
// Middleware & Plugins
// ----------------------------------------------------------------------------

await fastify.register(cors, {
  origin: [
    'http://localhost:5173',
    'http://127.0.0.1:5173',
    'http://localhost:3000',
    'http://127.0.0.1:3000',
  ],
  credentials: true,
  methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
});

await fastify.register(cookie, {
  secret: 'auth_lab_cookie_secret_dev_only_123',
});

// Helper to set HttpOnly refresh token cookie
function setRefreshCookie(reply: FastifyReply, refreshToken: string) {
  reply.setCookie(REFRESH_COOKIE_NAME, refreshToken, {
    httpOnly: true,
    secure: false, // Set to true in HTTPS production
    sameSite: 'lax',
    path: '/api/auth',
    maxAge: 7 * 24 * 60 * 60, // 7 days in seconds
  });
}

function clearRefreshCookie(reply: FastifyReply) {
  reply.clearCookie(REFRESH_COOKIE_NAME, {
    path: '/api/auth',
  });
}

// Access Token Authentication Hook for Protected Endpoints
async function requireAuth(request: FastifyRequest, reply: FastifyReply) {
  const authHeader = request.headers.authorization;
  if (!authHeader || !authHeader.startsWith('Bearer ')) {
    return reply.status(401).send({
      error: 'unauthorized',
      code: 'MISSING_BEARER_TOKEN',
      message: 'Missing or malformed Authorization header',
    });
  }

  const token = authHeader.slice('Bearer '.length).trim();
  const verification = tokenStore.verifyAccessToken(token);

  if (!verification.valid) {
    return reply.status(401).send({
      error: 'unauthorized',
      code: 'TOKEN_EXPIRED',
      message: 'Access token has expired or is invalid. Refresh required.',
    });
  }

  // Attach verified user to request
  (request as any).user = { userId: verification.userId };
}

// ----------------------------------------------------------------------------
// OAuth2 PKCE & Authentication Endpoints
// ----------------------------------------------------------------------------

/**
 * Step 1: Authorization endpoint
 * Client sends code_challenge (S256) and user identity.
 * Returns an authorization code.
 */
fastify.post('/api/auth/authorize', async (request: FastifyRequest, reply: FastifyReply) => {
  const body = (request.body as any) || {};
  const { codeChallenge, codeChallengeMethod, userId = 'user_alice' } = body;

  if (!codeChallenge || codeChallengeMethod !== 'S256') {
    return reply.status(400).send({
      error: 'invalid_request',
      message: 'codeChallenge is required and codeChallengeMethod must be S256',
    });
  }

  const code = tokenStore.createAuthorizationCode(codeChallenge, userId);
  return reply.send({
    code,
    state: body.state,
  });
});

/**
 * Step 2: Token exchange endpoint (RFC 7636)
 * Client trades authorization code + code_verifier for tokens.
 * Issues Access Token (JSON) and Refresh Token (HttpOnly Cookie).
 */
fastify.post('/api/auth/token', async (request: FastifyRequest, reply: FastifyReply) => {
  const body = (request.body as any) || {};
  const { code, codeVerifier } = body;

  if (!code || !codeVerifier) {
    return reply.status(400).send({
      error: 'invalid_request',
      message: 'Both code and codeVerifier are required',
    });
  }

  const result = tokenStore.consumeAuthorizationCode(code, codeVerifier);
  if (!result.valid || !result.userId) {
    return reply.status(400).send({
      error: 'invalid_grant',
      message: result.error || 'Failed to verify PKCE authorization code',
    });
  }

  const tokens = tokenStore.issueInitialTokens(result.userId);
  setRefreshCookie(reply, tokens.refreshToken);

  return reply.send({
    access_token: tokens.accessToken,
    expires_in: tokens.expiresIn,
    token_type: 'Bearer',
    user_id: result.userId,
    family_id: tokens.familyId,
  });
});

/**
 * Step 3: Refresh Token Rotation endpoint
 * Reads refresh token from HttpOnly cookie, rotates it, and issues new tokens.
 * If reuse of a rotated token is detected, revokes the whole family!
 */
fastify.post('/api/auth/refresh', async (request: FastifyRequest, reply: FastifyReply) => {
  const body = (request.body as any) || {};
  // Allow passing simulated token in body for the replay attack demonstration button
  const cookieToken = request.cookies[REFRESH_COOKIE_NAME];
  const refreshToken = body.simulatedStolenToken || cookieToken;

  if (!refreshToken) {
    return reply.status(401).send({
      error: 'unauthorized',
      code: 'NO_REFRESH_TOKEN',
      message: 'No refresh token provided in HttpOnly cookie or body',
    });
  }

  try {
    const rotation = tokenStore.rotateRefreshToken(refreshToken);

    if (rotation.status === 'success') {
      setRefreshCookie(reply, rotation.newRefreshToken);
      return reply.send({
        access_token: rotation.newAccessToken,
        expires_in: rotation.expiresIn,
        token_type: 'Bearer',
        family_id: rotation.familyId,
      });
    }

    if (rotation.status === 'reuse_detected') {
      clearRefreshCookie(reply);
      return reply.status(401).send({
        error: 'token_reuse_detected',
        code: 'TOKEN_FAMILY_REVOKED',
        message: 'Security Alert: Refresh token reuse detected! Token family revoked.',
        family_id: rotation.familyId,
      });
    }

    // Invalid or expired
    clearRefreshCookie(reply);
    return reply.status(401).send({
      error: 'invalid_grant',
      code: 'REFRESH_EXPIRED_OR_INVALID',
      message: rotation.message,
    });
  } catch (err: any) {
    fastify.log.error(err);
    return reply.status(500).send({
      error: 'server_error',
      message: err.message,
    });
  }
});

/**
 * Logout: Revokes current session and clears refresh cookie
 */
fastify.post('/api/auth/logout', async (request: FastifyRequest, reply: FastifyReply) => {
  clearRefreshCookie(reply);
  return reply.send({ success: true, message: 'Logged out successfully' });
});

/**
 * Educational helper: returns the last rotated token so the user can test
 * clicking "Simulate Stolen Refresh Token Replay" in the frontend.
 */
fastify.get('/api/auth/last-rotated-token', async (_request: FastifyRequest, reply: FastifyReply) => {
  const token = tokenStore.getLastRotatedUsedToken();
  return reply.send({ lastRotatedToken: token });
});

// ----------------------------------------------------------------------------
// 5 Protected Widget Endpoints for Dashboard Simulation
// ----------------------------------------------------------------------------

fastify.get('/api/widgets/analytics', { preHandler: [requireAuth] }, async (request) => {
  return {
    widget: 'analytics',
    data: { activeUsers: 1420, requestsPerSec: 382.4, errorRatePercent: 0.04 },
    timestamp: new Date().toISOString(),
    user: (request as any).user,
  };
});

fastify.get('/api/widgets/profile', { preHandler: [requireAuth] }, async (request) => {
  return {
    widget: 'profile',
    data: { name: 'Alice Engineer', role: 'Security Architect', avatar: '👩‍💻' },
    timestamp: new Date().toISOString(),
    user: (request as any).user,
  };
});

fastify.get('/api/widgets/notifications', { preHandler: [requireAuth] }, async (request) => {
  return {
    widget: 'notifications',
    data: { unreadCount: 3, items: ['New PR assigned', 'Deployment completed', 'Alert: 401 surge'] },
    timestamp: new Date().toISOString(),
    user: (request as any).user,
  };
});

fastify.get('/api/widgets/activity', { preHandler: [requireAuth] }, async (request) => {
  return {
    widget: 'activity',
    data: { recentAction: 'OAuth2 Token Refresh Mutex Test', status: 'In Progress' },
    timestamp: new Date().toISOString(),
    user: (request as any).user,
  };
});

fastify.get('/api/widgets/billing', { preHandler: [requireAuth] }, async (request) => {
  return {
    widget: 'billing',
    data: { currentPlan: 'Enterprise Dev', quotaUsedPercent: 62.5, nextInvoiceDays: 14 },
    timestamp: new Date().toISOString(),
    user: (request as any).user,
  };
});

// ----------------------------------------------------------------------------
// Start Server
// ----------------------------------------------------------------------------

const start = async () => {
  try {
    await fastify.listen({ port: PORT, host: '0.0.0.0' });
    console.log(`\n🚀 Auth Server listening at http://localhost:${PORT}`);
    console.log(`Protected widgets at http://localhost:${PORT}/api/widgets/*\n`);
  } catch (err) {
    fastify.log.error(err);
    process.exit(1);
  }
};

start();
