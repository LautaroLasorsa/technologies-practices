import { createHash, randomBytes } from 'node:crypto';

export interface RefreshTokenRecord {
  token: string;
  familyId: string;
  userId: string;
  isUsed: boolean;
  expiresAt: number;
}

export interface TokenFamilyRecord {
  familyId: string;
  userId: string;
  isRevoked: boolean;
  createdAt: number;
}

export interface AccessTokenRecord {
  token: string;
  userId: string;
  expiresAt: number;
}

export type RotationResult =
  | {
      status: 'success';
      newAccessToken: string;
      newRefreshToken: string;
      expiresIn: number;
      familyId: string;
    }
  | {
      status: 'reuse_detected';
      familyId: string;
      message: string;
    }
  | {
      status: 'invalid_or_expired';
      message: string;
    };

export interface AuthCodeRecord {
  code: string;
  codeChallenge: string;
  userId: string;
  expiresAt: number;
}

/**
 * TokenStore manages PKCE authorization codes, in-memory access tokens,
 * and rotating refresh token families with automatic reuse detection.
 */
export class TokenStore {
  // In-memory collections (for production, persist in Redis / Postgres)
  private authCodes = new Map<string, AuthCodeRecord>();
  private accessTokens = new Map<string, AccessTokenRecord>();
  private refreshTokens = new Map<string, RefreshTokenRecord>();
  private tokenFamilies = new Map<string, TokenFamilyRecord>();

  // Short lifespans for hands-on lab demonstration
  public readonly ACCESS_TOKEN_LIFESPAN_MS = 15_000; // 15 seconds
  public readonly REFRESH_TOKEN_LIFESPAN_MS = 7 * 24 * 60 * 60 * 1000; // 7 days

  /**
   * Helper: Generate a secure cryptographic random hex or url-safe string
   */
  private generateRandomToken(bytes: number = 32): string {
    return randomBytes(bytes).toString('hex');
  }

  // --------------------------------------------------------------------------
  // Authorization Code & PKCE Verification
  // --------------------------------------------------------------------------

  public createAuthorizationCode(codeChallenge: string, userId: string): string {
    const code = `code_${this.generateRandomToken(16)}`;
    this.authCodes.set(code, {
      code,
      codeChallenge,
      userId,
      expiresAt: Date.now() + 5 * 60 * 1000, // 5 minutes
    });
    return code;
  }

  public consumeAuthorizationCode(
    code: string,
    codeVerifier: string
  ): { valid: boolean; userId?: string; error?: string } {
    const record = this.authCodes.get(code);
    if (!record) {
      return { valid: false, error: 'Authorization code not found or expired' };
    }

    // Single-use code: delete immediately
    this.authCodes.delete(code);

    if (Date.now() > record.expiresAt) {
      return { valid: false, error: 'Authorization code expired' };
    }

    // RFC 7636 S256 verification: Base64URL(SHA-256(code_verifier))
    const hash = createHash('sha256').update(codeVerifier, 'utf8').digest();
    const calculatedChallenge = hash
      .toString('base64')
      .replace(/\+/g, '-')
      .replace(/\//g, '_')
      .replace(/=+$/, '');

    if (calculatedChallenge !== record.codeChallenge) {
      return { valid: false, error: 'PKCE code_verifier challenge verification failed' };
    }

    return { valid: true, userId: record.userId };
  }

  // --------------------------------------------------------------------------
  // Initial Token Issuance
  // --------------------------------------------------------------------------

  public issueInitialTokens(userId: string): {
    accessToken: string;
    refreshToken: string;
    expiresIn: number;
    familyId: string;
  } {
    const familyId = `fam_${this.generateRandomToken(12)}`;
    this.tokenFamilies.set(familyId, {
      familyId,
      userId,
      isRevoked: false,
      createdAt: Date.now(),
    });

    const accessToken = `at_${this.generateRandomToken(24)}`;
    this.accessTokens.set(accessToken, {
      token: accessToken,
      userId,
      expiresAt: Date.now() + this.ACCESS_TOKEN_LIFESPAN_MS,
    });

    const refreshToken = `rt_${this.generateRandomToken(32)}`;
    this.refreshTokens.set(refreshToken, {
      token: refreshToken,
      familyId,
      userId,
      isUsed: false,
      expiresAt: Date.now() + this.REFRESH_TOKEN_LIFESPAN_MS,
    });

    return {
      accessToken,
      refreshToken,
      expiresIn: Math.floor(this.ACCESS_TOKEN_LIFESPAN_MS / 1000),
      familyId,
    };
  }

  // --------------------------------------------------------------------------
  // Access Token Verification
  // --------------------------------------------------------------------------

  public verifyAccessToken(token: string): { valid: boolean; userId?: string } {
    const record = this.accessTokens.get(token);
    if (!record) return { valid: false };
    if (Date.now() > record.expiresAt) {
      this.accessTokens.delete(token);
      return { valid: false };
    }
    return { valid: true, userId: record.userId };
  }

  // --------------------------------------------------------------------------
  // Exercise 2: Refresh Token Rotation & Family Reuse Detection
  // --------------------------------------------------------------------------

  /**
   * Exercise 2: TODO(human)
   *
   * Implement Refresh Token Rotation with Token Family Reuse Detection.
   *
   * Context & Why it matters:
   * 1. Refresh Token Rotation (RTR): Every time a client exchanges a refresh token
   *    at /api/auth/refresh, that token must be immediately invalidated (marked isUsed = true
   *    or replaced), and a brand-new refresh token issued in the same family.
   * 2. Reuse Detection: If an attacker steals a refresh token (e.g., from network inspection
   *    or memory) and later tries to present it AFTER the legitimate client already rotated it,
   *    we detect that record.isUsed === true!
   * 3. Immediate Family Revocation: In that event, the attacker and the victim both hold
   *    refresh tokens in that family. We MUST immediately revoke the entire family
   *    (family.isRevoked = true and purge all related access tokens), forcing both to log in
   *    again and neutralizing the breach.
   *
   * Steps to implement:
   * 1. Retrieve the record from this.refreshTokens.get(oldRefreshToken).
   *    If not found, return { status: 'invalid_or_expired', message: 'Token not found' }.
   * 2. Check if the token's family exists in this.tokenFamilies.
   *    If not found or family.isRevoked is true, return:
   *    { status: 'invalid_or_expired', message: 'Token family revoked or invalid' }.
   * 3. Check expiration: if Date.now() > record.expiresAt, return:
   *    { status: 'invalid_or_expired', message: 'Refresh token expired' }.
   * 4. REUSE DETECTION (CRUCIAL):
   *    If record.isUsed is true:
   *      - Mark family.isRevoked = true
   *      - Revoke all access tokens belonging to this user
   *      - Return { status: 'reuse_detected', familyId: record.familyId, message: 'Refresh token reuse detected! Family revoked.' }
   * 5. ROTATION (Happy path):
   *      - Mark record.isUsed = true
   *      - Generate newRefreshToken and newAccessToken
   *      - Store newRefreshToken with isUsed = false, familyId = record.familyId, userId = record.userId
   *      - Store newAccessToken with userId = record.userId
   *      - Return {
   *          status: 'success',
   *          newAccessToken,
   *          newRefreshToken,
   *          expiresIn: Math.floor(this.ACCESS_TOKEN_LIFESPAN_MS / 1000),
   *          familyId: record.familyId
   *        }
   */
  public rotateRefreshToken(oldRefreshToken: string): RotationResult {
    // TODO(human): Implement Refresh Token Rotation with Family Reuse Detection
    throw new Error('TODO(human): Exercise 2 - Implement rotateRefreshToken in backend/src/tokenStore.ts');
  }

  // --------------------------------------------------------------------------
  // Session Revocation & Simulation Helpers
  // --------------------------------------------------------------------------

  public revokeFamily(familyId: string): void {
    const family = this.tokenFamilies.get(familyId);
    if (family) {
      family.isRevoked = true;
      // Invalidate all access tokens for this user
      for (const [key, at] of this.accessTokens.entries()) {
        if (at.userId === family.userId) {
          this.accessTokens.delete(key);
        }
      }
    }
  }

  public getFamily(familyId: string): TokenFamilyRecord | undefined {
    return this.tokenFamilies.get(familyId);
  }

  /**
   * Helper for the educational UI: Find the most recently rotated (used) token
   * to simulate an attacker replaying a stolen token.
   */
  public getLastRotatedUsedToken(): string | null {
    for (const record of this.refreshTokens.values()) {
      if (record.isUsed) {
        return record.token;
      }
    }
    return null;
  }
}
