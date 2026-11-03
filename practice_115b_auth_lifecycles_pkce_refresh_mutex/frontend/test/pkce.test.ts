import { generateCodeVerifier, deriveCodeChallenge } from '../src/auth/pkce.js';

function assert(condition: boolean, msg: string) {
  if (!condition) {
    console.error(`❌ FAILED: ${msg}`);
    process.exit(1);
  } else {
    console.log(`✅ PASSED: ${msg}`);
  }
}

async function runTests() {
  console.log('--- Running PKCE Unit Tests (RFC 7636) ---');

  // Test 1: Verifier generation
  try {
    const verifier = generateCodeVerifier(64);
    assert(verifier.length === 64, `Verifier length is 64 (got ${verifier.length})`);
    assert(
      /^[A-Za-z0-9\-._~]+$/.test(verifier),
      'Verifier contains only RFC 7636 unreserved characters'
    );

    const verifier2 = generateCodeVerifier(64);
    assert(verifier !== verifier2, 'Subsequent verifiers are cryptographically unique');
  } catch (err: any) {
    console.error('❌ Exercise 1A failed:', err.message);
    process.exit(1);
  }

  // Test 2: Official RFC 7636 Appendix B Test Vector
  // Verifier:  dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk
  // Challenge: E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM
  try {
    const rfcVerifier = 'dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk';
    const expectedChallenge = 'E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM';

    const derivedChallenge = await deriveCodeChallenge(rfcVerifier);
    assert(
      derivedChallenge === expectedChallenge,
      `Derived challenge matches official RFC 7636 test vector!\nExpected: ${expectedChallenge}\nActual:   ${derivedChallenge}`
    );
  } catch (err: any) {
    console.error('❌ Exercise 1B failed:', err.message);
    process.exit(1);
  }

  console.log('\n🎉 ALL Exercise 1 PKCE Tests Passed!\n');
}

runTests().catch((e) => {
  console.error(e);
  process.exit(1);
});
