import { TokenStore } from '../src/tokenStore.js';

function assert(condition: boolean, msg: string) {
  if (!condition) {
    console.error(`❌ FAILED: ${msg}`);
    process.exit(1);
  } else {
    console.log(`✅ PASSED: ${msg}`);
  }
}

async function runTests() {
  console.log('--- Running TokenStore & Refresh Rotation Tests ---');
  const store = new TokenStore();

  // Test 1: Initial token issuance
  const initial = store.issueInitialTokens('user_123');
  assert(initial.accessToken.startsWith('at_'), 'Initial access token generated');
  assert(initial.refreshToken.startsWith('rt_'), 'Initial refresh token generated');
  assert(initial.familyId.startsWith('fam_'), 'Family ID generated');

  // Test 2: Rotate refresh token (Happy path)
  let rotation1;
  try {
    rotation1 = store.rotateRefreshToken(initial.refreshToken);
  } catch (err: any) {
    console.error('❌ Exercise 2 not implemented yet:', err.message);
    process.exit(1);
  }

  assert(rotation1.status === 'success', 'First rotation succeeds');
  if (rotation1.status === 'success') {
    assert(rotation1.newRefreshToken !== initial.refreshToken, 'New refresh token differs from old');
    assert(rotation1.familyId === initial.familyId, 'Rotation stays in the same family');

    // Test 3: Reuse Detection!
    console.log('Testing reuse of old rotated token...');
    const reuseAttempt = store.rotateRefreshToken(initial.refreshToken);
    assert(reuseAttempt.status === 'reuse_detected', 'Replay of old token detected as reuse');
    assert(reuseAttempt.familyId === initial.familyId, 'Reuse correctly identifies compromised family');

    const family = store.getFamily(initial.familyId);
    assert(family?.isRevoked === true, 'Token family is immediately revoked upon reuse');

    // Test 4: Subsequent refresh with legitimate token also rejected because family revoked
    const postCompromise = store.rotateRefreshToken(rotation1.newRefreshToken);
    assert(
      postCompromise.status === 'invalid_or_expired' || postCompromise.status === 'reuse_detected',
      'Tokens in revoked family cannot be refreshed'
    );
  }

  console.log('\n🎉 ALL Exercise 2 Tests Passed!\n');
}

runTests().catch((e) => {
  console.error(e);
  process.exit(1);
});
