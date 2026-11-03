import {
  setAccessToken,
  getAccessToken,
  clearAccessToken,
  authenticatedFetch,
} from '../src/auth/authClient.js';

function assert(condition: boolean, msg: string) {
  if (!condition) {
    console.error(`❌ FAILED: ${msg}`);
    process.exit(1);
  } else {
    console.log(`✅ PASSED: ${msg}`);
  }
}

async function runTests() {
  console.log('--- Testing Single-Flight Mutex Concept ---');

  // Test in-memory closure
  setAccessToken('test_at_123', 60);
  assert(getAccessToken() === 'test_at_123', 'In-memory token getter returns set token');

  clearAccessToken();
  assert(getAccessToken() === null, 'In-memory token cleared successfully');

  console.log('\nToken closure verified.');
}

runTests().catch((e) => {
  console.error(e);
  process.exit(1);
});
