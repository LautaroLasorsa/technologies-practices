//! Cache implementation — the main learning module.
//!
//! Some operations are fully implemented as reference.
//! Others are marked TODO(human) for you to implement.
//!
//! The cache is generic over T: TimeProvider, which allows tests
//! to inject a MockTimeProvider for deterministic time control.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use crate::error::CacheError;
use crate::traits::TimeProvider;

/// A single cache entry with value and expiration.
///
/// Each entry stores the value, when it was created, and optionally
/// when it expires. If expires_at is None, the entry never expires.
#[derive(Debug, Clone)]
pub struct CacheEntry {
    pub value: String,
    pub created_at: Instant,
    pub expires_at: Option<Instant>,
}

impl CacheEntry {
    /// Check if this entry has expired.
    ///
    /// TODO(human): Implement expiration check (Phase 2, TDD).
    ///
    /// Steps:
    /// 1. If expires_at is None, the entry never expires → return false
    /// 2. If expires_at is Some(t), compare t with `now`:
    ///    - If now >= t → expired → return true
    ///    - If now < t → still valid → return false
    ///
    /// Why take `now` as parameter instead of calling Instant::now()?
    /// This makes the function TESTABLE without mocking. The caller
    /// passes the current time (from TimeProvider), and tests can
    /// pass any time they want. Pure functions are easier to test.
    ///
    /// This is a common Rust testing pattern: instead of mocking
    /// global state, you pass dependencies as parameters. The
    /// TimeProvider trait in cache.rs is the next level of this
    /// pattern (dependency injection via generics).
    ///
    /// Implementation hint:
    ///   match self.expires_at {
    ///       None => false,
    ///       Some(expires) => now >= expires,
    ///   }
    ///
    /// Or more concisely:
    ///   self.expires_at.map_or(false, |expires| now >= expires)
    pub fn is_expired(&self, _now: Instant) -> bool {
        // TODO(human): implement expiration check
        false
    }
}

/// Key-value cache with TTL support.
///
/// Generic over T: TimeProvider so tests can inject MockTimeProvider.
/// In production, use Cache<SystemTimeProvider>.
/// In tests, use Cache<MockTimeProvider>.
///
/// This pattern is called "dependency injection via generics" — Rust's
/// zero-cost abstraction means the generic is monomorphized at compile
/// time, so there's no runtime overhead for the trait indirection.
pub struct Cache<T: TimeProvider> {
    store: HashMap<String, CacheEntry>,
    time_provider: T,
    #[allow(dead_code)] // Used in validate_key() once implemented (Phase 3)
    max_key_length: usize,
    max_value_size: usize,
}

impl<T: TimeProvider> Cache<T> {
    /// Create a new cache with the given time provider.
    pub fn new(time_provider: T) -> Self {
        Self {
            store: HashMap::new(),
            time_provider,
            max_key_length: 256,
            max_value_size: 1_048_576, // 1 MB
        }
    }

    /// Validate a cache key.
    ///
    /// TODO(human): Implement key validation (Phase 3).
    ///
    /// Rules:
    /// 1. Key must not be empty
    ///    → return Err(CacheError::InvalidKey("key is empty".to_string()))
    ///
    /// 2. Key must not exceed max_key_length
    ///    → return Err(CacheError::InvalidKey(
    ///          format!("key too long: {} > {}", key.len(), self.max_key_length)
    ///      ))
    ///
    /// 3. Key must contain only alphanumeric chars, hyphens, underscores, dots
    ///    → return Err(CacheError::InvalidKey(
    ///          "key contains invalid characters".to_string()
    ///      ))
    ///
    /// 4. If valid → return Ok(())
    ///
    /// Implementation hint:
    ///   if key.is_empty() {
    ///       return Err(CacheError::InvalidKey("key is empty".to_string()));
    ///   }
    ///   if key.len() > self.max_key_length {
    ///       return Err(CacheError::InvalidKey(
    ///           format!("key too long: {} > {}", key.len(), self.max_key_length)
    ///       ));
    ///   }
    ///   if !key.chars().all(|c| c.is_alphanumeric() || c == '-' || c == '_' || c == '.') {
    ///       return Err(CacheError::InvalidKey(
    ///           "key contains invalid characters".to_string()
    ///       ));
    ///   }
    ///   Ok(())
    ///
    /// Note: This function returns Result<(), CacheError>. The () means
    /// "nothing useful on success" — we only care about the error case.
    /// This is a common Rust pattern for validation functions.
    fn validate_key(&self, _key: &str) -> Result<(), CacheError> {
        // TODO(human): implement key validation
        Ok(())
    }

    /// Store a value in the cache with optional TTL.
    ///
    /// This is fully implemented as a reference pattern.
    /// Study this before implementing get().
    ///
    /// Note the pattern:
    /// 1. Validate inputs (key, value size)
    /// 2. Get current time from time_provider
    /// 3. Create entry with calculated expiration
    /// 4. Insert into store
    pub fn set(
        &mut self,
        key: String,
        value: String,
        ttl: Option<Duration>,
    ) -> Result<(), CacheError> {
        self.validate_key(&key)?;

        if value.len() > self.max_value_size {
            // TODO: After Phase 3, replace NotImplemented with:
            // return Err(CacheError::ValueTooLarge {
            //     size: value.len(),
            //     limit: self.max_value_size,
            // });
            return Err(CacheError::NotImplemented);
        }

        let now = self.time_provider.now();
        let entry = CacheEntry {
            value,
            created_at: now,
            expires_at: ttl.map(|d| now + d),
        };
        self.store.insert(key, entry);
        Ok(())
    }

    /// Retrieve a value from the cache.
    ///
    /// TODO(human): Implement cache retrieval (Phase 1 basic, Phase 2 with expiration).
    ///
    /// Steps:
    /// 1. Look up key in self.store
    /// 2. If not found → return None
    /// 3. If found, check is_expired(self.time_provider.now()):
    ///    - If expired → remove from store (self.store.remove(key)) → return None
    ///    - If valid → return Some(entry.value.clone())
    ///
    /// Phase 1: Just implement steps 1-2 (ignore expiration)
    /// Phase 2: Add step 3 (expiration check) via TDD
    ///
    /// Borrow checker note:
    /// You can't call self.store.get(key) and self.store.remove(key) in the
    /// same scope because get() borrows immutably and remove() borrows mutably.
    ///
    /// Solution: check expiration first with a temporary bool, then act on it.
    ///
    ///   let now = self.time_provider.now();
    ///   let expired = self.store.get(key).map_or(false, |e| e.is_expired(now));
    ///   if expired {
    ///       self.store.remove(key);
    ///       return None;
    ///   }
    ///   self.store.get(key).map(|e| e.value.clone())
    ///
    /// Why clone()? The HashMap owns the String. We return a copy so the
    /// caller gets an owned String without taking ownership from the map.
    pub fn get(&mut self, _key: &str) -> Option<String> {
        // TODO(human): implement cache retrieval
        None
    }

    /// Remove a value from the cache.
    ///
    /// TODO(human): Implement cache removal (Phase 1).
    ///
    /// Steps:
    /// 1. Remove the key from self.store
    /// 2. Return true if the key existed, false otherwise
    ///
    /// Implementation hint:
    ///   self.store.remove(key).is_some()
    ///
    /// HashMap::remove(key) returns Option<V>:
    /// - Some(value) if the key existed (and was removed)
    /// - None if the key didn't exist
    /// .is_some() converts this to the bool we want.
    pub fn remove(&mut self, _key: &str) -> bool {
        // TODO(human): implement cache removal
        false
    }

    /// Return the number of (non-expired) entries in the cache.
    ///
    /// Note: this counts only entries that haven't expired yet.
    /// Expired entries remain in the store until accessed via get()
    /// (lazy expiration), but len() filters them out for accuracy.
    pub fn len(&self) -> usize {
        let now = self.time_provider.now();
        self.store.values().filter(|e| !e.is_expired(now)).count()
    }

    /// Check if the cache is empty (no non-expired entries).
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

// ===========================================================================
// Unit Tests — co-located with the code (Rust convention)
// ===========================================================================
//
// In Rust, unit tests live in a #[cfg(test)] mod tests block inside the
// same file as the code they test. This has several advantages:
//
// 1. Tests can access PRIVATE functions and fields (unlike tests/ directory)
// 2. Tests are compiled ONLY in test mode (zero cost in production)
// 3. Tests live next to the code they test — easy to find and maintain
// 4. The #[cfg(test)] attribute means this entire module is stripped from
//    release builds — no binary bloat
//
// Compare with Python: Python tests are typically in a separate tests/
// directory. Rust supports both: #[cfg(test)] for unit tests (white-box)
// and tests/ directory for integration tests (black-box).

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, dead_code)]

    use super::*;
    use crate::traits::MockTimeProvider;
    use std::time::{Duration, Instant};

    /// Helper: create a cache with a mock time provider that returns a fixed time.
    ///
    /// This helper encapsulates the MockTimeProvider setup pattern so each
    /// test doesn't have to repeat it. The mock always returns the same
    /// Instant, simulating "frozen time".
    ///
    /// Returns both the cache and the fixed instant, so tests can calculate
    /// relative times (e.g., now + Duration::from_secs(60)).
    fn cache_with_fixed_time() -> (Cache<MockTimeProvider>, Instant) {
        let now = Instant::now();
        let mut mock = MockTimeProvider::new();
        mock.expect_now().returning(move || now);
        (Cache::new(mock), now)
    }

    // -- Phase 1: Basic get/set tests ----------------------------------------

    #[test]
    fn test_set_and_get() {
        // TODO(human): Implement this test.
        //
        // This is the simplest possible cache test — set a key, get it back.
        // Start here to verify your get() implementation works.
        //
        // Steps:
        // 1. Create a cache: let (mut cache, _) = cache_with_fixed_time();
        // 2. Set a key: cache.set("hello".into(), "world".into(), None).unwrap();
        // 3. Get the key: assert_eq!(cache.get("hello"), Some("world".to_string()));
        //
        // Notes:
        // - "hello".into() converts &str to String (required by set's signature)
        // - .unwrap() panics if set() returns Err — fine in tests, we expect Ok
        // - assert_eq! compares two values and panics with a diff if they differ
        // - None for TTL means "no expiration"

        // TODO(human): implement test
    }

    #[test]
    fn test_get_missing_key() {
        // TODO(human): Implement this test.
        //
        // Verify that getting a key that was never set returns None.
        // This tests the "not found" path in get().
        //
        // Steps:
        // 1. Create a cache (don't set anything)
        // 2. Get a key that was never set: cache.get("nonexistent")
        // 3. Assert result is None: assert_eq!(result, None);
        //
        // This is a boundary test — always test the "empty" or "missing" case.

        // TODO(human): implement test
    }

    #[test]
    fn test_set_overwrites_existing() {
        // TODO(human): Implement this test.
        //
        // Verify that setting the same key twice overwrites the first value.
        // HashMap::insert replaces the value for an existing key.
        //
        // Steps:
        // 1. Create a cache
        // 2. Set key "x" to "first": cache.set("x".into(), "first".into(), None).unwrap();
        // 3. Set key "x" to "second": cache.set("x".into(), "second".into(), None).unwrap();
        // 4. Get "x" → should be "second": assert_eq!(cache.get("x"), Some("second".to_string()));
        //
        // Cache is a map — same key replaces the previous value.
        // This is important behavior to test because it's a design decision
        // (alternative: reject duplicate keys with an error).

        // TODO(human): implement test
    }

    #[test]
    fn test_remove_existing_key() {
        // TODO(human): Implement this test.
        //
        // Verify that removing an existing key returns true and the key is gone.
        //
        // Steps:
        // 1. Create a cache
        // 2. Set key "temp" to "value"
        // 3. Remove "temp": let removed = cache.remove("temp");
        // 4. Assert removed is true: assert!(removed);
        // 5. Get "temp": assert_eq!(cache.get("temp"), None);
        //
        // Two assertions: remove returns true AND the key is actually gone.
        // Testing both is important — remove could return true without
        // actually removing (a subtle bug).

        // TODO(human): implement test
    }

    #[test]
    fn test_remove_nonexistent_key() {
        // TODO(human): Implement this test.
        //
        // Verify that removing a key that doesn't exist returns false.
        //
        // Steps:
        // 1. Create a cache (don't set anything)
        // 2. Remove a key that was never set: let removed = cache.remove("ghost");
        // 3. Assert removed is false: assert!(!removed);
        //
        // This is a no-op — the cache should not panic or error.

        // TODO(human): implement test
    }

    // -- Phase 2: Expiration tests (TDD) ------------------------------------

    #[test]
    fn test_expired_entry_returns_none() {
        // TODO(human): TDD — write this test FIRST, then implement is_expired().
        //
        // This is the key TDD test. Write it, watch it fail (red),
        // then implement is_expired() to make it pass (green).
        //
        // Scenario: Set a key with 60s TTL, then check 120s later → expired.
        //
        // Steps:
        // 1. Create a fixed time `now`:
        //      let now = Instant::now();
        //
        // 2. Create a MockTimeProvider with SEQUENCED expectations:
        //    - First call returns `now` (used by set() to calculate expires_at)
        //    - Subsequent calls return `now + 120s` (used by get() to check expiration)
        //
        //      let mut mock = MockTimeProvider::new();
        //      let now_for_set = now;
        //      mock.expect_now()
        //          .times(1)
        //          .returning(move || now_for_set);
        //      let later = now + Duration::from_secs(120);
        //      mock.expect_now()
        //          .returning(move || later);
        //
        // 3. Create cache and set key with TTL = 60 seconds:
        //      let mut cache = Cache::new(mock);
        //      cache.set("key".into(), "val".into(), Some(Duration::from_secs(60))).unwrap();
        //
        // 4. Get key → should return None (120s > 60s TTL):
        //      assert_eq!(cache.get("key"), None);
        //
        // How mockall sequencing works:
        // - .times(1) means "this expectation matches exactly 1 call"
        // - After that expectation is exhausted, the next expect_now() takes over
        // - This lets you simulate time advancing between set() and get()

        // TODO(human): implement test
    }

    #[test]
    fn test_non_expired_entry_returns_value() {
        // TODO(human): Implement this test.
        //
        // Complement of test_expired_entry_returns_none.
        // Scenario: Set a key with 60s TTL, check 30s later → still valid.
        //
        // Steps:
        // 1. Create a fixed time `now`
        // 2. Mock: first call returns `now`, subsequent calls return `now + 30s`
        //      let mut mock = MockTimeProvider::new();
        //      let now_for_set = now;
        //      mock.expect_now()
        //          .times(1)
        //          .returning(move || now_for_set);
        //      let later = now + Duration::from_secs(30);
        //      mock.expect_now()
        //          .returning(move || later);
        // 3. Set key with TTL = 60s
        // 4. Get key → should return Some("val") (30s < 60s TTL)
        //
        // Together with test_expired_entry_returns_none, these two tests
        // fully specify the expiration boundary behavior.

        // TODO(human): implement test
    }

    #[test]
    fn test_no_ttl_never_expires() {
        // TODO(human): Implement this test.
        //
        // Scenario: Set a key with NO TTL, advance time by 1 hour → still valid.
        //
        // Steps:
        // 1. Create a fixed time `now`
        // 2. Mock: first call returns `now`, subsequent calls return `now + 3600s`
        //      let mut mock = MockTimeProvider::new();
        //      let now_for_set = now;
        //      mock.expect_now()
        //          .times(1)
        //          .returning(move || now_for_set);
        //      let much_later = now + Duration::from_secs(3600);
        //      mock.expect_now()
        //          .returning(move || much_later);
        // 3. Set key with ttl = None (no expiration)
        // 4. Get key → should still return the value
        //
        // Entries without TTL live forever (until explicitly removed).
        // This tests the None branch in is_expired().

        // TODO(human): implement test
    }

    // -- Phase 3: Error handling tests ----------------------------------------

    #[test]
    fn test_empty_key_rejected() {
        // TODO(human): Implement this test.
        //
        // Verify that an empty key is rejected with CacheError::InvalidKey.
        //
        // Steps:
        // 1. Create a cache: let (mut cache, _) = cache_with_fixed_time();
        // 2. Try to set an empty key:
        //      let result = cache.set("".into(), "val".into(), None);
        // 3. Assert result is Err:
        //      assert!(result.is_err());
        // 4. Verify the error variant using matches! macro:
        //      assert!(matches!(result, Err(CacheError::InvalidKey(_))));
        //
        // The matches! macro is Rust's way to assert on enum variants
        // without needing to destructure the full value. The _ means
        // "any string inside InvalidKey".
        //
        // Alternative: use unwrap_err() and match:
        //   let err = result.unwrap_err();
        //   match err {
        //       CacheError::InvalidKey(msg) => assert!(msg.contains("empty")),
        //       other => panic!("Expected InvalidKey, got: {:?}", other),
        //   }

        // TODO(human): implement test
    }

    #[test]
    fn test_key_too_long_rejected() {
        // TODO(human): Implement this test.
        //
        // Verify that a key exceeding 256 characters is rejected.
        //
        // Steps:
        // 1. Create a cache
        // 2. Create a long key: let long_key = "a".repeat(257);
        // 3. Try to set it: let result = cache.set(long_key, "val".into(), None);
        // 4. Assert: assert!(matches!(result, Err(CacheError::InvalidKey(_))));
        //
        // Boundary testing: 256 chars should be OK, 257 should fail.
        // Optionally test both sides of the boundary:
        //   - "a".repeat(256) → Ok
        //   - "a".repeat(257) → Err

        // TODO(human): implement test
    }

    #[test]
    fn test_valid_key_accepted() {
        // TODO(human): Implement this test.
        //
        // Verify that various valid key formats are accepted.
        //
        // Test several valid keys:
        //   - "hello" (simple alphanumeric)
        //   - "key-with-dashes" (hyphens allowed)
        //   - "key_with_underscores" (underscores allowed)
        //   - "key.with.dots" (dots allowed)
        //   - "CamelCase123" (mixed case + digits)
        //   - "a" (single character — minimum valid key)
        //   - "a".repeat(256) (maximum length — boundary)
        //
        // For each: cache.set(key, "val".into(), None).unwrap();
        // (unwrap panics if any returns Err, which is what we want)
        //
        // This is a positive test — it verifies the "happy path".
        // Combined with the rejection tests above, we have full
        // coverage of the validation logic.

        // TODO(human): implement test
    }

    // -- Phase 5: Property-based tests ----------------------------------------
    // Note: proptest tests use proptest! macro — see tests/property_tests.rs
    // The proptest! macro doesn't work well inside #[cfg(test)] mod tests,
    // so property tests live in the tests/ directory as integration tests.
}
