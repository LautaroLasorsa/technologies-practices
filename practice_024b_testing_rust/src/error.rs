//! Cache error types.
//!
//! Phase 3: Implement custom error variants using thiserror.

use thiserror::Error;

/// Errors that can occur during cache operations.
///
/// TODO(human): Add error variants to this enum.
///
/// Required variants:
/// 1. InvalidKey(String) — key is empty or too long (> 256 chars)
///    Display: "Invalid cache key: {reason}"
///    Example: InvalidKey("key is empty".to_string())
///
/// 2. ValueTooLarge { size: usize, limit: usize } — value exceeds size limit
///    Display: "Value too large: {size} bytes exceeds limit of {limit} bytes"
///
/// 3. Expired — entry was found but has expired (internal use)
///    Display: "Cache entry has expired"
///
/// 4. FetchError(String) — HTTP fetch failed (used in async cache-aside)
///    Display: "Failed to fetch: {0}"
///
/// Hint: Use #[error("...")] attribute for each variant.
/// The thiserror derive macro turns #[error("...")] into a Display impl.
///
/// Example of how a variant with #[error] looks:
///
///   #[error("Invalid cache key: {0}")]
///   InvalidKey(String),
///
/// For struct-style variants (named fields), reference fields by name:
///
///   #[error("Value too large: {size} bytes exceeds limit of {limit} bytes")]
///   ValueTooLarge { size: usize, limit: usize },
///
/// After implementing, delete the NotImplemented variant and update
/// src/cache.rs set() to use ValueTooLarge instead of NotImplemented.
#[derive(Error, Debug)]
pub enum CacheError {
    // TODO(human): Define the error variants described above

    // Placeholder variant — remove after implementing the real variants
    #[error("Not yet implemented")]
    NotImplemented,
}
