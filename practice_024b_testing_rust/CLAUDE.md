# Practice 024b: Testing Patterns — Rust

## Technologies

- **Rust built-in test framework** — `#[test]`, `#[cfg(test)]`, `cargo test` for unit and integration tests
- **mockall 0.13** — Trait mocking with `#[automock]`, expectations, and call verification
- **proptest** — Property-based testing with strategies, shrinking, and invariant checking
- **tokio** — Async runtime for testing async code with `#[tokio::test]`
- **thiserror 2.0** — Derive macro for clean custom error types

## Stack

- Rust 1.77+ (cargo)

## Description

Build a Cache service and its comprehensive test suite, practicing every major Rust testing pattern. The domain is a key-value cache with TTL expiration: get/set operations, key validation, time-based expiry, and async HTTP fetching for cache-aside pattern. This domain naturally demonstrates trait-based mocking (TimeProvider, HttpClient), property-based invariants (keys valid, expired entries never returned), error handling (Result + custom errors), and async testing.

Focus is on testing patterns, not the cache itself — the cache code is partially provided and partially TODO(human).

### What you'll learn

1. **Built-in testing** — `#[test]`, `#[cfg(test)]`, assert macros, test organization
2. **TDD workflow** — red-green-refactor cycle for adding expiration
3. **Error handling** — custom error types with thiserror, testing Result variants
4. **Mocking with mockall** — `#[automock]`, expectations, trait-based dependency injection
5. **Property-based testing** — proptest strategies, shrinking, invariants
6. **Async testing** — `#[tokio::test]`, testing async operations
7. **Integration tests** — `tests/` directory, black-box testing

## Instructions

### Phase 1: Foundation — Built-in Testing (~15 min)

1. From this folder: `cargo build` to verify everything compiles
2. Explore the project structure: `src/` has the domain code, `tests/` has integration and property tests
3. Run the existing tests: `cargo test` — stubs compile, tests are empty
4. Open `src/cache.rs` — read the fully-implemented `set()` as a reference pattern
5. **User implements:** `get()` and `remove()` in `src/cache.rs`
6. **User implements:** all tests in the Phase 1 section of `src/cache.rs` (`test_set_and_get`, `test_get_missing_key`, `test_set_overwrites_existing`, `test_remove_existing_key`, `test_remove_nonexistent_key`)
7. Run: `cargo test` — verify all Phase 1 tests pass
8. Key question: Why co-locate tests with code in Rust (`#[cfg(test)]`)? How does this differ from Python's separate `tests/` directory convention?

### Phase 2: TDD — Adding Expiration (~20 min)

1. Write the failing tests FIRST (`test_expired_entry_returns_none`, `test_non_expired_entry_returns_value`, `test_no_ttl_never_expires`)
2. Run tests — observe compile errors and failures (this is "red" in Rust TDD: compile errors count!)
3. **User implements:** `CacheEntry::is_expired()` in `src/cache.rs` to make tests pass
4. Update `get()` to check expiration and remove expired entries
5. Run: `cargo test` — verify all Phase 2 tests pass (green)
6. Key question: How does TDD differ in Rust? (Compile errors = first "red" phase, unlike Python where code runs even with type errors)

### Phase 3: Error Handling (~15 min)

1. **User implements:** `CacheError` variants in `src/error.rs` using thiserror derive macros
2. **User implements:** `validate_key()` in `src/cache.rs`
3. Update `set()` — replace `CacheError::NotImplemented` with `CacheError::ValueTooLarge` once the variant exists
4. **User implements:** error tests (`test_empty_key_rejected`, `test_key_too_long_rejected`, `test_valid_key_accepted`)
5. Run: `cargo test` — verify all Phase 3 tests pass
6. Key question: Why test error variants explicitly in Rust? (Pattern matching is exhaustive — if you add a variant, the compiler forces you to handle it everywhere, but business logic correctness still needs tests)

### Phase 4: Mocking with mockall (~20 min)

1. Study `src/traits.rs` — `TimeProvider` already has `#[automock]`, `MockTimeProvider` is auto-generated
2. **User implements:** Add `#[cfg_attr(test, automock)]` to `HttpClient` trait in `src/traits.rs`
3. **User implements:** Phase 2 expiration tests using `MockTimeProvider` with sequenced expectations
4. Study how `cache_with_fixed_time()` helper creates a cache with mocked time
5. Key question: Why does Rust require trait abstraction for mocking? (No monkey-patching like Python — Rust's type system requires compile-time dispatch, so you must design for testability with traits)

### Phase 5: Property-Based Testing with proptest (~20 min)

1. Open `tests/property_tests.rs` — read the strategy definitions and test scaffolds
2. **User implements:** `valid_key()` strategy using proptest's regex strategy
3. **User implements:** all property tests (`prop_set_get_roundtrip`, `prop_remove_makes_key_absent`, `prop_len_matches_entries`, `prop_empty_key_always_rejected`)
4. Run: `cargo test --test property_tests` — observe proptest generating hundreds of cases
5. Key question: What invariants should a cache guarantee? (set/get roundtrip, removal is final, length accuracy, validation always rejects invalid input)

### Phase 6: Async Testing (~15 min)

1. **User implements:** async cache operations — a `fetch_or_get()` method that uses `HttpClient` trait
2. **User implements:** tests using `#[tokio::test]` and `MockHttpClient`
3. Key question: How does `#[tokio::test]` differ from `#[test]`? (It sets up a tokio runtime, allowing `.await` in test body)

### Phase 7: Integration Tests (~10 min)

1. Open `tests/integration_test.rs` — these test only the public API (black-box)
2. **User implements:** `test_full_lifecycle`, `test_key_isolation`, `test_error_recovery`
3. Run: `cargo test --test integration_test` — verify all integration tests pass
4. Key question: What's the difference between `#[cfg(test)] mod tests` in `src/` and files in `tests/`? (Unit tests have access to private items; integration tests only see `pub` API)

## Motivation

- **Rust-specific patterns**: Trait-based mocking and ownership-aware testing are unique to Rust and require different mental models than Python/JS testing
- **Complementary to 024a**: Compare Python (pytest + Hypothesis) vs Rust (built-in + mockall + proptest) testing philosophy side by side
- **Production quality**: Rust's type system catches many bugs at compile time, but tests verify BUSINESS LOGIC that types cannot express (e.g., "expired entries are never returned")
- **Career differentiator**: Few developers write comprehensive Rust test suites with property testing — this is a strong signal of engineering maturity

## References

- [The Rust Book: Writing Tests](https://doc.rust-lang.org/book/ch11-00-testing.html)
- [mockall Documentation](https://docs.rs/mockall/latest/mockall/)
- [proptest Book](https://proptest-rs.github.io/proptest/intro.html)
- [Tokio: Testing](https://tokio.rs/tokio/topics/testing)
- [thiserror Documentation](https://docs.rs/thiserror/latest/thiserror/)

## Commands

### Setup

| Command | Description |
|---------|-------------|
| `cargo build` | Compile the project and verify all stubs are valid |
| `cargo check` | Fast type-check without producing binaries |

### Run Tests

| Command | Description |
|---------|-------------|
| `cargo test` | Run all tests (unit + integration + property + doc tests) |
| `cargo test -- --nocapture` | Run all tests with stdout/stderr visible (useful for debugging println!) |
| `cargo test -- --show-output` | Run all tests and show output from passing tests too |

### Specific Tests

| Command | Description |
|---------|-------------|
| `cargo test test_set_and_get` | Run a single test by name |
| `cargo test test_expired` | Run all tests matching "test_expired" |
| `cargo test --lib` | Run only unit tests (co-located in src/) |
| `cargo test --test integration_test` | Run only the integration test file |
| `cargo test --test property_tests` | Run only property-based tests |

### Coverage

| Command | Description |
|---------|-------------|
| `cargo install cargo-tarpaulin` | Install tarpaulin coverage tool (one-time setup, Linux only) |
| `cargo tarpaulin --out Html` | Generate HTML coverage report (Linux only) |
| `cargo test 2>&1` | On Windows, use test output to manually verify coverage |

## State

`not-started`
