# Practice 024b: Testing Patterns — Rust

## Technologies

- **Rust built-in test framework** — `#[test]`, `#[cfg(test)]`, `cargo test` for unit and integration tests
- **mockall 0.13** — Trait mocking with `#[automock]`, expectations, and call verification
- **proptest** — Property-based testing with strategies, shrinking, and invariant checking
- **tokio** — Async runtime for testing async code with `#[tokio::test]`
- **thiserror 2.0** — Derive macro for clean custom error types

## Stack

- Rust 1.77+ (cargo)

## Theoretical Context

### What are Rust's testing patterns and what problems do they solve?

Rust's testing ecosystem solves the **type-safe testing** problem: how do you test code in a language with strict compile-time guarantees (ownership, lifetimes) where traditional mocking techniques (monkey-patching, runtime reflection) don't work? The answer: **trait-based dependency injection**, **property-based testing** with compile-time guarantees, and **integration tests** that verify the public API without accessing internals.

**Built-in testing** (`#[test]`, `#[cfg(test)]`) is zero-dependency and fast — no setup, just `cargo test`. Tests live alongside code in `src/` (unit tests) or in `tests/` (integration tests).

**mockall** solves the **trait mocking** problem. Since Rust has no runtime type introspection, you can't replace methods at runtime like Python's `unittest.mock`. Instead, mockall generates mock implementations of traits via `#[automock]`, letting you set expectations (`expect_method().times(1).return_const(42)`) at compile time.

**proptest** brings property-based testing to Rust with **strategies**, **shrinking**, and **regression testing** (caches failing cases in `proptest-regressions/` to prevent regressions).

### How they work internally

**Rust's built-in test framework:**

1. **Test discovery**: `cargo test` compiles your code with `--test`, which enables `#[cfg(test)]` modules and marks `#[test]` functions as test entry points.
2. **Execution**: Each test runs in its own thread (parallel by default; use `--test-threads=1` for serial). The test harness captures stdout/stderr unless `--nocapture` is passed.
3. **Assertions**: `assert!`, `assert_eq!`, `assert_ne!` panic on failure. The test harness catches panics and reports them as test failures.
4. **Doc tests**: Code in `///` doc comments is extracted and compiled as separate tests, ensuring examples stay correct.

**Co-location vs separation:**

- **Unit tests** (`#[cfg(test)] mod tests` in `src/`): Have access to private items, test internal implementation. Co-located for convenience.
- **Integration tests** (`tests/*.rs`): Compiled as separate binaries, only see the crate's public API. Black-box testing.

**mockall: trait mocking via code generation:**

1. **`#[automock]` attribute**: Applied to a trait definition. mockall generates a `MockTraitName` struct with:
   - Mock methods that store expectations (call count, arguments, return values).
   - An expectation API (`.returning(|x| ...)`, `.times(1)`, `.with(predicate)`).
2. **Expectation checking**: When the mock method is called, it checks if the call matches stored expectations (arguments, call count). If not, it panics (test failure).
3. **Ownership and lifetimes**: mockall respects Rust's type system — you can mock traits with lifetimes, generic parameters, and `&mut` receivers. This is impossible in runtime-reflection-based mocking (Python, Java).

**proptest: property-based testing with strategies:**

1. **Strategy**: A type implementing the `Strategy` trait, describing how to generate random values. Examples: `any::<u32>()`, `prop::collection::vec(any::<String>(), 0..10)`, `prop::string::string_regex("[a-z]{1,20}")`.
2. **Execution**: proptest runs your test function 256 times (default) with generated inputs. If a test fails, it triggers **shrinking**.
3. **Shrinking**: proptest binary-searches the input space, trying simpler values (smaller numbers, shorter strings, fewer elements) that still fail. Stops at the minimal failing example.
4. **Regression file**: Failing cases are saved to `proptest-regressions/<test_name>.txt` and re-run on subsequent `cargo test` to prevent regressions.

**TDD in Rust (red-green-refactor):**

- **Red**: In Rust, "red" includes **compile errors** (a failed test might not compile yet because the function signature doesn't exist). This is a feature — the compiler guides implementation.
- **Green**: Implement the minimal code to make the test pass.
- **Refactor**: Extract functions, apply SOLID principles, run tests to ensure behavior unchanged.

### Key concepts

| Concept | Description |
|---------|-------------|
| **`#[test]`** | Attribute marking a function as a test; `cargo test` runs all such functions |
| **`#[cfg(test)]`** | Conditional compilation — code inside is only compiled during `cargo test` |
| **Unit tests** | Co-located in `src/` files, have access to private items via `#[cfg(test)] mod tests` |
| **Integration tests** | Separate files in `tests/`, compiled as independent binaries, only see public API |
| **`assert!` / `assert_eq!` / `assert_ne!`** | Panic on failure; test harness catches panic and reports failure |
| **Doc tests** | Code in `///` doc comments, extracted and compiled as tests |
| **Trait-based mocking** | Using trait abstractions + dependency injection to enable test doubles (mockall automates this) |
| **`#[automock]`** | mockall attribute generating a mock implementation of a trait with expectation API |
| **`#[tokio::test]`** | Async test macro setting up a Tokio runtime for testing async functions |
| **proptest strategy** | A type describing how to generate random test data (e.g., `any::<u32>()`, `vec(0..10, any::<String>())`) |
| **Shrinking** | proptest's algorithm for minimizing a failing input to the simplest counterexample |
| **Regression file** | proptest saves failing cases to `proptest-regressions/` to prevent regressions in future runs |

### Ecosystem context

**Alternatives and trade-offs:**

| Approach | Strengths | Weaknesses |
|----------|-----------|------------|
| **Built-in `#[test]`** | Zero dependencies, fast, parallel by default | No fixtures, parametrization requires macros/loops |
| **mockall** | Type-safe trait mocking, compile-time checks | Requires trait abstraction upfront (can't mock free functions or structs) |
| **proptest** | Finds edge cases automatically, shrinks to minimal failing input | Slower (256 runs/test), requires thinking in invariants |
| **rstest** (not used here) | Parametrization and fixtures like pytest | Extra dependency, less idiomatic than built-in tests |
| **mockers / mockito** | Alternative mocking libraries | Less mature than mockall, different API styles |

**When to use each:**

- **Built-in tests**: Default choice for unit tests and simple integration tests.
- **mockall**: When testing code with external dependencies (network, filesystem, time) via trait abstraction.
- **proptest**: Testing invariants on data structures, parsers, encoders, math functions — anywhere properties hold universally.
- **Async tests (`#[tokio::test]`)**: Testing async functions that return `Future` (common in web servers, async I/O).

**Limitations:**

- **mockall**: Requires designing for testability upfront (trait abstractions). Can't mock a concrete `struct` or free function without wrapping it in a trait.
- **proptest**: Not suitable for tests requiring specific inputs (e.g., "user 123 exists in DB"). Use for algorithmic correctness, not business logic with fixtures.
- **Compile-time testing**: Rust's type system catches many bugs at compile time, but tests still verify **business logic** that types can't express (e.g., "expired entries are never returned").

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
