# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  Phase 4: CTest — Test registration, labels, fixtures, properties         ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# This file demonstrates CTest features beyond basic add_test().
#
# Key concepts:
#   - LABELS: categorize tests for selective execution (ctest -L unit)
#   - TIMEOUT: fail tests that run too long
#   - WILL_FAIL: test is expected to return non-zero (invert pass/fail)
#   - FIXTURES_SETUP / FIXTURES_CLEANUP: ordered test dependencies
#   - PASS_REGULAR_EXPRESSION: test passes only if stdout matches a regex
#
# Docs: https://cmake.org/cmake/help/latest/command/set_tests_properties.html
# Docs: https://cmake.org/cmake/help/latest/prop_test/FIXTURES_REQUIRED.html

# ─── Test executable: unit tests ─────────────────────────────────────────────

add_executable(test_mymath test_mymath.cpp)
target_link_libraries(test_mymath PRIVATE mymath)

# ─── Test executable: advanced tests (fixtures, edge cases) ──────────────────

add_executable(test_advanced test_advanced.cpp)
target_link_libraries(test_advanced PRIVATE mymath)


# ─── Register tests with CTest ──────────────────────────────────────────────
#
# add_test(NAME <name> COMMAND <executable> [args...])
#
# NAME: unique identifier used by ctest to refer to this test
# COMMAND: the executable to run (use the target name — CMake resolves the path)

# TODO(human): Register the test_mymath executable as a CTest test.
#
# 1. Use add_test() with NAME "mymath_unit_tests" and COMMAND test_mymath
# 2. Set the LABELS property to "unit" so you can run just unit tests with:
#      ctest -L unit
# 3. Set TIMEOUT to 10 (seconds) — math tests should be near-instant
#
# Pattern:
#   add_test(NAME mymath_unit_tests COMMAND test_mymath)
#   set_tests_properties(mymath_unit_tests PROPERTIES
#       LABELS "unit"
#       TIMEOUT 10
#   )
#
# Fill in below:

add_test(NAME mymath_unit_tests COMMAND test_mymath)
set_tests_properties(mymath_unit_tests PROPERTIES
    LABELS "unit"
    TIMEOUT 10
)


# ─── Advanced test: edge cases ───────────────────────────────────────────────
#
# The test_advanced executable tests edge cases (negative input, overflow, etc.)
# It uses a non-zero exit code to signal specific test modes via command-line args.

add_test(NAME mymath_edge_cases COMMAND test_advanced --edge-cases)
set_tests_properties(mymath_edge_cases PROPERTIES
    LABELS "edge_case"
    TIMEOUT 10
)

# ─── Advanced test: expected failure ─────────────────────────────────────────
#
# WILL_FAIL: inverts the pass/fail logic.
#   - If the test returns 0 (success), CTest marks it as FAILED
#   - If the test returns non-zero (failure), CTest marks it as PASSED
#
# Useful for verifying that bad input is correctly rejected.

add_test(NAME mymath_expected_failure COMMAND test_advanced --should-fail)
set_tests_properties(mymath_expected_failure PROPERTIES
    LABELS "edge_case"
    WILL_FAIL TRUE
    TIMEOUT 5
)

# ─── Advanced test: output verification ──────────────────────────────────────
#
# PASS_REGULAR_EXPRESSION: test passes only if stdout matches the given regex.
# This is useful when the exit code doesn't convey enough information.

add_test(NAME mymath_output_check COMMAND test_advanced --print-sequence)
set_tests_properties(mymath_output_check PROPERTIES
    LABELS "output"
    PASS_REGULAR_EXPRESSION "0, 1, 1, 2, 3, 5"
    TIMEOUT 5
)

# ─── Fixtures: ordered test dependencies ─────────────────────────────────────
#
# CTest fixtures let you define setup → test → cleanup ordering.
# Unlike test framework fixtures (Google Test SetUp/TearDown), these are
# SEPARATE test executables that CTest ensures run in the correct order.
#
# Use case: a "setup" test creates a temp file, the main test reads it,
# and a "cleanup" test removes it.
#
# FIXTURES_SETUP: this test sets up the named fixture
# FIXTURES_REQUIRED: this test requires the named fixture (runs after setup)
# FIXTURES_CLEANUP: this test cleans up the named fixture (runs after main test)

add_test(NAME fixture_setup COMMAND test_advanced --fixture-setup)
set_tests_properties(fixture_setup PROPERTIES
    FIXTURES_SETUP "MathFixture"
    LABELS "fixture"
)

add_test(NAME fixture_main COMMAND test_advanced --fixture-main)
set_tests_properties(fixture_main PROPERTIES
    FIXTURES_REQUIRED "MathFixture"
    LABELS "fixture"
)

add_test(NAME fixture_cleanup COMMAND test_advanced --fixture-cleanup)
set_tests_properties(fixture_cleanup PROPERTIES
    FIXTURES_CLEANUP "MathFixture"
    LABELS "fixture"
)


# ─── Summary of registered tests ────────────────────────────────────────────
#
# Name                      │ Labels     │ Special
# ──────────────────────────┼────────────┼──────────────────────
# mymath_unit_tests         │ unit       │
# mymath_edge_cases         │ edge_case  │
# mymath_expected_failure   │ edge_case  │ WILL_FAIL
# mymath_output_check       │ output     │ PASS_REGULAR_EXPRESSION
# fixture_setup             │ fixture    │ FIXTURES_SETUP
# fixture_main              │ fixture    │ FIXTURES_REQUIRED
# fixture_cleanup           │ fixture    │ FIXTURES_CLEANUP
#
# Run selectively:
#   ctest --preset=msvc-debug -L unit          # only unit tests
#   ctest --preset=msvc-debug -L edge_case     # only edge case tests
#   ctest --preset=msvc-debug -L fixture       # fixtures run in order
#   ctest --preset=msvc-debug -N               # list tests without running
