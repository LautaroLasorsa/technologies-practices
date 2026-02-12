// ╔══════════════════════════════════════════════════════════════════════════════╗
// ║  Phase 4: Unit tests for mymath library                                   ║
// ╚══════════════════════════════════════════════════════════════════════════════╝
//
// Simple assert-based tests — no framework needed.
// Registered with CTest in tests/CMakeLists.txt.
//
// The point is not the tests themselves but how CMake + CTest discovers,
// runs, and reports them.

#include "mymath.h"

#include <cassert>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

// ─── Helpers ────────────────────────────────────────────────────────────────

static int tests_passed = 0;
static int tests_failed = 0;

void check(bool condition, const char* test_name) {
    if (condition) {
        std::cout << "  [PASS] " << test_name << "\n";
        ++tests_passed;
    } else {
        std::cout << "  [FAIL] " << test_name << "\n";
        ++tests_failed;
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

void test_add() {
    // ── Exercise Context ──────────────────────────────────────────────────
    // This exercise writes tests that CTest will discover and run.
    // These tests validate the mymath library across different build configurations.

    // TODO(human): Test mymath::add with several cases.
    //
    // Suggestions:
    //   - add(0, 0) == 0
    //   - add(2, 3) == 5
    //   - add(-1, 1) == 0
    //   - add(-5, -3) == -8
    //
    // Use the check() helper above.
    //
    // Placeholder:
    check(mymath::add(0, 0) == 0, "add(0, 0) == 0");
    check(mymath::add(2, 3) == 5, "add(2, 3) == 5");
    check(mymath::add(-1, 1) == 0, "add(-1, 1) == 0");
    check(mymath::add(-5, -3) == -8, "add(-5, -3) == -8");
}

void test_multiply() {
    // ── Exercise Context ──────────────────────────────────────────────────
    // CTest can filter tests by labels. These unit tests will get a "unit" label,
    // allowing selective execution (ctest -L unit).

    // TODO(human): Test mymath::multiply with several cases.
    //
    // Suggestions:
    //   - multiply(0, 100) == 0
    //   - multiply(3, 7) == 21
    //   - multiply(-2, 5) == -10
    //   - multiply(-3, -4) == 12
    //
    // Placeholder:
    check(mymath::multiply(0, 100) == 0, "multiply(0, 100) == 0");
    check(mymath::multiply(3, 7) == 21, "multiply(3, 7) == 21");
    check(mymath::multiply(-2, 5) == -10, "multiply(-2, 5) == -10");
    check(mymath::multiply(-3, -4) == 12, "multiply(-3, -4) == 12");
}

void test_factorial() {
    // ── Exercise Context ──────────────────────────────────────────────────
    // This tests a more complex function. CTest will report pass/fail for each
    // registered test, making it easy to identify which functions break.

    // TODO(human): Test mymath::factorial with several cases.
    //
    // Suggestions:
    //   - factorial(0) == 1
    //   - factorial(1) == 1
    //   - factorial(5) == 120
    //   - factorial(10) == 3628800
    //   - factorial(20) == 2432902008176640000 (fits in int64_t)
    //   - factorial(-1) == 0 (error case)
    //
    // Placeholder:
    check(mymath::factorial(0) == 1, "factorial(0) == 1");
    check(mymath::factorial(1) == 1, "factorial(1) == 1");
    check(mymath::factorial(5) == 120, "factorial(5) == 120");
    check(mymath::factorial(10) == 3628800, "factorial(10) == 3628800");
    check(mymath::factorial(-1) == 0, "factorial(-1) == 0 (error case)");
}

void test_fibonacci() {
    // ── Exercise Context ──────────────────────────────────────────────────
    // Fibonacci tests validate correctness across edge cases (n=0, n=1, negative).
    // Real libraries test edge cases exhaustively—CTest makes this practical.

    // TODO(human): Test mymath::fibonacci with several cases.
    //
    // Suggestions:
    //   - fibonacci(0) == 0
    //   - fibonacci(1) == 1
    //   - fibonacci(2) == 1
    //   - fibonacci(10) == 55
    //   - fibonacci(20) == 6765
    //   - fibonacci(-1) == -1 (error case)
    //
    // Placeholder:
    check(mymath::fibonacci(0) == 0, "fibonacci(0) == 0");
    check(mymath::fibonacci(1) == 1, "fibonacci(1) == 1");
    check(mymath::fibonacci(2) == 1, "fibonacci(2) == 1");
    check(mymath::fibonacci(10) == 55, "fibonacci(10) == 55");
    check(mymath::fibonacci(-1) == -1, "fibonacci(-1) == -1 (error case)");
}

void test_fibonacci_sequence() {
    // ── Exercise Context ──────────────────────────────────────────────────
    // This tests a container-returning function. CTest doesn't care about implementation—
    // it just runs the executable and checks the exit code.

    // TODO(human): Test mymath::fibonacci_sequence.
    //
    // Suggestions:
    //   - fibonacci_sequence(0) returns empty vector
    //   - fibonacci_sequence(1) returns {0}
    //   - fibonacci_sequence(7) returns {0, 1, 1, 2, 3, 5, 8}
    //
    // Placeholder:
    auto seq0 = mymath::fibonacci_sequence(0);
    check(seq0.empty(), "fibonacci_sequence(0) is empty");

    auto seq1 = mymath::fibonacci_sequence(1);
    check(seq1.size() == 1 && seq1[0] == 0, "fibonacci_sequence(1) == {0}");

    auto seq7 = mymath::fibonacci_sequence(7);
    std::vector<int64_t> expected = {0, 1, 1, 2, 3, 5, 8};
    check(seq7 == expected, "fibonacci_sequence(7) == {0,1,1,2,3,5,8}");
}

// ─── Main ───────────────────────────────────────────────────────────────────

int main() {
    std::cout << "=== MyMath Unit Tests ===\n\n";

    test_add();
    test_multiply();
    test_factorial();
    test_fibonacci();
    test_fibonacci_sequence();

    std::cout << "\n--- Results: " << tests_passed << " passed, "
              << tests_failed << " failed ---\n";

    return tests_failed > 0 ? 1 : 0;
}
