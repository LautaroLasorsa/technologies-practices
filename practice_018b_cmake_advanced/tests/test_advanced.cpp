// ╔══════════════════════════════════════════════════════════════════════════════╗
// ║  Phase 4: Advanced CTest features — fixtures, expected failures, output   ║
// ╚══════════════════════════════════════════════════════════════════════════════╝
//
// This test executable supports multiple modes via command-line arguments,
// demonstrating CTest properties:
//   --edge-cases      → Test edge cases (negative input, boundary values)
//   --should-fail     → Intentionally returns non-zero (for WILL_FAIL tests)
//   --print-sequence  → Prints Fibonacci sequence (for PASS_REGULAR_EXPRESSION)
//   --fixture-setup   → Simulates fixture setup (creates state)
//   --fixture-main    → Simulates fixture test (uses state)
//   --fixture-cleanup → Simulates fixture cleanup (removes state)
//
// Each mode is registered as a separate CTest test in tests/CMakeLists.txt.

#include "mymath.h"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

// ─── Edge case tests ────────────────────────────────────────────────────────

int run_edge_cases() {
    std::cout << "=== Edge Case Tests ===\n\n";

    int failed = 0;

    // Factorial of 0
    if (mymath::factorial(0) != 1) {
        std::cout << "  [FAIL] factorial(0) should be 1\n";
        ++failed;
    } else {
        std::cout << "  [PASS] factorial(0) == 1\n";
    }

    // Factorial of negative
    if (mymath::factorial(-5) != 0) {
        std::cout << "  [FAIL] factorial(-5) should be 0\n";
        ++failed;
    } else {
        std::cout << "  [PASS] factorial(-5) == 0\n";
    }

    // Fibonacci of negative
    if (mymath::fibonacci(-10) != -1) {
        std::cout << "  [FAIL] fibonacci(-10) should be -1\n";
        ++failed;
    } else {
        std::cout << "  [PASS] fibonacci(-10) == -1\n";
    }

    // Large factorial (20! fits in int64_t)
    int64_t f20 = mymath::factorial(20);
    if (f20 != 2432902008176640000LL) {
        std::cout << "  [FAIL] factorial(20) should be 2432902008176640000\n";
        ++failed;
    } else {
        std::cout << "  [PASS] factorial(20) == 2432902008176640000\n";
    }

    // Empty fibonacci sequence
    auto seq = mymath::fibonacci_sequence(0);
    if (!seq.empty()) {
        std::cout << "  [FAIL] fibonacci_sequence(0) should be empty\n";
        ++failed;
    } else {
        std::cout << "  [PASS] fibonacci_sequence(0) is empty\n";
    }

    std::cout << "\n" << (failed == 0 ? "All edge case tests passed." : "Some edge case tests failed.") << "\n";
    return failed > 0 ? 1 : 0;
}


// ─── Expected failure ───────────────────────────────────────────────────────

int run_should_fail() {
    // This test intentionally returns non-zero.
    // CTest's WILL_FAIL property will invert the result: non-zero → PASS.
    std::cout << "This test intentionally fails (returns exit code 1).\n";
    std::cout << "CTest WILL_FAIL property inverts this to a PASS.\n";
    return 1;
}


// ─── Output verification ────────────────────────────────────────────────────

int run_print_sequence() {
    // CTest's PASS_REGULAR_EXPRESSION checks stdout for "0, 1, 1, 2, 3, 5"
    auto seq = mymath::fibonacci_sequence(10);
    std::cout << "Fibonacci(10): ";
    for (size_t i = 0; i < seq.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << seq[i];
    }
    std::cout << "\n";
    return 0;
}


// ─── Fixture simulation ────────────────────────────────────────────────────

static const char* FIXTURE_FILE = "test_fixture_data.tmp";

int run_fixture_setup() {
    std::cout << "=== Fixture Setup ===\n";
    std::cout << "Creating temporary state file: " << FIXTURE_FILE << "\n";

    // Write some test data that the main fixture test will read
    std::ofstream out(FIXTURE_FILE);
    if (!out) {
        std::cerr << "Failed to create fixture file!\n";
        return 1;
    }
    out << "42\n";
    out << "fibonacci_check\n";
    out.close();

    std::cout << "Fixture setup complete.\n";
    return 0;
}

int run_fixture_main() {
    std::cout << "=== Fixture Main Test ===\n";
    std::cout << "Reading state from: " << FIXTURE_FILE << "\n";

    std::ifstream in(FIXTURE_FILE);
    if (!in) {
        std::cerr << "Fixture file not found — did setup run first?\n";
        return 1;
    }

    int value;
    std::string check_type;
    in >> value >> check_type;
    in.close();

    std::cout << "Got value=" << value << ", check=" << check_type << "\n";

    // Verify fibonacci(value) makes sense
    if (check_type == "fibonacci_check") {
        int64_t result = mymath::fibonacci(value);
        std::cout << "fibonacci(" << value << ") = " << result << "\n";
        // fib(42) = 267914296
        if (result == 267914296) {
            std::cout << "  [PASS] Correct!\n";
            return 0;
        } else {
            std::cout << "  [FAIL] Expected 267914296\n";
            return 1;
        }
    }

    std::cerr << "Unknown check type: " << check_type << "\n";
    return 1;
}

int run_fixture_cleanup() {
    std::cout << "=== Fixture Cleanup ===\n";
    std::cout << "Removing temporary state file: " << FIXTURE_FILE << "\n";

    if (std::remove(FIXTURE_FILE) == 0) {
        std::cout << "Cleanup complete.\n";
    } else {
        std::cout << "File already removed or not found (OK).\n";
    }
    return 0;
}


// ─── Main: dispatch by argument ─────────────────────────────────────────────

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: test_advanced <mode>\n";
        std::cerr << "Modes: --edge-cases, --should-fail, --print-sequence,\n";
        std::cerr << "       --fixture-setup, --fixture-main, --fixture-cleanup\n";
        return 1;
    }

    const char* mode = argv[1];

    if (std::strcmp(mode, "--edge-cases") == 0)      return run_edge_cases();
    if (std::strcmp(mode, "--should-fail") == 0)      return run_should_fail();
    if (std::strcmp(mode, "--print-sequence") == 0)   return run_print_sequence();
    if (std::strcmp(mode, "--fixture-setup") == 0)    return run_fixture_setup();
    if (std::strcmp(mode, "--fixture-main") == 0)     return run_fixture_main();
    if (std::strcmp(mode, "--fixture-cleanup") == 0)  return run_fixture_cleanup();

    std::cerr << "Unknown mode: " << mode << "\n";
    return 1;
}
