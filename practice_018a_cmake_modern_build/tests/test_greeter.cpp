// ╔══════════════════════════════════════════════════════════════════════════════╗
// ║  Phase 3: Test executable — tests for the greeter library                 ║
// ╚══════════════════════════════════════════════════════════════════════════════╝
//
// Simple assert-based tests (no framework needed).
// This executable links against the `greeter` library:
//   target_link_libraries(test_greeter PRIVATE greeter)
//
// CMake sees this as just another executable target — the fact that it's a
// "test" is a convention. We register it with CTest via add_test().

#include "greeter.h"

#include <cassert>
#include <iostream>
#include <string>
#include <vector>

// ─── Helper ─────────────────────────────────────────────────────────────────

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

void test_constructor() {
    // TODO(human): Test that the Greeter stores the name correctly.
    //
    // 1. Create a Greeter with name "Alice"
    // 2. Check that greeter.name() == "Alice"
    //
    // Placeholder:
    Greeter g("Alice");
    check(g.name() == "Alice", "constructor stores name");
}

void test_greet() {
    // TODO(human): Test that greet() returns the expected format.
    //
    // 1. Create a Greeter with name "Bob"
    // 2. Call greet() and check the result contains "Bob"
    //    (exact format depends on your implementation)
    //
    // Placeholder:
    Greeter g("Bob");
    std::string result = g.greet();
    check(result.find("Bob") != std::string::npos, "greet() contains name");
    check(!result.empty(), "greet() is not empty");
}

void test_greet_many() {
    // TODO(human): Test greet_many() with multiple names.
    //
    // 1. Create a Greeter with name "Host"
    // 2. Call greet_many({"Alice", "Bob", "Charlie"})
    // 3. Check: result has 3 elements
    // 4. Check: each element contains the corresponding guest name
    // 5. Check: each element contains the host name "Host"
    //
    // Placeholder:
    Greeter g("Host");
    std::vector<std::string> guests = {"Alice", "Bob", "Charlie"};
    auto results = g.greet_many(guests);

    check(results.size() == 3, "greet_many() returns correct count");

    for (size_t i = 0; i < guests.size(); ++i) {
        std::string test_label = "greet_many() result contains guest: " + guests[i];
        check(results[i].find(guests[i]) != std::string::npos, test_label.c_str());
        std::string host_label = "greet_many() result contains host for: " + guests[i];
        check(results[i].find("Host") != std::string::npos, host_label.c_str());
    }
}

void test_greet_many_empty() {
    // TODO(human): Test greet_many() with an empty list.
    //
    // Should return an empty vector.
    //
    // Placeholder:
    Greeter g("Solo");
    auto results = g.greet_many({});
    check(results.empty(), "greet_many() with empty input returns empty");
}

// ─── Main ───────────────────────────────────────────────────────────────────

int main() {
    std::cout << "=== Greeter Library Tests ===\n\n";

    test_constructor();
    test_greet();
    test_greet_many();
    test_greet_many_empty();

    std::cout << "\n--- Results: " << tests_passed << " passed, "
              << tests_failed << " failed ---\n";

    return tests_failed > 0 ? 1 : 0;
}
