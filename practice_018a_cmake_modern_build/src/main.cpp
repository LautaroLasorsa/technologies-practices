// ╔══════════════════════════════════════════════════════════════════════════════╗
// ║  Phase 1 + Phase 4: Main executable — uses greeter library + config header ║
// ╚══════════════════════════════════════════════════════════════════════════════╝
//
// This executable demonstrates:
//   - Linking against the `greeter` library target (Phase 1 & 3)
//   - Using a configure_file-generated header for version info (Phase 4)
//   - target_link_libraries(PRIVATE greeter) — main_app depends on greeter

#include "greeter.h"
#include "config.h"  // Generated at configure time (Phase 4)

#include <iostream>
#include <string>
#include <vector>

// TODO(human): Implement print_version_info()
//
// Print the project name, version (major.minor.patch), compiler ID,
// and whether greeting emoji is enabled.
//
// Use the macros from config.h:
//   PROJECT_NAME, PROJECT_VERSION, VERSION_MAJOR, VERSION_MINOR, VERSION_PATCH,
//   CXX_COMPILER_ID, ENABLE_GREETING_EMOJI
//
// Example output:
//   practice_018a v1.0.0 (compiled with MSVC)
//   Greeting emoji: enabled
//
// Hint: ENABLE_GREETING_EMOJI is an int (0 or 1), not a string.
void print_version_info() {
    std::cout << PROJECT_NAME << " v" << PROJECT_VERSION
              << " (compiled with " << CXX_COMPILER_ID << ")\n";
    std::cout << "Greeting emoji: "
              << (ENABLE_GREETING_EMOJI ? "enabled" : "disabled") << "\n";
}

int main() {
    print_version_info();
    std::cout << "---\n";

    // TODO(human): Create a Greeter, call greet(), and call greet_many()
    //
    // 1. Create a Greeter with your name (or any name)
    // 2. Print the result of greet()
    // 3. Create a vector of 3 names, call greet_many(), print each result
    //
    // This exercises the library linkage from Phase 3.
    //
    // Placeholder:
    Greeter g("CMake Learner");
    std::cout << g.greet() << "\n";

    std::vector<std::string> friends = {"Alice", "Bob", "Charlie"};
    for (const auto& greeting : g.greet_many(friends)) {
        std::cout << greeting << "\n";
    }

    return 0;
}
