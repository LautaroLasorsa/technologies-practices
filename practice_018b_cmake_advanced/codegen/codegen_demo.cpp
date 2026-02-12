// ╔══════════════════════════════════════════════════════════════════════════════╗
// ║  Phase 5: Code generation demo — print embedded file contents             ║
// ╚══════════════════════════════════════════════════════════════════════════════╝
//
// This file uses a GENERATED header (embedded_data.h) that is created at
// build time by the add_custom_command in codegen/CMakeLists.txt.
//
// The header contains the contents of data.txt as a constexpr string_view.
// If you modify data.txt and rebuild, the embedded content updates automatically.

#include "embedded_data.h"  // Generated at build time — see codegen/CMakeLists.txt

#include <iostream>
#include <string_view>

int main() {
    // ── Exercise Context ──────────────────────────────────────────────────
    // This exercise demonstrates add_custom_command for build-time code generation.
    // The embedded_data.h header was generated from data.txt during build.
    // This pattern is how real projects embed assets without runtime file I/O.

    // TODO(human): Print the embedded data and demonstrate it works.
    //
    // The generated header defines:
    //   namespace generated {
    //       inline constexpr std::string_view embedded_data = R"EMBED(...)EMBED";
    //   }
    //
    // Tasks:
    //   1. Print the embedded data to stdout
    //   2. Print the size of the embedded data (number of characters)
    //   3. Count the number of lines in the embedded data
    //
    // Hints:
    //   - generated::embedded_data is a std::string_view
    //   - Count lines by iterating and counting '\n' characters
    //   - std::count from <algorithm> works on string_view iterators
    //
    // Placeholder (compiles, but replace with your implementation):

    std::cout << "=== Embedded File Contents ===\n\n";
    std::cout << generated::embedded_data << "\n";
    std::cout << "=== End of Embedded Data ===\n";
    std::cout << "Size: " << generated::embedded_data.size() << " characters\n";

    return 0;
}
