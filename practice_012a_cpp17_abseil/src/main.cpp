// ============================================================================
// Practice 012a: C++17 Features & abseil-cpp
// ============================================================================
//
// This is the entry point that lists all phases and how to build/run them.
// Each phase is a separate executable -- run them individually.
//
// Build:
//   mkdir build && cd build && cmake .. && cmake --build .
//
// Run individual phases:
//   ./phase1_bindings          (or .\Debug\phase1_bindings.exe on Windows)
//   ./phase2_optional_variant
//   ./phase3_strings
//   ./phase4_containers
//   ./phase5_errors
//   ./phase6_constexpr
//
// Or build all at once:
//   cmake --build . --target all_phases
// ============================================================================

#include <iostream>
#include <string>

int main() {
    std::cout << R"(
  ╔═══════════════════════════════════════════════════════════╗
  ║       Practice 012a: C++17 Features & abseil-cpp         ║
  ╠═══════════════════════════════════════════════════════════╣
  ║                                                           ║
  ║  Phase 1: Structured Bindings & Control Flow              ║
  ║    -> phase1_bindings                                     ║
  ║                                                           ║
  ║  Phase 2: std::optional, std::variant & std::visit        ║
  ║    -> phase2_optional_variant                             ║
  ║                                                           ║
  ║  Phase 3: Abseil String Processing                        ║
  ║    -> phase3_strings                                      ║
  ║                                                           ║
  ║  Phase 4: Abseil Containers & Hashing                     ║
  ║    -> phase4_containers                                   ║
  ║                                                           ║
  ║  Phase 5: Error Handling with absl::StatusOr              ║
  ║    -> phase5_errors                                       ║
  ║                                                           ║
  ║  Phase 6: if-constexpr & Fold Expressions                 ║
  ║    -> phase6_constexpr                                    ║
  ║                                                           ║
  ╚═══════════════════════════════════════════════════════════╝

  Each phase is a separate executable.
  Build all: cmake --build . --target all_phases
  Then run each one individually.

  Tip: Start with Phase 1 and work through in order.
  Each TODO(human) marker shows what YOU need to implement.
)" << std::endl;

    return 0;
}
