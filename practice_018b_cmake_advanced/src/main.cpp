// ╔══════════════════════════════════════════════════════════════════════════════╗
// ║  Main executable — uses the mymath library                                ║
// ╚══════════════════════════════════════════════════════════════════════════════╝
//
// Demonstrates linking against a project-local library.
// Built with: cmake --build --preset=msvc-debug --target main_app

#include "mymath.h"

#include <cstdint>
#include <iostream>
#include <vector>

int main() {
    std::cout << "=== Practice 018b: Advanced CMake ===\n\n";

    // Basic operations
    std::cout << "add(3, 4) = " << mymath::add(3, 4) << "\n";
    std::cout << "multiply(6, 7) = " << mymath::multiply(6, 7) << "\n";
    std::cout << "factorial(10) = " << mymath::factorial(10) << "\n";
    std::cout << "fibonacci(10) = " << mymath::fibonacci(10) << "\n";

    // Fibonacci sequence
    std::cout << "\nFirst 15 Fibonacci numbers: ";
    auto seq = mymath::fibonacci_sequence(15);
    for (size_t i = 0; i < seq.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << seq[i];
    }
    std::cout << "\n";

    return 0;
}
