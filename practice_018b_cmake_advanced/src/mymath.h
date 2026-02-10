#ifndef MYMATH_H
#define MYMATH_H

// ╔══════════════════════════════════════════════════════════════════════════════╗
// ║  Simple math library — the CMake configuration is the learning material    ║
// ╚══════════════════════════════════════════════════════════════════════════════╝
//
// Four functions: add, multiply, factorial, fibonacci.
// Intentionally simple C++ — the focus is on CMake, not the math.

#include <cstdint>
#include <vector>

namespace mymath {

/// Add two integers.
int add(int a, int b);

/// Multiply two integers.
int multiply(int a, int b);

/// Compute n! (factorial). Returns 0 for negative input.
/// Uses int64_t to handle factorials up to 20.
int64_t factorial(int n);

/// Compute the n-th Fibonacci number (0-indexed).
/// fib(0) = 0, fib(1) = 1, fib(2) = 1, fib(3) = 2, ...
/// Returns -1 for negative input.
int64_t fibonacci(int n);

/// Return the first `count` Fibonacci numbers starting from fib(0).
std::vector<int64_t> fibonacci_sequence(int count);

} // namespace mymath

#endif // MYMATH_H
