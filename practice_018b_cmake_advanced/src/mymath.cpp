// ╔══════════════════════════════════════════════════════════════════════════════╗
// ║  mymath library implementation                                            ║
// ╚══════════════════════════════════════════════════════════════════════════════╝

#include "mymath.h"

#include <vector>

namespace mymath {

int add(int a, int b) {
    return a + b;
}

int multiply(int a, int b) {
    return a * b;
}

int64_t factorial(int n) {
    if (n < 0) return 0;
    int64_t result = 1;
    for (int i = 2; i <= n; ++i) {
        result *= i;
    }
    return result;
}

int64_t fibonacci(int n) {
    if (n < 0) return -1;
    if (n == 0) return 0;
    if (n == 1) return 1;

    int64_t prev = 0, curr = 1;
    for (int i = 2; i <= n; ++i) {
        int64_t next = prev + curr;
        prev = curr;
        curr = next;
    }
    return curr;
}

std::vector<int64_t> fibonacci_sequence(int count) {
    std::vector<int64_t> seq;
    seq.reserve(count > 0 ? count : 0);
    for (int i = 0; i < count; ++i) {
        seq.push_back(fibonacci(i));
    }
    return seq;
}

} // namespace mymath
