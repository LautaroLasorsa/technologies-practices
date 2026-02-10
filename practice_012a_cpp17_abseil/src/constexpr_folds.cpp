// ============================================================================
// Phase 6: if-constexpr & Fold Expressions
// ============================================================================
//
// These are the most "template-metaprogramming" features of C++17.
// In CP you use templates rarely. In production, these enable:
//   - Type-safe serialization/deserialization
//   - Zero-cost generic logging
//   - Compile-time dispatch (like Rust's monomorphization)
//
// if constexpr:
//   Evaluates a condition at COMPILE TIME. The false branch is discarded
//   entirely -- it doesn't even need to compile. This replaces SFINAE
//   and tag dispatch for many use cases.
//
// Fold expressions:
//   Expand parameter packs with an operator. For example:
//     (args + ...) expands to arg1 + arg2 + arg3 + ...
//     (std::cout << ... << args) prints all args
//
// Docs:
//   https://en.cppreference.com/w/cpp/language/if#constexpr_if
//   https://en.cppreference.com/w/cpp/language/fold
// ============================================================================

#include <iostream>
#include <string>
#include <type_traits>

#include "absl/strings/str_cat.h"

// ─── Helpers ─────────────────────────────────────────────────────────────────

void print_section(const std::string &title) {
  std::cout << "\n=== " << title << " ===\n\n";
}

// ─── Exercise 1: Variadic print with fold expressions ────────────────────────
//
// Write a function that prints all its arguments separated by ", ".
//
// In CP you might write a macro: #define DBG(x) cout<<#x<<" = "<<(x)<<endl;
// Fold expressions let you do this generically for any number of args.

// TODO(human): Implement print_all using a fold expression.
//
// Signature:
//   template <typename... Args>
//   void print_all(const Args&... args)
//
// Body: use a fold expression to print each arg followed by a separator.
//
// Option A - comma fold with a helper:
//   std::cout << "  ";
//   ((std::cout << args << ", "), ...);   // unary right fold
//   std::cout << "\n";
//
// Option B - cleaner with a lambda:
//   std::string sep;
//   ((std::cout << sep << args, sep = ", "), ...);
//   std::cout << "\n";
//
// The key insight: (expr, ...) expands expr for each arg in the pack.
// The comma operator sequences the expressions left to right.

template <typename... Args> void print_all(const Args &...args) {
  // TODO(human): Replace this stub with a fold expression.
  // See hints above.
  std::cout << "   ";
  ((std::cout << args << ", "), ...);
  std::cout << " \n";
}

// ─── Exercise 2: Fold expression for sum/product/all/any ─────────────────────
//
// Fold expressions work with any binary operator.

// TODO(human): Implement sum_all using a fold expression.
//   template <typename... Args>
//   auto sum_all(Args... args) { return (args + ...); }
//
// The pattern (args + ...) is a "unary right fold".
// It expands to: arg1 + (arg2 + (arg3 + ...))

template <typename... Args> auto sum_all(Args... args) {
  // TODO(human): return (args + ...);
  return (args + ...);
}

// TODO(human): Implement all_true using a fold expression.
//   Hint: (args && ...) -- unary right fold with &&
//   Returns true only if all arguments are truthy.

template <typename... Args> bool all_true(Args... args) {
  // TODO(human): return (args && ...);
  return (args && ...);
}

// TODO(human): Implement any_true using a fold expression.
//   Hint: (args || ...) -- unary right fold with ||

template <typename... Args> bool any_true(Args... args) {
  // TODO(human): return (args || ...);
  return (args || ...);
}

// ─── Exercise 3: Type-aware serializer with if-constexpr ─────────────────────
//
// Write a function that converts a value to a "serialized" string
// representation, handling different types at compile time.
//
// Without if-constexpr, you'd need template specialization or SFINAE.
// With if-constexpr, the code reads like a normal if-else.

// TODO(human): Implement serialize using if-constexpr.
//
// template <typename T>
// std::string serialize(const T& value) {
//     if constexpr (std::is_integral_v<T>) {
//         return absl::StrCat("int:", value);
//     } else if constexpr (std::is_floating_point_v<T>) {
//         return absl::StrCat("float:", value);
//     } else if constexpr (std::is_same_v<T, std::string>) {
//         return absl::StrCat("str:\"", value, "\"");
//     } else if constexpr (std::is_same_v<T, bool>) {
//         // Note: bool is also integral! Put this check BEFORE is_integral_v
//         // or use is_same_v to be specific.
//         return value ? "bool:true" : "bool:false";
//     } else {
//         // This is a compile-time error for unsupported types.
//         static_assert(sizeof(T) == 0, "Unsupported type for serialize");
//     }
// }
//
// KEY INSIGHT: Without if-constexpr, ALL branches must compile for ALL types.
// With if-constexpr, only the taken branch needs to compile.
// For example, absl::StrCat("int:", value) wouldn't compile for std::string,
// but it's fine because that branch is discarded at compile time.

template <typename T> std::string serialize(const T &value) {
  // TODO(human): Implement with if-constexpr. See hints above.
  // Be careful about the order: check bool before integral!
  if constexpr (std::is_same_v<T, bool>) {
    return value ? "bool:true" : "bool:false";
  } else if constexpr (std::is_integral_v<T>) {
    return absl::StrCat("int:", value);
  } else if constexpr (std::is_floating_point_v<T>) {
    return absl::StrCat("float:", value);
  } else if constexpr (std::is_same_v<T, std::string>) {
    return absl::StrCat("str:\"", value, "\"");
  } else {
    static_assert(sizeof(T) == 0, "Unsupported type for serialize");
  }
}

// ─── Exercise 4: Variadic serialize_all combining folds + if-constexpr ───────
//
// Combine fold expressions with if-constexpr to serialize multiple
// heterogeneous values into a single comma-separated string.

// TODO(human): Implement serialize_all.
//
// template <typename... Args>
// std::string serialize_all(const Args&... args) {
//     std::string result;
//     std::string sep;
//     // Fold expression that calls serialize() on each arg:
//     ((result += sep + serialize(args), sep = ", "), ...);
//     return result;
// }
//
// This is the payoff: fold expressions handle the iteration,
// if-constexpr handles the per-type logic, and it all resolves
// at compile time with zero runtime overhead.

template <typename... Args> std::string serialize_all(const Args &...args) {
  // TODO(human): Implement. See hints above.
  std::string result, sep;
  ((absl::StrAppend(&result, sep, serialize(args)), sep = ", "), ...);
  return result;
}

// ─── Exercise 5: Compile-time size calculator ────────────────────────────────
//
// Use if-constexpr to compute the "serialized size" of a type at compile time.
// This is useful for pre-allocating buffers.

template <typename T> constexpr size_t serialized_size_hint() {
  // TODO(human): Return an estimated serialized size for each type.
  //
  // if constexpr (std::is_same_v<T, int>) {
  //     return 4 + 11;  // "int:" + max int digits
  // } else if constexpr (std::is_same_v<T, double>) {
  //     return 6 + 20;  // "float:" + max double digits
  // } else if constexpr (std::is_same_v<T, std::string>) {
  //     return 6 + 64;  // "str:\"" + estimated max length + "\""
  // } else {
  //     return 32;  // fallback estimate
  // }
  //
  // Note: constexpr means this is evaluated at compile time.
  // You can use it as a template parameter or array size!
  if constexpr (std::is_same_v<T, int>) {
    return 15;
  } else if constexpr (std::is_same_v<T, double>) {
    return 26;
  } else if constexpr (std::is_same_v<T, std::string>) {
    return 70;
  } else {
    return 32;
  }
}

// ─── Main ────────────────────────────────────────────────────────────────────

int main() {
  std::cout << "Phase 6: if-constexpr & Fold Expressions\n";
  std::cout << std::string(55, '=') << "\n";

  // --- Exercise 1: Variadic print ---
  print_section("Exercise 1: print_all with fold expressions");
  std::cout << "  print_all(1, 2.5, \"hello\", 'x'):\n";
  print_all(1, 2.5, "hello", 'x');

  std::cout << "  print_all(\"one\", \"two\", \"three\"):\n";
  print_all("one", "two", "three");

  // --- Exercise 2: Fold operators ---
  print_section("Exercise 2: Fold expression operators");
  std::cout << "  sum_all(1, 2, 3, 4, 5) = " << sum_all(1, 2, 3, 4, 5) << "\n";
  std::cout << "  sum_all(1.5, 2.5, 3.0) = " << sum_all(1.5, 2.5, 3.0) << "\n";
  std::cout << "  all_true(true, true, true) = " << std::boolalpha
            << all_true(true, true, true) << "\n";
  std::cout << "  all_true(true, false, true) = " << std::boolalpha
            << all_true(true, false, true) << "\n";
  std::cout << "  any_true(false, false, true) = " << std::boolalpha
            << any_true(false, false, true) << "\n";
  std::cout << "  any_true(false, false, false) = " << std::boolalpha
            << any_true(false, false, false) << "\n";

  // --- Exercise 3: Type-aware serializer ---
  print_section("Exercise 3: Serialize with if-constexpr");
  std::cout << "  serialize(42)       = " << serialize(42) << "\n";
  std::cout << "  serialize(3.14)     = " << serialize(3.14) << "\n";
  std::cout << "  serialize(\"hello\") = " << serialize(std::string("hello"))
            << "\n";
  // Note: serialize(true) tests bool vs integral priority

  // --- Exercise 4: Combine folds + if-constexpr ---
  print_section("Exercise 4: serialize_all (folds + if-constexpr)");
  std::cout << "  serialize_all(42, 3.14, \"hello\"):\n";
  std::cout << "  " << serialize_all(42, 3.14, std::string("hello")) << "\n";

  // --- Exercise 5: Compile-time size hints ---
  print_section("Exercise 5: constexpr size hints");
  std::cout << "  serialized_size_hint<int>()    = "
            << serialized_size_hint<int>() << " bytes\n";
  std::cout << "  serialized_size_hint<double>() = "
            << serialized_size_hint<double>() << " bytes\n";
  std::cout << "  serialized_size_hint<std::string>() = "
            << serialized_size_hint<std::string>() << " bytes\n";

  // Compile-time usage proof: array sized by constexpr function
  constexpr size_t buf_size = serialized_size_hint<int>();
  [[maybe_unused]] char buffer[buf_size > 0 ? buf_size : 1];
  std::cout << "\n  Allocated buffer of " << sizeof(buffer)
            << " bytes at compile time.\n";

  return 0;
}
