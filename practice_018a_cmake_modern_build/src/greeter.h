#ifndef GREETER_H
#define GREETER_H

// ╔══════════════════════════════════════════════════════════════════════════════╗
// ║  Phase 3: Library target — public header                                   ║
// ╚══════════════════════════════════════════════════════════════════════════════╝
//
// This header is part of the `greeter` library target (add_library in CMakeLists.txt).
// It demonstrates:
//   - PUBLIC include directories (consumers see this header)
//   - Separation of interface (.h) from implementation (.cpp)
//   - target_compile_features(cxx_std_17) propagation

#include <string>
#include <vector>

/// A simple greeter that produces formatted greetings.
/// The point is NOT the C++ — it's how CMake builds and links this library.
class Greeter {
public:
    /// Construct a greeter with a given name.
    explicit Greeter(std::string name);

    // TODO(human): Implement these methods in greeter.cpp
    //
    // greet() should return a greeting string like "Hello, <name>!"
    // If ENABLE_GREETING_EMOJI is defined and truthy, append a waving hand.
    //
    // Hint: You'll need to #include "config.h" in greeter.cpp to check the option.
    [[nodiscard]] std::string greet() const;

    // TODO(human): Implement greet_many()
    //
    // Given a list of names, return a vector of greetings.
    // Use std::transform or a range-for to apply greet-like logic to each name.
    [[nodiscard]] std::vector<std::string> greet_many(const std::vector<std::string>& names) const;

    [[nodiscard]] const std::string& name() const { return name_; }

private:
    std::string name_;
};

#endif // GREETER_H
