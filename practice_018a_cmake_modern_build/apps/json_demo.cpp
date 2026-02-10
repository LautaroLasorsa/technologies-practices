// ╔══════════════════════════════════════════════════════════════════════════════╗
// ║  Phase 2: FetchContent demo — nlohmann/json + fmt                         ║
// ╚══════════════════════════════════════════════════════════════════════════════╝
//
// This executable uses two libraries pulled from GitHub via FetchContent:
//   - nlohmann/json: parse and manipulate JSON
//   - fmt: format strings (the library behind C++20's std::format)
//
// The CMake side:
//   target_link_libraries(json_demo PRIVATE nlohmann_json::nlohmann_json fmt::fmt)
//
// Because FetchContent_MakeAvailable was called in cmake/FetchDeps.cmake,
// these targets are available as if they were part of our project.

#include <nlohmann/json.hpp>
#include <fmt/format.h>
#include <fmt/ranges.h>

#include <iostream>
#include <string>
#include <vector>

using json = nlohmann::json;

int main() {
    // ─── Part A: Parse a JSON string ────────────────────────────────────────
    //
    // TODO(human): Parse the JSON string below into a nlohmann::json object.
    //
    // JSON to parse:
    //   {
    //     "project": "cmake_practice",
    //     "version": [1, 0, 0],
    //     "features": ["FetchContent", "configure_file", "install"],
    //     "metadata": {
    //       "author": "your_name",
    //       "language": "C++17"
    //     }
    //   }
    //
    // Hint: Use json::parse(R"(...)") or the ""_json literal.
    // Docs: https://json.nlohmann.me/api/basic_json/parse/
    //
    // Placeholder:
    auto data = json::parse(R"({
        "project": "cmake_practice",
        "version": [1, 0, 0],
        "features": ["FetchContent", "configure_file", "install"],
        "metadata": {
            "author": "your_name",
            "language": "C++17"
        }
    })");

    // ─── Part B: Extract values and format with fmt ─────────────────────────
    //
    // TODO(human): Extract fields from the parsed JSON and format them using fmt.
    //
    // 1. Get "project" as a std::string
    // 2. Get "version" as a std::vector<int>
    // 3. Get "metadata"."language" as a std::string
    // 4. Get "features" as a std::vector<std::string>
    //
    // Then print using fmt::format / fmt::print:
    //   "Project: cmake_practice"
    //   "Version: 1.0.0"
    //   "Language: C++17"
    //   "Features: [FetchContent, configure_file, install]"
    //
    // Hint for version: fmt::format("{}.{}.{}", v[0], v[1], v[2])
    // Hint for features: fmt::join(features, ", ")
    //
    // Docs:
    //   - https://json.nlohmann.me/api/basic_json/get/
    //   - https://fmt.dev/latest/api/
    //
    // Placeholder:
    std::string project = data["project"].get<std::string>();
    auto version = data["version"].get<std::vector<int>>();
    std::string language = data["metadata"]["language"].get<std::string>();
    auto features = data["features"].get<std::vector<std::string>>();

    fmt::print("Project: {}\n", project);
    fmt::print("Version: {}.{}.{}\n", version[0], version[1], version[2]);
    fmt::print("Language: {}\n", language);
    fmt::print("Features: [{}]\n", fmt::join(features, ", "));

    // ─── Part C: Modify and serialize back to string ────────────────────────
    //
    // TODO(human): Add a new field to the JSON and print the result.
    //
    // 1. Add "build_system": "CMake" to the root object
    // 2. Add your name to metadata.author
    // 3. Serialize back to a pretty-printed JSON string with 2-space indent
    //
    // Hint: data["key"] = value;  and  data.dump(2)
    //
    // Placeholder:
    data["build_system"] = "CMake";
    data["metadata"]["author"] = "student";

    fmt::print("\nModified JSON:\n{}\n", data.dump(2));

    return 0;
}
