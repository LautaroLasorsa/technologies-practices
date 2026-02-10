#include "serialization.hpp"

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/serialization/vector.hpp>

#include <fstream>
#include <iostream>

void save_results(const DijkstraResult &result, const std::string &filename) {
  // ========================================================================
  // TODO(human): Serialize `result` to a text archive
  // ========================================================================
  //
  // Steps:
  //   1. Open an std::ofstream to `filename`
  //   2. Create a boost::archive::text_oarchive from the ofstream
  //   3. Use operator<< to write `result` into the archive:
  //        archive << result;
  //
  // That's it. Boost.Serialization will call DijkstraResult::serialize()
  // (defined in types.hpp) which serializes source_vertex, distances,
  // and predecessors. The std::vector<int> fields are handled automatically
  // because we included <boost/serialization/vector.hpp>.
  //
  // The text archive produces a human-readable file -- open it afterward
  // to see the format (useful for debugging).
  //
  // Docs:
  // https://www.boost.org/doc/libs/release/libs/serialization/doc/tutorial.html
  // ========================================================================

  std::ofstream of(filename);
  boost::archive::text_oarchive tof(of);
  tof << result;
}

DijkstraResult load_results(const std::string &filename) {
  DijkstraResult result;

  // ========================================================================
  // TODO(human): Deserialize `result` from a text archive
  // ========================================================================
  //
  // Mirror of save_results:
  //   1. Open an std::ifstream from `filename`
  //   2. Create a boost::archive::text_iarchive from the ifstream
  //   3. Use operator>> to read into `result`:
  //        archive >> result;
  //
  // Key insight: The same serialize() method handles both directions.
  // When called with a text_oarchive, operator& writes data.
  // When called with a text_iarchive, operator& reads data.
  // This is why serialize() uses `ar &` instead of `ar <<` or `ar >>`.
  // ========================================================================
  std::ifstream in(filename);
  boost::archive::text_iarchive fin(in);
  fin >> result;
  return result;
}

// Verification helper (provided)
bool verify_roundtrip(const DijkstraResult &original,
                      const DijkstraResult &loaded) {
  bool ok = true;

  if (original.source_vertex != loaded.source_vertex) {
    std::cerr << "  MISMATCH: source_vertex " << original.source_vertex
              << " != " << loaded.source_vertex << "\n";
    ok = false;
  }

  if (original.distances != loaded.distances) {
    std::cerr << "  MISMATCH: distances differ\n";
    ok = false;
  }

  if (original.predecessors != loaded.predecessors) {
    std::cerr << "  MISMATCH: predecessors differ\n";
    ok = false;
  }

  if (ok) {
    std::cout << "  Round-trip verification: PASSED\n";
  } else {
    std::cout << "  Round-trip verification: FAILED\n";
  }

  return ok;
}
