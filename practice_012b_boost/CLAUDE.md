# Practice 012b: Boost Deep-Dive

## Technologies

- **Boost.Asio** -- Asynchronous I/O, timers, and TCP networking
- **Boost.Graph (BGL)** -- Generic graph data structures and algorithms (BFS, DFS, Dijkstra)
- **Boost.Program_options** -- Type-safe command-line argument parsing
- **Boost.Serialization** -- Object serialization to text/binary archives
- **CMake** -- Build system with `find_package(Boost)`
- **vcpkg** -- C++ package manager for Boost installation

## Stack

- C++17
- Boost >= 1.74
- CMake >= 3.16

## Description

Build a **graph analysis CLI tool** that ties four core Boost libraries into a single cohesive project. The tool reads a graph from a file (or generates a sample one), runs classic algorithms on it (BFS, DFS, Dijkstra), serializes results to disk, and exposes an async TCP server that answers shortest-path queries. Command-line arguments are parsed with Boost.Program_options.

This practice focuses on Boost libraries that remain **essential in production C++** and are **not superseded by C++17 STL**: Asio (no STL equivalent for async I/O), BGL (no STL graph library), Program_options (no STL CLI parser), and Serialization (no STL serialization).

### What you'll learn

1. **Boost.Program_options** -- Defining named/positional options, validation, help generation
2. **Boost.Graph (BGL)** -- `adjacency_list`, property maps, BFS/DFS visitors, Dijkstra's algorithm
3. **Boost.Serialization** -- Intrusive and non-intrusive serialization, text archives, versioning
4. **Boost.Asio** -- `io_context`, `steady_timer`, async TCP acceptor/session, completion handlers
5. **CMake + Boost** -- Finding and linking Boost components in a modern CMake project

## Instructions

### Phase 0: Setup (~10 min)

1. Install Boost via vcpkg: `vcpkg install boost` (or `vcpkg install boost-asio boost-graph boost-program-options boost-serialization`)
2. Configure CMake with vcpkg toolchain: `cmake -B build -DCMAKE_TOOLCHAIN_FILE=[vcpkg-root]/scripts/buildsystems/vcpkg.cmake`
3. Build: `cmake --build build`
4. Run: `./build/Debug/boost_practice --help`

### Phase 1: CLI Parsing with Boost.Program_options (~15 min)

1. Understand `options_description`, `variables_map`, `positional_options_description`
2. **User implements:** Define CLI options (--vertices, --edges, --source, --mode, --port)
3. **User implements:** Parse and validate arguments, print help on --help
4. Key question: How does `notify(vm)` differ from just checking `vm.count()`?

### Phase 2: Graph Construction & Algorithms with BGL (~25 min)

1. Understand `adjacency_list` template parameters: OutEdgeList, VertexList, Directed, properties
2. **User implements:** Build a weighted directed graph from generated edges
3. **User implements:** BFS visitor that records discovery order
4. **User implements:** Dijkstra's shortest paths from a source vertex
5. Key question: Why does BGL use external property maps instead of storing data in vertices directly?

### Phase 3: Serialization with Boost.Serialization (~15 min)

1. Understand archives (`text_oarchive`, `text_iarchive`) and the `serialize()` method
2. **User implements:** Serialize Dijkstra results (distances + predecessors) to a text file
3. **User implements:** Deserialize and verify round-trip correctness
4. Key question: What does the `version` parameter in `serialize()` enable?

### Phase 4: Async TCP Server with Boost.Asio (~25 min)

1. Understand `io_context`, `tcp::acceptor`, `tcp::socket`, async operation chains
2. **User implements:** Async accept loop that spawns sessions
3. **User implements:** Session that reads a source vertex, runs Dijkstra, and writes back distances
4. **User implements:** Graceful shutdown with a steady_timer timeout
5. Key question: Why must the session be kept alive with `shared_from_this()`?

### Phase 5: Integration & Testing (~10 min)

1. Run the full pipeline: generate graph, compute paths, serialize, start server
2. Test with `netcat` or a simple TCP client: send vertex ID, receive distances
3. Discussion: How would you add SSL/TLS using Boost.Asio's ssl::stream?

## Motivation

- **Boost.Asio** is the foundation of C++ async networking (used in Beast, gRPC C++ core, many trading systems). Understanding its completion handler model is essential for high-performance C++.
- **Boost.Graph** provides battle-tested generic graph algorithms. For someone with a competitive programming background, BGL connects algorithmic knowledge to production-grade C++ APIs.
- **Boost.Program_options** is the de-facto standard for C++ CLI parsing in large codebases (CMake itself uses it internally).
- **Boost.Serialization** teaches intrusive/non-intrusive serialization patterns that appear across C++ systems (game engines, scientific computing, financial systems).
- **Market demand**: Boost proficiency is a hard requirement for C++ roles in finance, networking, embedded systems, and game development.

## References

- [Boost.Asio Tutorial](https://www.boost.org/doc/libs/master/doc/html/boost_asio/tutorial.html)
- [Boost.Graph Quick Tour](https://www.boost.org/doc/libs/release/libs/graph/doc/quick_tour.html)
- [Boost.Program_options Tutorial](https://www.boost.org/doc/libs/release/doc/html/program_options/tutorial.html)
- [Boost.Serialization Tutorial](https://www.boost.org/doc/libs/release/libs/serialization/doc/tutorial.html)
- [The Boost C++ Libraries (Online Book)](https://theboostcpplibraries.com/)
- [vcpkg Boost Package](https://vcpkg.io/en/package/boost)

## Commands

All commands are run from the `practice_012b_boost/` folder root. The cmake binary on this machine is at `C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe`. Boost is installed via vcpkg at `literal:%VCPKG_ROOT%`.

### Configure

| Command | Description |
|---------|-------------|
| `powershell.exe -Command "& 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe' -S . -B build -DCMAKE_TOOLCHAIN_FILE='literal:%VCPKG_ROOT%\scripts\buildsystems\vcpkg.cmake' 2>&1"` | Configure the project with vcpkg toolchain (finds Boost automatically) |

### Build

| Command | Description |
|---------|-------------|
| `powershell.exe -Command "& 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe' --build build --config Release 2>&1"` | Build the boost_practice executable (Release) |
| `powershell.exe -Command "& 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe' --build build --config Debug 2>&1"` | Build the boost_practice executable (Debug) |

### Run

| Command | Description |
|---------|-------------|
| `build\Release\boost_practice.exe --help` | Show all CLI options and usage |
| `build\Release\boost_practice.exe` | Run all phases with defaults (10 vertices, 20 edges, source 0, port 9090) |
| `build\Release\boost_practice.exe --mode graph` | Run graph phase only (BFS, Dijkstra, serialization) -- no TCP server |
| `build\Release\boost_practice.exe --mode server` | Run TCP server phase only (listens on port 9090, 30s timeout) |
| `build\Release\boost_practice.exe --mode graph --vertices 20 --edges 40 --source 5` | Run graph phase with custom graph size and source vertex |
| `build\Release\boost_practice.exe --mode graph --output results.txt` | Run graph phase and serialize Dijkstra results to a custom output file |
| `build\Release\boost_practice.exe --mode server --port 8080` | Run TCP server on a custom port |
| `build\Release\boost_practice.exe --vertices 15 --edges 30 --source 3 --port 8080 --output results.txt` | Run all phases with fully custom configuration |

## State

`completed`
