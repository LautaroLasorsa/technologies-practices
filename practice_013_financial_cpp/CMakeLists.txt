cmake_minimum_required(VERSION 3.20)
project(financial_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# --- Dependencies via vcpkg ---
# Install with: vcpkg install quickfix:x64-windows quantlib:x64-windows
# Build with:  cmake -B build -S . -DCMAKE_TOOLCHAIN_FILE=[vcpkg-root]/scripts/buildsystems/vcpkg.cmake

find_package(quickfix CONFIG REQUIRED)
find_package(QuantLib CONFIG REQUIRED)

# --- QuickFIX Acceptor (mock exchange) ---
add_executable(fix_acceptor
    src/fix_application.cpp
    src/fix_acceptor_main.cpp
)
target_link_libraries(fix_acceptor PRIVATE quickfix)
target_include_directories(fix_acceptor PRIVATE ${CMAKE_SOURCE_DIR}/src)

# --- QuickFIX Initiator (trading client) ---
add_executable(fix_initiator
    src/fix_client.cpp
)
target_link_libraries(fix_initiator PRIVATE quickfix)
target_include_directories(fix_initiator PRIVATE ${CMAKE_SOURCE_DIR}/src)

# --- QuantLib Option Pricing ---
add_executable(option_pricing
    src/option_pricing.cpp
)
target_link_libraries(option_pricing PRIVATE QuantLib::QuantLib)

# --- QuantLib Bond Pricing ---
add_executable(bond_pricing
    src/bond_pricing.cpp
)
target_link_libraries(bond_pricing PRIVATE QuantLib::QuantLib)
