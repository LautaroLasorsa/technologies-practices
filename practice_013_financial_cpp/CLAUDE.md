# Practice 013: Financial C++ -- QuickFIX & QuantLib

## Technologies

- **QuickFIX** -- Open-source C++ FIX (Financial Information eXchange) protocol engine for electronic trading
- **QuantLib** -- Comprehensive C++ library for quantitative finance (option/bond pricing, term structures, Greeks)
- **FIX 4.4** -- Industry-standard protocol for real-time electronic trading communication
- **vcpkg** -- C++ package manager for installing both libraries

## Stack

- C++17
- CMake 3.20+
- vcpkg (dependency management)
- Windows (MSVC)

## Description

Build two focused mini-projects that exercise the two pillars of financial C++ infrastructure:

1. **FIX Order Flow (QuickFIX)** -- An acceptor (mock exchange) and initiator (trading client) communicate via the FIX 4.4 protocol. The client sends a `NewOrderSingle`, the exchange validates it and replies with an `ExecutionReport`. This teaches FIX session management, message construction, field extraction, and the Application/MessageCracker pattern.

2. **Quantitative Pricing (QuantLib)** -- Price a European call option using the Black-Scholes model and a fixed-rate bond using a flat yield curve. Compute Greeks (delta, gamma, theta, vega) and bond analytics (clean/dirty price, yield). This teaches QuantLib's instrument-engine architecture, term structures, and date handling.

### What you'll learn

1. **FIX protocol fundamentals** -- Sessions, messages, tags, BeginString/SenderCompID/TargetCompID
2. **QuickFIX Application pattern** -- Inheriting `FIX::Application` + `FIX::MessageCracker`, lifecycle callbacks
3. **FIX message construction** -- Building `NewOrderSingle` and `ExecutionReport` with typed field objects
4. **QuantLib instrument-engine architecture** -- Instruments hold market data, engines compute prices
5. **Black-Scholes pricing** -- `BlackScholesProcess` + `AnalyticEuropeanEngine` for European options
6. **Bond analytics** -- `FixedRateBond` + `DiscountingBondEngine` for clean/dirty price and yield
7. **Greeks computation** -- Delta, gamma, theta, vega as sensitivities of option price to market parameters

## Instructions

### Phase 0: Setup (~10 min)

1. Install QuickFIX and QuantLib via vcpkg:
   ```
   vcpkg install quickfix:x64-windows quantlib:x64-windows
   ```
2. Configure and build the CMake project:
   ```
   cmake -B build -S . -DCMAKE_TOOLCHAIN_FILE=[vcpkg-root]/scripts/buildsystems/vcpkg.cmake
   cmake --build build --config Release
   ```
3. Verify both libraries link correctly by building the project.

### Phase 1: QuickFIX Acceptor -- Mock Exchange (~20 min)

1. Understand the FIX Application interface: `onCreate`, `onLogon`, `onLogout`, `toAdmin`, `fromAdmin`, `toApp`, `fromApp`
2. **User implements:** Session lifecycle callbacks (`onCreate`, `onLogon`, `onLogout`) with logging
3. **User implements:** `onMessage(const FIX44::NewOrderSingle&, ...)` -- extract Symbol, Side, OrderQty, Price, ClOrdID
4. **User implements:** Build and send an `ExecutionReport` back with order acknowledgment fields
5. Key question: Why does QuickFIX use `MessageCracker` instead of a single `fromApp` handler?

### Phase 2: QuickFIX Initiator -- Trading Client (~15 min)

1. **User implements:** Build a `FIX44::NewOrderSingle` message with required fields (ClOrdID, Side, TransactTime, OrdType, Symbol, OrderQty, Price)
2. **User implements:** Send the message via `FIX::Session::sendToTarget`
3. Run both acceptor and initiator -- observe the FIX message flow in the logs
4. Key question: What is the difference between `HandlInst` values 1, 2, and 3?

### Phase 3: QuantLib Option Pricing (~20 min)

1. Understand QuantLib's architecture: Quote -> TermStructure -> Process -> Engine -> Instrument
2. **User implements:** Create a `BlackScholesProcess` from market data (spot, risk-free rate, volatility)
3. **User implements:** Create a `VanillaOption` with `PlainVanillaPayoff` and `EuropeanExercise`
4. **User implements:** Set `AnalyticEuropeanEngine` and compute NPV
5. **User implements:** Extract Greeks: `delta()`, `gamma()`, `theta()`, `vega()`, `rho()`
6. Key question: Why does QuantLib separate the instrument from its pricing engine?

### Phase 4: QuantLib Bond Pricing (~15 min)

1. **User implements:** Build a flat yield curve with `FlatForward`
2. **User implements:** Create a `FixedRateBond` with coupon schedule
3. **User implements:** Set `DiscountingBondEngine` and compute `cleanPrice()`, `dirtyPrice()`, `yield()`
4. Key question: What is the difference between clean price and dirty price?

### Phase 5: Integration & Exploration (~10 min)

1. Experiment: Change volatility and observe how Greeks change
2. Experiment: Change the yield curve and observe bond price sensitivity
3. Discussion: How would you connect QuickFIX order flow to QuantLib pricing in a real system?

## Motivation

- **Industry standard**: FIX is the dominant protocol in electronic trading; QuickFIX is the reference open-source implementation used in production at trading firms
- **Quantitative finance**: QuantLib is the gold standard for derivatives pricing in C++ -- used by banks, hedge funds, and fintech companies
- **C++ in finance**: High-frequency trading, risk engines, and pricing libraries are overwhelmingly C++; this practice bridges the gap between general C++ knowledge and domain-specific financial programming
- **Career relevance**: Financial C++ (FIX, pricing, Greeks) is a high-demand, high-compensation skill set in capital markets technology
- **Complementary to current stack**: Extends C++ proficiency beyond competitive programming into production financial systems

## References

- [QuickFIX Official Site](https://quickfixengine.org/)
- [QuickFIX GitHub Repository](https://github.com/quickfix/quickfix)
- [QuickFIX Configuration Guide](https://quickfixengine.org/c/documentation/getting-started/configuration.html)
- [QuickFIX Executor Example](https://github.com/quickfix/quickfix/blob/master/examples/executor/C++/Application.cpp)
- [QuantLib Official Site](https://www.quantlib.org/)
- [QuantLib GitHub Repository](https://github.com/lballabio/QuantLib)
- [QuantLib Bond Example](https://github.com/lballabio/QuantLib/blob/master/Examples/Bonds/Bonds.cpp)
- [Implementing QuantLib: Black-Scholes](https://www.implementingquantlib.com/2023/11/black-scholes.html)
- [FIX Protocol Specification](https://www.fixtrading.org/standards/)

## Commands

### Phase 0: Dependencies (vcpkg)

| Command | Description |
|---------|-------------|
| `vcpkg install quickfix:x64-windows quantlib:x64-windows` | Install QuickFIX and QuantLib via vcpkg |

### Phase 0: CMake Configure & Build

| Command | Description |
|---------|-------------|
| `powershell.exe -Command "& 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe' -B build -S . -DCMAKE_TOOLCHAIN_FILE=%VCPKG_ROOT%/scripts/buildsystems/vcpkg.cmake 2>&1"` | Configure CMake project with vcpkg toolchain |
| `powershell.exe -Command "& 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe' --build build --config Release 2>&1"` | Build all targets (Release) |
| `powershell.exe -Command "& 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe' --build build --config Release --target fix_acceptor 2>&1"` | Build only the FIX acceptor (mock exchange) |
| `powershell.exe -Command "& 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe' --build build --config Release --target fix_initiator 2>&1"` | Build only the FIX initiator (trading client) |
| `powershell.exe -Command "& 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe' --build build --config Release --target option_pricing 2>&1"` | Build only the option pricing executable |
| `powershell.exe -Command "& 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe' --build build --config Release --target bond_pricing 2>&1"` | Build only the bond pricing executable |
| `build.bat` | Configure + build all targets (Release) via convenience script |
| `build.bat configure` | Only run CMake configuration |
| `build.bat build` | Only run CMake build (assumes already configured) |
| `build.bat clean` | Remove the build directory |

### FIX Data Dictionary Setup

| Command | Description |
|---------|-------------|
| `copy "%VCPKG_ROOT%\installed\x64-windows\share\quickfix\FIX44.xml" config\` | Copy FIX44.xml data dictionary to config/ directory |

### Phase 1-2: Run QuickFIX Demo (two separate terminals)

| Command | Description |
|---------|-------------|
| `build\Release\fix_acceptor.exe` | Start the FIX acceptor (mock exchange) -- run first, from project root |
| `build\Release\fix_initiator.exe` | Start the FIX initiator (trading client) -- run second, in separate terminal |

### Phase 3: Run QuantLib Option Pricing

| Command | Description |
|---------|-------------|
| `build\Release\option_pricing.exe` | Run European option pricing with Black-Scholes and compute Greeks |

### Phase 4: Run QuantLib Bond Pricing

| Command | Description |
|---------|-------------|
| `build\Release\bond_pricing.exe` | Run fixed-rate bond pricing with flat yield curve |

### LSP (clangd) Support

| Command | Description |
|---------|-------------|
| `gen_compile_commands.bat` | Generate compile_commands.json for clangd (Ninja-based build) |

## State

`not-started`
