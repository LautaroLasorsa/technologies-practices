# Practice 020b: HFT Systems -- Order Book, Matching Engine & Feed Handler

## Technologies

- **C++17** -- std::variant, std::optional, structured bindings, constexpr, enum class
- **abseil-cpp** -- absl::flat_hash_map (O(1) order lookups), absl::StrFormat (output), absl::StatusOr (error handling)
- **CMake 3.16+** -- Build system with FetchContent for abseil-cpp
- **HFT Domain** -- Limit order books, matching engines, market data feeds, order management, signal generation

## Stack

- C++17
- abseil-cpp (fetched via CMake FetchContent)

## Description

Build the **core components of a high-frequency trading system**: a limit order book, matching engine, market data feed handler, signal generator, and order management system. This practice applies the low-latency patterns from 020a (SPSC queues, memory pools, cache optimization, TSC timing) to real financial domain structures.

This is not a toy -- these are simplified but architecturally faithful versions of what runs inside Nasdaq, CME, ICE, and every crypto exchange.

### What you'll learn

1. **Limit Order Book** -- Price-time priority (FIFO), bid/ask sides, L2 snapshots, O(1) cancel via hash map
2. **Matching Engine** -- Aggressive vs passive orders, fills/partial fills, IOC/FOK order types, trade generation
3. **Feed Handler** -- Incremental updates vs snapshots, sequence number gap detection, book reconstruction
4. **Signal Generation** -- Mid-price, spread, book imbalance, simple market-making signals
5. **Order Management** -- Order lifecycle state machine, pre-trade risk checks, position/PnL tracking
6. **End-to-End Simulation** -- Wire all components into a simulated trading loop with PnL tracking

### Prerequisites

- **Practice 020a (HFT Low-Latency C++)** -- SPSC queues, memory pools, cache optimization, CRTP dispatch, TSC timing
- **Practice 012a (C++17 & abseil-cpp)** -- std::variant, absl::flat_hash_map, absl::StatusOr, structured bindings

## Instructions

### Phase 1: Limit Order Book (~25 min)

1. **Concepts:** Price levels, bid/ask sides, price-time priority (FIFO), L2 order book representation
2. **User implements:** `PriceLevel::add_order()` -- append order to FIFO queue at a price level
3. **User implements:** `OrderBook::add_order()` -- insert into correct side, maintain sorted price levels, register in lookup map
4. **User implements:** `OrderBook::cancel_order()` -- O(1) lookup via flat_hash_map, remove from price level
5. **User implements:** `OrderBook::get_l2_snapshot()` -- return top N bid/ask levels with aggregate quantities
6. Key question: Why do real exchanges use intrusive linked lists instead of std::deque for orders at a price level?

### Phase 2: Matching Engine (~25 min)

1. **Concepts:** Price-time priority matching, aggressive vs passive orders, fill/partial-fill, trade generation
2. **User implements:** `MatchingEngine::match_order()` -- walk opposite side of book, generate trades
3. **User implements:** Handling of partial fills (both incoming and resting orders)
4. **User implements:** Order types: LIMIT (rests if not filled), IOC (immediate-or-cancel), FOK (fill-or-kill)
5. Key question: What's the difference between a maker and a taker? Why do exchanges charge different fees?

### Phase 3: Market Data Feed Handler (~20 min)

1. **Concepts:** Market data protocols (ITCH, MDP3), incremental updates vs snapshots, sequence numbers, gap detection
2. **User implements:** `FeedHandler::on_message()` -- apply incremental updates to local order book
3. **User implements:** Sequence number gap detection and snapshot recovery logic
4. Key question: Why is gap detection critical? What happens if you miss a cancel message?

### Phase 4: Signal Generation (~20 min)

1. **Concepts:** Mid-price, spread, book imbalance, VWAP, trade signals
2. **User implements:** `calculate_mid_price()`, `calculate_spread()`, `book_imbalance()`
3. **User implements:** `SignalGenerator` that emits BUY/SELL/HOLD based on imbalance + spread
4. Key question: Why is book imbalance predictive of short-term price movement?

### Phase 5: Order Management System (~15 min)

1. **Concepts:** Order lifecycle (New -> Acked -> PartialFill -> Filled/Cancelled), position tracking, risk checks
2. **User implements:** `OrderManager` state machine with `send_order()`, `on_ack()`, `on_fill()`, `on_cancel()`
3. **User implements:** Pre-trade risk checks: max order size, max position, max notional value
4. **User implements:** Position tracker: net position, realized PnL, unrealized PnL (mark-to-market)
5. Key question: Why must risk checks happen synchronously before order submission, never asynchronously?

### Phase 6: End-to-End Simulation (~20 min)

1. **Concepts:** Wire everything together into a simulated trading loop
2. **User implements:** Synthetic market data generator (random walk with mean reversion)
3. **User implements:** Full loop: FeedHandler -> OrderBook -> SignalGenerator -> OMS -> MatchingEngine -> PnL
4. **User implements:** Performance stats: trades, fills, PnL, latency per tick
5. Key question: In production, which of these components run on the same thread? Which are separate?

## Motivation

- **Direct career relevance**: HFT/systematic trading firms (Citadel Securities, Jane Street, Jump Trading, Optiver, IMC, Tower Research) build exactly these components. This is the #1 most valued skill for C++ quant dev roles.
- **Exchange technology**: Nasdaq, CME, ICE, LSEG all run matching engines in C++. Understanding the architecture from the inside out is essential for exchange-side roles.
- **Builds on 020a**: Practice 020a covered generic low-latency patterns. This practice applies them to the domain where they matter most.
- **Interview preparation**: "Implement a limit order book" and "design a matching engine" are among the most common HFT interview questions. This practice builds working implementations.
- **CP crossover**: Order book operations (sorted structures, FIFO queues, hash maps) are algorithmic problems you already understand -- now applied to a $6T/day market.

## References

- [How to Build a Fast Limit Order Book (WK Selph)](https://web.archive.org/web/20110219163448/http://howtohft.wordpress.com/2011/02/15/how-to-build-a-fast-limit-order-book/) -- Classic article on LOB data structure design
- [Nasdaq ITCH 5.0 Protocol Spec](https://www.nasdaqtrader.com/content/technicalsupport/specifications/dataproducts/NQTVITCHSpecification.pdf) -- Real market data protocol
- [CME MDP 3.0 Market Data](https://www.cmegroup.com/confluence/display/EPICSANDBOX/CME+MDP+3.0+Market+Data) -- CME's market data protocol
- [SEC Market Structure Resources](https://www.sec.gov/marketstructure) -- US equity market structure overview
- [Abseil C++ Official](https://abseil.io/) -- absl::flat_hash_map, StatusOr, StrFormat docs
- [Trading and Exchanges by Larry Harris](https://global.oup.com/academic/product/trading-and-exchanges-9780195144703) -- Definitive textbook on market microstructure

## Commands

### Setup & Build

| Command | Description |
|---------|-------------|
| `powershell.exe -Command "& 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe' -S . -B build -DCMAKE_BUILD_TYPE=Release"` | Configure CMake (Release); first run fetches abseil-cpp via FetchContent |
| `powershell.exe -Command "& 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe' --build build --config Release --target all_phases"` | Build all 6 phases (Release) |
| `powershell.exe -Command "& 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe' -S . -B build -DCMAKE_BUILD_TYPE=Debug"` | Configure CMake (Debug) |
| `powershell.exe -Command "& 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe' --build build --config Debug --target all_phases"` | Build all 6 phases (Debug) |
| `build.bat` | Build all targets (Release) via helper script |
| `build.bat debug` | Build all targets (Debug) via helper script |
| `build.bat clean` | Remove the build directory |
| `gen_compile_commands.bat` | Generate `compile_commands.json` for clangd (Ninja + MSVC) |

### Build Individual Phases

| Command | Description |
|---------|-------------|
| `powershell.exe -Command "& 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe' --build build --config Release --target phase1_orderbook"` | Build Phase 1 only |
| `powershell.exe -Command "& 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe' --build build --config Release --target phase2_matching"` | Build Phase 2 only |
| `powershell.exe -Command "& 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe' --build build --config Release --target phase3_feed"` | Build Phase 3 only |
| `powershell.exe -Command "& 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe' --build build --config Release --target phase4_signal"` | Build Phase 4 only |
| `powershell.exe -Command "& 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe' --build build --config Release --target phase5_oms"` | Build Phase 5 only |
| `powershell.exe -Command "& 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe' --build build --config Release --target phase6_simulation"` | Build Phase 6 only |
| `build.bat phase1` | Build Phase 1 via helper script |
| `build.bat phase2` | Build Phase 2 via helper script |
| `build.bat phase3` | Build Phase 3 via helper script |
| `build.bat phase4` | Build Phase 4 via helper script |
| `build.bat phase5` | Build Phase 5 via helper script |
| `build.bat phase6` | Build Phase 6 via helper script |

### Run Phases

| Command | Description |
|---------|-------------|
| `build\Release\phase1_orderbook.exe` | Run Phase 1: Limit Order Book |
| `build\Release\phase2_matching.exe` | Run Phase 2: Matching Engine |
| `build\Release\phase3_feed.exe` | Run Phase 3: Market Data Feed Handler |
| `build\Release\phase4_signal.exe` | Run Phase 4: Signal Generation |
| `build\Release\phase5_oms.exe` | Run Phase 5: Order Management System |
| `build\Release\phase6_simulation.exe` | Run Phase 6: End-to-End Simulation |

## State

`not-started`
