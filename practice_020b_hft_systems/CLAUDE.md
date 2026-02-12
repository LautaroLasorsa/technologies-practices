# Practice 020b: HFT Systems -- Order Book, Matching Engine & Feed Handler

## Technologies

- **C++17** -- std::variant, std::optional, structured bindings, constexpr, enum class
- **abseil-cpp** -- absl::flat_hash_map (O(1) order lookups), absl::StrFormat (output), absl::StatusOr (error handling)
- **CMake 3.16+** -- Build system with FetchContent for abseil-cpp
- **HFT Domain** -- Limit order books, matching engines, market data feeds, order management, signal generation

## Stack

- C++17
- abseil-cpp (fetched via CMake FetchContent)

## Theoretical Context

### What Limit Order Books and Matching Engines Are

A limit order book (LOB) is the foundational data structure of every financial exchange. It maintains all resting (unmatched) orders organized by price and time, implementing the **price-time priority rule**: at each price level, orders execute in first-in-first-out (FIFO) order. The matching engine is the stateful process that accepts incoming orders and either matches them against the book (generating trades) or adds them to the book as resting orders.

This seemingly simple data structure is the engine of global finance. Nasdaq processes ~1 billion order book messages per day. The CME's Glob ex handles 100M+ orders/day across futures and options. Every stock, futures, options, and crypto exchange runs a matching engine at its core.

The cancel-to-fill ratio on modern exchanges is ~30:1—for every trade executed, roughly 30 orders are placed and then cancelled. This is driven by high-frequency market makers who continuously update quotes as prices move. The implication: **O(1) cancel performance is non-negotiable**. A hash map from order ID to location is the standard solution.

### How Order Books Work Internally

An order book maintains two sides: **bids** (buy orders) and **asks** (sell orders). Each side is a sorted collection of price levels. Each price level contains a FIFO queue of orders at that price.

**Data structure design:**
- `std::map<price_t, PriceLevel>` for each side (bids and asks)
  - Map gives O(log N) insert/erase and O(1) access to best price (begin/rbegin)
  - N = number of distinct active price levels (~20-100 in practice, so log N ~ 5-7)
- `std::deque<Order>` at each price level for FIFO time priority
  - In production, intrusive doubly-linked lists enable O(1) removal; deque is O(N) but simpler
- `absl::flat_hash_map<OrderId, OrderLocation>` for O(1) cancel lookup
  - Without this, cancel requires scanning every order on one side—unacceptable at scale

**Matching process:**
1. Incoming **aggressive order** (market or marketable limit) walks the opposite side's price levels
2. Starting from best price (highest bid for sells, lowest ask for buys), match against resting orders
3. Generate **Trade** messages for each fill (maker + taker)
4. Update order remaining quantities; remove fully-filled orders
5. If incoming order not fully filled and is a limit order, rest it in the book at its limit price

**Market data generation:**
- **L1 (top-of-book)**: best bid and ask prices + quantities
- **L2 (depth)**: top N price levels on each side with aggregate quantity
- **L3 (full book)**: every individual order (exchange-internal only; not broadcast publicly)

Nasdaq ITCH, CME MDP 3.0, and other market data protocols are essentially serialized order book updates: Add, Modify, Cancel, Trade messages.

### Key Concepts

| Concept | Description |
|---------|-------------|
| **Price-time priority** | Orders match first by price (best price first), then by arrival time (FIFO) within each price level. |
| **Maker vs taker** | Maker = resting order providing liquidity; taker = aggressive order removing liquidity. Makers often pay lower fees. |
| **Aggressive order** | Market order or marketable limit (crosses the spread). Always matches immediately (or IOC/FOK). |
| **Passive order** | Limit order that does not cross the spread. Rests in the book until matched or cancelled. |
| **IOC (Immediate-Or-Cancel)** | Order type: execute as much as possible immediately, cancel the rest. Never rests in the book. |
| **FOK (Fill-Or-Kill)** | Order type: execute entire quantity immediately or cancel the whole order. All-or-nothing semantics. |
| **Spread** | Difference between best ask and best bid. Tighter spreads indicate more liquid markets. |
| **Book imbalance** | Ratio of bid liquidity to ask liquidity at top of book. Predictive of short-term price movement. |
| **Feed handler** | Component that consumes market data protocol messages (ITCH, MDP, FIX FAST) and reconstructs the order book locally. |
| **Sequence number** | Monotonic counter in market data feed. Gap = missed message, triggers snapshot recovery. |

### Ecosystem Context and Trade-offs

Real exchange matching engines optimize for:
1. **Deterministic latency** (P99.9 < 100 microseconds from order receipt to ack)
2. **Throughput** (millions of messages/second peak)
3. **Fairness** (FIFO order processing, no priority inversions)

**Design trade-offs:**
- `std::map` vs `absl::flat_hash_map` for price levels: map wins because sorted iteration (best bid/ask) is frequent
- `std::deque` vs intrusive list at each level: intrusive list enables O(1) removal but requires custom memory management
- Centralized matching (one thread) vs distributed (sharded by symbol): centralized is simpler and ensures strict FIFO; distributed scales but complicates cross-symbol orders
- Synchronous order-by-order matching vs batched auctions: synchronous is standard for continuous markets; batching (IEX, some crypto exchanges) reduces latency arbitrage but sacrifices continuous price discovery

**Alternatives:**
- **Pro-rata matching** (CME, some futures): fills split proportionally by resting quantity, not FIFO. Rewards large orders.
- **Hybrid matching** (some options exchanges): FIFO for public orders, pro-rata for market makers
- **Frequent batch auctions** (IEX, EBS): batch orders over tiny intervals (350 microseconds), match at single clearing price

Most major equity exchanges (NYSE, Nasdaq, LSE, Euronext) and crypto exchanges (Coinbase, Binance, Kraken) use price-time priority with continuous matching—the design implemented in this practice.

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
