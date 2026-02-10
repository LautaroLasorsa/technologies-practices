# Practice 021a: Solana Fundamentals -- Anchor & Accounts Model

## Technologies

- **Solana** -- High-throughput L1 blockchain with parallel transaction execution and sub-second finality
- **Anchor** -- Solana's dominant smart contract framework: IDL generation, account validation, (de)serialization, and testing scaffolding
- **Rust** -- Programs (smart contracts) are compiled to BPF/SBF bytecode from Rust
- **LiteSVM** -- In-process Solana VM for fast local testing (replaces deprecated bankrun)
- **anchor-litesvm** -- Bridge library connecting Anchor's TypeScript client with LiteSVM
- **TypeScript** -- Client-side test scripts using @coral-xyz/anchor and litesvm packages

## Stack

- Rust 1.89+ (inside Docker container)
- Solana CLI v2.1+ (Agave validator)
- Anchor CLI v0.32+ (via AVM)
- Node.js / TypeScript (for tests)
- Docker (dev container -- replaces WSL requirement)

## Description

Build a series of on-chain programs that progressively teach Solana's **account model** -- the single most important concept that distinguishes Solana from Ethereum and other blockchains. You start with a trivial counter, then learn PDAs (deterministic addressing), CPI (inter-program calls), and events. Everything runs locally -- no devnet, no real SOL.

### What you'll learn

1. **Solana architecture** -- Programs are stateless code; all state lives in separate Accounts (owner, lamports, data, rent)
2. **Anchor macros** -- `#[program]`, `#[derive(Accounts)]`, `#[account]`, and constraint attributes (`init`, `mut`, `has_one`, `seeds`)
3. **Account size & rent** -- Calculating `space`, the 8-byte discriminator, rent exemption
4. **PDAs (Program Derived Addresses)** -- Deterministic addresses derived from seeds, no private key, programs can "sign" via seeds
5. **CPI (Cross-Program Invocation)** -- Programs calling other programs, PDA signers, System Program transfers
6. **Events & errors** -- `emit!()`, `#[error_code]`, structured on-chain logging
7. **Testing with LiteSVM** -- Fast in-process tests using anchor-litesvm, transaction construction, error assertions

### Solana vs Ethereum mental model

| Concept | Ethereum | Solana |
|---------|----------|--------|
| Smart contract state | Stored inside the contract | Stored in separate **Accounts** owned by the program |
| Contract address | Deployed code has an address | Programs and accounts have **separate** addresses |
| Function calls | Tx calls contract method | Tx sends **instruction** to program, passing accounts as args |
| State access | Contract reads its own storage | Program receives accounts explicitly -- **no global state** |
| Deterministic addresses | CREATE2 | **PDAs** -- derived from seeds, no private key |
| Inter-contract calls | External calls | **CPI** -- program invokes another program's instruction |

### Solana for Rust developers

You already know Rust. Key differences in Solana/Anchor:
- `#[account]` is like a `#[derive(Serialize, Deserialize)]` struct that auto-(de)serializes from on-chain bytes via Borsh
- `#[derive(Accounts)]` is like a function signature declaring which accounts (and their constraints) an instruction needs
- PDAs are like a `HashMap<(seed1, seed2, ...), Account>` where the key is deterministic
- CPI is like calling another library's function, but the callee validates account ownership
- There's no `std` -- programs run in a BPF sandbox with limited compute budget (200k CU default)

## Instructions

### Phase 1: Environment Setup & First Program (~20 min)

1. Follow `setup.md` to build the Docker container (`dev.bat up`) and verify tools (`dev.bat versions`)
2. Run `dev.bat init solana_practice` then `dev.bat shell` and `cd solana_practice`
3. Explore the generated structure: `programs/`, `tests/`, `Anchor.toml`, `migrations/`
4. Run `anchor build` -- observe the compiled `.so` in `target/deploy/`
5. Run `anchor test` -- observe the default test passing
6. **User modifies:** Change the `initialize` instruction to accept a `greeting: String` parameter and store it in a new `GreetingAccount`
7. Key insight: Programs are deployed code (stateless). The greeting lives in a separate Account that the program owns.

### Phase 2: Accounts & Data Storage (~25 min)

1. **Concepts:** Account model (owner, lamports, data), account size calculation, Borsh serialization, Anchor constraints
2. Copy relevant code from `phase2_counter/lib.rs` into your program
3. **User implements:** `initialize` -- create a Counter account with `count: u64` and `authority: Pubkey`
4. **User implements:** `increment` -- read Counter, increment count, write back
5. **User implements:** `decrement` -- same as increment but subtract, using `require!(counter.count > 0, ...)`
6. **User implements:** Access control -- only `authority` can modify (use `has_one = authority` constraint)
7. Key insight: `space = 8 + 8 + 32` -- the 8 is Anchor's discriminator (unique per account type), then 8 bytes for u64, 32 for Pubkey

### Phase 3: PDAs (Program Derived Addresses) (~25 min)

1. **Concepts:** What PDAs are, seed derivation, bump seeds, why PDAs can't be regular keypairs
2. Copy relevant code from `phase3_pda/lib.rs` into your program
3. **User implements:** PDA-based counter using seeds `[b"counter", user.key().as_ref()]` -- each user gets their own counter
4. **User implements:** Client-side PDA derivation using `PublicKey.findProgramAddressSync()`
5. **User implements:** A "user profile" PDA -- stores name + bio, one per user
6. **User implements:** `close` instruction that reclaims rent (lamports) to the user
7. Key insight: PDAs are like a deterministic HashMap -- you can compute the address without storing it. Analogy: `HashMap<(b"counter", user_pubkey), CounterAccount>`

### Phase 4: Testing with LiteSVM (~20 min)

1. **Concepts:** LiteSVM as in-process VM, anchor-litesvm provider, transaction lifecycle
2. Install test deps: `npm install --save-dev litesvm anchor-litesvm`
3. Copy test templates from `phase4_tests/counter.test.ts`
4. **User implements:** Test that creates a counter, increments 3 times, verifies count == 3
5. **User implements:** Test error path -- decrement below zero should fail with custom error
6. **User implements:** Test unauthorized access -- wrong signer should fail
7. **User implements:** Test PDA derivation -- verify address matches program expectation
8. Key insight: LiteSVM runs Solana in-process -- no validator needed, tests are 25x faster than solana-test-validator

### Phase 5: Cross-Program Invocations (CPI) (~20 min)

1. **Concepts:** CPI context, signed invocations, PDA as signer, System Program
2. Copy relevant code from `phase5_cpi/lib.rs` into a new program (or extend existing)
3. **User implements:** `deposit` -- user transfers SOL to a PDA vault via CPI to System Program
4. **User implements:** `withdraw` -- vault PDA "signs" the CPI using its seeds to send SOL back
5. **User implements:** Authority check -- only the original depositor can withdraw
6. Key insight: PDAs can "sign" CPIs -- the runtime verifies the seeds match. This is how programs control funds without private keys.

### Phase 6: Events & Error Handling (~15 min)

1. **Concepts:** Anchor events (`emit!`), custom error codes, program logs
2. Copy relevant code from `phase6_events/lib.rs`
3. **User implements:** Custom error enum with meaningful messages (`CounterOverflow`, `Unauthorized`, `InsufficientFunds`)
4. **User implements:** Emit events on state changes -- `CounterChanged { old_count, new_count, user }`
5. **User implements:** Test event emission in LiteSVM tests
6. Key insight: Events are how off-chain systems (frontends, indexers, analytics) react to on-chain state changes -- like Ethereum's event logs

## Motivation

- **Market demand**: Solana developer roles are among the highest-paying in blockchain ($150k-$300k+), and the ecosystem is growing rapidly (DePIN, DeFi, payments)
- **Rust leverage**: You already know Rust well -- Solana/Anchor lets you apply that skill in a new, high-value domain
- **Account model novelty**: Solana's account model is fundamentally different from Ethereum's contract storage -- understanding it is the key hurdle, and this practice attacks it directly
- **Complements existing skills**: Your experience with systems programming (C++, Rust) and async patterns (Python/FastAPI) maps naturally to Solana's parallel execution model
- **Local-first development**: Everything runs locally via LiteSVM and solana-test-validator -- no wallet setup, no tokens, no devnet friction

## References

- [Anchor Framework Documentation](https://www.anchor-lang.com/docs)
- [Anchor 0.32.0 Release Notes](https://www.anchor-lang.com/docs/updates/release-notes/0-32-0)
- [Solana Developer Documentation](https://solana.com/docs)
- [Solana Account Model](https://solana.com/docs/core/accounts)
- [Solana Programs](https://solana.com/docs/core/programs)
- [Solana PDAs](https://solana.com/docs/core/pda)
- [Solana CPI](https://solana.com/docs/core/cpi)
- [LiteSVM GitHub](https://github.com/LiteSVM/litesvm)
- [anchor-litesvm GitHub](https://github.com/brimigs/anchor-litesvm)
- [RareSkills -- Anchor Account Types](https://rareskills.io/post/anchor-account-types)
- [RareSkills -- Initializing Accounts](https://rareskills.io/post/solana-initialize-account)
- [QuickNode -- LiteSVM Testing Guide](https://www.quicknode.com/guides/solana-development/tooling/litesvm)
- [Helius -- Beginner's Guide to Anchor](https://www.helius.dev/blog/an-introduction-to-anchor-a-beginners-guide-to-building-solana-programs)

## Commands

### Docker Container Setup

| Command | Description |
|---------|-------------|
| `dev.bat up` | Build Docker image and start the dev container in background |
| `dev.bat down` | Stop and remove the container |
| `dev.bat versions` | Print installed tool versions (rustc, solana, anchor, node) |
| `dev.bat logs` | Follow container logs |

### Development Workflow (from Windows)

| Command | Description |
|---------|-------------|
| `dev.bat shell` | Open a bash shell inside the container |
| `dev.bat init solana_practice` | Initialize a new Anchor project named `solana_practice` inside the container |
| `dev.bat build` | Run `anchor build` inside the container (compiles program to `.so`) |
| `dev.bat test` | Run `anchor test` inside the container (validator + deploy + tests) |
| `dev.bat validator` | Start `solana-test-validator` inside the container |

### Inside the Container (via `dev.bat shell`)

| Command | Description |
|---------|-------------|
| `anchor init solana_practice` | Initialize a new Anchor project |
| `anchor build` | Compile the Solana program to `target/deploy/*.so` |
| `anchor test` | Build, deploy to local validator, and run TypeScript tests |
| `anchor test --skip-build` | Run tests without rebuilding the program |
| `solana-test-validator` | Start a local Solana validator (for manual testing) |
| `npm install --save-dev litesvm anchor-litesvm` | Install LiteSVM test dependencies (Phase 4) |
| `solana-keygen new --no-bip39-passphrase --force` | Generate a new local keypair |
| `solana config set --url localhost` | Set Solana CLI to use local validator |

## State

`not-started`
