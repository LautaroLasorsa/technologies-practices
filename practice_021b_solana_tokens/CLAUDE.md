# Practice 021b: Solana Tokens — SPL, Escrow & PDA Vaults

## Technologies

- **Anchor Framework** — Solana's dominant program framework (account validation, CPI helpers, IDL generation)
- **SPL Token Program** — The standard token program all Solana tokens use (USDC, wrapped SOL, NFTs)
- **Token-2022 (Token Extensions)** — Next-gen token program with built-in transfer fees, non-transferable tokens, confidential transfers
- **Associated Token Account (ATA) Program** — Deterministic token account derivation per (wallet, mint) pair
- **LiteSVM** — Fast in-process Solana VM for Rust-native testing (no validator needed)
- **anchor-spl** — Anchor's SPL integration crate (`token`, `token_interface`, `associated_token`, `token_2022`)

## Stack

- Rust 1.89+ (inside Docker container)
- Solana CLI, Anchor CLI, LiteSVM (all inside Docker)
- Docker (dev container -- replaces WSL requirement)

## Theoretical Context

### What SPL Token and Token-2022 Are

The **SPL Token Program** is Solana's standard token interface—a single on-chain program that manages ALL fungible and non-fungible tokens (including USDC, wrapped SOL, NFTs, stablecoins, memecoins). Unlike Ethereum where each ERC-20 is a separate deployed contract, Solana has ONE canonical program (address `TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA`) that every token uses.

This architecture has profound implications. Token creation is cheap (no contract deployment gas). Token operations (mint, transfer, burn) have uniform cost. Wallets and explorers know exactly how to parse any token—there's no "non-standard ERC-20" problem. But it also means less flexibility: custom token logic requires wrapping the SPL program in a higher-level program.

**Token-2022 (Token Extensions)** is the next-generation token program that adds built-in features previously requiring custom programs: transfer fees, non-transferable (soulbound) tokens, confidential transfers (ZK proofs), interest-bearing tokens, and more. It's backward-compatible with SPL Token but provides opt-in extensions.

### How SPL Token Works Internally

**Core concepts:**
1. **Mint Account**: Defines a token type. Stores supply, decimals, and mint/freeze authority. ONE mint = ONE token (like USDC has one mint).
2. **Token Account / ATA**: Holds a balance of a specific token for a specific wallet. Structure: `{mint, owner, amount, delegate, state}`.
3. **Associated Token Account (ATA)**: Deterministic PDA derived from `(owner, mint)`. Every wallet has exactly one ATA per token. Simplifies UX—you don't need to track multiple accounts per token.

**Key operations (CPIs to the Token Program):**
- `mint_to(mint_authority, destination, amount)`: Create new tokens. Requires mint authority signature.
- `transfer / transfer_checked`: Move tokens between accounts. `transfer_checked` validates decimals to prevent mint confusion attacks.
- `burn(owner, source, amount)`: Destroy tokens. Reduces supply.
- `approve(owner, delegate, amount)`: Delegate spending authority (like ERC-20 approve). Used by DeFi programs.

**Authority model:**
- **Mint authority**: Can mint new tokens. Often a PDA controlled by a program (program = "central bank"). Can be set to None (fixed supply).
- **Freeze authority**: Can freeze token accounts (prevent transfers). Used by regulated stablecoins and securities.
- **Close authority**: Can close empty token accounts to reclaim rent.

**Escrow pattern:** The canonical Solana DeFi primitive. Two parties want to atomically swap token A for token B:
1. Maker deposits token A into a **vault PDA** (program-controlled account).
2. Maker creates an **Escrow Account** (state PDA) recording terms: `{maker, taker, vault, mint_a, mint_b, amount_a, amount_b}`.
3. Taker sends token B to maker via CPI.
4. Program uses vault PDA's signature (via `invoke_signed`) to send token A to taker.
5. Escrow account closed, maker reclaims rent.

This pattern underpins DEX limit orders, OTC swaps, and vesting contracts.

### Key Concepts

| Concept | Description |
|---------|-------------|
| **Mint Account** | Defines a token: supply, decimals, authorities. One mint per token type (like USDC has one global mint). |
| **Token Account (ATA)** | Holds a balance of a specific token for a specific owner. PDA derived from `(owner, mint)`. |
| **Mint Authority** | Pubkey (often a PDA) that can mint new tokens. If None, supply is fixed. |
| **Transfer Fee** | Token-2022 extension: automatic fee deducted on every transfer, collected into a designated account. |
| **Non-Transferable (Soulbound)** | Token-2022 extension: tokens bound to one account, cannot be transferred. Used for credentials, reputation. |
| **Confidential Transfers** | Token-2022 extension: encrypt balances and amounts using ZK proofs. Preserves privacy on public blockchain. |
| **Escrow Pattern** | Atomic swap primitive: deposit token A into vault PDA, taker provides token B, vault releases A. |
| **CPI (Cross-Program Invocation)** | Calling another program (e.g., Token Program) from your program. PDAs can sign CPIs to act as authorities. |
| **PDA as Mint Authority** | Mint owned by a program-derived address. Program logic controls when tokens can be minted. |
| **AMM (Automated Market Maker)** | Constant product formula (x * y = k). Users trade against a liquidity pool instead of an orderbook. |

### Ecosystem Context and Trade-offs

The single Token Program architecture enables **composability**: every DeFi protocol (Raydium, Orca, Marinade, Jupiter) uses the same transfer/mint/burn CPIs. No need for per-token integration—if it's SPL, it works.

**Trade-offs:**
- **Uniform cost** → No per-token gas optimization. Ethereum allows custom ERC-20 implementations to optimize gas.
- **Less flexibility** → Custom token logic (vesting, rebasing, fee-on-transfer before Token-2022) requires wrapping programs. ERC-20 is just code—do anything.
- **Account proliferation** → Every (wallet, token) pair needs an ATA. Creates ~1M accounts for popular tokens like USDC. Ethereum ERC-20 is a single mapping.
- **Rent burden** → Each ATA requires ~0.002 SOL rent deposit. Closing accounts reclaims rent. Ethereum storage is "pay once, store forever."

**Alternatives:**
- **Ethereum ERC-20**: Custom contracts per token, flexible but less composable
- **Cosmos IBC**: Native token transfers across chains, but less programmability
- **Sui Coin Standard**: Similar single-program model, typed in Move

Solana's model optimizes for **speed** (parallel execution), **composability** (one standard), and **low fees** (predictable costs). Token-2022 bridges the flexibility gap by moving common patterns (transfer fees, non-transferability) into the core program, avoiding the "every feature needs a wrapper program" problem.

## Prerequisites

- **Practice 021a completed** — you should already understand:
  - Anchor program structure (`declare_id!`, `#[program]`, `#[derive(Accounts)]`)
  - Account model (owned accounts, system accounts, rent)
  - PDAs (seeds, bumps, `find_program_address`)
  - CPIs (Cross-Program Invocations)
  - LiteSVM test setup

## Description

Build progressively complex token programs that cover the full spectrum of Solana DeFi primitives:

1. **SPL Token basics** — mints, ATAs, minting, transferring (the atoms of all Solana DeFi)
2. **Access control via PDA mint authority** — program-controlled token supply
3. **Escrow pattern** — THE canonical Solana pattern: atomic two-party token swap
4. **PDA vaults & staking** — program-controlled token custody with time-based rewards
5. **Token-2022 extensions** — transfer fees, non-transferable (soulbound) tokens
6. **Mini DEX (AMM)** — constant product market maker combining everything

**Key insight**: ALL Solana tokens (USDC, SOL wrappers, memecoins, NFTs) use the same Token Program. Unlike Ethereum where each ERC-20 is a separate contract, Solana has ONE program that manages ALL tokens. This is a fundamentally different architecture.

## Instructions

### Phase 1: SPL Token Basics (~20 min)

**Concepts**: Mint Account (defines a token), Token Account / ATA (holds a balance), decimals, supply.

1. Copy `phase1_spl/lib.rs` into your Anchor project's `programs/<name>/src/lib.rs`
2. Read the account structs — they're complete. Understand each constraint.
3. **User implements:** `create_mint` — just needs the instruction body (Anchor `init` does the work)
4. **User implements:** `create_token_account` — initialize an ATA for a user
5. **User implements:** `mint_tokens(amount)` — CPI to `token_interface::mint_to`
6. **User implements:** `transfer_tokens(amount)` — CPI to `token_interface::transfer_checked`
7. Key question: Why does `transfer_checked` require `decimals` but `mint_to` doesn't?

### Phase 2: Token Metadata & Access Control (~20 min)

**Concepts**: Mint authority, freeze authority, PDA as mint authority (program = central bank).

1. Copy `phase2_managed/lib.rs` into your project
2. **User implements:** `initialize_managed_mint` — create a mint where a PDA is the authority
3. **User implements:** `managed_mint(amount)` — mint using PDA signer seeds (`with_signer`)
4. **User implements:** `burn_tokens(amount)` — CPI to `token_interface::burn`
5. **User implements:** `airdrop(amounts)` — mint to multiple recipients via remaining_accounts
6. Key question: Why is PDA-as-authority safer than using a wallet keypair as authority?

### Phase 3: Escrow Pattern (~30 min)

**Concepts**: Escrow = conditional atomic swap. Maker deposits token A, taker provides token B, escrow releases both. This is THE canonical Solana tutorial project.

1. Copy `phase3_escrow/lib.rs` into your project
2. Read the `EscrowAccount` state struct — understand every field
3. **User implements:** `make_escrow` — maker deposits token A into vault PDA, creates escrow state
4. **User implements:** `take_escrow` — taker sends token B to maker, vault releases token A to taker
5. **User implements:** `cancel_escrow` — maker reclaims token A, escrow account closed
6. Copy `phase3_escrow/escrow.test.ts` for test structure reference (adapt to LiteSVM if preferred)
7. Key question: Why must the vault be a PDA owned by the program, not just the maker's ATA?

**Real-world**: Raydium, Orca, Jupiter all use escrow variants for swaps and limit orders.

### Phase 4: PDA Vaults & Treasury (~20 min)

**Concepts**: Program-controlled token custody, staking with time-based rewards, fee collection.

1. Copy `phase4_vault/lib.rs` into your project
2. **User implements:** `stake(amount)` — transfer tokens to vault PDA, create stake receipt
3. **User implements:** `unstake` — calculate rewards via `Clock::get()?.unix_timestamp`, return tokens + rewards
4. **User implements:** `collect_fees(amount)` — transfer from user to treasury PDA
5. Key question: How would you handle the case where the vault doesn't have enough tokens for rewards?

**Real-world**: Marinade Finance (mSOL), Jito staking, and all Solana staking protocols use this pattern.

### Phase 5: Token-2022 Extensions (~20 min)

**Concepts**: Token-2022 moves common patterns from custom programs into the token program itself.

1. Copy `phase5_token2022/lib.rs` into your project
2. **User implements:** Create a mint with transfer fees (fee basis points + max fee)
3. **User implements:** Create a non-transferable (soulbound) token
4. Read the conceptual notes on confidential transfers (ZK proofs — understand, don't implement)
5. Key question: What use cases does a non-transferable token enable? (credentials, reputation, membership)

**Note**: If Token-2022 Anchor support is incomplete for some extensions, this phase is partially conceptual.

### Phase 6: End-to-End: Mini DEX (~25 min)

**Concepts**: Automated Market Maker using constant product formula (x * y = k).

1. Copy `phase6_dex/lib.rs` into your project
2. **User implements:** `initialize_pool` — create liquidity pool PDA holding two token vaults
3. **User implements:** `add_liquidity(amount_a, amount_b)` — deposit both tokens, mint LP tokens
4. **User implements:** `swap(amount_in, min_amount_out)` — constant product math, apply fee
5. Copy `phase6_dex/dex.test.ts` for test reference
6. Key question: Why must all AMM math use integer arithmetic (u64), never floating point?

**Real-world**: Raydium CPMM, Orca Whirlpools, and Uniswap V2 all use variants of x*y=k.

## Motivation

- **DeFi literacy**: Tokens, escrows, and AMMs are the foundation of ALL DeFi — understanding them is non-negotiable for Solana development
- **Rust mastery in production context**: Anchor programs are real Rust with real constraints (no_std-like, BPF target, 200K compute budget)
- **Complementary to 021a**: Accounts + PDAs (021a) are the "data model"; tokens + escrows (021b) are the "business logic" layer
- **Market demand**: Solana DeFi TVL is multi-billion; every protocol needs developers who understand these primitives
- **Interview-relevant**: The escrow pattern and AMM math are standard Solana interview questions
- **Portfolio differentiator**: Demonstrates blockchain engineering beyond "hello world" — actual DeFi building blocks

## References

- [Anchor SPL Token Basics](https://www.anchor-lang.com/docs/tokens/basics) — Official Anchor token documentation
- [Anchor Escrow Example](https://github.com/solana-foundation/anchor/blob/master/tests/escrow/programs/escrow/src/lib.rs) — Official escrow test in Anchor repo
- [SPL Token Program](https://spl.solana.com/token) — Solana Program Library token docs
- [Token-2022 Program](https://spl.solana.com/token-2022) — Token Extensions specification
- [Solana Cookbook: Tokens](https://solana.com/docs) — Solana official documentation
- [LiteSVM Testing](https://www.anchor-lang.com/docs/testing/litesvm) — Anchor LiteSVM guide
- [Anchor Token-2022 Guide](https://www.quicknode.com/guides/solana-development/anchor/token-2022) — QuickNode Token Extensions tutorial
- [Uniswap V2 AMM on Solana](https://github.com/rust-trust/anchor-uniswap-v2) — Reference constant-product AMM
- [Anchor Escrow Tutorial (HackMD)](https://hackmd.io/@ironaddicteddog/solana-anchor-escrow) — Step-by-step escrow walkthrough
- [RareSkills: SPL Token Transfers](https://rareskills.io/post/spl-token-transfer) — Detailed token transfer guide

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
| `dev.bat init solana_tokens` | Initialize a new Anchor project named `solana_tokens` inside the container |
| `dev.bat build` | Run `anchor build` inside the container (compiles program to `.so`) |
| `dev.bat test` | Run `anchor test` inside the container (validator + deploy + tests) |
| `dev.bat validator` | Start `solana-test-validator` inside the container |

### Inside the Container (via `dev.bat shell`)

| Command | Description |
|---------|-------------|
| `anchor init solana_tokens` | Initialize a new Anchor project |
| `anchor build` | Compile the Solana program to `target/deploy/*.so` |
| `anchor test` | Build, deploy to local validator, and run TypeScript tests |
| `anchor test --skip-build` | Run tests without rebuilding the program |
| `solana-test-validator` | Start a local Solana validator (for manual testing) |
| `npm install --save-dev litesvm anchor-litesvm` | Install LiteSVM test dependencies |
| `npm install @coral-xyz/anchor @solana/web3.js` | Install Anchor and Solana JS client libraries |
| `solana-keygen new --no-bip39-passphrase --force` | Generate a new local keypair |
| `solana config set --url localhost` | Set Solana CLI to use local validator |

## State

`not-started`
