// =============================================================================
// Phase 2: Counter Program -- Accounts & Data Storage
// =============================================================================
//
// REFERENCE FILE: Copy relevant parts into programs/solana_practice/src/lib.rs
//
// KEY CONCEPTS:
//
// 1. ACCOUNT MODEL -- Unlike Ethereum where contracts store their own state,
//    Solana programs are STATELESS. All data lives in separate Accounts.
//    Think of it like: the program is a function, accounts are its arguments.
//
// 2. ANCHOR MACROS:
//    - #[program]      -- Marks the module containing instruction handlers
//    - #[derive(Accounts)] -- Declares which accounts an instruction needs + constraints
//    - #[account]      -- Marks a struct as an account data type (auto Borsh serialization)
//
// 3. SPACE CALCULATION:
//    Every account needs a fixed size declared at creation time.
//    space = 8 (discriminator) + sum of field sizes
//
//    Common sizes:
//      bool    = 1 byte
//      u8      = 1 byte
//      u16     = 2 bytes
//      u32     = 4 bytes
//      u64     = 8 bytes
//      i64     = 8 bytes
//      u128    = 16 bytes
//      Pubkey  = 32 bytes
//      String  = 4 (length prefix) + content bytes
//      Vec<T>  = 4 (length prefix) + (count * size_of::<T>())
//      Option<T> = 1 + size_of::<T>()
//
// 4. CONSTRAINTS:
//    init       -- Create a new account (allocates space, pays rent)
//    mut        -- Account data will be modified
//    has_one    -- Verify that account.field == another_account.key()
//    payer      -- Who pays for account creation (rent deposit)
//    space      -- How many bytes to allocate
//
// 5. RENT:
//    Solana charges rent for storing data on-chain. If you deposit enough
//    lamports (~ 2 years of rent), the account becomes "rent-exempt" and
//    lives forever. Anchor's `init` constraint handles this automatically.
//
// =============================================================================

use anchor_lang::prelude::*;

// This is your program's on-chain address. `anchor build` generates it.
// Replace with your actual program ID from target/deploy/solana_practice-keypair.json
declare_id!("11111111111111111111111111111111");

#[program]
pub mod solana_practice {
    use super::*;

    // =========================================================================
    // Exercise 1: Initialize a Counter account
    // =========================================================================
    //
    // This instruction creates a new Counter account on-chain.
    // The account is owned by this program and stores a count + authority.
    //
    // Analogy: Like `malloc` + writing initial values. The account didn't
    // exist before this instruction -- Anchor's `init` constraint creates it.
    //
    pub fn initialize(ctx: Context<Initialize>) -> Result<()> {
        let counter = &mut ctx.accounts.counter;

        // ── Exercise Context ──────────────────────────────────────────────────
        // This exercise teaches account initialization with Anchor's `init` constraint.
        // Understanding how accounts are created (space calculation, rent-exemption, owner
        // assignment) is fundamental to all Solana programming.
        //
        // TODO(human): Set counter.authority to the user's public key
        //   Hint: ctx.accounts.user.key()
        //
        // TODO(human): Set counter.count to 0
        //

        Ok(())
    }

    // =========================================================================
    // Exercise 2: Increment the counter
    // =========================================================================
    //
    // Reads the current count, adds 1, writes it back.
    //
    // Notice: we mark `counter` as `mut` in the Accounts struct below --
    // without this, Anchor would reject any writes to the account data.
    //
    // Rust parallel: This is like `&mut counter.count += 1`, but the
    // "borrow checker" is Anchor's constraint validation, not rustc.
    //
    pub fn increment(ctx: Context<Increment>) -> Result<()> {
        let counter = &mut ctx.accounts.counter;

        // ── Exercise Context ──────────────────────────────────────────────────
        // This exercise teaches account mutation and the `mut` constraint. Anchor validates
        // writable accounts at runtime—without `#[account(mut)]`, writes would fail. This is
        // Solana's version of Rust's borrow checker, enforced by the runtime not rustc.
        //
        // TODO(human): Increment counter.count by 1
        //   Hint: counter.count += 1;  (yes, it's that simple)
        //   But think about: what happens at u64::MAX? (see Phase 6 for overflow handling)
        //

        Ok(())
    }

    // =========================================================================
    // Exercise 3: Decrement with a safety check
    // =========================================================================
    //
    // Same as increment, but subtracts 1.
    // Must NOT go below zero -- use Anchor's require! macro.
    //
    // The require! macro is like assert! but returns a proper Anchor error
    // instead of panicking. On-chain panics waste compute and give bad errors.
    //
    pub fn decrement(ctx: Context<Decrement>) -> Result<()> {
        let counter = &mut ctx.accounts.counter;

        // TODO(human): Check that counter.count > 0 using require!
        //   Hint: require!(counter.count > 0, ErrorCode::CounterUnderflow);
        //   You'll define ErrorCode in Phase 6. For now, use a simple error:
        //   require!(counter.count > 0, ProgramError::InvalidArgument);
        //   Or use: require_gt!(counter.count, 0);
        //
        // TODO(human): Decrement counter.count by 1
        //

        Ok(())
    }
}

// =============================================================================
// Account Data Structures
// =============================================================================
//
// #[account] tells Anchor:
//   1. Add an 8-byte discriminator (unique hash of "account:Counter")
//   2. Implement Borsh serialization/deserialization
//   3. Set the account owner to this program's ID
//
// The discriminator prevents account type confusion -- if someone passes
// a UserProfile account where a Counter is expected, the discriminator
// check fails before your code even runs.
//

#[account]
pub struct Counter {
    pub authority: Pubkey,   // 32 bytes -- who can modify this counter
    pub count: u64,          // 8 bytes  -- the counter value
}

// Total space = 8 (discriminator) + 32 (authority) + 8 (count) = 48 bytes

// =============================================================================
// Instruction Account Structs (#[derive(Accounts)])
// =============================================================================
//
// Each instruction needs a struct declaring which accounts it uses.
// This is the "type signature" of the instruction.
//
// Anchor validates ALL constraints BEFORE your instruction code runs.
// If any constraint fails, the transaction is rejected -- your handler
// never executes. This is a security feature.
//

// --- Initialize ---
// Creates a new Counter account. The `user` pays for rent.
#[derive(Accounts)]
pub struct Initialize<'info> {
    // `init` = create this account (must not already exist)
    // `payer = user` = user's SOL pays for rent deposit
    // `space = 8 + 32 + 8` = discriminator + authority + count
    #[account(init, payer = user, space = 8 + 32 + 8)]
    pub counter: Account<'info, Counter>,

    // `mut` because SOL is deducted from user to pay rent
    #[account(mut)]
    pub user: Signer<'info>,

    // System Program is needed for account creation (it's the "allocator")
    pub system_program: Program<'info, System>,
}

// --- Increment ---
// Modifies an existing Counter. Only the authority can call this.
#[derive(Accounts)]
pub struct Increment<'info> {
    // `mut` = we will modify this account's data
    // `has_one = authority` = counter.authority must equal the authority account's key
    //
    // Exercise 4: Add `has_one = authority` to enforce access control
    //
    // TODO(human): Add the `has_one = authority` constraint
    //   Before: #[account(mut)]
    //   After:  #[account(mut, has_one = authority)]
    //
    #[account(mut)]
    pub counter: Account<'info, Counter>,

    // The signer -- Anchor verifies this account actually signed the transaction.
    // With `has_one = authority`, Anchor also checks counter.authority == authority.key()
    pub authority: Signer<'info>,
}

// --- Decrement ---
// Same constraints as Increment -- authority must sign.
#[derive(Accounts)]
pub struct Decrement<'info> {
    // TODO(human): Same pattern as Increment -- add `has_one = authority`
    #[account(mut)]
    pub counter: Account<'info, Counter>,

    pub authority: Signer<'info>,
}

// =============================================================================
// MENTAL MODEL SUMMARY
// =============================================================================
//
// Transaction flow:
//   1. Client constructs a transaction with:
//      - Instruction data (which function + arguments)
//      - Account list (which accounts to pass)
//      - Signers (who signed the transaction)
//
//   2. Runtime delivers the instruction to your program with the accounts
//
//   3. Anchor's #[derive(Accounts)] validates:
//      - Account ownership (is it owned by this program?)
//      - Discriminator (is it the right account type?)
//      - Constraints (has_one, seeds, mut, signer, etc.)
//
//   4. Only THEN does your handler code run
//
//   5. Anchor serializes modified account data back to the account
//
// This is fundamentally different from Ethereum where you just call
// contract.increment() and the contract accesses its own storage.
// In Solana, the CLIENT must know which accounts to pass.
// =============================================================================
