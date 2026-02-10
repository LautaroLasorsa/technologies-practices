// =============================================================================
// Phase 6: Events & Error Handling
// =============================================================================
//
// REFERENCE FILE: Copy relevant parts into your program's lib.rs
//
// KEY CONCEPTS:
//
// 1. CUSTOM ERRORS:
//    Anchor's #[error_code] macro creates typed errors with codes and messages.
//    Each variant gets a unique error code (starting at 6000 + offset).
//    Clients can match on these codes for specific error handling.
//
//    Compared to Ethereum: like Solidity's `error InsufficientFunds(uint256 balance);`
//    But in Solana, errors are returned (not reverted), and the error code is
//    part of the transaction log.
//
// 2. EVENTS:
//    Anchor's `emit!()` macro serializes an event struct into the transaction log.
//    Off-chain systems (frontends, indexers, bots) parse these logs to react
//    to on-chain state changes.
//
//    Compared to Ethereum: exactly like Solidity's `emit Transfer(from, to, amount);`
//    Events are not stored in account data -- they exist only in transaction logs.
//
//    Use cases:
//      - Frontend: update UI when a counter changes
//      - Indexer: build a database of all counter changes (like a subgraph)
//      - Bot: react to deposits/withdrawals
//
// 3. WHY NOT JUST USE msg!()?
//    msg!() writes a string to the program log. It's good for debugging but:
//      - Unstructured: clients have to parse strings
//      - No type safety: typos silently break parsing
//      - No IDL integration: Anchor events are in the IDL, so clients auto-parse them
//
//    emit!() writes structured, typed data that Anchor clients can decode automatically.
//
// =============================================================================

use anchor_lang::prelude::*;

declare_id!("11111111111111111111111111111111");

// =============================================================================
// Exercise 1: Custom Error Enum
// =============================================================================
//
// Define meaningful errors with the #[error_code] macro.
// Each variant gets a code and a human-readable message.
//
// These replace generic ProgramError variants with domain-specific errors
// that make debugging and client-side error handling much easier.
//
// Rust parallel: This is like a custom Error enum that implements std::error::Error,
// but Anchor handles the serialization and code assignment.
//

#[error_code]
pub enum VaultError {
    // TODO(human): Define at least 3 custom errors
    //
    // Hint: Each variant has a #[msg("...")] attribute for the message.
    //
    //   #[msg("Counter would underflow below zero")]
    //   CounterUnderflow,
    //
    //   #[msg("Only the authority can perform this action")]
    //   Unauthorized,
    //
    //   #[msg("Insufficient funds in vault for withdrawal")]
    //   InsufficientFunds,
    //
    //   #[msg("Counter would overflow above u64::MAX")]
    //   CounterOverflow,
    //
    //   #[msg("Name exceeds maximum length of 32 bytes")]
    //   NameTooLong,
    //
    //   #[msg("Bio exceeds maximum length of 256 bytes")]
    //   BioTooLong,
    //

    // Placeholder -- replace with your errors
    #[msg("Placeholder error")]
    Placeholder,
}

// =============================================================================
// Exercise 2: Event Structs
// =============================================================================
//
// Events are emitted with `emit!(EventName { field1, field2, ... });`
// They appear in the transaction logs and can be parsed by Anchor clients.
//
// Design principle: include enough context for off-chain systems to
// understand what happened WITHOUT fetching additional on-chain data.
//

// TODO(human): Define a CounterChanged event
//
// Hint:
//   #[event]
//   pub struct CounterChanged {
//       pub old_count: u64,
//       pub new_count: u64,
//       pub user: Pubkey,
//       pub timestamp: i64,  // Clock::get()?.unix_timestamp
//   }
//

// TODO(human): Define a VaultDeposit event
//
// Hint:
//   #[event]
//   pub struct VaultDeposit {
//       pub user: Pubkey,
//       pub amount: u64,
//       pub total_deposited: u64,  // running total after deposit
//   }
//

// TODO(human): Define a VaultWithdrawal event
//
// Hint:
//   #[event]
//   pub struct VaultWithdrawal {
//       pub user: Pubkey,
//       pub amount: u64,
//       pub remaining: u64,  // balance after withdrawal
//   }
//

// Placeholder event -- replace with your events
#[event]
pub struct PlaceholderEvent {
    pub message: String,
}

// =============================================================================
// Using errors and events in instruction handlers
// =============================================================================
//
// Here's how your Phase 2 increment would look with errors + events:
//

#[program]
pub mod solana_practice {
    use super::*;

    pub fn increment(ctx: Context<Increment>) -> Result<()> {
        let counter = &mut ctx.accounts.counter;
        let old_count = counter.count;

        // Use custom error instead of generic ProgramError
        // TODO(human): Replace ProgramError with your custom error
        //
        // Hint:
        //   counter.count = counter.count
        //       .checked_add(1)
        //       .ok_or(VaultError::CounterOverflow)?;
        //
        // Why checked_add? Prevents u64 overflow. In competitive programming
        // you might not care, but on-chain, overflow = vulnerability.
        //
        counter.count += 1;

        // Emit an event for off-chain systems
        // TODO(human): Emit CounterChanged event
        //
        // Hint:
        //   emit!(CounterChanged {
        //       old_count,
        //       new_count: counter.count,
        //       user: ctx.accounts.authority.key(),
        //       timestamp: Clock::get()?.unix_timestamp,
        //   });
        //
        // Note: Clock::get()? fetches the current cluster time.
        // In LiteSVM tests, you can manipulate this with .setClock().
        //

        Ok(())
    }

    pub fn decrement(ctx: Context<Decrement>) -> Result<()> {
        let counter = &mut ctx.accounts.counter;
        let old_count = counter.count;

        // TODO(human): Use require! with your custom error
        //
        // Hint:
        //   require!(counter.count > 0, VaultError::CounterUnderflow);
        //   counter.count -= 1;
        //
        // Alternative with checked_sub:
        //   counter.count = counter.count
        //       .checked_sub(1)
        //       .ok_or(VaultError::CounterUnderflow)?;
        //

        // TODO(human): Emit CounterChanged event (same as increment)
        //

        Ok(())
    }

    pub fn deposit(ctx: Context<Deposit>, amount: u64) -> Result<()> {
        // ... CPI transfer logic from Phase 5 ...

        // TODO(human): Emit VaultDeposit event after successful transfer
        //
        // Hint:
        //   emit!(VaultDeposit {
        //       user: ctx.accounts.user.key(),
        //       amount,
        //       total_deposited: ctx.accounts.vault_data.total_deposited,
        //   });
        //

        Ok(())
    }

    pub fn withdraw(ctx: Context<Withdraw>, amount: u64) -> Result<()> {
        let vault_data = &ctx.accounts.vault_data;

        // TODO(human): Replace ProgramError with custom error
        //
        // Before:
        //   require!(vault_data.total_deposited >= amount, ProgramError::InsufficientFunds);
        // After:
        //   require!(vault_data.total_deposited >= amount, VaultError::InsufficientFunds);
        //

        // ... CPI transfer logic from Phase 5 ...

        // TODO(human): Emit VaultWithdrawal event
        //
        // Hint:
        //   emit!(VaultWithdrawal {
        //       user: ctx.accounts.authority.key(),
        //       amount,
        //       remaining: ctx.accounts.vault_data.total_deposited,
        //   });
        //

        Ok(())
    }
}

// =============================================================================
// Placeholder account structs (use your real ones from Phase 2-5)
// =============================================================================

#[account]
pub struct Counter {
    pub authority: Pubkey,
    pub count: u64,
}

#[derive(Accounts)]
pub struct Increment<'info> {
    #[account(mut, has_one = authority)]
    pub counter: Account<'info, Counter>,
    pub authority: Signer<'info>,
}

#[derive(Accounts)]
pub struct Decrement<'info> {
    #[account(mut, has_one = authority)]
    pub counter: Account<'info, Counter>,
    pub authority: Signer<'info>,
}

// Placeholder Deposit/Withdraw -- use real ones from Phase 5
#[derive(Accounts)]
pub struct Deposit<'info> {
    #[account(mut)]
    pub user: Signer<'info>,
    /// CHECK: placeholder
    pub vault_data: AccountInfo<'info>,
    pub system_program: Program<'info, System>,
}

#[derive(Accounts)]
pub struct Withdraw<'info> {
    #[account(mut)]
    pub authority: Signer<'info>,
    /// CHECK: placeholder
    pub vault_data: AccountInfo<'info>,
    pub system_program: Program<'info, System>,
}

// =============================================================================
// TESTING EVENTS IN LiteSVM
// =============================================================================
//
// Exercise 3: In your test file, verify events are emitted.
//
// With anchor-litesvm, you can inspect transaction logs for events.
// Anchor events are base64-encoded in the program's log output.
//
// Pattern (TypeScript):
//
// ```typescript
// // Option A: Use Anchor's event parser
// const listener = program.addEventListener("CounterChanged", (event) => {
//   console.log("Event:", event);
//   expect(event.oldCount.toNumber()).toBe(0);
//   expect(event.newCount.toNumber()).toBe(1);
// });
//
// await program.methods.increment()
//   .accounts({ counter: counterPda, authority: user.publicKey })
//   .signers([user])
//   .rpc();
//
// // Give the event listener time to process
// await new Promise((resolve) => setTimeout(resolve, 500));
// program.removeEventListener(listener);
//
// // Option B: Parse from transaction logs directly
// // (More reliable in LiteSVM since event listeners may not work)
// const txSig = await program.methods.increment()
//   .accounts({ counter: counterPda, authority: user.publicKey })
//   .signers([user])
//   .rpc();
//
// // Inspect the transaction for event data in logs
// ```
//
// Note: Event listener support in LiteSVM may vary. If it doesn't work,
// verify events by checking the program logs for the base64-encoded event data.
//
// =============================================================================
//
// ERROR CODES REFERENCE
// =============================================================================
//
// Anchor error codes start at 6000 for custom errors.
// Your first variant = 6000, second = 6001, etc.
//
// In tests:
//   try {
//     await program.methods.decrement().accounts({...}).signers([...]).rpc();
//   } catch (err) {
//     // err.error.errorCode.code === "CounterUnderflow"
//     // err.error.errorCode.number === 6000
//     // err.error.errorMessage === "Counter would underflow below zero"
//     expect(err.error.errorCode.code).toBe("CounterUnderflow");
//   }
//
// This structured error handling is MUCH better than parsing error strings.
//
// =============================================================================
