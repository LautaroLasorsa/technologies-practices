// =============================================================================
// Phase 5: Cross-Program Invocations (CPI)
// =============================================================================
//
// REFERENCE FILE: Copy relevant parts into your program's lib.rs
//
// KEY CONCEPTS:
//
// 1. WHAT IS CPI?
//    CPI = one program calling another program's instruction.
//    The most common CPI is calling the System Program to transfer SOL.
//
//    In Ethereum, contracts can directly call other contracts.
//    In Solana, programs must explicitly construct a CPI instruction
//    with the required accounts and invoke it.
//
// 2. PDA AS SIGNER:
//    Regular accounts sign transactions with their private key.
//    PDAs don't HAVE private keys. Instead, the program that derived
//    the PDA can "sign" on its behalf by providing the seeds.
//
//    The runtime verifies: hash(seeds + program_id) == PDA address.
//    If it matches, the PDA is considered a valid signer.
//
//    This is THE mechanism for programs to control funds:
//      - User deposits SOL into a PDA vault (normal System Program transfer)
//      - Program withdraws from vault via CPI with PDA signer seeds
//      - No private key needed -- the program IS the authority
//
// 3. CPI CONTEXT:
//    Anchor provides `CpiContext` to simplify CPI calls:
//
//    // Without PDA signer:
//    let cpi_ctx = CpiContext::new(system_program, transfer_accounts);
//    anchor_lang::system_program::transfer(cpi_ctx, amount)?;
//
//    // With PDA signer:
//    let seeds = &[b"vault", user.as_ref(), &[bump]];
//    let signer_seeds = &[&seeds[..]];
//    let cpi_ctx = CpiContext::new_with_signer(
//        system_program, transfer_accounts, signer_seeds
//    );
//    anchor_lang::system_program::transfer(cpi_ctx, amount)?;
//
// 4. ANALOGY:
//    Think of CPI like calling another library's function:
//      my_program::withdraw()  -->  system_program::transfer()
//
//    But the callee (System Program) verifies that the signer is
//    authorized. For PDAs, the runtime does this verification automatically
//    when the seeds match.
//
// =============================================================================

use anchor_lang::prelude::*;
use anchor_lang::system_program;

declare_id!("11111111111111111111111111111111");

#[program]
pub mod solana_practice {
    use super::*;

    // =========================================================================
    // Exercise 1: Deposit SOL into a vault PDA
    // =========================================================================
    //
    // The user transfers SOL from their wallet to a PDA vault.
    // This uses CPI to call the System Program's transfer instruction.
    //
    // Flow:
    //   1. User signs the transaction
    //   2. Our program constructs a CPI to system_program::transfer
    //   3. SOL moves from user's account to the vault PDA
    //
    // The vault PDA is derived from: seeds = [b"vault", user.key().as_ref()]
    // Each user gets their own vault.
    //
    pub fn deposit(ctx: Context<Deposit>, amount: u64) -> Result<()> {
        // ── Exercise Context ──────────────────────────────────────────────────
        // This exercise teaches Cross-Program Invocations (CPI)—the mechanism for programs
        // to call other programs. Understanding CPI is essential for composability in Solana:
        // every DeFi protocol (DEX, lending, staking) relies on CPI to transfer tokens and SOL.
        //
        // TODO(human): Create the CPI context for system_program::transfer
        //
        // Hint: Build the CPI accounts struct and context:
        //
        //   let cpi_accounts = system_program::Transfer {
        //       from: ctx.accounts.user.to_account_info(),
        //       to: ctx.accounts.vault.to_account_info(),
        //   };
        //   let cpi_program = ctx.accounts.system_program.to_account_info();
        //   let cpi_ctx = CpiContext::new(cpi_program, cpi_accounts);
        //
        // TODO(human): Execute the CPI transfer
        //
        //   system_program::transfer(cpi_ctx, amount)?;
        //

        // Update the vault's tracking data
        let vault_data = &mut ctx.accounts.vault_data;
        vault_data.total_deposited = vault_data
            .total_deposited
            .checked_add(amount)
            .ok_or(ProgramError::ArithmeticOverflow)?;

        Ok(())
    }

    // =========================================================================
    // Exercise 2: Withdraw SOL from the vault PDA
    // =========================================================================
    //
    // The vault PDA "signs" the CPI transfer using its seeds.
    // This is the key insight: the program controls the PDA's funds
    // without a private key -- it uses the seeds to prove ownership.
    //
    // Flow:
    //   1. Authority (original depositor) signs the transaction
    //   2. Our program verifies authority via has_one constraint
    //   3. Our program constructs a CPI with PDA signer seeds
    //   4. SOL moves from vault PDA back to the user
    //
    pub fn withdraw(ctx: Context<Withdraw>, amount: u64) -> Result<()> {
        let vault_data = &ctx.accounts.vault_data;

        // Check sufficient balance
        require!(
            vault_data.total_deposited >= amount,
            ProgramError::InsufficientFunds
        );

        // TODO(human): Build the signer seeds for the vault PDA
        //
        // The vault PDA was derived with seeds = [b"vault", authority.key().as_ref()]
        // We stored the bump in vault_data.bump during initialization.
        //
        // Hint:
        //   let authority_key = ctx.accounts.authority.key();
        //   let seeds = &[
        //       b"vault".as_ref(),
        //       authority_key.as_ref(),
        //       &[vault_data.vault_bump],
        //   ];
        //   let signer_seeds = &[&seeds[..]];
        //
        // Why the weird &[&seeds[..]] type?
        // CPI can have MULTIPLE PDA signers. So the outer slice is a
        // list of signer seed sets. We have just one, but the API
        // supports many. It's like Vec<Vec<&[u8]>> but with slices.
        //

        // TODO(human): Create the CPI context WITH PDA signer
        //
        // Hint:
        //   let cpi_accounts = system_program::Transfer {
        //       from: ctx.accounts.vault.to_account_info(),
        //       to: ctx.accounts.authority.to_account_info(),
        //   };
        //   let cpi_ctx = CpiContext::new_with_signer(
        //       ctx.accounts.system_program.to_account_info(),
        //       cpi_accounts,
        //       signer_seeds,
        //   );
        //   system_program::transfer(cpi_ctx, amount)?;
        //

        // Update tracking
        let vault_data = &mut ctx.accounts.vault_data;
        vault_data.total_deposited = vault_data
            .total_deposited
            .checked_sub(amount)
            .ok_or(ProgramError::ArithmeticOverflow)?;

        Ok(())
    }

    // =========================================================================
    // Helper: Initialize vault tracking account
    // =========================================================================
    //
    // Creates the vault PDA (to hold SOL) and a data account (to track state).
    // We need two accounts because:
    //   - vault PDA: holds SOL (lamports), no data
    //   - vault_data PDA: stores metadata (authority, bump, total_deposited)
    //
    // Why separate? The vault needs to be a SystemAccount (owned by System Program)
    // to receive SOL transfers. The data account is owned by our program.
    //
    pub fn initialize_vault(ctx: Context<InitializeVault>) -> Result<()> {
        let vault_data = &mut ctx.accounts.vault_data;

        // TODO(human): Set vault_data fields
        //
        // Hint:
        //   vault_data.authority = ctx.accounts.user.key();
        //   vault_data.vault_bump = ctx.bumps.vault;
        //   vault_data.data_bump = ctx.bumps.vault_data;
        //   vault_data.total_deposited = 0;
        //

        Ok(())
    }
}

// =============================================================================
// Account Data Structures
// =============================================================================

#[account]
pub struct VaultData {
    pub authority: Pubkey,       // 32 bytes -- who can withdraw
    pub vault_bump: u8,          // 1 byte  -- bump for the vault PDA (SOL holder)
    pub data_bump: u8,           // 1 byte  -- bump for this data PDA
    pub total_deposited: u64,    // 8 bytes -- tracking total SOL deposited
}
// space = 8 + 32 + 1 + 1 + 8 = 50 bytes

// =============================================================================
// Instruction Account Structs
// =============================================================================

#[derive(Accounts)]
pub struct InitializeVault<'info> {
    // The vault PDA that will HOLD SOL
    // It's a SystemAccount (not #[account]) because it just holds lamports.
    //
    // Why SystemAccount? Because we need the System Program to own it
    // so that system_program::transfer can move SOL in/out.
    //
    /// CHECK: This is a PDA that holds SOL. It's not a data account.
    /// We validate it via seeds constraint.
    #[account(
        mut,
        seeds = [b"vault", user.key().as_ref()],
        bump,
    )]
    pub vault: SystemAccount<'info>,

    // The data PDA that tracks vault state (separate from the SOL holder)
    #[account(
        init,
        payer = user,
        space = 8 + 32 + 1 + 1 + 8,
        seeds = [b"vault_data", user.key().as_ref()],
        bump,
    )]
    pub vault_data: Account<'info, VaultData>,

    #[account(mut)]
    pub user: Signer<'info>,

    pub system_program: Program<'info, System>,
}

#[derive(Accounts)]
pub struct Deposit<'info> {
    // Vault PDA (receives SOL)
    /// CHECK: PDA validated by seeds. Holds lamports only.
    #[account(
        mut,
        seeds = [b"vault", user.key().as_ref()],
        bump = vault_data.vault_bump,
    )]
    pub vault: SystemAccount<'info>,

    // Vault data (tracks deposits)
    #[account(
        mut,
        seeds = [b"vault_data", user.key().as_ref()],
        bump = vault_data.data_bump,
        has_one = authority,
    )]
    pub vault_data: Account<'info, VaultData>,

    // The depositor (must be the authority)
    #[account(mut)]
    pub user: Signer<'info>,

    // Authority check -- redundant with user for deposit, but explicit is good
    /// CHECK: Validated by has_one on vault_data
    pub authority: AccountInfo<'info>,

    pub system_program: Program<'info, System>,
}

#[derive(Accounts)]
pub struct Withdraw<'info> {
    // Vault PDA (sends SOL via CPI)
    /// CHECK: PDA validated by seeds. The program "signs" for this PDA.
    #[account(
        mut,
        seeds = [b"vault", authority.key().as_ref()],
        bump = vault_data.vault_bump,
    )]
    pub vault: SystemAccount<'info>,

    // Vault data (tracks withdrawals)
    #[account(
        mut,
        seeds = [b"vault_data", authority.key().as_ref()],
        bump = vault_data.data_bump,
        has_one = authority,
    )]
    pub vault_data: Account<'info, VaultData>,

    // Only the authority can withdraw
    // TODO(human): Why is authority marked as `mut`?
    //   Answer: Because they RECEIVE lamports (their balance increases).
    //   Any account whose lamport balance changes must be `mut`.
    #[account(mut)]
    pub authority: Signer<'info>,

    pub system_program: Program<'info, System>,
}

// =============================================================================
// CPI MENTAL MODEL
// =============================================================================
//
//  ┌─────────────────────┐      CPI        ┌──────────────────────┐
//  │   Your Program      │ ──────────────>  │   System Program     │
//  │                     │                  │                      │
//  │  withdraw()         │   transfer()     │   Moves lamports     │
//  │  - checks authority │   with PDA       │   from vault -> user │
//  │  - builds CPI ctx   │   signer seeds   │   Verifies signer    │
//  │  - invokes transfer │                  │                      │
//  └─────────────────────┘                  └──────────────────────┘
//
//  The runtime verifies:
//    hash(seeds + your_program_id) == vault PDA address
//  If yes, the vault is considered to have "signed" the transfer.
//
// Compare with Ethereum:
//   In Solidity, you'd just do: payable(user).transfer(amount)
//   In Solana, you must:
//     1. Construct the CPI instruction
//     2. Provide all accounts involved
//     3. Sign with PDA seeds
//   More explicit, but also more auditable and composable.
//
// =============================================================================
