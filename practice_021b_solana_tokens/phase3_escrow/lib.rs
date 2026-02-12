// =============================================================================
// Phase 3: Escrow Pattern — The Canonical Solana DeFi Primitive
// =============================================================================
//
// WHAT IS AN ESCROW?
// An escrow is a trusted intermediary that holds assets until conditions are met.
// In TradFi, this is a lawyer or bank. In DeFi, it's a smart contract (program).
//
// THE SCENARIO:
// ┌─────────────────────────────────────────────────────────────────────┐
// │ Alice (Maker) wants to trade 100 USDC for 1 SOL                   │
// │ Bob (Taker) wants to trade 1 SOL for 100 USDC                     │
// │                                                                    │
// │ Problem: If Alice sends first, Bob might not send back.            │
// │ Solution: Escrow holds Alice's USDC until Bob provides SOL.        │
// │                                                                    │
// │ Flow:                                                              │
// │ 1. MAKE: Alice deposits 100 USDC into escrow vault (PDA)          │
// │    → Creates EscrowAccount with terms: "100 USDC for 1 SOL"       │
// │                                                                    │
// │ 2. TAKE: Bob sees the escrow, agrees to terms                     │
// │    → Bob sends 1 SOL to Alice                                     │
// │    → Escrow releases 100 USDC to Bob                              │
// │    → EscrowAccount is closed (rent returned to Alice)             │
// │                                                                    │
// │ 3. CANCEL (alternative): Alice changes her mind                   │
// │    → Escrow returns 100 USDC to Alice                             │
// │    → EscrowAccount is closed                                      │
// └─────────────────────────────────────────────────────────────────────┘
//
// WHY IS THIS THE CANONICAL SOLANA PATTERN?
// Because it combines EVERYTHING you've learned:
// - Account model (multiple accounts in one instruction)
// - PDAs (vault address derived from escrow state)
// - CPIs (transferring tokens via Token Program)
// - State management (EscrowAccount stores terms)
// - Closing accounts (returning rent to maker)
//
// REAL-WORLD USES:
// - DEX limit orders (Serum/OpenBook, Jupiter Limit)
// - OTC trading platforms
// - NFT marketplaces (escrow the NFT, release on payment)
// - Lending protocols (escrow collateral)
// - Raydium concentrated liquidity positions
//
// DEPENDENCIES (Cargo.toml):
// ```toml
// [dependencies]
// anchor-lang = { version = "0.31", features = ["init-if-needed"] }
// anchor-spl = { version = "0.31", features = ["token", "associated_token"] }
// ```

use anchor_lang::prelude::*;
use anchor_spl::{
    associated_token::AssociatedToken,
    token_interface::{
        self, CloseAccount, Mint, TokenAccount, TokenInterface, TransferChecked,
    },
};

declare_id!("Escrow11111111111111111111111111111111111111");

/// Seed prefix for escrow state accounts.
pub const ESCROW_SEED: &[u8] = b"escrow";

/// Seed prefix for the token vault PDA.
pub const VAULT_SEED: &[u8] = b"vault";

#[program]
pub mod escrow {
    use super::*;

    // =========================================================================
    // Exercise 1: Make Escrow — Maker deposits Token A
    // =========================================================================
    //
    // The maker creates an escrow offer:
    // "I'm depositing X of token_a. I want Y of token_b in return."
    //
    // What happens:
    // 1. EscrowAccount is created (stores the terms)
    // 2. Token A is transferred from maker's ATA to the vault PDA
    //
    // The vault is a Token Account owned by a PDA — the program controls it.
    // Nobody (not even the maker) can withdraw from the vault except through
    // the program's take_escrow or cancel_escrow instructions.
    //
    // Security: The vault PDA is derived from the escrow account's key.
    // This means each escrow has its own unique vault — no shared pool.
    //
    pub fn make_escrow(
        ctx: Context<MakeEscrow>,
        escrow_id: u64,
        deposit_amount: u64,
        receive_amount: u64,
    ) -> Result<()> {
        // ── Exercise Context ──────────────────────────────────────────────────
        // This exercise teaches the escrow pattern—THE foundational primitive of Solana DeFi.
        // Every atomic swap, DEX limit order, and marketplace uses this: deposit into a vault
        // PDA, record terms, release when conditions met. This is Solana's answer to Ethereum's
        // smart contract escrows.
        //
        // TODO(human): Implement make_escrow in two steps.
        //
        // Step 1: Save the escrow terms to the EscrowAccount.
        //
        //   let escrow = &mut ctx.accounts.escrow;
        //   escrow.maker = ctx.accounts.maker.key();
        //   escrow.token_a_mint = ctx.accounts.token_a_mint.key();
        //   escrow.token_b_mint = ctx.accounts.token_b_mint.key();
        //   escrow.deposit_amount = deposit_amount;
        //   escrow.receive_amount = receive_amount;
        //   escrow.escrow_id = escrow_id;
        //   escrow.bump = ctx.bumps.escrow;
        //   escrow.vault_bump = ctx.bumps.vault;
        //
        // Step 2: Transfer token A from maker's ATA to the vault.
        //
        //   let transfer_accounts = TransferChecked {
        //       mint: ctx.accounts.token_a_mint.to_account_info(),
        //       from: ctx.accounts.maker_token_a.to_account_info(),
        //       to: ctx.accounts.vault.to_account_info(),
        //       authority: ctx.accounts.maker.to_account_info(),
        //   };
        //   let cpi_ctx = CpiContext::new(
        //       ctx.accounts.token_program.to_account_info(),
        //       transfer_accounts,
        //   );
        //   token_interface::transfer_checked(
        //       cpi_ctx,
        //       deposit_amount,
        //       ctx.accounts.token_a_mint.decimals,
        //   )?;
        //
        // Question: Why do we store both bumps? (Answer: we need them in
        // take_escrow and cancel_escrow to sign with the PDA.)
        //
        // Question: Why a separate vault per escrow instead of one big pool?
        // (Answer: isolation — one escrow's tokens can't be mixed with another's.
        //  This is also how Raydium AMM pools work — separate vault per pair.)
        todo!("Save escrow state, transfer token A to vault")
    }

    // =========================================================================
    // Exercise 2: Take Escrow — Taker completes the swap
    // =========================================================================
    //
    // The taker agrees to the escrow terms:
    // 1. Taker sends `receive_amount` of token B to the maker
    // 2. Vault releases `deposit_amount` of token A to the taker
    // 3. Vault account is closed (rent returned to maker)
    // 4. Escrow account is closed (rent returned to maker)
    //
    // This is ATOMIC — either both transfers happen or neither does.
    // If any CPI fails, the entire transaction is rolled back.
    // This atomicity is what makes escrows trustless.
    //
    pub fn take_escrow(ctx: Context<TakeEscrow>) -> Result<()> {
        // TODO(human): Implement take_escrow in three steps.
        //
        // Read the escrow terms first:
        //   let escrow = &ctx.accounts.escrow;
        //   let deposit_amount = escrow.deposit_amount;
        //   let receive_amount = escrow.receive_amount;
        //
        // Step 1: Taker sends token B to maker.
        //   (Regular transfer — taker signs directly)
        //
        //   let taker_to_maker = TransferChecked {
        //       mint: ctx.accounts.token_b_mint.to_account_info(),
        //       from: ctx.accounts.taker_token_b.to_account_info(),
        //       to: ctx.accounts.maker_token_b.to_account_info(),
        //       authority: ctx.accounts.taker.to_account_info(),
        //   };
        //   let cpi_ctx = CpiContext::new(
        //       ctx.accounts.token_program.to_account_info(),
        //       taker_to_maker,
        //   );
        //   token_interface::transfer_checked(
        //       cpi_ctx,
        //       receive_amount,
        //       ctx.accounts.token_b_mint.decimals,
        //   )?;
        //
        // Step 2: Vault releases token A to taker (PDA-signed).
        //   Build signer seeds for the escrow PDA:
        //
        //   let escrow_id_bytes = escrow.escrow_id.to_le_bytes();
        //   let signer_seeds: &[&[&[u8]]] = &[&[
        //       ESCROW_SEED,
        //       ctx.accounts.maker.to_account_info().key.as_ref(),
        //       &escrow_id_bytes,
        //       &[escrow.bump],
        //   ]];
        //
        //   let vault_to_taker = TransferChecked {
        //       mint: ctx.accounts.token_a_mint.to_account_info(),
        //       from: ctx.accounts.vault.to_account_info(),
        //       to: ctx.accounts.taker_token_a.to_account_info(),
        //       authority: ctx.accounts.escrow.to_account_info(),
        //   };
        //   let cpi_ctx = CpiContext::new(
        //       ctx.accounts.token_program.to_account_info(),
        //       vault_to_taker,
        //   ).with_signer(signer_seeds);
        //   token_interface::transfer_checked(
        //       cpi_ctx,
        //       deposit_amount,
        //       ctx.accounts.token_a_mint.decimals,
        //   )?;
        //
        // Step 3: Close the vault account (return rent to maker).
        //   let close_accounts = CloseAccount {
        //       account: ctx.accounts.vault.to_account_info(),
        //       destination: ctx.accounts.maker.to_account_info(),
        //       authority: ctx.accounts.escrow.to_account_info(),
        //   };
        //   let cpi_ctx = CpiContext::new(
        //       ctx.accounts.token_program.to_account_info(),
        //       close_accounts,
        //   ).with_signer(signer_seeds);
        //   token_interface::close_account(cpi_ctx)?;
        //
        // The escrow account itself is closed by Anchor's `close = maker`
        // constraint in the Accounts struct.
        //
        // CRITICAL INSIGHT: Both transfers are in the same transaction.
        // If the vault-to-taker transfer fails (e.g., insufficient funds),
        // the taker-to-maker transfer is also rolled back. This is atomicity.
        todo!("Transfer B to maker, transfer A to taker, close vault")
    }

    // =========================================================================
    // Exercise 3: Cancel Escrow — Maker reclaims Token A
    // =========================================================================
    //
    // The maker changes their mind and wants their tokens back.
    // 1. Vault returns token A to maker
    // 2. Vault account is closed (rent returned)
    // 3. Escrow account is closed (rent returned)
    //
    // Only the MAKER can cancel — enforced by the Signer constraint.
    //
    pub fn cancel_escrow(ctx: Context<CancelEscrow>) -> Result<()> {
        // TODO(human): Implement cancel_escrow.
        //
        // This is similar to take_escrow Step 2 + Step 3, but simpler:
        // - Transfer vault tokens back to maker (not taker)
        // - Close the vault
        // - Escrow account closed by Anchor's `close = maker`
        //
        // Steps:
        // 1. Build signer seeds for the escrow PDA (same pattern as take_escrow)
        //
        // 2. Transfer token A from vault back to maker_token_a:
        //    let vault_to_maker = TransferChecked { ... };
        //    (PDA-signed with escrow seeds)
        //
        // 3. Close the vault account:
        //    let close_accounts = CloseAccount { ... };
        //    (PDA-signed with escrow seeds)
        //
        // Hint: The vault authority is the escrow PDA (same as take_escrow).
        // The escrow PDA signs both the transfer and the close.
        todo!("Return tokens to maker, close vault, close escrow")
    }
}

// =============================================================================
// State
// =============================================================================

/// The escrow state account. Stores the terms of the trade.
///
/// Layout: 8 (discriminator) + 32 + 32 + 32 + 8 + 8 + 8 + 1 + 1 = 130 bytes
///
/// Why store bumps? Because we need them to reconstruct PDA signer seeds
/// in take_escrow and cancel_escrow. Storing them avoids re-deriving
/// (which costs compute units).
#[account]
#[derive(InitSpace)]
pub struct EscrowAccount {
    /// The maker's wallet pubkey (who created the escrow).
    pub maker: Pubkey,          // 32 bytes

    /// The mint of the token the maker deposited (token A).
    pub token_a_mint: Pubkey,   // 32 bytes

    /// The mint of the token the maker wants to receive (token B).
    pub token_b_mint: Pubkey,   // 32 bytes

    /// Amount of token A deposited in the vault.
    pub deposit_amount: u64,    // 8 bytes

    /// Amount of token B the maker wants in return.
    pub receive_amount: u64,    // 8 bytes

    /// Unique ID for this escrow (allows multiple escrows per maker).
    pub escrow_id: u64,         // 8 bytes

    /// Bump for the escrow PDA.
    pub bump: u8,               // 1 byte

    /// Bump for the vault PDA.
    pub vault_bump: u8,         // 1 byte
}

// =============================================================================
// Account Validation Structs
// =============================================================================

/// Create a new escrow: maker deposits token A.
#[derive(Accounts)]
#[instruction(escrow_id: u64)]
pub struct MakeEscrow<'info> {
    /// The maker creating the escrow (pays for everything).
    #[account(mut)]
    pub maker: Signer<'info>,

    /// The mint of token A (what maker is depositing).
    pub token_a_mint: InterfaceAccount<'info, Mint>,

    /// The mint of token B (what maker wants to receive).
    pub token_b_mint: InterfaceAccount<'info, Mint>,

    /// Maker's token A account (tokens come from here).
    #[account(
        mut,
        associated_token::mint = token_a_mint,
        associated_token::authority = maker,
        associated_token::token_program = token_program,
    )]
    pub maker_token_a: InterfaceAccount<'info, TokenAccount>,

    /// The escrow state account (PDA).
    /// Seeds: ["escrow", maker_pubkey, escrow_id]
    /// This allows a maker to have multiple active escrows.
    #[account(
        init,
        payer = maker,
        space = 8 + EscrowAccount::INIT_SPACE,
        seeds = [ESCROW_SEED, maker.key().as_ref(), &escrow_id.to_le_bytes()],
        bump,
    )]
    pub escrow: Account<'info, EscrowAccount>,

    /// The vault token account (PDA) — holds token A during escrow.
    /// Seeds: ["vault", escrow_pubkey]
    /// Authority: the escrow PDA (so the program can release tokens).
    #[account(
        init,
        payer = maker,
        token::mint = token_a_mint,
        token::authority = escrow,
        token::token_program = token_program,
        seeds = [VAULT_SEED, escrow.key().as_ref()],
        bump,
    )]
    pub vault: InterfaceAccount<'info, TokenAccount>,

    pub token_program: Interface<'info, TokenInterface>,
    pub system_program: Program<'info, System>,
}

/// Complete the escrow: taker provides token B, receives token A.
#[derive(Accounts)]
pub struct TakeEscrow<'info> {
    /// The taker completing the escrow.
    #[account(mut)]
    pub taker: Signer<'info>,

    /// The maker (receives rent back when accounts close).
    /// CHECK: Validated by escrow.maker constraint.
    #[account(mut)]
    pub maker: SystemAccount<'info>,

    /// Token A mint.
    pub token_a_mint: InterfaceAccount<'info, Mint>,

    /// Token B mint.
    pub token_b_mint: InterfaceAccount<'info, Mint>,

    /// Taker's token A account (receives token A from vault).
    #[account(
        init_if_needed,
        payer = taker,
        associated_token::mint = token_a_mint,
        associated_token::authority = taker,
        associated_token::token_program = token_program,
    )]
    pub taker_token_a: InterfaceAccount<'info, TokenAccount>,

    /// Taker's token B account (sends token B to maker).
    #[account(
        mut,
        associated_token::mint = token_b_mint,
        associated_token::authority = taker,
        associated_token::token_program = token_program,
    )]
    pub taker_token_b: InterfaceAccount<'info, TokenAccount>,

    /// Maker's token B account (receives token B from taker).
    #[account(
        init_if_needed,
        payer = taker,
        associated_token::mint = token_b_mint,
        associated_token::authority = maker,
        associated_token::token_program = token_program,
    )]
    pub maker_token_b: InterfaceAccount<'info, TokenAccount>,

    /// The escrow state. Verified: maker matches, mints match.
    /// `close = maker` → rent is returned to maker when this account is closed.
    #[account(
        mut,
        close = maker,
        seeds = [ESCROW_SEED, maker.key().as_ref(), &escrow.escrow_id.to_le_bytes()],
        bump = escrow.bump,
        has_one = maker,
        has_one = token_a_mint,
        has_one = token_b_mint,
    )]
    pub escrow: Account<'info, EscrowAccount>,

    /// The vault holding token A.
    #[account(
        mut,
        token::mint = token_a_mint,
        token::authority = escrow,
        token::token_program = token_program,
        seeds = [VAULT_SEED, escrow.key().as_ref()],
        bump = escrow.vault_bump,
    )]
    pub vault: InterfaceAccount<'info, TokenAccount>,

    pub token_program: Interface<'info, TokenInterface>,
    pub associated_token_program: Program<'info, AssociatedToken>,
    pub system_program: Program<'info, System>,
}

/// Cancel the escrow: maker reclaims token A.
#[derive(Accounts)]
pub struct CancelEscrow<'info> {
    /// The maker (only they can cancel).
    #[account(mut)]
    pub maker: Signer<'info>,

    /// Token A mint (needed for transfer_checked).
    pub token_a_mint: InterfaceAccount<'info, Mint>,

    /// Maker's token A account (tokens returned here).
    #[account(
        mut,
        associated_token::mint = token_a_mint,
        associated_token::authority = maker,
        associated_token::token_program = token_program,
    )]
    pub maker_token_a: InterfaceAccount<'info, TokenAccount>,

    /// The escrow state. `close = maker` returns rent.
    #[account(
        mut,
        close = maker,
        seeds = [ESCROW_SEED, maker.key().as_ref(), &escrow.escrow_id.to_le_bytes()],
        bump = escrow.bump,
        has_one = maker,
        has_one = token_a_mint,
    )]
    pub escrow: Account<'info, EscrowAccount>,

    /// The vault holding token A.
    #[account(
        mut,
        token::mint = token_a_mint,
        token::authority = escrow,
        token::token_program = token_program,
        seeds = [VAULT_SEED, escrow.key().as_ref()],
        bump = escrow.vault_bump,
    )]
    pub vault: InterfaceAccount<'info, TokenAccount>,

    pub token_program: Interface<'info, TokenInterface>,
}
