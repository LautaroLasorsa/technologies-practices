// =============================================================================
// Phase 4: PDA Vaults & Treasury — Staking, Rewards, Fee Collection
// =============================================================================
//
// KEY CONCEPT: PDA Vaults
// ┌─────────────────────────────────────────────────────────────────────┐
// │ A "vault" is a Token Account whose authority is a PDA.             │
// │ This means only the PROGRAM can move tokens in/out of it.          │
// │                                                                    │
// │ User deposits tokens → vault PDA                                   │
// │ Program logic decides when to release → user gets tokens back      │
// │                                                                    │
// │ This is the foundation of ALL DeFi:                                │
// │ - Staking: deposit tokens → earn rewards over time                 │
// │ - Lending: deposit collateral → borrow against it                  │
// │ - Yield farming: deposit LP tokens → earn extra tokens             │
// │ - Treasury: program collects fees from operations                  │
// └─────────────────────────────────────────────────────────────────────┘
//
// STAKING PATTERN:
// 1. User calls `stake(amount)` → tokens move to vault, receipt PDA created
// 2. Time passes (rewards accrue based on unix_timestamp)
// 3. User calls `unstake` → program calculates rewards, returns tokens + reward
//
// The "receipt" is a PDA account that records:
// - Who staked
// - How much
// - When (timestamp)
// This receipt is the user's proof of deposit.
//
// REWARD CALCULATION:
// Simple linear model: reward = staked_amount * rate_per_second * seconds_staked
// All math in u64 — no floating point! Use basis points (1 bp = 0.01%).
//
// Real-world reference:
// - Marinade Finance (mSOL): Users deposit SOL, receive mSOL receipt token.
//   The exchange rate increases over time as staking rewards accrue.
// - Jito (jitoSOL): Same pattern with MEV rewards added.
//
// TREASURY PATTERN:
// A treasury is just a vault that collects fees. The program can charge a fee
// on operations (e.g., 0.3% on swaps) and deposit it into the treasury PDA.
// Only authorized admins can withdraw from the treasury.
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
        self, Mint, TokenAccount, TokenInterface, TransferChecked,
    },
};

declare_id!("Vault111111111111111111111111111111111111111");

pub const VAULT_SEED: &[u8] = b"vault";
pub const STAKE_RECEIPT_SEED: &[u8] = b"stake-receipt";
pub const TREASURY_SEED: &[u8] = b"treasury";

/// Reward rate: 1 basis point per hour = 0.01% per hour.
/// In production, this would be configurable and much more sophisticated.
/// Using basis points (1 bp = 0.01%) avoids floating point.
///
/// Calculation: reward = staked_amount * REWARD_RATE_BPS * hours_staked / 10_000
/// The / 10_000 converts basis points back to a fraction.
pub const REWARD_RATE_BPS_PER_HOUR: u64 = 1; // 0.01% per hour

#[program]
pub mod staking_vault {
    use super::*;

    // =========================================================================
    // Exercise 1: Initialize the staking vault and treasury
    // =========================================================================
    //
    // Creates the vault token account (where staked tokens go) and the
    // treasury token account (where fees accumulate).
    // Both are PDAs — the program controls them.
    //
    pub fn initialize(ctx: Context<Initialize>) -> Result<()> {
        // TODO(human): Initialize the vault config.
        //
        //   let config = &mut ctx.accounts.vault_config;
        //   config.admin = ctx.accounts.admin.key();
        //   config.mint = ctx.accounts.mint.key();
        //   config.total_staked = 0;
        //   config.vault_bump = ctx.bumps.vault;
        //   config.treasury_bump = ctx.bumps.treasury;
        //   config.config_bump = ctx.bumps.vault_config;
        //
        // The vault and treasury token accounts are created by Anchor's init
        // constraint — no additional logic needed for them.
        //
        // Hint: Just save state + return Ok(()).
        todo!("Initialize vault config state")
    }

    // =========================================================================
    // Exercise 2: Stake tokens
    // =========================================================================
    //
    // User deposits tokens into the vault and receives a "stake receipt" PDA
    // that records the deposit details.
    //
    // Flow:
    // 1. Transfer tokens from user's ATA to vault
    // 2. Create a StakeReceipt PDA with amount, timestamp, owner
    // 3. Update total_staked in vault config
    //
    pub fn stake(ctx: Context<Stake>, amount: u64) -> Result<()> {
        // TODO(human): Implement staking in three steps.
        //
        // Step 1: Transfer tokens from staker to vault.
        //
        //   let transfer_accounts = TransferChecked {
        //       mint: ctx.accounts.mint.to_account_info(),
        //       from: ctx.accounts.staker_token_account.to_account_info(),
        //       to: ctx.accounts.vault.to_account_info(),
        //       authority: ctx.accounts.staker.to_account_info(),
        //   };
        //   let cpi_ctx = CpiContext::new(
        //       ctx.accounts.token_program.to_account_info(),
        //       transfer_accounts,
        //   );
        //   token_interface::transfer_checked(
        //       cpi_ctx, amount, ctx.accounts.mint.decimals
        //   )?;
        //
        // Step 2: Record the stake in the receipt PDA.
        //
        //   let receipt = &mut ctx.accounts.stake_receipt;
        //   receipt.owner = ctx.accounts.staker.key();
        //   receipt.amount = amount;
        //   receipt.staked_at = Clock::get()?.unix_timestamp;
        //   receipt.bump = ctx.bumps.stake_receipt;
        //
        // Step 3: Update vault config.
        //
        //   let config = &mut ctx.accounts.vault_config;
        //   config.total_staked = config.total_staked
        //       .checked_add(amount)
        //       .ok_or(ErrorCode::MathOverflow)?;
        //
        // Question: Why use checked_add instead of plain +?
        // (Answer: u64 overflow would silently wrap around.
        //  checked_add returns None on overflow, which we convert to an error.
        //  On Solana, this is CRITICAL — an overflow bug could let attackers
        //  mint infinite tokens. See: Wormhole hack, $325M lost.)
        todo!("Transfer to vault, create receipt, update config")
    }

    // =========================================================================
    // Exercise 3: Unstake tokens + claim rewards
    // =========================================================================
    //
    // User redeems their stake receipt for original tokens + accrued rewards.
    //
    // Reward calculation (simple linear model):
    //   seconds_staked = now - staked_at
    //   hours_staked = seconds_staked / 3600
    //   reward = amount * REWARD_RATE_BPS_PER_HOUR * hours_staked / 10_000
    //
    // The vault must have enough tokens to cover the reward. In production,
    // you'd have a separate reward pool funded by protocol revenue.
    //
    // IMPORTANT: The vault signs the outgoing transfer (PDA signer seeds).
    //
    pub fn unstake(ctx: Context<Unstake>) -> Result<()> {
        // TODO(human): Implement unstaking with rewards.
        //
        // Step 1: Calculate rewards.
        //
        //   let receipt = &ctx.accounts.stake_receipt;
        //   let now = Clock::get()?.unix_timestamp;
        //   let seconds_staked = (now - receipt.staked_at) as u64;
        //   let hours_staked = seconds_staked / 3600;
        //
        //   let reward = receipt.amount
        //       .checked_mul(REWARD_RATE_BPS_PER_HOUR)?
        //       .checked_mul(hours_staked)?
        //       .checked_div(10_000)
        //       .unwrap_or(0);
        //
        //   let total_payout = receipt.amount
        //       .checked_add(reward)
        //       .ok_or(ErrorCode::MathOverflow)?;
        //
        // Step 2: Verify vault has enough tokens.
        //
        //   require!(
        //       ctx.accounts.vault.amount >= total_payout,
        //       ErrorCode::InsufficientVaultBalance
        //   );
        //
        // Step 3: Transfer from vault to staker (PDA-signed).
        //
        //   Build signer_seeds for the vault config PDA (which is the vault's authority).
        //   The vault token account's authority is vault_config.
        //
        //   let mint_key = ctx.accounts.mint.key();
        //   let signer_seeds: &[&[&[u8]]] = &[&[
        //       VAULT_SEED,
        //       mint_key.as_ref(),
        //       &[ctx.accounts.vault_config.vault_bump],
        //   ]];
        //
        //   (Transfer total_payout from vault to staker_token_account)
        //
        // Step 4: Update vault config.
        //
        //   config.total_staked = config.total_staked
        //       .checked_sub(receipt.amount)
        //       .ok_or(ErrorCode::MathOverflow)?;
        //
        // The stake_receipt account is closed by Anchor's `close = staker`.
        //
        // Question: What happens if many users stake but the vault doesn't have
        // enough rewards tokens? This is the "bank run" problem. Real protocols
        // solve this by:
        // 1. Minting reward tokens (inflationary model — Marinade)
        // 2. Using protocol revenue as rewards (fee model — Jito)
        // 3. Capping APY based on available rewards
        todo!("Calculate reward, transfer from vault, update config")
    }

    // =========================================================================
    // Exercise 4: Collect fees into treasury
    // =========================================================================
    //
    // Any instruction can call this to collect fees. The treasury PDA
    // accumulates fees that can later be withdrawn by the admin.
    //
    // In a real protocol, you'd call this during swaps, mints, etc.:
    //   let fee = amount * FEE_BPS / 10_000;
    //   collect_fee(fee)?;
    //
    pub fn collect_fee(ctx: Context<CollectFee>, amount: u64) -> Result<()> {
        // TODO(human): Transfer fee from user to treasury.
        //
        // This is a standard transfer_checked CPI — no PDA signing needed
        // because the user (fee payer) signs the transfer.
        //
        //   let transfer_accounts = TransferChecked {
        //       mint: ctx.accounts.mint.to_account_info(),
        //       from: ctx.accounts.payer_token_account.to_account_info(),
        //       to: ctx.accounts.treasury.to_account_info(),
        //       authority: ctx.accounts.payer.to_account_info(),
        //   };
        //   (Create CpiContext, call transfer_checked)
        //
        // In a real protocol, the fee collection would be embedded inside
        // other instructions (swap, borrow, etc.), not a standalone instruction.
        // We separate it here for clarity.
        todo!("Transfer fee from payer to treasury")
    }
}

// =============================================================================
// State
// =============================================================================

/// Global vault configuration. One per token mint.
#[account]
#[derive(InitSpace)]
pub struct VaultConfig {
    /// Admin who can withdraw from treasury.
    pub admin: Pubkey,       // 32
    /// The token mint this vault accepts.
    pub mint: Pubkey,        // 32
    /// Total tokens currently staked.
    pub total_staked: u64,   // 8
    /// Bump for the vault token account PDA.
    pub vault_bump: u8,      // 1
    /// Bump for the treasury token account PDA.
    pub treasury_bump: u8,   // 1
    /// Bump for this config account PDA.
    pub config_bump: u8,     // 1
}

/// A receipt proving a user has staked tokens.
/// One per staker per staking action.
#[account]
#[derive(InitSpace)]
pub struct StakeReceipt {
    /// The staker's wallet.
    pub owner: Pubkey,       // 32
    /// Amount of tokens staked.
    pub amount: u64,         // 8
    /// Unix timestamp when staked (for reward calculation).
    pub staked_at: i64,      // 8
    /// Bump for this receipt PDA.
    pub bump: u8,            // 1
}

// =============================================================================
// Account Validation Structs
// =============================================================================

/// Initialize the vault, treasury, and config.
#[derive(Accounts)]
pub struct Initialize<'info> {
    #[account(mut)]
    pub admin: Signer<'info>,

    /// The token mint this vault will accept.
    pub mint: InterfaceAccount<'info, Mint>,

    /// Vault config (PDA). Stores global staking state.
    #[account(
        init,
        payer = admin,
        space = 8 + VaultConfig::INIT_SPACE,
        seeds = [b"config", mint.key().as_ref()],
        bump,
    )]
    pub vault_config: Account<'info, VaultConfig>,

    /// Vault token account. Authority = vault_config PDA.
    /// This is where staked tokens are held.
    #[account(
        init,
        payer = admin,
        token::mint = mint,
        token::authority = vault_config,
        token::token_program = token_program,
        seeds = [VAULT_SEED, mint.key().as_ref()],
        bump,
    )]
    pub vault: InterfaceAccount<'info, TokenAccount>,

    /// Treasury token account. Authority = vault_config PDA.
    /// This is where collected fees accumulate.
    #[account(
        init,
        payer = admin,
        token::mint = mint,
        token::authority = vault_config,
        token::token_program = token_program,
        seeds = [TREASURY_SEED, mint.key().as_ref()],
        bump,
    )]
    pub treasury: InterfaceAccount<'info, TokenAccount>,

    pub token_program: Interface<'info, TokenInterface>,
    pub system_program: Program<'info, System>,
}

/// Stake tokens into the vault.
#[derive(Accounts)]
pub struct Stake<'info> {
    #[account(mut)]
    pub staker: Signer<'info>,

    pub mint: InterfaceAccount<'info, Mint>,

    /// Vault config (tracks total_staked).
    #[account(
        mut,
        seeds = [b"config", mint.key().as_ref()],
        bump = vault_config.config_bump,
        has_one = mint,
    )]
    pub vault_config: Account<'info, VaultConfig>,

    /// Staker's token account (tokens come from here).
    #[account(
        mut,
        associated_token::mint = mint,
        associated_token::authority = staker,
        associated_token::token_program = token_program,
    )]
    pub staker_token_account: InterfaceAccount<'info, TokenAccount>,

    /// The vault (tokens go here).
    #[account(
        mut,
        token::mint = mint,
        token::authority = vault_config,
        token::token_program = token_program,
        seeds = [VAULT_SEED, mint.key().as_ref()],
        bump = vault_config.vault_bump,
    )]
    pub vault: InterfaceAccount<'info, TokenAccount>,

    /// Stake receipt PDA. Created for this staker.
    /// Seeds: ["stake-receipt", staker_pubkey, mint_pubkey]
    /// This means one receipt per (staker, mint) pair.
    /// For multiple stakes, you'd add a nonce/counter.
    #[account(
        init,
        payer = staker,
        space = 8 + StakeReceipt::INIT_SPACE,
        seeds = [STAKE_RECEIPT_SEED, staker.key().as_ref(), mint.key().as_ref()],
        bump,
    )]
    pub stake_receipt: Account<'info, StakeReceipt>,

    pub token_program: Interface<'info, TokenInterface>,
    pub system_program: Program<'info, System>,
}

/// Unstake tokens and claim rewards.
#[derive(Accounts)]
pub struct Unstake<'info> {
    #[account(mut)]
    pub staker: Signer<'info>,

    pub mint: InterfaceAccount<'info, Mint>,

    /// Vault config.
    #[account(
        mut,
        seeds = [b"config", mint.key().as_ref()],
        bump = vault_config.config_bump,
        has_one = mint,
    )]
    pub vault_config: Account<'info, VaultConfig>,

    /// Staker's token account (tokens returned here).
    #[account(
        mut,
        associated_token::mint = mint,
        associated_token::authority = staker,
        associated_token::token_program = token_program,
    )]
    pub staker_token_account: InterfaceAccount<'info, TokenAccount>,

    /// The vault (tokens come from here).
    #[account(
        mut,
        token::mint = mint,
        token::authority = vault_config,
        token::token_program = token_program,
        seeds = [VAULT_SEED, mint.key().as_ref()],
        bump = vault_config.vault_bump,
    )]
    pub vault: InterfaceAccount<'info, TokenAccount>,

    /// Stake receipt (proves the staker deposited tokens).
    /// `close = staker` → rent returned to staker when receipt is consumed.
    #[account(
        mut,
        close = staker,
        seeds = [STAKE_RECEIPT_SEED, staker.key().as_ref(), mint.key().as_ref()],
        bump = stake_receipt.bump,
        has_one = owner @ ErrorCode::NotStakeOwner,
    )]
    pub stake_receipt: Account<'info, StakeReceipt>,

    pub token_program: Interface<'info, TokenInterface>,
}

/// Collect fees into the treasury.
#[derive(Accounts)]
pub struct CollectFee<'info> {
    #[account(mut)]
    pub payer: Signer<'info>,

    pub mint: InterfaceAccount<'info, Mint>,

    /// Payer's token account (fee comes from here).
    #[account(
        mut,
        associated_token::mint = mint,
        associated_token::authority = payer,
        associated_token::token_program = token_program,
    )]
    pub payer_token_account: InterfaceAccount<'info, TokenAccount>,

    /// Treasury token account (fees accumulate here).
    #[account(
        mut,
        token::mint = mint,
        token::authority = vault_config,
        token::token_program = token_program,
        seeds = [TREASURY_SEED, mint.key().as_ref()],
        bump = vault_config.treasury_bump,
    )]
    pub treasury: InterfaceAccount<'info, TokenAccount>,

    /// Vault config (needed to verify treasury seeds).
    #[account(
        seeds = [b"config", mint.key().as_ref()],
        bump = vault_config.config_bump,
    )]
    pub vault_config: Account<'info, VaultConfig>,

    pub token_program: Interface<'info, TokenInterface>,
}

// =============================================================================
// Errors
// =============================================================================

#[error_code]
pub enum ErrorCode {
    #[msg("Arithmetic overflow")]
    MathOverflow,

    #[msg("Vault does not have enough tokens for payout")]
    InsufficientVaultBalance,

    #[msg("Only the original staker can unstake")]
    NotStakeOwner,
}
