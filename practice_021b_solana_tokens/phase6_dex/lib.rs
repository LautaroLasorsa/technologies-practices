// =============================================================================
// Phase 6: Mini DEX — Constant Product AMM (x * y = k)
// =============================================================================
//
// WHAT IS AN AMM (Automated Market Maker)?
// ┌─────────────────────────────────────────────────────────────────────┐
// │ Traditional exchange: Order book with bids and asks.               │
// │ Matching engine pairs buyers with sellers.                         │
// │                                                                    │
// │ AMM: No order book. A MATHEMATICAL FORMULA sets the price.         │
// │ Liquidity is held in a POOL (two token vaults).                    │
// │ Anyone can swap by interacting with the pool.                      │
// │                                                                    │
// │ The constant product formula: x * y = k                            │
// │ - x = amount of token A in the pool                                │
// │ - y = amount of token B in the pool                                │
// │ - k = constant (doesn't change during swaps)                       │
// │                                                                    │
// │ Example:                                                           │
// │ Pool has 1000 A and 1000 B → k = 1,000,000                        │
// │ Swap 100 A for B:                                                  │
// │   new_x = 1000 + 100 = 1100                                       │
// │   new_y = k / new_x = 1,000,000 / 1100 = 909.09                  │
// │   tokens_out = 1000 - 909.09 = 90.91 B                            │
// │                                                                    │
// │ Notice: You put in 100 A but only get ~91 B (not 100).            │
// │ This is SLIPPAGE — the price moves against you.                    │
// │ Larger swaps = more slippage. This is by design.                   │
// └─────────────────────────────────────────────────────────────────────┘
//
// LP TOKENS (Liquidity Provider Tokens):
// When you add liquidity, you receive LP tokens proportional to your share
// of the pool. When you remove liquidity, you burn LP tokens and receive
// your proportional share of both tokens.
//
// LP token formula on initial deposit:
//   lp_tokens = sqrt(amount_a * amount_b)
//
// LP token formula on subsequent deposits:
//   lp_tokens = min(amount_a / reserve_a, amount_b / reserve_b) * total_lp_supply
//
// FEES:
// A swap fee (e.g., 0.3%) is deducted from every swap. This fee stays in
// the pool, increasing the value of LP tokens over time. This is how
// liquidity providers earn yield.
//
// ALL MATH IS INTEGER (u64):
// - No floating point — it's non-deterministic across validators
// - Division truncates (rounds down) — this is safe because it slightly
//   favors the pool (not the swapper), preventing drainage attacks
// - Use checked_mul / checked_div to prevent overflow
// - For sqrt, use integer square root (Newton's method or bit manipulation)
//
// REAL-WORLD PROTOCOLS USING THIS:
// - Uniswap V2 (Ethereum) — the OG constant product AMM
// - Raydium CPMM (Solana) — constant product AMM with OpenBook integration
// - Orca Whirlpools (Solana) — concentrated liquidity (Uni V3 style)
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
        self, Burn, Mint, MintTo, TokenAccount, TokenInterface, TransferChecked,
    },
};

declare_id!("MiniDEX1111111111111111111111111111111111111");

/// Swap fee in basis points. 30 bps = 0.3% (same as Uniswap V2).
pub const SWAP_FEE_BPS: u64 = 30;

/// Basis point denominator.
pub const BPS_DENOMINATOR: u64 = 10_000;

/// Seeds for the pool state PDA.
pub const POOL_SEED: &[u8] = b"pool";

/// Seeds for pool vaults.
pub const VAULT_A_SEED: &[u8] = b"vault-a";
pub const VAULT_B_SEED: &[u8] = b"vault-b";

/// Seed for LP token mint.
pub const LP_MINT_SEED: &[u8] = b"lp-mint";

#[program]
pub mod mini_dex {
    use super::*;

    // =========================================================================
    // Exercise 1: Initialize Liquidity Pool
    // =========================================================================
    //
    // Creates:
    // - PoolState PDA (stores pool configuration)
    // - Vault A (PDA token account for token A)
    // - Vault B (PDA token account for token B)
    // - LP Mint (PDA mint for liquidity provider tokens)
    //
    // The pool state PDA is the authority for all vaults and the LP mint.
    // This means only the program can move tokens or mint/burn LP tokens.
    //
    pub fn initialize_pool(ctx: Context<InitializePool>) -> Result<()> {
        // ── Exercise Context ──────────────────────────────────────────────────
        // This exercise teaches AMM pool initialization—creating the vault PDAs and LP token
        // mint that form the foundation of a constant-product market maker. This is how Uniswap
        // V2, Raydium CPMM, and Orca pools work: two token vaults + deterministic pricing.
        //
        // TODO(human): Initialize the pool state.
        //
        //   let pool = &mut ctx.accounts.pool;
        //   pool.token_a_mint = ctx.accounts.token_a_mint.key();
        //   pool.token_b_mint = ctx.accounts.token_b_mint.key();
        //   pool.lp_mint = ctx.accounts.lp_mint.key();
        //   pool.vault_a = ctx.accounts.vault_a.key();
        //   pool.vault_b = ctx.accounts.vault_b.key();
        //   pool.total_lp_supply = 0;
        //   pool.bump = ctx.bumps.pool;
        //   pool.vault_a_bump = ctx.bumps.vault_a;
        //   pool.vault_b_bump = ctx.bumps.vault_b;
        //   pool.lp_mint_bump = ctx.bumps.lp_mint;
        //
        // Return Ok(()) — the vaults and LP mint are created by Anchor constraints.
        todo!("Initialize pool state")
    }

    // =========================================================================
    // Exercise 2: Add Liquidity
    // =========================================================================
    //
    // Liquidity provider deposits both tokens and receives LP tokens.
    //
    // FIRST deposit (pool is empty):
    //   lp_tokens_minted = integer_sqrt(amount_a * amount_b)
    //   (Geometric mean — penalizes unbalanced deposits)
    //
    // SUBSEQUENT deposits:
    //   lp_tokens_minted = min(
    //       amount_a * total_lp_supply / reserve_a,
    //       amount_b * total_lp_supply / reserve_b,
    //   )
    //   This ensures proportional deposits. If you deposit more of one token
    //   than needed, you lose the excess (incentivizes balanced deposits).
    //
    // Why sqrt for initial deposit?
    // - If we used amount_a * amount_b, the LP supply would be enormous
    //   and subsequent LPs would get dust-level tokens.
    // - sqrt normalizes the scale. Uniswap V2 uses this exact formula.
    //
    pub fn add_liquidity(
        ctx: Context<AddLiquidity>,
        amount_a: u64,
        amount_b: u64,
    ) -> Result<()> {
        // TODO(human): Implement add_liquidity.
        //
        // Step 1: Get current reserves from vaults.
        //   let reserve_a = ctx.accounts.vault_a.amount;
        //   let reserve_b = ctx.accounts.vault_b.amount;
        //
        // Step 2: Calculate LP tokens to mint.
        //   let lp_tokens = if ctx.accounts.pool.total_lp_supply == 0 {
        //       // First deposit — use geometric mean
        //       integer_sqrt(
        //           (amount_a as u128)
        //               .checked_mul(amount_b as u128)
        //               .ok_or(ErrorCode::MathOverflow)?
        //       )? as u64
        //   } else {
        //       // Subsequent deposits — proportional to smaller ratio
        //       let lp_a = (amount_a as u128)
        //           .checked_mul(ctx.accounts.pool.total_lp_supply as u128)
        //           .ok_or(ErrorCode::MathOverflow)?
        //           .checked_div(reserve_a as u128)
        //           .ok_or(ErrorCode::MathOverflow)? as u64;
        //       let lp_b = (amount_b as u128)
        //           .checked_mul(ctx.accounts.pool.total_lp_supply as u128)
        //           .ok_or(ErrorCode::MathOverflow)?
        //           .checked_div(reserve_b as u128)
        //           .ok_or(ErrorCode::MathOverflow)? as u64;
        //       lp_a.min(lp_b)
        //   };
        //   require!(lp_tokens > 0, ErrorCode::InsufficientLiquidity);
        //
        // Step 3: Transfer token A from provider to vault_a.
        //   (Regular transfer_checked — provider signs)
        //
        // Step 4: Transfer token B from provider to vault_b.
        //   (Regular transfer_checked — provider signs)
        //
        // Step 5: Mint LP tokens to provider's LP token account.
        //   (PDA-signed — pool state PDA is LP mint authority)
        //
        //   let signer_seeds = pool_signer_seeds!(ctx);
        //   let cpi_accounts = MintTo {
        //       mint: ctx.accounts.lp_mint.to_account_info(),
        //       to: ctx.accounts.provider_lp_account.to_account_info(),
        //       authority: ctx.accounts.pool.to_account_info(),
        //   };
        //   token_interface::mint_to(
        //       CpiContext::new(...).with_signer(signer_seeds),
        //       lp_tokens,
        //   )?;
        //
        // Step 6: Update pool state.
        //   ctx.accounts.pool.total_lp_supply += lp_tokens;
        //
        // Hint: Use u128 for intermediate calculations to avoid overflow.
        // 1000 A * 1000 B = 1,000,000 — fits in u64.
        // But larger amounts could overflow u64 during multiplication.
        todo!("Calculate LP tokens, transfer both tokens to vaults, mint LP tokens")
    }

    // =========================================================================
    // Exercise 3: Swap
    // =========================================================================
    //
    // The core AMM operation. User provides token A, receives token B
    // (or vice versa).
    //
    // CONSTANT PRODUCT FORMULA:
    //   (reserve_a + amount_in_after_fee) * (reserve_b - amount_out) = k
    //
    // Solving for amount_out:
    //   amount_in_after_fee = amount_in * (10000 - SWAP_FEE_BPS) / 10000
    //   amount_out = reserve_b * amount_in_after_fee / (reserve_a + amount_in_after_fee)
    //
    // The fee (0.3%) stays in the pool, increasing reserve value for LPs.
    //
    // SLIPPAGE PROTECTION:
    // The `min_amount_out` parameter prevents front-running attacks.
    // If the actual output is less than min_amount_out, the tx fails.
    // Without this, a MEV bot could sandwich your trade:
    //   1. Bot buys before you (price goes up)
    //   2. Your trade executes at worse price
    //   3. Bot sells after you (profits from the difference)
    //
    pub fn swap(
        ctx: Context<Swap>,
        amount_in: u64,
        min_amount_out: u64,
        a_to_b: bool, // true = swap A for B, false = swap B for A
    ) -> Result<()> {
        // TODO(human): Implement the swap.
        //
        // Step 1: Determine input/output vaults and mints.
        //
        //   let (reserve_in, reserve_out) = if a_to_b {
        //       (ctx.accounts.vault_a.amount, ctx.accounts.vault_b.amount)
        //   } else {
        //       (ctx.accounts.vault_b.amount, ctx.accounts.vault_a.amount)
        //   };
        //
        // Step 2: Calculate output amount using constant product formula.
        //
        //   // Apply fee: amount_in_after_fee = amount_in * (10000 - 30) / 10000
        //   let amount_in_after_fee = (amount_in as u128)
        //       .checked_mul((BPS_DENOMINATOR - SWAP_FEE_BPS) as u128)
        //       .ok_or(ErrorCode::MathOverflow)?
        //       .checked_div(BPS_DENOMINATOR as u128)
        //       .ok_or(ErrorCode::MathOverflow)?;
        //
        //   // amount_out = reserve_out * amount_in_after_fee / (reserve_in + amount_in_after_fee)
        //   let numerator = (reserve_out as u128)
        //       .checked_mul(amount_in_after_fee)
        //       .ok_or(ErrorCode::MathOverflow)?;
        //   let denominator = (reserve_in as u128)
        //       .checked_add(amount_in_after_fee)
        //       .ok_or(ErrorCode::MathOverflow)?;
        //   let amount_out = numerator
        //       .checked_div(denominator)
        //       .ok_or(ErrorCode::MathOverflow)? as u64;
        //
        // Step 3: Check slippage protection.
        //   require!(amount_out >= min_amount_out, ErrorCode::SlippageExceeded);
        //   require!(amount_out > 0, ErrorCode::ZeroOutput);
        //
        // Step 4: Transfer input tokens from user to input vault.
        //   (Regular transfer — user signs)
        //
        // Step 5: Transfer output tokens from output vault to user.
        //   (PDA-signed — pool state PDA is vault authority)
        //
        //   Build signer_seeds from pool seeds:
        //   let token_a_key = ctx.accounts.pool.token_a_mint;
        //   let token_b_key = ctx.accounts.pool.token_b_mint;
        //   let signer_seeds: &[&[&[u8]]] = &[&[
        //       POOL_SEED,
        //       token_a_key.as_ref(),
        //       token_b_key.as_ref(),
        //       &[ctx.accounts.pool.bump],
        //   ]];
        //
        // Note: The fee is NOT transferred anywhere — it simply stays in the
        // input vault (the input amount is fully deposited, but the output
        // is calculated AFTER deducting the fee). This implicitly increases
        // the pool's k value, benefiting LP holders.
        //
        // Question: Why does integer division (truncation) favor the pool?
        // (Answer: amount_out is rounded DOWN, so the pool keeps slightly
        //  more tokens than the formula requires. Over many swaps, this
        //  slightly grows k, making the pool more resistant to drainage.)
        todo!("Calculate output, check slippage, transfer tokens")
    }
}

// =============================================================================
// Helper: Integer Square Root
// =============================================================================
//
// Newton's method for computing floor(sqrt(n)) in pure integer arithmetic.
// Used for calculating initial LP token supply.
//
// This is the same algorithm Uniswap V2 uses.
//
pub fn integer_sqrt(n: u128) -> Result<u128> {
    // TODO(human): Implement integer square root.
    //
    // Newton's method:
    //   Start with x = n
    //   Iterate: x = (x + n/x) / 2
    //   Stop when x doesn't change (convergence)
    //
    // Implementation:
    //   if n == 0 { return Ok(0); }
    //   let mut x = n;
    //   let mut y = (x + 1) / 2;
    //   while y < x {
    //       x = y;
    //       y = (x + n / x) / 2;
    //   }
    //   Ok(x)
    //
    // Why this works:
    // - Newton's method for f(x) = x² - n converges quadratically
    // - Each iteration approximately doubles the number of correct digits
    // - For u128, convergence takes at most ~64 iterations (usually ~7-10)
    //
    // Example: sqrt(1_000_000) = 1000
    //   Iteration 1: x = 500001, y = 250001
    //   Iteration 2: x = 250001, y = 125003
    //   ... (converges to 1000)
    //
    // CP analogy: This is like binary search on the answer —
    // but Newton's method converges faster (quadratic vs linear).
    todo!("Implement integer_sqrt using Newton's method")
}

// =============================================================================
// State
// =============================================================================

/// The liquidity pool state. One per token pair.
#[account]
#[derive(InitSpace)]
pub struct PoolState {
    /// Mint of token A.
    pub token_a_mint: Pubkey,    // 32
    /// Mint of token B.
    pub token_b_mint: Pubkey,    // 32
    /// LP token mint (minted when adding liquidity).
    pub lp_mint: Pubkey,         // 32
    /// Vault holding token A reserves.
    pub vault_a: Pubkey,         // 32
    /// Vault holding token B reserves.
    pub vault_b: Pubkey,         // 32
    /// Total LP tokens in circulation.
    pub total_lp_supply: u64,    // 8
    /// Bump for the pool PDA.
    pub bump: u8,                // 1
    /// Bump for vault A PDA.
    pub vault_a_bump: u8,        // 1
    /// Bump for vault B PDA.
    pub vault_b_bump: u8,        // 1
    /// Bump for LP mint PDA.
    pub lp_mint_bump: u8,        // 1
}

// =============================================================================
// Account Validation Structs
// =============================================================================

/// Initialize a new liquidity pool for a token pair.
#[derive(Accounts)]
pub struct InitializePool<'info> {
    #[account(mut)]
    pub creator: Signer<'info>,

    /// Token A mint.
    pub token_a_mint: InterfaceAccount<'info, Mint>,

    /// Token B mint.
    pub token_b_mint: InterfaceAccount<'info, Mint>,

    /// Pool state PDA. Seeds: ["pool", token_a_mint, token_b_mint]
    #[account(
        init,
        payer = creator,
        space = 8 + PoolState::INIT_SPACE,
        seeds = [POOL_SEED, token_a_mint.key().as_ref(), token_b_mint.key().as_ref()],
        bump,
    )]
    pub pool: Account<'info, PoolState>,

    /// Vault for token A. Authority = pool PDA.
    #[account(
        init,
        payer = creator,
        token::mint = token_a_mint,
        token::authority = pool,
        token::token_program = token_program,
        seeds = [VAULT_A_SEED, pool.key().as_ref()],
        bump,
    )]
    pub vault_a: InterfaceAccount<'info, TokenAccount>,

    /// Vault for token B. Authority = pool PDA.
    #[account(
        init,
        payer = creator,
        token::mint = token_b_mint,
        token::authority = pool,
        token::token_program = token_program,
        seeds = [VAULT_B_SEED, pool.key().as_ref()],
        bump,
    )]
    pub vault_b: InterfaceAccount<'info, TokenAccount>,

    /// LP token mint. Authority = pool PDA.
    /// Decimals = 6 (arbitrary choice, could match token A or B).
    #[account(
        init,
        payer = creator,
        mint::decimals = 6,
        mint::authority = pool,
        mint::freeze_authority = pool,
        seeds = [LP_MINT_SEED, pool.key().as_ref()],
        bump,
    )]
    pub lp_mint: InterfaceAccount<'info, Mint>,

    pub token_program: Interface<'info, TokenInterface>,
    pub system_program: Program<'info, System>,
}

/// Add liquidity to the pool.
#[derive(Accounts)]
pub struct AddLiquidity<'info> {
    #[account(mut)]
    pub provider: Signer<'info>,

    /// Pool state.
    #[account(
        mut,
        seeds = [POOL_SEED, pool.token_a_mint.as_ref(), pool.token_b_mint.as_ref()],
        bump = pool.bump,
    )]
    pub pool: Account<'info, PoolState>,

    /// Token A mint (needed for transfer_checked).
    pub token_a_mint: InterfaceAccount<'info, Mint>,

    /// Token B mint (needed for transfer_checked).
    pub token_b_mint: InterfaceAccount<'info, Mint>,

    /// Provider's token A account (deposits from here).
    #[account(
        mut,
        associated_token::mint = token_a_mint,
        associated_token::authority = provider,
        associated_token::token_program = token_program,
    )]
    pub provider_token_a: InterfaceAccount<'info, TokenAccount>,

    /// Provider's token B account (deposits from here).
    #[account(
        mut,
        associated_token::mint = token_b_mint,
        associated_token::authority = provider,
        associated_token::token_program = token_program,
    )]
    pub provider_token_b: InterfaceAccount<'info, TokenAccount>,

    /// Provider's LP token account (LP tokens minted here).
    #[account(
        init_if_needed,
        payer = provider,
        associated_token::mint = lp_mint,
        associated_token::authority = provider,
        associated_token::token_program = token_program,
    )]
    pub provider_lp_account: InterfaceAccount<'info, TokenAccount>,

    /// Vault A (receives token A).
    #[account(
        mut,
        token::mint = token_a_mint,
        token::authority = pool,
        token::token_program = token_program,
        seeds = [VAULT_A_SEED, pool.key().as_ref()],
        bump = pool.vault_a_bump,
    )]
    pub vault_a: InterfaceAccount<'info, TokenAccount>,

    /// Vault B (receives token B).
    #[account(
        mut,
        token::mint = token_b_mint,
        token::authority = pool,
        token::token_program = token_program,
        seeds = [VAULT_B_SEED, pool.key().as_ref()],
        bump = pool.vault_b_bump,
    )]
    pub vault_b: InterfaceAccount<'info, TokenAccount>,

    /// LP mint (pool PDA is authority).
    #[account(
        mut,
        mint::authority = pool,
        seeds = [LP_MINT_SEED, pool.key().as_ref()],
        bump = pool.lp_mint_bump,
    )]
    pub lp_mint: InterfaceAccount<'info, Mint>,

    pub token_program: Interface<'info, TokenInterface>,
    pub associated_token_program: Program<'info, AssociatedToken>,
    pub system_program: Program<'info, System>,
}

/// Swap tokens using the constant product formula.
#[derive(Accounts)]
pub struct Swap<'info> {
    #[account(mut)]
    pub user: Signer<'info>,

    /// Pool state.
    #[account(
        mut,
        seeds = [POOL_SEED, pool.token_a_mint.as_ref(), pool.token_b_mint.as_ref()],
        bump = pool.bump,
    )]
    pub pool: Account<'info, PoolState>,

    /// Token A mint.
    pub token_a_mint: InterfaceAccount<'info, Mint>,

    /// Token B mint.
    pub token_b_mint: InterfaceAccount<'info, Mint>,

    /// User's token A account.
    #[account(
        mut,
        associated_token::mint = token_a_mint,
        associated_token::authority = user,
        associated_token::token_program = token_program,
    )]
    pub user_token_a: InterfaceAccount<'info, TokenAccount>,

    /// User's token B account.
    #[account(
        mut,
        associated_token::mint = token_b_mint,
        associated_token::authority = user,
        associated_token::token_program = token_program,
    )]
    pub user_token_b: InterfaceAccount<'info, TokenAccount>,

    /// Vault A.
    #[account(
        mut,
        token::mint = token_a_mint,
        token::authority = pool,
        token::token_program = token_program,
        seeds = [VAULT_A_SEED, pool.key().as_ref()],
        bump = pool.vault_a_bump,
    )]
    pub vault_a: InterfaceAccount<'info, TokenAccount>,

    /// Vault B.
    #[account(
        mut,
        token::mint = token_b_mint,
        token::authority = pool,
        token::token_program = token_program,
        seeds = [VAULT_B_SEED, pool.key().as_ref()],
        bump = pool.vault_b_bump,
    )]
    pub vault_b: InterfaceAccount<'info, TokenAccount>,

    pub token_program: Interface<'info, TokenInterface>,
}

// =============================================================================
// Errors
// =============================================================================

#[error_code]
pub enum ErrorCode {
    #[msg("Arithmetic overflow")]
    MathOverflow,

    #[msg("Output amount is less than minimum (slippage exceeded)")]
    SlippageExceeded,

    #[msg("Swap would produce zero output")]
    ZeroOutput,

    #[msg("Must provide non-zero liquidity")]
    InsufficientLiquidity,

    #[msg("Pool reserves are empty — cannot swap")]
    EmptyPool,
}
