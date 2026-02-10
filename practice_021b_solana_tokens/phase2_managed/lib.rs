// =============================================================================
// Phase 2: Managed Token — PDA as Mint Authority & Access Control
// =============================================================================
//
// KEY CONCEPT: PDA as Mint Authority
// ┌─────────────────────────────────────────────────────────────────────┐
// │ Regular mint: mint_authority = some_wallet_keypair                  │
// │   → Anyone with the private key can mint (risky if key leaks)      │
// │                                                                    │
// │ PDA-controlled mint: mint_authority = PDA("mint-authority", [seed]) │
// │   → ONLY the program can mint (no private key exists for PDAs)     │
// │   → Program logic decides WHEN and HOW MUCH to mint                │
// │   → This is how real protocols work:                               │
// │     - Marinade's mSOL mint authority = Marinade program PDA        │
// │     - Raydium LP token mint authority = Raydium program PDA        │
// │                                                                    │
// │ Analogy: PDA authority = the program is the central bank.          │
// │ Only the program's code can authorize new money printing.          │
// └─────────────────────────────────────────────────────────────────────┘
//
// SIGNING WITH PDAs:
// PDAs have no private key, so they can't sign normally. Instead, the program
// uses `CpiContext::new(...).with_signer(signer_seeds)` to prove it "owns" the
// PDA. The runtime verifies: seeds + bump → PDA address matches.
//
// BURN:
// Burning = decreasing supply. The token account owner can burn their own tokens.
// Think of it as the inverse of minting.
//
// AIRDROP (remaining_accounts pattern):
// Anchor's `remaining_accounts` lets you pass a variable number of accounts.
// This is useful for airdropping to N recipients without defining N account fields.
// Real protocols use this for batch operations, whitelist mints, etc.
//
// DEPENDENCIES (Cargo.toml):
// ```toml
// [dependencies]
// anchor-lang = { version = "0.31", features = ["init-if-needed"] }
// anchor-spl = { version = "0.31", features = ["token", "associated_token"] }
// ```

use anchor_lang::prelude::*;
use anchor_spl::token_interface::{
    self, Burn, Mint, MintTo, TokenAccount, TokenInterface,
};

declare_id!("ManagedToken1111111111111111111111111111111");

/// Seeds for the mint authority PDA.
/// The PDA derived from these seeds will be the ONLY entity that can mint.
pub const MINT_AUTHORITY_SEED: &[u8] = b"mint-authority";

#[program]
pub mod managed_token {
    use super::*;

    // =========================================================================
    // Exercise 1: Initialize a managed mint (PDA = mint authority)
    // =========================================================================
    //
    // The mint is created with `mint::authority = mint_authority_pda.key()`.
    // This means NO wallet can mint — only our program, by signing with the
    // PDA's seeds.
    //
    // This is the "program as central bank" pattern.
    //
    pub fn initialize_managed_mint(_ctx: Context<InitializeManagedMint>) -> Result<()> {
        // TODO(human): Return Ok(()).
        // Anchor's init constraint handles everything. But verify you understand:
        // - The mint_authority PDA is derived from seeds = [MINT_AUTHORITY_SEED]
        // - This PDA is set as the mint's authority in the constraint
        // - No wallet keypair can mint tokens — only the program via CPI
        //
        // Why is this safer? Because:
        // 1. No private key exists for PDAs — can't be leaked/stolen
        // 2. Minting logic is enforced by program code — auditable, deterministic
        // 3. You can add arbitrary conditions: max supply, time locks, governance votes
        todo!("Return Ok(())")
    }

    // =========================================================================
    // Exercise 2: Mint tokens using PDA signer seeds
    // =========================================================================
    //
    // Since the mint authority is a PDA, we can't sign with a keypair.
    // Instead, we use `CpiContext::new(...).with_signer(signer_seeds)`.
    //
    // The runtime checks: do these seeds + bump produce the PDA address?
    // If yes, the CPI is authorized. If no, it fails.
    //
    // Pattern:
    //   let signer_seeds: &[&[&[u8]]] = &[&[SEED, &[bump]]];
    //   let cpi_ctx = CpiContext::new(program, accounts).with_signer(signer_seeds);
    //
    pub fn managed_mint(ctx: Context<ManagedMint>, amount: u64) -> Result<()> {
        // TODO(human): Implement PDA-signed minting.
        //
        // Steps:
        // 1. Get the bump from ctx.bumps.mint_authority
        //
        // 2. Build signer seeds:
        //    let signer_seeds: &[&[&[u8]]] = &[&[
        //        MINT_AUTHORITY_SEED,
        //        &[ctx.bumps.mint_authority],
        //    ]];
        //
        // 3. Create MintTo accounts struct:
        //    MintTo {
        //        mint: ctx.accounts.mint.to_account_info(),
        //        to: ctx.accounts.recipient_token_account.to_account_info(),
        //        authority: ctx.accounts.mint_authority.to_account_info(),
        //    }
        //
        // 4. Create CpiContext with .with_signer(signer_seeds)
        //
        // 5. Call token_interface::mint_to(cpi_ctx, amount)?;
        //
        // Key difference from Phase 1: authority is the PDA, not the signer!
        // The signer (admin) triggers the instruction, but the PDA authorizes the mint.
        //
        // Hint: The with_signer call is what makes PDA signing work.
        // Without it, the Token Program would reject the CPI because
        // "the authority didn't sign the transaction."
        let _bump = ctx.bumps.mint_authority;
        todo!("Build signer_seeds, CpiContext with_signer, call mint_to")
    }

    // =========================================================================
    // Exercise 3: Burn tokens
    // =========================================================================
    //
    // Burning decreases supply. The TOKEN ACCOUNT OWNER (not mint authority)
    // can burn their own tokens. Think of it as "I'm destroying my own money."
    //
    // Real-world uses:
    // - Deflationary tokens (burn on transfer)
    // - Redeeming tokens (burn staking receipt to unstake)
    // - NFT burning (destroy to claim reward)
    //
    pub fn burn_tokens(ctx: Context<BurnTokens>, amount: u64) -> Result<()> {
        // TODO(human): Implement burn CPI.
        //
        // Steps:
        // 1. Create a Burn struct:
        //    Burn {
        //        mint: ctx.accounts.mint.to_account_info(),
        //        from: ctx.accounts.token_account.to_account_info(),
        //        authority: ctx.accounts.owner.to_account_info(),
        //    }
        //
        // 2. Create CpiContext::new(token_program, burn_accounts)
        //
        // 3. Call token_interface::burn(cpi_ctx, amount)?;
        //
        // Note: No PDA signing needed here — the owner is a regular Signer.
        // The owner is burning their OWN tokens, so they sign directly.
        todo!("Create Burn struct, CpiContext, call burn")
    }

    // =========================================================================
    // Exercise 4: Airdrop to multiple recipients
    // =========================================================================
    //
    // Airdrop = mint tokens to multiple wallets in one transaction.
    //
    // Pattern: Use `ctx.remaining_accounts` to pass a variable number of
    // token accounts. The `amounts` vec has one entry per recipient.
    //
    // remaining_accounts is a Vec<AccountInfo> — you iterate over it.
    // Each account must be a mutable token account for the correct mint.
    //
    // Real-world: NFT minting sites, token launches, loyalty rewards.
    //
    // SECURITY NOTE: In production, you'd verify each remaining_account is
    // actually a token account for the correct mint. For learning, we trust
    // the caller (but the comment shows what you'd check).
    //
    pub fn airdrop(ctx: Context<Airdrop>, amounts: Vec<u64>) -> Result<()> {
        // TODO(human): Implement batch minting via remaining_accounts.
        //
        // Steps:
        // 1. Verify: amounts.len() == ctx.remaining_accounts.len()
        //    If not, return err!(ErrorCode::AirdropLengthMismatch)
        //
        // 2. Build signer seeds (same as managed_mint):
        //    let signer_seeds: &[&[&[u8]]] = &[&[
        //        MINT_AUTHORITY_SEED,
        //        &[ctx.bumps.mint_authority],
        //    ]];
        //
        // 3. Iterate over remaining_accounts and amounts together:
        //    for (recipient_account_info, amount) in
        //        ctx.remaining_accounts.iter().zip(amounts.iter())
        //    {
        //        let cpi_accounts = MintTo {
        //            mint: ctx.accounts.mint.to_account_info(),
        //            to: recipient_account_info.clone(),
        //            authority: ctx.accounts.mint_authority.to_account_info(),
        //        };
        //        let cpi_ctx = CpiContext::new(
        //            ctx.accounts.token_program.to_account_info(),
        //            cpi_accounts,
        //        ).with_signer(signer_seeds);
        //        token_interface::mint_to(cpi_ctx, *amount)?;
        //    }
        //
        // 4. Return Ok(())
        //
        // Warning: Each CPI costs compute units. Solana's max is 200K per
        // instruction (1.4M per transaction with compute budget). With ~10K CU
        // per mint_to, you can airdrop to ~15-20 recipients per transaction.
        // Real airdrops use multiple transactions or compression (ZK compression).
        require!(
            amounts.len() == ctx.remaining_accounts.len(),
            ErrorCode::AirdropLengthMismatch
        );
        todo!("Iterate over remaining_accounts, mint to each")
    }
}

// =============================================================================
// Account Validation Structs
// =============================================================================

/// Initialize a mint where a PDA controls minting.
#[derive(Accounts)]
pub struct InitializeManagedMint<'info> {
    /// Admin wallet paying for creation. NOT the mint authority.
    #[account(mut)]
    pub admin: Signer<'info>,

    /// The token mint. Authority is the PDA, not the admin.
    #[account(
        init,
        payer = admin,
        mint::decimals = 6,
        mint::authority = mint_authority.key(),
        mint::freeze_authority = mint_authority.key(),
    )]
    pub mint: InterfaceAccount<'info, Mint>,

    /// The PDA that will be the mint authority.
    /// CHECK: This is a PDA — no data, just an address used as authority.
    #[account(
        seeds = [MINT_AUTHORITY_SEED],
        bump,
    )]
    pub mint_authority: SystemAccount<'info>,

    pub token_program: Interface<'info, TokenInterface>,
    pub system_program: Program<'info, System>,
}

/// Mint tokens using the PDA authority.
#[derive(Accounts)]
pub struct ManagedMint<'info> {
    /// The admin triggering the mint (for access control).
    /// In production, you'd verify this is an authorized admin.
    #[account(mut)]
    pub admin: Signer<'info>,

    /// The mint (supply changes → mutable).
    #[account(
        mut,
        constraint = mint.mint_authority.unwrap() == mint_authority.key()
            @ ErrorCode::InvalidMintAuthority,
    )]
    pub mint: InterfaceAccount<'info, Mint>,

    /// The PDA mint authority. Not a Signer — we sign via signer_seeds.
    /// CHECK: PDA verified by seeds constraint.
    #[account(
        seeds = [MINT_AUTHORITY_SEED],
        bump,
    )]
    pub mint_authority: SystemAccount<'info>,

    /// Recipient's token account.
    #[account(
        mut,
        token::mint = mint,
        token::token_program = token_program,
    )]
    pub recipient_token_account: InterfaceAccount<'info, TokenAccount>,

    pub token_program: Interface<'info, TokenInterface>,
}

/// Burn tokens (owner destroys their own tokens).
#[derive(Accounts)]
pub struct BurnTokens<'info> {
    /// The token account owner who wants to burn.
    pub owner: Signer<'info>,

    /// The mint (supply decreases → mutable).
    #[account(mut)]
    pub mint: InterfaceAccount<'info, Mint>,

    /// The token account to burn from. Must belong to owner.
    #[account(
        mut,
        token::mint = mint,
        token::authority = owner,
        token::token_program = token_program,
    )]
    pub token_account: InterfaceAccount<'info, TokenAccount>,

    pub token_program: Interface<'info, TokenInterface>,
}

/// Airdrop tokens to multiple recipients via remaining_accounts.
#[derive(Accounts)]
pub struct Airdrop<'info> {
    /// Admin triggering the airdrop.
    #[account(mut)]
    pub admin: Signer<'info>,

    /// The mint (supply increases → mutable).
    #[account(
        mut,
        constraint = mint.mint_authority.unwrap() == mint_authority.key()
            @ ErrorCode::InvalidMintAuthority,
    )]
    pub mint: InterfaceAccount<'info, Mint>,

    /// PDA mint authority.
    /// CHECK: PDA verified by seeds.
    #[account(
        seeds = [MINT_AUTHORITY_SEED],
        bump,
    )]
    pub mint_authority: SystemAccount<'info>,

    pub token_program: Interface<'info, TokenInterface>,
    // remaining_accounts: Vec<AccountInfo> — the recipient token accounts.
    // Passed dynamically, not defined in the struct.
}

// =============================================================================
// Custom Errors
// =============================================================================

#[error_code]
pub enum ErrorCode {
    #[msg("Mint authority does not match the expected PDA")]
    InvalidMintAuthority,

    #[msg("Amounts vector length must match remaining_accounts length")]
    AirdropLengthMismatch,
}
