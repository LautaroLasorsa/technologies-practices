// =============================================================================
// Phase 5: Token-2022 Extensions — Next-Gen Token Standard
// =============================================================================
//
// WHAT IS TOKEN-2022?
// ┌─────────────────────────────────────────────────────────────────────┐
// │ SPL Token (original): Simple mint/transfer/burn. Any extra feature │
// │ (fees, freezing logic, metadata) requires a SEPARATE program.      │
// │                                                                    │
// │ Token-2022: Same operations PLUS built-in "extensions":            │
// │ - Transfer fees (protocol takes a cut on every transfer)           │
// │ - Non-transferable tokens (soulbound — can't be sent)              │
// │ - Confidential transfers (ZK proofs hide amounts)                  │
// │ - Interest-bearing tokens (balance increases over time)            │
// │ - Permanent delegate (someone can always transfer your tokens)     │
// │ - Transfer hook (custom program called on every transfer)          │
// │ - Metadata (name, symbol, URI stored on-chain — no Metaplex)      │
// │                                                                    │
// │ Why this matters:                                                  │
// │ - Less code: Features that required custom programs are now native │
// │ - Cheaper: Fewer CPIs = fewer compute units                        │
// │ - Safer: Audited once in the Token Program, not per-project       │
// │ - Composable: Wallets/DEXs automatically support extensions        │
// └─────────────────────────────────────────────────────────────────────┘
//
// ANCHOR SUPPORT:
// anchor-spl provides `token_interface` which works with BOTH Token Program
// and Token-2022. The key types:
// - Interface<'info, TokenInterface> → accepts either program
// - InterfaceAccount<'info, Mint> → works with both programs
// - InterfaceAccount<'info, TokenAccount> → works with both programs
//
// To use Token-2022 specifically, pass the Token-2022 program ID instead
// of the Token Program ID when creating accounts.
//
// NOTE: Anchor's Token-2022 support is still evolving. Some extensions
// (like TransferFee) require manual account space calculation. This phase
// may be partially conceptual depending on the Anchor version.
//
// DEPENDENCIES (Cargo.toml):
// ```toml
// [dependencies]
// anchor-lang = { version = "0.31", features = ["init-if-needed"] }
// anchor-spl = { version = "0.31", features = ["token", "token_2022", "associated_token"] }
// spl-token-2022 = "5"
// ```

use anchor_lang::prelude::*;
use anchor_spl::token_interface::{
    self, Mint, MintTo, TokenAccount, TokenInterface,
};

declare_id!("Token2022Ext11111111111111111111111111111111");

#[program]
pub mod token_2022_extensions {
    use super::*;

    // =========================================================================
    // Exercise 1: Create a Token-2022 Mint with Transfer Fees
    // =========================================================================
    //
    // Transfer fees are charged automatically by the Token Program on every
    // transfer. You configure:
    // - fee_basis_points: e.g., 50 = 0.5% fee
    // - maximum_fee: cap on the fee in base units (prevents huge fees on large transfers)
    //
    // The fee is withheld in the destination token account and can be
    // "harvested" by the fee authority into a designated account.
    //
    // Real-world: stablecoins with transfer tax, deflationary tokens.
    //
    // HOW IT WORKS:
    // 1. Create mint with TransferFeeConfig extension
    // 2. On every transfer, Token-2022 automatically deducts the fee
    // 3. Fee accumulates as "withheld" tokens in recipient accounts
    // 4. Fee authority calls harvest_withheld_tokens_to_mint to collect
    //
    // NOTE: Anchor may not have built-in init constraints for extensions.
    // If so, you'll need to manually:
    // 1. Calculate account size: Mint::LEN + ExtensionType::try_calculate_account_len(&[TransferFeeConfig])
    // 2. Create account via system_program::create_account
    // 3. Initialize extension via spl_token_2022::instruction::initialize_transfer_fee_config
    // 4. Initialize mint via spl_token_2022::instruction::initialize_mint2
    //
    pub fn create_transfer_fee_mint(
        ctx: Context<CreateTransferFeeMint>,
        decimals: u8,
        fee_basis_points: u16,
        maximum_fee: u64,
    ) -> Result<()> {
        // TODO(human): Create a Token-2022 mint with transfer fee extension.
        //
        // If Anchor supports it natively (check your version):
        //   The init constraint might support extensions directly.
        //   Check: https://www.anchor-lang.com/docs/tokens
        //
        // If manual setup is needed:
        //
        // Step 1: Calculate required space.
        //   use spl_token_2022::extension::ExtensionType;
        //   let space = ExtensionType::try_calculate_account_len::<
        //       spl_token_2022::state::Mint
        //   >(&[ExtensionType::TransferFeeConfig])?;
        //
        // Step 2: Create the account (System Program CPI).
        //   let rent = Rent::get()?;
        //   let lamports = rent.minimum_balance(space);
        //   anchor_lang::system_program::create_account(
        //       CpiContext::new(
        //           ctx.accounts.system_program.to_account_info(),
        //           anchor_lang::system_program::CreateAccount {
        //               from: ctx.accounts.payer.to_account_info(),
        //               to: ctx.accounts.mint.to_account_info(),
        //           },
        //       ),
        //       lamports,
        //       space as u64,
        //       &spl_token_2022::id(),
        //   )?;
        //
        // Step 3: Initialize the transfer fee config extension.
        //   (CPI to Token-2022's initialize_transfer_fee_config)
        //
        // Step 4: Initialize the mint.
        //   (CPI to Token-2022's initialize_mint2)
        //
        // This is more manual than regular SPL Token because extensions
        // require specific initialization order:
        //   create_account → initialize_extensions → initialize_mint
        //
        // The extension MUST be initialized BEFORE the mint itself.
        //
        // Hint: Study the spl-token-2022 crate's instruction module for
        // the exact CPI signatures.
        msg!("Transfer fee mint: {} bps, max fee: {}", fee_basis_points, maximum_fee);
        todo!("Create Token-2022 mint with TransferFeeConfig extension")
    }

    // =========================================================================
    // Exercise 2: Create a Non-Transferable (Soulbound) Token
    // =========================================================================
    //
    // A non-transferable token CANNOT be sent to anyone. Once minted to a
    // wallet, it stays there forever (or until burned).
    //
    // Use cases:
    // - Proof of attendance (POAPs)
    // - Reputation tokens (can't be bought/sold)
    // - KYC verification badges
    // - Game achievements
    // - Professional certifications
    //
    // Implementation: Add NonTransferable extension to the mint.
    // All token accounts for this mint automatically can't transfer.
    //
    // This is Solana's equivalent of Ethereum's "Soulbound Tokens" (SBTs),
    // proposed by Vitalik Buterin in 2022. But on Solana, it's just an
    // extension flag — no custom contract needed.
    //
    pub fn create_soulbound_mint(
        ctx: Context<CreateSoulboundMint>,
        decimals: u8,
    ) -> Result<()> {
        // TODO(human): Create a Token-2022 mint with NonTransferable extension.
        //
        // Similar pattern to transfer fee mint, but simpler:
        // 1. Calculate space with ExtensionType::NonTransferable
        // 2. Create account
        // 3. Initialize NonTransferable extension
        // 4. Initialize mint
        //
        // The NonTransferable extension has no configuration — it's just a flag.
        //
        // After creation, mint_to works normally, but transfer_checked
        // will FAIL with an error. The only way to "remove" the token is burn.
        //
        // Question: If you can't transfer soulbound tokens, how do you
        // revoke credentials? (Answer: the mint authority or freeze authority
        // can freeze the account, or the holder can burn.)
        msg!("Creating soulbound (non-transferable) mint with {} decimals", decimals);
        todo!("Create Token-2022 mint with NonTransferable extension")
    }

    // =========================================================================
    // Exercise 3: Mint tokens (works with both Token and Token-2022)
    // =========================================================================
    //
    // This demonstrates the beauty of token_interface — the SAME instruction
    // works for both SPL Token and Token-2022 mints.
    //
    // The caller just passes the correct token_program (either Token or
    // Token-2022) and Anchor's Interface<TokenInterface> handles it.
    //
    pub fn mint_tokens(ctx: Context<MintExtTokens>, amount: u64) -> Result<()> {
        // TODO(human): Implement mint via token_interface (same as Phase 1).
        //
        //   let cpi_accounts = MintTo {
        //       mint: ctx.accounts.mint.to_account_info(),
        //       to: ctx.accounts.token_account.to_account_info(),
        //       authority: ctx.accounts.authority.to_account_info(),
        //   };
        //   let cpi_ctx = CpiContext::new(
        //       ctx.accounts.token_program.to_account_info(),
        //       cpi_accounts,
        //   );
        //   token_interface::mint_to(cpi_ctx, amount)?;
        //
        // Key insight: This exact code works for Token-2022 mints too!
        // token_interface abstracts over both programs.
        let _ = amount;
        todo!("Same as Phase 1 mint_tokens — token_interface handles both programs")
    }
}

// =============================================================================
// Account Validation Structs
// =============================================================================

/// Create a Token-2022 mint with transfer fee extension.
/// NOTE: This struct may need adjustment based on Anchor's Token-2022 support.
/// If Anchor doesn't support extension init constraints directly, you'll use
/// UncheckedAccount for the mint and initialize manually in the instruction.
#[derive(Accounts)]
pub struct CreateTransferFeeMint<'info> {
    #[account(mut)]
    pub payer: Signer<'info>,

    /// The mint account (will be a Token-2022 mint).
    /// If Anchor supports extensions:
    ///   #[account(init, payer = payer, mint::decimals = ..., extensions::transfer_fee::...)]
    /// If manual:
    ///   Use a Signer (keypair) and initialize in the instruction body.
    ///
    /// CHECK: Initialized manually in instruction body if Anchor doesn't
    /// support Token-2022 extension init constraints.
    #[account(mut)]
    pub mint: Signer<'info>,

    /// Token-2022 program (NOT the original Token Program).
    /// Program ID: TokenzQdBNbLqP5VEhdkAS6EPFLC1PHnBqCXEpPxuEb
    pub token_program: Interface<'info, TokenInterface>,

    pub system_program: Program<'info, System>,
}

/// Create a non-transferable (soulbound) Token-2022 mint.
#[derive(Accounts)]
pub struct CreateSoulboundMint<'info> {
    #[account(mut)]
    pub payer: Signer<'info>,

    /// CHECK: Initialized manually in instruction body.
    #[account(mut)]
    pub mint: Signer<'info>,

    pub token_program: Interface<'info, TokenInterface>,
    pub system_program: Program<'info, System>,
}

/// Mint tokens — works with any TokenInterface-compatible program.
#[derive(Accounts)]
pub struct MintExtTokens<'info> {
    #[account(mut)]
    pub authority: Signer<'info>,

    #[account(mut)]
    pub mint: InterfaceAccount<'info, Mint>,

    #[account(
        mut,
        token::mint = mint,
        token::token_program = token_program,
    )]
    pub token_account: InterfaceAccount<'info, TokenAccount>,

    /// Accepts both Token Program and Token-2022.
    pub token_program: Interface<'info, TokenInterface>,
}

// =============================================================================
// CONCEPTUAL: Confidential Transfers (ZK-SNARKs)
// =============================================================================
//
// Token-2022 supports confidential transfers where:
// - On-chain balances are encrypted (ElGamal encryption)
// - Transfers use zero-knowledge proofs to verify:
//   * The sender has enough balance (without revealing the balance)
//   * The transfer amount is non-negative
//   * The new balances are correct
//
// This is the most advanced Token-2022 extension. Implementation requires:
// 1. ElGamal keypair generation (client-side)
// 2. Range proof generation (client-side, using bulletproofs)
// 3. On-chain verification (~100K+ compute units per transfer)
//
// For this practice, understanding the concept is sufficient.
// If interested, see: https://spl.solana.com/confidential-token/deep-dive/overview
//
// Real-world: Private DeFi, compliance-friendly privacy (auditors can be
// given the decryption key), institutional trading with hidden order sizes.
//
// =============================================================================
// CONCEPTUAL: Other Extensions
// =============================================================================
//
// InterestBearing:
//   Balance display increases over time (like a savings account).
//   Actual tokens don't change — just the "UI balance" = amount * (1 + rate * time).
//   Use case: tokenized bonds, lending receipts.
//
// PermanentDelegate:
//   A designated authority can transfer tokens FROM any holder.
//   Use case: regulatory compliance (freeze + seize), subscription billing.
//   WARNING: This is controversial — gives power to revoke tokens from users.
//
// TransferHook:
//   A custom program is called on EVERY transfer.
//   The hook program can approve/reject/modify the transfer.
//   Use case: royalties (NFTs), sanctions screening, custom fee logic.
//   This is the most flexible extension — it's basically middleware.
//
// MintCloseAuthority:
//   Allows closing the mint account (normally, mints can't be closed).
//   Use case: temporary tokens, one-time-use tokens, cleanup.
//
// DefaultAccountState:
//   New token accounts start frozen (must be thawed by the freeze authority).
//   Use case: KYC-gated tokens — users must pass verification before trading.
