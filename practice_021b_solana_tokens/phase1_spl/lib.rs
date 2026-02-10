// =============================================================================
// Phase 1: SPL Token Basics — Mints, ATAs, Minting & Transferring
// =============================================================================
//
// MENTAL MODEL (coming from Ethereum):
// - Ethereum: Each token is a SEPARATE contract (ERC-20). USDC has its own contract,
//   WETH has its own, etc. Each implements transfer(), balanceOf(), etc.
// - Solana: There is ONE Token Program that manages ALL tokens. Each token is just
//   a "Mint Account" (metadata) + many "Token Accounts" (balances). The program
//   ID is the same for USDC, SOL wrappers, memecoins, NFTs — everything.
//
// ACCOUNT TYPES:
// ┌─────────────────────────────────────────────────────────────────────┐
// │ Mint Account                                                       │
// │ - Defines a token (like a "class definition")                      │
// │ - Fields: supply, decimals, mint_authority, freeze_authority        │
// │ - One per token type (e.g., one Mint for USDC)                     │
// │ - Owned by the Token Program                                       │
// ├─────────────────────────────────────────────────────────────────────┤
// │ Token Account (aka Associated Token Account / ATA)                 │
// │ - Holds a balance of a specific token for a specific owner         │
// │ - Fields: mint, owner, amount, delegate, state                     │
// │ - Many per mint (one per user who holds the token)                 │
// │ - Also owned by the Token Program                                  │
// │ - ATA = deterministic address derived from (wallet, mint)          │
// ├─────────────────────────────────────────────────────────────────────┤
// │ Key insight: You CANNOT hold tokens in your wallet directly.       │
// │ You need a Token Account for EACH token type you hold.             │
// │ ATAs solve "where is my token account?" via deterministic address. │
// └─────────────────────────────────────────────────────────────────────┘
//
// DECIMALS:
// Solana tokens use u64 amounts with a `decimals` field on the Mint.
// If decimals=6 and you want 1.5 tokens, you store 1_500_000.
// This is identical to how USDC works (6 decimals) or SOL (9 decimals).
// NEVER use floating point — it's non-deterministic across validators.
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
    token_interface::{self, Mint, MintTo, TokenAccount, TokenInterface, TransferChecked},
};

declare_id!("TokenBasics111111111111111111111111111111111");

#[program]
pub mod spl_token_basics {
    use super::*;

    // =========================================================================
    // Exercise 1: Create a new token mint
    // =========================================================================
    //
    // This instruction creates a new Mint account. Anchor's `init` constraint
    // handles the System Program CPI to allocate space and the Token Program
    // CPI to initialize the mint. Your instruction body can be empty — the
    // account validation does all the work.
    //
    // After this executes, a new token "type" exists on-chain with:
    // - 6 decimal places (like USDC)
    // - signer as mint authority (can create new tokens)
    // - signer as freeze authority (can freeze token accounts)
    //
    pub fn create_mint(_ctx: Context<CreateMint>) -> Result<()> {
        // TODO(human): This one is free — Anchor's init constraint does everything.
        // Just return Ok(()). But UNDERSTAND what the constraints below do:
        // 1. `init` → allocates account via System Program
        // 2. `payer = signer` → signer pays rent
        // 3. `mint::decimals = 6` → sets decimal precision
        // 4. `mint::authority = signer.key()` → who can mint new tokens
        //
        // Question: What would happen if you set mint::authority to a PDA instead?
        // (Answer: Only the program could mint — see Phase 2)
        todo!("Return Ok(())")
    }

    // =========================================================================
    // Exercise 2: Create an Associated Token Account (ATA) for a user
    // =========================================================================
    //
    // An ATA is a Token Account at a deterministic address derived from:
    //   PDA(wallet_pubkey, token_program_id, mint_pubkey)
    //
    // This means given a wallet and a mint, you can ALWAYS compute where the
    // token account is — no need to store or look up addresses.
    //
    // `init_if_needed` is used because the ATA might already exist (e.g., if
    // someone already sent tokens to this wallet for this mint).
    //
    pub fn create_token_account(_ctx: Context<CreateTokenAccount>) -> Result<()> {
        // TODO(human): Again, Anchor does the heavy lifting via constraints.
        // The `associated_token::mint` and `associated_token::authority`
        // constraints tell Anchor to derive the correct ATA address and
        // initialize it if it doesn't exist.
        //
        // Return Ok(()). But think about: why is `init_if_needed` better than
        // `init` here? (Hint: what if two transactions try to create the same
        // ATA in the same slot?)
        todo!("Return Ok(())")
    }

    // =========================================================================
    // Exercise 3: Mint tokens to a user's ATA
    // =========================================================================
    //
    // Minting = increasing supply. Only the mint_authority can do this.
    // This is a CPI (Cross-Program Invocation) to the Token Program's
    // `mint_to` instruction.
    //
    // CPI pattern:
    // 1. Build the accounts struct (MintTo { mint, to, authority })
    // 2. Create a CpiContext with the token program
    // 3. Call token_interface::mint_to(ctx, amount)
    //
    // The `amount` is in BASE UNITS. If decimals=6 and you want 100 tokens,
    // pass 100_000_000 (100 * 10^6).
    //
    pub fn mint_tokens(ctx: Context<MintTokens>, amount: u64) -> Result<()> {
        // TODO(human): Implement the CPI to mint tokens.
        //
        // Steps:
        // 1. Create a MintTo struct with the three required accounts:
        //    - mint: ctx.accounts.mint.to_account_info()
        //    - to: ctx.accounts.token_account.to_account_info()
        //    - authority: ctx.accounts.signer.to_account_info()
        //
        // 2. Create a CpiContext::new(token_program, mint_to_accounts)
        //    - token_program: ctx.accounts.token_program.to_account_info()
        //
        // 3. Call token_interface::mint_to(cpi_ctx, amount)?;
        //
        // Hint: This is ~5 lines. Look at the MintTo import at the top.
        //
        // Reference: https://www.anchor-lang.com/docs/tokens/basics/mint-tokens
        let _cpi_accounts = MintTo {
            mint: ctx.accounts.mint.to_account_info(),
            to: ctx.accounts.token_account.to_account_info(),
            authority: ctx.accounts.signer.to_account_info(),
        };
        todo!("Create CpiContext and call token_interface::mint_to")
    }

    // =========================================================================
    // Exercise 4: Transfer tokens between users
    // =========================================================================
    //
    // Transfer = move tokens from one Token Account to another.
    // Uses `transfer_checked` (not plain `transfer`) because it requires
    // passing the mint and decimals — this is a safety check ensuring you're
    // transferring the right token type with correct decimal handling.
    //
    // Why transfer_checked > transfer:
    // - Validates the mint matches both token accounts
    // - Validates the decimals parameter matches the mint
    // - Prevents accidentally sending 1_000_000 of a 9-decimal token
    //   when you meant to send 1_000_000 of a 6-decimal token
    //
    pub fn transfer_tokens(ctx: Context<TransferTokens>, amount: u64) -> Result<()> {
        // TODO(human): Implement the CPI to transfer tokens.
        //
        // Steps:
        // 1. Get decimals from the mint: ctx.accounts.mint.decimals
        //
        // 2. Create a TransferChecked struct:
        //    - mint: ctx.accounts.mint.to_account_info()
        //    - from: ctx.accounts.sender_token_account.to_account_info()
        //    - to: ctx.accounts.recipient_token_account.to_account_info()
        //    - authority: ctx.accounts.signer.to_account_info()
        //
        // 3. Create CpiContext::new(token_program, transfer_accounts)
        //
        // 4. Call token_interface::transfer_checked(cpi_ctx, amount, decimals)?;
        //
        // Hint: The `decimals` param is what makes this "checked" — the Token
        // Program will verify it matches the mint's actual decimals.
        //
        // Reference: https://www.anchor-lang.com/docs/tokens/basics/transfer-tokens
        let _decimals = ctx.accounts.mint.decimals;
        let _cpi_accounts = TransferChecked {
            mint: ctx.accounts.mint.to_account_info(),
            from: ctx.accounts.sender_token_account.to_account_info(),
            to: ctx.accounts.recipient_token_account.to_account_info(),
            authority: ctx.accounts.signer.to_account_info(),
        };
        todo!("Create CpiContext and call token_interface::transfer_checked")
    }
}

// =============================================================================
// Account Validation Structs
// =============================================================================
//
// These are COMPLETE — do not modify. Study the constraints to understand
// what Anchor does automatically.

/// Creates a new token mint with the signer as authority.
/// The mint is created as a new keypair (not a PDA).
#[derive(Accounts)]
pub struct CreateMint<'info> {
    /// The wallet paying for account creation (rent).
    #[account(mut)]
    pub signer: Signer<'info>,

    /// The new mint account. Anchor will:
    /// 1. Allocate space via System Program
    /// 2. Initialize mint data via Token Program
    /// 3. Set decimals=6, authority=signer, freeze_authority=signer
    #[account(
        init,
        payer = signer,
        mint::decimals = 6,
        mint::authority = signer.key(),
        mint::freeze_authority = signer.key(),
    )]
    pub mint: InterfaceAccount<'info, Mint>,

    /// Token Program (or Token-2022) — the program that owns Mint accounts.
    pub token_program: Interface<'info, TokenInterface>,

    /// System Program — needed for account creation (allocating space + rent).
    pub system_program: Program<'info, System>,
}

/// Creates an Associated Token Account (ATA) for a user.
/// Uses `init_if_needed` because the ATA might already exist.
#[derive(Accounts)]
pub struct CreateTokenAccount<'info> {
    /// The wallet paying for ATA creation.
    #[account(mut)]
    pub signer: Signer<'info>,

    /// The ATA to create. Anchor derives the address from (signer, mint).
    /// `init_if_needed` = don't fail if it already exists.
    #[account(
        init_if_needed,
        payer = signer,
        associated_token::mint = mint,
        associated_token::authority = signer,
        associated_token::token_program = token_program,
    )]
    pub token_account: InterfaceAccount<'info, TokenAccount>,

    /// The mint this ATA will hold tokens for.
    pub mint: InterfaceAccount<'info, Mint>,

    /// Token Program — owns the Token Account.
    pub token_program: Interface<'info, TokenInterface>,

    /// ATA Program — derives the deterministic ATA address.
    pub associated_token_program: Program<'info, AssociatedToken>,

    /// System Program — needed for account creation.
    pub system_program: Program<'info, System>,
}

/// Mints new tokens to a token account.
/// The signer must be the mint authority.
#[derive(Accounts)]
pub struct MintTokens<'info> {
    /// The mint authority (must match mint.mint_authority).
    #[account(mut)]
    pub signer: Signer<'info>,

    /// The mint to create tokens for. Must be mutable (supply changes).
    #[account(mut)]
    pub mint: InterfaceAccount<'info, Mint>,

    /// The destination token account. Must be mutable (balance changes).
    /// Constraint: this token account must be for the correct mint.
    #[account(
        mut,
        associated_token::mint = mint,
        associated_token::authority = signer,
        associated_token::token_program = token_program,
    )]
    pub token_account: InterfaceAccount<'info, TokenAccount>,

    /// Token Program — executes the mint_to CPI.
    pub token_program: Interface<'info, TokenInterface>,
}

/// Transfers tokens from sender to recipient.
/// The signer must own the sender's token account.
#[derive(Accounts)]
pub struct TransferTokens<'info> {
    /// The owner of the sender's token account.
    #[account(mut)]
    pub signer: Signer<'info>,

    /// The mint (needed for transfer_checked to validate decimals).
    #[account(mut)]
    pub mint: InterfaceAccount<'info, Mint>,

    /// Source token account (balance decreases).
    #[account(
        mut,
        associated_token::mint = mint,
        associated_token::authority = signer,
        associated_token::token_program = token_program,
    )]
    pub sender_token_account: InterfaceAccount<'info, TokenAccount>,

    /// Destination token account (balance increases).
    /// Note: no authority constraint — anyone can RECEIVE tokens.
    #[account(
        mut,
        token::mint = mint,
        token::token_program = token_program,
    )]
    pub recipient_token_account: InterfaceAccount<'info, TokenAccount>,

    /// Token Program.
    pub token_program: Interface<'info, TokenInterface>,
}
