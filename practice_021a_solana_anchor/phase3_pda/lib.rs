// =============================================================================
// Phase 3: PDAs (Program Derived Addresses)
// =============================================================================
//
// REFERENCE FILE: Copy relevant parts into your program's lib.rs
//
// KEY CONCEPTS:
//
// 1. WHAT IS A PDA?
//    A PDA is an address derived from a set of seeds + a program ID.
//    Unlike regular addresses (keypairs), PDAs have NO private key.
//    They live "off the Ed25519 curve" -- you can't sign with them normally.
//
//    Derivation: PDA = hash(seeds, program_id, bump)
//    The "bump" is a byte (0-255) that pushes the address off the curve.
//    Anchor finds the bump automatically.
//
// 2. WHY PDAs?
//    - DETERMINISTIC: Given the same seeds, you always get the same address.
//      No need to store the address -- recompute it from seeds.
//    - UNIQUE: seeds = [b"counter", user_pubkey] guarantees one counter per user.
//    - PROGRAM-CONTROLLED: Only the program that derived the PDA can "sign"
//      transactions involving it (via CPI with seeds). This is crucial for vaults.
//
// 3. ANALOGY FOR RUST DEVELOPERS:
//    Think of PDAs as a deterministic HashMap:
//
//      HashMap<(&[u8], Pubkey), Account>
//
//    seeds = the key
//    PDA   = the deterministic "address" computed from the key
//    Account at that address = the value
//
//    Example: seeds = [b"counter", user.key()]
//    This is like: counters[("counter", user_pubkey)] = CounterAccount { ... }
//
//    You don't need to store "where is Alice's counter?" -- just recompute:
//      PDA = find_program_address([b"counter", alice_pubkey], program_id)
//
// 4. BUMP SEED:
//    The bump is a single byte (0-255) appended to the seeds to ensure
//    the resulting address is NOT on the Ed25519 curve. Anchor tries 255,
//    then 254, ..., until it finds a valid PDA. The first valid bump is
//    called the "canonical bump". Always use the canonical bump.
//
//    Store the bump in the account data so you don't waste compute
//    re-deriving it in subsequent instructions.
//
// 5. PDA vs KEYPAIR ACCOUNT:
//    | Property         | Keypair Account      | PDA                     |
//    |------------------|----------------------|-------------------------|
//    | Has private key  | Yes                  | No                      |
//    | Created by       | Anyone with the key  | Only the deriving program|
//    | Address from     | Random generation    | Deterministic from seeds|
//    | Can sign txs     | With private key     | Via CPI with seeds      |
//    | Use case         | User wallets         | Program-owned state     |
//
// =============================================================================

use anchor_lang::prelude::*;

declare_id!("11111111111111111111111111111111");

#[program]
pub mod solana_practice {
    use super::*;

    // =========================================================================
    // Exercise 1: PDA-based Counter
    // =========================================================================
    //
    // Each user gets their own counter, derived from their public key.
    // No need to pass the counter's address -- the program computes it.
    //
    // seeds = [b"counter", user.key().as_ref()]
    //
    // This means:
    //   - Alice's counter PDA = hash("counter" + alice_pubkey + bump)
    //   - Bob's counter PDA   = hash("counter" + bob_pubkey + bump)
    //   - They're different addresses, deterministically derived
    //
    pub fn initialize_pda_counter(ctx: Context<InitializePdaCounter>) -> Result<()> {
        let counter = &mut ctx.accounts.counter;

        // ── Exercise Context ──────────────────────────────────────────────────
        // This exercise teaches PDA (Program Derived Address) creation—the key mechanism
        // for deterministic account addressing in Solana. PDAs are like a HashMap where keys
        // (seeds) deterministically map to addresses, eliminating the need to store addresses.
        //
        // TODO(human): Set counter.authority to the user's public key
        //   Hint: ctx.accounts.user.key()
        //
        // TODO(human): Set counter.count to 0
        //
        // TODO(human): Store the bump for later use
        //   Hint: counter.bump = ctx.bumps.counter;
        //   The bump is passed via ctx.bumps (a hashmap of account_name -> bump)
        //

        Ok(())
    }

    pub fn increment_pda_counter(ctx: Context<IncrementPdaCounter>) -> Result<()> {
        let counter = &mut ctx.accounts.counter;

        // TODO(human): Increment counter.count
        //

        Ok(())
    }

    // =========================================================================
    // Exercise 3: User Profile PDA
    // =========================================================================
    //
    // A more realistic example: each user gets exactly one profile.
    // seeds = [b"profile", user.key().as_ref()]
    //
    // Profile stores: name (String), bio (String), authority (Pubkey), bump (u8)
    //
    // Space calculation for strings:
    //   String = 4 bytes (length prefix) + max_content_bytes
    //   If name can be up to 32 chars: 4 + 32 = 36 bytes
    //   If bio can be up to 256 chars:  4 + 256 = 260 bytes
    //
    pub fn create_profile(
        ctx: Context<CreateProfile>,
        name: String,
        bio: String,
    ) -> Result<()> {
        let profile = &mut ctx.accounts.profile;

        // TODO(human): Validate name length (max 32 bytes)
        //   Hint: require!(name.len() <= 32, ...);
        //
        // TODO(human): Validate bio length (max 256 bytes)
        //
        // TODO(human): Set profile fields: authority, name, bio, bump
        //   Hint: profile.authority = ctx.accounts.user.key();
        //         profile.name = name;
        //         profile.bio = bio;
        //         profile.bump = ctx.bumps.profile;
        //

        Ok(())
    }

    pub fn update_profile(
        ctx: Context<UpdateProfile>,
        name: String,
        bio: String,
    ) -> Result<()> {
        let profile = &mut ctx.accounts.profile;

        // TODO(human): Same validation as create_profile
        //
        // TODO(human): Update name and bio
        //   Hint: profile.name = name; profile.bio = bio;
        //

        Ok(())
    }

    // =========================================================================
    // Exercise 4: Close an account (reclaim rent)
    // =========================================================================
    //
    // When you no longer need an account, you can close it to reclaim
    // the rent deposit (lamports). Anchor's `close = destination` constraint
    // handles this: it zeroes the data, transfers lamports, and marks
    // the account as closed.
    //
    // This is important for cost management -- you don't want to pay rent
    // for accounts you no longer need.
    //
    pub fn close_profile(ctx: Context<CloseProfile>) -> Result<()> {
        // Anchor's `close` constraint handles everything.
        // The profile's lamports are transferred to `user`.
        // No code needed here -- the constraint does the work.
        //
        // But you might want to emit an event (Phase 6) or log something:
        msg!("Profile closed for user: {}", ctx.accounts.user.key());
        Ok(())
    }
}

// =============================================================================
// Account Data Structures
// =============================================================================

#[account]
pub struct PdaCounter {
    pub authority: Pubkey, // 32 bytes
    pub count: u64,        // 8 bytes
    pub bump: u8,          // 1 byte -- store the bump to avoid re-deriving
}
// space = 8 + 32 + 8 + 1 = 49 bytes

#[account]
pub struct UserProfile {
    pub authority: Pubkey, // 32 bytes
    pub name: String,      // 4 + 32 = 36 bytes (max 32 char name)
    pub bio: String,       // 4 + 256 = 260 bytes (max 256 char bio)
    pub bump: u8,          // 1 byte
}
// space = 8 + 32 + 36 + 260 + 1 = 337 bytes

// =============================================================================
// Instruction Account Structs
// =============================================================================

// --- Initialize PDA Counter ---
//
// The `seeds` and `bump` attributes tell Anchor:
//   1. Derive the PDA from these seeds
//   2. Verify the account address matches the derived PDA
//   3. Find the canonical bump automatically
//
// `init` + `seeds` + `bump` = create a PDA-based account
//
#[derive(Accounts)]
pub struct InitializePdaCounter<'info> {
    // seeds = [b"counter", user.key().as_ref()] -- deterministic per user
    // bump -- Anchor finds the canonical bump automatically
    // init -- create the account
    // payer = user -- user pays rent
    #[account(
        init,
        payer = user,
        space = 8 + 32 + 8 + 1,
        seeds = [b"counter", user.key().as_ref()],
        bump,
    )]
    pub counter: Account<'info, PdaCounter>,

    #[account(mut)]
    pub user: Signer<'info>,

    pub system_program: Program<'info, System>,
}

// --- Increment PDA Counter ---
//
// For existing PDA accounts, use `seeds` + `bump` WITHOUT `init`.
// This tells Anchor to VERIFY the address matches, not create it.
//
// The stored bump (counter.bump) is used for verification, which is
// cheaper than re-deriving: `bump = counter.bump`
//
#[derive(Accounts)]
pub struct IncrementPdaCounter<'info> {
    // TODO(human): Add the correct seeds and bump constraint
    //   Before: #[account(mut)]
    //   After:  #[account(mut, seeds = [b"counter", user.key().as_ref()], bump = counter.bump)]
    //
    // Why `bump = counter.bump`? Re-deriving the bump costs compute units.
    // Since we stored it during init, we reuse it here. This is a common pattern.
    //
    #[account(
        mut,
        seeds = [b"counter", user.key().as_ref()],
        bump = counter.bump,
    )]
    pub counter: Account<'info, PdaCounter>,

    pub user: Signer<'info>,
}

// --- Create Profile ---
#[derive(Accounts)]
pub struct CreateProfile<'info> {
    #[account(
        init,
        payer = user,
        space = 8 + 32 + 36 + 260 + 1,  // discriminator + authority + name + bio + bump
        seeds = [b"profile", user.key().as_ref()],
        bump,
    )]
    pub profile: Account<'info, UserProfile>,

    #[account(mut)]
    pub user: Signer<'info>,

    pub system_program: Program<'info, System>,
}

// --- Update Profile ---
//
// has_one = authority ensures only the profile owner can update it.
// seeds + bump verify the PDA address is correct.
//
#[derive(Accounts)]
pub struct UpdateProfile<'info> {
    #[account(
        mut,
        has_one = authority,
        seeds = [b"profile", authority.key().as_ref()],
        bump = profile.bump,
    )]
    pub profile: Account<'info, UserProfile>,

    pub authority: Signer<'info>,
}

// --- Close Profile ---
//
// The `close = user` constraint:
//   1. Zeroes out the account data
//   2. Transfers all lamports to `user`
//   3. Sets the account owner to the System Program (effectively deleting it)
//
#[derive(Accounts)]
pub struct CloseProfile<'info> {
    // TODO(human): Add the close constraint
    //   Hint: #[account(mut, close = user, has_one = authority, seeds = [...], bump = ...)]
    //
    #[account(
        mut,
        close = user,
        has_one = authority,
        seeds = [b"profile", authority.key().as_ref()],
        bump = profile.bump,
    )]
    pub profile: Account<'info, UserProfile>,

    #[account(mut)]
    pub authority: Signer<'info>,

    /// CHECK: Receives the closed account's lamports. Can be any account.
    /// We use `authority` as both signer and receiver here for simplicity,
    /// but they could be separate.
    #[account(mut)]
    pub user: SystemAccount<'info>,
}

// =============================================================================
// CLIENT-SIDE PDA DERIVATION (for tests -- TypeScript)
// =============================================================================
//
// Exercise 2: In your test file, derive the PDA client-side:
//
// ```typescript
// const [counterPda, bump] = PublicKey.findProgramAddressSync(
//   [Buffer.from("counter"), user.publicKey.toBuffer()],
//   program.programId
// );
// ```
//
// This computes the SAME address the program will derive on-chain.
// You pass `counterPda` as the `counter` account in your transaction.
//
// The beauty: you never stored this address anywhere. Both client and
// program independently compute it from the same seeds.
//
// Rust equivalent (off-chain):
// ```rust
// let (pda, bump) = Pubkey::find_program_address(
//     &[b"counter", user_pubkey.as_ref()],
//     &program_id,
// );
// ```
//
// =============================================================================
