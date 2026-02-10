// =============================================================================
// Phase 6: Mini DEX Integration Test Template
// =============================================================================
//
// This test exercises the full AMM lifecycle:
// 1. Create two token mints (Token A and Token B)
// 2. Initialize the liquidity pool
// 3. Add initial liquidity (and verify LP tokens minted)
// 4. Perform swaps (both directions) and verify constant product holds
// 5. Verify fee accumulation (k should increase after swaps with fees)
//
// To adapt to LiteSVM (Rust), follow the same flow but:
// - Use litesvm::LiteSVM for the runtime
// - Build transactions with solana_sdk
// - Deploy program with svm.add_program_from_file()
//
// NOTE: This is a REFERENCE template. Uncomment and fill TODO(human) sections.

import * as anchor from "@coral-xyz/anchor";
import { Program } from "@coral-xyz/anchor";
import { MiniDex } from "../target/types/mini_dex";
import {
  createMint,
  createAssociatedTokenAccount,
  mintTo,
  getAccount,
  TOKEN_PROGRAM_ID,
  getAssociatedTokenAddress,
} from "@solana/spl-token";
import { assert } from "chai";

describe("mini_dex", () => {
  const provider = anchor.AnchorProvider.env();
  anchor.setProvider(provider);
  const program = anchor.workspace.MiniDex as Program<MiniDex>;

  // Actors
  const creator = anchor.web3.Keypair.generate();
  const liquidityProvider = anchor.web3.Keypair.generate();
  const trader = anchor.web3.Keypair.generate();

  // Mints
  let tokenAMint: anchor.web3.PublicKey;
  let tokenBMint: anchor.web3.PublicKey;

  // Derived PDAs (computed in before())
  let poolPda: anchor.web3.PublicKey;
  let vaultAPda: anchor.web3.PublicKey;
  let vaultBPda: anchor.web3.PublicKey;
  let lpMintPda: anchor.web3.PublicKey;

  // Token amounts
  const INITIAL_LIQUIDITY_A = new anchor.BN(1_000_000_000); // 1000 tokens (6 decimals)
  const INITIAL_LIQUIDITY_B = new anchor.BN(1_000_000_000); // 1000 tokens
  const SWAP_AMOUNT = new anchor.BN(100_000_000);           // 100 tokens

  before(async () => {
    // =====================================================================
    // Setup: Fund accounts, create mints, distribute tokens
    // =====================================================================

    // TODO(human): Airdrop SOL to all actors
    //
    // const airdropAmount = 10 * anchor.web3.LAMPORTS_PER_SOL;
    // await provider.connection.requestAirdrop(creator.publicKey, airdropAmount);
    // await provider.connection.requestAirdrop(liquidityProvider.publicKey, airdropAmount);
    // await provider.connection.requestAirdrop(trader.publicKey, airdropAmount);
    // ... confirm transactions ...

    // TODO(human): Create token A and token B mints
    //
    // tokenAMint = await createMint(
    //   provider.connection, creator, creator.publicKey, null, 6
    // );
    // tokenBMint = await createMint(
    //   provider.connection, creator, creator.publicKey, null, 6
    // );

    // TODO(human): Create ATAs and mint tokens to LP and trader
    //
    // const lpTokenAAccount = await createAssociatedTokenAccount(
    //   provider.connection, liquidityProvider, tokenAMint, liquidityProvider.publicKey
    // );
    // await mintTo(
    //   provider.connection, creator, tokenAMint, lpTokenAAccount,
    //   creator, INITIAL_LIQUIDITY_A.toNumber() * 2
    // );
    // ... repeat for token B and trader ...

    // TODO(human): Derive all PDA addresses
    //
    // [poolPda] = anchor.web3.PublicKey.findProgramAddressSync(
    //   [Buffer.from("pool"), tokenAMint.toBuffer(), tokenBMint.toBuffer()],
    //   program.programId,
    // );
    // [vaultAPda] = anchor.web3.PublicKey.findProgramAddressSync(
    //   [Buffer.from("vault-a"), poolPda.toBuffer()],
    //   program.programId,
    // );
    // ... vault B and LP mint PDAs ...
  });

  // =========================================================================
  // Test 1: Initialize Pool
  // =========================================================================
  it("initializes a liquidity pool", async () => {
    // TODO(human): Call initialize_pool
    //
    // await program.methods
    //   .initializePool()
    //   .accounts({
    //     creator: creator.publicKey,
    //     tokenAMint,
    //     tokenBMint,
    //     pool: poolPda,
    //     vaultA: vaultAPda,
    //     vaultB: vaultBPda,
    //     lpMint: lpMintPda,
    //     tokenProgram: TOKEN_PROGRAM_ID,
    //     systemProgram: anchor.web3.SystemProgram.programId,
    //   })
    //   .signers([creator])
    //   .rpc();

    // TODO(human): Verify pool state
    //
    // const pool = await program.account.poolState.fetch(poolPda);
    // assert.equal(pool.tokenAMint.toBase58(), tokenAMint.toBase58());
    // assert.equal(pool.tokenBMint.toBase58(), tokenBMint.toBase58());
    // assert.equal(pool.totalLpSupply.toNumber(), 0);
  });

  // =========================================================================
  // Test 2: Add Initial Liquidity
  // =========================================================================
  it("adds initial liquidity", async () => {
    // TODO(human): Call add_liquidity with INITIAL_LIQUIDITY_A and INITIAL_LIQUIDITY_B
    //
    // const lpTokenAccount = await getAssociatedTokenAddress(
    //   lpMintPda, liquidityProvider.publicKey
    // );
    //
    // await program.methods
    //   .addLiquidity(INITIAL_LIQUIDITY_A, INITIAL_LIQUIDITY_B)
    //   .accounts({
    //     provider: liquidityProvider.publicKey,
    //     pool: poolPda,
    //     tokenAMint,
    //     tokenBMint,
    //     providerTokenA: lpProviderTokenA,
    //     providerTokenB: lpProviderTokenB,
    //     providerLpAccount: lpTokenAccount,
    //     vaultA: vaultAPda,
    //     vaultB: vaultBPda,
    //     lpMint: lpMintPda,
    //     tokenProgram: TOKEN_PROGRAM_ID,
    //     associatedTokenProgram: anchor.utils.token.ASSOCIATED_PROGRAM_ID,
    //     systemProgram: anchor.web3.SystemProgram.programId,
    //   })
    //   .signers([liquidityProvider])
    //   .rpc();

    // TODO(human): Verify LP tokens minted
    //
    // For initial deposit: lp_tokens = sqrt(1_000_000_000 * 1_000_000_000)
    //                                = sqrt(10^18) = 10^9 = 1_000_000_000
    //
    // const lpAccount = await getAccount(provider.connection, lpTokenAccount);
    // assert.equal(Number(lpAccount.amount), 1_000_000_000);
    //
    // // Verify vault balances
    // const vaultA = await getAccount(provider.connection, vaultAPda);
    // assert.equal(Number(vaultA.amount), INITIAL_LIQUIDITY_A.toNumber());
  });

  // =========================================================================
  // Test 3: Swap A → B
  // =========================================================================
  it("swaps token A for token B", async () => {
    // TODO(human): Call swap with a_to_b = true
    //
    // Expected output calculation:
    //   reserve_a = 1_000_000_000, reserve_b = 1_000_000_000
    //   amount_in = 100_000_000 (100 tokens)
    //   amount_in_after_fee = 100_000_000 * 9970 / 10000 = 99_700_000
    //   amount_out = 1_000_000_000 * 99_700_000 / (1_000_000_000 + 99_700_000)
    //              = 99_700_000_000_000_000 / 1_099_700_000
    //              ≈ 90_661_812
    //
    // Set min_amount_out to something slightly less (e.g., 90_000_000)
    // to account for rounding.
    //
    // const minAmountOut = new anchor.BN(90_000_000);
    //
    // await program.methods
    //   .swap(SWAP_AMOUNT, minAmountOut, true)
    //   .accounts({
    //     user: trader.publicKey,
    //     pool: poolPda,
    //     tokenAMint,
    //     tokenBMint,
    //     userTokenA: traderTokenA,
    //     userTokenB: traderTokenB,
    //     vaultA: vaultAPda,
    //     vaultB: vaultBPda,
    //     tokenProgram: TOKEN_PROGRAM_ID,
    //   })
    //   .signers([trader])
    //   .rpc();

    // TODO(human): Verify balances after swap
    //
    // After swap:
    //   vault_a should have ~1_100_000_000 (increased by swap_amount)
    //   vault_b should have ~909_338_188 (decreased by amount_out)
    //
    // Verify k increased (due to fees):
    //   old_k = 1_000_000_000 * 1_000_000_000 = 10^18
    //   new_k = 1_100_000_000 * 909_338_188 > 10^18 ← fees grew k!
    //
    // const vaultA = await getAccount(provider.connection, vaultAPda);
    // const vaultB = await getAccount(provider.connection, vaultBPda);
    // const newK = BigInt(vaultA.amount) * BigInt(vaultB.amount);
    // const oldK = BigInt(1_000_000_000) * BigInt(1_000_000_000);
    // assert.isTrue(newK > oldK, "k should increase due to fees");
  });

  // =========================================================================
  // Test 4: Swap B → A (reverse direction)
  // =========================================================================
  it("swaps token B for token A", async () => {
    // TODO(human): Call swap with a_to_b = false
    //
    // This tests that swapping works in both directions.
    // The output amount will differ because reserves are no longer equal.
    //
    // Key insight: After the A→B swap, there's more A and less B in the pool.
    // So swapping B→A should give MORE A per B than the initial 1:1 rate.
    // This is the AMM naturally adjusting the price based on supply/demand.
  });

  // =========================================================================
  // Test 5: Verify constant product invariant
  // =========================================================================
  it("maintains k increasing (fees grow the pool)", async () => {
    // TODO(human): Fetch vault balances and verify k >= initial_k
    //
    // const vaultA = await getAccount(provider.connection, vaultAPda);
    // const vaultB = await getAccount(provider.connection, vaultBPda);
    // const k = BigInt(vaultA.amount) * BigInt(vaultB.amount);
    // const initialK = BigInt(INITIAL_LIQUIDITY_A.toNumber()) * BigInt(INITIAL_LIQUIDITY_B.toNumber());
    //
    // assert.isTrue(k >= initialK, "k must never decrease");
    //
    // console.log(`Initial k: ${initialK}`);
    // console.log(`Current k: ${k}`);
    // console.log(`Growth: ${Number(k - initialK) / Number(initialK) * 100}%`);
  });
});
