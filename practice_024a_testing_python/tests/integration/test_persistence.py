"""Integration tests — wallet persistence via repository.

Phase 4: Test the full lifecycle: create → save → load → verify.
"""

import pytest
from wallet.models import Wallet
from wallet.service import WalletService
from wallet.repository import WalletRepository


@pytest.mark.integration
class TestWalletPersistence:
    """Integration tests for wallet save/load lifecycle."""

    def test_save_and_load_wallet(self, service: WalletService, repository: WalletRepository):
        """Save a wallet, load it back, verify data integrity.

        TODO(human): Implement this integration test.
        1. Create a Wallet(owner="Test User")
        2. Deposit 500.0 into it
        3. Save it: repository.save(wallet)
        4. Load it: loaded = repository.load(wallet.id)
        5. Assert loaded is not None
        6. Assert loaded.owner == "Test User"
        7. Assert loaded.balance == 500.0
        8. Assert len(loaded.transactions) == 1
        """
        # TODO(human): implement integration test
        pass

    def test_transfer_persisted_wallets(
        self, service: WalletService, repository: WalletRepository
    ):
        """Transfer between wallets that are saved to repository.

        TODO(human): Implement this integration test.
        1. Create two wallets: alice (balance=1000) and bob (balance=0)
        2. Save both to repository
        3. Load both from repository (simulates real DB round-trip)
        4. Transfer 300 from loaded alice to loaded bob
        5. Save both updated wallets
        6. Load both again and verify balances (alice=700, bob=300)

        This tests the full persistence lifecycle with business logic.
        """
        # TODO(human): implement integration test
        pass
