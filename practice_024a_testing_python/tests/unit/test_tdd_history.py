"""TDD exercise — transaction history filtering.

Phase 4: Practice the red-green-refactor cycle.

These tests are written FIRST (red). You then implement
get_history() in wallet/service.py to make them pass (green).
"""

import pytest
from wallet.models import Wallet, TransactionType
from wallet.service import WalletService


@pytest.mark.unit
class TestTransactionHistory:
    """Tests for WalletService.get_history() — TDD exercise.

    These tests define the EXPECTED behavior. They will FAIL
    until you implement get_history() in service.py.

    TDD cycle:
    1. Run these tests → all FAIL (red)
    2. Implement get_history() → all PASS (green)
    3. Refactor if needed → still PASS
    """

    def test_empty_history(self, service: WalletService, wallet: Wallet):
        """New wallet has empty transaction history."""
        history = service.get_history(wallet)
        assert history == []

    def test_history_contains_deposits(self, service: WalletService, wallet: Wallet):
        """History includes deposit transactions."""
        service.deposit(wallet, 100.0)
        service.deposit(wallet, 200.0)
        history = service.get_history(wallet)
        assert len(history) == 2
        assert all(t.type == TransactionType.DEPOSIT for t in history)

    def test_history_filter_by_type(self, service: WalletService, wallet: Wallet):
        """Filter history by transaction type."""
        service.deposit(wallet, 100.0)
        service.deposit(wallet, 200.0)
        # Need withdraw implemented first:
        service.withdraw(wallet, 50.0)

        deposits = service.get_history(wallet, transaction_type=TransactionType.DEPOSIT)
        withdrawals = service.get_history(wallet, transaction_type=TransactionType.WITHDRAW)

        assert len(deposits) == 2
        assert len(withdrawals) == 1

    def test_history_returns_copy(self, service: WalletService, wallet: Wallet):
        """get_history() returns a new list, not the internal reference.

        Modifying the returned list should NOT affect the wallet's transactions.
        This prevents accidental mutation of internal state.
        """
        service.deposit(wallet, 100.0)
        history = service.get_history(wallet)
        history.clear()  # Clear the returned list
        assert len(service.get_history(wallet)) == 1  # Original unchanged
