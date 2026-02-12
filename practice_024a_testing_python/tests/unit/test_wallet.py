"""Unit tests for wallet operations.

Phase 1: pytest fundamentals — fixtures, parametrize, assertions.
Phase 2: mocking — test transfer with mocked notification.

Reference: deposit tests are fully implemented. Use them as patterns.
"""

import pytest
from wallet.models import Wallet, TransactionType
from wallet.service import WalletService, InsufficientFundsError, InvalidAmountError


# ═══════════════════════════════════════════════════════════════════════════
# Deposit tests (FULLY IMPLEMENTED — use as reference)
# ═══════════════════════════════════════════════════════════════════════════

class TestDeposit:
    """Tests for WalletService.deposit() — reference implementation."""

    @pytest.mark.unit
    def test_deposit_increases_balance(self, service: WalletService, wallet: Wallet):
        """Basic deposit adds to balance."""
        service.deposit(wallet, 100.0)
        assert wallet.balance == 100.0

    @pytest.mark.unit
    def test_deposit_returns_transaction(self, service: WalletService, wallet: Wallet):
        """Deposit returns a Transaction with correct type and amount."""
        txn = service.deposit(wallet, 50.0)
        assert txn.type == TransactionType.DEPOSIT
        assert txn.amount == 50.0

    @pytest.mark.unit
    def test_deposit_appends_to_history(self, service: WalletService, wallet: Wallet):
        """Each deposit adds one transaction to history."""
        service.deposit(wallet, 100.0)
        service.deposit(wallet, 200.0)
        assert len(wallet.transactions) == 2

    @pytest.mark.unit
    @pytest.mark.parametrize("amount", [0, -1, -100.5])
    def test_deposit_rejects_invalid_amount(
        self, service: WalletService, wallet: Wallet, amount: float
    ):
        """Deposit raises InvalidAmountError for zero or negative amounts."""
        with pytest.raises(InvalidAmountError):
            service.deposit(wallet, amount)

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "deposits, expected_balance",
        [
            ([100.0], 100.0),
            ([100.0, 200.0], 300.0),
            ([50.0, 50.0, 50.0], 150.0),
            ([0.01], 0.01),
        ],
    )
    def test_deposit_accumulates(
        self,
        service: WalletService,
        wallet: Wallet,
        deposits: list[float],
        expected_balance: float,
    ):
        """Multiple deposits accumulate correctly."""
        for amount in deposits:
            service.deposit(wallet, amount)
        assert wallet.balance == pytest.approx(expected_balance)


# ═══════════════════════════════════════════════════════════════════════════
# Withdraw tests — TODO(human)
# ═══════════════════════════════════════════════════════════════════════════

class TestWithdraw:
    """Tests for WalletService.withdraw().

    TODO(human): Implement these test methods. Follow the TestDeposit
    patterns above. Each test should be focused on ONE behavior.

    Use these fixtures (defined in conftest.py):
    - wallet: empty wallet (balance=0)
    - funded_wallet: wallet with balance=1000.0
    - service: WalletService instance
    """

    @pytest.mark.unit
    def test_withdraw_decreases_balance(
        self, service: WalletService, funded_wallet: Wallet
    ):
        """Withdrawing 200 from a 1000-balance wallet leaves 800.

        # ── Exercise Context ──────────────────────────────────────────────────
        # This teaches writing focused unit tests that verify a single behavior.
        # Following the Arrange-Act-Assert pattern, this test sets up state (fixture provides
        # a funded wallet), performs one action (withdraw), and checks one outcome (balance decreased).

        TODO(human): Implement this test.
        1. Call service.withdraw(funded_wallet, 200.0)
        2. Assert funded_wallet.balance == 800.0
        """
        # TODO(human): implement test
        pass

    @pytest.mark.unit
    def test_withdraw_returns_transaction(
        self, service: WalletService, funded_wallet: Wallet
    ):
        """Withdraw returns a Transaction with type=WITHDRAW.

        TODO(human): Implement this test.
        1. Call service.withdraw(funded_wallet, 100.0)
        2. Assert the returned transaction has type == TransactionType.WITHDRAW
        3. Assert the returned transaction has amount == 100.0
        """
        # TODO(human): implement test
        pass

    @pytest.mark.unit
    def test_withdraw_insufficient_funds(
        self, service: WalletService, wallet: Wallet
    ):
        """Withdrawing more than balance raises InsufficientFundsError.

        TODO(human): Implement this test.
        1. Use pytest.raises(InsufficientFundsError) as context manager
        2. Try to withdraw 100.0 from an empty wallet
        3. Optionally check exc.requested and exc.available attributes
        """
        # TODO(human): implement test
        pass

    @pytest.mark.unit
    @pytest.mark.parametrize("amount", [0, -1, -50.0])
    def test_withdraw_rejects_invalid_amount(
        self, service: WalletService, funded_wallet: Wallet, amount: float
    ):
        """Withdraw raises InvalidAmountError for zero or negative amounts.

        TODO(human): Implement this test.
        Same pattern as test_deposit_rejects_invalid_amount.
        """
        # TODO(human): implement test
        pass

    @pytest.mark.unit
    def test_withdraw_exact_balance(
        self, service: WalletService, funded_wallet: Wallet
    ):
        """Withdrawing exact balance leaves zero (not an error).

        TODO(human): Implement this test.
        1. Withdraw 1000.0 from funded_wallet (which has 1000.0)
        2. Assert balance == 0.0
        """
        # TODO(human): implement test
        pass


# ═══════════════════════════════════════════════════════════════════════════
# Transfer tests with mocking — TODO(human)
# ═══════════════════════════════════════════════════════════════════════════

class TestTransfer:
    """Tests for WalletService.transfer() with mocked NotificationService.

    TODO(human): Implement these tests after implementing transfer() in service.py.

    Key concepts:
    - mocker.Mock() creates a mock object that records all calls
    - mock.assert_called_once_with(...) verifies the mock was called correctly
    - Passing the mock as notifier to WalletService tests the real code
      with a fake dependency (Dependency Injection pattern)
    """

    @pytest.mark.unit
    def test_transfer_moves_funds(self, service: WalletService):
        """Transfer moves exact amount between wallets.

        TODO(human): Implement this test.
        1. Create two wallets: alice (balance=500) and bob (balance=100)
        2. Call service.transfer(alice, bob, 200.0)
        3. Assert alice.balance == 300.0
        4. Assert bob.balance == 300.0
        """
        # TODO(human): implement test
        pass

    @pytest.mark.unit
    def test_transfer_sends_notification(self, mocker):
        """Transfer calls notifier.send() with correct arguments.

        TODO(human): Implement this test.
        1. Create a mock: mock_notifier = mocker.Mock()
        2. Create WalletService(notifier=mock_notifier)
        3. Create two wallets and transfer between them
        4. Assert: mock_notifier.send.assert_called_once_with(
               recipient=<to_wallet.owner>,
               subject="Transfer received",
               body=<expected message>
           )

        Key insight: The mock records every call. You can verify
        not just THAT it was called, but WITH WHAT ARGUMENTS.
        """
        # TODO(human): implement test
        pass

    @pytest.mark.unit
    def test_transfer_no_notification_without_notifier(self, service: WalletService):
        """Transfer works without notifier (notifier=None).

        TODO(human): Implement this test.
        1. service fixture has no notifier (notifier=None)
        2. Transfer should succeed without errors
        3. Verify balances changed correctly
        """
        # TODO(human): implement test
        pass

    @pytest.mark.unit
    def test_transfer_insufficient_funds(self, service: WalletService):
        """Transfer raises InsufficientFundsError if from_wallet is short.

        TODO(human): Implement this test.
        1. Create alice with balance=50
        2. Try to transfer 100 from alice to bob
        3. Verify InsufficientFundsError is raised
        4. Verify neither balance changed (atomicity)
        """
        # TODO(human): implement test
        pass
