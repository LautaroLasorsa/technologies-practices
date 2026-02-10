"""Wallet service — core business logic.

Some operations are fully implemented as reference.
Others are marked TODO(human) for you to implement.
"""

from wallet.models import Transaction, TransactionType, Wallet
from wallet.notifications import NotificationService


class InsufficientFundsError(Exception):
    """Raised when a withdrawal exceeds available balance."""

    def __init__(self, requested: float, available: float):
        self.requested = requested
        self.available = available
        super().__init__(
            f"Insufficient funds: requested {requested}, available {available}"
        )


class InvalidAmountError(Exception):
    """Raised when amount is zero or negative."""

    def __init__(self, amount: float):
        self.amount = amount
        super().__init__(f"Invalid amount: {amount}. Must be positive.")


class WalletService:
    """Manages wallet operations with optional notification support."""

    def __init__(self, notifier: NotificationService | None = None):
        self.notifier = notifier

    def deposit(self, wallet: Wallet, amount: float) -> Transaction:
        """Deposit funds into a wallet.

        This is fully implemented as a reference pattern.
        Study this before implementing withdraw() and transfer().
        """
        if amount <= 0:
            raise InvalidAmountError(amount)

        wallet.balance += amount
        txn = Transaction(
            type=TransactionType.DEPOSIT,
            amount=amount,
            description=f"Deposit of {amount:.2f}",
        )
        wallet.transactions.append(txn)
        return txn

    def withdraw(self, wallet: Wallet, amount: float) -> Transaction:
        """Withdraw funds from a wallet.

        TODO(human): Implement withdrawal logic. Follow the deposit() pattern above.

        Steps:
        1. Validate amount > 0, raise InvalidAmountError if not
        2. Check wallet.balance >= amount, raise InsufficientFundsError if not
        3. Decrease wallet.balance by amount
        4. Create a Transaction with type=WITHDRAW and descriptive message
        5. Append transaction to wallet.transactions
        6. Return the transaction

        Key insight: This is where property-based testing shines —
        you can verify that for ANY valid amount, the balance after
        withdrawal equals (original_balance - amount), and that
        withdrawing more than the balance ALWAYS raises InsufficientFundsError.
        """
        # TODO(human): implement withdrawal
        pass

    def transfer(
        self,
        from_wallet: Wallet,
        to_wallet: Wallet,
        amount: float,
    ) -> tuple[Transaction, Transaction]:
        """Transfer funds between wallets with notification.

        TODO(human): Implement transfer logic.

        Steps:
        1. Validate amount > 0, raise InvalidAmountError if not
        2. Check from_wallet.balance >= amount, raise InsufficientFundsError if not
        3. Decrease from_wallet.balance by amount
        4. Increase to_wallet.balance by amount
        5. Create TRANSFER_OUT transaction for from_wallet
        6. Create TRANSFER_IN transaction for to_wallet
        7. Append transactions to respective wallets
        8. If self.notifier is not None, call:
           self.notifier.send(
               recipient=to_wallet.owner,
               subject="Transfer received",
               body=f"You received {amount:.2f} from {from_wallet.owner}"
           )
        9. Return (out_txn, in_txn)

        Key insight for testing: The notifier is injected via __init__.
        In tests, you can pass a mock and verify it was called with
        the correct arguments — without sending real notifications.
        This is Dependency Injection for testability.
        """
        # TODO(human): implement transfer with notification
        pass

    def get_balance(self, wallet: Wallet) -> float:
        """Return current wallet balance."""
        return wallet.balance

    def get_history(
        self,
        wallet: Wallet,
        transaction_type: TransactionType | None = None,
    ) -> list[Transaction]:
        """Get transaction history, optionally filtered by type.

        TODO(human): Implement this in Phase 4 (TDD).
        This method will be built using the TDD workflow:
        1. First write the test (red)
        2. Then implement this method (green)
        3. Then refactor

        Steps:
        1. If transaction_type is None, return all transactions (copy of list)
        2. If transaction_type is provided, return only matching transactions
        3. Always return a new list (don't expose internal state)

        Hint: return [t for t in wallet.transactions if ...]
        """
        # TODO(human): implement in Phase 4 (TDD)
        pass
