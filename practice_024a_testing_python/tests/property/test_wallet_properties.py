"""Property-based tests using Hypothesis.

Phase 3: Verify wallet invariants hold for ANY valid input,
not just hand-picked examples.

Hypothesis automatically generates hundreds of test cases and
shrinks failures to the minimal reproducing example.
"""

import pytest
from hypothesis import given, assume, settings
from hypothesis import strategies as st

from wallet.models import Wallet
from wallet.service import WalletService, InsufficientFundsError, InvalidAmountError


# Reusable strategies
positive_amount = st.floats(min_value=0.01, max_value=100_000.0, allow_nan=False, allow_infinity=False)
non_positive_amount = st.floats(max_value=0.0, allow_nan=False, allow_infinity=False)


@pytest.mark.property
class TestWalletProperties:
    """Property-based tests for wallet invariants."""

    @given(amount=positive_amount)
    def test_deposit_always_increases_balance(self, amount: float):
        """PROPERTY: For any positive amount, deposit increases balance by exactly that amount.

        TODO(human): Implement this property test.
        1. Create a Wallet and WalletService
        2. Record initial balance
        3. Deposit the given amount
        4. Assert new balance == initial + amount (use pytest.approx for floats)

        This test will run ~100 times with different random amounts.
        If it fails, Hypothesis will shrink to the SMALLEST failing amount.

        Hint: Hypothesis handles the @given decorator — just implement
        the body. The 'amount' parameter receives generated values.
        """
        # TODO(human): implement property test
        pass

    @given(amount=positive_amount)
    def test_withdraw_then_deposit_restores_balance(self, amount: float):
        """PROPERTY: withdraw(x) then deposit(x) restores original balance.

        TODO(human): Implement this property test.
        1. Create a Wallet with balance=100_000 (enough for any withdrawal)
        2. Record initial balance
        3. Withdraw amount, then deposit same amount
        4. Assert balance == initial (use pytest.approx)

        This verifies deposit and withdraw are inverses for valid amounts.
        """
        # TODO(human): implement property test
        pass

    @given(amount=non_positive_amount)
    def test_deposit_rejects_non_positive(self, amount: float):
        """PROPERTY: Deposit ALWAYS rejects zero and negative amounts.

        TODO(human): Implement this property test.
        1. Create a Wallet and WalletService
        2. Use pytest.raises(InvalidAmountError) to verify rejection
        3. Assert wallet balance is unchanged (still 0)

        This tests the boundary: Hypothesis will try 0, -0.0, very small
        negatives, very large negatives, and edge cases you might not think of.
        """
        # TODO(human): implement property test
        pass

    @given(
        initial_balance=st.floats(min_value=0.0, max_value=10_000.0, allow_nan=False, allow_infinity=False),
        withdraw_amount=st.floats(min_value=0.01, max_value=20_000.0, allow_nan=False, allow_infinity=False),
    )
    def test_balance_never_negative(self, initial_balance: float, withdraw_amount: float):
        """PROPERTY: Balance is NEVER negative after any operation.

        TODO(human): Implement this property test.
        1. Create a Wallet with the given initial_balance
        2. Try to withdraw withdraw_amount
        3. If withdraw_amount <= initial_balance:
           - Withdrawal should succeed
           - Assert balance >= 0 (use pytest.approx(0, abs=1e-9) for boundary)
        4. If withdraw_amount > initial_balance:
           - Withdrawal should raise InsufficientFundsError
           - Assert balance unchanged

        This is the MOST IMPORTANT invariant: no sequence of operations
        should ever leave a wallet with negative balance.

        Hint: Use try/except InsufficientFundsError to handle both cases.
        """
        # TODO(human): implement property test
        pass

    @given(amounts=st.lists(positive_amount, min_size=1, max_size=20))
    def test_multiple_deposits_equal_sum(self, amounts: list[float]):
        """PROPERTY: Multiple deposits result in balance == sum of amounts.

        TODO(human): Implement this property test.
        1. Create a Wallet and WalletService
        2. Deposit each amount in the list
        3. Assert balance == sum(amounts) (use pytest.approx)

        Hypothesis generates lists of 1-20 random positive amounts.
        This verifies deposit is associative and commutative with addition.
        """
        # TODO(human): implement property test
        pass

    @given(
        balance=st.floats(min_value=100.0, max_value=10_000.0, allow_nan=False, allow_infinity=False),
        amount=st.floats(min_value=0.01, max_value=100.0, allow_nan=False, allow_infinity=False),
    )
    def test_transfer_preserves_total(self, balance: float, amount: float):
        """PROPERTY: Transfer preserves total money across both wallets.

        TODO(human): Implement this property test.
        1. Create alice (balance=balance) and bob (balance=0)
        2. Record total = alice.balance + bob.balance
        3. Transfer amount from alice to bob
        4. Assert alice.balance + bob.balance == total (use pytest.approx)

        This is the conservation law: money is neither created nor destroyed.
        The total across all wallets must remain constant after any transfer.
        """
        # TODO(human): implement property test
        pass
