"""Shared test fixtures.

Fixtures defined here are available to ALL tests automatically.
No import needed — pytest discovers conftest.py files by convention.
"""

import pytest
from wallet.models import Wallet
from wallet.service import WalletService
from wallet.repository import WalletRepository


@pytest.fixture
def wallet():
    """Create a fresh wallet with zero balance."""
    return Wallet(owner="Alice")


@pytest.fixture
def funded_wallet():
    """Create a wallet pre-loaded with 1000.00."""
    w = Wallet(owner="Bob", balance=1000.0)
    return w


@pytest.fixture
def service():
    """WalletService without notification (unit testing)."""
    return WalletService()


@pytest.fixture
def repository():
    """Fresh in-memory repository for each test."""
    return WalletRepository()
