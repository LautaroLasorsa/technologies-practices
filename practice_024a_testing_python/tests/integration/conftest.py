"""Integration test fixtures — heavier setup, wider scope."""

import pytest
from wallet.repository import WalletRepository


@pytest.fixture(scope="module")
def shared_repository():
    """Repository shared across all tests in a module.

    scope="module" means this fixture is created ONCE per test file,
    not once per test function. Use for expensive setup.
    """
    return WalletRepository()
