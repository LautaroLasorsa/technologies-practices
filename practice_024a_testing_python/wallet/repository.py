"""In-memory wallet repository for integration testing."""

from wallet.models import Wallet


class WalletRepository:
    """Simple in-memory repository — stores wallets in a dict."""

    def __init__(self):
        self._store: dict[str, Wallet] = {}

    def save(self, wallet: Wallet) -> None:
        self._store[wallet.id] = wallet

    def load(self, wallet_id: str) -> Wallet | None:
        return self._store.get(wallet_id)

    def delete(self, wallet_id: str) -> bool:
        if wallet_id in self._store:
            del self._store[wallet_id]
            return True
        return False

    def list_all(self) -> list[Wallet]:
        return list(self._store.values())
