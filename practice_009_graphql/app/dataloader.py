"""DataLoader setup for batching and caching within a single request.

Strawberry integrates with the DataLoader pattern to solve the N+1 problem.
Instead of each BookType.author resolver making an individual lookup, all
author_ids are collected and resolved in one batch call.

See: https://strawberry.rocks/docs/guides/dataloaders
"""

from __future__ import annotations

from strawberry.dataloader import DataLoader
from strawberry.fastapi import BaseContext

from data import Author


# ---------------------------------------------------------------------------
# Batch loading function
# ---------------------------------------------------------------------------

async def load_authors(keys: list[int]) -> list[Author | None]:
    """Batch-load authors by their IDs.

    TODO(human): Implement this batch function.

    Contract (MUST be followed for DataLoader to work correctly):
        - Receives a list of author IDs (keys).
        - Returns a list of the SAME LENGTH as keys.
        - Each position i in the result corresponds to keys[i].
        - If an author is not found, that position should be None.

    Steps:
        1. Import get_authors_by_ids from the data module.
        2. Call get_authors_by_ids(keys) — it already preserves order.
        3. Return the result directly.

    Why this matters:
        Without DataLoader, querying 10 books with their authors triggers
        10 individual get_author_by_id() calls. With DataLoader, all 10
        author_ids are collected into one call to this batch function.

    Key question to think about:
        Why must DataLoader instances be per-request and not global singletons?
        (Answer: caching across requests would serve stale data and leak memory.)

    Hint:
        from data import get_authors_by_ids
        return get_authors_by_ids(keys)
    """
    raise NotImplementedError("TODO(human): implement load_authors batch function")


# ---------------------------------------------------------------------------
# Context with DataLoader (wired into FastAPI)
# ---------------------------------------------------------------------------

class GraphQLContext(BaseContext):
    """Custom context that provides a DataLoader for author batching."""

    def __init__(self) -> None:
        super().__init__()
        # A fresh DataLoader per request — no cross-request caching.
        self.author_loader: DataLoader[int, Author | None] = DataLoader(load_fn=load_authors)


async def get_context() -> GraphQLContext:
    """Factory for the per-request GraphQL context.

    Strawberry calls this for every incoming request and passes the result
    as `info.context` to all resolvers.
    """
    return GraphQLContext()
