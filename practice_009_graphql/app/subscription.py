"""GraphQL Subscription resolvers for the Bookstore API."""

from __future__ import annotations

from typing import AsyncGenerator

import strawberry

from data import book_added_queue
from gql_types import BookType


@strawberry.type
class Subscription:
    """Root subscription type — real-time events over WebSocket."""

    @strawberry.subscription
    async def book_added(self) -> AsyncGenerator[BookType, None]:
        """Yield a BookType each time a new book is added via mutation.

        TODO(human): Implement this subscription.

        Steps:
            1. Create an infinite loop (while True).
            2. Inside the loop, await book_added_queue.get() to wait for
               the next book that was pushed by the add_book mutation.
            3. Convert the Book model to BookType using BookType.from_model().
            4. Yield the BookType.

        How it works:
            - asyncio.Queue.get() is an awaitable that suspends until an item
              is available — no busy-waiting or polling.
            - Strawberry wraps this async generator in a WebSocket handler;
              each yielded value is sent to the subscribed client.
            - When the client disconnects, Strawberry closes the generator.

        Hint:
            book = await book_added_queue.get()
            yield BookType.from_model(book)
        """
        raise NotImplementedError("TODO(human): implement book_added subscription")
        # The yield below is needed so Python recognizes this as an async generator.
        # Remove it (along with the raise above) when you implement the real logic.
        yield  # type: ignore[misc]
