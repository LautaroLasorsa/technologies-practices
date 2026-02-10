"""GraphQL Query resolvers for the Bookstore API."""

from __future__ import annotations

from typing import Optional

import strawberry

from gql_types import BookType, BookConnection


@strawberry.type
class Query:
    """Root query type — read operations."""

    @strawberry.field
    def books(self) -> list[BookType]:
        """Return all books in the store.

        TODO(human): Implement this resolver.

        Steps:
            1. Call get_all_books() from data module to get all Book models.
            2. Convert each Book to BookType using BookType.from_model().
            3. Return the list of BookType objects.

        Hint: A list comprehension works well here.
              from data import get_all_books
        """
        raise NotImplementedError("TODO(human): implement books query")

    @strawberry.field
    def book(self, id: int) -> Optional[BookType]:
        """Return a single book by its ID, or null if not found.

        TODO(human): Implement this resolver.

        Steps:
            1. Call get_book_by_id(id) from the data module.
            2. If the book exists, convert it to BookType with BookType.from_model().
            3. If the book is None, return None.

        Hint: from data import get_book_by_id
        """
        raise NotImplementedError("TODO(human): implement book query")

    @strawberry.field
    def books_paginated(
        self,
        first: int = 5,
        after: Optional[str] = None,
    ) -> BookConnection:
        """Return a paginated list of books (Relay Connection spec).

        TODO(human): Implement this in Phase 5.

        Steps:
            1. Import paginate_books from the pagination module.
            2. Call paginate_books(first, after) and return the result.

        Hint: from pagination import paginate_books
        """
        raise NotImplementedError("TODO(human): implement paginated books query")
