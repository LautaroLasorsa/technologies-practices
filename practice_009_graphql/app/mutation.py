"""GraphQL Mutation resolvers for the Bookstore API."""

from __future__ import annotations

import strawberry

from gql_types import BookInput, BookType, MutationResult


@strawberry.type
class Mutation:
    """Root mutation type — write operations."""

    @strawberry.mutation
    async def add_book(self, input: BookInput) -> MutationResult:
        """Create a new book in the store.

        TODO(human): Implement this mutation.

        Steps:
            1. Validate the input:
               - title must not be empty
               - year must be positive
               - author_id must reference an existing author (use get_author_by_id)
               Collect validation errors in a list of strings.
            2. If there are validation errors, return MutationResult with
               success=False, a message like "Validation failed", and the errors list.
            3. If valid, call add_book() from the data module to create the book.
            4. Push the new book to the subscription queue:
               await book_added_queue.put(new_book)
            5. Return MutationResult with success=True, the created BookType, and
               a message like "Book created successfully".

        Hint:
            from data import add_book as store_add_book, get_author_by_id, book_added_queue

        Why errors-as-data? Because the mutation technically succeeds (HTTP 200),
        but the business logic has errors. The client can inspect the `errors`
        field without needing to parse exception messages.
        """
        raise NotImplementedError("TODO(human): implement add_book mutation")

    @strawberry.mutation
    def update_book(self, id: int, input: BookInput) -> MutationResult:
        """Update an existing book.

        TODO(human): Implement this mutation.

        Steps:
            1. Call update_book() from the data module with the book's ID and
               the fields from input (title, year, genre, description).
            2. If update_book returns None, the book was not found:
               return MutationResult(success=False, message="Book not found",
                                     errors=["No book with id {id}"])
            3. If successful, return MutationResult with success=True, the
               updated BookType, and a message like "Book updated successfully".

        Hint:
            from data import update_book as store_update_book
        """
        raise NotImplementedError("TODO(human): implement update_book mutation")
