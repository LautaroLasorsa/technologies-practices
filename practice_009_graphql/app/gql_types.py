"""Strawberry GraphQL types for the Bookstore API.

Defines the public schema types that clients see: BookType, AuthorType,
input types, mutation results, and pagination types (Relay Connection spec).
"""

from __future__ import annotations

from typing import Optional

import strawberry
from strawberry.types import Info

from data import Author, Book, get_author_by_id


# ---------------------------------------------------------------------------
# Core types
# ---------------------------------------------------------------------------

@strawberry.type
class AuthorType:
    """A book author."""

    id: int
    name: str
    bio: str

    @staticmethod
    def from_model(author: Author) -> AuthorType:
        return AuthorType(id=author.id, name=author.name, bio=author.bio)


@strawberry.type
class BookType:
    """A book in the store."""

    id: int
    title: str
    author_id: int
    year: int
    genre: str
    description: str

    @staticmethod
    def from_model(book: Book) -> BookType:
        return BookType(
            id=book.id,
            title=book.title,
            author_id=book.author_id,
            year=book.year,
            genre=book.genre,
            description=book.description,
        )

    @strawberry.field
    async def author(self, info: Info) -> Optional[AuthorType]:
        """Resolve the author for this book.

        TODO(human): Implement this resolver.

        PHASE 2 — Simple version (causes N+1 problem):
            Use get_author_by_id(self.author_id) to look up the author.
            Convert to AuthorType using AuthorType.from_model().
            Return None if the author is not found.

        PHASE 4 — DataLoader version (solves N+1):
            Replace the direct lookup with:
                loader = info.context.author_loader
                author = await loader.load(self.author_id)
            This batches all author lookups into a single call.

        Hint: Start with the simple version in Phase 2, then refactor
        in Phase 4 once you understand the N+1 problem.
        """
        raise NotImplementedError("TODO(human): implement author resolver")


# ---------------------------------------------------------------------------
# Input types
# ---------------------------------------------------------------------------

@strawberry.input
class BookInput:
    """Input for creating or updating a book."""

    title: str
    author_id: int
    year: int
    genre: str
    description: str = ""


# ---------------------------------------------------------------------------
# Mutation result
# ---------------------------------------------------------------------------

@strawberry.type
class MutationResult:
    """Standard mutation result with errors-as-data pattern."""

    success: bool
    message: str
    book: Optional[BookType] = None
    errors: list[str] = strawberry.field(default_factory=list)


# ---------------------------------------------------------------------------
# Pagination types (Relay Connection spec)
# ---------------------------------------------------------------------------

@strawberry.type
class PageInfo:
    """Pagination metadata following the Relay Connection spec."""

    has_next_page: bool
    has_previous_page: bool
    start_cursor: Optional[str] = None
    end_cursor: Optional[str] = None


@strawberry.type
class BookEdge:
    """A single edge in a paginated book list."""

    cursor: str
    node: BookType


@strawberry.type
class BookConnection:
    """Paginated list of books following the Relay Connection spec."""

    edges: list[BookEdge]
    page_info: PageInfo
    total_count: int
