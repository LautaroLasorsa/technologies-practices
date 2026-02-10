"""In-memory data store for the Bookstore API.

Contains sample books and authors, CRUD helpers, and a subscription event queue.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Domain models (plain Python — NOT Strawberry types)
# ---------------------------------------------------------------------------

@dataclass
class Author:
    id: int
    name: str
    bio: str


@dataclass
class Book:
    id: int
    title: str
    author_id: int
    year: int
    genre: str
    description: str = ""


# ---------------------------------------------------------------------------
# Pre-populated data
# ---------------------------------------------------------------------------

_authors: list[Author] = [
    Author(id=1, name="Gabriel Garcia Marquez", bio="Colombian novelist, Nobel Prize 1982"),
    Author(id=2, name="Jorge Luis Borges", bio="Argentine short-story writer and essayist"),
    Author(id=3, name="Julio Cortazar", bio="Argentine novelist and short-story writer"),
    Author(id=4, name="Isabel Allende", bio="Chilean novelist, known for magical realism"),
    Author(id=5, name="Mario Vargas Llosa", bio="Peruvian novelist, Nobel Prize 2010"),
]

_next_book_id: int = 11

_books: list[Book] = [
    Book(id=1, title="One Hundred Years of Solitude", author_id=1, year=1967, genre="Magical Realism",
         description="The multi-generational story of the Buendia family in Macondo"),
    Book(id=2, title="Love in the Time of Cholera", author_id=1, year=1985, genre="Romance",
         description="A love story spanning fifty years"),
    Book(id=3, title="Ficciones", author_id=2, year=1944, genre="Short Stories",
         description="A collection of fantastical short stories"),
    Book(id=4, title="The Aleph", author_id=2, year=1949, genre="Short Stories",
         description="Stories exploring infinity and identity"),
    Book(id=5, title="Hopscotch", author_id=3, year=1963, genre="Experimental Fiction",
         description="A novel that can be read in multiple orders"),
    Book(id=6, title="Blow-Up and Other Stories", author_id=3, year=1959, genre="Short Stories",
         description="Surreal short stories blending reality and fantasy"),
    Book(id=7, title="The House of the Spirits", author_id=4, year=1982, genre="Magical Realism",
         description="A family saga set against political upheaval in Chile"),
    Book(id=8, title="Eva Luna", author_id=4, year=1987, genre="Fiction",
         description="The story of a young storyteller in South America"),
    Book(id=9, title="The Time of the Hero", author_id=5, year=1963, genre="Fiction",
         description="Life in a Lima military academy"),
    Book(id=10, title="Conversation in the Cathedral", author_id=5, year=1969, genre="Fiction",
         description="A sprawling portrait of Peruvian society under dictatorship"),
]


# ---------------------------------------------------------------------------
# Subscription event queue
# ---------------------------------------------------------------------------

book_added_queue: asyncio.Queue[Book] = asyncio.Queue()


# ---------------------------------------------------------------------------
# CRUD helpers
# ---------------------------------------------------------------------------

def get_all_books() -> list[Book]:
    """Return a shallow copy of all books."""
    return list(_books)


def get_book_by_id(book_id: int) -> Book | None:
    """Find a book by its ID, or None if not found."""
    return next((b for b in _books if b.id == book_id), None)


def get_author_by_id(author_id: int) -> Author | None:
    """Find an author by their ID, or None if not found."""
    return next((a for a in _authors if a.id == author_id), None)


def get_authors_by_ids(author_ids: list[int]) -> list[Author | None]:
    """Batch-fetch authors by IDs, preserving order.

    Returns a list of the same length as *author_ids*: each position holds the
    matching Author or None if no author has that ID.
    """
    index = {a.id: a for a in _authors}
    return [index.get(aid) for aid in author_ids]


def add_book(title: str, author_id: int, year: int, genre: str, description: str = "") -> Book:
    """Create and store a new book. Returns the created book."""
    global _next_book_id
    book = Book(
        id=_next_book_id,
        title=title,
        author_id=author_id,
        year=year,
        genre=genre,
        description=description,
    )
    _next_book_id += 1
    _books.append(book)
    return book


def update_book(
    book_id: int,
    *,
    title: str | None = None,
    year: int | None = None,
    genre: str | None = None,
    description: str | None = None,
) -> Book | None:
    """Update a book's fields in-place. Returns the updated book or None if not found."""
    book = get_book_by_id(book_id)
    if book is None:
        return None
    if title is not None:
        book.title = title
    if year is not None:
        book.year = year
    if genre is not None:
        book.genre = genre
    if description is not None:
        book.description = description
    return book
