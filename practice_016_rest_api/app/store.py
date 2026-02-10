"""In-memory data store for the Book Collection API.

Fully implemented — no TODOs here. This is intentionally simple
(Python dicts) so the focus stays on API design, not persistence.
"""

from datetime import datetime, timezone

from app.models import generate_id


# ---------------------------------------------------------------------------
# Storage: plain dicts keyed by id
# ---------------------------------------------------------------------------

authors: dict[str, dict] = {}
books: dict[str, dict] = {}

# Idempotency cache: maps Idempotency-Key -> (status_code, response_body)
idempotency_cache: dict[str, tuple[int, dict]] = {}


# ---------------------------------------------------------------------------
# Seed data (so the API isn't empty on first run)
# ---------------------------------------------------------------------------

def seed() -> None:
    """Populate the store with sample data for immediate experimentation."""
    if authors or books:
        return  # already seeded

    now = datetime.now(timezone.utc)

    # Authors
    a1_id = "auth_marquez1"
    a2_id = "auth_orwell01"
    a3_id = "auth_tolkien1"

    authors[a1_id] = {
        "id": a1_id,
        "name": "Gabriel Garcia Marquez",
        "bio": "Colombian novelist, Nobel Prize in Literature 1982.",
        "created_at": now,
    }
    authors[a2_id] = {
        "id": a2_id,
        "name": "George Orwell",
        "bio": "English novelist and essayist, known for dystopian fiction.",
        "created_at": now,
    }
    authors[a3_id] = {
        "id": a3_id,
        "name": "J.R.R. Tolkien",
        "bio": "English writer and philologist, author of The Lord of the Rings.",
        "created_at": now,
    }

    # Books
    sample_books = [
        ("One Hundred Years of Solitude", a1_id, "Magical Realism", 1967, "9780060883287"),
        ("Love in the Time of Cholera", a1_id, "Romance", 1985, "9780307389732"),
        ("Chronicle of a Death Foretold", a1_id, "Novella", 1981, "9781400034710"),
        ("1984", a2_id, "Dystopian", 1949, "9780451524935"),
        ("Animal Farm", a2_id, "Political Satire", 1945, "9780451526342"),
        ("The Hobbit", a3_id, "Fantasy", 1937, "9780547928227"),
        ("The Fellowship of the Ring", a3_id, "Fantasy", 1954, "9780547928210"),
        ("The Two Towers", a3_id, "Fantasy", 1954, "9780547928203"),
        ("The Return of the King", a3_id, "Fantasy", 1955, "9780547928197"),
    ]

    for title, author_id, genre, year, isbn in sample_books:
        book_id = generate_id("book")
        books[book_id] = {
            "id": book_id,
            "title": title,
            "author_id": author_id,
            "genre": genre,
            "year": year,
            "isbn": isbn,
            "created_at": now,
            "updated_at": now,
        }
