"""Cursor-based pagination helpers (Relay Connection spec).

Cursors are opaque base64-encoded strings that encode a book's position
in the list. Clients pass `after` (a cursor) and `first` (page size) to
navigate forward through the result set.

Why cursors instead of offset/limit?
    - Offsets break when items are inserted or deleted between pages.
    - Cursors are stable: they refer to a specific item, not a position.

See: https://graphql.org/learn/pagination/
     https://relay.dev/graphql/connections.htm
"""

from __future__ import annotations

import base64
from typing import Optional

from data import get_all_books
from gql_types import BookConnection, BookEdge, BookType, PageInfo


# ---------------------------------------------------------------------------
# Cursor encoding/decoding (fully implemented)
# ---------------------------------------------------------------------------

def encode_cursor(index: int) -> str:
    """Encode a list index as an opaque base64 cursor string."""
    return base64.b64encode(f"cursor:{index}".encode()).decode()


def decode_cursor(cursor: str) -> int:
    """Decode a base64 cursor string back to a list index.

    Raises ValueError if the cursor format is invalid.
    """
    decoded = base64.b64decode(cursor.encode()).decode()
    if not decoded.startswith("cursor:"):
        raise ValueError(f"Invalid cursor format: {decoded}")
    return int(decoded.removeprefix("cursor:"))


# ---------------------------------------------------------------------------
# Pagination function
# ---------------------------------------------------------------------------

def paginate_books(first: int = 5, after: Optional[str] = None) -> BookConnection:
    """Return a paginated slice of books as a Relay Connection.

    TODO(human): Implement cursor-based pagination.

    Parameters:
        first  — how many books to return (page size).
        after  — cursor of the last item on the previous page (or None for
                 the first page).

    Steps:
        1. Get all books with get_all_books().
        2. Determine the start index:
           - If `after` is None, start at index 0.
           - Otherwise, decode the cursor to get the index of the last-seen
             item, then start at index + 1.
        3. Slice the books list: books[start : start + first].
        4. Build BookEdge objects for each book in the slice. Each edge has:
           - cursor: encode_cursor(original_index_in_full_list)
           - node: BookType.from_model(book)
        5. Build PageInfo:
           - has_next_page: True if there are more books after this slice
           - has_previous_page: True if start > 0
           - start_cursor: cursor of the first edge (or None if empty)
           - end_cursor: cursor of the last edge (or None if empty)
        6. Return BookConnection(edges=edges, page_info=page_info,
                                 total_count=len(all_books)).

    Example (first=3, after=None with 10 books):
        start=0, slice=[book0, book1, book2]
        edges = [Edge(cursor="cursor:0", node=book0), ...]
        page_info = PageInfo(has_next=True, has_prev=False,
                             start_cursor="cursor:0", end_cursor="cursor:2")
        total_count = 10

    Hint: Use encode_cursor(start + i) for each edge's cursor,
          where i is the position within the slice.
    """
    raise NotImplementedError("TODO(human): implement paginate_books")
