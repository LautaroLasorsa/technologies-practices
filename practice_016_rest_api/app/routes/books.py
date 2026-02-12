"""Book endpoints — the main learning module.

This file has the route structure and boilerplate already in place.
Your job is to implement the endpoint bodies marked with TODO(human).

Use the authors.py file as a reference — it shows the same patterns
(status codes, HATEOAS links, error handling) fully implemented.

Concepts to practice:
- Correct HTTP status codes for each method
- RFC 9457 error responses via raise_problem()
- Offset pagination with _pagination metadata
- Query param filtering and sorting
- HATEOAS _links on single and collection responses
- Idempotency-Key header for safe POST retries
"""

from datetime import datetime, timezone

from fastapi import APIRouter, Header, Query, Request, Response

from app.models import (
    BookCreate,
    BookResponse,
    BookUpdate,
    Link,
    PaginatedBooks,
    PaginationMeta,
    generate_id,
)
from app import store
from app.errors import raise_problem

router = APIRouter(prefix="/v1/books", tags=["Books"])


# ---------------------------------------------------------------------------
# Helpers (fully implemented)
# ---------------------------------------------------------------------------

def _book_to_response(book: dict) -> BookResponse:
    """Convert an internal book dict to a BookResponse with HATEOAS links.

    # ── Exercise Context ──────────────────────────────────────────────────
    # This exercise teaches HATEOAS link construction. HATEOAS (Hypermedia As
    # The Engine Of Application State) is REST's Level 3 (Richardson Maturity
    # Model): responses guide clients to available actions via links. This decouples
    # clients from server URL schemes and enables API discoverability. Production
    # APIs use HATEOAS for workflows (next steps in a multi-step process) and
    # navigation (related resources).
    # ──────────────────────────────────────────────────────────────────────

    TODO(human): Build the _links list. Each book response should include:
      - "self"       -> GET /v1/books/{book_id}
      - "author"     -> GET /v1/authors/{author_id}
      - "collection" -> GET /v1/books
      - "update"     -> PUT /v1/books/{book_id}
      - "delete"     -> DELETE /v1/books/{book_id}

    Hint: use the Link model — Link(href="...", method="...", rel="...")
    """
    links: list[Link] = []
    # TODO(human): populate `links` with the HATEOAS links described above

    return BookResponse(**book, links=links)


# ---------------------------------------------------------------------------
# GET /v1/books — List books (paginated, filterable, sortable)
# ---------------------------------------------------------------------------

@router.get(
    "",
    response_model=PaginatedBooks,
    summary="List books with pagination, filtering, and sorting",
    operation_id="listBooks",
)
async def list_books(
    request: Request,
    # Pagination
    limit: int = Query(10, ge=1, le=100, description="Items per page"),
    offset: int = Query(0, ge=0, description="Number of items to skip"),
    # Filtering
    author_id: str | None = Query(None, description="Filter by author ID"),
    genre: str | None = Query(None, description="Filter by genre (case-insensitive)"),
    year_min: int | None = Query(None, description="Filter: year >= year_min"),
    year_max: int | None = Query(None, description="Filter: year <= year_max"),
    # Sorting
    sort_by: str = Query("title", pattern="^(title|year)$", description="Field to sort by"),
    order: str = Query("asc", pattern="^(asc|desc)$", description="Sort order"),
) -> PaginatedBooks:
    """Return a paginated, filtered, sorted list of books.

    # ── Exercise Context ──────────────────────────────────────────────────
    # This exercise teaches RESTful collection endpoint design: pagination,
    # filtering, sorting, and HATEOAS navigation links. Offset pagination is
    # simple but inefficient for large datasets (cursor-based pagination scales
    # better). Returning next/prev links enables client pagination without hardcoding
    # URL logic. Production APIs add ETag headers for caching and conditional requests.
    # ──────────────────────────────────────────────────────────────────────

    TODO(human): Implement this endpoint. Steps:

    1. FILTERING — Start with all books from store.books.values(), then:
       - If author_id is provided, keep only books where book["author_id"] == author_id
       - If genre is provided, keep only books where book["genre"].lower() == genre.lower()
       - If year_min is provided, keep only books where book["year"] >= year_min
       - If year_max is provided, keep only books where book["year"] <= year_max

    2. SORTING — Sort the filtered list:
       - Use sort_by as the dict key (either "title" or "year")
       - Reverse if order == "desc"

    3. PAGINATION — Apply offset/limit:
       - total = len(filtered_list)  (before slicing!)
       - page = filtered_list[offset : offset + limit]

    4. PAGINATION LINKS — Build next/prev URLs:
       - base_url = str(request.url).split("?")[0]  (strip existing query params)
       - next_url = f"{base_url}?limit={limit}&offset={offset + limit}" if offset + limit < total else None
       - prev_url = f"{base_url}?limit={limit}&offset={max(0, offset - limit)}" if offset > 0 else None

    5. BUILD RESPONSE — Return PaginatedBooks with:
       - data: list of _book_to_response(book) for each book in the page
       - pagination: PaginationMeta(total=total, limit=limit, offset=offset, next=next_url, prev=prev_url)
       - links: collection-level HATEOAS links (self, next, prev, first)

    Hint: Look at the PaginatedBooks and PaginationMeta models in models.py.
    """
    # TODO(human): implement filtering, sorting, pagination, and return PaginatedBooks
    return PaginatedBooks(
        data=[],
        pagination=PaginationMeta(total=0, limit=limit, offset=offset, next=None, prev=None),
        links=[],
    )


# ---------------------------------------------------------------------------
# POST /v1/books — Create a book
# ---------------------------------------------------------------------------

@router.post(
    "",
    response_model=BookResponse,
    status_code=201,
    summary="Create a new book",
    operation_id="createBook",
)
async def create_book(
    body: BookCreate,
    response: Response,
    idempotency_key: str | None = Header(None, alias="Idempotency-Key"),
) -> BookResponse:
    """Create a new book resource.

    # ── Exercise Context ──────────────────────────────────────────────────
    # This exercise teaches idempotent POST via the Idempotency-Key header.
    # POST is not naturally idempotent (retries create duplicates), but clients
    # can send a unique key; the server caches responses keyed by this header.
    # This is critical for payment APIs (prevent double-charging on retries) and
    # production systems with unreliable networks. Stripe, PayPal, and Twilio all
    # use idempotency keys. Return 201 Created with Location header (RFC 7231).
    # ──────────────────────────────────────────────────────────────────────

    TODO(human): Implement this endpoint. Steps:

    1. IDEMPOTENCY CHECK — If idempotency_key is not None:
       - Look it up in store.idempotency_cache
       - If found, return the cached (status_code, body) immediately
         (set response.status_code to the cached status code and return the cached body)

    2. VALIDATE AUTHOR — Check that body.author_id exists in store.authors:
       - If not, call raise_problem(404, f"Author '{body.author_id}' does not exist.", ...)

    3. CREATE BOOK — Build the book dict:
       - id = generate_id("book")
       - created_at = updated_at = datetime.now(timezone.utc)
       - Store it in store.books[book_id]

    4. SET LOCATION HEADER:
       - response.headers["Location"] = f"/v1/books/{book_id}"

    5. CACHE FOR IDEMPOTENCY — If idempotency_key was provided:
       - Store (201, response_dict) in store.idempotency_cache[idempotency_key]

    6. RETURN — _book_to_response(book_dict)

    Key concept: POST is not idempotent by default. The Idempotency-Key header
    lets clients safely retry failed requests without creating duplicates.
    """
    # TODO(human): implement creation with idempotency support
    raise_problem(501, "Not implemented yet — this is your task!")


# ---------------------------------------------------------------------------
# GET /v1/books/{book_id} — Get a single book
# ---------------------------------------------------------------------------

@router.get(
    "/{book_id}",
    response_model=BookResponse,
    summary="Get a book by ID",
    operation_id="getBook",
)
async def get_book(book_id: str) -> BookResponse:
    """Return a single book by its ID.

    # ── Exercise Context ──────────────────────────────────────────────────
    # This exercise teaches safe, idempotent GET semantics. GET must not modify
    # state (safe) and must return the same result on repeat (idempotent). This
    # enables HTTP caching (proxies, CDNs) and safe retries. Return 200 OK with
    # HATEOAS links to guide clients. Production APIs add ETag headers for
    # conditional requests (304 Not Modified when client cache is fresh).
    # ──────────────────────────────────────────────────────────────────────

    TODO(human): Implement this endpoint. Steps:

    1. Look up book_id in store.books
    2. If not found, call raise_problem(404, f"Book '{book_id}' does not exist.", instance=f"/v1/books/{book_id}")
    3. If found, return _book_to_response(book)

    Key concept: GET is safe (no side effects) and idempotent (same result on repeat).
    Return 200 on success, 404 if missing.
    """
    # TODO(human): implement single-book retrieval
    raise_problem(501, "Not implemented yet — this is your task!")


# ---------------------------------------------------------------------------
# PUT /v1/books/{book_id} — Full replace
# ---------------------------------------------------------------------------

@router.put(
    "/{book_id}",
    response_model=BookResponse,
    summary="Replace a book entirely",
    operation_id="replaceBook",
)
async def replace_book(book_id: str, body: BookUpdate) -> BookResponse:
    """Fully replace an existing book.

    # ── Exercise Context ──────────────────────────────────────────────────
    # This exercise teaches PUT semantics: full replacement (all fields required).
    # PUT is idempotent: sending the same body twice yields identical results.
    # This differs from PATCH (partial update, may not be idempotent depending on
    # format). Production APIs often use optimistic concurrency (If-Match/ETag) to
    # prevent lost updates from concurrent clients. Return 200 OK with updated
    # resource (some APIs return 204 No Content without body for efficiency).
    # ──────────────────────────────────────────────────────────────────────

    TODO(human): Implement this endpoint. Steps:

    1. Look up book_id in store.books
    2. If not found, raise_problem(404, ...)
    3. Validate that body.author_id exists in store.authors (if not, raise_problem(404, ...))
    4. Update ALL fields from body, set updated_at = datetime.now(timezone.utc)
       Keep the original id and created_at unchanged.
    5. Return _book_to_response(updated_book)

    Key concept: PUT means full replacement — ALL fields in the body are required.
    This differs from PATCH which allows partial updates.
    Return 200 on success (the resource existed and was replaced).
    """
    # TODO(human): implement full replacement
    raise_problem(501, "Not implemented yet — this is your task!")


# ---------------------------------------------------------------------------
# DELETE /v1/books/{book_id} — Delete a book
# ---------------------------------------------------------------------------

@router.delete(
    "/{book_id}",
    status_code=204,
    summary="Delete a book",
    operation_id="deleteBook",
)
async def delete_book(book_id: str) -> None:
    """Delete a book by its ID.

    # ── Exercise Context ──────────────────────────────────────────────────
    # This exercise teaches DELETE semantics: idempotent removal. Deleting a
    # deleted resource can return 404 (resource no longer exists) or 204 (operation
    # succeeded, no-op). Both are valid; 204 emphasizes idempotency. Return 204
    # No Content with empty body (RFC 7231). Production APIs may implement soft
    # deletes (mark as deleted, don't physically remove) for audit trails and undo.
    # ──────────────────────────────────────────────────────────────────────

    TODO(human): Implement this endpoint. Steps:

    1. Check if book_id exists in store.books
    2. If not found, raise_problem(404, ...)
    3. Delete it: del store.books[book_id]
    4. Return None (FastAPI will send 204 No Content with empty body)

    Key concept: DELETE is idempotent — deleting the same resource twice should
    not error (some APIs return 404 on second delete, others return 204; both
    are acceptable, but returning 404 is more informative).
    Return 204 No Content on success — no body needed.
    """
    # TODO(human): implement deletion
    raise_problem(501, "Not implemented yet — this is your task!")
