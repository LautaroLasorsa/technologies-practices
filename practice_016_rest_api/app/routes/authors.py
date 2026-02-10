"""Author endpoints.

Fully implemented — serves as a reference for the user when implementing
the book endpoints. Demonstrates:
- Proper status codes (201 for creation, 200 for reads)
- Location header on POST
- HATEOAS _links in responses
"""

from datetime import datetime, timezone

from fastapi import APIRouter, Response

from app.models import AuthorCreate, AuthorResponse, Link, generate_id
from app import store
from app.errors import raise_problem

router = APIRouter(prefix="/v1/authors", tags=["Authors"])


def _build_author_links(author_id: str) -> list[Link]:
    """Build HATEOAS links for a single author."""
    return [
        Link(href=f"/v1/authors/{author_id}", method="GET", rel="self"),
        Link(href=f"/v1/books?author_id={author_id}", method="GET", rel="books"),
        Link(href="/v1/authors", method="GET", rel="collection"),
    ]


@router.get(
    "",
    response_model=list[AuthorResponse],
    summary="List all authors",
    operation_id="listAuthors",
)
async def list_authors() -> list[AuthorResponse]:
    results = []
    for author in store.authors.values():
        results.append(
            AuthorResponse(
                **author,
                links=_build_author_links(author["id"]),
            )
        )
    return results


@router.post(
    "",
    response_model=AuthorResponse,
    status_code=201,
    summary="Create a new author",
    operation_id="createAuthor",
)
async def create_author(body: AuthorCreate, response: Response) -> AuthorResponse:
    author_id = generate_id("auth")
    now = datetime.now(timezone.utc)

    author = {
        "id": author_id,
        "name": body.name,
        "bio": body.bio,
        "created_at": now,
    }
    store.authors[author_id] = author

    response.headers["Location"] = f"/v1/authors/{author_id}"

    return AuthorResponse(**author, links=_build_author_links(author_id))


@router.get(
    "/{author_id}",
    response_model=AuthorResponse,
    summary="Get an author by ID",
    operation_id="getAuthor",
)
async def get_author(author_id: str) -> AuthorResponse:
    author = store.authors.get(author_id)
    if author is None:
        raise_problem(404, f"Author '{author_id}' does not exist.", instance=f"/v1/authors/{author_id}")

    return AuthorResponse(**author, links=_build_author_links(author_id))
