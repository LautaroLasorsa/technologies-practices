"""Pydantic models for the Book Collection API.

Fully implemented — no TODOs here. These are the data contracts
that define what the API accepts and returns.

Key concepts demonstrated:
- Pydantic v2 model definition with Field metadata
- Separate Create/Update/Response models (never expose internal IDs in input)
- HATEOAS link model
- RFC 9457 Problem Detail model
- Pagination metadata model
"""

from datetime import datetime
from uuid import uuid4

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# HATEOAS Link
# ---------------------------------------------------------------------------

class Link(BaseModel):
    """A single hypermedia link (HATEOAS).

    Example: {"href": "/v1/books/abc123", "method": "GET", "rel": "self"}
    """

    href: str = Field(..., description="URL of the linked resource")
    method: str = Field("GET", description="HTTP method for the link")
    rel: str = Field(..., description="Relationship type (self, collection, author, next, prev, ...)")


# ---------------------------------------------------------------------------
# Author models
# ---------------------------------------------------------------------------

class AuthorCreate(BaseModel):
    """Payload for creating a new author."""

    name: str = Field(..., min_length=1, max_length=200, examples=["Gabriel Garcia Marquez"])
    bio: str | None = Field(None, max_length=2000, examples=["Colombian novelist and Nobel laureate."])


class AuthorResponse(BaseModel):
    """Author as returned by the API (includes server-generated fields)."""

    id: str = Field(..., examples=["auth_7f3a9b2e"])
    name: str = Field(..., examples=["Gabriel Garcia Marquez"])
    bio: str | None = Field(None, examples=["Colombian novelist and Nobel laureate."])
    created_at: datetime
    links: list[Link] = Field(default_factory=list, alias="_links")

    model_config = {"populate_by_name": True}


# ---------------------------------------------------------------------------
# Book models
# ---------------------------------------------------------------------------

class BookCreate(BaseModel):
    """Payload for creating a new book (POST /v1/books)."""

    title: str = Field(..., min_length=1, max_length=500, examples=["One Hundred Years of Solitude"])
    author_id: str = Field(..., examples=["auth_7f3a9b2e"])
    genre: str = Field(..., min_length=1, max_length=100, examples=["Magical Realism"])
    year: int = Field(..., ge=1000, le=2100, examples=[1967])
    isbn: str | None = Field(None, pattern=r"^\d{13}$", examples=["9780060883287"])


class BookUpdate(BaseModel):
    """Payload for full replacement of a book (PUT /v1/books/{book_id}).

    All fields required because PUT means full replacement.
    """

    title: str = Field(..., min_length=1, max_length=500, examples=["One Hundred Years of Solitude"])
    author_id: str = Field(..., examples=["auth_7f3a9b2e"])
    genre: str = Field(..., min_length=1, max_length=100, examples=["Magical Realism"])
    year: int = Field(..., ge=1000, le=2100, examples=[1967])
    isbn: str | None = Field(None, pattern=r"^\d{13}$", examples=["9780060883287"])


class BookResponse(BaseModel):
    """Book as returned by the API (includes server-generated fields + HATEOAS links)."""

    id: str = Field(..., examples=["book_a1b2c3d4"])
    title: str = Field(..., examples=["One Hundred Years of Solitude"])
    author_id: str = Field(..., examples=["auth_7f3a9b2e"])
    genre: str = Field(..., examples=["Magical Realism"])
    year: int = Field(..., examples=[1967])
    isbn: str | None = Field(None, examples=["9780060883287"])
    created_at: datetime
    updated_at: datetime
    links: list[Link] = Field(default_factory=list, alias="_links")

    model_config = {"populate_by_name": True}


# ---------------------------------------------------------------------------
# Pagination metadata
# ---------------------------------------------------------------------------

class PaginationMeta(BaseModel):
    """Metadata for paginated responses."""

    total: int = Field(..., description="Total number of items matching the query")
    limit: int = Field(..., description="Maximum items per page")
    offset: int = Field(..., description="Number of items skipped")
    next: str | None = Field(None, description="URL for the next page (null if last)")
    prev: str | None = Field(None, description="URL for the previous page (null if first)")


class PaginatedBooks(BaseModel):
    """Paginated collection of books with metadata and HATEOAS links."""

    data: list[BookResponse]
    pagination: PaginationMeta = Field(alias="_pagination")
    links: list[Link] = Field(default_factory=list, alias="_links")

    model_config = {"populate_by_name": True}


# ---------------------------------------------------------------------------
# RFC 9457 Problem Detail
# ---------------------------------------------------------------------------

class ProblemDetail(BaseModel):
    """RFC 9457 Problem Details for HTTP APIs.

    See: https://www.rfc-editor.org/rfc/rfc9457.html

    Example:
        {
            "type": "https://api.example.com/problems/not-found",
            "title": "Resource Not Found",
            "status": 404,
            "detail": "Book with id 'book_xyz' does not exist.",
            "instance": "/v1/books/book_xyz"
        }
    """

    type: str = Field(
        "about:blank",
        description="URI identifying the problem type",
        examples=["https://api.example.com/problems/not-found"],
    )
    title: str = Field(..., description="Short human-readable summary", examples=["Resource Not Found"])
    status: int = Field(..., description="HTTP status code", examples=[404])
    detail: str = Field(..., description="Explanation specific to this occurrence", examples=["Book 'book_xyz' does not exist."])
    instance: str | None = Field(None, description="URI of the request that caused the problem", examples=["/v1/books/book_xyz"])


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    """Response for GET /v1/health."""

    status: str = Field("ok", examples=["ok"])
    version: str = Field("1.0.0", examples=["1.0.0"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def generate_id(prefix: str) -> str:
    """Generate a short prefixed UUID (e.g., 'book_a1b2c3d4')."""
    return f"{prefix}_{uuid4().hex[:8]}"
