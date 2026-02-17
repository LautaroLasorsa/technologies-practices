"""
Practice 058a — FastAPI Router with SQLModel Endpoints

This module defines the HTTP API endpoints. FastAPI + SQLModel integration
provides automatic:
  - Request body validation (HeroCreate → Pydantic validation on POST body)
  - Response serialization (HeroRead → only declared fields appear in JSON)
  - OpenAPI/Swagger docs generation (from type annotations + models)
  - Dependency injection (Depends(get_session) → fresh Session per request)

The `response_model` parameter on each endpoint controls what the client sees.
Even though the CRUD functions return full Hero objects (with all SQLAlchemy
internals), FastAPI serializes them through HeroRead, stripping anything not
in the schema. This is the "read model" half of the read/write model pattern.
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlmodel import Session

from app.crud import create_hero, delete_hero, get_hero_by_id, get_heroes, update_hero
from app.database import get_session
from app.models import HeroCreate, HeroRead, HeroUpdate

router = APIRouter(prefix="/heroes", tags=["heroes"])


# ---------------------------------------------------------------------------
# POST /heroes/ — Create a hero
# ---------------------------------------------------------------------------


@router.post("/", response_model=HeroRead, status_code=201)
def create_hero_endpoint(
    hero: HeroCreate,
    session: Session = Depends(get_session),
) -> HeroRead:
    # TODO(human): Create a hero and return it as HeroRead.
    #
    # WHAT TO DO:
    #   1. Call the CRUD function: create_hero(session, hero)
    #      This handles add → commit → refresh and returns a Hero with an id.
    #
    #   2. Return the result directly.
    #      FastAPI will serialize it through the response_model=HeroRead schema,
    #      including only the fields defined in HeroRead (id, name, secret_name,
    #      age, team_id). Any extra SQLAlchemy internals are stripped.
    #
    # HOW IT WORKS:
    #   - `hero: HeroCreate` in the signature tells FastAPI to parse the JSON
    #     request body as HeroCreate. If validation fails (missing name, wrong
    #     type), FastAPI returns a 422 Unprocessable Entity AUTOMATICALLY.
    #   - `session: Session = Depends(get_session)` triggers the dependency
    #     injection: FastAPI calls get_session(), gets a Session, and passes it.
    #   - `response_model=HeroRead` + `status_code=201` → on success, the
    #     response is 201 Created with the hero JSON.
    #
    # WHY THIS MATTERS:
    #   This single function signature replaces what would typically require:
    #   manual JSON parsing, validation, error formatting, session management,
    #   and response serialization. SQLModel + FastAPI handles all of it
    #   through type annotations.
    #
    # EXPECTED RESULT:
    #   POST /heroes/ with body {"name": "Spider-Man", "secret_name": "Peter Parker"}
    #   → 201 {"id": 1, "name": "Spider-Man", "secret_name": "Peter Parker", "age": null, "team_id": null}
    #
    # HINT:
    #   return create_hero(session, hero)

    raise NotImplementedError("TODO(human)")


# ---------------------------------------------------------------------------
# GET /heroes/ — List heroes with pagination
# ---------------------------------------------------------------------------


@router.get("/", response_model=list[HeroRead])
def list_heroes_endpoint(
    offset: int = 0,
    limit: int = Query(default=100, le=100),
    session: Session = Depends(get_session),
) -> list[HeroRead]:
    # TODO(human): Return a paginated list of heroes.
    #
    # WHAT TO DO:
    #   1. Call the CRUD function: get_heroes(session, offset, limit)
    #
    #   2. Return the result.
    #      FastAPI serializes each Hero through HeroRead automatically.
    #
    # HOW IT WORKS:
    #   - `offset: int = 0` and `limit: int = Query(default=100, le=100)` are
    #     query parameters. FastAPI reads them from the URL:
    #     GET /heroes/?offset=10&limit=20
    #   - `Query(le=100)` adds validation: limit must be <= 100. If the client
    #     sends limit=500, FastAPI returns 422 without calling your function.
    #   - `response_model=list[HeroRead]` → the response is a JSON array of
    #     hero objects. FastAPI validates EACH element against HeroRead.
    #
    # WHY THIS MATTERS:
    #   Pagination prevents loading the entire table into memory. The le=100
    #   guard prevents clients from requesting unbounded result sets that could
    #   crash the server or timeout.
    #
    # EXPECTED RESULT:
    #   GET /heroes/?offset=0&limit=2 → [{"id": 1, ...}, {"id": 2, ...}]
    #
    # HINT:
    #   return get_heroes(session, offset, limit)

    raise NotImplementedError("TODO(human)")


# ---------------------------------------------------------------------------
# GET /heroes/{hero_id} — Get a single hero
# ---------------------------------------------------------------------------


@router.get("/{hero_id}", response_model=HeroRead)
def read_hero_endpoint(
    hero_id: int,
    session: Session = Depends(get_session),
) -> HeroRead:
    # TODO(human): Fetch a hero by ID, returning 404 if not found.
    #
    # WHAT TO DO:
    #   1. Call the CRUD function: get_hero_by_id(session, hero_id)
    #
    #   2. If the result is None, raise HTTPException(status_code=404,
    #      detail="Hero not found"). This tells FastAPI to return a 404
    #      JSON response: {"detail": "Hero not found"}.
    #
    #   3. Otherwise, return the hero.
    #
    # HOW IT WORKS:
    #   - `hero_id: int` in the path is a path parameter. FastAPI extracts it
    #     from the URL: GET /heroes/42 → hero_id=42. If the client sends
    #     /heroes/abc, FastAPI returns 422 (not a valid int).
    #   - HTTPException is FastAPI's way of returning error responses. It
    #     short-circuits the endpoint and returns the specified status code.
    #
    # WHY THIS MATTERS:
    #   Proper 404 handling is REST API 101. Without it, the client gets a
    #   200 with null/empty body, which is confusing and breaks client-side
    #   error handling. The HTTPException pattern standardizes error responses
    #   across all endpoints.
    #
    # EXPECTED RESULT:
    #   GET /heroes/1 → 200 {"id": 1, "name": "Spider-Man", ...}
    #   GET /heroes/999 → 404 {"detail": "Hero not found"}
    #
    # HINT:
    #   hero = get_hero_by_id(session, hero_id)
    #   if not hero:
    #       raise HTTPException(status_code=404, detail="Hero not found")
    #   return hero

    raise NotImplementedError("TODO(human)")


# ---------------------------------------------------------------------------
# PATCH /heroes/{hero_id} — Partial update
# ---------------------------------------------------------------------------


@router.patch("/{hero_id}", response_model=HeroRead)
def update_hero_endpoint(
    hero_id: int,
    hero_data: HeroUpdate,
    session: Session = Depends(get_session),
) -> HeroRead:
    """Partially update a hero. Only fields present in the request body are changed."""
    hero = update_hero(session, hero_id, hero_data)
    if not hero:
        raise HTTPException(status_code=404, detail="Hero not found")
    return hero


# ---------------------------------------------------------------------------
# DELETE /heroes/{hero_id} — Delete a hero
# ---------------------------------------------------------------------------


@router.delete("/{hero_id}", status_code=204)
def delete_hero_endpoint(
    hero_id: int,
    session: Session = Depends(get_session),
) -> None:
    """Delete a hero by ID. Returns 204 No Content on success, 404 if not found."""
    if not delete_hero(session, hero_id):
        raise HTTPException(status_code=404, detail="Hero not found")
