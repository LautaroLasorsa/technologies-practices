"""
Practice 058a — SQLModel Model Definitions

This module defines the data models for the Hero/Team domain. SQLModel's key
innovation is that a SINGLE class can serve as both a SQLAlchemy table model
AND a Pydantic validation schema.

The distinction is controlled by the `table=True` parameter:
  - `class Hero(SQLModel, table=True)` → creates a real database table.
    SQLModel generates a SQLAlchemy Table behind the scenes, so Hero instances
    can be added to sessions, queried, committed, etc.
  - `class HeroCreate(SQLModel)` → a plain Pydantic model (no table).
    Used for request validation, serialization, and data transfer. No
    database interaction.

This separation (table model vs. read/write models) is the recommended pattern
from SQLModel's official docs. It prevents leaking internal DB details (like
auto-generated IDs) into API inputs, and lets you control which fields are
required vs. optional for each operation (create, read, update).

Relationship() creates an in-Python link between related models. It does NOT
create a database column — the actual DB link is the Foreign Key. Relationship()
just tells SQLModel to load the related objects when you access the attribute.
The `back_populates` parameter makes the relationship bidirectional: if you
set hero.team, then team.heroes automatically includes that hero.
"""

from sqlmodel import Field, Relationship, SQLModel


# ---------------------------------------------------------------------------
# Table models — these map to real PostgreSQL tables
# ---------------------------------------------------------------------------


class Team(SQLModel, table=True):
    """Team table model.

    Represents a team that heroes can belong to. The `heroes` relationship
    attribute provides access to all Hero objects linked to this team via
    their `team_id` foreign key. SQLModel uses SQLAlchemy's relationship
    loading under the hood — by default it's lazy-loaded (queried on access).
    """

    # TODO(human): Define the Team table model with the following columns:
    #
    # WHAT TO DO:
    #   1. `id`: Optional[int] field, primary key. Use Field(default=None,
    #      primary_key=True). It's Optional because the DB auto-generates it —
    #      when creating a new Team, you pass id=None and PostgreSQL assigns
    #      the next sequence value.
    #
    #   2. `name`: str field, indexed for fast lookups. Use Field(index=True).
    #      The index creates a B-tree index on the name column in PostgreSQL,
    #      making WHERE name = '...' queries O(log n) instead of O(n).
    #
    #   3. `headquarters`: str field. Plain column, no special attributes.
    #
    #   4. `heroes`: list["Hero"] relationship attribute. Use:
    #        heroes: list["Hero"] = Relationship(back_populates="team")
    #      This tells SQLModel: "when I access team.heroes, load all Hero rows
    #      whose team_id points to this team's id." The string "Hero" is a
    #      forward reference (Hero is defined below). back_populates="team"
    #      means Hero.team is the other side of this relationship.
    #
    # WHY THIS MATTERS:
    #   This single class definition replaces what would be THREE separate
    #   things in raw SQLAlchemy: a Table() declaration, a mapped class, and
    #   a Pydantic schema. SQLModel unifies them. The Relationship() attribute
    #   doesn't add a column to the DB — it's purely a Python-side convenience
    #   that triggers SQL JOINs or subqueries when accessed.
    #
    # EXPECTED RESULT:
    #   After implementing, the `teams` table in PostgreSQL will have columns:
    #   id (serial primary key), name (varchar with index), headquarters (varchar).
    #
    # HINT:
    #   id: int | None = Field(default=None, primary_key=True)
    #   heroes: list["Hero"] = Relationship(back_populates="team")

    raise NotImplementedError("TODO(human)")


class Hero(SQLModel, table=True):
    """Hero table model.

    Each hero optionally belongs to a team via team_id foreign key. The `team`
    relationship attribute provides direct access to the related Team object
    without writing a manual JOIN query.
    """

    # TODO(human): Define the Hero table model with the following columns:
    #
    # WHAT TO DO:
    #   1. `id`: Optional[int] field, primary key. Same pattern as Team.id.
    #
    #   2. `name`: str field, indexed. This is the hero's public name
    #      (e.g., "Spider-Man"). The index speeds up name-based searches.
    #
    #   3. `secret_name`: str field. The hero's real identity (e.g., "Peter
    #      Parker"). No index needed — we rarely search by secret name.
    #
    #   4. `age`: Optional[int] field with default=None. Some heroes don't
    #      have a known age. Use: age: int | None = Field(default=None).
    #
    #   5. `team_id`: Optional[int] foreign key pointing to the teams table.
    #      Use: team_id: int | None = Field(default=None, foreign_key="team.id")
    #      The string "team.id" refers to the `id` column of the `team` table
    #      (SQLModel lowercases the class name to derive the table name).
    #      It's Optional because a hero can exist without belonging to any team.
    #
    #   6. `team`: Optional["Team"] relationship attribute. Use:
    #        team: Team | None = Relationship(back_populates="heroes")
    #      This is the other side of Team.heroes. When you set hero.team = some_team,
    #      SQLModel automatically sets hero.team_id = some_team.id AND adds this
    #      hero to some_team.heroes.
    #
    # WHY THIS MATTERS:
    #   The foreign_key parameter creates the actual DB constraint (ON DELETE, etc.).
    #   The Relationship() is the Python-side ORM convenience. Understanding this
    #   distinction is critical: foreign_key = database integrity,
    #   Relationship() = Python object navigation. You need BOTH for a working
    #   bidirectional relationship.
    #
    # EXPECTED RESULT:
    #   The `hero` table will have: id (serial PK), name (varchar, indexed),
    #   secret_name (varchar), age (integer, nullable), team_id (integer, nullable,
    #   FK → team.id).
    #
    # HINT:
    #   team_id: int | None = Field(default=None, foreign_key="team.id")
    #   team: Team | None = Relationship(back_populates="heroes")

    raise NotImplementedError("TODO(human)")


# ---------------------------------------------------------------------------
# Read/write models — Pydantic-only (no table=True), used for API schemas
# ---------------------------------------------------------------------------
# These models control what data the API accepts (Create) and returns (Read).
# Without them, the API would expose internal details like auto-generated IDs
# in creation requests, or require all fields on partial updates.


class HeroCreate(SQLModel):
    """Schema for creating a new hero (POST request body).

    No `id` field — the database generates it. No `team_id` here either;
    team assignment is handled separately or via a dedicated endpoint.
    """

    name: str
    secret_name: str
    age: int | None = None
    team_id: int | None = None


class HeroRead(SQLModel):
    """Schema for reading a hero (GET response body).

    Includes `id` because the client needs it for subsequent requests
    (update, delete, fetch by ID). All fields are present.
    """

    id: int
    name: str
    secret_name: str
    age: int | None = None
    team_id: int | None = None


class HeroUpdate(SQLModel):
    """Schema for partial hero updates (PATCH request body).

    ALL fields are Optional — the client sends only the fields they want
    to change. We use model_dump(exclude_unset=True) to distinguish between
    "field not sent" (excluded) and "field explicitly set to None" (included).
    """

    name: str | None = None
    secret_name: str | None = None
    age: int | None = None
    team_id: int | None = None


class TeamCreate(SQLModel):
    """Schema for creating a new team (POST request body)."""

    name: str
    headquarters: str


class TeamRead(SQLModel):
    """Schema for reading a team (GET response body)."""

    id: int
    name: str
    headquarters: str


class TeamReadWithHeroes(SQLModel):
    """Schema for reading a team with its heroes eagerly loaded."""

    id: int
    name: str
    headquarters: str
    heroes: list[HeroRead] = []


class TeamUpdate(SQLModel):
    """Schema for partial team updates (PATCH request body)."""

    name: str | None = None
    headquarters: str | None = None
