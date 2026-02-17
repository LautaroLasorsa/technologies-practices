"""
Practice 058a — CRUD Operations with SQLModel

This module implements Create, Read, Update, Delete operations using SQLModel's
query API. SQLModel wraps SQLAlchemy 2.0's `select()` statement — you build
queries with Python objects instead of raw SQL strings.

Key patterns:
  - select(Hero) → SELECT * FROM hero
  - select(Hero).where(Hero.id == 1) → SELECT * FROM hero WHERE id = 1
  - select(Hero).offset(10).limit(5) → SELECT * FROM hero OFFSET 10 LIMIT 5
  - session.add(obj) → stages an INSERT
  - session.commit() → flushes all staged changes to the DB
  - session.refresh(obj) → re-reads the object from DB (gets auto-generated fields like id)
  - session.delete(obj) → stages a DELETE
  - model.model_dump(exclude_unset=True) → Pydantic method returning only fields
    the client explicitly set (crucial for partial updates)
"""

from sqlmodel import Session, select

from app.models import Hero, HeroCreate, HeroUpdate


# ---------------------------------------------------------------------------
# Create
# ---------------------------------------------------------------------------


def create_hero(session: Session, hero_data: HeroCreate) -> Hero:
    # TODO(human): Create a new Hero in the database from a HeroCreate schema.
    #
    # WHAT TO DO:
    #   1. Convert the HeroCreate Pydantic model to a Hero table model:
    #        hero = Hero.model_validate(hero_data)
    #      model_validate() copies all matching fields from hero_data into a
    #      new Hero instance. Since Hero has table=True, this instance can
    #      be added to a session.
    #
    #   2. Add the hero to the session:  session.add(hero)
    #      This stages an INSERT — nothing hits the DB yet.
    #
    #   3. Commit the transaction:  session.commit()
    #      This flushes the INSERT to PostgreSQL and assigns the auto-generated
    #      id. The transaction is finalized.
    #
    #   4. Refresh the object:  session.refresh(hero)
    #      After commit, the hero object's id field is stale (still None in
    #      Python). refresh() re-reads it from the DB, populating id with
    #      the value PostgreSQL assigned.
    #
    #   5. Return the hero with its populated id.
    #
    # WHY THIS MATTERS:
    #   The add → commit → refresh cycle is the fundamental write pattern in
    #   SQLAlchemy/SQLModel. Understanding why refresh() is needed after
    #   commit() is important: commit() invalidates the session's identity
    #   map, so accessing attributes after commit requires a re-fetch.
    #
    # EXPECTED RESULT:
    #   Returns a Hero with a non-None id, persisted in PostgreSQL.
    #
    # HINT:
    #   hero = Hero.model_validate(hero_data)
    #   session.add(hero)
    #   session.commit()
    #   session.refresh(hero)
    #   return hero

    raise NotImplementedError("TODO(human)")


# ---------------------------------------------------------------------------
# Read (list with pagination)
# ---------------------------------------------------------------------------


def get_heroes(session: Session, offset: int = 0, limit: int = 100) -> list[Hero]:
    # TODO(human): Query all heroes with offset/limit pagination.
    #
    # WHAT TO DO:
    #   1. Build the query:  select(Hero).offset(offset).limit(limit)
    #      - select(Hero) generates: SELECT hero.id, hero.name, ... FROM hero
    #      - .offset(offset) adds: OFFSET {offset}
    #      - .limit(limit) adds: LIMIT {limit}
    #
    #   2. Execute the query:  session.exec(statement)
    #      session.exec() is SQLModel's wrapper around session.execute(). It
    #      returns properly typed results — you get Hero objects, not raw rows.
    #
    #   3. Collect results:  heroes = results.all()
    #      .all() fetches all matching rows into a Python list.
    #
    #   4. Return the list.
    #
    # WHY THIS MATTERS:
    #   Offset/limit pagination is the simplest pagination strategy. For large
    #   datasets, cursor-based pagination (WHERE id > last_seen_id LIMIT N)
    #   is more efficient because OFFSET requires scanning and discarding rows.
    #   But offset/limit is standard for small-to-medium datasets and is what
    #   most REST APIs implement.
    #
    # EXPECTED RESULT:
    #   Returns a list of Hero objects. With offset=0, limit=10, returns the
    #   first 10 heroes ordered by insertion order.
    #
    # HINT:
    #   statement = select(Hero).offset(offset).limit(limit)
    #   return session.exec(statement).all()

    raise NotImplementedError("TODO(human)")


# ---------------------------------------------------------------------------
# Read (single by ID)
# ---------------------------------------------------------------------------


def get_hero_by_id(session: Session, hero_id: int) -> Hero | None:
    # TODO(human): Fetch a single hero by primary key.
    #
    # WHAT TO DO:
    #   Use session.get(Hero, hero_id) — this is the optimal way to fetch
    #   by primary key. It:
    #     1. First checks the session's identity map (in-memory cache of
    #        already-loaded objects). If the hero was loaded earlier in this
    #        session, it returns instantly without hitting the DB.
    #     2. If not cached, issues: SELECT * FROM hero WHERE id = {hero_id}
    #     3. Returns None if no row exists with that id.
    #
    # WHY THIS MATTERS:
    #   session.get() vs select().where(): get() is optimized for PK lookups
    #   and uses the identity map. select().where(Hero.id == hero_id) always
    #   hits the DB. For PK lookups, always prefer session.get().
    #
    # EXPECTED RESULT:
    #   Returns the Hero if found, None otherwise.
    #
    # HINT:
    #   return session.get(Hero, hero_id)

    raise NotImplementedError("TODO(human)")


# ---------------------------------------------------------------------------
# Update (partial)
# ---------------------------------------------------------------------------


def update_hero(session: Session, hero_id: int, hero_data: HeroUpdate) -> Hero | None:
    # TODO(human): Partially update a hero — only change fields the client sent.
    #
    # WHAT TO DO:
    #   1. Fetch the existing hero: session.get(Hero, hero_id).
    #      Return None if not found.
    #
    #   2. Get only the fields the client explicitly set:
    #        update_dict = hero_data.model_dump(exclude_unset=True)
    #      This is the critical line. model_dump(exclude_unset=True) returns
    #      a dict containing ONLY the fields that were present in the request
    #      body. If the client sends {"name": "New Name"}, the dict is
    #      {"name": "New Name"} — age, secret_name, team_id are NOT included,
    #      so they won't be overwritten.
    #
    #   3. Apply the updates:
    #        for key, value in update_dict.items():
    #            setattr(hero, key, value)
    #      setattr() dynamically sets hero.name = "New Name", etc.
    #
    #   4. Add, commit, refresh, and return the updated hero.
    #
    # WHY THIS MATTERS:
    #   Without exclude_unset=True, model_dump() would include ALL fields
    #   with their defaults (None). A PATCH request with just {"name": "X"}
    #   would accidentally set age=None, team_id=None, etc. — destroying
    #   existing data. This is the #1 mistake in REST API update endpoints.
    #
    # EXPECTED RESULT:
    #   Only the specified fields are updated. Unmentioned fields retain
    #   their previous values.
    #
    # HINT:
    #   hero = session.get(Hero, hero_id)
    #   if not hero:
    #       return None
    #   update_dict = hero_data.model_dump(exclude_unset=True)
    #   for key, value in update_dict.items():
    #       setattr(hero, key, value)
    #   session.add(hero)
    #   session.commit()
    #   session.refresh(hero)
    #   return hero

    raise NotImplementedError("TODO(human)")


# ---------------------------------------------------------------------------
# Delete
# ---------------------------------------------------------------------------


def delete_hero(session: Session, hero_id: int) -> bool:
    # TODO(human): Delete a hero by ID. Return True if deleted, False if not found.
    #
    # WHAT TO DO:
    #   1. Fetch the hero: session.get(Hero, hero_id).
    #      Return False if not found.
    #
    #   2. Delete it: session.delete(hero)
    #      This stages a DELETE FROM hero WHERE id = {hero_id}.
    #
    #   3. Commit the transaction: session.commit()
    #
    #   4. Return True.
    #
    # WHY THIS MATTERS:
    #   Deleting by fetching first (vs. DELETE WHERE id = X directly) ensures
    #   SQLAlchemy's cascade rules and relationship cleanup are honored. If
    #   Hero had dependent objects (e.g., hero_powers), session.delete()
    #   would handle cascading deletes according to the relationship config.
    #   A raw DELETE bypasses this.
    #
    # EXPECTED RESULT:
    #   Returns True if the hero existed and was deleted, False otherwise.
    #
    # HINT:
    #   hero = session.get(Hero, hero_id)
    #   if not hero:
    #       return False
    #   session.delete(hero)
    #   session.commit()
    #   return True

    raise NotImplementedError("TODO(human)")
