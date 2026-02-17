"""
Practice 058a — Relationship Demonstrations

This module demonstrates SQLModel relationship patterns: creating related
objects in a single transaction, and eagerly loading related objects to
avoid the N+1 query problem.

Key concepts:
  - Transaction: a group of operations that either ALL succeed or ALL fail.
    If any step raises an exception before commit(), all changes are rolled back.
  - Lazy loading: relationships are not loaded until you access the attribute.
    team.heroes triggers a SELECT behind the scenes. Fine for single objects,
    but causes N+1 queries when iterating over a list of teams.
  - Eager loading (selectinload): loads related objects in a SINGLE additional
    query using SELECT ... WHERE team_id IN (1, 2, 3, ...). Eliminates N+1.
"""

from sqlmodel import Session, select
from sqlalchemy.orm import selectinload

from app.models import Hero, Team, TeamReadWithHeroes


# ---------------------------------------------------------------------------
# Create team with heroes in one transaction
# ---------------------------------------------------------------------------


def create_team_with_heroes(
    session: Session, team_name: str, headquarters: str, hero_names: list[str]
) -> Team:
    # TODO(human): Create a Team and multiple Heroes in a single transaction.
    #
    # WHAT TO DO:
    #   1. Create a Team instance: Team(name=team_name, headquarters=headquarters)
    #
    #   2. Create Hero instances for each name in hero_names. For each hero,
    #      set `team=team` (the relationship attribute, NOT team_id). This is
    #      the SQLModel/SQLAlchemy way — when you set the relationship attribute,
    #      the foreign key is automatically populated on commit.
    #      Example: Hero(name=name, secret_name=f"{name} Secret", team=team)
    #
    #   3. Add the team to the session: session.add(team)
    #      Because the heroes reference the team via Relationship(), SQLAlchemy's
    #      cascade behavior automatically adds the heroes too. You don't need
    #      to add each hero individually.
    #
    #   4. Commit and refresh the team.
    #
    #   5. Return the team.
    #
    # WHY THIS MATTERS:
    #   This demonstrates transactional integrity. If creating the 3rd hero
    #   fails (e.g., a constraint violation), the entire transaction rolls back —
    #   no team and no heroes are created. Without transactions, you could end
    #   up with a team but only 2 of 3 heroes, leaving the DB in an
    #   inconsistent state.
    #
    #   Setting `team=team` (relationship) instead of `team_id=team.id` is
    #   important: at this point team.id is still None (not committed yet).
    #   SQLAlchemy resolves the foreign key DURING commit, once the team's id
    #   is assigned. This is called "relationship-based assignment."
    #
    # EXPECTED RESULT:
    #   A Team row and N Hero rows are created in a single commit. All heroes
    #   have team_id pointing to the new team.
    #
    # HINT:
    #   team = Team(name=team_name, headquarters=headquarters)
    #   for name in hero_names:
    #       Hero(name=name, secret_name=f"{name} Secret", team=team)
    #   session.add(team)
    #   session.commit()
    #   session.refresh(team)
    #   return team

    raise NotImplementedError("TODO(human)")


# ---------------------------------------------------------------------------
# Read team with eagerly loaded heroes
# ---------------------------------------------------------------------------


def get_team_with_heroes(session: Session, team_id: int) -> Team | None:
    # TODO(human): Fetch a team with its heroes in a single efficient query.
    #
    # WHAT TO DO:
    #   1. Build a select statement with eager loading:
    #        statement = select(Team).where(Team.id == team_id).options(
    #            selectinload(Team.heroes)
    #        )
    #      selectinload(Team.heroes) tells SQLAlchemy: "after fetching the
    #      team, immediately fetch all its heroes in a second query using
    #      SELECT * FROM hero WHERE team_id IN (...)"
    #
    #   2. Execute and get the first result:
    #        team = session.exec(statement).first()
    #
    #   3. Return the team (or None if not found).
    #
    # WHY THIS MATTERS:
    #   Without selectinload, accessing team.heroes would trigger a LAZY load —
    #   a separate SQL query executed behind the scenes. This is the N+1 problem:
    #   if you load 100 teams and access .heroes on each, you get 1 query for
    #   teams + 100 individual queries for heroes = 101 queries total.
    #
    #   With selectinload, it's always 2 queries total regardless of how many
    #   teams you load: one for teams, one for all their heroes.
    #
    #   Other loading strategies:
    #   - joinedload: uses a LEFT JOIN (1 query, but duplicates team data)
    #   - subqueryload: uses a subquery (2 queries, like selectinload)
    #   - lazyload (default): queries on attribute access (N+1 risk)
    #
    # EXPECTED RESULT:
    #   Returns a Team object with team.heroes already populated as a list
    #   of Hero objects, loaded via 2 SQL queries (not N+1).
    #
    # HINT:
    #   statement = select(Team).where(Team.id == team_id).options(
    #       selectinload(Team.heroes)
    #   )
    #   return session.exec(statement).first()

    raise NotImplementedError("TODO(human)")
