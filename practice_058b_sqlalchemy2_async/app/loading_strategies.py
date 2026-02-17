"""Eager Loading Strategies — Solving the N+1 Query Problem.

The N+1 problem is the most common ORM performance pitfall:
  - Query 1: SELECT * FROM departments (returns N departments)
  - Queries 2..N+1: SELECT * FROM employees WHERE department_id = ? (one per dept)

This module demonstrates the problem and three eager loading solutions:
  - selectinload: SELECT ... WHERE id IN (...) -- batch loads in 1 extra query
  - joinedload:   LEFT JOIN -- loads everything in 1 query (can duplicate parent rows)
  - subqueryload: SELECT ... WHERE id IN (SELECT ...) -- subquery-based batch load

Run after starting PostgreSQL:
    uv run python -m app.loading_strategies
"""

from sqlalchemy import select, event, text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload, selectinload, subqueryload

from app.models import Department, Employee


# ── Query Counter (boilerplate) ───────────────────────────────────────
#
# This helper counts SQL statements emitted during a block of code.
# It hooks into SQLAlchemy's event system to intercept every query.


class QueryCounter:
    """Counts SQL queries emitted within a context manager.

    Usage:
        counter = QueryCounter(session)
        with counter:
            await session.execute(select(Employee))
        print(f"Queries: {counter.count}")
    """

    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self.count = 0
        self._sync_engine = session.bind.sync_engine  # type: ignore[union-attr]

    def _on_execute(self, conn, clauseelement, multiparams, params, execution_options):  # noqa: ANN001
        self.count += 1

    def __enter__(self) -> "QueryCounter":
        event.listen(self._sync_engine, "before_execute", self._on_execute)
        return self

    def __exit__(self, *args: object) -> None:
        event.remove(self._sync_engine, "before_execute", self._on_execute)


# ── N+1 Problem Demonstration ────────────────────────────────────────


async def demonstrate_n_plus_one(session: AsyncSession) -> None:
    """Show the N+1 problem: loading departments then accessing employees.

    # TODO(human): Query departments and trigger lazy loading of employees
    #
    # This exercise makes the N+1 problem visible. You'll see exactly how
    # many queries are emitted when you naively access a relationship.
    #
    # In SYNC SQLAlchemy, lazy loading "just works" silently -- it fires a
    # SELECT behind the scenes. In ASYNC SQLAlchemy, lazy loading raises
    # MissingGreenlet because it can't do sync I/O in an async context.
    # To demonstrate, we use run_sync() to simulate what sync code does.
    #
    # Steps:
    #   1. Query all departments (no eager loading):
    #      result = await session.execute(select(Department))
    #      departments = result.scalars().all()
    #
    #   2. Use run_sync to simulate lazy loading (async-safe workaround):
    #      counter = QueryCounter(session)
    #      with counter:
    #          def access_employees(sync_session):
    #              for dept in departments:
    #                  # This triggers a lazy load SELECT for each department
    #                  emp_count = len(dept.employees)
    #                  print(f"    {dept.name}: {emp_count} employees")
    #          await session.run_sync(access_employees)
    #
    #   3. Print the total query count:
    #      print(f"  Total queries: {counter.count}")
    #      -- You should see N+1 queries: 1 for departments + N for employees
    #      -- With 5 departments, that's 6 queries total.
    #
    # Why this matters:
    #   With 5 departments, it's 6 queries (barely noticeable).
    #   With 1000 departments, it's 1001 queries (several seconds).
    #   With 10,000 departments, it's 10,001 queries (minutes).
    #   The problem scales linearly with data volume.
    """
    raise NotImplementedError("TODO(human)")


# ── Fix: selectinload ────────────────────────────────────────────────


async def fix_with_selectinload(session: AsyncSession) -> None:
    """Solve N+1 with selectinload — batches related objects in one extra query.

    # TODO(human): Rewrite the department query with selectinload
    #
    # selectinload emits a second query:
    #   SELECT * FROM employees WHERE department_id IN (1, 2, 3, 4, 5)
    # This loads ALL employees for ALL departments in a single round trip.
    #
    # Steps:
    #   1. Query departments with eager loading:
    #      stmt = select(Department).options(selectinload(Department.employees))
    #      result = await session.execute(stmt)
    #      departments = result.scalars().all()
    #
    #   2. Count queries:
    #      counter = QueryCounter(session)
    #      -- Note: the queries were already emitted during execute().
    #      -- Access dept.employees without triggering any additional queries:
    #      for dept in departments:
    #          print(f"    {dept.name}: {len(dept.employees)} employees")
    #
    #   3. Print: "Total additional queries: 0"
    #      -- All data was loaded by the initial execute() call.
    #      -- The execute() itself emitted exactly 2 queries:
    #         1. SELECT * FROM departments
    #         2. SELECT * FROM employees WHERE department_id IN (...)
    #
    # selectinload is the RECOMMENDED default for one-to-many relationships
    # in async code because:
    #   - Always exactly 2 queries (predictable performance)
    #   - No row duplication (unlike joinedload)
    #   - Works with LIMIT/OFFSET on the parent query
    """
    raise NotImplementedError("TODO(human)")


# ── Comparison: All Loading Strategies ────────────────────────────────


async def compare_loading_strategies(session: AsyncSession) -> None:
    """Compare selectinload vs joinedload vs subqueryload side by side.

    # TODO(human): Run the same query with each loading strategy and observe
    #
    # This exercise builds intuition for when to use each strategy.
    #
    # For each strategy, load departments with employees and print the
    # query count. The key difference is HOW they load related objects:
    #
    # 1. selectinload:
    #    stmt = select(Department).options(selectinload(Department.employees))
    #    -- Emits: SELECT depts... THEN SELECT emps WHERE dept_id IN (...)
    #    -- 2 queries total. The IN list contains all parent IDs.
    #    -- Best for: most cases, especially one-to-many. Predictable.
    #
    # 2. joinedload:
    #    stmt = select(Department).options(joinedload(Department.employees))
    #    -- Emits: SELECT depts LEFT JOIN employees ON ...
    #    -- 1 query total. BUT: parent rows are duplicated per child.
    #       A dept with 3 employees appears 3 times in the result set.
    #       SQLAlchemy de-duplicates in Python (via identity map).
    #    -- Best for: many-to-one / one-to-one (no duplication).
    #       AVOID for one-to-many with LIMIT (the LIMIT applies to the
    #       joined result, not the parent count — you get fewer parents).
    #    -- IMPORTANT: When using joinedload with select() (2.0 style),
    #       you must add .unique() before .scalars():
    #       result.unique().scalars().all()
    #       This de-duplicates the joined rows. Without .unique(),
    #       SQLAlchemy raises an error.
    #
    # 3. subqueryload:
    #    stmt = select(Department).options(subqueryload(Department.employees))
    #    -- Emits: SELECT emps WHERE dept_id IN (SELECT dept.id FROM depts)
    #    -- 2 queries, but the second uses a subquery instead of an IN list.
    #    -- Best for: when the parent query is complex (the subquery
    #       re-executes it to get IDs, vs selectin which passes them).
    #    -- Rare in practice. selectinload is usually sufficient.
    #
    # Steps for each strategy:
    #   a) Execute the query with .options(strategy(Department.employees))
    #   b) For joinedload, use result.unique().scalars().all()
    #      For selectinload/subqueryload, use result.scalars().all()
    #   c) Print each dept's employee count
    #   d) Note which strategy suits which scenario
    #
    # Print a summary table at the end:
    #   Strategy      | Queries | Duplication | LIMIT-safe | Best for
    #   selectinload  | 2       | No          | Yes        | one-to-many (default)
    #   joinedload    | 1       | Yes         | No         | many-to-one, one-to-one
    #   subqueryload  | 2       | No          | Yes        | complex parent queries
    """
    raise NotImplementedError("TODO(human)")


# ── Demo Runner ───────────────────────────────────────────────────────


async def run_loading_demo(session: AsyncSession) -> None:
    """Run all loading strategy demonstrations."""
    print("\n" + "=" * 70)
    print("PHASE 4: Loading Strategies & N+1 Problem")
    print("=" * 70)

    print("\n--- 1. The N+1 Problem (lazy loading) ---")
    await demonstrate_n_plus_one(session)

    # Expire all to clear cached state between demos
    session.expire_all()

    print("\n--- 2. Fix with selectinload ---")
    await fix_with_selectinload(session)

    session.expire_all()

    print("\n--- 3. Comparing All Strategies ---")
    await compare_loading_strategies(session)


if __name__ == "__main__":
    import asyncio

    from app.engine import create_engine_factory, create_session_factory

    async def main() -> None:
        engine = create_engine_factory()
        session_factory = create_session_factory(engine)
        async with session_factory() as session:
            await run_loading_demo(session)
        await engine.dispose()

    asyncio.run(main())
