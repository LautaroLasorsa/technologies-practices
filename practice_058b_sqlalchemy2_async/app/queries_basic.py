"""Basic CRUD Operations with SQLAlchemy 2.0 select() Style.

Demonstrates the modern query API that replaced the legacy Query interface:
  - Legacy (1.x):  session.query(Employee).filter(Employee.name == "Alice").all()
  - Modern (2.0):  session.execute(select(Employee).where(Employee.name == "Alice"))

The 2.0 style uses standalone select() statements that are composed functionally.
This is more explicit, more composable, and works identically in sync and async.

Run after starting PostgreSQL:
    uv run python -m app.queries_basic
"""

from decimal import Decimal

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models import Department, Employee


# ── Query: Employees by Department ────────────────────────────────────


async def get_employees_by_department(
    session: AsyncSession,
    dept_name: str,
) -> list[Employee]:
    """Fetch all employees in a department, eagerly loading their relationships.

    # TODO(human): Write a select() query joining Employee to Department
    #
    # This exercise teaches you the 2.0 select() + join + eager loading pattern
    # that replaces session.query() from 1.x.
    #
    # Steps:
    #   1. Build a select(Employee) statement
    #   2. Use .join(Employee.department) to JOIN with the departments table
    #       -- SQLAlchemy infers the JOIN condition from the relationship
    #          definition and the ForeignKey. You don't need to write
    #          ON employees.department_id = departments.id manually.
    #   3. Use .where(Department.name == dept_name) to filter
    #   4. Use .options(selectinload(Employee.department)) to eager-load
    #       -- selectinload emits a second SELECT ... WHERE id IN (...)
    #          to batch-load all related departments in one query.
    #       -- Without this, accessing emp.department in async raises
    #          MissingGreenlet because lazy loading requires sync I/O.
    #   5. Execute: result = await session.execute(stmt)
    #   6. Extract: employees = result.scalars().all()
    #       -- .scalars() unwraps Row objects into the ORM entity directly
    #       -- .all() materializes the result into a Python list
    #
    # Return the list of Employee objects.
    #
    # Why scalars()?
    #   session.execute() returns Row objects (tuples). For a simple
    #   select(Employee), each Row contains one element: the Employee.
    #   .scalars() extracts that single element, so you get Employee
    #   objects directly instead of Row(Employee,) wrappers.
    """
    raise NotImplementedError("TODO(human)")


# ── Create: Insert New Employee ───────────────────────────────────────


async def create_employee(
    session: AsyncSession,
    name: str,
    email: str,
    salary: float,
    dept_id: int,
    manager_id: int | None = None,
) -> Employee:
    """Insert a new employee using the 2.0 Unit of Work pattern.

    # TODO(human): Create and persist a new Employee instance
    #
    # This exercise teaches the SQLAlchemy 2.0 insert pattern:
    #
    # Steps:
    #   1. Instantiate: employee = Employee(name=..., email=..., salary=...,
    #                                       department_id=dept_id,
    #                                       manager_id=manager_id)
    #       -- Convert salary to Decimal for the Numeric column:
    #          Decimal(str(salary)) avoids float precision issues.
    #          e.g., Decimal(str(85000.50)) -> Decimal('85000.50')
    #          vs    Decimal(85000.50) -> Decimal('85000.5000000000...072')
    #
    #   2. Add to session: session.add(employee)
    #       -- This puts the object in the session's "new" set (pending INSERT).
    #       -- No SQL is emitted yet. The session tracks it in memory.
    #
    #   3. Flush: await session.flush()
    #       -- Emits the INSERT SQL and populates the auto-generated id.
    #       -- flush() writes to DB but does NOT commit the transaction.
    #       -- After flush, employee.id is set (the DB returned it).
    #       -- flush vs commit: flush writes SQL to the DB within the
    #          current transaction; commit makes it permanent.
    #          You flush when you need the id before committing.
    #
    #   4. Refresh: await session.refresh(employee)
    #       -- Reloads ALL columns from the DB (including server_default
    #          values like created_at that were set by NOW()).
    #       -- Without refresh, created_at would be None in Python
    #          because the Python object doesn't know what the DB set.
    #
    # Return the employee with all fields populated.
    #
    # Important: This function does NOT commit. The caller decides when
    # to commit (or rollback). This follows the Repository pattern --
    # operations are composed, and transaction boundaries are controlled
    # at a higher level.
    """
    raise NotImplementedError("TODO(human)")


# ── Update: Partial Update ───────────────────────────────────────────


async def update_salary(
    session: AsyncSession,
    employee_id: int,
    new_salary: float,
) -> Employee:
    """Update an employee's salary using the 2.0 pattern.

    # TODO(human): Fetch an employee by id and update their salary
    #
    # This exercise teaches the load-modify-flush update pattern:
    #
    # Steps:
    #   1. Fetch: result = await session.execute(
    #                select(Employee).where(Employee.id == employee_id)
    #             )
    #             employee = result.scalar_one_or_none()
    #       -- scalar_one_or_none() returns exactly one result or None.
    #       -- If no row matches, it returns None (not an exception).
    #       -- scalar_one() would raise NoResultFound if no match.
    #
    #   2. Guard: if employee is None, raise ValueError(f"Employee {employee_id} not found")
    #
    #   3. Mutate: employee.salary = Decimal(str(new_salary))
    #       -- SQLAlchemy's Unit of Work detects this change automatically.
    #       -- The session tracks dirty attributes via Python descriptors.
    #       -- On flush/commit, it emits UPDATE employees SET salary = ...
    #          only for the changed columns (partial UPDATE).
    #
    #   4. Flush: await session.flush()
    #       -- Writes the UPDATE to the DB (still within the transaction).
    #
    #   5. Refresh: await session.refresh(employee)
    #       -- Ensures the Python object matches the DB state.
    #
    # Return the updated employee.
    #
    # Why load-then-modify instead of a raw UPDATE statement?
    #   The ORM pattern gives you: validation, hooks (events), relationship
    #   updates, and audit trails. For bulk updates (thousands of rows),
    #   use session.execute(update(Employee).where(...).values(...)) instead
    #   -- it bypasses the ORM identity map for performance.
    """
    raise NotImplementedError("TODO(human)")


# ── Demo Runner ───────────────────────────────────────────────────────


async def run_basic_queries_demo(session: AsyncSession) -> None:
    """Run all basic CRUD demonstrations."""
    print("\n" + "=" * 70)
    print("PHASE 2: Basic CRUD with select() Style")
    print("=" * 70)

    # Query employees in Engineering
    print("\n--- Employees in Engineering ---")
    engineers = await get_employees_by_department(session, "Engineering")
    for emp in engineers:
        print(f"  {emp.name} (${emp.salary:,.2f})")

    # Create a new employee
    print("\n--- Creating New Employee ---")
    new_emp = await create_employee(
        session,
        name="Karen White",
        email="karen@co.com",
        salary=96000.00,
        dept_id=1,
        manager_id=1,
    )
    print(f"  Created: {new_emp} (id={new_emp.id})")
    await session.commit()

    # Update salary
    print("\n--- Updating Salary ---")
    updated = await update_salary(session, new_emp.id, 101000.00)
    print(f"  Updated: {updated.name} -> ${updated.salary:,.2f}")
    await session.commit()

    # Verify the update
    print("\n--- Engineering After Changes ---")
    engineers = await get_employees_by_department(session, "Engineering")
    for emp in engineers:
        print(f"  {emp.name} (${emp.salary:,.2f})")


if __name__ == "__main__":
    import asyncio

    from app.engine import create_engine_factory, create_session_factory

    async def main() -> None:
        engine = create_engine_factory()
        session_factory = create_session_factory(engine)
        async with session_factory() as session:
            await run_basic_queries_demo(session)
        await engine.dispose()

    asyncio.run(main())
