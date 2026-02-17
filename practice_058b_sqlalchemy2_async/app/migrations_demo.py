"""Alembic Migration Workflow Demonstration.

Alembic is SQLAlchemy's dedicated migration tool. It tracks schema changes
as versioned Python scripts, allowing you to:
  - Autogenerate migrations by diffing models against the DB
  - Upgrade (apply) and downgrade (revert) schema changes
  - Maintain a linear or branching migration history

This module guides you through the migration workflow for adding a new
column to the employees table.

Run after completing the Alembic setup:
    uv run python -m app.migrations_demo
"""

from sqlalchemy import inspect, text
from sqlalchemy.ext.asyncio import AsyncSession


# ── Migration Exercise ────────────────────────────────────────────────


async def verify_phone_column(session: AsyncSession) -> bool:
    """Verify that the 'phone' column was successfully added to the employees table.

    # TODO(human): Add a phone column to Employee via Alembic, then verify here
    #
    # This exercise teaches the complete Alembic migration workflow.
    # It is different from the other exercises -- you don't write Python
    # function logic here. Instead, you:
    #
    # STEP-BY-STEP WORKFLOW (run these commands in the terminal):
    #
    #   1. First, add the column to the Employee model in models.py:
    #      phone: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    #      -- Add this line in your Employee class, after the existing columns.
    #      -- Optional[str] + nullable=True means the column allows NULL.
    #      -- This represents a schema change: the Python model now has a
    #         column that the database does NOT have yet.
    #
    #   2. Generate the migration script:
    #      uv run alembic revision --autogenerate -m "add phone column"
    #      -- Alembic connects to the DB, reads the current schema, compares
    #         it to Base.metadata (your Python models), and generates a
    #         migration script with the diff.
    #      -- The generated file appears in alembic/versions/ as:
    #         <revision_id>_add_phone_column.py
    #      -- Open it and inspect the upgrade() and downgrade() functions.
    #         upgrade() should contain: op.add_column('employees', sa.Column('phone', ...))
    #         downgrade() should contain: op.drop_column('employees', 'phone')
    #
    #   3. Apply the migration:
    #      uv run alembic upgrade head
    #      -- "head" means "apply all unapplied migrations up to the latest."
    #      -- Alembic tracks which migrations have been applied in a table
    #         called alembic_version in the database.
    #      -- After this, the DB has the phone column.
    #
    #   4. Verify by running this module:
    #      uv run python -m app.migrations_demo
    #      -- This function checks if the column exists.
    #
    # WHAT TO IMPLEMENT IN THIS FUNCTION:
    #   Use raw SQL to check if the phone column exists:
    #
    #   result = await session.execute(
    #       text(
    #           "SELECT column_name FROM information_schema.columns "
    #           "WHERE table_name = 'employees' AND column_name = 'phone'"
    #       )
    #   )
    #   row = result.first()
    #
    #   if row:
    #       print("  [OK] 'phone' column exists in employees table")
    #       return True
    #   else:
    #       print("  [MISSING] 'phone' column not found -- run the Alembic workflow above")
    #       return False
    #
    # BONUS -- Other useful Alembic commands:
    #   uv run alembic downgrade -1     # Revert the last migration
    #   uv run alembic history          # Show migration history
    #   uv run alembic current          # Show current DB revision
    #   uv run alembic heads            # Show latest available revision
    """
    raise NotImplementedError("TODO(human)")


# ── Demo Runner ───────────────────────────────────────────────────────


async def run_migrations_demo(session: AsyncSession) -> None:
    """Run the migration verification."""
    print("\n" + "=" * 70)
    print("PHASE 5: Alembic Migrations")
    print("=" * 70)

    print("\n--- Verifying Phone Column Migration ---")
    exists = await verify_phone_column(session)
    if exists:
        print("\n  Migration workflow completed successfully!")
        print("  The phone column is now part of the employees table.")
    else:
        print("\n  Follow the steps in the TODO(human) comment to complete this exercise.")
        print("  You need to:")
        print("    1. Add 'phone' column to Employee model in models.py")
        print("    2. Run: uv run alembic revision --autogenerate -m \"add phone column\"")
        print("    3. Run: uv run alembic upgrade head")
        print("    4. Re-run this module to verify")


if __name__ == "__main__":
    import asyncio

    from app.engine import create_engine_factory, create_session_factory

    async def main() -> None:
        engine = create_engine_factory()
        session_factory = create_session_factory(engine)
        async with session_factory() as session:
            await run_migrations_demo(session)
        await engine.dispose()

    asyncio.run(main())
