"""SQLAlchemy 2.0 Declarative ORM Models.

Defines the database schema using the modern 2.0 Mapped[] + mapped_column() style.
This replaces the legacy Column() + relationship() pattern from SQLAlchemy 1.x.

Key concepts demonstrated:
  - DeclarativeBase: The new base class (replaces declarative_base() factory)
  - Mapped[T]: Type annotation that tells SQLAlchemy + mypy about column types
  - mapped_column(): Column definition with full type safety
  - relationship(): Defines Python-level ORM relationships (not DB foreign keys)
  - Association table: Many-to-many via a plain Table (not a mapped class)
  - Self-referential FK: employee.manager_id -> employees.id

Run: This module is imported by other modules; it has no standalone entry point.
"""

from __future__ import annotations

import datetime
from decimal import Decimal
from typing import Optional

from sqlalchemy import (
    Column,
    Date,
    ForeignKey,
    Integer,
    Numeric,
    String,
    Table,
    func,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
)


# ── DeclarativeBase ───────────────────────────────────────────────────
#
# In SQLAlchemy 2.0, you subclass DeclarativeBase instead of calling
# declarative_base(). This gives you a Base class that:
#   1. Has a MetaData registry (Base.metadata) for all tables
#   2. Has a registry for all mapped classes
#   3. Integrates with type checkers (mypy, pyright)
#
# Every model class inherits from Base, and SQLAlchemy automatically
# registers it in the metadata. This metadata is what Alembic uses
# to autogenerate migrations.


class Base(DeclarativeBase):
    pass


# ── Association Table (Many-to-Many) ─────────────────────────────────
#
# For a pure many-to-many relationship (no extra columns beyond the FKs),
# SQLAlchemy recommends a plain Table object instead of a mapped class.
# This is lighter weight -- no ORM identity, no session tracking.
#
# If you needed extra columns (e.g., assigned_date, hours_worked), you
# would use an Association Object pattern (a full mapped class) instead.
#
# Note: The "role" column here makes this a borderline case -- we include
# it in the Table for simplicity, but in production you might promote it
# to a full mapped class to query roles via ORM.

employee_projects = Table(
    "employee_projects",
    Base.metadata,
    Column("employee_id", Integer, ForeignKey("employees.id"), primary_key=True),
    Column("project_id", Integer, ForeignKey("projects.id"), primary_key=True),
    Column("role", String(100), default="member"),
)


# ── Project Model (boilerplate -- fully implemented) ──────────────────
#
# Demonstrates mapped_column() for all common SQL types:
#   - Integer primary key with autoincrement
#   - String with max length
#   - Nullable columns (Optional[T] + nullable=True)
#   - Decimal/Numeric for money (never use float for money!)
#   - Date columns


class Project(Base):
    __tablename__ = "projects"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    status: Mapped[str] = mapped_column(String(50), default="active")
    budget: Mapped[Optional[Decimal]] = mapped_column(Numeric(12, 2), nullable=True)
    start_date: Mapped[Optional[datetime.date]] = mapped_column(Date, nullable=True)
    end_date: Mapped[Optional[datetime.date]] = mapped_column(Date, nullable=True)

    # Many-to-many relationship back to Employee via association table.
    # back_populates="projects" means Employee.projects mirrors this list.
    employees: Mapped[list[Employee]] = relationship(
        secondary=employee_projects,
        back_populates="projects",
    )

    def __repr__(self) -> str:
        return f"<Project(id={self.id}, name={self.name!r}, status={self.status!r})>"


# ── Department Model ──────────────────────────────────────────────────


class Department(Base):
    """Department entity with a one-to-many relationship to Employee.

    # TODO(human): Define the Department model using SQLAlchemy 2.0 mapped_column()
    #
    # This exercise teaches you the modern SQLAlchemy 2.0 declarative style:
    #
    # TABLE NAME:
    #   Set __tablename__ = "departments" to map this class to the existing
    #   departments table created by init.sql. SQLAlchemy uses __tablename__
    #   to know which DB table this Python class corresponds to.
    #
    # COLUMNS (use mapped_column() for each):
    #   - id: Mapped[int]        -- Integer, primary_key=True, autoincrement=True
    #   - name: Mapped[str]      -- String(100), nullable=False
    #   - budget: Mapped[Decimal] -- Numeric(12, 2), nullable=False
    #   - created_at: Mapped[Optional[datetime.datetime]]
    #       -- Use server_default=func.now() so the DB sets the timestamp.
    #       -- func.now() generates SQL NOW() -- it's a server-side default,
    #          meaning the DB clock sets the value, not the Python process.
    #          This ensures consistency across distributed app servers.
    #
    # RELATIONSHIP:
    #   - employees: Mapped[list["Employee"]]
    #       -- relationship(back_populates="department")
    #       -- This creates a Python-level collection. When you access
    #          dept.employees, SQLAlchemy emits a SELECT to load them.
    #       -- back_populates="department" means Employee.department is
    #          the other side of this bidirectional relationship.
    #       -- By default, relationships use "lazy loading" -- the SELECT
    #          is emitted when you first access the attribute. In async
    #          mode, lazy loading raises an error because it would need
    #          to do I/O synchronously inside an async context. You MUST
    #          use eager loading (selectinload, joinedload) or explicit
    #          awaitable loading to avoid this.
    #
    # REPR:
    #   - __repr__ returning f"<Department(id={self.id}, name={self.name!r})>"
    #
    # WHY mapped_column() OVER Column():
    #   In 1.x, you wrote:  id = Column(Integer, primary_key=True)
    #   In 2.0, you write:  id: Mapped[int] = mapped_column(Integer, primary_key=True)
    #
    #   The Mapped[int] annotation tells both SQLAlchemy and type checkers
    #   that self.id is an int. This eliminates the need for SQLAlchemy stubs
    #   and gives you full IDE autocompletion + type safety. The runtime
    #   behavior is identical -- mapped_column() is just Column() with type
    #   integration.
    """

    __tablename__ = "departments"

    raise NotImplementedError("TODO(human)")


# ── Employee Model ────────────────────────────────────────────────────


class Employee(Base):
    """Employee entity with self-referential FK (manager) and many-to-many (projects).

    # TODO(human): Define the Employee model with all relationships
    #
    # This is the most complex model -- it demonstrates three relationship patterns
    # that appear constantly in real-world ORMs:
    #
    # TABLE NAME:
    #   __tablename__ = "employees"
    #
    # COLUMNS (use mapped_column() for each):
    #   - id: Mapped[int]                    -- Integer, primary_key=True
    #   - name: Mapped[str]                  -- String(100), nullable=False
    #   - email: Mapped[str]                 -- String(255), unique=True, nullable=False
    #   - salary: Mapped[Decimal]            -- Numeric(10, 2), nullable=False
    #   - hire_date: Mapped[datetime.date]   -- Date, nullable=False
    #   - department_id: Mapped[Optional[int]]  -- ForeignKey("departments.id")
    #   - manager_id: Mapped[Optional[int]]     -- ForeignKey("employees.id")
    #       ^^ This is the self-referential FK. It points to the SAME table
    #          (employees.id). This creates a tree structure: Alice manages
    #          Bob, Bob manages Carol, etc.
    #   - created_at: Mapped[Optional[datetime.datetime]]
    #       -- server_default=func.now()
    #
    # RELATIONSHIPS:
    #
    #   1. department: Mapped[Optional["Department"]]
    #       -- relationship(back_populates="employees")
    #       -- The "many" side of dept <-> employees. Each employee belongs
    #          to at most one department.
    #
    #   2. manager: Mapped[Optional["Employee"]]
    #       -- relationship(back_populates="subordinates", remote_side=[id])
    #       -- Self-referential: manager_id FK points to another Employee.
    #       -- remote_side=[id] tells SQLAlchemy which side is the "one"
    #          in this one-to-many: the side with the primary key (id)
    #          is the "remote" (parent) side. Without remote_side,
    #          SQLAlchemy can't determine direction in a self-join.
    #
    #   3. subordinates: Mapped[list["Employee"]]
    #       -- relationship(back_populates="manager")
    #       -- The reverse of the manager relationship. Alice.subordinates
    #          returns [Bob, Carol, Jack].
    #
    #   4. projects: Mapped[list["Project"]]
    #       -- relationship(secondary=employee_projects, back_populates="employees")
    #       -- Many-to-many via the association table. Employee can be on
    #          multiple projects; Project can have multiple employees.
    #       -- "secondary=employee_projects" tells SQLAlchemy to JOIN through
    #          the association table when loading this relationship.
    #
    # REPR:
    #   __repr__ returning f"<Employee(id={self.id}, name={self.name!r}, dept_id={self.department_id})>"
    #
    # IMPORTANT -- ASYNC AND LAZY LOADING:
    #   All four relationships above default to lazy="select" (lazy loading).
    #   In async code, accessing emp.department without prior eager loading
    #   raises sqlalchemy.exc.MissingGreenlet because the ORM would need
    #   to emit a synchronous SQL query inside an async context. Solutions:
    #     a) Use selectinload()/joinedload() in the original query
    #     b) Use await session.run_sync(lambda s: emp.department)
    #     c) Set lazy="selectin" on the relationship (always eager)
    #   We'll explore these strategies in loading_strategies.py.
    """

    __tablename__ = "employees"

    raise NotImplementedError("TODO(human)")
