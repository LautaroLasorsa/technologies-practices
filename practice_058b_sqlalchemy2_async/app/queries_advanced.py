"""Advanced Query Patterns with SQLAlchemy 2.0.

Demonstrates SQL patterns that go beyond basic CRUD:
  - Aggregations with GROUP BY (func.avg, func.min, func.max)
  - Correlated subqueries (employees earning above dept average)
  - Recursive CTEs (management hierarchy traversal)
  - Window functions (RANK() OVER PARTITION BY)

Each pattern maps 1:1 to a SQL concept. SQLAlchemy generates the SQL --
you express intent in Python. Understanding the generated SQL is key to
using the ORM effectively.

Run after starting PostgreSQL:
    uv run python -m app.queries_advanced
"""

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Department, Employee


# ── Aggregation: Department Salary Stats ──────────────────────────────


async def department_salary_stats(session: AsyncSession) -> list[dict]:
    """Compute salary statistics (avg, min, max, count) grouped by department.

    # TODO(human): Write a GROUP BY query with aggregate functions
    #
    # This exercise teaches you how SQLAlchemy maps to SQL aggregations.
    # The generated SQL should look like:
    #
    #   SELECT departments.name,
    #          AVG(employees.salary)   AS avg_salary,
    #          MIN(employees.salary)   AS min_salary,
    #          MAX(employees.salary)   AS max_salary,
    #          COUNT(employees.id)     AS emp_count
    #   FROM employees
    #   JOIN departments ON employees.department_id = departments.id
    #   GROUP BY departments.name
    #   ORDER BY avg_salary DESC
    #
    # Steps:
    #   1. Build the select with explicit columns:
    #      stmt = select(
    #          Department.name,
    #          func.avg(Employee.salary).label("avg_salary"),
    #          func.min(Employee.salary).label("min_salary"),
    #          func.max(Employee.salary).label("max_salary"),
    #          func.count(Employee.id).label("emp_count"),
    #      )
    #       -- func.avg/min/max/count generate SQL aggregate functions
    #       -- .label("name") assigns a column alias (AS name in SQL)
    #       -- When selecting specific columns (not entities), the result
    #          rows are named tuples, not ORM objects.
    #
    #   2. Add join: .join(Department, Employee.department_id == Department.id)
    #       -- Explicit join condition. You could also use
    #          .join(Employee.department) if the relationship exists.
    #
    #   3. Group: .group_by(Department.name)
    #
    #   4. Order: .order_by(func.avg(Employee.salary).desc())
    #       -- .desc() generates DESC in SQL
    #
    #   5. Execute and collect results into a list of dicts:
    #      result = await session.execute(stmt)
    #      return [
    #          {
    #              "department": row.name,
    #              "avg_salary": float(row.avg_salary),
    #              "min_salary": float(row.min_salary),
    #              "max_salary": float(row.max_salary),
    #              "emp_count": row.emp_count,
    #          }
    #          for row in result.all()
    #      ]
    #       -- result.all() returns Row objects with named attributes
    #          matching the .label() aliases.
    """
    raise NotImplementedError("TODO(human)")


# ── Subquery: Employees Above Department Average ──────────────────────


async def employees_above_department_avg(session: AsyncSession) -> list[dict]:
    """Find employees earning more than their department's average salary.

    # TODO(human): Write a correlated subquery or CTE-based query
    #
    # This exercise teaches subquery composition -- a fundamental SQL pattern
    # for comparing individual rows against aggregated values.
    #
    # Approach: Use a subquery to compute department averages, then join.
    #
    # The generated SQL should resemble:
    #
    #   SELECT employees.name, employees.salary,
    #          departments.name AS dept_name, dept_avg.avg_sal
    #   FROM employees
    #   JOIN departments ON employees.department_id = departments.id
    #   JOIN (
    #       SELECT department_id, AVG(salary) AS avg_sal
    #       FROM employees
    #       GROUP BY department_id
    #   ) AS dept_avg ON employees.department_id = dept_avg.department_id
    #   WHERE employees.salary > dept_avg.avg_sal
    #   ORDER BY employees.salary DESC
    #
    # Steps:
    #   1. Build the subquery:
    #      dept_avg_subq = (
    #          select(
    #              Employee.department_id,
    #              func.avg(Employee.salary).label("avg_sal"),
    #          )
    #          .group_by(Employee.department_id)
    #          .subquery("dept_avg")
    #      )
    #       -- .subquery("dept_avg") wraps the SELECT as a derived table
    #          with alias "dept_avg". It can now be joined like a real table.
    #       -- Access its columns via dept_avg_subq.c.department_id
    #          and dept_avg_subq.c.avg_sal (.c is the columns collection).
    #
    #   2. Build the outer select:
    #      stmt = (
    #          select(
    #              Employee.name,
    #              Employee.salary,
    #              Department.name.label("dept_name"),
    #              dept_avg_subq.c.avg_sal,
    #          )
    #          .join(Department, Employee.department_id == Department.id)
    #          .join(dept_avg_subq,
    #                Employee.department_id == dept_avg_subq.c.department_id)
    #          .where(Employee.salary > dept_avg_subq.c.avg_sal)
    #          .order_by(Employee.salary.desc())
    #      )
    #
    #   3. Execute and convert to list of dicts with keys:
    #      "name", "salary" (float), "department" (dept_name), "dept_avg" (float avg_sal)
    #
    # Why subquery instead of a window function?
    #   Both work. A subquery is more portable and easier to reason about.
    #   Window functions (covered in salary_rank_by_department) can do
    #   this too, but subqueries are the canonical way to compare
    #   individual rows against group aggregates in standard SQL.
    """
    raise NotImplementedError("TODO(human)")


# ── Recursive CTE: Management Hierarchy ──────────────────────────────


async def management_hierarchy(
    session: AsyncSession,
    employee_id: int,
) -> list[Employee]:
    """Traverse the management chain upward from an employee to the top.

    # TODO(human): Write a recursive CTE to walk the manager chain
    #
    # This exercise teaches recursive Common Table Expressions (CTEs) --
    # the SQL standard way to traverse tree/graph structures in relational data.
    #
    # The self-referential FK (manager_id -> employees.id) forms a tree:
    #   Alice (id=1, manager=NULL)  -- top-level
    #     ├── Bob (id=2, manager=1)
    #     ├── Carol (id=3, manager=1)
    #     └── Jack (id=10, manager=1)
    #   David (id=4, manager=NULL)  -- top-level
    #     └── Eve (id=5, manager=4)
    #
    # For employee_id=3 (Carol), this should return: [Carol, Alice]
    # For employee_id=9 (Ivy), this should return: [Ivy, Henry]
    #
    # The generated SQL should look like:
    #
    #   WITH RECURSIVE mgmt_chain AS (
    #       -- Base case: start with the given employee
    #       SELECT id, name, manager_id
    #       FROM employees
    #       WHERE id = :employee_id
    #
    #       UNION ALL
    #
    #       -- Recursive step: join to get the manager
    #       SELECT e.id, e.name, e.manager_id
    #       FROM employees e
    #       JOIN mgmt_chain mc ON e.id = mc.manager_id
    #   )
    #   SELECT * FROM mgmt_chain
    #
    # Steps:
    #   1. Base case (anchor):
    #      anchor = (
    #          select(Employee.id, Employee.name, Employee.manager_id)
    #          .where(Employee.id == employee_id)
    #          .cte(name="mgmt_chain", recursive=True)
    #      )
    #       -- .cte(recursive=True) produces WITH RECURSIVE
    #       -- The anchor is the starting point of the recursion
    #
    #   2. Create alias for the recursive reference:
    #      emp_alias = Employee.__table__.alias("e")
    #       -- We need a table alias because we're joining employees
    #          to the CTE (which also references employees data).
    #
    #   3. Recursive step:
    #      recursive = (
    #          select(emp_alias.c.id, emp_alias.c.name, emp_alias.c.manager_id)
    #          .join(anchor, emp_alias.c.id == anchor.c.manager_id)
    #      )
    #       -- Joins employees (aliased) where emp.id == cte.manager_id
    #       -- This walks UP the tree: from child to parent
    #
    #   4. Union: cte = anchor.union_all(recursive)
    #       -- UNION ALL combines anchor + recursive results
    #       -- UNION ALL (not UNION) because we want all rows including
    #          potential duplicates (though trees don't have cycles)
    #
    #   5. Final query joining CTE back to Employee for full ORM objects:
    #      stmt = select(Employee).join(cte, Employee.id == cte.c.id)
    #
    #   6. Execute: result = await session.execute(stmt)
    #      Return result.scalars().all()
    """
    raise NotImplementedError("TODO(human)")


# ── Window Function: Salary Rank by Department ────────────────────────


async def salary_rank_by_department(session: AsyncSession) -> list[dict]:
    """Rank employees by salary within each department using window functions.

    # TODO(human): Write a query using RANK() OVER (PARTITION BY ... ORDER BY ...)
    #
    # Window functions perform calculations across a set of rows that are
    # related to the current row -- without collapsing them into groups
    # (unlike GROUP BY). Each row keeps its identity while gaining access
    # to aggregate/ranking information.
    #
    # The generated SQL:
    #
    #   SELECT employees.name, employees.salary,
    #          departments.name AS dept_name,
    #          RANK() OVER (
    #              PARTITION BY employees.department_id
    #              ORDER BY employees.salary DESC
    #          ) AS salary_rank
    #   FROM employees
    #   JOIN departments ON employees.department_id = departments.id
    #   ORDER BY departments.name, salary_rank
    #
    # Steps:
    #   1. Define the window function:
    #      salary_rank = func.rank().over(
    #          partition_by=Employee.department_id,
    #          order_by=Employee.salary.desc(),
    #      ).label("salary_rank")
    #       -- func.rank() generates the RANK() SQL function
    #       -- .over() defines the window: PARTITION BY divides rows into
    #          groups (per department), ORDER BY sorts within each group.
    #       -- RANK() assigns 1 to the highest salary in each department,
    #          2 to the next, etc. Ties get the same rank, and the next
    #          rank is skipped (e.g., 1, 1, 3 if two people tie for first).
    #       -- Alternative: func.dense_rank() doesn't skip (1, 1, 2).
    #       -- Alternative: func.row_number() never ties (1, 2, 3).
    #
    #   2. Build the select:
    #      stmt = (
    #          select(
    #              Employee.name,
    #              Employee.salary,
    #              Department.name.label("dept_name"),
    #              salary_rank,
    #          )
    #          .join(Department, Employee.department_id == Department.id)
    #          .order_by(Department.name, salary_rank)
    #      )
    #
    #   3. Execute and convert to list of dicts with keys:
    #      "name", "salary" (float), "department" (dept_name), "rank" (salary_rank)
    #
    # When to use window functions vs GROUP BY:
    #   GROUP BY: when you want ONE row per group (aggregated)
    #   Window:   when you want ALL rows, each annotated with group-level info
    """
    raise NotImplementedError("TODO(human)")


# ── Demo Runner ───────────────────────────────────────────────────────


async def run_advanced_queries_demo(session: AsyncSession) -> None:
    """Run all advanced query demonstrations."""
    print("\n" + "=" * 70)
    print("PHASE 3: Advanced Query Patterns")
    print("=" * 70)

    # Department salary statistics
    print("\n--- Department Salary Statistics ---")
    stats = await department_salary_stats(session)
    for s in stats:
        print(
            f"  {s['department']:12s}  "
            f"avg=${s['avg_salary']:>10,.2f}  "
            f"min=${s['min_salary']:>10,.2f}  "
            f"max=${s['max_salary']:>10,.2f}  "
            f"count={s['emp_count']}"
        )

    # Employees above department average
    print("\n--- Employees Above Department Average ---")
    above_avg = await employees_above_department_avg(session)
    for e in above_avg:
        print(
            f"  {e['name']:16s}  ${e['salary']:>10,.2f}  "
            f"(dept avg: ${e['dept_avg']:>10,.2f})  "
            f"[{e['department']}]"
        )

    # Management hierarchy for Carol (id=3)
    print("\n--- Management Hierarchy for Carol (id=3) ---")
    chain = await management_hierarchy(session, 3)
    for i, emp in enumerate(chain):
        indent = "  " + "  " * i
        print(f"{indent}-> {emp.name} (id={emp.id})")

    # Management hierarchy for Ivy (id=9)
    print("\n--- Management Hierarchy for Ivy (id=9) ---")
    chain = await management_hierarchy(session, 9)
    for i, emp in enumerate(chain):
        indent = "  " + "  " * i
        print(f"{indent}-> {emp.name} (id={emp.id})")

    # Salary rank by department
    print("\n--- Salary Rank by Department ---")
    ranks = await salary_rank_by_department(session)
    current_dept = ""
    for r in ranks:
        if r["department"] != current_dept:
            current_dept = r["department"]
            print(f"\n  [{current_dept}]")
        print(f"    #{r['rank']}  {r['name']:16s}  ${r['salary']:>10,.2f}")


if __name__ == "__main__":
    import asyncio

    from app.engine import create_engine_factory, create_session_factory

    async def main() -> None:
        engine = create_engine_factory()
        session_factory = create_session_factory(engine)
        async with session_factory() as session:
            await run_advanced_queries_demo(session)
        await engine.dispose()

    asyncio.run(main())
