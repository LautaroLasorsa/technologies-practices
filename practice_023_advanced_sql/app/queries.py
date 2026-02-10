"""Advanced SQL queries — the main learning module.

Work through the TODO(human) markers in order, matching the phases
in CLAUDE.md. Each function should contain the SQL string to execute.

Use app.connection.run_query(sql) to execute and see results.
Use app.connection.run_explain(sql) to see EXPLAIN ANALYZE output.

How to work through this file:
  1. Read the docstring for each function — it explains what to write
  2. Replace the empty sql = "" with your SQL query
  3. Uncomment the function call in the __main__ block at the bottom
  4. Run: uv run python -m app.queries
  5. Check the output, tweak, repeat
"""

from app.connection import run_query, run_explain


# ═══════════════════════════════════════════════════════════════════════════
# Phase 2: Window Functions
# ═══════════════════════════════════════════════════════════════════════════


def window_rank_products_by_price():
    """Rank products by unit_price within each category.

    TODO(human): Write a SELECT that returns:
      - category name (from the categories table, aliased as "category")
      - product name (from the products table, aliased as "product")
      - unit_price
      - row_number: sequential position within category (no ties)
      - rank: position with gaps after ties
      - dense_rank: position without gaps after ties

    Use three window functions, all with the SAME window specification:
      OVER (PARTITION BY c.name ORDER BY p.unit_price DESC)

    PARTITION BY splits rows into groups (one per category).
    ORDER BY within the partition determines ranking order.

    Tables needed:
      products p JOIN categories c ON p.category_id = c.category_id

    Expected output example (for one category):
      category   | product          | unit_price | row_number | rank | dense_rank
      -----------+------------------+------------+------------+------+-----------
      Beverages  | Chang Beer       |      19.00 |          1 |    1 |          1
      Beverages  | Chai Tea         |      18.00 |          2 |    2 |          2
      ...

    Understanding the difference between these three ranking functions is
    essential because they behave differently when two products have the
    same price:
      ROW_NUMBER: always gives unique sequential numbers (1,2,3,4) — ties
                  get arbitrary ordering (nondeterministic within ties).
      RANK:       gives 1,2,2,4 — after a tie at position 2, it SKIPS to 4
                  (the next position accounts for the number of tied rows).
      DENSE_RANK: gives 1,2,2,3 — after a tie at position 2, the next
                  distinct value gets 3 (no gaps).

    When to use each:
      ROW_NUMBER — when you need exactly N rows (e.g., "give me 1 row per group")
      RANK       — when you need sports-style ranking ("no 3rd place if two 2nd places")
      DENSE_RANK — when you need "top N distinct values" (e.g., "top 3 price tiers")

    Hint: All three functions go in the SELECT clause:
      ROW_NUMBER() OVER (PARTITION BY ... ORDER BY ...) AS row_number,
      RANK()       OVER (PARTITION BY ... ORDER BY ...) AS rank,
      DENSE_RANK() OVER (PARTITION BY ... ORDER BY ...) AS dense_rank
    """
    sql = ""
    # TODO(human): Write the SQL query described above
    run_query(sql)


def window_top3_per_supplier():
    """Find top 3 products per supplier by total units sold.

    TODO(human): Write a query that:

    Step 1 — Aggregate total units sold per product:
      JOIN products with order_details to compute:
        total_sold = SUM(od.quantity)
      GROUP BY supplier_id, product_id, product name
      Also JOIN suppliers to get the supplier company_name.

    Step 2 — Rank products within each supplier:
      Use DENSE_RANK() OVER (PARTITION BY s.supplier_id ORDER BY total_sold DESC)
      to assign a rank to each product within its supplier.

    Step 3 — Filter to keep only top 3:
      You CANNOT use window functions in WHERE (they're computed after WHERE).
      Wrap the window function in a CTE (Common Table Expression):

        WITH ranked AS (
          SELECT
            s.company_name AS supplier,
            p.name AS product,
            SUM(od.quantity) AS total_sold,
            DENSE_RANK() OVER (
              PARTITION BY s.supplier_id
              ORDER BY SUM(od.quantity) DESC
            ) AS rnk
          FROM products p
          JOIN order_details od ON p.product_id = od.product_id
          JOIN suppliers s ON p.supplier_id = s.supplier_id
          GROUP BY s.supplier_id, s.company_name, p.product_id, p.name
        )
        SELECT supplier, product, total_sold, rnk
        FROM ranked
        WHERE rnk <= 3
        ORDER BY supplier, rnk

    Return columns: supplier, product, total_sold, rnk

    Why DENSE_RANK instead of ROW_NUMBER here?
      If two products tie in total_sold, DENSE_RANK gives them the same rank.
      This means you might get 4 products for a supplier if 2 tie at rank 2,
      which is usually the desired behavior ("top 3 selling levels").
      ROW_NUMBER would arbitrarily pick one of the tied products — less fair.

    Why can't window functions appear in WHERE?
      SQL evaluation order: FROM → WHERE → GROUP BY → HAVING → SELECT → ORDER BY
      Window functions are computed during SELECT, AFTER WHERE has already filtered.
      So you must compute the rank in a subquery/CTE first, then filter in the outer query.
    """
    sql = ""
    # TODO(human): Write the SQL query described above
    run_query(sql)


def window_lag_monthly_growth():
    """Calculate month-over-month order count growth using LAG.

    TODO(human): Write a query that shows how order volume changes month to month.

    Structure the query as a CTE for clarity:

    Step 1 — CTE "monthly": count orders per month
      SELECT
        DATE_TRUNC('month', order_date) AS month,
        COUNT(*) AS order_count
      FROM orders
      GROUP BY DATE_TRUNC('month', order_date)

      DATE_TRUNC('month', ...) truncates a date to the first of its month:
        '2024-03-15' → '2024-03-01'
      This groups all orders in March together regardless of day.

    Step 2 — Outer query: use LAG to compare with previous month
      SELECT
        month,
        order_count,
        LAG(order_count, 1) OVER (ORDER BY month) AS prev_month_count,
        ROUND(
          100.0 * (order_count - LAG(order_count, 1) OVER (ORDER BY month))
                / LAG(order_count, 1) OVER (ORDER BY month),
          1
        ) AS growth_pct
      FROM monthly
      ORDER BY month

    Return columns: month, order_count, prev_month_count, growth_pct

    How LAG works:
      LAG(column, offset, default) OVER (ORDER BY ...)
      - Looks BACK `offset` rows in the window ordering
      - LAG(order_count, 1) = "the order_count from 1 row before this one"
      - First row has no previous → returns NULL (or `default` if specified)

    Related function — LEAD:
      LEAD(column, 1) looks FORWARD 1 row instead of backward.
      Useful for "what happens next month?" analysis.

    The growth formula:
      growth_pct = 100 * (current - previous) / previous
      Positive = growth, negative = decline, NULL = first month (no comparison).

    Hint: You reference LAG(...) multiple times in the same SELECT — that's fine.
    PostgreSQL evaluates all window functions in the same pass. You could also
    wrap it in another CTE level to avoid repeating the LAG expression.
    """
    sql = ""
    # TODO(human): Write the SQL query described above
    run_query(sql)


def window_running_total():
    """Cumulative revenue per employee over time.

    TODO(human): Write a query that shows each employee's revenue accumulating
    order by order over time.

    Step 1 — Compute revenue per order:
      JOIN orders → order_details to calculate:
        order_revenue = SUM(od.unit_price * od.quantity * (1 - od.discount))
      GROUP BY employee, order_id, order_date
      Also JOIN employees to get employee name.

    Step 2 — Apply running total with a window frame:
      SUM(order_revenue) OVER (
        PARTITION BY e.employee_id
        ORDER BY o.order_date, o.order_id
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
      ) AS running_total

    Structure as a CTE:
      WITH order_revenues AS (
        SELECT
          e.employee_id,
          e.first_name || ' ' || e.last_name AS employee,
          o.order_id,
          o.order_date,
          ROUND(SUM(od.unit_price * od.quantity * (1 - od.discount)), 2) AS order_revenue
        FROM orders o
        JOIN order_details od ON o.order_id = od.order_id
        JOIN employees e ON o.employee_id = e.employee_id
        GROUP BY e.employee_id, e.first_name, e.last_name, o.order_id, o.order_date
      )
      SELECT
        employee,
        order_date,
        order_revenue,
        SUM(order_revenue) OVER (
          PARTITION BY employee_id
          ORDER BY order_date, order_id
          ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS running_total
      FROM order_revenues
      ORDER BY employee, order_date, order_id

    Return columns: employee, order_date, order_revenue, running_total

    Understanding the window frame clause:
      ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        → "from the very first row of this partition up to the current row"
        → This gives a CUMULATIVE sum.

      The frame clause is optional (it's the default when ORDER BY is present),
      but being explicit makes your intent crystal clear to readers.

    Other useful frame clauses:
      ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
        → 3-row moving average (current + 2 before)
      ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
        → total across entire partition (same value for every row)
      ROWS BETWEEN CURRENT ROW AND 2 FOLLOWING
        → forward-looking 3-row window

    Why ORDER BY includes order_id:
      Multiple orders can have the same date. Adding order_id ensures
      deterministic ordering within the same date.
    """
    sql = ""
    # TODO(human): Write the SQL query described above
    run_query(sql)


def window_ntile_quartiles():
    """Divide customers into spending quartiles using NTILE.

    TODO(human): Write a query that buckets customers into 4 spending groups.

    Step 1 — Compute total spent per customer:
      JOIN customers → orders → order_details to calculate:
        total_spent = SUM(od.unit_price * od.quantity * (1 - od.discount))
      GROUP BY customer

    Step 2 — Assign quartiles:
      NTILE(4) OVER (ORDER BY total_spent DESC) AS quartile
      (1 = top spenders, 4 = lowest spenders)

    Structure as a CTE:
      WITH customer_spending AS (
        SELECT
          c.customer_id,
          c.company_name,
          ROUND(SUM(od.unit_price * od.quantity * (1 - od.discount)), 2) AS total_spent
        FROM customers c
        JOIN orders o ON c.customer_id = o.customer_id
        JOIN order_details od ON o.order_id = od.order_id
        GROUP BY c.customer_id, c.company_name
      )
      SELECT
        company_name,
        total_spent,
        NTILE(4) OVER (ORDER BY total_spent DESC) AS quartile
      FROM customer_spending
      ORDER BY quartile, total_spent DESC

    Return columns: company_name, total_spent, quartile

    How NTILE works:
      NTILE(n) divides the ordered result set into n roughly equal-sized buckets.
      - 20 customers / 4 buckets = 5 customers per bucket
      - If not evenly divisible, earlier buckets get one extra row
      - Each row is assigned a bucket number from 1 to n

    NTILE vs RANK vs PERCENT_RANK:
      NTILE(4):      forces exactly 4 groups, roughly equal size
      RANK():        position-based, gaps possible, unequal group sizes
      PERCENT_RANK(): 0-1 scale, = (rank - 1) / (total_rows - 1)

    Use cases:
      NTILE(4)   → "divide customers into quartiles for marketing campaigns"
      NTILE(10)  → "assign decile scores for credit risk modeling"
      NTILE(100) → "compute percentile rankings"
    """
    sql = ""
    # TODO(human): Write the SQL query described above
    run_query(sql)


# ═══════════════════════════════════════════════════════════════════════════
# Phase 3: Common Table Expressions (CTEs)
# ═══════════════════════════════════════════════════════════════════════════


def cte_high_value_customers():
    """Multi-step CTE to identify high-value customer segments.

    TODO(human): Write a query using WITH (non-recursive) that classifies
    customers into spending segments using multiple CTE steps.

    Step 1 — CTE "customer_totals":
      Compute total spent per customer:
        SELECT
          c.customer_id,
          c.company_name,
          ROUND(SUM(od.unit_price * od.quantity * (1 - od.discount)), 2) AS total_spent
        FROM customers c
        JOIN orders o ON c.customer_id = o.customer_id
        JOIN order_details od ON o.order_id = od.order_id
        GROUP BY c.customer_id, c.company_name

    Step 2 — CTE "customer_frequency":
      Count distinct orders per customer:
        SELECT
          customer_id,
          COUNT(DISTINCT order_id) AS order_count
        FROM orders
        GROUP BY customer_id

    Step 3 — CTE "customer_segments":
      JOIN the two CTEs and classify customers using CASE:
        SELECT
          ct.customer_id,
          ct.company_name,
          ct.total_spent,
          cf.order_count,
          CASE
            WHEN ct.total_spent > 5000 AND cf.order_count > 10 THEN 'VIP'
            WHEN ct.total_spent > 1000 THEN 'Regular'
            ELSE 'Occasional'
          END AS segment
        FROM customer_totals ct
        JOIN customer_frequency cf ON ct.customer_id = cf.customer_id

    Step 4 — Final SELECT (aggregated):
        SELECT
          segment,
          COUNT(*) AS customer_count,
          ROUND(AVG(total_spent), 2) AS avg_spent,
          ROUND(AVG(order_count), 1) AS avg_orders
        FROM customer_segments
        GROUP BY segment
        ORDER BY avg_spent DESC

    Return columns: segment, customer_count, avg_spent, avg_orders

    Full syntax structure:
      WITH
        customer_totals AS ( ... ),
        customer_frequency AS ( ... ),
        customer_segments AS ( ... )
      SELECT ... FROM customer_segments GROUP BY ...

    Why CTEs instead of nested subqueries?
      1. Each CTE has a meaningful name — self-documenting code
      2. A CTE can be referenced multiple times without re-executing
      3. Easier to debug — test each CTE independently by commenting out later ones
      4. Reads top-to-bottom like a pipeline: compute totals → compute frequency →
         join & classify → aggregate

    Note: In PostgreSQL, CTEs are "optimization fences" before v12 (the optimizer
    couldn't push predicates into CTEs). Since v12, PostgreSQL can inline CTEs when
    beneficial. You're on v17, so no performance concern.
    """
    sql = ""
    # TODO(human): Write the SQL query described above
    run_query(sql)


def cte_recursive_employee_hierarchy():
    """Recursive CTE to traverse the employee reporting hierarchy.

    TODO(human): Write a WITH RECURSIVE query that builds the full
    reporting tree from CEO down to individual contributors.

    The recursive CTE has two parts connected by UNION ALL:

    Part 1 — Anchor (base case):
      SELECT employees where reports_to IS NULL — these are the top-level
      employees (CEO / VP). Start them at depth = 1.
        SELECT
          employee_id,
          first_name,
          last_name,
          title,
          reports_to,
          1 AS depth,
          first_name || ' ' || last_name AS path
        FROM employees
        WHERE reports_to IS NULL

    Part 2 — Recursive step:
      JOIN employees ON e.reports_to = hierarchy.employee_id
      This finds all employees who report to someone already in the result set.
      Increment depth and append to the path:
        SELECT
          e.employee_id,
          e.first_name,
          e.last_name,
          e.title,
          e.reports_to,
          h.depth + 1,
          h.path || ' -> ' || e.first_name || ' ' || e.last_name
        FROM employees e
        JOIN hierarchy h ON e.reports_to = h.employee_id

    Final SELECT:
      SELECT
        REPEAT('  ', depth - 1) || first_name || ' ' || last_name AS employee,
        title,
        depth,
        path
      FROM hierarchy
      ORDER BY path

    Return columns: employee (indented), title, depth, path

    How WITH RECURSIVE works (step by step):
      1. Execute the anchor query → produces initial rows (the CEO)
      2. Execute the recursive term, joining against the rows from step 1
         → produces the CEO's direct reports (managers)
      3. Execute the recursive term again, joining against rows from step 2
         → produces the managers' direct reports (staff)
      4. Repeat until the recursive term produces no new rows → STOP
      5. Final result = UNION ALL of all iterations

    The REPEAT('  ', depth - 1) is a visual trick:
      depth=1 → no indent     → "Andrew Fuller"
      depth=2 → 2 spaces      → "  Janet Leverling"
      depth=3 → 4 spaces      → "    Nancy Davolio"
      This creates a nice tree-like output in the terminal.

    ORDER BY path gives a natural tree ordering because path is built as:
      "Andrew Fuller"
      "Andrew Fuller -> Janet Leverling"
      "Andrew Fuller -> Janet Leverling -> Nancy Davolio"
      Alphabetical sorting of these strings groups each subtree together.

    Safety note: PostgreSQL has no built-in cycle detection for recursive CTEs
    (unlike Oracle's CONNECT BY with NOCYCLE). If your data has cycles
    (A reports to B, B reports to A), the query runs forever. PostgreSQL 14+
    added CYCLE detection syntax: CYCLE employee_id SET is_cycle USING path_array.
    """
    sql = ""
    # TODO(human): Write the SQL query described above
    run_query(sql)


def cte_recursive_category_tree():
    """Recursive CTE to traverse the category tree.

    TODO(human): Write a WITH RECURSIVE query that builds the full
    category hierarchy from root categories down to subcategories.

    Part 1 — Anchor (base case):
      Root categories have parent_category_id IS NULL.
      Start them at depth = 0.
        SELECT
          category_id,
          name,
          parent_category_id,
          0 AS depth,
          name AS full_path
        FROM categories
        WHERE parent_category_id IS NULL

    Part 2 — Recursive step:
      Find categories whose parent_category_id matches an already-found category:
        SELECT
          c.category_id,
          c.name,
          c.parent_category_id,
          tree.depth + 1,
          tree.full_path || ' > ' || c.name
        FROM categories c
        JOIN category_tree tree ON c.parent_category_id = tree.category_id

    Final SELECT:
      SELECT
        REPEAT('  ', depth) || name AS category,
        depth,
        full_path
      FROM category_tree
      ORDER BY full_path

    Return columns: category (indented), depth, full_path

    This is the exact same recursive pattern as the employee hierarchy,
    applied to a different domain (product categories instead of people).
    Self-referential foreign keys (parent_category_id → category_id) are
    the classic use case for recursive CTEs.

    Without recursion, you'd need:
      - One query per depth level
      - You wouldn't know how many levels deep the tree goes
      - The code would break if someone adds deeper nesting

    With recursive CTE:
      - One query handles ANY depth
      - Adding deeper categories requires zero code changes
      - The full_path column gives a breadcrumb trail:
        "Beverages > Hot Beverages"
        "Produce > Organic Produce"

    Real-world uses of recursive CTEs on tree structures:
      - Organizational charts (employees)
      - Category/taxonomy trees (e-commerce)
      - Bill of Materials (manufacturing — parts contain sub-parts)
      - File system paths (folders contain subfolders)
      - Comment threads (replies to replies)
    """
    sql = ""
    # TODO(human): Write the SQL query described above
    run_query(sql)


# ═══════════════════════════════════════════════════════════════════════════
# Phase 4: Query Optimization
# ═══════════════════════════════════════════════════════════════════════════


def optimization_slow_query():
    """Run a query WITHOUT indexes and inspect the execution plan.

    TODO(human): Write a query that filters orders by date range and
    then use run_explain() to see how PostgreSQL executes it.

    The query:
      SELECT o.order_id, o.order_date, c.company_name, o.freight
      FROM orders o
      JOIN customers c ON o.customer_id = c.customer_id
      WHERE o.order_date BETWEEN '2024-01-01' AND '2024-06-30'
      ORDER BY o.order_date

    Then call: run_explain(sql)

    What to look for in the EXPLAIN ANALYZE output:

    1. "Seq Scan on orders" — this means PostgreSQL is reading EVERY row
       in the orders table, then checking the WHERE condition on each.
       With 200 orders this is fine, but with 10M rows it's catastrophic.

    2. "actual time=X..Y" — X is startup time (before first row produced),
       Y is total time to produce all rows from this node.

    3. "rows=N" — Two numbers: estimated (from statistics) and actual.
       If these differ wildly, your table statistics are stale (run ANALYZE).

    4. "Planning Time" and "Execution Time" at the bottom — total time.
       Planning Time is the optimizer thinking about the best plan.
       Execution Time is actually running it.

    5. "Buffers: shared hit=N read=M" — N pages served from cache,
       M pages read from disk. More hits = better caching.

    This establishes the baseline BEFORE adding indexes.
    After running this, proceed to optimization_create_index() to see
    the improvement.

    Hint: Just write the SELECT query and pass it to run_explain(sql).
    The function wraps it in EXPLAIN (ANALYZE, BUFFERS, FORMAT TEXT) for you.
    """
    sql = ""
    # TODO(human): Write the SQL query and call run_explain(sql)


def optimization_create_index():
    """Create an index and re-run the query to measure improvement.

    TODO(human): Do two things in this function:

    1. Create a B-tree index on orders(order_date):
         CREATE INDEX IF NOT EXISTS idx_orders_date ON orders(order_date);
       Execute this with run_query() — it's a DDL statement.

    2. Re-run the SAME query from optimization_slow_query() and use
       run_explain() to see the new execution plan.

    What changes in the EXPLAIN output:

    BEFORE (no index):
      Seq Scan on orders  (cost=0.00..5.00 rows=100 ...)
        Filter: (order_date >= '2024-01-01' AND order_date <= '2024-06-30')

    AFTER (with index):
      Index Scan using idx_orders_date on orders  (cost=0.14..3.50 rows=100 ...)
        Index Cond: (order_date >= '2024-01-01' AND order_date <= '2024-06-30')

    Key differences:
      - "Seq Scan" → "Index Scan" or "Bitmap Index Scan"
      - "Filter" → "Index Cond" (condition pushed into the index lookup)
      - Lower cost estimates and actual time values
      - Fewer buffers read (index narrows down which pages to read)

    Types of index scans in PostgreSQL:
      Index Scan:        reads index → fetches matching table rows one by one.
                         Best for small result sets.
      Bitmap Index Scan: reads index → builds a bitmap of matching pages →
                         reads pages in physical order. Best for medium result sets.
      Seq Scan:          reads entire table. Best for large result sets (>~10-20%
                         of table). The optimizer chooses whichever is cheapest.

    B-tree indexes (the default) work best for:
      - Range queries: BETWEEN, <, >, <=, >=
      - Equality: =
      - Sorting: ORDER BY (can skip the sort step entirely!)
      - Prefix matching: LIKE 'abc%' (but NOT LIKE '%abc')

    Other PostgreSQL index types:
      Hash:  only equality (=), slightly faster than B-tree for pure equality
      GIN:   full-text search, JSONB containment, array operations
      GiST:  geometric data, range types, nearest-neighbor searches
      BRIN:  very large tables with naturally ordered data (time series)
    """
    # TODO(human): Create index with run_query(), then re-run the date range query with run_explain()
    pass


def optimization_composite_index():
    """Experiment with a composite (multi-column) index.

    TODO(human): Do three things:

    1. Create a composite index on orders(customer_id, order_date):
         CREATE INDEX IF NOT EXISTS idx_orders_customer_date
         ON orders(customer_id, order_date);
       Execute with run_query().

    2. Run a query that benefits from BOTH columns in the index:
         SELECT o.order_id, o.order_date, o.freight
         FROM orders o
         WHERE o.customer_id = 5
           AND o.order_date BETWEEN '2024-01-01' AND '2024-06-30'
       Execute with run_explain() to inspect the plan.

    3. Try a query that only filters on order_date (the SECOND column):
         SELECT o.order_id, o.order_date, o.freight
         FROM orders o
         WHERE o.order_date BETWEEN '2024-01-01' AND '2024-06-30'
       Execute with run_explain() and compare — does it use idx_orders_customer_date?

    The "leftmost prefix" rule for composite indexes:
      An index on (A, B) can be used for queries filtering on:
        ✓ A alone         → uses the index
        ✓ A AND B         → uses the index (most efficient)
        ✗ B alone         → CANNOT use this index (needs a separate index on B)

    Analogy — think of a phone book sorted by (last_name, first_name):
      ✓ "Find all Smiths"                    → easy, they're grouped together
      ✓ "Find John Smith"                    → easy, go to Smiths, then find John
      ✗ "Find all Johns" (any last name)     → impossible without scanning the whole book

    Column order matters! When designing composite indexes, put:
      1. Equality columns first (customer_id = 5)
      2. Range columns second (order_date BETWEEN ...)
    This is because the B-tree can narrow to an exact subtree for equality,
    then scan a range within that subtree.

    Covering indexes (bonus concept):
      If the index contains ALL columns the query needs, PostgreSQL can
      answer the query from the index alone without touching the table.
      This is called an "index-only scan" — the fastest possible access.
      CREATE INDEX idx_covering ON orders(customer_id, order_date) INCLUDE (freight);
    """
    # TODO(human): Create composite index, run both queries with run_explain()
    pass


# ═══════════════════════════════════════════════════════════════════════════
# Phase 5: Advanced Joins & PostgreSQL Features
# ═══════════════════════════════════════════════════════════════════════════


def lateral_top3_per_customer():
    """LATERAL JOIN: Find top 3 most ordered products per customer.

    TODO(human): Write a query using LATERAL to solve the classic
    "top N per group" problem elegantly.

    The query:
      SELECT c.company_name, top.product_name, top.times_ordered
      FROM customers c,
      LATERAL (
        SELECT p.name AS product_name, COUNT(*) AS times_ordered
        FROM orders o
        JOIN order_details od ON o.order_id = od.order_id
        JOIN products p ON od.product_id = p.product_id
        WHERE o.customer_id = c.customer_id
        GROUP BY p.name
        ORDER BY times_ordered DESC
        LIMIT 3
      ) AS top
      ORDER BY c.company_name, top.times_ordered DESC

    Return columns: company_name, product_name, times_ordered

    How LATERAL works:
      In a normal JOIN, the right side CANNOT reference the left side.
      LATERAL removes this restriction — the subquery CAN reference columns
      from preceding FROM items (here, c.customer_id from the customers table).

    It's like a "foreach" loop in SQL:
      FOR EACH customer c:
        run the subquery (top 3 products for THIS customer)
        join the results back

    Why LATERAL over alternatives?

    Alternative 1 — Window function + filter:
      WITH ranked AS (
        SELECT customer_id, product_name, times_ordered,
               ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY times_ordered DESC) AS rn
        FROM ...
      )
      SELECT * FROM ranked WHERE rn <= 3
      → Works, but computes the rank for ALL rows in ALL groups first,
        then throws away most of them. Wasteful for large tables.

    Alternative 2 — Correlated subquery in SELECT:
      SELECT c.company_name,
             (SELECT p.name FROM ... WHERE o.customer_id = c.customer_id
              ORDER BY count DESC LIMIT 1)
      → Can only return ONE column and ONE row. Can't get top 3.

    LATERAL is:
      ✓ The most readable (intent is clear: "for each customer, get top 3")
      ✓ Often the most efficient (the LIMIT 3 stops early per group)
      ✓ Flexible (the subquery can return multiple rows and columns)

    Syntax note: "FROM customers c, LATERAL (...)" is equivalent to
    "FROM customers c CROSS JOIN LATERAL (...)". The comma syntax is shorter.
    If a customer has NO orders, they won't appear (implicit INNER JOIN).
    Use "LEFT JOIN LATERAL (...) ON TRUE" to include them with NULLs.
    """
    sql = ""
    # TODO(human): Write the LATERAL JOIN query described above
    run_query(sql)


def selfjoin_products_ordered_together():
    """Self-join: Find pairs of products frequently ordered together.

    TODO(human): Write a query that identifies which product pairs appear
    together in the same order most often. This is the classic "market basket
    analysis" pattern used by recommendation engines ("customers who bought X
    also bought Y").

    The query:
      SELECT
        p1.name AS product_1,
        p2.name AS product_2,
        COUNT(*) AS times_together
      FROM order_details od1
      JOIN order_details od2
        ON od1.order_id = od2.order_id
        AND od1.product_id < od2.product_id
      JOIN products p1 ON od1.product_id = p1.product_id
      JOIN products p2 ON od2.product_id = p2.product_id
      GROUP BY p1.name, p2.name
      ORDER BY times_together DESC
      LIMIT 10

    Return columns: product_1, product_2, times_together

    Why the self-join works:
      order_details has multiple rows per order (one per product in that order).
      Joining order_details to itself ON the same order_id pairs up every
      product with every other product in the same order.

    The critical trick: od1.product_id < od2.product_id

      Without it:
        (Chai Tea, Chang Beer) AND (Chang Beer, Chai Tea) — counted twice!
        (Chai Tea, Chai Tea) — paired with itself!

      With < instead of !=:
        Only (Chai Tea, Chang Beer) — each pair appears exactly once
        No self-pairs — a product can't be "ordered together" with itself

      Using < (not !=) is better than != + DISTINCT because:
        - < produces half the rows → less work for GROUP BY
        - No need for DISTINCT → simpler plan

    Performance consideration:
      This self-join produces O(n^2) rows per order (where n = items per order).
      For orders with 5 items: 5*4/2 = 10 pairs. Manageable.
      For orders with 100 items: 100*99/2 = 4,950 pairs. Still okay.
      For very large datasets, consider pre-aggregating or sampling.

    Real-world applications:
      - Amazon: "Frequently bought together"
      - Netflix: "Because you watched X" (co-viewing pairs)
      - Grocery stores: product placement (beer and diapers on same aisle)
    """
    sql = ""
    # TODO(human): Write the self-join query described above
    run_query(sql)


def generate_series_date_gaps():
    """GENERATE_SERIES: Fill gaps in the order time series.

    TODO(human): Write a query that shows order counts for EVERY month
    in 2024, including months with zero orders (which would otherwise
    be missing from a simple GROUP BY).

    The query:
      SELECT
        months.month,
        COALESCE(order_counts.cnt, 0) AS order_count
      FROM
        GENERATE_SERIES(
          '2024-01-01'::date,
          '2024-12-01'::date,
          '1 month'::interval
        ) AS months(month)
      LEFT JOIN (
        SELECT
          DATE_TRUNC('month', order_date) AS month,
          COUNT(*) AS cnt
        FROM orders
        WHERE order_date >= '2024-01-01' AND order_date < '2025-01-01'
        GROUP BY DATE_TRUNC('month', order_date)
      ) AS order_counts ON months.month = order_counts.month
      ORDER BY months.month

    Return columns: month, order_count

    How GENERATE_SERIES works:
      GENERATE_SERIES(start, stop, step) produces a set of values:
        GENERATE_SERIES(1, 5, 1)           → 1, 2, 3, 4, 5
        GENERATE_SERIES('2024-01-01'::date, '2024-12-01'::date, '1 month')
          → 2024-01-01, 2024-02-01, ..., 2024-12-01

    Why this matters:
      Without GENERATE_SERIES, a GROUP BY on order_date only shows months
      that HAVE orders. If March had zero orders, March wouldn't appear at all.
      This breaks:
        - Time series charts (gaps create misleading trend lines)
        - Month-over-month calculations (LAG would skip months)
        - Dashboard displays (users expect all 12 months to show)

    LEFT JOIN is essential:
      The GENERATE_SERIES goes on the LEFT side, and the actual data on the RIGHT.
      Months with no orders get NULL from the right side → COALESCE converts to 0.
      If you used INNER JOIN, months with no orders would disappear (defeating the purpose).

    GENERATE_SERIES is PostgreSQL-specific. Alternatives in other databases:
      - MySQL: Create a calendar table or use recursive CTE to generate dates
      - SQL Server: Use a numbers table or recursive CTE
      - BigQuery: GENERATE_DATE_ARRAY()

    Other uses of GENERATE_SERIES:
      - Generate test data: GENERATE_SERIES(1, 1000000)
      - Create time slots: GENERATE_SERIES(start, end, '30 minutes')
      - Build a calendar/date dimension table for a data warehouse
    """
    sql = ""
    # TODO(human): Write the GENERATE_SERIES query described above
    run_query(sql)


def json_aggregation():
    """JSON aggregation: Build JSON order summaries per customer.

    TODO(human): Write a query that produces a JSON array of recent orders
    for each customer, demonstrating PostgreSQL's powerful JSON capabilities.

    The query:
      SELECT
        c.company_name,
        COUNT(o.order_id) AS total_orders,
        jsonb_agg(
          jsonb_build_object(
            'order_id', o.order_id,
            'date', o.order_date,
            'total', (
              SELECT ROUND(SUM(od.unit_price * od.quantity * (1 - od.discount)), 2)
              FROM order_details od
              WHERE od.order_id = o.order_id
            )
          ) ORDER BY o.order_date DESC
        ) AS recent_orders
      FROM customers c
      JOIN orders o ON c.customer_id = o.customer_id
      GROUP BY c.customer_id, c.company_name
      ORDER BY total_orders DESC
      LIMIT 5

    Return columns: company_name, total_orders, recent_orders (JSONB array)

    How the JSON functions work:

    jsonb_build_object(key1, value1, key2, value2, ...):
      Creates a JSON object from key-value pairs:
        jsonb_build_object('name', 'Chai', 'price', 18.00)
        → {"name": "Chai", "price": 18.00}

    jsonb_agg(expression ORDER BY ...):
      Aggregates multiple JSON values into a JSON array:
        [{"order_id": 1, "date": "2024-12-15", "total": 523.40},
         {"order_id": 2, "date": "2024-11-20", "total": 128.00},
         ...]
      The ORDER BY inside jsonb_agg controls the array element ordering
      (newest orders first, in this case).

    Why build JSON in SQL?
      In application code, the naive approach is:
        1. Query customers (1 query)
        2. For each customer, query their orders (N queries)
        3. For each order, query order details (N*M queries)
        → N+1 problem: hundreds of database round-trips

      With JSON aggregation:
        1. One single query returns everything
        2. Application receives pre-formatted JSON
        3. Zero N+1 problem, minimal round-trips

    JSONB vs JSON in PostgreSQL:
      JSON:  stored as text, preserves formatting and key order, no indexing
      JSONB: stored as binary, faster to query, supports GIN indexes,
             doesn't preserve key order or whitespace
      Rule of thumb: always use JSONB unless you need exact text preservation.

    GIN indexes on JSONB (bonus):
      CREATE INDEX idx_jsonb ON table USING GIN (jsonb_column);
      Enables fast queries like:
        WHERE jsonb_column @> '{"status": "active"}'  (containment)
        WHERE jsonb_column ? 'key_name'                (key existence)
    """
    sql = ""
    # TODO(human): Write the JSON aggregation query described above
    run_query(sql)


# ═══════════════════════════════════════════════════════════════════════════
# Main: Run all queries in order
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("  Advanced SQL — Work through each function, replacing the empty")
    print("  sql strings with your queries. Uncomment functions as you go.")
    print("=" * 70)
    print()

    # Phase 2: Window Functions
    # print("--- Phase 2: Window Functions ---")
    # window_rank_products_by_price()
    # window_top3_per_supplier()
    # window_lag_monthly_growth()
    # window_running_total()
    # window_ntile_quartiles()

    # Phase 3: CTEs
    # print("--- Phase 3: CTEs ---")
    # cte_high_value_customers()
    # cte_recursive_employee_hierarchy()
    # cte_recursive_category_tree()

    # Phase 4: Query Optimization
    # print("--- Phase 4: Query Optimization ---")
    # optimization_slow_query()
    # optimization_create_index()
    # optimization_composite_index()

    # Phase 5: Advanced Joins & PostgreSQL Features
    # print("--- Phase 5: Advanced Joins & PostgreSQL Features ---")
    # lateral_top3_per_customer()
    # selfjoin_products_ordered_together()
    # generate_series_date_gaps()
    # json_aggregation()

    print("Uncomment the functions above as you implement each TODO(human).")
    print("Run with: uv run python -m app.queries")
