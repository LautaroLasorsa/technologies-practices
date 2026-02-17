-- =============================================================================
-- Seed data for SQLAlchemy 2.0 Async ORM practice.
-- Loaded by PostgreSQL on first container start via docker-entrypoint-initdb.d.
-- =============================================================================

CREATE TABLE departments (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    budget DECIMAL(12,2) NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE employees (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    salary DECIMAL(10,2) NOT NULL,
    hire_date DATE NOT NULL,
    department_id INTEGER REFERENCES departments(id),
    manager_id INTEGER REFERENCES employees(id),
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE projects (
    id SERIAL PRIMARY KEY,
    name VARCHAR(200) NOT NULL,
    status VARCHAR(50) DEFAULT 'active',
    budget DECIMAL(12,2),
    start_date DATE,
    end_date DATE
);

CREATE TABLE employee_projects (
    employee_id INTEGER REFERENCES employees(id),
    project_id INTEGER REFERENCES projects(id),
    role VARCHAR(100) DEFAULT 'member',
    PRIMARY KEY (employee_id, project_id)
);

-- ── Departments ──────────────────────────────────────────────────────

INSERT INTO departments (name, budget) VALUES
    ('Engineering', 500000),
    ('Marketing', 200000),
    ('Sales', 300000),
    ('HR', 150000),
    ('Research', 400000);

-- ── Employees ────────────────────────────────────────────────────────

INSERT INTO employees (name, email, salary, hire_date, department_id, manager_id) VALUES
    ('Alice Johnson',  'alice@co.com',  120000, '2020-01-15', 1, NULL),
    ('Bob Smith',      'bob@co.com',     95000, '2020-03-20', 1, 1),
    ('Carol Williams', 'carol@co.com',   88000, '2021-06-01', 1, 1),
    ('David Brown',    'david@co.com',  110000, '2019-11-10', 2, NULL),
    ('Eve Davis',      'eve@co.com',     92000, '2021-01-05', 2, 4),
    ('Frank Miller',   'frank@co.com',  105000, '2020-07-22', 3, NULL),
    ('Grace Lee',      'grace@co.com',   87000, '2022-02-14', 3, 6),
    ('Henry Wilson',   'henry@co.com',  130000, '2018-05-30', 5, NULL),
    ('Ivy Moore',      'ivy@co.com',     98000, '2021-09-15', 5, 8),
    ('Jack Taylor',    'jack@co.com',    91000, '2022-04-01', 1, 1);

-- ── Projects ─────────────────────────────────────────────────────────

INSERT INTO projects (name, status, budget, start_date, end_date) VALUES
    ('Platform Rewrite',  'active',    200000, '2024-01-01', '2024-12-31'),
    ('Mobile App',        'active',    150000, '2024-03-01', '2024-09-30'),
    ('Data Pipeline',     'completed',  80000, '2023-06-01', '2024-01-31'),
    ('Marketing Portal',  'active',     60000, '2024-02-01', NULL),
    ('Sales Dashboard',   'active',     45000, '2024-04-01', NULL);

-- ── Project Assignments ──────────────────────────────────────────────

INSERT INTO employee_projects (employee_id, project_id, role) VALUES
    (1, 1, 'lead'),    (2, 1, 'member'),  (3, 1, 'member'),
    (2, 2, 'lead'),    (10, 2, 'member'),
    (8, 3, 'lead'),    (9, 3, 'member'),
    (4, 4, 'lead'),    (5, 4, 'member'),
    (6, 5, 'lead'),    (7, 5, 'member');
