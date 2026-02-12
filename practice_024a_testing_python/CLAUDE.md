# Practice 024a: Testing Patterns — Python

## Technologies

- **pytest 8.x** — Python's most popular testing framework with fixtures, parametrize, and markers
- **Hypothesis 6.x** — Property-based testing library for automatic test case generation and shrinking
- **pytest-mock** — Thin wrapper around unittest.mock for advanced mocking with call tracking
- **pytest-cov** — Coverage measurement plugin with terminal and HTML reporting

## Stack

- Python 3.12+ (uv)

## Theoretical Context

### What are pytest, fixtures, and property-based testing, and what problems do they solve?

**pytest** is Python's de facto testing framework, solving the problem of **test complexity** and **boilerplate** in unittest (the standard library). pytest's key innovations: automatic test discovery, powerful fixtures with dependency injection, parametrization for data-driven tests, and expressive assertions (plain `assert`, no `self.assertEqual` noise).

**Fixtures** solve the **setup/teardown duplication** problem. Instead of repeating resource creation (`setup_method`) in every test class, fixtures are reusable, composable functions that pytest automatically injects by parameter name. They support scopes (function, class, module, session) for controlling lifecycle and sharing expensive resources.

**Property-based testing (Hypothesis)** solves the **example bias** problem. Traditional tests use hand-picked examples (`test_add_positive_numbers()`), which miss edge cases (what about overflow? negative numbers? zero?). Property-based testing generates **hundreds of random inputs** from strategies, finds failing cases via **shrinking** (automatically minimizing the failing input to the simplest counterexample), and verifies **invariants** that should hold for ALL valid inputs.

### How they work internally

**pytest test discovery and execution:**

1. **Discovery**: pytest recursively searches for files matching `test_*.py` or `*_test.py`, then collects functions/classes matching `test_*` or `Test*`.
2. **Fixture resolution**: For each test function, pytest inspects its parameters, matches them to registered fixtures (from `conftest.py` or the same file), builds a **dependency graph**, and resolves in topological order.
3. **Execution**: pytest runs setup (fixtures), then the test body, then teardown (fixture finalizers). If a fixture has `scope="module"`, it's created once and reused across all tests in that module.
4. **Assertion rewriting**: pytest hooks into the import system to rewrite `assert` statements at bytecode level, enabling rich failure messages (shows left/right values, not just "AssertionError").

**Fixture scopes and caching:**

- `scope="function"` (default): Fresh instance per test. Full isolation but slower.
- `scope="class"`: Shared across all tests in a class. Faster but tests can interfere.
- `scope="module"`: Shared across all tests in a file. Use for expensive setup (database connection).
- `scope="session"`: Shared across the entire test run. Use for global resources (Docker containers).

Fixtures can `yield` a value, run test code, then execute teardown after the yield (context manager pattern).

**Hypothesis: strategies, shrinking, and invariant testing:**

1. **Strategy**: A description of how to generate random test data (e.g., `st.integers(min_value=0, max_value=100)`, `st.lists(st.text())`, `st.floats(allow_nan=False)`). Strategies can be composed (e.g., `st.tuples(st.integers(), st.booleans())`).
2. **Execution**: Hypothesis calls your test function 100+ times (configurable) with different generated inputs. If any input fails, Hypothesis **shrinks** it — repeatedly tries smaller/simpler inputs that still fail, stopping at the minimal failing example.
3. **Shrinking algorithm**: Hypothesis uses a **greedy best-first search** that tries deleting elements, reducing numbers, replacing strings with shorter versions, etc., always keeping changes that preserve the failure.
4. **Database**: Hypothesis caches failing examples in `.hypothesis/` so they're re-run on future executions (regression detection).

**Property testing mental model:** Instead of "assert f(5) == 25", write "for all x, f(f_inverse(x)) == x" (round-trip property) or "for all deposits d, balance_after >= balance_before" (invariant).

### Key concepts

| Concept | Description |
|---------|-------------|
| **Fixture** | A reusable setup function that pytest injects by parameter name; supports scopes and teardown via yield |
| **conftest.py** | A special file where fixtures are defined and automatically discovered across the test suite |
| **Parametrize** | `@pytest.mark.parametrize` runs the same test with multiple input/output pairs (data-driven testing) |
| **Marker** | `@pytest.mark.unit`, `@pytest.mark.slow` — tags for categorizing tests; run subsets via `pytest -m unit` |
| **Monkeypatch** | pytest fixture for replacing attributes/env vars at runtime, automatically reverts after the test |
| **mocker (pytest-mock)** | Wraps `unittest.mock` with pytest-friendly API; creates mocks/spies, verifies calls, and auto-cleans up |
| **Property-based testing** | Testing approach that generates random inputs and checks invariants, not specific examples |
| **Hypothesis strategy** | A specification of how to generate random test data (e.g., `st.integers()`, `st.lists(st.text())`) |
| **Shrinking** | Hypothesis's algorithm for minimizing a failing input to the simplest counterexample |
| **Invariant** | A property that should always hold (e.g., "sorted list is never longer than input list") |
| **@given decorator** | Hypothesis decorator that wraps a test function, injecting generated arguments from strategies |

### Ecosystem context

**Alternatives and trade-offs:**

| Framework | Strengths | Weaknesses |
|-----------|-----------|------------|
| **unittest** | Standard library, no dependencies | Verbose (self.assertEqual, setUp/tearDown), no auto-discovery by default |
| **pytest** | Minimal boilerplate, powerful fixtures, parametrize, plugins | Extra dependency (though nearly universal) |
| **Hypothesis** | Finds edge cases automatically, shrinks to minimal failing input | Slower (runs 100+ cases), requires thinking in invariants |
| **doctest** | Tests embedded in docstrings, ensures docs stay correct | Limited assertions, poor failure messages, hard to maintain |

**When to use each:**

- **pytest fixtures**: Anything needing setup/teardown (DB connections, API clients, test data)
- **parametrize**: Testing the same logic with multiple inputs (boundary values, equivalence classes)
- **mocking (mocker)**: Isolating unit tests from external dependencies (network, filesystem, time, random)
- **Hypothesis**: Testing invariants that should hold for all inputs (parsers, encoders, data structures, math functions)

**Limitations:**

- **Hypothesis**: Requires rethinking tests as properties, not examples. Not suitable for tests that depend on specific inputs (e.g., "user 123 exists").
- **Fixtures**: Over-use can make tests hard to understand (magic injection). Balance with explicit setup when clarity matters.
- **Mocking**: Over-mocking tests implementation details instead of behavior (brittle tests). Mock only external dependencies, not internal calls.

## Description

Build a Wallet service and its comprehensive test suite, practicing every major Python testing pattern. The domain is a simple banking wallet: deposit, withdraw, transfer between wallets, and transaction history. This domain naturally demonstrates invariants (balance never negative), external dependencies (notification service), stateful behavior (transaction lifecycle), and TDD workflow.

Focus is on testing patterns, not the wallet itself — the wallet code is partially provided and partially TODO(human).

### What you'll learn

1. **pytest fundamentals** — fixtures with scopes, parametrize for data-driven tests, markers for categorization
2. **Mocking patterns** — monkeypatch vs pytest-mock, dependency injection for testability
3. **Property-based testing** — Hypothesis @given, strategies, shrinking, invariant-based testing
4. **Integration testing** — repository pattern, fixture scoping, test isolation
5. **TDD workflow** — red-green-refactor cycle with transaction history feature
6. **Test organization** — conftest hierarchy, naming conventions, running subsets

## Instructions

### Phase 1: Setup & pytest Fundamentals (~20 min)

1. From this folder: `uv sync`
2. Explore the project structure: `wallet/` has the domain code, `tests/` has the test suite
3. Run the existing tests: `uv run pytest -v` — some pass (deposit), most fail (TODO stubs)
4. Open `wallet/service.py` — read the fully-implemented `deposit()` as a reference pattern
5. **User implements:** `withdraw()` in `wallet/service.py` following the deposit pattern
6. **User implements:** all tests in `TestWithdraw` class in `tests/unit/test_wallet.py`
7. Run: `uv run pytest tests/unit/test_wallet.py::TestWithdraw -v` — verify all pass
8. Key question: Why use fixtures instead of setUp/tearDown? What advantage does pytest's fixture injection give over xUnit-style setup?

### Phase 2: Mocking & Dependency Injection (~20 min)

1. Claude explains: monkeypatch replaces attributes/env vars (scope-limited patching); mocker (pytest-mock) wraps unittest.mock with call tracking, assertion helpers, and spy capabilities
2. **User implements:** `transfer()` in `wallet/service.py` — moves funds and optionally notifies
3. **User implements:** all tests in `TestTransfer` class in `tests/unit/test_wallet.py`
4. Run: `uv run pytest tests/unit/test_wallet.py::TestTransfer -v`
5. Key question: When to use monkeypatch vs mocker? (Answer: monkeypatch for replacing values/env; mocker for verifying interactions — calls, arguments, call count)

### Phase 3: Property-Based Testing (~25 min)

1. Claude explains: Hypothesis `@given` generates random inputs from strategies. If a test fails, Hypothesis *shrinks* the input to the smallest failing example. Property tests verify invariants that hold for ALL valid inputs, not just hand-picked examples.
2. Open `tests/property/test_wallet_properties.py` — read the strategy definitions at the top
3. **User implements:** all property tests in `TestWalletProperties`
4. Run: `uv run pytest tests/property/ -v` — observe Hypothesis generating hundreds of cases
5. Key question: What invariants does the wallet guarantee? (balance >= 0, deposit+withdraw inverse, transfer conserves total)

### Phase 4: Integration Testing & TDD (~20 min)

1. Open `tests/unit/test_tdd_history.py` — these tests are already written but FAIL
2. Run: `uv run pytest tests/unit/test_tdd_history.py -v` — observe the RED phase
3. **User implements:** `get_history()` in `wallet/service.py` to make all tests pass (GREEN)
4. Refactor if needed — keep tests passing (REFACTOR)
5. **User implements:** persistence tests in `tests/integration/test_persistence.py`
6. Run: `uv run pytest tests/integration/ -v`
7. Key question: How does fixture scope affect test isolation? (scope="function" = fresh per test; scope="module" = shared, faster but tests can interfere)

### Phase 5: Coverage & Organization (~10 min)

1. Run coverage: `uv run pytest --cov=wallet --cov-report=term-missing`
2. Identify uncovered lines — are they TODO stubs or missed branches?
3. Run by marker: `uv run pytest -m unit`, `uv run pytest -m property`, `uv run pytest -m integration`
4. Final discussion: How would you organize tests for a larger project? When do property tests add the most value?

## Motivation

- **Quality engineering**: Testing is the #1 differentiator between junior and senior engineers
- **Python ecosystem**: pytest + Hypothesis is the gold standard for Python testing
- **Directly applicable**: AutoScheduler.AI Python services need comprehensive test suites
- **Interview essential**: Testing patterns and TDD are common senior-level interview topics

## References

- [pytest Documentation](https://docs.pytest.org/en/stable/)
- [Hypothesis Documentation](https://hypothesis.readthedocs.io/en/latest/)
- [pytest-mock Documentation](https://pytest-mock.readthedocs.io/en/latest/)
- [Real Python: Effective Python Testing](https://realpython.com/pytest-python-testing/)

## Commands

### Setup

| Command | Description |
|---------|-------------|
| `uv sync` | Install all dependencies (pytest, hypothesis, pytest-mock, pytest-cov) |

### Run Tests

| Command | Description |
|---------|-------------|
| `uv run pytest` | Run all tests with default options |
| `uv run pytest -v` | Run all tests with verbose output (shows each test name and result) |
| `uv run pytest -x` | Stop on first failure (useful during development) |
| `uv run pytest -k "test_deposit"` | Run only tests matching the keyword expression "test_deposit" |

### Run by Marker

| Command | Description |
|---------|-------------|
| `uv run pytest -m unit` | Run only unit tests (fast, isolated) |
| `uv run pytest -m property` | Run only property-based tests (Hypothesis) |
| `uv run pytest -m integration` | Run only integration tests (slower, may use I/O) |

### Run Specific Files / Classes

| Command | Description |
|---------|-------------|
| `uv run pytest tests/unit/test_wallet.py::TestDeposit -v` | Run only deposit tests |
| `uv run pytest tests/unit/test_wallet.py::TestWithdraw -v` | Run only withdraw tests |
| `uv run pytest tests/unit/test_wallet.py::TestTransfer -v` | Run only transfer tests |
| `uv run pytest tests/unit/test_tdd_history.py -v` | Run TDD history tests (Phase 4) |
| `uv run pytest tests/property/ -v` | Run all property-based tests |
| `uv run pytest tests/integration/ -v` | Run all integration tests |

### Coverage

| Command | Description |
|---------|-------------|
| `uv run pytest --cov=wallet --cov-report=term-missing` | Run all tests with coverage, show uncovered lines |
| `uv run pytest --cov=wallet --cov-report=html` | Generate HTML coverage report in `htmlcov/` |

## State

`not-started`
