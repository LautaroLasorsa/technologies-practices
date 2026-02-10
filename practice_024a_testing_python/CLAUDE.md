# Practice 024a: Testing Patterns — Python

## Technologies

- **pytest 8.x** — Python's most popular testing framework with fixtures, parametrize, and markers
- **Hypothesis 6.x** — Property-based testing library for automatic test case generation and shrinking
- **pytest-mock** — Thin wrapper around unittest.mock for advanced mocking with call tracking
- **pytest-cov** — Coverage measurement plugin with terminal and HTML reporting

## Stack

- Python 3.12+ (uv)

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
