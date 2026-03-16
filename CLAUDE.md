# SchwabGym

High-fidelity RL environment for algorithmic trading with API parity to Charles Schwab's `schwab-py`.

## Quick Reference

```bash
# Run tests
pytest                          # full suite (148 tests, ~4s)
pytest tests/test_client.py     # single file
pytest -k "test_market_order"   # by name pattern
pytest -m "not slow"            # skip slow tests

# Lint & format (ruff replaces black + isort + flake8)
ruff check .                    # lint
ruff check --fix .              # lint + auto-fix
ruff format .                   # format

# Type checking
mypy schwabgym                  # type check source (not tests)

# Coverage
pytest --cov --cov-report=term-missing
```

## Architecture

```
schwabgym/
  client.py          # MockClient - main entry point, delegates to components
  prices.py          # PriceEngine - market data replay and time advancement
  account.py         # Account - positions, cash, PDT enforcement, margin
  order_manager.py   # OrderManager - order lifecycle (place/cancel/replace)
  orders.py          # MockEquities/MockResponse - order builders, API response
  environment.py     # SchwabTradingEnv - Gymnasium wrapper for RL
  fees.py            # FeeCalculator - SEC Section 31, FINRA TAF
  data.py            # Data loading, cleaning, technical indicators
  market_calendar.py # Trading sessions, holidays
  streamer.py        # MockStreamer - streaming data simulation
  physics/           # Execution engines
    base.py          #   ExecutionEngine ABC
    fast.py          #   FastExecutionEngine - instant fills
    realistic.py     #   RealisticExecutionEngine - Square Root Law (default)
    hybrid.py        #   HybridExecutionEngine - probabilistic switching
    almgren_chriss.py #  AlmgrenChrissOptimalExecutor
```

**Key design principle**: MockClient mirrors `schwab.client.Client` so that switching from simulation to live trading only requires changing the import.

## Code Conventions

- **Python 3.10+** - use modern syntax (`X | Y` unions, not `Union[X, Y]`)
- **Formatting**: ruff format (line-length 88, same as black)
- **Imports**: sorted by ruff (isort profile=black)
- **Type hints**: gradually being added; `mypy` runs in CI with `check_untyped_defs = true`
- **Tests**: pytest, one `test_*.py` per module, fixtures in `conftest.py`
- **Greek letters in docstrings**: intentionally used in physics formulas (σ, λ, η, γ, ×) - ruff is configured to allow them

## Testing Patterns

- Fixtures `sample_data`, `client`, `account_hash`, `fast_client` are in `tests/conftest.py`
- `sample_data` generates 100 bars of synthetic OHLCV data starting at $100
- `client` creates a MockClient with $10,000 initial cash and realistic physics
- Tests use `env.step()` which returns `(obs, reward, terminated, truncated, info)` - unused vars are prefixed with `_` or left unprefixed (ruff allows this in tests)
- Market orders: use `MockEquities.equity_buy_market(symbol, qty)`
- Limit orders: use `MockEquities.equity_buy_limit(symbol, qty, price)`

## Common Workflows

**Adding a new feature to MockClient**:
1. Add the core logic to the appropriate component (prices.py, account.py, order_manager.py)
2. Expose through client.py if it's part of the schwab-py API surface
3. Add tests in the corresponding `test_*.py`
4. Run `ruff check --fix . && ruff format .` then `mypy schwabgym`

**Adding a new physics engine**:
1. Create `schwabgym/physics/your_engine.py` extending `ExecutionEngine` from `base.py`
2. Implement `calculate_execution_price()` and `should_limit_fill()`
3. Export from `schwabgym/physics/__init__.py`
4. Add tests in `tests/test_physics_your_engine.py`

## Claude Code

Hooks in `.claude/settings.json` auto-format every file after Claude edits it (ruff check --fix + ruff format). No manual formatting needed.
