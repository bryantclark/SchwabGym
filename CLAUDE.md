# SchwabGym

Trading simulator with API parity to `schwab-py`. Switching from sim to live = changing the import.

## Commands

```bash
make test                       # run full test suite
make lint                       # ruff check + format check
make typecheck                  # mypy schwabgym
make check                      # all of the above
make fix                        # auto-fix lint + format

pytest tests/test_client.py     # single file
pytest -k "test_market_order"   # by name pattern
pytest --cov                    # with coverage report
```

## Structure

```
schwabgym/
  client.py          # MockClient — main entry, delegates to components
  prices.py          # PriceEngine — market data replay, time advancement
  account.py         # Account — positions, cash, PDT, margin
  order_manager.py   # OrderManager — order lifecycle (place/cancel/replace)
  orders.py          # MockEquities/MockResponse — order builders, API response
  environment.py     # SchwabTradingEnv — Gymnasium wrapper (unopinionated)
  fees.py            # FeeCalculator — SEC Section 31, FINRA TAF
  data.py            # Data loading, cleaning, technical indicators
  streamer.py        # MockStreamer — streaming data simulation
  physics/           # Execution engines
    base.py          #   ExecutionEngine ABC
    fast.py          #   FastExecutionEngine — instant fills
    realistic.py     #   RealisticExecutionEngine — Square Root Law (default)
    hybrid.py        #   HybridExecutionEngine — probabilistic switching
    almgren_chriss.py#   AlmgrenChrissOptimalExecutor
```

## Conventions

- Python 3.10+ — use `X | Y` unions, not `Union[X, Y]`
- ruff for lint + format (line-length 88)
- Greek letters in physics docstrings are intentional (σ, λ, η) — ruff ignores them
- One `test_*.py` per source module, fixtures in `tests/conftest.py`

## Gotchas

- **`latency_mode`**: defaults to `True` (orders activate after 1 step). Tests use `latency_mode=False` for immediate fills. If a test unexpectedly fails to fill an order, check this.
- **SchwabTradingEnv is unopinionated**: it takes `observation_fn`, `reward_fn`, `action_fn` callbacks. It does NOT compute indicators or rewards internally.
- **Order builders**: use `MockEquities.equity_buy_market(symbol, qty)` not raw dicts.
- **Physics engines**: default is `RealisticExecutionEngine` (Square Root Law). Use `FastExecutionEngine` for speed in tests.
- **Worktrees**: `pip install -e .` points to the main repo. `conftest.py` has a `sys.path` insert so tests use the local worktree code.

## Hooks

`.claude/settings.json` auto-formats on every Edit/Write (ruff check --fix + ruff format). No manual formatting needed.
