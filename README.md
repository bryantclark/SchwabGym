# SchwabGym

A simulated trading environment compatible with the Charles Schwab Trader API (via `schwab-py`).

[![CI](https://github.com/bryantclark/SchwabGym/actions/workflows/ci.yml/badge.svg)](https://github.com/bryantclark/SchwabGym/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

SchwabGym aims to mirror the interface of the `schwab-py` library so that algorithmic trading agents can be developed and tested against historical data before deployment. It implements key market mechanics and regulatory rules found in the real trading environment, such as execution delays, partial fills, margin requirements, and Pattern Day Trader (PDT) restrictions.

**Goal:** switching from simulation to live trading should only require changing import statements. Full API parity is not yet achieved — see [Supported Methods](#supported-methods) for current coverage and [Limitations](#limitations) for known gaps.

## Project Structure

```
schwabgym/
├── client.py            # MockClient — main entry, delegates to components
├── orders.py            # MockEquities/MockResponse — order builders, API response
├── fees.py              # FeeCalculator — SEC Section 31, FINRA TAF
├── account.py           # Account — positions, cash, PDT, margin
├── order_manager.py     # OrderManager — order lifecycle (place/cancel/replace)
├── prices.py            # PriceEngine — market data replay, time advancement
├── data.py              # Data loading, cleaning, technical indicators
├── streamer.py          # MockStreamer — streaming data simulation
├── environment.py       # SchwabTradingEnv — Gymnasium wrapper (unopinionated)
└── physics/             # Execution engines
    ├── base.py          #   ExecutionEngine ABC
    ├── fast.py          #   FastExecutionEngine — instant fills
    ├── realistic.py     #   RealisticExecutionEngine — Square Root Law (default)
    ├── hybrid.py        #   HybridExecutionEngine — probabilistic switching
    └── almgren_chriss.py#   AlmgrenChrissOptimalExecutor
```

## Comparison with `schwab-py`

SchwabGym provides mock implementations of the most commonly used `schwab-py` classes. Coverage is growing but not yet complete.

| Feature | `schwab-py` (Live) | `SchwabGym` (Simulation) |
|---------|--------------------|--------------------------|
| **Client** | `schwab.client.Client` | `schwabgym.MockClient` |
| **Orders** | `schwab.orders.equities` | `schwabgym.orders.MockEquities` |
| **Quotes** | Real-time market data | Historical data replay (Pandas DataFrame) |
| **Execution** | Exchange matching engine | Local physics engine (`schwabgym.physics`) |
| **Fees** | Actual broker/regulatory fees | Estimated regulatory fees (SEC/FINRA) |

### Supported Methods

The `MockClient` supports the following methods with signatures matching `schwab-py`:

- `get_account_numbers()`
- `get_account(account_hash, *, fields=None)`
- `get_accounts(*, fields=None)`
- `get_quote(symbol, *, fields=None)`
- `get_quotes(symbols, *, fields=None, indicative=None)`
- `get_price_history(symbol, *, period_type, period, frequency_type, frequency, start_datetime, end_datetime, need_extended_hours_data, need_previous_close)`
- `get_price_history_every_minute(symbol, ...)` (+ every_five/ten/fifteen/thirty_minutes, every_day, every_week)
- `place_order(account_hash, order_spec)`
- `preview_order(account_hash, order_spec)`
- `cancel_order(order_id, account_hash)`
- `replace_order(account_hash, order_id, order_spec)`
- `get_order(order_id, account_hash)`
- `get_orders_for_account(account_hash, *, max_results, from_entered_datetime, to_entered_datetime, status)`
- `get_orders_for_all_linked_accounts(*, ...)`
- `get_transaction(account_hash, transaction_id)`
- `get_transactions(account_hash, *, start_date, end_date, transaction_types, symbol)`
- `get_user_preferences()`

## Regulatory and Market Simulation

The simulator attempts to replicate specific constraints of the US equity market:

1.  **Regulatory Fees**:
    *   **SEC Section 31**: Calculated based on transaction value.
    *   **FINRA TAF**: Calculated per share/contract with applicable caps.

2.  **Pattern Day Trading (PDT)**:
    *   Enforces the rule restricting accounts with less than $25,000 equity from executing more than 3 day trades in a rolling 5-business-day period.

3.  **Margin Requirements**:
    *   Simulates Regulation T initial margin (50%) and maintenance margin requirements.

4.  **Market Physics**:
    *   **Slippage**: Can use the Square Root Law (`ΔP = Y×σ×sqrt(Q/V)`) to estimate market impact based on order size and historical volatility.
    *   **Limit Orders**: Fills are determined by checking if the historical price range (High/Low) crossed the limit price.

## Installation

```bash
pip install schwabgym
```

Optional extras:

```bash
pip install schwabgym[rl]      # stable-baselines3 + torch
pip install schwabgym[live]    # schwab-py for live trading
pip install schwabgym[plot]    # matplotlib
pip install schwabgym[dev]     # pytest, ruff, mypy
```

## Usage Example

### Simulation

```python
from schwabgym import MockClient, load_and_clean_data
from schwabgym.orders import MockEquities as eq

# Load historical data
df = load_and_clean_data('AAPL_5min.csv')

# Initialize simulator
client = MockClient(df, initial_cash=25000)
account_hash = client.get_account_numbers().json()[0]['hashValue']

# Trading loop
while client.advance_time():
    # Get quote
    quote = client.get_quotes('AAPL')
    price = quote.json()['AAPL']['quote']['lastPrice']

    # Place order
    if some_condition:
        order = eq.equity_buy_market('AAPL', 100)
        client.place_order(account_hash, order)
```

### Transition to Live Trading

To deploy, replace the `schwabgym` imports with `schwab-py` imports.

```python
# from schwabgym import MockClient as Client
# from schwabgym.orders import MockEquities as eq
from schwab.client import Client
from schwab.orders import equities as eq
from schwab import auth

client = auth.easy_client(...)
# ... logic remains the same ...
```

## Physics Engines

Control execution realism with pluggable physics engines:

```python
from schwabgym import MockClient, FastExecutionEngine, RealisticExecutionEngine, HybridExecutionEngine

# Fast: instant fills, fixed slippage (best for prototyping)
client = MockClient(df, execution_engine=FastExecutionEngine())

# Realistic: Square Root Law market impact (default, best for validation)
client = MockClient(df, execution_engine=RealisticExecutionEngine(impact_coefficient=0.7))

# Hybrid: domain randomization (best for RL training)
client = MockClient(df, execution_engine=HybridExecutionEngine(realistic_probability=0.3))
```

| Mode | Speed | Market Impact | Best For |
|------|-------|---------------|----------|
| Fast | ~10k steps/s | Fixed slippage | Prototyping, hyperparameter search |
| Realistic | ~1k steps/s | Square Root Law | Final validation, execution-sensitive strategies |
| Hybrid | ~7k steps/s | Mixed | RL training with domain randomization |

See [`schwabgym/physics/PHYSICS_ENGINE.md`](schwabgym/physics/PHYSICS_ENGINE.md) for the full guide including Almgren-Chriss optimal execution.

## Gymnasium RL Environment

`SchwabTradingEnv` is an unopinionated Gymnasium wrapper. You inject your own observation, reward, and action logic:

```python
import numpy as np
from gymnasium import spaces
from schwabgym import MockClient, SchwabTradingEnv, load_and_clean_data

df = load_and_clean_data('AAPL_5min.csv')
client = MockClient(df, initial_cash=25000)

def my_obs_fn(client):
    """Return your observation vector."""
    hist = client.get_price_history('AAPL').json()['candles']
    prices = [c['close'] for c in hist[-20:]]
    return np.array(prices, dtype=np.float32)

def my_reward_fn(client):
    """Return scalar reward."""
    return 0.0  # your reward logic

def my_action_fn(client, action):
    """Translate agent actions to orders."""
    pass  # your order logic

env = SchwabTradingEnv(
    client=client,
    observation_fn=my_obs_fn,
    reward_fn=my_reward_fn,
    action_fn=my_action_fn,
    observation_space=spaces.Box(low=0, high=500, shape=(20,), dtype=np.float32),
    action_space=spaces.Discrete(3),
)

obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(0)
```

## Streaming

`MockStreamer` provides basic async streaming simulation for `MockClient`:

```python
from schwabgym import MockClient, MockStreamer

client = MockClient(df)
streamer = client.streamer  # MockStreamer instance

# Subscribe and stream (async)
async def on_message(msg):
    print(msg['service'], msg['content'])

await streamer.start(on_message)
```

## Testing

Run the test suite with pytest:

```bash
pytest tests/
```

## Limitations

SchwabGym is under active development. The goal is full drop-in parity with `schwab-py`, but the following gaps remain:

- **Synthetic market-data helpers:** `get_option_chain`, `get_option_expiration_chain`, `get_movers`, `get_market_hours`, `get_instruments`, and `get_instrument_by_cusip` now return deterministic simulator-generated payloads. They are useful for local testing, but they are not live Schwab reference data.
- **Order builders are only partially parity-complete.** SchwabGym order helpers now return fluent `MockOrderBuilder` instances, and `place_order()` / `preview_order()` also accept real `schwab-py` `OrderBuilder` objects. The common setter flow and vertical spread helpers are covered, but the full generic `OrderBuilder` surface is not implemented yet.
- **Streaming:** `MockStreamer` provides basic Level 1 equity streaming for simulator-driven tests, not full `schwab-py` `StreamClient` parity.
- **Enums:** `schwab-py` uses nested enums (`Client.Account.Fields`, `Client.Order.Status`, etc.). SchwabGym accepts raw strings instead.
- **Price history:** The simulator filters candles by the requested time window and resamples to coarser bars when possible, but it cannot synthesize finer bars than the frequency of the loaded dataset.
- **Data loading:** `load_and_clean_data()` now fails fast on missing files. Use `allow_dummy=True` or `generate_dummy_data()` when synthetic data is intentional.
- **Fee schedule:** Regulatory fees (SEC Section 31, FINRA TAF) are approximations based on published rate schedules. They may drift as rates change.

Contributions to close these gaps are welcome.

## Disclaimer

This project is an independent simulation tool and is **not affiliated with Charles Schwab & Co., Inc.** It is intended for educational and testing purposes only. Simulation results do not guarantee future performance in live markets.
