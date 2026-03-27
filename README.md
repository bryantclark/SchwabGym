# SchwabGym

A simulated trading environment compatible with the Charles Schwab Trader API (via `schwab-py`).

[![CI](https://github.com/bryantclark/SchwabGym/actions/workflows/ci.yml/badge.svg)](https://github.com/bryantclark/SchwabGym/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

SchwabGym mimics the interface of the `schwab-py` library to allow algorithmic trading agents to be developed and tested against historical data before deployment. It implements key market mechanics and regulatory rules found in the real trading environment, such as execution delays, partial fills, margin requirements, and Pattern Day Trader (PDT) restrictions.

By adhering to the `schwab-py` API signature, switching from simulation to live trading involves changing import statements rather than rewriting strategy logic.

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

SchwabGym provides mock implementations of key `schwab-py` classes.

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
- `get_account(account_hash, fields=None)`
- `get_quotes(symbols)`
- `get_price_history(symbol, ...)`
- `place_order(account_hash, order)`
- `cancel_order(account_hash, order_id)`
- `replace_order(account_hash, order_id, order_spec)`
- `get_order(account_hash, order_id)`
- `get_orders_for_account(account_hash, ...)`

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
account_hash = client.get_account_numbers().json()['hashValue']

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

## Testing

Run the test suite with pytest:

```bash
pytest tests/
```

## Disclaimer

This project is an independent simulation tool and is **not affiliated with Charles Schwab & Co., Inc.** It is intended for educational and testing purposes only. Simulation results do not guarantee future performance in live markets.
