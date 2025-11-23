# SchwabGym

**A high-fidelity reinforcement learning environment for training algorithmic trading agents on the Charles Schwab platform.**

[![CI](https://github.com/bryantclark/SchwabGym/actions/workflows/ci.yml/badge.svg)](https://github.com/bryantclark/SchwabGym/actions/workflows/ci.yml)
[![TestPyPI](https://github.com/bryantclark/SchwabGym/actions/workflows/release.yml/badge.svg)](https://github.com/bryantclark/SchwabGym/actions/workflows/release.yml)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI](https://badge.fury.io/py/schwabgym.svg)](https://badge.fury.io/py/schwabgym)

Developed by [Bryant Clark](https://github.com/bryantclark)

---

## Publishing & Secrets

This repository includes a two-stage release workflow that first publishes to TestPyPI and then — after manual approval in the `production` environment — publishes to PyPI.

1) Create API tokens
    - TestPyPI: create an API token at https://test.pypi.org/manage/account/#api-tokens and copy the token.
    - PyPI: create an API token at https://pypi.org/manage/account/#api-tokens and copy the token.

2) Add repository secrets
    - In GitHub: `Settings` → `Secrets and variables` → `Actions` → `New repository secret`.
    - Add `TEST_PYPI_API_TOKEN` (token from TestPyPI).
    - Add `PYPI_API_TOKEN` (token from PyPI).

3) Configure environment protection for production
    - In GitHub: `Settings` → `Environments` → `New environment` and create `production`.
    - Add required reviewers so the `publish-prod` job requires manual approval before running.

4) Trigger a release (local)

```bash
# create a version tag locally
git tag v0.1.0
# push the tag to GitHub (this triggers the release workflow)
git push origin v0.1.0
```

Notes
- The workflow publishes to TestPyPI using `TEST_PYPI_API_TOKEN` and creates a test GitHub Release.
- After test publish completes, the `publish-prod` job will wait for approval in the `production` environment and then publish to PyPI using `PYPI_API_TOKEN`.
- Do not store tokens in source — keep them only in GitHub Secrets.

## Overview

SchwabGym is a trading simulator designed for training deep reinforcement learning agents that deploy to live markets via the Charles Schwab API. SchwabGym provides **high API fidelity** with `schwab-py`, ensuring that your trained agents require **minimal code changes** when transitioning from simulation to live trading.

## Project Structure

The repository layout (top-level) is shown below. This representation is intended to help contributors quickly find modules, tests, and examples. Some subfolders contain additional files (e.g. `tensorboard` event logs, cached data) and are elided for brevity.

```
examples/
    basic_trading.py
    hybrid_training_demo.py
    rl_trading.py
schwabgym/
    __init__.py
    client.py
    data.py
    environment.py
    fees.py
    orders.py
    physics/
        __init__.py
        almgren_chriss.py
        base.py
        fast.py
        hybrid.py
        PHYSICS_ENGINE.md
        realistic.py
tests/
    conftest.py
    test_client.py
    test_data.py
    test_environment.py
    test_fees.py
    test_orders.py
    test_physics_almgren_chriss.py
    test_physics_fast.py
    test_physics_hybrid.py
    test_physics_init.py
    test_physics_realistic.py
    test_physics.py
.gitignore
LICENSE
LIVE_TRADING.md
MY_INTERNAL_NOTES.md
README.md
requirements.txt
setup.py
```

### Why SchwabGym?

**The Problem**: Most trading simulators fail when agents are deployed to production because they don't model realistic market microstructure, execution friction, or API-specific constraints. Agents trained on idealized fills and zero slippage often underperform in real-world markets.

**The Solution**: SchwabGym implements realistic market physics including the Square Root Law of market impact, volume-constrained fills, and regulatory constraints (PDT rules, SEC/FINRA fees, margin requirements). It replicates the JSON schemas and authentication flows of the Schwab Trader API.

**The Result**: Agents trained in SchwabGym can be deployed to live trading with greater confidence, having already experienced realistic execution conditions during training.

---

## 🎯 Key Features

### Realistic Market Physics
- **Square Root Law Market Impact**: Implements the `ΔP = Y×σ×sqrt(Q/V)` model to simulate slippage
- **Volume-Constrained Fills**: Limit orders fill probabilistically based on available liquidity
- **Brownian Bridge Simulation**: Generates realistic intraday price paths for path-dependent orders
- **Bid-Ask Spread**: Simulates spread crossing costs
- **Regulatory Friction**: Estimates SEC Section 31 fees, FINRA TAF, and exchange fees

### Schwab API Fidelity
- **Schema Matching**: JSON responses mirror the production API structure
- **OAuth Flow Simulation**: Token lifecycle management logic
- **Account Hash System**: Enforces Schwab's encrypted account identifier workflow
- **Pattern Day Trading**: Full PDT rule enforcement (4 day trades in 5 days = account restriction)
- **Margin Requirements**: Regulation T initial margin (50%) and maintenance margin (30%)

### Deep Learning Optimized
- **Gymnasium Compatible**: Drop-in replacement for OpenAI Gym with Stable Baselines3 support
- **Vectorization Ready**: Supports parallel environments
- **Domain Randomization**: Hybrid physics modes for robust Sim-to-Real transfer

### Minimal Code Deployment
```python
# Training
from schwabgym import MockClient
client = MockClient(df)

# Live Trading (minimal changes)
from schwab.client import Client  # Real Schwab API
client = auth.easy_client(...)     # Everything else identical
```

---

## Installation

```bash
pip install schwabgym
```

**Dependencies**:
```bash
pip install pandas numpy gymnasium matplotlib schwab-py
```

**Optional (for RL training)**:
```bash
pip install stable-baselines3 torch
```

---

## Quick Start

### Basic Simulation

```python
from schwabgym import MockClient, load_and_clean_data
from schwabgym.orders import MockEquities as eq

# Load historical data (Alpha Vantage, Yahoo Finance, etc.)
df = load_and_clean_data('AAPL_5min.csv')

# Initialize simulator (defaults to Realistic physics)
client = MockClient(df, initial_cash=25000)
account_hash = client.account_linked().json()['hashValue']

# Trading loop
while client.advance_time():
    # Get quote (exactly like schwab-py)
    quote = client.quote('AAPL')
    price = quote.json()['AAPL']['quote']['lastPrice']
    
    # Get account state
    acct = client.account_details(account_hash).json()
    cash = acct['securitiesAccount']['currentBalances']['cashBalance']
    
    # Place order
    if cash > price * 100:
        order = eq.equity_buy_market('AAPL', 100)
        client.place_order(account_hash, order)
```

### Reinforcement Learning Training

```python
from schwabgym import SchwabTradingEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv

# Create vectorized environment (4 parallel envs)
def make_env():
    df = load_and_clean_data('AAPL_5min.csv')
    return SchwabTradingEnv(df, ticker='AAPL', initial_cash=25000)

env = SubprocVecEnv([make_env for _ in range(4)])

# Train PPO agent
model = PPO('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=1_000_000)

# Evaluate
obs, _ = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)

# Visualize
env.envs[0].render()
```

---

## Architecture

### Dual-Price System

SchwabGym maintains two price states to mirror real-world data providers:

1. **Raw Prices** (`Close` column): Used for order execution, margin calculations, and P&L
   - Represents actual traded prices on "the tape"
   - Used by the matching engine for fills
   - What you see in real-time quotes

2. **Adjusted Prices** (`AdjClose` column): Used for technical analysis and indicators
   - Accounts for stock splits and dividends
   - Used in `price_history()` endpoint
   - What you use for backtesting and strategy development

This dual-state design prevents "look-ahead bias" where adjusted prices from the future contaminate historical trading decisions.

### Physics Engines

SchwabGym supports two execution modes:

#### Realistic Mode (Default)

**Recommended for production training.** Implements realistic market microstructure:

```python
from schwabgym.physics import RealisticExecutionEngine

engine = RealisticExecutionEngine(
    impact_coefficient=0.7,      # Y in Square Root Law
    participation_rate=0.10,     # Max 10% of bar volume
    queue_depth_factor=2.0       # Estimated orders ahead of us
)

client = MockClient(df, execution_engine=engine)
```

**Features**:
- Square Root Law: `Impact = Y × σ × sqrt(Q/V)`
- Volume-based fill probabilities for limit orders
- Brownian Bridge intraday path generation
- Pessimistic execution (agent sees worst-case fills)

**Use When**: Training production-bound agents, final validation

#### Fast Mode (Testing/Prototyping)

**Simplified physics for rapid iteration:**

```python
from schwabgym.physics import FastExecutionEngine

engine = FastExecutionEngine(base_slippage=0.02)
client = MockClient(df, execution_engine=engine)
```

**Features**:
- Fixed slippage model
- Binary limit order fills (price touched = filled)
- Minimal computational overhead

**Use When**: Debugging strategy logic, quick prototypes

---

## Data Integration

### Alpha Vantage (Recommended)

```python
import requests
import pandas as pd
from schwabgym import load_and_clean_data

# Download from Alpha Vantage
API_KEY = 'your_key'
url = f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=AAPL&interval=5min&apikey={API_KEY}&outputsize=full&datatype=csv'

response = requests.get(url)
with open('AAPL_5min.csv', 'wb') as f:
    f.write(response.content)

# Load and clean
df = load_and_clean_data('AAPL_5min.csv')
```

### Supported Formats

- **Alpha Vantage**: Daily, Intraday, Adjusted
- **Yahoo Finance**: Historical data
- **Custom CSVs**: Any OHLCV format with timestamps

The `load_and_clean_data()` function automatically:
- Detects and normalizes column names
- Reconstructs raw prices from adjusted data
- Calculates volatility proxy for impact model
- Validates data integrity

---

## Advanced Features

### Pattern Day Trading Enforcement

SchwabGym rigorously enforces PDT rules:

```python
# Scenario: Account has $20,000 equity
# Agent makes 4 day trades in 5 days

# On the 5th day trade attempt:
response = client.place_order(account_hash, order)
# Returns: 403 Forbidden - "Pattern Day Trader Restriction"

# Account is now flagged and restricted until:
# 1. Equity rises above $25,000, OR
# 2. 5 days pass without day trades
```

This forces agents to learn position sizing and holding periods appropriate for their capital level.

### Regulatory Fees

```python
# Built-in fee calculator accounts for SEC and FINRA fees
fee_calculator.calculate_sec_fee(
    transaction_date=datetime.date(2024, 12, 1),
    notional_value=100000
)
```

Ensures agents trained on historical data learn the correct fee structure.

### Almgren-Chriss Optimal Execution

For large orders, use the optimal execution framework:

```python
from schwabgym.execution import AlmgrenChrissOptimalExecutor

executor = AlmgrenChrissOptimalExecutor(
    lambda_risk=0.01,  # Risk aversion parameter
    eta_temp=0.1,      # Temporary impact coefficient
    gamma_perm=0.05    # Permanent impact coefficient
)

# Compute schedule for 50,000 shares over 1 day
trajectory = executor.compute_trajectory(
    total_shares=50000,
    T=1.0,      # Time horizon (days)
    N=10,       # Number of child orders
    volatility=0.02  # Daily volatility
)

# Returns: [8431, 6890, 5893, 5198, 4694, ...]
# Front-loaded schedule minimizes timing risk
```

This demonstrates how to split "parent orders" into "child orders" that balance market impact vs. timing risk.

---

## Gymnasium Environment

### Observation Space

The `SchwabTradingEnv` provides an 8-dimensional observation vector:

| Index | Feature | Description | Range |
|-------|---------|-------------|-------|
| 0 | RSI | Relative Strength Index | [0, 100] |
| 1 | Price/SMA | Price relative to 20-period SMA | [0.8, 1.2] |
| 2 | Price/BB Upper | Price relative to upper Bollinger Band | [0.9, 1.1] |
| 3 | MACD | Moving Average Convergence Divergence | [-5, 5] |
| 4 | Position | Current position size / 1000 | [-10, 10] |
| 5 | Profit Factor | Price / Average cost basis | [0.5, 1.5] |
| 6 | Time (sin) | Cyclical time encoding | [-1, 1] |
| 7 | Time (cos) | Cyclical time encoding | [-1, 1] |

All features are Z-score normalized with online statistics tracking.

### Action Space

Continuous 2D action vector:

- **Action[0]**: Signal strength [-1, +1]
  - < -0.33: Sell/Short
  - [-0.33, 0.33]: Hold
  - > 0.33: Buy/Cover

- **Action[1]**: Position size [0, 1]
  - Fraction of available buying power to use

### Reward Function

```python
reward = log(new_equity / old_equity)
```

Log returns ensure reward scale consistency across different account sizes and price levels.

**Terminal conditions**:
- Account value < 50% of initial capital (margin call)
- End of data reached

---

## Example Strategies

### Mean Reversion

```python
from schwabgym import MockClient, load_and_clean_data
from schwabgym.orders import MockEquities as eq

df = load_and_clean_data('SPY_1min.csv')
client = MockClient(df, initial_cash=100000)
account_hash = client.account_linked().json()['hashValue']

while client.advance_time():
    # Get recent candles
    hist = client.price_history('SPY').json()['candles']
    prices = [c['close'] for c in hist[-20:]]
    
    current_price = prices[-1]
    sma_20 = sum(prices) / 20
    
    # Get account state
    acct = client.account_details(account_hash).json()
    cash = acct['securitiesAccount']['currentBalances']['cashBalance']
    
    # Mean reversion logic
    if current_price < sma_20 * 0.98 and cash > 5000:
        qty = int(5000 / current_price)
        order = eq.equity_buy_market('SPY', qty)
        client.place_order(account_hash, order)
    
    elif current_price > sma_20 * 1.02:
        # Find position
        for pos in acct['securitiesAccount']['positions']:
            if pos['instrument']['symbol'] == 'SPY':
                qty = pos['longQuantity']
                order = eq.equity_sell_market('SPY', int(qty))
                client.place_order(account_hash, order)
                break

# Print results
final = client.account_details(account_hash).json()
print(f"Final Equity: ${final['securitiesAccount']['currentBalances']['liquidationValue']:,.2f}")
```

### Pairs Trading

```python
from schwabgym import MockClient, load_and_clean_data
from schwabgym.orders import MockEquities as eq
import numpy as np

# Load data for two correlated assets
df_spy = load_and_clean_data('SPY_5min.csv')
df_qqq = load_and_clean_data('QQQ_5min.csv')

# Merge on timestamp
df = df_spy.merge(df_qqq, left_index=True, right_index=True, suffixes=('_SPY', '_QQQ'))

client = MockClient(df, initial_cash=50000)
account_hash = client.account_linked().json()['hashValue']

# Calculate spread
df['spread'] = df['Close_SPY'] / df['Close_QQQ']
df['spread_zscore'] = (df['spread'] - df['spread'].rolling(60).mean()) / df['spread'].rolling(60).std()

while client.advance_time():
    row = df.iloc[client.current_step]
    zscore = row['spread_zscore']
    
    if not np.isnan(zscore):
        if zscore > 2.0:  # SPY expensive relative to QQQ
            # Short SPY, Long QQQ
            order_spy = eq.equity_sell_short_market('SPY', 100)
            order_qqq = eq.equity_buy_market('QQQ', 100)
            client.place_order(account_hash, order_spy)
            client.place_order(account_hash, order_qqq)
        
        elif zscore < -2.0:  # SPY cheap relative to QQQ
            # Long SPY, Short QQQ
            order_spy = eq.equity_buy_market('SPY', 100)
            order_qqq = eq.equity_sell_short_market('QQQ', 100)
            client.place_order(account_hash, order_spy)
            client.place_order(account_hash, order_qqq)
```

---

## Deployment to Live Trading

When your agent is ready for production:

### Step 1: Setup Schwab Developer Account

1. Create account at [developer.schwab.com](https://developer.schwab.com)
2. Create an app and get API keys
3. Set callback URL to `https://127.0.0.1:8182/`
4. Wait for app approval (can take 2-5 days)

### Step 2: Switch Client

```python
# BEFORE (Simulation)
from schwabgym import MockClient
client = MockClient(df)

# AFTER (Live)
from schwab import auth
client = auth.easy_client(
    api_key='YOUR_API_KEY',
    app_secret='YOUR_APP_SECRET',
    callback_url='https://127.0.0.1:8182/',
    token_path='./token.json'
)
```

### Step 3: Verify

**Everything else remains identical:**
- `client.quote('AAPL')` → Same JSON structure
- `client.place_order(hash, order)` → Same method signature
- `client.account_details(hash)` → Same response format

**Minimal code changes needed!**

---

## Testing

```bash
# Run test suite
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=schwabgym --cov-report=html

# Run specific test
pytest tests/test_mock_client.py::TestPatternDayTrading -v
```

Test coverage includes:
- Order execution accuracy
- Fee calculations
- PDT rule enforcement
- Market impact modeling
- Data loader edge cases
- API response format validation

---

## Project Structure

```
schwabgym/
├── README.md                      # This file
├── LICENSE                        # MIT License
├── setup.py                       # Package configuration
├── requirements.txt               # Dependencies
├── schwabgym/
│   ├── __init__.py               # Package exports
│   ├── client.py                 # MockClient (core simulator)
│   ├── orders.py                 # Order builders (Schwab API compatibility)
│   ├── data.py                   # Data loading and preprocessing
│   ├── environment.py            # Gymnasium trading environment
│   ├── physics/
│   │   ├── __init__.py
│   │   ├── fast.py               # Fast execution engine
│   │   ├── realistic.py          # Realistic execution engine
│   │   └── almgren_chriss.py     # Optimal execution
│   ├── fees.py                   # Regulatory fee calculator
│   └── utils.py                  # Utilities
├── examples/
│   ├── basic_trading.py          # Simple mean reversion
│   ├── pairs_trading.py          # Pairs trading strategy
│   ├── rl_training.py            # PPO agent training
│   └── live_deployment.py        # Live trading template
├── tests/
│   ├── test_client.py            # Client tests
│   ├── test_physics.py           # Physics engine tests
│   ├── test_environment.py       # Environment tests
│   └── test_fees.py              # Fee calculation tests
└── docs/
    ├── PHYSICS.md                # Physics engine details
    ├── API_REFERENCE.md          # Complete API docs
    └── DEPLOYMENT.md             # Live trading guide
```

---

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Run tests (`pytest tests/ -v`)
4. Commit changes (`git commit -m 'Add amazing feature'`)
5. Push to branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

---

## Citation

If you use SchwabGym in your research, please cite:

```bibtex
@software{schwabgym2024,
  author = {Clark, Bryant},
  title = {SchwabGym: High-Fidelity RL Environment for Algorithmic Trading},
  year = {2024},
  url = {https://github.com/bryantclark/SchwabGym}
}
```

---

## Acknowledgments

- **schwab-py** ([alexgolec/schwab-py](https://github.com/alexgolec/schwab-py)): Excellent API wrapper that inspired this project
- **Gymnasium** ([Farama-Foundation/Gymnasium](https://github.com/Farama-Foundation/Gymnasium)): Modern RL environment standard
- **Almgren & Chriss (2000)**: "Optimal Execution of Portfolio Transactions" - foundation for execution modeling
- **Alpha Vantage**: Free market data API for development

---

## License

MIT License - see [LICENSE](LICENSE) file for details.

---

## Disclaimer

**SchwabGym is an unofficial simulator and is not affiliated with Charles Schwab & Co., Inc.**

- Use at your own risk
- No guarantees of accuracy or profitability
- Paper trading and backtesting results do not guarantee live trading success
- Always test thoroughly before deploying real capital
- Consult with a financial advisor before trading

**Trading involves substantial risk of loss. Only trade with capital you can afford to lose.**

---

## Support

- **Documentation**: [schwabgym.readthedocs.io](https://schwabgym.readthedocs.io) (coming soon)
- **Issues**: [GitHub Issues](https://github.com/bryantclark/SchwabGym/issues)
- **Discussions**: [GitHub Discussions](https://github.com/bryantclark/SchwabGym/discussions)

---

**Built with ❤️ for algorithmic traders by [Bryant Clark](https://github.com/bryantclark)**
