"""
SchwabGym: High-Fidelity RL Environment for Algorithmic Trading
================================================================

A production-grade trading simulator designed for training deep reinforcement
learning agents that deploy to live markets via the Charles Schwab API.

Author: Bryant Clark
Repository: https://github.com/bryantclark/SchwabGym
License: MIT

Key Features:
    - Perfect API parity with schwab-py
    - Institutional-grade market physics (Square Root Law)
    - Pattern Day Trading enforcement
    - Regulatory fee calculations (SEC, FINRA)
    - Gymnasium-compatible RL environment
    - Zero-code deployment to live trading

Example Usage:
    >>> from schwabgym import MockClient, load_and_clean_data
    >>> from schwabgym.orders import MockEquities as eq
    >>>
    >>> df = load_and_clean_data('AAPL_5min.csv')
    >>> client = MockClient(df, initial_cash=25000)
    >>>
    >>> # Trading exactly like schwab-py
    >>> account_hash = client.account_linked().json()['hashValue']
    >>> quote = client.quote('AAPL')
    >>> order = eq.equity_buy_market('AAPL', 100)
    >>> client.place_order(account_hash, order)

For RL Training:
    >>> from schwabgym import SchwabTradingEnv
    >>> from stable_baselines3 import PPO
    >>>
    >>> env = SchwabTradingEnv(df, ticker='AAPL')
    >>> model = PPO('MlpPolicy', env, device='cuda')
    >>> model.learn(total_timesteps=10_000_000)
"""

__version__ = "1.0.0"
__author__ = "Bryant Clark"
__license__ = "MIT"
__repository__ = "https://github.com/bryantclark/SchwabGym"

# Order builders (compatibility layer)
from schwabgym import orders

# Core simulator
from schwabgym.client import MockClient

# Data utilities
from schwabgym.data import (
    add_technical_indicators,
    generate_dummy_data,
    load_and_clean_data,
    resample_data,
    split_train_test,
)

# Gymnasium environment
from schwabgym.environment import SchwabTradingEnv, ZScoreNormalizer

# Fee calculator
from schwabgym.fees import FeeCalculator

# Physics engines
from schwabgym.physics import (
    AlmgrenChrissOptimalExecutor,
    FastExecutionEngine,
    HybridExecutionEngine,
    RealisticExecutionEngine,
    create_execution_engine,
)

__all__ = [
    # Core
    "MockClient",
    # Data
    "load_and_clean_data",
    "generate_dummy_data",
    "resample_data",
    "add_technical_indicators",
    "split_train_test",
    # Environment
    "SchwabTradingEnv",
    "ZScoreNormalizer",
    # Physics
    "FastExecutionEngine",
    "RealisticExecutionEngine",
    "HybridExecutionEngine",
    "AlmgrenChrissOptimalExecutor",
    "create_execution_engine",
    # Fees
    "FeeCalculator",
    # Orders
    "orders",
]


def get_version():
    """Get the current version of SchwabGym."""
    return __version__


def get_info():
    """Get package information."""
    return {
        "version": __version__,
        "author": __author__,
        "license": __license__,
        "repository": __repository__,
        "description": "High-fidelity RL environment for algorithmic trading",
    }


# ASCII banner for CLI
BANNER = r"""
  ____       _                 _     ____
 / ___|  ___| |____      ____ _| |__ / ___|_   _ _ __ ___
 \___ \ / __| '_ \ \ /\ / / _` | '_ \| |  _| | | | '_ ` _ \
  ___) | (__| | | \ V  V / (_| | |_) | |_| | |_| | | | | | |
 |____/ \___|_| |_|\_/\_/ \__,_|_.__/ \____|\__, |_| |_| |_|
                                            |___/
        High-Fidelity RL Environment for Algorithmic Trading
        Version: {version} | Author: {author}
        {repository}
""".format(
    version=__version__, author=__author__, repository=__repository__
)


def print_banner():
    """Print the SchwabGym banner."""
    print(BANNER)


def check_dependencies():
    """Check if all required dependencies are installed."""
    required = {
        "pandas": "pandas",
        "numpy": "numpy",
        "gymnasium": "gymnasium",
        "matplotlib": "matplotlib",
    }

    optional = {
        "schwab": "schwab-py (for live trading)",
        "stable_baselines3": "stable-baselines3 (for RL)",
        "torch": "torch (for RL)",
    }

    missing_required = []
    missing_optional = []

    for module, name in required.items():
        try:
            __import__(module)
        except ImportError:
            missing_required.append(name)

    for module, name in optional.items():
        try:
            __import__(module)
        except ImportError:
            missing_optional.append(name)

    if missing_required:
        raise ImportError(
            f"Missing required dependencies: {', '.join(missing_required)}\n"
            f"Install with: pip install {' '.join(missing_required)}"
        )

    if missing_optional:
        import warnings

        warnings.warn(
            f"Optional dependencies not installed: {', '.join(missing_optional)}\n"
            f"Some features may not be available."
        )

    return True


# Run dependency check on import
try:
    check_dependencies()
except ImportError as e:
    import warnings

    warnings.warn(str(e))
