"""
Pytest Configuration and Shared Fixtures
=========================================

Shared test fixtures for SchwabGym test suite.

Author: Bryant Clark
Repository: https://github.com/bryantclark/SchwabGym
"""

import os
import sys

# Ensure tests import from this worktree, not the editable install of the main repo
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_data():
    """
    Generate sample OHLCV data for testing.

    Returns 100 bars of realistic synthetic data.
    """
    from schwabgym import generate_dummy_data

    return generate_dummy_data("TEST", periods=100, start_price=100.0)


@pytest.fixture
def client(sample_data):
    """
    Create a MockClient instance with sample data.

    Uses realistic physics engine (default).
    Legacy tests assume 0 latency, so we force latency_mode=False.
    """
    from schwabgym import MockClient

    return MockClient(sample_data, initial_cash=10000.0, latency_mode=False)


@pytest.fixture
def account_hash(client):
    """Get account hash from client."""
    return client.get_account_numbers().json()[0]["hashValue"]


@pytest.fixture
def fast_client(sample_data):
    """
    Create a MockClient with fast physics engine.

    Useful for speed-sensitive tests.
    """
    from schwabgym import MockClient
    from schwabgym.physics import FastExecutionEngine

    engine = FastExecutionEngine()
    # Also disable latency for fast client tests
    return MockClient(
        sample_data, initial_cash=10000.0, execution_engine=engine, latency_mode=False
    )


@pytest.fixture
def multi_asset_data():
    """Create deterministic multi-asset data for market-data API tests."""
    from schwabgym import generate_dummy_data

    return {
        "TEST": generate_dummy_data("TEST", periods=40, start_price=100.0),
        "ALT": generate_dummy_data("ALT", periods=40, start_price=50.0),
    }


@pytest.fixture
def multi_asset_client(multi_asset_data):
    """Create a multi-asset client for movers/instrument tests."""
    from schwabgym import MockClient

    return MockClient(multi_asset_data, initial_cash=10000.0, latency_mode=False)


@pytest.fixture
def alpha_vantage_csv(tmp_path):
    """
    Create a temporary CSV in Alpha Vantage format.

    Useful for testing data loader.
    """
    dates = pd.date_range(start="2024-01-01", periods=50, freq="5min")
    data = {
        "timestamp": dates,
        "open": np.random.uniform(99, 101, 50),
        "high": np.random.uniform(100, 102, 50),
        "low": np.random.uniform(98, 100, 50),
        "close": np.random.uniform(99, 101, 50),
        "volume": np.random.randint(100000, 200000, 50),
    }

    df = pd.DataFrame(data)
    csv_path = tmp_path / "test_data.csv"
    df.to_csv(csv_path, index=False)

    return str(csv_path)


@pytest.fixture
def fee_calculator():
    """Create FeeCalculator instance."""
    from schwabgym.fees import FeeCalculator

    return FeeCalculator()
