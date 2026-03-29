"""
Performance Benchmarks for SchwabGym Physics Engines
=====================================================

Run with: pytest tests/test_benchmarks.py --benchmark-enable
Skip with: pytest -m "not benchmark" (default, benchmarks are opt-in)

Requires: pip install pytest-benchmark
"""

import numpy as np
import pytest

pytest.importorskip("pytest_benchmark")

from schwabgym.physics.fast import FastExecutionEngine
from schwabgym.physics.hybrid import HybridExecutionEngine
from schwabgym.physics.realistic import RealisticExecutionEngine

pytestmark = pytest.mark.benchmark


@pytest.fixture
def market_data():
    """Typical OHLCV bar for benchmarking."""
    return {
        "Open": 150.00,
        "High": 151.50,
        "Low": 149.20,
        "Close": 150.80,
        "Volume": 500_000,
        "Volatility": 0.015,
    }


@pytest.fixture
def fast_engine():
    return FastExecutionEngine()


@pytest.fixture
def realistic_engine():
    return RealisticExecutionEngine(seed=42)


@pytest.fixture
def hybrid_engine():
    return HybridExecutionEngine(seed=42)


# ---------------------------------------------------------------------------
# calculate_execution_price benchmarks
# ---------------------------------------------------------------------------


def test_bench_fast_execution_price(benchmark, fast_engine, market_data):
    """Benchmark FastExecutionEngine.calculate_execution_price."""
    benchmark(
        fast_engine.calculate_execution_price,
        base_price=150.80,
        quantity=100,
        instruction="BUY",
        market_data=market_data,
    )


def test_bench_realistic_execution_price(benchmark, realistic_engine, market_data):
    """Benchmark RealisticExecutionEngine.calculate_execution_price."""
    benchmark(
        realistic_engine.calculate_execution_price,
        base_price=150.80,
        quantity=100,
        instruction="BUY",
        market_data=market_data,
    )


def test_bench_hybrid_execution_price(benchmark, hybrid_engine, market_data):
    """Benchmark HybridExecutionEngine.calculate_execution_price."""

    def run():
        hybrid_engine.select_engine_for_step()
        return hybrid_engine.calculate_execution_price(
            base_price=150.80,
            quantity=100,
            instruction="BUY",
            market_data=market_data,
        )

    benchmark(run)


# ---------------------------------------------------------------------------
# should_limit_fill benchmarks
# ---------------------------------------------------------------------------


def test_bench_fast_limit_fill(benchmark, fast_engine):
    """Benchmark FastExecutionEngine.should_limit_fill."""
    benchmark(
        fast_engine.should_limit_fill,
        limit_price=150.00,
        market_high=151.50,
        market_low=149.20,
        volume=500_000,
        quantity=100,
    )


def test_bench_realistic_limit_fill(benchmark, realistic_engine):
    """Benchmark RealisticExecutionEngine.should_limit_fill."""
    benchmark(
        realistic_engine.should_limit_fill,
        limit_price=150.00,
        market_high=151.50,
        market_low=149.20,
        volume=500_000,
        quantity=100,
    )


# ---------------------------------------------------------------------------
# Brownian Bridge benchmark
# ---------------------------------------------------------------------------


def test_bench_brownian_bridge(benchmark, realistic_engine):
    """Benchmark RealisticExecutionEngine.simulate_intrabar_path."""
    benchmark(
        realistic_engine.simulate_intrabar_path,
        open_price=150.00,
        high=151.50,
        low=149.20,
        close=150.80,
        n_ticks=60,
    )


# ---------------------------------------------------------------------------
# Batch throughput: simulate N steps
# ---------------------------------------------------------------------------


def test_bench_realistic_batch_1000_steps(benchmark, realistic_engine):
    """Benchmark 1000 execution price calculations (realistic engine throughput)."""
    rng = np.random.default_rng(42)
    bars = [
        {
            "Open": float(o),
            "High": float(h),
            "Low": float(lo),
            "Close": float(c),
            "Volume": int(v),
        }
        for o, h, lo, c, v in zip(
            rng.uniform(148, 152, 1000),
            rng.uniform(151, 153, 1000),
            rng.uniform(147, 149, 1000),
            rng.uniform(149, 151, 1000),
            rng.integers(100_000, 1_000_000, 1000),
        )
    ]

    def run_batch():
        for bar in bars:
            realistic_engine.calculate_execution_price(
                base_price=bar["Close"],
                quantity=100,
                instruction="BUY",
                market_data=bar,
            )

    benchmark(run_batch)


def test_bench_fast_batch_1000_steps(benchmark, fast_engine):
    """Benchmark 1000 execution price calculations (fast engine throughput)."""
    rng = np.random.default_rng(42)
    bars = [
        {
            "Open": float(o),
            "High": float(h),
            "Low": float(lo),
            "Close": float(c),
            "Volume": int(v),
        }
        for o, h, lo, c, v in zip(
            rng.uniform(148, 152, 1000),
            rng.uniform(151, 153, 1000),
            rng.uniform(147, 149, 1000),
            rng.uniform(149, 151, 1000),
            rng.integers(100_000, 1_000_000, 1000),
        )
    ]

    def run_batch():
        for bar in bars:
            fast_engine.calculate_execution_price(
                base_price=bar["Close"],
                quantity=100,
                instruction="BUY",
                market_data=bar,
            )

    benchmark(run_batch)
