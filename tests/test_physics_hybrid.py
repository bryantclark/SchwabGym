import numpy as np
import pytest

from schwabgym.physics.fast import FastExecutionEngine
from schwabgym.physics.hybrid import HybridExecutionEngine
from schwabgym.physics.realistic import RealisticExecutionEngine


def test_hybrid_always_fast():
    # probability 0 -> always fast
    engine = HybridExecutionEngine(realistic_probability=0.0, seed=42)
    # calculate price should match fast engine
    fast = FastExecutionEngine()
    base_price = 100.0
    qty = 10
    instr = "BUY"
    market_data = {"Open": 100, "High": 101, "Low": 99, "Close": 100, "Volume": 1000}
    price_hybrid = engine.calculate_execution_price(base_price, qty, instr, market_data)
    price_fast = fast.calculate_execution_price(base_price, qty, instr, market_data)
    assert price_hybrid == pytest.approx(price_fast)
    # should_limit_fill should match fast (binary)
    assert (
        engine.should_limit_fill(
            limit_price=100, market_high=101, market_low=99, volume=1000, quantity=10
        )
        is True
    )


def test_hybrid_always_realistic():
    engine = HybridExecutionEngine(realistic_probability=1.0, seed=42)
    realistic = RealisticExecutionEngine()
    base_price = 100.0
    qty = 10
    instr = "SELL"
    market_data = {"Open": 100, "High": 101, "Low": 99, "Close": 100, "Volume": 1000}
    price_hybrid = engine.calculate_execution_price(base_price, qty, instr, market_data)
    price_real = realistic.calculate_execution_price(
        base_price, qty, instr, market_data
    )
    assert price_hybrid == pytest.approx(price_real)
    # limit fill may be probabilistic; set high volume to guarantee fill
    engine.realistic.alpha = 1.0  # ensure fill probability 1
    engine.realistic.beta = 0.0
    assert (
        engine.should_limit_fill(
            limit_price=100, market_high=101, market_low=99, volume=1000, quantity=1
        )
        is True
    )


def test_hybrid_statistics():
    engine = HybridExecutionEngine(realistic_probability=0.3, seed=123)
    stats = engine.get_statistics()
    assert stats["mode"] == "hybrid"
    assert stats["realistic_probability"] == 0.3
    # current_mode may be None before any call
    assert stats["current_mode"] in (None, "fast", "realistic")
