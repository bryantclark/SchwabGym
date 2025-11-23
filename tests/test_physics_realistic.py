import pytest
import numpy as np
from schwabgym.physics.realistic import RealisticExecutionEngine

def test_realistic_calculate_execution_price_buy():
    engine = RealisticExecutionEngine(impact_coefficient=0.5, participation_rate=0.1, queue_depth_factor=1.0)
    base_price = 100.0
    qty = 1000
    market_data = {'High': 101.0, 'Low': 99.0, 'Volume': 100000}
    price = engine.calculate_execution_price(base_price, qty, 'BUY', market_data)
    # Impact should be positive, price > base_price
    assert price > base_price

def test_realistic_calculate_execution_price_sell():
    engine = RealisticExecutionEngine(impact_coefficient=0.5, participation_rate=0.1, queue_depth_factor=1.0)
    base_price = 100.0
    qty = 1000
    market_data = {'High': 101.0, 'Low': 99.0, 'Volume': 100000}
    price = engine.calculate_execution_price(base_price, qty, 'SELL', market_data)
    assert price < base_price

def test_realistic_should_limit_fill_always_true_when_liquidity_high():
    engine = RealisticExecutionEngine(impact_coefficient=0.5, participation_rate=1.0, queue_depth_factor=0.0)
    # Set high volume and low quantity to guarantee fill
    filled = engine.should_limit_fill(limit_price=100.0, market_high=101.0, market_low=99.0, volume=100000, quantity=1)
    assert filled is True

def test_simulate_intrabar_path_properties():
    engine = RealisticExecutionEngine()
    path = engine.simulate_intrabar_path(open_price=100.0, high=102.0, low=98.0, close=101.0, n_ticks=60)
    assert len(path) == 60
    # Ensure path stays within high/low bounds
    assert path.min() >= 98.0
    assert path.max() <= 102.0
    # Ensure start and end match open and close
    assert path[0] == pytest.approx(100.0)
    assert path[-1] == pytest.approx(101.0)
