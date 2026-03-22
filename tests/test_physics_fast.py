import pytest

from schwabgym.physics.fast import FastExecutionEngine


def test_fast_calculate_execution_price_buy():
    engine = FastExecutionEngine(base_slippage=0.02)
    price = engine.calculate_execution_price(
        base_price=100.0, quantity=10, instruction="BUY", market_data={}
    )
    assert price == pytest.approx(100.02)


def test_fast_calculate_execution_price_sell():
    engine = FastExecutionEngine(base_slippage=0.02)
    price = engine.calculate_execution_price(
        base_price=100.0, quantity=10, instruction="SELL", market_data={}
    )
    assert price == pytest.approx(99.98)


def test_fast_should_limit_fill_true():
    engine = FastExecutionEngine()
    # price touched within range
    assert (
        engine.should_limit_fill(
            limit_price=100.0,
            market_high=101.0,
            market_low=99.0,
            volume=1000,
            quantity=10,
        )
        is True
    )


def test_fast_should_limit_fill_false():
    engine = FastExecutionEngine()
    # price not touched
    assert (
        engine.should_limit_fill(
            limit_price=105.0,
            market_high=101.0,
            market_low=99.0,
            volume=1000,
            quantity=10,
        )
        is False
    )
