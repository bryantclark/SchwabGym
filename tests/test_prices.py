"""
Tests for Price Engine
======================
"""

import pandas as pd
import pytest
from datetime import datetime
from schwabgym.prices import PriceEngine

@pytest.fixture
def sample_data():
    dates = pd.date_range(start="2023-01-01", periods=10, freq="1min")
    data = {
        "Open": [100.0] * 10,
        "High": [101.0] * 10,
        "Low": [99.0] * 10,
        "Close": [100.0] * 10,
        "Volume": [1000] * 10,
        "Volatility": [0.01] * 10
    }
    return pd.DataFrame(data, index=dates)

class TestPriceEngine:
    def test_initialization(self, sample_data):
        engine = PriceEngine(sample_data)
        assert engine.current_step == 0
        assert engine.max_steps == 9

    def test_missing_columns(self):
        df = pd.DataFrame({"Open": [100]})
        with pytest.raises(ValueError):
            PriceEngine(df)

    def test_advance_time(self, sample_data):
        engine = PriceEngine(sample_data)
        assert engine.advance_time() is True
        assert engine.current_step == 1

        # Advance to end
        for _ in range(8):
            engine.advance_time()
        assert engine.current_step == 9

        # Advance past end
        assert engine.advance_time() is False
        assert engine.current_step == 9

    def test_reset(self, sample_data):
        engine = PriceEngine(sample_data)
        engine.advance_time()
        engine.reset()
        assert engine.current_step == 0

    def test_get_current_price(self, sample_data):
        engine = PriceEngine(sample_data)
        price = engine.get_current_price("TEST")
        assert price == 100.0

    def test_get_quotes_data(self, sample_data):
        engine = PriceEngine(sample_data)
        quotes = engine.get_quotes_data(["AAPL", "GOOG"])
        assert "AAPL" in quotes
        assert "GOOG" in quotes
        assert quotes["AAPL"]["quote"]["lastPrice"] == 100.0

        # Test dynamic spread
        bid = quotes["AAPL"]["quote"]["bidPrice"]
        ask = quotes["AAPL"]["quote"]["askPrice"]
        assert bid < 100.0
        assert ask > 100.0

    def test_get_price_history(self, sample_data):
        engine = PriceEngine(sample_data)
        engine.advance_time() # step 1
        history = engine.get_price_history_data("AAPL")
        assert len(history) == 2 # step 0 and 1
        assert history[0]["open"] == 100.0
