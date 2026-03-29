"""
Tests for Price Engine
======================
"""

import pandas as pd
import pytest

from schwabgym.prices import PriceEngine


@pytest.fixture
def sample_data():
    """Local fixture with predictable data for price engine tests."""
    dates = pd.date_range(start="2023-01-01", periods=10, freq="1min")
    data = {
        "Open": [100.0] * 10,
        "High": [101.0] * 10,
        "Low": [99.0] * 10,
        "Close": [100.0] * 10,
        "Volume": [1000] * 10,
        "Volatility": [0.01] * 10,
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
        engine.advance_time()  # step 1
        history = engine.get_price_history_data("AAPL")
        assert len(history) == 2  # step 0 and 1
        assert history[0]["open"] == 100.0

    def test_get_price_history_filters_date_range(self, sample_data):
        engine = PriceEngine(sample_data)
        for _ in range(9):
            engine.advance_time()

        history = engine.get_price_history_data(
            "AAPL",
            start_datetime=sample_data.index[2].to_pydatetime(),
            end_datetime=sample_data.index[4].to_pydatetime(),
        )
        assert len(history) == 3
        assert history[0]["datetime"] == int(sample_data.index[2].timestamp() * 1000)
        assert history[-1]["datetime"] == int(sample_data.index[4].timestamp() * 1000)

    def test_get_price_history_resamples_to_requested_frequency(self, sample_data):
        engine = PriceEngine(sample_data)
        for _ in range(9):
            engine.advance_time()

        history = engine.get_price_history_data(
            "AAPL", frequency_type="minute", frequency=5
        )
        assert len(history) == 2


@pytest.fixture
def price_engine_setup():
    df = pd.DataFrame(
        {
            "Open": [100.0],
            "High": [105.0],
            "Low": [95.0],
            "Close": [101.0],
            "Volume": [1000],
        },
        index=pd.to_datetime(["2023-01-01"]),
    )
    return PriceEngine(df)


def test_get_price_unknown_symbol_fallback(price_engine_setup):
    engine = price_engine_setup
    # Single asset mode fallback
    price = engine.get_current_price("UNKNOWN_SYMBOL")
    assert price == 101.0


def test_get_quotes_mixed_symbols(price_engine_setup):
    engine = price_engine_setup

    quotes = engine.get_quotes_data(["DEFAULT", "UNKNOWN"])

    # DEFAULT should be there
    assert "DEFAULT" in quotes
    assert quotes["DEFAULT"]["quote"]["lastPrice"] == 101.0

    # UNKNOWN should be there because of fallback in single-asset mode
    assert "UNKNOWN" in quotes
    assert quotes["UNKNOWN"]["quote"]["lastPrice"] == 101.0


def test_multi_asset_behavior():
    df1 = pd.DataFrame(
        {"Open": [10], "High": [11], "Low": [9], "Close": [10], "Volume": [100]},
        index=pd.to_datetime(["2023-01-01"]),
    )

    df2 = pd.DataFrame(
        {"Open": [20], "High": [21], "Low": [19], "Close": [20], "Volume": [200]},
        index=pd.to_datetime(["2023-01-01"]),
    )

    data = {"SYM1": df1, "SYM2": df2}
    engine = PriceEngine(data)

    # Test known symbols
    assert engine.get_current_price("SYM1") == 10.0
    assert engine.get_current_price("SYM2") == 20.0

    # Unknown symbol in multi-asset mode should raise KeyError
    with pytest.raises(KeyError, match="UNKNOWN"):
        engine.get_current_price("UNKNOWN")


def test_price_history_empty(price_engine_setup):
    engine = price_engine_setup
    # Request history for unknown symbol in multi-asset mode (simulated by mocking data len > 1?)
    # Or just test the logic directly if we can.

    # Actually, let's test the single asset fallback for history
    history = engine.get_price_history_data("UNKNOWN")
    assert len(history) == 1
    assert history[0]["close"] == 101.0
