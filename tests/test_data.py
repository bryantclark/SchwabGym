"""
Data Loading Tests
==================

Test data loading and preprocessing functions.

Author: Bryant Clark
"""

import numpy as np
import pandas as pd
import pytest

from schwabgym import (
    add_technical_indicators,
    generate_dummy_data,
    load_and_clean_data,
    resample_data,
    split_train_test,
)


class TestLoadAndCleanData:
    """Test load_and_clean_data function."""

    def test_load_valid_csv(self, alpha_vantage_csv):
        """Test loading a valid CSV file."""
        df = load_and_clean_data(alpha_vantage_csv)

        # Check required columns exist
        assert "Open" in df.columns
        assert "High" in df.columns
        assert "Low" in df.columns
        assert "Close" in df.columns
        assert "Volume" in df.columns
        assert "AdjClose" in df.columns
        assert "Volatility" in df.columns

    def test_load_nonexistent_file(self):
        """Test that nonexistent files fail fast by default."""
        with pytest.raises(FileNotFoundError, match="allow_dummy=True"):
            load_and_clean_data("nonexistent_file.csv")

    def test_load_nonexistent_file_allow_dummy(self):
        """Test explicit dummy-data fallback for missing files."""
        df = load_and_clean_data("nonexistent_file.csv", allow_dummy=True)

        assert len(df) > 0
        assert "Close" in df.columns

    def test_index_is_datetime(self, alpha_vantage_csv):
        """Test that index is converted to datetime."""
        df = load_and_clean_data(alpha_vantage_csv)

        assert isinstance(df.index, pd.DatetimeIndex)

    def test_data_sorted_chronologically(self, alpha_vantage_csv):
        """Test that data is sorted in ascending time order."""
        df = load_and_clean_data(alpha_vantage_csv)

        assert df.index.is_monotonic_increasing

    def test_no_nan_values(self, alpha_vantage_csv):
        """Test that NaN values are handled."""
        df = load_and_clean_data(alpha_vantage_csv)

        assert not df.isnull().any().any()

    def test_volatility_calculated(self, alpha_vantage_csv):
        """Test that volatility is calculated."""
        df = load_and_clean_data(alpha_vantage_csv)

        assert "Volatility" in df.columns
        assert (df["Volatility"] >= 0).all()  # Volatility should be non-negative


class TestGenerateDummyData:
    """Test generate_dummy_data function."""

    def test_correct_length(self):
        """Test that dummy data has correct length."""
        df = generate_dummy_data("TEST", periods=200)

        assert len(df) == 200

    def test_all_columns_present(self):
        """Test that all required columns are present."""
        df = generate_dummy_data("TEST", periods=100)

        required = ["Open", "High", "Low", "Close", "AdjClose", "Volume", "Volatility"]
        for col in required:
            assert col in df.columns

    def test_ohlc_consistency(self):
        """Test that OHLC data is internally consistent."""
        df = generate_dummy_data("TEST", periods=100)

        # High should be >= Close
        assert (df["High"] >= df["Close"]).all()

        # Low should be <= Close
        assert (df["Low"] <= df["Close"]).all()

        # High should be >= Low
        assert (df["High"] >= df["Low"]).all()

    def test_volume_positive(self):
        """Test that volume is always positive."""
        df = generate_dummy_data("TEST", periods=100)

        assert (df["Volume"] > 0).all()

    def test_price_around_start_price(self):
        """Test that prices stay around starting price."""
        start_price = 150.0
        df = generate_dummy_data("TEST", periods=100, start_price=start_price)

        mean_price = df["Close"].mean()

        # Should be within 30% of start price (with some randomness)
        assert 100 < mean_price < 200


class TestResampleData:
    """Test resample_data function."""

    def test_resample_reduces_length(self):
        """Test that resampling reduces data length."""
        df_1min = generate_dummy_data("TEST", periods=100, freq="1min")
        df_5min = resample_data(df_1min, "5min")

        assert len(df_5min) < len(df_1min)

    def test_resampled_ohlc_consistency(self):
        """Test OHLC consistency after resampling."""
        df_1min = generate_dummy_data("TEST", periods=100, freq="1min")
        df_5min = resample_data(df_1min, "5min")

        assert (df_5min["High"] >= df_5min["Close"]).all()
        assert (df_5min["Low"] <= df_5min["Close"]).all()

    def test_volume_sums(self):
        """Test that volume sums correctly."""
        df_1min = generate_dummy_data("TEST", periods=10, freq="1min")
        df_5min = resample_data(df_1min, "5min")

        # First 5 minutes of 1min data should sum to first 5min bar
        original_sum = df_1min["Volume"][:5].sum()
        resampled_first = df_5min["Volume"].iloc[0]

        assert abs(original_sum - resampled_first) < 1  # Allow rounding


class TestAddTechnicalIndicators:
    """Test add_technical_indicators function."""

    def test_indicators_added(self):
        """Test that indicators are added to dataframe."""
        df = generate_dummy_data("TEST", periods=100)
        df_with_indicators = add_technical_indicators(df)

        expected_indicators = [
            "SMA_10",
            "SMA_20",
            "SMA_50",
            "EMA_12",
            "EMA_26",
            "MACD",
            "MACD_Signal",
            "RSI",
            "BB_Middle",
            "BB_Upper",
            "BB_Lower",
        ]

        for indicator in expected_indicators:
            assert indicator in df_with_indicators.columns

    def test_rsi_bounds(self):
        """Test that RSI is bounded between 0 and 100."""
        df = generate_dummy_data("TEST", periods=100)
        df_with_indicators = add_technical_indicators(df)

        rsi = df_with_indicators["RSI"].dropna()

        assert (rsi >= 0).all()
        assert (rsi <= 100).all()

    def test_bollinger_bands_order(self):
        """Test that Bollinger Bands are in correct order."""
        df = generate_dummy_data("TEST", periods=100)
        df_with_indicators = add_technical_indicators(df)

        # Drop NaN rows
        df_clean = df_with_indicators.dropna()

        # Upper should be >= Middle >= Lower
        assert (df_clean["BB_Upper"] >= df_clean["BB_Middle"]).all()
        assert (df_clean["BB_Middle"] >= df_clean["BB_Lower"]).all()


class TestSplitTrainTest:
    """Test split_train_test function."""

    def test_split_ratio(self):
        """Test that split ratio is correct."""
        df = generate_dummy_data("TEST", periods=100)
        train, test = split_train_test(df, train_ratio=0.8)

        assert len(train) == 80
        assert len(test) == 20

    def test_no_overlap(self):
        """Test that train and test don't overlap."""
        df = generate_dummy_data("TEST", periods=100)
        train, test = split_train_test(df, train_ratio=0.7)

        # Last timestamp of train should be before first of test
        assert train.index[-1] < test.index[0]

    def test_chronological_order(self):
        """Test that train comes before test chronologically."""
        df = generate_dummy_data("TEST", periods=100)
        train, test = split_train_test(df, train_ratio=0.8)

        assert train.index[0] < test.index[0]
        assert train.index[-1] < test.index[-1]

    def test_invalid_split_ratio(self):
        """Test that invalid split ratio raises error."""
        df = generate_dummy_data("TEST", periods=100)

        with pytest.raises(ValueError):
            split_train_test(df, train_ratio=1.5)

        with pytest.raises(ValueError):
            split_train_test(df, train_ratio=-0.1)

    def test_empty_dataframe(self):
        """Test splitting empty dataframe."""
        df = pd.DataFrame()

        # Should handle gracefully or raise specific error
        # Assuming implementation returns empty frames or raises error
        # Let's check what it does
        try:
            train, test = split_train_test(df)
            assert len(train) == 0
            assert len(test) == 0
        except (ValueError, IndexError):
            pass  # Acceptable behavior

    def test_nan_handling(self, tmp_path):
        """Test that NaNs are filled."""
        # Create CSV with NaNs
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=10),
                "open": [100.0] * 9 + [np.nan],
                "high": [101.0] * 10,
                "low": [99.0] * 10,
                "close": [100.0] * 10,
                "volume": [1000] * 10,
            }
        )
        csv_path = tmp_path / "nan_test.csv"
        df.to_csv(csv_path, index=False)

        loaded_df = load_and_clean_data(str(csv_path))
        assert not loaded_df.isnull().any().any()

    def test_negative_price_handling(self, tmp_path):
        """Test that negative prices are filtered."""
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=10),
                "open": [100.0] * 9 + [-50.0],  # One negative price
                "high": [101.0] * 10,
                "low": [99.0] * 10,
                "close": [100.0] * 10,
                "volume": [1000] * 10,
            }
        )
        csv_path = tmp_path / "neg_test.csv"
        df.to_csv(csv_path, index=False)

        loaded_df = load_and_clean_data(str(csv_path))
        assert (loaded_df["Open"] > 0).all()

    def test_timestamp_error(self, tmp_path):
        """Test timestamp parsing error."""
        df = pd.DataFrame(
            {
                "timestamp": ["invalid_date"] * 10,
                "open": [100.0] * 10,
                "high": [101.0] * 10,
                "low": [99.0] * 10,
                "close": [100.0] * 10,
                "volume": [1000] * 10,
            }
        )
        csv_path = tmp_path / "bad_time.csv"
        df.to_csv(csv_path, index=False)

        with pytest.raises(ValueError, match="Could not parse timestamp"):
            load_and_clean_data(str(csv_path))

    def test_missing_close_column(self, tmp_path):
        """Test handling of missing close column (use adj_close)."""
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=10),
                "open": [100.0] * 10,
                "high": [101.0] * 10,
                "low": [99.0] * 10,
                "adj_close": [100.0] * 10,  # Only adj_close
                "volume": [1000] * 10,
            }
        )
        csv_path = tmp_path / "no_close.csv"
        df.to_csv(csv_path, index=False)

        loaded_df = load_and_clean_data(str(csv_path))
        assert "Close" in loaded_df.columns
        assert (loaded_df["Close"] == loaded_df["AdjClose"]).all()

    def test_negative_volume(self, tmp_path):
        """Test negative volume handling."""
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=10),
                "open": [100.0] * 10,
                "high": [101.0] * 10,
                "low": [99.0] * 10,
                "close": [100.0] * 10,
                "volume": [-100] * 10,  # Negative volume
            }
        )
        csv_path = tmp_path / "neg_vol.csv"
        df.to_csv(csv_path, index=False)

        loaded_df = load_and_clean_data(str(csv_path))
        assert (loaded_df["Volume"] == 0).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
