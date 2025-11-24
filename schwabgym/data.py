"""
SchwabGym Data Loading & Processing
====================================

Utilities for loading market data from various sources and preparing
it for simulation with dual-state reconstruction.

Author: Bryant Clark
Repository: https://github.com/bryantclark/SchwabGym
License: MIT
"""

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def load_and_clean_data(filepath: str, symbol: Optional[str] = None) -> pd.DataFrame:
    """
    Load and preprocess historical market data with dual-price state reconstruction.

    This function handles multiple data formats and reconstructs both raw and adjusted
    price states needed for realistic simulation:
    - Raw prices (Close): Used for order execution and PnL
    - Adjusted prices (AdjClose): Used for technical analysis

    Supports data from:
    - Alpha Vantage (daily, intraday, adjusted)
    - Yahoo Finance
    - Custom CSVs with OHLCV columns

    Args:
        filepath (str): Path to CSV file
        symbol (str, optional): Symbol name (for logging)

    Returns:
        pd.DataFrame: Cleaned dataframe with columns:
            - Open, High, Low, Close: Raw prices
            - AdjClose: Adjusted prices
            - Volume: Share volume
            - Volatility: Calculated volatility proxy

    Raises:
        FileNotFoundError: If file doesn't exist (generates dummy data instead)
        ValueError: If required columns are missing

    Example:
        >>> df = load_and_clean_data('AAPL_1min.csv')
        >>> print(df.head())
        >>> # df.index is DatetimeIndex
        >>> # df['Close'] = raw execution prices
        >>> # df['AdjClose'] = adjusted analytical prices
    """
    logger.info(f"Loading data from {filepath}")

    try:
        df = pd.read_csv(filepath)
        logger.info(f"Loaded {len(df)} rows")
    except FileNotFoundError:
        logger.warning(f"File not found: {filepath}. Generating dummy data.")
        return generate_dummy_data(symbol or "DUMMY")

    # ==================== NORMALIZE COLUMN NAMES ====================

    # Convert all column names to lowercase for consistent handling
    df.columns = [c.lower().strip() for c in df.columns]

    # Common column name variations
    rename_map = {
        # Timestamp columns
        "time": "timestamp",
        "date": "timestamp",
        "datetime": "timestamp",
        # Price columns
        "adjusted close": "adj_close",
        "adj_close": "adj_close",
        "adjusted_close": "adj_close",
        "close": "close",
        # Volume columns
        "vol": "volume",
        "volume": "volume",
        # OHLC columns (usually correct)
        "open": "open",
        "high": "high",
        "low": "low",
    }

    df.rename(columns=rename_map, inplace=True)
    logger.debug(f"Normalized columns: {list(df.columns)}")

    # ==================== TIMESTAMP PROCESSING ====================

    if "timestamp" in df.columns:
        try:
            # Use format='mixed' to handle inconsistent formats without warning
            df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed")
            df.set_index("timestamp", inplace=True)
            df.sort_index(ascending=True, inplace=True)
            logger.info(f"Data range: {df.index[0]} to {df.index[-1]}")
        except Exception as e:
            logger.error(f"Error parsing timestamps: {e}")
            raise ValueError("Could not parse timestamp column")
    else:
        logger.warning("No timestamp column found. Using sequential index.")
        df.index = pd.date_range(start="2020-01-01", periods=len(df), freq="1min")

    # ==================== DUAL-STATE RECONSTRUCTION ====================

    # Ensure we have both raw and adjusted prices
    if "adj_close" not in df.columns:
        if "close" in df.columns:
            # Assume close is adjusted (Yahoo default), copy to adj_close
            df["adj_close"] = df["close"]
            logger.info("Using 'close' as both raw and adjusted prices")
        else:
            raise ValueError("Must have either 'close' or 'adj_close' column")

    if "close" not in df.columns:
        # Have adj_close but no close, use adj_close as raw
        df["close"] = df["adj_close"]
        logger.warning("No 'close' column found, using 'adj_close' as raw price")

    # ==================== ENSURE REQUIRED COLUMNS ====================

    required_cols = ["open", "high", "low", "close", "volume"]
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        logger.warning(f"Missing columns {missing_cols}, filling with 'close' value")
        for col in missing_cols:
            if col == "volume":
                df[col] = 100000  # Default volume
            else:
                df[col] = df["close"]

    # ==================== CAPITALIZE FOR SIMULATOR ====================

    # Simulator expects capitalized column names
    final_map = {
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "adj_close": "AdjClose",
        "volume": "Volume",
    }

    df.rename(columns=final_map, inplace=True)

    # ==================== CALCULATE VOLATILITY ====================

    # Volatility proxy: (High - Low) / Close
    # This is used for market impact calculations
    if "Volatility" not in df.columns:
        df["Volatility"] = (df["High"] - df["Low"]) / df["Close"]
        df["Volatility"] = df["Volatility"].fillna(0.01)  # Default volatility
        logger.debug("Calculated volatility from price range")

    # ==================== DATA VALIDATION ====================

    # Check for NaN values
    if df.isnull().any().any():
        logger.warning("Found NaN values, forward filling...")
        # Updated to use ffill/bfill to silence FutureWarning
        df.ffill(inplace=True)
        df.bfill(inplace=True)

    # Check for non-positive prices
    price_cols = ["Open", "High", "Low", "Close", "AdjClose"]
    for col in price_cols:
        if (df[col] <= 0).any():
            logger.error(f"Found non-positive values in {col}")
            df = df[df[col] > 0]
            logger.warning(f"Removed rows with invalid prices")

    # Check for volume
    if (df["Volume"] < 0).any():
        logger.warning("Found negative volume, setting to 0")
        df.loc[df["Volume"] < 0, "Volume"] = 0

    # ==================== FINAL VALIDATION ====================

    required_final = [
        "Open",
        "High",
        "Low",
        "Close",
        "AdjClose",
        "Volume",
        "Volatility",
    ]
    if not all(col in df.columns for col in required_final):
        raise ValueError(f"Missing required columns after processing: {required_final}")

    logger.info(f"Data cleaning complete. Shape: {df.shape}")
    logger.info(f"Columns: {list(df.columns)}")

    return df


def generate_dummy_data(
    symbol: str = "DUMMY",
    periods: int = 1000,
    start_price: float = 100.0,
    volatility: float = 0.02,
    freq: str = "1min",
) -> pd.DataFrame:
    """
    Generate synthetic market data for testing.

    Creates realistic-looking price data using geometric Brownian motion
    with intraday patterns.

    Args:
        symbol (str): Symbol name
        periods (int): Number of time periods
        start_price (float): Initial price
        volatility (float): Price volatility (standard deviation)
        freq (str): Time frequency ('1min', '5min', '1H', '1D')

    Returns:
        pd.DataFrame: Synthetic market data

    Example:
        >>> df = generate_dummy_data('TEST', periods=500)
        >>> print(df.shape)
        (500, 7)
    """
    logger.info(f"Generating {periods} periods of dummy data for {symbol}")

    # Generate timestamps
    dates = pd.date_range(start="2023-01-01 09:30:00", periods=periods, freq=freq)

    # Generate price path using geometric Brownian motion
    returns = np.random.normal(0, volatility, periods)
    price_path = start_price * np.exp(np.cumsum(returns))

    # Add intraday volatility
    intraday_vol = np.random.uniform(0.002, 0.008, periods)

    # Generate OHLC
    opens = price_path
    highs = price_path * (1 + intraday_vol)
    lows = price_path * (1 - intraday_vol)
    closes = price_path * (1 + np.random.normal(0, volatility / 2, periods))

    # Ensure OHLC consistency (High >= Close >= Low, etc.)
    for i in range(periods):
        high = max(opens[i], closes[i])
        low = min(opens[i], closes[i])

        highs[i] = max(high, highs[i])
        lows[i] = min(low, lows[i])

    # Generate volume (random with trend)
    base_volume = 100000
    volume = np.random.poisson(base_volume, periods)

    # Create dataframe
    df = pd.DataFrame(
        {
            "Open": opens,
            "High": highs,
            "Low": lows,
            "Close": closes,
            "AdjClose": closes,  # Same as close for dummy data
            "Volume": volume,
            "Volatility": intraday_vol,
        },
        index=dates,
    )

    logger.info(f"Generated dummy data: {df.shape[0]} rows")
    logger.info(f"Price range: ${df['Close'].min():.2f} - ${df['Close'].max():.2f}")

    return df


def resample_data(df: pd.DataFrame, timeframe: str = "5min") -> pd.DataFrame:
    """
    Resample data to a different timeframe.

    Aggregates OHLCV data to a coarser timeframe while maintaining
    dual-price state.

    Args:
        df (pd.DataFrame): Input dataframe
        timeframe (str): Target timeframe ('5min', '15min', '1H', '1D', etc.)

    Returns:
        pd.DataFrame: Resampled dataframe

    Example:
        >>> df_5min = resample_data(df_1min, '5min')
    """
    logger.info(f"Resampling data to {timeframe}")

    resampled = df.resample(timeframe).agg(
        {
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "AdjClose": "last",
            "Volume": "sum",
            "Volatility": "mean",
        }
    )

    # Remove any NaN rows created by resampling
    resampled.dropna(inplace=True)

    logger.info(f"Resampled to {len(resampled)} rows")

    return resampled


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add common technical indicators to dataframe.

    Adds:
    - SMA (Simple Moving Average)
    - EMA (Exponential Moving Average)
    - RSI (Relative Strength Index)
    - MACD (Moving Average Convergence Divergence)
    - Bollinger Bands

    Args:
        df (pd.DataFrame): Input dataframe

    Returns:
        pd.DataFrame: Dataframe with added indicator columns

    Note:
        This is optional - the trading environment calculates indicators
        internally. Use this for custom strategies.

    Example:
        >>> df = add_technical_indicators(df)
        >>> print(df[['Close', 'SMA_20', 'RSI']].head())
    """
    logger.info("Adding technical indicators")

    # Simple Moving Averages
    df["SMA_10"] = df["AdjClose"].rolling(window=10).mean()
    df["SMA_20"] = df["AdjClose"].rolling(window=20).mean()
    df["SMA_50"] = df["AdjClose"].rolling(window=50).mean()

    # Exponential Moving Average
    df["EMA_12"] = df["AdjClose"].ewm(span=12, adjust=False).mean()
    df["EMA_26"] = df["AdjClose"].ewm(span=26, adjust=False).mean()

    # MACD
    df["MACD"] = df["EMA_12"] - df["EMA_26"]
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    # RSI
    delta = df["AdjClose"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    df["BB_Middle"] = df["AdjClose"].rolling(window=20).mean()
    bb_std = df["AdjClose"].rolling(window=20).std()
    df["BB_Upper"] = df["BB_Middle"] + (bb_std * 2)
    df["BB_Lower"] = df["BB_Middle"] - (bb_std * 2)

    logger.info("Technical indicators added")

    return df


def split_train_test(
    df: pd.DataFrame, train_ratio: float = 0.8
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into training and testing sets.

    Args:
        df (pd.DataFrame): Input dataframe
        train_ratio (float): Fraction of data for training (0-1)

    Returns:
        tuple: (train_df, test_df)

    Example:
        >>> train, test = split_train_test(df, train_ratio=0.8)
        >>> print(f"Train: {len(train)}, Test: {len(test)}")
    """
    if not 0 < train_ratio < 1:
        raise ValueError(f"Train ratio must be between 0 and 1, got {train_ratio}")

    split_idx = int(len(df) * train_ratio)

    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    logger.info(f"Split data: {len(train_df)} train, {len(test_df)} test")

    return train_df, test_df
