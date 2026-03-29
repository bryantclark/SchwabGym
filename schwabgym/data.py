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

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# --- Synthetic data constants ---
BASE_SPREAD = 0.01  # Default bid-ask spread ($)
HIGH_VOL_SPREAD = 0.03  # Spread during high-volatility regimes ($)
MARKET_OPEN_SPREAD_BUMP = 0.02  # Additional spread at open ($)
HIGH_VOL_QUANTILE = 0.9  # Threshold for "high volatility"
DEFAULT_VOLUME = 100_000  # Default volume for missing data
PARETO_SHAPE = 1.16  # Trade-size distribution (80/20 rule)
BASE_TRADE_SIZE = 100  # Minimum trade size for Pareto distribution


def add_synthetic_quotes(df: pd.DataFrame) -> pd.DataFrame:
    """Generates Bid/Ask prices based on volatility regimes."""
    df["Spread"] = BASE_SPREAD

    if "Volatility" in df.columns:
        high_vol_mask = df["Volatility"] > df["Volatility"].quantile(HIGH_VOL_QUANTILE)
        df.loc[high_vol_mask, "Spread"] = HIGH_VOL_SPREAD

    if isinstance(df.index, pd.DatetimeIndex):
        market_hours = df.index.indexer_between_time("09:30", "10:00")
        df.iloc[market_hours, df.columns.get_loc("Spread")] += MARKET_OPEN_SPREAD_BUMP

    # Calculate Bid/Ask from Close (or use High/Low for more variance)
    df["BidPrice"] = df["Close"] - (df["Spread"] / 2)
    df["AskPrice"] = df["Close"] + (df["Spread"] / 2)

    return df


def add_liquidity_depth(
    df: pd.DataFrame, rng: np.random.Generator | None = None
) -> pd.DataFrame:
    """Estimates Bid/Ask sizes based on volume."""
    if rng is None:
        rng = np.random.default_rng()

    liquidity_factor = rng.uniform(0.01, 0.05, size=len(df))

    df["BidSize"] = (df["Volume"] * liquidity_factor).astype(int)
    # Make AskSize slightly different to create imbalance signals
    df["AskSize"] = (
        df["Volume"] * liquidity_factor * rng.uniform(0.8, 1.2, size=len(df))
    ).astype(int)

    # Ensure at least 1 share/contract (using 100 as standard lot size base if volume permits, else 1)
    df["BidSize"] = df["BidSize"].clip(lower=1)
    df["AskSize"] = df["AskSize"].clip(lower=1)

    return df


def add_last_trade_size(
    df: pd.DataFrame, rng: np.random.Generator | None = None
) -> pd.DataFrame:
    """Estimates LastSize to simulate trade granularity."""
    if rng is None:
        rng = np.random.default_rng()

    # Simulate: Most trades are small (100), occasional large blocks
    # Use a Pareto distribution to simulate rare large prints
    sizes = (rng.pareto(PARETO_SHAPE, len(df)) + 1) * BASE_TRADE_SIZE

    # Clip to be realistic (cannot exceed bar volume)
    df["LastSize"] = np.minimum(sizes, df["Volume"]).astype(int)
    # Ensure LastSize is at least 1 if Volume > 0
    df["LastSize"] = np.maximum(df["LastSize"], np.where(df["Volume"] > 0, 1, 0))
    return df


def mark_data_quality(df: pd.DataFrame) -> pd.DataFrame:
    """Injects data quality flags."""
    df["DataValid"] = True

    # Mark flat-line periods (no price movement) as suspicious
    # Real bots often pause if price hasn't moved for 5 minutes (data feed crash?)
    df.loc[df["Close"].pct_change(periods=5) == 0, "DataValid"] = False

    # Mark low-volume periods as "Low Confidence"
    low_vol_threshold = df["Volume"].quantile(0.05)
    df.loc[df["Volume"] < low_vol_threshold, "DataValid"] = False

    return df


def load_and_clean_data(
    filepath: str, symbol: str | None = None, *, allow_dummy: bool = False
) -> pd.DataFrame:
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
        allow_dummy (bool): If True, missing files generate synthetic data instead
            of raising FileNotFoundError.

    Returns:
        pd.DataFrame: Cleaned dataframe with columns:
            - Open, High, Low, Close: Raw prices
            - AdjClose: Adjusted prices
            - Volume: Share volume
            - Volatility: Calculated volatility proxy

    Raises:
        FileNotFoundError: If file doesn't exist and allow_dummy is False
        ValueError: If required columns are missing

    Example:
        >>> df = load_and_clean_data("AAPL_1min.csv")
        >>> print(df.head())
        >>> # df.index is DatetimeIndex
        >>> # df['Close'] = raw execution prices
        >>> # df['AdjClose'] = adjusted analytical prices
    """
    logger.info(f"Loading data from {filepath}")

    try:
        df = pd.read_csv(filepath)
        logger.info(f"Loaded {len(df)} rows")
    except FileNotFoundError as exc:
        if not allow_dummy:
            raise FileNotFoundError(
                f"File not found: {filepath}. Pass allow_dummy=True to generate "
                "synthetic data explicitly."
            ) from exc
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
            raise ValueError("Could not parse timestamp column") from e
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
                df[col] = DEFAULT_VOLUME
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

    # Parkinson volatility estimator: σ = ln(H/L) / sqrt(4·ln(2))
    # More efficient than close-to-close; uses high-low range properly
    if "Volatility" not in df.columns:
        df["Volatility"] = np.log(df["High"] / df["Low"]) / np.sqrt(4 * np.log(2))
        df["Volatility"] = df["Volatility"].fillna(0.01)  # Default volatility
        logger.debug("Calculated Parkinson volatility from price range")

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
            logger.warning("Removed rows with invalid prices")

    # Check for volume
    if (df["Volume"] < 0).any():
        logger.warning("Found negative volume, setting to 0")
        df.loc[df["Volume"] < 0, "Volume"] = 0

    # ==================== SYNTHETIC DATA ENRICHMENT ====================
    _rng = np.random.default_rng()
    df = add_synthetic_quotes(df)
    df = add_liquidity_depth(df, rng=_rng)
    df = add_last_trade_size(df, rng=_rng)
    df = mark_data_quality(df)

    # ==================== FINAL VALIDATION ====================

    required_final = [
        "Open",
        "High",
        "Low",
        "Close",
        "AdjClose",
        "Volume",
        "Volatility",
        "BidPrice",
        "AskPrice",
        "BidSize",
        "AskSize",
        "LastSize",
        "DataValid",
    ]
    missing_final = [col for col in required_final if col not in df.columns]
    if missing_final:
        logger.warning(f"Missing expected columns after processing: {missing_final}")

    logger.info(f"Data cleaning complete. Shape: {df.shape}")
    logger.info(f"Columns: {list(df.columns)}")

    return df


def generate_dummy_data(
    symbol: str = "DUMMY",
    periods: int = 1000,
    start_price: float = 100.0,
    volatility: float = 0.02,
    freq: str = "1min",
    seed: int | None = None,
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
        >>> df = generate_dummy_data("TEST", periods=500)
        >>> print(df.shape)
        (500, 7)
    """
    logger.info(f"Generating {periods} periods of dummy data for {symbol}")

    rng = np.random.default_rng(seed)

    # Generate timestamps
    dates = pd.date_range(start="2023-01-01 09:30:00", periods=periods, freq=freq)

    # Generate price path using geometric Brownian motion
    returns = rng.normal(0, volatility, periods)
    price_path = start_price * np.exp(np.cumsum(returns))

    # Add intraday volatility
    intraday_vol = rng.uniform(0.002, 0.008, periods)

    # Generate OHLC
    opens = price_path
    highs = price_path * (1 + intraday_vol)
    lows = price_path * (1 - intraday_vol)
    closes = price_path * (1 + rng.normal(0, volatility / 2, periods))

    # Ensure OHLC consistency (High >= Close >= Low, etc.)
    for i in range(periods):
        high = max(opens[i], closes[i])
        low = min(opens[i], closes[i])

        highs[i] = max(high, highs[i])
        lows[i] = min(low, lows[i])

    # Generate volume (random with trend)
    volume = rng.poisson(DEFAULT_VOLUME, periods)

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

    # Enrich dummy data too
    df = add_synthetic_quotes(df)
    df = add_liquidity_depth(df, rng=rng)
    df = add_last_trade_size(df, rng=rng)
    df = mark_data_quality(df)

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
        >>> df_5min = resample_data(df_1min, "5min")
    """
    logger.info(f"Resampling data to {timeframe}")

    agg_dict = {
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "AdjClose": "last",
        "Volume": "sum",
        "Volatility": "mean",
    }

    # Add new columns to aggregation if they exist
    if "BidPrice" in df.columns:
        agg_dict.update(
            {
                "BidPrice": "last",
                "AskPrice": "last",
                "BidSize": "last",
                "AskSize": "last",
                "LastSize": "last",
                "DataValid": "all",
            }
        )

    resampled = df.resample(timeframe).agg(agg_dict)

    # Remove any NaN rows created by resampling
    resampled.dropna(inplace=True)

    logger.info(f"Resampled to {len(resampled)} rows")

    return resampled


def _add_indicators_talib(df: pd.DataFrame) -> pd.DataFrame:
    """Add indicators using TA-Lib (faster, handles edge cases better)."""
    import talib

    prices = df["AdjClose"].values

    # Simple Moving Averages
    df["SMA_10"] = talib.SMA(prices, timeperiod=10)
    df["SMA_20"] = talib.SMA(prices, timeperiod=20)
    df["SMA_50"] = talib.SMA(prices, timeperiod=50)

    # Exponential Moving Averages
    df["EMA_12"] = talib.EMA(prices, timeperiod=12)
    df["EMA_26"] = talib.EMA(prices, timeperiod=26)

    # MACD
    df["MACD"], df["MACD_Signal"], _ = talib.MACD(
        prices, fastperiod=12, slowperiod=26, signalperiod=9
    )

    # RSI
    df["RSI"] = talib.RSI(prices, timeperiod=14)

    # Bollinger Bands
    df["BB_Upper"], df["BB_Middle"], df["BB_Lower"] = talib.BBANDS(
        prices, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0
    )

    return df


def _add_indicators_pandas(df: pd.DataFrame) -> pd.DataFrame:
    """Fallback indicator calculation using pandas (no TA-Lib required)."""
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

    # RSI (Wilder's smoothing)
    delta = df["AdjClose"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    df["RSI"] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    df["BB_Middle"] = df["AdjClose"].rolling(window=20).mean()
    bb_std = df["AdjClose"].rolling(window=20).std()
    df["BB_Upper"] = df["BB_Middle"] + (bb_std * 2)
    df["BB_Lower"] = df["BB_Middle"] - (bb_std * 2)

    return df


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add common technical indicators to dataframe.

    Uses TA-Lib when available (faster, battle-tested edge-case handling).
    Falls back to pandas implementation otherwise.

    Adds:
    - SMA (Simple Moving Average) — 10, 20, 50
    - EMA (Exponential Moving Average) — 12, 26
    - RSI (Relative Strength Index) — 14
    - MACD (Moving Average Convergence Divergence) — 12/26/9
    - Bollinger Bands — 20, 2σ

    Args:
        df (pd.DataFrame): Input dataframe with 'AdjClose' column

    Returns:
        pd.DataFrame: Dataframe with added indicator columns

    Note:
        This is optional - the trading environment calculates indicators
        internally. Use this for custom strategies.

    Example:
        >>> df = add_technical_indicators(df)
        >>> print(df[["Close", "SMA_20", "RSI"]].head())
    """
    logger.info("Adding technical indicators")

    try:
        df = _add_indicators_talib(df)
        logger.info("Technical indicators added (TA-Lib)")
    except ImportError:
        df = _add_indicators_pandas(df)
        logger.info("Technical indicators added (pandas fallback)")

    return df


def split_train_test(
    df: pd.DataFrame, train_ratio: float = 0.8
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into training and testing sets.
    """
    if not 0 < train_ratio < 1:
        raise ValueError(f"Train ratio must be between 0 and 1, got {train_ratio}")

    split_idx = int(len(df) * train_ratio)

    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    logger.info(f"Split data: {len(train_df)} train, {len(test_df)} test")

    return train_df, test_df
