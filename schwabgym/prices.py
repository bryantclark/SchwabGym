"""
SchwabGym Price Engine
======================

Handles market data simulation, time advancement, and price retrieval.
"""

import datetime
import logging
from typing import Dict, List, Optional, Union

import pandas as pd

logger = logging.getLogger(__name__)


class PriceEngine:
    """
    Manages historical market data and simulation time.

    Attributes:
        df (pd.DataFrame): Historical OHLCV data.
        current_step (int): Current simulation time index.
        max_steps (int): Total time steps available.
    """

    def __init__(self, market_data_df: pd.DataFrame):
        """
        Initialize the price engine.

        Args:
            market_data_df (pd.DataFrame): Historical market data.
        """
        # Validate required columns
        required_cols = {"Open", "High", "Low", "Close", "Volume"}
        if not required_cols.issubset(market_data_df.columns):
            missing = required_cols - set(market_data_df.columns)
            raise ValueError(
                f"Missing required columns: {missing}\n"
                f"Required: {required_cols}\n"
                f"Found: {set(market_data_df.columns)}"
            )

        self.df = market_data_df
        self.current_step = 0
        self.max_steps = len(self.df) - 1

    def advance_time(self) -> bool:
        """
        Advance simulator by one time step.

        Returns:
            bool: True if successfully advanced, False if at end of data.
        """
        if self.current_step >= self.max_steps:
            logger.warning("Reached end of market data")
            return False

        self.current_step += 1
        return True

    def reset(self) -> None:
        """Reset time to beginning."""
        self.current_step = 0

    def get_current_time(self) -> datetime.datetime:
        """Get current timestamp."""
        return self.df.index[self.current_step]

    def get_current_price(self, symbol: str, col: str = "Close") -> float:
        """
        Get current price for a symbol.

        Note: Currently simulator is single-asset based on the DF.
        The symbol arg is largely ignored in single-asset mode but kept for API shape.
        """
        # In multi-asset future, look up by symbol.
        # For now, we assume the DF applies to the symbol being queried.
        return float(self.df.iloc[self.current_step][col])

    def get_current_ohlcv(self) -> Dict[str, Union[float, int]]:
        """Get current step's full OHLCV data."""
        row = self.df.iloc[self.current_step]
        return {
            "Open": float(row["Open"]),
            "High": float(row["High"]),
            "Low": float(row["Low"]),
            "Close": float(row["Close"]),
            "Volume": int(row["Volume"]),
            "Volatility": float(row.get("Volatility", 0.01)),
        }

    def get_quotes_data(self, symbols: List[str]) -> Dict[str, Dict]:
        """
        Generate quote data for symbols.

        Args:
            symbols: List of symbols to quote.

        Returns:
            Dict: Quote data keyed by symbol.
        """
        response_body = {}
        ts_ms = int(self.get_current_time().timestamp() * 1000)

        row = self.df.iloc[self.current_step]
        price = float(row["Close"])
        volume = int(row["Volume"])
        volatility = float(row.get("Volatility", 0.01))

        # Dynamic spread
        spread_factor = 0.0005 * (1 + (volatility * 100))
        bid_price = price * (1 - spread_factor)
        ask_price = price * (1 + spread_factor)

        for sym in symbols:
            # For now, all symbols get the same price from the single DF
            response_body[sym] = {
                "quote": {
                    "symbol": sym,
                    "lastPrice": price,
                    "closePrice": price,
                    "bidPrice": bid_price,
                    "askPrice": ask_price,
                    "totalVolume": volume,
                    "tradeTime": ts_ms,
                }
            }
        return response_body

    def get_price_history_data(self, symbol: str) -> List[Dict]:
        """
        Get historical candles up to current step.

        Args:
            symbol: Ticker symbol.

        Returns:
            List[Dict]: List of candle dicts.
        """
        LOOKBACK = 50
        start_idx = max(0, self.current_step - LOOKBACK + 1)

        col_close = "AdjClose" if "AdjClose" in self.df.columns else "Close"

        subset = self.df.iloc[start_idx : self.current_step + 1]
        candles = []

        for ts, row in subset.iterrows():
            candles.append(
                {
                    "open": float(row["Open"]),
                    "high": float(row["High"]),
                    "low": float(row["Low"]),
                    "close": float(row[col_close]),
                    "volume": int(row["Volume"]),
                    "datetime": int(ts.timestamp() * 1000),
                }
            )
        return candles
