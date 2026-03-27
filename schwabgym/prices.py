"""
SchwabGym Price Engine
======================

Handles market data simulation, time advancement, and price retrieval.
"""

import datetime
import logging

import pandas as pd

logger = logging.getLogger(__name__)


class PriceEngine:
    """
    Manages historical market data and simulation time.

    Attributes:
        data (Dict[str, pd.DataFrame]): Historical OHLCV data keyed by symbol.
        current_step (int): Current simulation time index.
        max_steps (int): Total time steps available.
    """

    def __init__(self, market_data: pd.DataFrame | dict[str, pd.DataFrame]):
        """
        Initialize the price engine.

        Args:
            market_data (pd.DataFrame or Dict[str, pd.DataFrame]): Historical market data.
        """
        if isinstance(market_data, pd.DataFrame):
            # Backwards compatibility: Wrap single DF in a default key
            self.data = {"DEFAULT": market_data}
            self.main_symbol = "DEFAULT"
        elif isinstance(market_data, dict):
            self.data = market_data
            # Pick first key as main symbol for time sync
            self.main_symbol = next(iter(market_data))
        else:
            raise TypeError("market_data must be a DataFrame or Dict[str, DataFrame]")

        self.current_step = 0

        # Validate all DFs and find min length
        lengths = []
        required_cols = {"Open", "High", "Low", "Close", "Volume"}

        for sym, df in self.data.items():
            if not required_cols.issubset(df.columns):
                missing = required_cols - set(df.columns)
                raise ValueError(
                    f"Symbol {sym} missing required columns: {missing}\n"
                    f"Required: {required_cols}"
                )
            lengths.append(len(df))

        if not lengths:
            raise ValueError("No data provided")

        # Use minimum length to ensure we don't go out of bounds on any asset
        self.max_steps = min(lengths) - 1

    @property
    def df(self):
        """Backwards compatibility for single-asset access."""
        return self.data[self.main_symbol]

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
        ts: pd.Timestamp = self.data[self.main_symbol].index[self.current_step]
        return ts.to_pydatetime()  # type: ignore[no-any-return, return-value]

    def _resolve_dataframe(self, symbol: str) -> pd.DataFrame:
        """Resolve a symbol to its DataFrame.

        In single-asset mode, any symbol maps to the loaded data.
        In multi-asset mode, unknown symbols raise KeyError.
        """
        target_df = self.data.get(symbol)
        if target_df is not None:
            return target_df
        if len(self.data) == 1:
            return self.data[self.main_symbol]
        raise KeyError(
            f"Symbol '{symbol}' not in market data. Available: {list(self.data.keys())}"
        )

    def get_current_price(self, symbol: str, col: str = "Close") -> float:
        """Get current price for a symbol."""
        return float(self._resolve_dataframe(symbol).iloc[self.current_step][col])

    def get_current_ohlcv(self, symbol: str | None = None) -> dict[str, float | int]:
        """Get current step's full OHLCV data."""
        row = self._resolve_dataframe(symbol or self.main_symbol).iloc[
            self.current_step
        ]
        return {
            "Open": float(row["Open"]),
            "High": float(row["High"]),
            "Low": float(row["Low"]),
            "Close": float(row["Close"]),
            "Volume": int(row["Volume"]),
            "Volatility": float(row.get("Volatility", 0.01)),
        }

    def get_quotes_data(self, symbols: list[str]) -> dict[str, dict]:
        """
        Generate quote data for symbols.

        Args:
            symbols: List of symbols to quote.

        Returns:
            Dict: Quote data keyed by symbol.
        """
        response_body = {}
        ts_ms = int(self.get_current_time().timestamp() * 1000)

        for sym in symbols:
            try:
                df = self._resolve_dataframe(sym)
            except KeyError:
                continue  # Skip unknown symbols in multi-asset mode

            row = df.iloc[self.current_step]
            price = float(row["Close"])
            volume = int(row["Volume"])

            # Use pre-calculated columns if available (from data.py enhancements)
            if "BidPrice" in df.columns and "AskPrice" in df.columns:
                bid_price = float(row["BidPrice"])
                ask_price = float(row["AskPrice"])
            else:
                # Fallback calculation
                volatility = float(row.get("Volatility", 0.01))
                spread_factor = 0.0005 * (1 + (volatility * 100))
                bid_price = price * (1 - spread_factor)
                ask_price = price * (1 + spread_factor)

            bid_size = int(row.get("BidSize", 100))
            ask_size = int(row.get("AskSize", 100))
            last_size = int(row.get("LastSize", 100))

            response_body[sym] = {
                "quote": {
                    "symbol": sym,
                    "lastPrice": price,
                    "closePrice": price,
                    "bidPrice": bid_price,
                    "askPrice": ask_price,
                    "bidSize": bid_size,
                    "askSize": ask_size,
                    "lastSize": last_size,
                    "totalVolume": volume,
                    "tradeTime": ts_ms,
                    "quoteTime": ts_ms,
                }
            }
        return response_body

    def get_price_history_data(self, symbol: str) -> list[dict]:
        """
        Get historical candles up to current step.

        Args:
            symbol: Ticker symbol.

        Returns:
            List[Dict]: List of candle dicts.
        """
        LOOKBACK = 50
        target_df = self._resolve_dataframe(symbol)
        start_idx = max(0, self.current_step - LOOKBACK + 1)
        col_close = "AdjClose" if "AdjClose" in target_df.columns else "Close"

        subset = target_df.iloc[start_idx : self.current_step + 1]
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
