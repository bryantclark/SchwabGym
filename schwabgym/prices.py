"""
SchwabGym Price Engine
======================

Handles market data simulation, time advancement, and price retrieval.
"""

import datetime
import logging

import pandas as pd

logger = logging.getLogger(__name__)

_HISTORY_LOOKBACK = 50
_RESAMPLE_RULES: dict[tuple[str, int], str] = {
    ("minute", 1): "1min",
    ("minute", 5): "5min",
    ("minute", 10): "10min",
    ("minute", 15): "15min",
    ("minute", 30): "30min",
    ("daily", 1): "1D",
    ("weekly", 1): "W-MON",
    ("monthly", 1): "MS",
}


def _coerce_history_arg(value):
    if hasattr(value, "value"):
        value = value.value
    if isinstance(value, str):
        return value.lower()
    return value


def _normalize_timestamp(
    value: datetime.datetime | int | float | None, index: pd.DatetimeIndex
) -> pd.Timestamp | None:
    if value is None:
        return None

    if isinstance(value, int | float):
        unit = "ms" if abs(value) >= 10_000_000_000 else "s"
        ts = pd.Timestamp(value, unit=unit)
    else:
        ts = pd.Timestamp(value)

    if index.tz is None and ts.tzinfo is not None:
        return ts.tz_convert(None)
    if index.tz is not None and ts.tzinfo is None:
        return ts.tz_localize(index.tz)
    if index.tz is not None and ts.tzinfo is not None:
        return ts.tz_convert(index.tz)
    return ts


def _resolve_period_start(
    period_type, period, end_timestamp: pd.Timestamp
) -> pd.Timestamp | None:
    period_type = _coerce_history_arg(period_type)
    period = int(period) if period is not None else None

    if period_type == "ytd":
        return pd.Timestamp(
            year=end_timestamp.year, month=1, day=1, tz=end_timestamp.tz
        )
    if period is None:
        return None
    if period_type == "day":
        return end_timestamp - pd.Timedelta(days=period)
    if period_type == "month":
        return end_timestamp - pd.DateOffset(months=period)
    if period_type == "year":
        return end_timestamp - pd.DateOffset(years=period)
    return None


def _approx_frequency_seconds(offset) -> float | None:
    if offset is None:
        return None
    if isinstance(offset, pd.offsets.Week):
        return float(offset.n * 7 * 24 * 60 * 60)
    if isinstance(offset, pd.offsets.MonthBegin):
        return float(offset.n * 30 * 24 * 60 * 60)

    try:
        return float(offset.nanos) / 1_000_000_000
    except ValueError:
        return None


def _infer_offset(index: pd.DatetimeIndex):
    if len(index) >= 3:
        inferred = pd.infer_freq(index)
        if inferred is not None:
            return pd.tseries.frequencies.to_offset(inferred)
    if len(index) >= 2:
        return pd.tseries.frequencies.to_offset(index[1] - index[0])
    return None


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

    def _filter_history_frame(
        self,
        symbol: str,
        *,
        period_type=None,
        period=None,
        start_datetime: datetime.datetime | int | float | None = None,
        end_datetime: datetime.datetime | int | float | None = None,
    ) -> pd.DataFrame:
        """Slice history to the visible data up to the current simulation step."""
        target_df = self._resolve_dataframe(symbol).iloc[: self.current_step + 1]
        if target_df.empty:
            return target_df

        if (
            start_datetime is None
            and end_datetime is None
            and period_type is None
            and period is None
        ):
            return target_df.iloc[max(0, len(target_df) - _HISTORY_LOOKBACK) :]

        end_ts = _normalize_timestamp(end_datetime, target_df.index)
        if end_ts is None:
            end_ts = target_df.index[-1]

        start_ts = _normalize_timestamp(start_datetime, target_df.index)
        if start_ts is None:
            start_ts = _resolve_period_start(period_type, period, end_ts)

        subset = target_df
        if start_ts is not None:
            subset = subset[subset.index >= start_ts]
        if end_ts is not None:
            subset = subset[subset.index <= end_ts]
        return subset

    def _resample_history_frame(
        self, subset: pd.DataFrame, *, frequency_type=None, frequency=None
    ) -> pd.DataFrame:
        """Resample to coarser intervals when the request asks for them."""
        frequency_type = _coerce_history_arg(frequency_type)
        frequency = int(frequency) if frequency is not None else None
        if subset.empty or frequency_type is None or frequency is None:
            return subset

        rule = _RESAMPLE_RULES.get((frequency_type, frequency))
        if rule is None:
            logger.warning(
                "Unsupported price history frequency request frequency_type=%s frequency=%s",
                frequency_type,
                frequency,
            )
            return subset

        source_offset = _infer_offset(subset.index)
        target_offset = pd.tseries.frequencies.to_offset(rule)
        source_seconds = _approx_frequency_seconds(source_offset)
        target_seconds = _approx_frequency_seconds(target_offset)

        if source_seconds is not None and target_seconds is not None:
            if target_seconds < source_seconds:
                logger.warning(
                    "Requested %s bars from %s source data; returning source frequency.",
                    rule,
                    getattr(source_offset, "freqstr", source_offset),
                )
                return subset
            if target_seconds == source_seconds:
                return subset

        agg = {
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum",
        }
        if "AdjClose" in subset.columns:
            agg["AdjClose"] = "last"
        if "Volatility" in subset.columns:
            agg["Volatility"] = "mean"

        resampled = subset.resample(rule, label="left", closed="left").agg(agg)
        resampled.dropna(subset=["Open", "High", "Low", "Close"], inplace=True)
        return resampled

    def get_price_history_data(
        self,
        symbol: str,
        *,
        period_type=None,
        period=None,
        frequency_type=None,
        frequency=None,
        start_datetime: datetime.datetime | int | float | None = None,
        end_datetime: datetime.datetime | int | float | None = None,
    ) -> list[dict]:
        """
        Get historical candles up to current step.

        Args:
            symbol: Ticker symbol.

        Returns:
            List[Dict]: List of candle dicts.
        """
        subset = self._filter_history_frame(
            symbol,
            period_type=period_type,
            period=period,
            start_datetime=start_datetime,
            end_datetime=end_datetime,
        )
        subset = self._resample_history_frame(
            subset, frequency_type=frequency_type, frequency=frequency
        )

        # schwab-py returns raw Close prices, not adjusted.
        col_close = "Close"

        # Vectorized: build candles without iterrows()
        candles = [
            {
                "open": float(o),
                "high": float(h),
                "low": float(lo),
                "close": float(c),
                "volume": int(v),
                "datetime": int(ts.timestamp() * 1000),
            }
            for ts, o, h, lo, c, v in zip(
                subset.index,
                subset["Open"].values,
                subset["High"].values,
                subset["Low"].values,
                subset[col_close].values,
                subset["Volume"].values,
            )
        ]
        return candles
