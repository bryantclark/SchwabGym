"""
SchwabGym Core Client
=====================

Simulator that replicates the Charles Schwab Trader API.
Now refactored to delegate logic to specialized components.

Author: Bryant Clark
Repository: https://github.com/bryantclark/SchwabGym
License: MIT
"""

import datetime
import hashlib
import logging
import re
from typing import Any

from schwabgym.account import Account
from schwabgym.order_manager import OrderManager
from schwabgym.orders import MockResponse, OptionSymbol
from schwabgym.physics import ExecutionEngine, RealisticExecutionEngine
from schwabgym.prices import PriceEngine
from schwabgym.streamer import MockStreamer

# Configure logging
logger = logging.getLogger(__name__)


def _coerce_enum_value(value):
    if hasattr(value, "value"):
        return value.value
    return value


def _normalize_datetime_filter(value) -> datetime.datetime | None:
    if value is None:
        return None

    value = _coerce_enum_value(value)

    if isinstance(value, datetime.datetime):
        if value.tzinfo is not None:
            return value.astimezone(datetime.timezone.utc).replace(tzinfo=None)
        return value

    if isinstance(value, datetime.date):
        return datetime.datetime.combine(value, datetime.time.min)

    if isinstance(value, int | float):
        ts = value / 1000 if abs(value) >= 10_000_000_000 else value
        return datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc).replace(
            tzinfo=None
        )

    if isinstance(value, str):
        normalized = value.replace("Z", "+00:00")
        parsed = datetime.datetime.fromisoformat(normalized)
        if parsed.tzinfo is not None:
            return parsed.astimezone(datetime.timezone.utc).replace(tzinfo=None)
        return parsed

    raise TypeError(f"Unsupported datetime filter value: {value!r}")


def _order_symbol(order: dict[str, Any]) -> str | None:
    legs = order.get("orderLegCollection", [])
    if not legs:
        return None
    symbol = legs[0].get("instrument", {}).get("symbol")
    return str(symbol) if symbol is not None else None


def _materialize_order_spec(order_spec: Any) -> Any:
    """Convert builder-like order specs into plain dict payloads."""
    if hasattr(order_spec, "build") and callable(order_spec.build):
        return order_spec.build()
    return order_spec


def _normalize_date_filter(value) -> datetime.date | None:
    """Normalize Schwab-style date inputs into ``datetime.date``."""
    normalized = _normalize_datetime_filter(value)
    if normalized is None:
        return None
    return normalized.date()


def _synthetic_cusip(symbol: str) -> str:
    """Generate a deterministic pseudo-CUSIP for simulator instruments."""
    digits = "".join(
        ch for ch in hashlib.sha256(symbol.encode()).hexdigest() if ch.isdigit()
    )
    if len(digits) < 9:
        digits = (digits + "0" * 9)[:9]
    return digits[:9]


def _option_strike_increment(price: float) -> float:
    """Choose a reasonable synthetic strike ladder increment."""
    if price < 25:
        return 1.0
    if price < 100:
        return 2.5
    if price < 250:
        return 5.0
    return 10.0


def _next_friday_expirations(
    anchor_date: datetime.date, *, count: int = 6
) -> list[datetime.date]:
    """Generate the next ``count`` Friday expirations including same-day Friday."""
    expirations: list[datetime.date] = []
    cursor = anchor_date

    while len(expirations) < count:
        days_until_friday = (4 - cursor.weekday()) % 7
        expiry = cursor + datetime.timedelta(days=days_until_friday)
        if expiry >= anchor_date:
            expirations.append(expiry)
        cursor = expiry + datetime.timedelta(days=7)

    return expirations


def _iso_session(
    date_value: datetime.date,
    start_hour: int,
    start_minute: int,
    end_hour: int,
    end_minute: int,
) -> dict[str, str]:
    """Build a Schwab-style session window payload."""
    start = datetime.datetime.combine(
        date_value, datetime.time(start_hour, start_minute)
    )
    end = datetime.datetime.combine(date_value, datetime.time(end_hour, end_minute))
    return {"start": start.isoformat(), "end": end.isoformat()}


class MockClient:
    """
    Simulator of schwab.client.Client.

    Delegates to:
    - PriceEngine: Market data
    - Account: Funds and positions
    - OrderManager: Order execution
    """

    def __init__(
        self,
        market_data_df=None,
        initial_cash: float = 25000.0,
        execution_engine: ExecutionEngine | None = None,
        # schwab-py compat params (accepted but unused in simulation)
        app_key: str | None = None,
        app_secret: str | None = None,
        callback_url: str | None = None,
        token_path: str | None = None,
        tokens_file: str | None = None,
        timeout: int | None = None,
        verbose: bool = False,
        **kwargs,
    ):
        """
        Initialize the MockClient.

        Args:
            market_data_df: Historical OHLCV data (DataFrame or dict of DataFrames).
            initial_cash: Starting account balance.
            execution_engine: Physics engine for order execution.
            app_key: Ignored (schwab-py signature compatibility).
            app_secret: Ignored (schwab-py signature compatibility).
            callback_url: Ignored (schwab-py signature compatibility).
            token_path: Ignored (schwab-py signature compatibility).
            tokens_file: Ignored (schwab-py signature compatibility).
            timeout: Ignored (schwab-py signature compatibility).
            verbose: Ignored (schwab-py signature compatibility).
            **kwargs: Additional options (latency_mode, market_data).
        """
        # Handle case where app_key is passed positionally as the first argument
        if isinstance(market_data_df, str):
            logger.warning(
                "MockClient received string for market_data_df. Assuming it is app_key."
            )
            market_data_df = kwargs.get("market_data")

        if market_data_df is None:
            # Check kwargs for 'market_data'
            market_data_df = kwargs.get("market_data")

        if market_data_df is None:
            from schwabgym.data import generate_dummy_data

            logger.warning("No market data provided. Generating DUMMY data.")
            market_data_df = generate_dummy_data()

        # Initialize components
        self.price_engine = PriceEngine(market_data_df)
        self.account = Account(initial_cash=initial_cash)
        self._cusip_symbol_cache: dict[str, str] = {}

        if execution_engine is None:
            execution_engine = RealisticExecutionEngine()
            logger.info("Using RealisticExecutionEngine")

        self.execution_engine = execution_engine
        if hasattr(self.execution_engine, "reset_episode"):
            self.execution_engine.reset_episode()

        # Default to latency mode unless explicitly disabled
        latency_mode = kwargs.get("latency_mode", True)

        self.order_manager = OrderManager(
            account=self.account,
            price_engine=self.price_engine,
            execution_engine=self.execution_engine,
            latency_mode=latency_mode,
            account_hash=self.account_hash,
        )

        # Streamer
        self.streamer = MockStreamer(self)

        logger.info(f"MockClient initialized: ${initial_cash:,.2f} starting capital")

    # ==================== PROPERTIES FOR COMPATIBILITY ====================

    @property
    def df(self):
        return self.price_engine.df

    @property
    def current_step(self):
        return self.price_engine.current_step

    @property
    def max_steps(self):
        return self.price_engine.max_steps

    @property
    def cash(self):
        return self.account.cash

    @cash.setter
    def cash(self, value):
        self.account.cash = value

    @property
    def positions(self):
        return self.account.positions

    @property
    def orders(self):
        return self.order_manager.orders

    @property
    def working_orders(self):
        return self.order_manager.working_orders

    @property
    def initial_cash(self):
        return self.account.initial_cash

    @property
    def day_trades(self):
        return self.account.day_trades

    @property
    def account_hash(self):
        return hashlib.sha256(self.account_number.encode()).hexdigest()[:12].upper()

    @property
    def account_number(self):
        return self.account.account_number

    # ==================== SIMULATION CONTROL ====================

    def advance_time(self) -> bool:
        """Advance simulator by one time step."""
        if not self.price_engine.advance_time():
            return False

        if hasattr(self.execution_engine, "prepare_step"):
            self.execution_engine.prepare_step()
        self.order_manager.process_working_orders()

        # Check for new day to reset day trading counters
        if self.current_step > 0:
            curr_date = self._get_current_time().date()
            prev_date = self.price_engine.df.index[self.current_step - 1].date()
            if curr_date > prev_date:
                self.account.on_new_day()

        return True

    def reset(self, initial_cash: float | None = None) -> None:
        """Reset simulator."""
        self.price_engine.reset()
        self.account.reset(initial_cash)
        self.order_manager.reset()
        if hasattr(self.execution_engine, "reset_episode"):
            self.execution_engine.reset_episode()
        logger.info("Simulator reset to initial state")

    # ==================== INTERNAL HELPERS ====================
    # Kept for compatibility with existing tests that access them directly

    def _get_current_time(self) -> datetime.datetime:
        return self.price_engine.get_current_time()

    def _get_current_raw_price(self, symbol: str) -> float:
        return self.price_engine.get_current_price(symbol)

    def _calculate_equity(self) -> float:
        return self.account.calculate_equity(self.price_engine.get_current_price)

    def _calculate_buying_power(self, equity) -> float:
        return self.account.calculate_buying_power(equity)

    def _process_working_orders(self):
        self.order_manager.process_working_orders()

    def _available_symbols(self) -> list[str]:
        """Return the simulator's known symbols."""
        symbols = list(self.price_engine.data.keys())
        if len(symbols) == 1 and symbols[0] == "DEFAULT":
            return []
        return symbols

    def _build_instrument_payload(
        self, symbol: str, *, include_fundamental: bool = False
    ) -> dict[str, Any]:
        """Build a deterministic instrument payload from simulator data."""
        df = self.price_engine._resolve_dataframe(symbol)
        visible = df.iloc[: self.current_step + 1]
        quote = self.get_quote(symbol).json()[symbol]["quote"]

        payload: dict[str, Any] = {
            "symbol": symbol,
            "description": f"{symbol} simulated equity",
            "exchange": "SIM",
            "exchangeName": "Simulator Exchange",
            "cusip": _synthetic_cusip(symbol),
            "assetType": "EQUITY",
        }
        self._cusip_symbol_cache[payload["cusip"]] = symbol

        if include_fundamental:
            current_price = float(quote["lastPrice"])
            avg_volume = int(visible["Volume"].mean()) if not visible.empty else 0
            trailing_pe = round(
                max(current_price / max(current_price * 0.06, 0.01), 1.0), 2
            )
            shares_outstanding = 1_000_000

            payload["fundamental"] = {
                "symbol": symbol,
                "high52": float(df["High"].max()),
                "low52": float(df["Low"].min()),
                "dividendAmount": 0.0,
                "dividendYield": 0.0,
                "peRatio": trailing_pe,
                "marketCap": round(current_price * shares_outstanding, 2),
                "avg10DaysVolume": avg_volume,
                "avg1YearVolume": avg_volume,
                "vol1DayAvg": int(quote["totalVolume"]),
                "sharesOutstanding": shares_outstanding,
            }

        return payload

    def _find_symbol_by_cusip(self, cusip: str) -> str | None:
        """Find the simulator symbol matching a synthetic CUSIP."""
        cached = self._cusip_symbol_cache.get(cusip)
        if cached is not None:
            return cached
        for symbol in self._available_symbols():
            if _synthetic_cusip(symbol) == cusip:
                return symbol
        return None

    # ==================== SCHWAB API INTERFACE ====================

    def get_account_numbers(self) -> MockResponse:
        return MockResponse(
            [{"accountNumber": self.account_number, "hashValue": self.account_hash}]
        )

    def get_account(self, account_hash: str, *, fields=None) -> MockResponse:
        if account_hash != self.account_hash:
            return MockResponse({"error": "Unauthorized"}, 401)

        # Calculate current values
        equity = self._calculate_equity()
        buying_power = self._calculate_buying_power(equity)
        long_mv, short_mv = self.account.calculate_market_value(
            self.price_engine.get_current_price
        )

        # Build positions array
        position_list = []
        for sym, pos in self.positions.items():
            current_price = self._get_current_raw_price(sym)
            qty = pos["quantity"]
            mv = qty * current_price

            position_list.append(
                {
                    "instrument": {"symbol": sym, "assetType": pos["assetType"]},
                    "longQuantity": qty if qty > 0 else 0,
                    "shortQuantity": abs(qty) if qty < 0 else 0,
                    "averagePrice": pos["avgPrice"],
                    "marketValue": mv,
                }
            )

        return MockResponse(
            {
                "securitiesAccount": {
                    "type": "MARGIN",
                    "accountNumber": self.account_number,
                    "roundTrips": len(self.account.day_trades),
                    "isDayTrader": self.account.is_pdt_flagged,
                    "currentBalances": {
                        "liquidationValue": equity,
                        "cashBalance": self.cash,
                        "buyingPower": buying_power,
                        "availableFunds": self.cash,
                        "longMarketValue": long_mv,
                        "shortMarketValue": short_mv,
                    },
                    "positions": position_list,
                }
            }
        )

    def get_accounts(self, *, fields=None) -> MockResponse:
        """Get all linked accounts (schwab-py parity: only one simulated account)."""
        account_resp = self.get_account(self.account_hash, fields=fields)
        return MockResponse([account_resp.json()])

    def get_quote(self, symbol: str, *, fields=None) -> MockResponse:
        """Get quote for a single symbol."""
        data = self.price_engine.get_quotes_data([symbol])
        return MockResponse(data)

    def get_quotes(
        self, symbols: str | list[str], *, fields=None, indicative=None
    ) -> MockResponse:
        if isinstance(symbols, str):
            symbols = [symbols]

        data = self.price_engine.get_quotes_data(symbols)
        return MockResponse(data)

    def get_price_history(
        self,
        symbol: str,
        *,
        period_type=None,
        period=None,
        frequency_type=None,
        frequency=None,
        start_datetime: datetime.datetime | None = None,
        end_datetime: datetime.datetime | None = None,
        need_extended_hours_data=None,
        need_previous_close=None,
    ) -> MockResponse:
        candles = self.price_engine.get_price_history_data(
            symbol,
            period_type=period_type,
            period=period,
            frequency_type=frequency_type,
            frequency=frequency,
            start_datetime=start_datetime,
            end_datetime=end_datetime,
        )
        logger.debug(f"Returned {len(candles)} candles for {symbol}")
        return MockResponse({"candles": candles, "symbol": symbol})

    def place_order(self, account_hash: str, order_spec: Any) -> MockResponse:
        if account_hash != self.account_hash:
            return MockResponse({"error": "Unauthorized"}, 401)

        # Check for Pattern Day Trader restriction
        if self.account.is_pdt_flagged:
            return MockResponse(
                {"error": "Order Rejected: Pattern Day Trader Restriction"}, 403
            )

        return self.order_manager.place_order(order_spec)

    def preview_order(self, account_hash: str, order_spec: Any) -> MockResponse:
        """Preview order and normalize builder objects into plain payloads."""
        if account_hash != self.account_hash:
            return MockResponse({"error": "Unauthorized"}, 401)
        return MockResponse({"orderStrategy": _materialize_order_spec(order_spec)})

    def cancel_order(self, order_id: int, account_hash: str) -> MockResponse:
        if account_hash != self.account_hash:
            return MockResponse({"error": "Unauthorized"}, 401)
        return self.order_manager.cancel_order(order_id)

    def replace_order(
        self, account_hash: str, order_id: int, order_spec: Any
    ) -> MockResponse:
        if account_hash != self.account_hash:
            return MockResponse({"error": "Unauthorized"}, 401)
        return self.order_manager.replace_order(order_id, order_spec)

    def get_order(self, order_id: int, account_hash: str) -> MockResponse:
        if account_hash != self.account_hash:
            return MockResponse({"error": "Unauthorized"}, 401)

        if order_id in self.orders:
            return MockResponse(self.orders[order_id])

        return MockResponse({"error": "Order not found"}, 404)

    def get_orders_for_account(
        self,
        account_hash: str,
        *,
        max_results: int | None = None,
        from_entered_datetime: str | None = None,
        to_entered_datetime: str | None = None,
        status: str | None = None,
    ) -> MockResponse:
        if account_hash != self.account_hash:
            return MockResponse({"error": "Unauthorized"}, 401)

        from_dt = _normalize_datetime_filter(from_entered_datetime)
        to_dt = _normalize_datetime_filter(to_entered_datetime)
        desired_status = _coerce_enum_value(status)
        result_orders = []

        for order in self.orders.values():
            if desired_status and order["status"] != desired_status:
                continue

            entered_dt = _normalize_datetime_filter(order.get("enteredTime"))
            if from_dt is not None and entered_dt is not None and entered_dt < from_dt:
                continue
            if to_dt is not None and entered_dt is not None and entered_dt > to_dt:
                continue

            result_orders.append(order)

        result_orders.sort(key=lambda x: x["orderId"])

        if max_results:
            result_orders = result_orders[-max_results:]

        return MockResponse(result_orders)

    def get_orders_for_all_linked_accounts(
        self,
        *,
        max_results: int | None = None,
        from_entered_datetime: str | None = None,
        to_entered_datetime: str | None = None,
        status: str | None = None,
    ) -> MockResponse:
        """Get orders across all linked accounts (single account in sim)."""
        return self.get_orders_for_account(
            self.account_hash,
            from_entered_datetime=from_entered_datetime,
            to_entered_datetime=to_entered_datetime,
            status=status,
            max_results=max_results,
        )

    def get_transaction(self, account_hash: str, transaction_id: int) -> MockResponse:
        """Get a single transaction by ID."""
        if account_hash != self.account_hash:
            return MockResponse({"error": "Unauthorized"}, 401)

        if transaction_id in self.orders:
            order = self.orders[transaction_id]
            if order["status"] == "FILLED":
                return MockResponse(order)

        return MockResponse({"error": "Transaction not found"}, 404)

    def get_transactions(
        self,
        account_hash: str,
        *,
        start_date: datetime.datetime | None = None,
        end_date: datetime.datetime | None = None,
        transaction_types=None,
        symbol: str | None = None,
    ) -> MockResponse:
        """Get simulated transaction history from filled orders."""
        if account_hash != self.account_hash:
            return MockResponse({"error": "Unauthorized"}, 401)

        start_dt = _normalize_datetime_filter(start_date)
        end_dt = _normalize_datetime_filter(end_date)

        allowed_types = None
        if transaction_types is not None:
            raw_types = (
                transaction_types
                if isinstance(transaction_types, list | tuple | set)
                else [transaction_types]
            )
            allowed_types = {str(_coerce_enum_value(t)).upper() for t in raw_types}

        transactions = []
        for order in self.orders.values():
            if order["status"] != "FILLED":
                continue
            if symbol is not None and _order_symbol(order) != symbol:
                continue
            if allowed_types is not None and "TRADE" not in allowed_types:
                continue

            entered_dt = _normalize_datetime_filter(order.get("enteredTime"))
            if (
                start_dt is not None
                and entered_dt is not None
                and entered_dt < start_dt
            ):
                continue
            if end_dt is not None and entered_dt is not None and entered_dt > end_dt:
                continue

            transactions.append(order)

        return MockResponse(transactions)

    def get_user_preferences(self) -> MockResponse:
        """Get user preferences (simulated)."""
        return MockResponse(
            {
                "accounts": [
                    {
                        "accountNumber": self.account_number,
                        "primaryAccount": True,
                        "type": "MARGIN",
                    }
                ],
                "streamerInfo": [
                    {
                        "streamerSocketUrl": "wss://simulated.schwabgym.local/ws",
                        "schwabClientCustomerId": "SIM_CUSTOMER",
                        "schwabClientCorrelId": "SIM_CORREL",
                    }
                ],
            }
        )

    # ==================== SYNTHETIC MARKET DATA ====================
    # These methods mirror schwab.client.Client, but they synthesize responses
    # from the simulator's loaded OHLCV data instead of calling Schwab.

    def get_option_chain(
        self,
        symbol: str,
        *,
        contract_type=None,
        strike_count=None,
        include_underlying_quote=None,
        strategy=None,
        interval=None,
        strike=None,
        strike_range=None,
        from_date=None,
        to_date=None,
        volatility=None,
        underlying_price=None,
        interest_rate=None,
        days_to_expiration=None,
        exp_month=None,
        option_type=None,
        entitlement=None,
    ) -> MockResponse:
        """Get a deterministic synthetic option chain for the loaded symbol."""
        try:
            df = self.price_engine._resolve_dataframe(symbol)
        except KeyError:
            return MockResponse({"error": f"Unknown symbol: {symbol}"}, 404)

        current_quote = self.get_quote(symbol).json()[symbol]["quote"]
        current_date = self._get_current_time().date()
        current_price = float(
            underlying_price
            if underlying_price is not None
            else current_quote["lastPrice"]
        )

        contract_type_value = (_coerce_enum_value(contract_type) or "ALL").upper()
        strategy_value = (_coerce_enum_value(strategy) or "SINGLE").upper()
        strike_range_value = (_coerce_enum_value(strike_range) or "ALL").upper()
        exp_month_value = (_coerce_enum_value(exp_month) or "ALL").upper()
        option_type_value = (_coerce_enum_value(option_type) or "ALL").upper()
        entitlement_value = _coerce_enum_value(entitlement)

        from_date_value = _normalize_date_filter(from_date)
        to_date_value = _normalize_date_filter(to_date)

        row = df.iloc[self.current_step]
        raw_volatility = float(
            volatility if volatility is not None else row.get("Volatility", 0.2)
        )
        vol_decimal = raw_volatility if raw_volatility <= 1 else raw_volatility / 100.0
        theoretical_volatility = round(vol_decimal * 100.0, 2)
        interest_rate_value = float(interest_rate) if interest_rate is not None else 0.0
        interval_value = (
            float(interval)
            if interval is not None
            else _option_strike_increment(current_price)
        )

        if strike is not None:
            strikes = [round(float(strike), 2)]
        else:
            strike_count_value = max(
                int(strike_count) if strike_count is not None else 2, 1
            )
            at_the_money = round(current_price / interval_value) * interval_value
            strikes = [
                round(at_the_money + (offset * interval_value), 2)
                for offset in range(-strike_count_value, strike_count_value + 1)
            ]
            strikes = sorted({value for value in strikes if value > 0})

        expirations = _next_friday_expirations(current_date)
        if from_date_value is not None:
            expirations = [
                expiry for expiry in expirations if expiry >= from_date_value
            ]
        if to_date_value is not None:
            expirations = [expiry for expiry in expirations if expiry <= to_date_value]
        if exp_month_value != "ALL":
            expirations = [
                expiry
                for expiry in expirations
                if expiry.strftime("%b").upper() == exp_month_value
            ]
        if days_to_expiration is not None:
            max_days = int(days_to_expiration)
            expirations = [
                expiry
                for expiry in expirations
                if (expiry - current_date).days <= max_days
            ]

        def include_strike(strike_price: float, put_call: str) -> bool:
            if strike_range_value in {"ALL", ""}:
                return True
            if strike_range_value in {"SAK", "STRIKES_ABOVE_MARKET"}:
                return strike_price > current_price
            if strike_range_value in {"SBK", "STRIKES_BELOW_MARKET"}:
                return strike_price < current_price
            if strike_range_value in {"SNK", "NTM", "STRIKES_NEAR_MARKET"}:
                return abs(strike_price - current_price) <= interval_value * 2
            if strike_range_value in {"ITM", "IN_THE_MONEY"}:
                return (
                    strike_price < current_price
                    if put_call == "CALL"
                    else strike_price > current_price
                )
            if strike_range_value in {"OTM", "OUT_OF_THE_MONEY"}:
                return (
                    strike_price > current_price
                    if put_call == "CALL"
                    else strike_price < current_price
                )
            return True

        quote_time_ms = int(self._get_current_time().timestamp() * 1000)
        call_map: dict[str, dict[str, list[dict[str, Any]]]] = {}
        put_map: dict[str, dict[str, list[dict[str, Any]]]] = {}

        def make_contract(
            expiry: datetime.date, strike_price: float, put_call: str
        ) -> dict[str, Any]:
            days = max((expiry - current_date).days, 0)
            years = max(days, 1) / 365.0
            intrinsic = (
                max(current_price - strike_price, 0.0)
                if put_call == "CALL"
                else max(strike_price - current_price, 0.0)
            )
            moneyness = abs(strike_price - current_price) / max(current_price, 1.0)
            extrinsic = max(
                0.05,
                current_price
                * vol_decimal
                * (years**0.5)
                * max(0.2, 1.0 - moneyness)
                * 0.15,
            )
            mark = round(intrinsic + extrinsic, 2)
            bid = round(max(mark - 0.05, 0.01), 2)
            ask = round(mark + 0.05, 2)
            delta_anchor = max(
                -1.0,
                min(
                    1.0, (current_price - strike_price) / max(current_price * 0.1, 1.0)
                ),
            )
            call_delta = round(0.5 + (0.5 * delta_anchor), 4)
            delta = call_delta if put_call == "CALL" else round(call_delta - 1.0, 4)
            contract_type_char = "C" if put_call == "CALL" else "P"
            option_symbol = OptionSymbol(
                symbol,
                expiry,
                contract_type_char,
                f"{strike_price:.3f}".rstrip("0").rstrip("."),
            ).build()

            return {
                "putCall": put_call,
                "symbol": option_symbol,
                "description": f"{symbol} {expiry.isoformat()} {strike_price:.2f} {put_call}",
                "exchangeName": "SIM",
                "bid": bid,
                "ask": ask,
                "last": mark,
                "mark": mark,
                "bidSize": 10,
                "askSize": 10,
                "lastSize": 1,
                "highPrice": round(mark * 1.1, 2),
                "lowPrice": round(max(mark * 0.9, 0.01), 2),
                "openPrice": mark,
                "closePrice": mark,
                "totalVolume": 0,
                "tradeDate": None,
                "tradeTimeInLong": quote_time_ms,
                "quoteTimeInLong": quote_time_ms,
                "netChange": 0.0,
                "volatility": theoretical_volatility,
                "delta": delta,
                "gamma": round(max(0.01, 0.08 * (1.0 - min(moneyness, 0.9))), 4),
                "theta": round(-extrinsic / max(days, 1), 4),
                "vega": round(extrinsic * 0.1, 4),
                "rho": round(interest_rate_value * years * 0.01, 4),
                "openInterest": 100,
                "timeValue": round(extrinsic, 4),
                "theoreticalOptionValue": mark,
                "theoreticalVolatility": theoretical_volatility,
                "optionDeliverablesList": [],
                "strikePrice": strike_price,
                "expirationDate": int(
                    datetime.datetime.combine(expiry, datetime.time()).timestamp()
                    * 1000
                ),
                "daysToExpiration": days,
                "expirationType": "W",
                "multiplier": 100,
                "settlementType": "P",
                "deliverableNote": "",
                "isIndexOption": False,
                "percentChange": 0.0,
                "markChange": 0.0,
                "markPercentChange": 0.0,
                "intrinsicValue": round(intrinsic, 4),
                "extrinsicValue": round(extrinsic, 4),
                "optionRoot": symbol,
                "exerciseType": "A",
                "high52Week": round(mark * 2, 2),
                "low52Week": 0.01,
                "pennyPilot": True,
                "inTheMoney": intrinsic > 0,
                "mini": False,
                "nonStandard": False,
            }

        if option_type_value not in {"ALL", "S", "STANDARD"}:
            expirations = []

        for expiry in expirations:
            days = max((expiry - current_date).days, 0)
            expiry_key = f"{expiry.isoformat()}:{days}"
            for strike_price in strikes:
                strike_key = f"{strike_price:.1f}"

                if contract_type_value in {"ALL", "CALL"} and include_strike(
                    strike_price, "CALL"
                ):
                    call_map.setdefault(expiry_key, {}).setdefault(
                        strike_key, []
                    ).append(make_contract(expiry, strike_price, "CALL"))

                if contract_type_value in {"ALL", "PUT"} and include_strike(
                    strike_price, "PUT"
                ):
                    put_map.setdefault(expiry_key, {}).setdefault(
                        strike_key, []
                    ).append(make_contract(expiry, strike_price, "PUT"))

        number_of_contracts = sum(
            len(contracts)
            for expiry_map in (call_map, put_map)
            for strike_map in expiry_map.values()
            for contracts in strike_map.values()
        )

        response: dict[str, Any] = {
            "symbol": symbol,
            "status": "SUCCESS",
            "strategy": strategy_value,
            "interval": interval_value,
            "isDelayed": False,
            "isIndex": False,
            "interestRate": interest_rate_value,
            "underlyingPrice": round(current_price, 4),
            "volatility": theoretical_volatility,
            "daysToExpiration": days_to_expiration,
            "numberOfContracts": number_of_contracts,
            "assetMainType": "EQUITY",
            "assetSubType": "COE",
            "callExpDateMap": call_map,
            "putExpDateMap": put_map,
            "entitlement": entitlement_value,
        }
        if include_underlying_quote:
            response["underlying"] = current_quote

        return MockResponse(response)

    def get_option_expiration_chain(self, symbol: str) -> MockResponse:
        """Return synthetic expiration dates for the requested symbol."""
        try:
            self.price_engine._resolve_dataframe(symbol)
        except KeyError:
            return MockResponse({"error": f"Unknown symbol: {symbol}"}, 404)

        current_date = self._get_current_time().date()
        expiration_list = [
            {
                "expirationDate": expiry.isoformat(),
                "daysToExpiration": (expiry - current_date).days,
                "expirationType": "W",
                "settlementType": "P",
                "optionRoots": symbol,
                "standard": True,
            }
            for expiry in _next_friday_expirations(current_date)
        ]

        return MockResponse(
            {
                "symbol": symbol,
                "status": "SUCCESS",
                "expirationList": expiration_list,
            }
        )

    def get_movers(
        self, index: str, *, sort_order=None, frequency=None
    ) -> MockResponse:
        """Get synthetic movers across the simulator's loaded symbols."""
        index_value = (_coerce_enum_value(index) or "").upper()
        sort_order_value = (
            _coerce_enum_value(sort_order) or "PERCENT_CHANGE_UP"
        ).upper()
        frequency_value = int(_coerce_enum_value(frequency) or 0)

        if index_value in {"OPTION_ALL", "OPTION_CALL", "OPTION_PUT"}:
            return MockResponse([])

        symbols = self._available_symbols() or [self.price_engine.main_symbol]
        movers = []

        for symbol in symbols:
            df = self.price_engine._resolve_dataframe(symbol)
            row = df.iloc[self.current_step]
            prev_close = (
                float(df.iloc[self.current_step - 1]["Close"])
                if self.current_step > 0
                else float(row["Open"])
            )
            last_price = float(row["Close"])
            change = last_price - prev_close
            percent_change = 0.0 if prev_close == 0 else (change / prev_close) * 100.0

            if abs(percent_change) < frequency_value:
                continue

            movers.append(
                {
                    "symbol": symbol,
                    "description": f"{symbol} simulated equity",
                    "direction": "up" if change >= 0 else "down",
                    "last": round(last_price, 4),
                    "change": round(change, 4),
                    "percentChange": round(percent_change, 4),
                    "totalVolume": int(row["Volume"]),
                }
            )

        if sort_order_value in {"VOLUME", "TRADES"}:
            movers.sort(key=lambda item: item["totalVolume"], reverse=True)
        elif sort_order_value == "PERCENT_CHANGE_DOWN":
            movers.sort(key=lambda item: item["percentChange"])
        else:
            movers.sort(key=lambda item: item["percentChange"], reverse=True)

        return MockResponse(movers[:10])

    def get_market_hours(self, markets, *, date=None) -> MockResponse:
        """Return deterministic market-hour schedules for supported market types."""
        market_values = (
            markets if isinstance(markets, list | tuple | set) else [markets]
        )
        normalized_markets = [
            _coerce_enum_value(value).lower() for value in market_values
        ]
        requested_date = _normalize_date_filter(date) or self._get_current_time().date()
        is_weekday = requested_date.weekday() < 5
        current_time = self._get_current_time().time()

        market_defs = {
            "equity": (
                "EQ",
                [_iso_session(requested_date, 7, 0, 9, 30)],
                [_iso_session(requested_date, 9, 30, 16, 0)],
                [_iso_session(requested_date, 16, 0, 20, 0)],
            ),
            "option": (
                "OP",
                [_iso_session(requested_date, 7, 0, 9, 30)],
                [_iso_session(requested_date, 9, 30, 16, 0)],
                [],
            ),
            "bond": ("BON", [], [_iso_session(requested_date, 8, 0, 17, 0)], []),
            "forex": ("FX", [], [_iso_session(requested_date, 0, 0, 23, 59)], []),
            "future": ("FUT", [], [_iso_session(requested_date, 0, 0, 23, 59)], []),
        }

        response: dict[str, Any] = {}
        for market in normalized_markets:
            product_code, pre_market, regular_market, post_market = market_defs.get(
                market,
                ("GEN", [], [_iso_session(requested_date, 9, 30, 16, 0)], []),
            )
            is_open = is_weekday and (
                datetime.time(9, 30) <= current_time <= datetime.time(16, 0)
                if market in {"equity", "option", "bond"}
                else True
            )
            response[market] = {
                product_code: {
                    "date": requested_date.isoformat(),
                    "marketType": market.upper(),
                    "exchange": "SIM",
                    "category": market,
                    "product": product_code,
                    "productName": f"Simulated {market.title()} Market",
                    "isOpen": bool(is_weekday and is_open),
                    "sessionHours": {
                        "preMarket": pre_market,
                        "regularMarket": regular_market if is_weekday else [],
                        "postMarket": post_market,
                    },
                }
            }

        return MockResponse(response)

    def get_instruments(self, symbols, projection) -> MockResponse:
        """Search the simulator's loaded instruments."""
        projection_value = (_coerce_enum_value(projection) or "").lower()

        symbol_list = [symbols] if isinstance(symbols, str) else list(symbols)
        known_symbols = self._available_symbols()
        candidates = known_symbols or list(dict.fromkeys(symbol_list))

        def description_for(symbol: str) -> str:
            return f"{symbol} simulated equity"

        matches: list[str] = []
        if projection_value in {"symbol-search", "fundamental"}:
            wanted = {value.upper() for value in symbol_list}
            matches = [symbol for symbol in candidates if symbol.upper() in wanted]
        elif projection_value == "symbol-regex":
            pattern = re.compile(symbol_list[0], re.IGNORECASE)
            matches = [symbol for symbol in candidates if pattern.search(symbol)]
        elif projection_value == "desc-search":
            query = symbol_list[0].lower()
            matches = [
                symbol
                for symbol in candidates
                if query in description_for(symbol).lower()
            ]
        elif projection_value == "desc-regex":
            pattern = re.compile(symbol_list[0], re.IGNORECASE)
            matches = [
                symbol
                for symbol in candidates
                if pattern.search(description_for(symbol))
            ]
        elif projection_value == "search":
            pattern = re.compile(symbol_list[0], re.IGNORECASE)
            matches = [
                symbol
                for symbol in candidates
                if pattern.search(symbol) or pattern.search(description_for(symbol))
            ]
        else:
            return MockResponse(
                {"error": f"Unsupported projection: {projection_value}"},
                400,
            )

        include_fundamental = projection_value == "fundamental"
        payload = {
            symbol: self._build_instrument_payload(
                symbol,
                include_fundamental=include_fundamental,
            )
            for symbol in matches
        }
        return MockResponse(payload)

    def get_instrument_by_cusip(self, cusip: str) -> MockResponse:
        """Look up a simulator instrument by deterministic synthetic CUSIP."""
        if not isinstance(cusip, str):
            raise ValueError("cusip must be passed as str")

        symbol = self._find_symbol_by_cusip(cusip)
        if symbol is None:
            return MockResponse({"error": "Instrument not found"}, 404)

        return MockResponse(self._build_instrument_payload(symbol))

    def set_timeout(self, timeout) -> None:
        """Set request timeout (no-op in simulation)."""
        pass

    def token_age(self) -> int:
        """Return token age in seconds (always zero in simulation)."""
        return 0

    # ==================== PRICE HISTORY CONVENIENCE ====================

    def _price_history_convenience(
        self,
        symbol: str,
        *,
        period_type=None,
        period=None,
        frequency_type=None,
        frequency=None,
        start_datetime: datetime.datetime | None = None,
        end_datetime: datetime.datetime | None = None,
        need_extended_hours_data=None,
        need_previous_close=None,
    ) -> MockResponse:
        """Shared implementation for price-history convenience methods."""
        return self.get_price_history(
            symbol,
            period_type=period_type,
            period=period,
            frequency_type=frequency_type,
            frequency=frequency,
            start_datetime=start_datetime,
            end_datetime=end_datetime,
            need_extended_hours_data=need_extended_hours_data,
            need_previous_close=need_previous_close,
        )

    def get_price_history_every_minute(self, symbol: str, **kwargs) -> MockResponse:
        """Fetch price history at one-minute granularity."""
        return self._price_history_convenience(
            symbol,
            period_type="day",
            period=1,
            frequency_type="minute",
            frequency=1,
            **kwargs,
        )

    def get_price_history_every_five_minutes(
        self, symbol: str, **kwargs
    ) -> MockResponse:
        """Fetch price history at five-minute granularity."""
        return self._price_history_convenience(
            symbol,
            period_type="day",
            period=1,
            frequency_type="minute",
            frequency=5,
            **kwargs,
        )

    def get_price_history_every_ten_minutes(
        self, symbol: str, **kwargs
    ) -> MockResponse:
        """Fetch price history at ten-minute granularity."""
        return self._price_history_convenience(
            symbol,
            period_type="day",
            period=1,
            frequency_type="minute",
            frequency=10,
            **kwargs,
        )

    def get_price_history_every_fifteen_minutes(
        self, symbol: str, **kwargs
    ) -> MockResponse:
        """Fetch price history at fifteen-minute granularity."""
        return self._price_history_convenience(
            symbol,
            period_type="day",
            period=1,
            frequency_type="minute",
            frequency=15,
            **kwargs,
        )

    def get_price_history_every_thirty_minutes(
        self, symbol: str, **kwargs
    ) -> MockResponse:
        """Fetch price history at thirty-minute granularity."""
        return self._price_history_convenience(
            symbol,
            period_type="day",
            period=1,
            frequency_type="minute",
            frequency=30,
            **kwargs,
        )

    def get_price_history_every_day(self, symbol: str, **kwargs) -> MockResponse:
        """Fetch price history at daily granularity."""
        return self._price_history_convenience(
            symbol,
            period_type="year",
            period=20,
            frequency_type="daily",
            frequency=1,
            **kwargs,
        )

    def get_price_history_every_week(self, symbol: str, **kwargs) -> MockResponse:
        """Fetch price history at weekly granularity."""
        return self._price_history_convenience(
            symbol,
            period_type="year",
            period=20,
            frequency_type="weekly",
            frequency=1,
            **kwargs,
        )


# Backward compatibility alias
Client = MockClient
