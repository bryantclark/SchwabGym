"""
SchwabGym Core Client
=====================

Simulator that replicates the Charles Schwab Trader API.

Author: Bryant Clark
Repository: https://github.com/bryantclark/SchwabGym
License: MIT
"""

import datetime
import logging
from collections import deque
from typing import Any, Dict, List, Optional, Union

from schwabgym.fees import FeeCalculator
from schwabgym.orders import MockResponse
from schwabgym.physics import ExecutionEngine, RealisticExecutionEngine

# Configure logging
logger = logging.getLogger(__name__)


class MockClient:
    """
    Simulator of schwab.client.Client.

    This class provides API compatibility with schwab-py, enabling agents
    to be trained in simulation using the same interface used for live trading.

    Attributes:
        df (pd.DataFrame): Historical OHLCV data with dual-price columns
        current_step (int): Current simulation time index
        max_steps (int): Total time steps available
        cash (float): Current cash balance
        positions (Dict): Open positions {symbol: {quantity, avgPrice, assetType}}
        execution_engine (ExecutionEngine): Physics model for order fills
        fee_calculator (FeeCalculator): Regulatory fee computation
    """

    # ==================== REGULATORY CONSTANTS ====================

    # Pattern Day Trading
    PDT_MIN_EQUITY = 25000.0  # Minimum equity to avoid PDT restriction
    PDT_DAY_TRADE_LIMIT = 4  # Max day trades in rolling window
    PDT_LOOKBACK_DAYS = 5  # Rolling window size

    # Margin Requirements (Regulation T)
    INITIAL_MARGIN_RATIO = 0.50  # 50% initial margin for new positions
    MAINTENANCE_MARGIN_RATIO = 0.30  # 30% maintenance margin

    def __init__(
        self,
        market_data_df,
        initial_cash: float = 25000.0,
        execution_engine: Optional[ExecutionEngine] = None,
    ):
        """
        Initialize the trading simulator.

        Args:
            market_data_df (pd.DataFrame): Historical market data with columns:
                - Open, High, Low, Close: Raw prices (execution)
                - AdjClose: Adjusted prices (analysis)
                - Volume: Share volume
                - Volatility: (optional) calculated if missing
            initial_cash (float): Starting cash balance
            execution_engine (ExecutionEngine, optional): Physics model.
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

        # Account state
        self.account_number = "12345678"  # Mock account number
        self.account_hash = "HASH_1234"  # Mock encrypted hash
        self.cash = float(initial_cash)
        self.positions: Dict[str, Dict[str, Any]] = {}
        self.orders: Dict[int, Dict] = {}  # Order history
        self.working_orders: List[Dict] = []  # Active limit/stop orders

        # Pattern Day Trading enforcement
        self.day_trades: deque = deque()  # Timestamps of day trades
        self.opened_positions_today: set = set()  # Symbols opened today
        self._is_pdt_flagged = False

        # Physics engine (defaults to realistic)
        if execution_engine is None:
            self.execution_engine = RealisticExecutionEngine()
            logger.info("Using RealisticExecutionEngine")
        else:
            self.execution_engine = execution_engine
            logger.info(f"Using {type(execution_engine).__name__}")

        # Fee calculator
        self.fee_calculator = FeeCalculator()

        logger.info(f"MockClient initialized: ${initial_cash:,.2f} starting capital")

    # ==================== SIMULATION CONTROL ====================

    def advance_time(self) -> bool:
        """
        Advance simulator by one time step.

        Returns:
            bool: True if successfully advanced, False if at end of data
        """
        if self.current_step >= self.max_steps:
            logger.warning("Reached end of market data")
            return False

        self.current_step += 1

        # Check working orders for fills
        self._process_working_orders()

        # Clear overnight positions for PDT tracking
        if self.current_step > 0:
            curr_date = self._get_current_time().date()
            prev_date = self.df.index[self.current_step - 1].date()
            if curr_date > prev_date:
                self.opened_positions_today.clear()
                logger.debug(f"New trading day: {curr_date}")

        return True

    def reset(self, initial_cash: Optional[float] = None) -> None:
        """
        Reset simulator to initial state.

        Args:
            initial_cash (float, optional): New starting cash.
                Uses original value if None.
        """
        self.current_step = 0
        if initial_cash is not None:
            self.cash = float(initial_cash)
        self.positions.clear()
        self.orders.clear()
        self.working_orders.clear()
        self.day_trades.clear()
        self.opened_positions_today.clear()
        self._is_pdt_flagged = False
        logger.info("Simulator reset to initial state")

    # ==================== INTERNAL HELPERS ====================

    def _get_current_raw_price(self, symbol: str) -> float:
        """Get current raw execution price (Close column)."""
        return float(self.df.iloc[self.current_step]["Close"])

    def _get_current_time(self) -> datetime.datetime:
        """Get current timestamp from dataframe index."""
        return self.df.index[self.current_step]

    def _calculate_market_value(self) -> tuple[float, float]:
        """
        Calculate current market value of positions.

        Returns:
            tuple: (long_market_value, short_market_value)
        """
        long_mv = 0.0
        short_mv = 0.0

        for sym, pos in self.positions.items():
            current_price = self._get_current_raw_price(sym)
            qty = pos["quantity"]
            mv = qty * current_price

            if qty > 0:
                long_mv += mv
            else:
                short_mv += abs(mv)

        return long_mv, short_mv

    def _calculate_equity(self) -> float:
        """Calculate total account equity (NAV)."""
        long_mv, short_mv = self._calculate_market_value()
        return self.cash + long_mv - short_mv

    def _calculate_buying_power(self, equity: float) -> float:
        """
        Calculate buying power based on margin requirements.

        Per Regulation T: 2:1 leverage for accounts > $2000

        Args:
            equity (float): Current account equity

        Returns:
            float: Available buying power
        """
        if equity < 2000:
            return self.cash  # No margin for small accounts
        return equity * 2.0  # Standard 2:1 margin

    def _check_pdt_rule(self, symbol: str, instruction: str, curr_qty: int) -> None:
        """
        Enforce Pattern Day Trading rules.

        Args:
            symbol (str): Ticker symbol
            instruction (str): Order instruction (BUY, SELL, etc.)
            curr_qty (int): Current position quantity

        Raises:
            ValueError: If PDT restriction triggered
        """
        # Update day trades window
        cutoff_date = self._get_current_time().date() - datetime.timedelta(
            days=self.PDT_LOOKBACK_DAYS
        )
        while self.day_trades and self.day_trades[0] < cutoff_date:
            self.day_trades.popleft()

        # Check if this order would create a day trade
        is_closing = (instruction in ["SELL", "SELL_TO_CLOSE"] and curr_qty > 0) or (
            instruction in ["BUY_TO_COVER", "BUY_TO_CLOSE"] and curr_qty < 0
        )

        if is_closing and symbol in self.opened_positions_today:
            # This is a day trade!
            current_equity = self._calculate_equity()
            day_trade_count = len(self.day_trades) + 1  # +1 for this trade

            if (
                day_trade_count >= self.PDT_DAY_TRADE_LIMIT
                and current_equity < self.PDT_MIN_EQUITY
            ):
                self._is_pdt_flagged = True
                raise ValueError(
                    f"403 Forbidden: Pattern Day Trader Restriction. "
                    f"Account equity ${current_equity:,.2f} < ${self.PDT_MIN_EQUITY:,.2f} "
                    f"and {day_trade_count} day trades in {self.PDT_LOOKBACK_DAYS} days."
                )

    def _process_working_orders(self):
        """Check working limit/stop orders for execution."""
        remaining_orders = []

        row = self.df.iloc[self.current_step]
        market_data = {
            "Open": float(row["Open"]),
            "High": float(row["High"]),
            "Low": float(row["Low"]),
            "Close": float(row["Close"]),
            "Volume": int(row["Volume"]),
            "Volatility": float(row.get("Volatility", 0.01)),
        }

        for order in self.working_orders:
            # Only support single leg for now in working orders
            leg = order["orderLegCollection"][0]
            symbol = leg["instrument"]["symbol"]
            qty = leg["quantity"]
            instruction = leg["instruction"]
            order_type = order.get("orderType", "MARKET")
            limit_price = float(order.get("price", 0))

            should_fill = False

            if order_type == "LIMIT":
                # Check if price touched
                if instruction in ["BUY", "BUY_TO_COVER"]:
                    # Buy limit: Fill if market low is <= limit price (price at or below limit)
                    if market_data["Low"] <= limit_price:
                        should_fill = True
                else:
                    # Sell limit: Fill if market high is >= limit price (price at or above limit)
                    if market_data["High"] >= limit_price:
                        should_fill = True

            if should_fill:
                try:
                    # Execute at limit price (or better? for now limit price)
                    self._execute_trade_leg(leg, limit_price)
                    logger.info(
                        f"Filled {order_type} order: {instruction} {qty} {symbol} @ {limit_price}"
                    )
                except Exception as e:
                    logger.error(f"Failed to execute filled order: {e}")
            else:
                remaining_orders.append(order)

        self.working_orders = remaining_orders

    def _execute_trade_leg(self, leg: Dict, exec_price: float):
        """Execute a single trade leg and update account state."""
        symbol = leg["instrument"]["symbol"]
        qty = leg["quantity"]
        instruction = leg["instruction"]
        asset_type = leg["instrument"].get("assetType", "EQUITY")

        total_cost = exec_price * qty

        # Calculate regulatory fees (sell-side only)
        reg_fees = 0.0
        if instruction in ["SELL", "SELL_SHORT", "SELL_TO_CLOSE"]:
            trade_date = self._get_current_time().date()
            reg_fees = self.fee_calculator.calculate_total_regulatory_fees(
                trade_date=trade_date,
                quantity=qty,
                price=exec_price,
                asset_type=asset_type,
            )

        # Initialize position if needed
        if symbol not in self.positions:
            self.positions[symbol] = {
                "quantity": 0,
                "avgPrice": 0.0,
                "assetType": asset_type,
            }
            self.opened_positions_today.add(symbol)

        curr_pos = self.positions[symbol]
        curr_qty = curr_pos["quantity"]

        # Execute based on instruction
        if instruction in ["BUY", "BUY_TO_COVER", "BUY_TO_OPEN"]:
            # Check buying power
            acct = self.get_account(self.account_hash).json()["securitiesAccount"]
            bp = acct["currentBalances"]["buyingPower"]

            if total_cost > bp:
                raise ValueError(
                    f"Insufficient Buying Power: Required {total_cost}, Available {bp}"
                )

            self.cash -= total_cost

            if curr_qty >= 0:
                # Adding to long or initiating long
                new_qty = curr_qty + qty
                new_avg = ((curr_qty * curr_pos["avgPrice"]) + total_cost) / new_qty
                curr_pos["quantity"] = new_qty
                curr_pos["avgPrice"] = new_avg
            else:
                # Covering short
                curr_pos["quantity"] += qty

        elif instruction in ["SELL", "SELL_SHORT", "SELL_TO_CLOSE", "SELL_TO_OPEN"]:
            # Check position availability for sells
            if instruction in ["SELL", "SELL_TO_CLOSE"] and curr_qty < qty:
                raise ValueError(
                    f"Position not available: Required {qty}, Available {curr_qty}"
                )

            # Check for PDT violation
            self._check_pdt_rule(symbol, instruction, curr_qty)

            self.cash += total_cost - reg_fees

            if curr_qty <= 0:
                # Adding to short or initiating short
                new_qty = curr_qty - qty
                existing_val = abs(curr_qty) * curr_pos["avgPrice"]
                new_avg = (existing_val + total_cost) / abs(new_qty)
                curr_pos["quantity"] = new_qty
                curr_pos["avgPrice"] = new_avg
            else:
                # Selling long
                curr_pos["quantity"] -= qty

            # Record day trade if closing same-day position
            if symbol in self.opened_positions_today:
                self.day_trades.append(self._get_current_time().date())
                logger.warning(
                    f"Day trade: {len(self.day_trades)} in {self.PDT_LOOKBACK_DAYS}-day window"
                )

        # Clean up zero positions
        if abs(self.positions[symbol]["quantity"]) < 1e-6:
            del self.positions[symbol]

    # ==================== SCHWAB API INTERFACE ====================
    # These methods provide exact parity with schwab.client.Client

    def get_account_numbers(self) -> MockResponse:
        """
        Get linked account information (schwab.client.Client.get_account_numbers).

        Returns:
            MockResponse: JSON with accountNumber and hashValue
        """
        return MockResponse(
            {"accountNumber": self.account_number, "hashValue": self.account_hash}
        )

    def get_account(
        self, account_hash: str, fields: Optional[str] = None
    ) -> MockResponse:
        """
        Get detailed account information (schwab.client.Client.get_account).

        Args:
            account_hash (str): Encrypted account hash from get_account_numbers()
            fields (str, optional): Fields to include (not used in simulator)

        Returns:
            MockResponse: JSON with securitiesAccount object
        """
        if account_hash != self.account_hash:
            return MockResponse({"error": "Unauthorized"}, 401)

        # Calculate current values
        equity = self._calculate_equity()
        buying_power = self._calculate_buying_power(equity)
        long_mv, short_mv = self._calculate_market_value()

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
                    "roundTrips": len(self.day_trades),
                    "isDayTrader": self._is_pdt_flagged,
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

    def get_quotes(
        self, symbols: Union[str, List[str]], fields: Optional[str] = None
    ) -> MockResponse:
        """
        Get current quote(s) (schwab.client.Client.get_quotes).

        Args:
            symbols (str or List[str]): Single symbol or list of symbols
            fields (str, optional): Fields to include

        Returns:
            MockResponse: JSON with quote data for each symbol
        """
        if isinstance(symbols, str):
            symbols = [symbols]

        response_body = {}
        ts_ms = int(self._get_current_time().timestamp() * 1000)

        for sym in symbols:
            price = self._get_current_raw_price(sym)
            row = self.df.iloc[self.current_step]

            response_body[sym] = {
                "quote": {
                    "symbol": sym,
                    "lastPrice": price,
                    "closePrice": price,
                    "bidPrice": price * 0.9995,  # Simulated spread
                    "askPrice": price * 1.0005,
                    "totalVolume": int(row["Volume"]),
                    "tradeTime": ts_ms,
                }
            }

        return MockResponse(response_body)

    def get_price_history(
        self,
        symbol: str,
        period_type: Optional[str] = None,
        period: Optional[int] = None,
        frequency_type: Optional[str] = None,
        frequency: Optional[int] = None,
        start_date: Optional[datetime.datetime] = None,
        end_date: Optional[datetime.datetime] = None,
        need_extended_hours_data: Optional[bool] = None,
    ) -> MockResponse:
        """
        Get historical OHLCV data (schwab.client.Client.get_price_history).

        Args:
            symbol (str): Ticker symbol

        Returns:
            MockResponse: JSON with 'candles' list
        """
        LOOKBACK = 50  # Return last 50 bars
        start_idx = max(0, self.current_step - LOOKBACK + 1)

        # Use adjusted close for historical analysis
        col_close = "AdjClose" if "AdjClose" in self.df.columns else "Close"

        subset = self.df.iloc[start_idx : self.current_step + 1]
        candles = []

        for ts, row in subset.iterrows():
            candles.append(
                {
                    "open": float(row["Open"]),
                    "high": float(row["High"]),
                    "low": float(row["Low"]),
                    "close": float(row[col_close]),  # Adjusted for analysis
                    "volume": int(row["Volume"]),
                    "datetime": int(ts.timestamp() * 1000),
                }
            )

        logger.debug(f"Returned {len(candles)} candles for {symbol}")
        return MockResponse({"candles": candles, "symbol": symbol})

    def place_order(self, account_hash: str, order: Dict[str, Any]) -> MockResponse:
        """
        Place an order (schwab.client.Client.place_order).

        Args:
            account_hash (str): Encrypted account hash
            order (Dict): Order specification from schwab.orders.equities

        Returns:
            MockResponse: Empty dict on success (201), error dict on failure
        """
        if account_hash != self.account_hash:
            return MockResponse({"error": "Unauthorized"}, 401)

        # Check PDT status
        if self._is_pdt_flagged:
            return MockResponse(
                {"error": "Order Rejected: Pattern Day Trader Restriction"}, 403
            )

        order_type = order.get("orderType", "MARKET")

        if order_type == "MARKET":
            # Execute immediately
            for leg in order["orderLegCollection"]:
                # Calculate execution price with impact
                row = self.df.iloc[self.current_step]
                market_data = {
                    "Open": float(row["Open"]),
                    "High": float(row["High"]),
                    "Low": float(row["Low"]),
                    "Close": float(row["Close"]),
                    "Volume": int(row["Volume"]),
                    "Volatility": float(row.get("Volatility", 0.01)),
                }

                exec_price = self.execution_engine.calculate_execution_price(
                    base_price=market_data["Close"],
                    quantity=leg["quantity"],
                    instruction=leg["instruction"],
                    market_data=market_data,
                )

                try:
                    self._execute_trade_leg(leg, exec_price)
                    logger.info(
                        f"Filled MARKET order: {leg['instruction']} {leg['quantity']} {leg['instrument']['symbol']} @ {exec_price}"
                    )
                except ValueError as e:
                    return MockResponse({"error": str(e)}, 400)

        elif order_type == "LIMIT":
            # Queue for later execution
            self.working_orders.append(order)
            logger.info(f"Queued LIMIT order: {order}")

        else:
            logger.warning(f"Unsupported order type: {order_type}")
            return MockResponse({"error": "Unsupported order type"}, 400)

        return MockResponse({}, 201)


# Backward compatibility alias
Client = MockClient
