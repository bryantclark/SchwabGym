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
from typing import Any

from schwabgym.account import Account
from schwabgym.order_manager import OrderManager
from schwabgym.orders import MockResponse
from schwabgym.physics import ExecutionEngine, RealisticExecutionEngine
from schwabgym.prices import PriceEngine
from schwabgym.streamer import MockStreamer

# Configure logging
logger = logging.getLogger(__name__)


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

        if execution_engine is None:
            execution_engine = RealisticExecutionEngine()
            logger.info("Using RealisticExecutionEngine")

        self.execution_engine = execution_engine

        # Default to latency mode unless explicitly disabled
        latency_mode = kwargs.get("latency_mode", True)

        self.order_manager = OrderManager(
            account=self.account,
            price_engine=self.price_engine,
            execution_engine=self.execution_engine,
            latency_mode=latency_mode,
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

    # ==================== SCHWAB API INTERFACE ====================

    def get_account_numbers(self) -> MockResponse:
        return MockResponse(
            {"accountNumber": self.account_number, "hashValue": self.account_hash}
        )

    def get_account(self, account_hash: str, fields: str | None = None) -> MockResponse:
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

    def get_accounts(self, fields: str | None = None) -> MockResponse:
        """Get all linked accounts (schwab-py parity: only one simulated account)."""
        account_resp = self.get_account(self.account_hash, fields=fields)
        return MockResponse([account_resp.json()])

    def get_quote(self, symbol: str, fields: str | None = None) -> MockResponse:
        """Get quote for a single symbol."""
        data = self.price_engine.get_quotes_data([symbol])
        return MockResponse(data)

    def get_quotes(
        self, symbols: str | list[str], fields: str | None = None
    ) -> MockResponse:
        if isinstance(symbols, str):
            symbols = [symbols]

        data = self.price_engine.get_quotes_data(symbols)
        return MockResponse(data)

    def get_price_history(
        self,
        symbol: str,
        period_type: str | None = None,
        period: int | None = None,
        frequency_type: str | None = None,
        frequency: int | None = None,
        start_date: datetime.datetime | None = None,
        end_date: datetime.datetime | None = None,
        need_extended_hours_data: bool | None = None,
    ) -> MockResponse:
        candles = self.price_engine.get_price_history_data(symbol)
        logger.debug(f"Returned {len(candles)} candles for {symbol}")
        return MockResponse({"candles": candles, "symbol": symbol})

    def place_order(self, account_hash: str, order: dict[str, Any]) -> MockResponse:
        if account_hash != self.account_hash:
            return MockResponse({"error": "Unauthorized"}, 401)

        # Check for Pattern Day Trader restriction
        if self.account.is_pdt_flagged:
            return MockResponse(
                {"error": "Order Rejected: Pattern Day Trader Restriction"}, 403
            )

        return self.order_manager.place_order(order)

    def cancel_order(self, account_hash: str, order_id: int) -> MockResponse:
        if account_hash != self.account_hash:
            return MockResponse({"error": "Unauthorized"}, 401)
        return self.order_manager.cancel_order(order_id)

    def replace_order(
        self, account_hash: str, order_id: int, order_spec: dict[str, Any]
    ) -> MockResponse:
        if account_hash != self.account_hash:
            return MockResponse({"error": "Unauthorized"}, 401)
        return self.order_manager.replace_order(order_id, order_spec)

    def get_order(self, account_hash: str, order_id: int) -> MockResponse:
        if account_hash != self.account_hash:
            return MockResponse({"error": "Unauthorized"}, 401)

        if order_id in self.orders:
            return MockResponse(self.orders[order_id])

        return MockResponse({"error": "Order not found"}, 404)

    def get_orders_for_account(
        self,
        account_hash: str,
        from_entered_datetime: str | None = None,
        to_entered_datetime: str | None = None,
        status: str | None = None,
        max_results: int | None = None,
    ) -> MockResponse:
        if account_hash != self.account_hash:
            return MockResponse({"error": "Unauthorized"}, 401)

        result_orders = list(self.orders.values())

        if status:
            result_orders = [o for o in result_orders if o["status"] == status]

        result_orders.sort(key=lambda x: x["orderId"])

        if max_results:
            result_orders = result_orders[-max_results:]

        return MockResponse(result_orders)

    def get_orders_for_all_linked_accounts(
        self,
        from_entered_datetime: str | None = None,
        to_entered_datetime: str | None = None,
        status: str | None = None,
        max_results: int | None = None,
    ) -> MockResponse:
        """Get orders across all linked accounts (single account in sim)."""
        return self.get_orders_for_account(
            self.account_hash,
            from_entered_datetime=from_entered_datetime,
            to_entered_datetime=to_entered_datetime,
            status=status,
            max_results=max_results,
        )

    def get_transactions(
        self,
        account_hash: str,
        start_date: datetime.datetime | None = None,
        end_date: datetime.datetime | None = None,
        transaction_types: str | None = None,
        symbol: str | None = None,
    ) -> MockResponse:
        """Get transaction history (stub — returns filled orders as transactions)."""
        if account_hash != self.account_hash:
            return MockResponse({"error": "Unauthorized"}, 401)

        filled = [o for o in self.orders.values() if o["status"] == "FILLED"]
        return MockResponse(filled)

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

    def get_price_history_every_minute(self, symbol: str, **kwargs) -> MockResponse:
        """Convenience wrapper — returns same data (sim has single frequency)."""
        return self.get_price_history(symbol, **kwargs)

    def get_price_history_every_five_minutes(
        self, symbol: str, **kwargs
    ) -> MockResponse:
        """Convenience wrapper — returns same data (sim has single frequency)."""
        return self.get_price_history(symbol, **kwargs)

    def get_price_history_every_fifteen_minutes(
        self, symbol: str, **kwargs
    ) -> MockResponse:
        """Convenience wrapper — returns same data (sim has single frequency)."""
        return self.get_price_history(symbol, **kwargs)

    def get_price_history_every_day(self, symbol: str, **kwargs) -> MockResponse:
        """Convenience wrapper — returns same data (sim has single frequency)."""
        return self.get_price_history(symbol, **kwargs)


# Backward compatibility alias
Client = MockClient
