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
import logging
from typing import Any, Dict, List, Optional, Union

from schwabgym.account import Account
from schwabgym.order_manager import OrderManager
from schwabgym.orders import MockResponse
from schwabgym.physics import ExecutionEngine, RealisticExecutionEngine
from schwabgym.prices import PriceEngine

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
        market_data_df,
        initial_cash: float = 25000.0,
        execution_engine: Optional[ExecutionEngine] = None,
    ):
        # Initialize components
        self.price_engine = PriceEngine(market_data_df)
        self.account = Account(initial_cash=initial_cash)

        if execution_engine is None:
            execution_engine = RealisticExecutionEngine()
            logger.info("Using RealisticExecutionEngine")

        self.execution_engine = execution_engine

        self.order_manager = OrderManager(
            account=self.account,
            price_engine=self.price_engine,
            execution_engine=self.execution_engine
        )

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
        return "HASH_1234"

    @property
    def account_number(self):
        return self.account.account_number

    # ==================== SIMULATION CONTROL ====================

    def advance_time(self) -> bool:
        """Advance simulator by one time step."""
        if not self.price_engine.advance_time():
            return False

        self.order_manager.process_working_orders()

        # New day check
        if self.current_step > 0:
            curr_date = self._get_current_time().date()
            prev_date = self.df.index[self.current_step - 1].date()
            if curr_date > prev_date:
                self.account.on_new_day()

        return True

    def reset(self, initial_cash: Optional[float] = None) -> None:
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

    def get_account(
        self, account_hash: str, fields: Optional[str] = None
    ) -> MockResponse:
        if account_hash != self.account_hash:
            return MockResponse({"error": "Unauthorized"}, 401)

        # Calculate current values
        equity = self._calculate_equity()
        buying_power = self._calculate_buying_power(equity)
        long_mv, short_mv = self.account.calculate_market_value(self.price_engine.get_current_price)

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
                    "isDayTrader": self.account._is_pdt_flagged,
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
        if isinstance(symbols, str):
            symbols = [symbols]

        data = self.price_engine.get_quotes_data(symbols)
        return MockResponse(data)

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
        candles = self.price_engine.get_price_history_data(symbol)
        logger.debug(f"Returned {len(candles)} candles for {symbol}")
        return MockResponse({"candles": candles, "symbol": symbol})

    def place_order(self, account_hash: str, order: Dict[str, Any]) -> MockResponse:
        if account_hash != self.account_hash:
            return MockResponse({"error": "Unauthorized"}, 401)

        # PDT Check happens in Account.execute_trade, but checking here for flagging?
        # The logic was moved to Account.check_pdt_rule which is called during execution.
        # However, schwab-py might reject upfront.
        # But for now, we rely on OrderManager handling it.
        # If user is ALREADY flagged, we should reject.
        if self.account._is_pdt_flagged:
             return MockResponse(
                {"error": "Order Rejected: Pattern Day Trader Restriction"}, 403
            )

        return self.order_manager.place_order(order)

    def cancel_order(self, account_hash: str, order_id: int) -> MockResponse:
        if account_hash != self.account_hash:
            return MockResponse({"error": "Unauthorized"}, 401)
        return self.order_manager.cancel_order(order_id)

    def replace_order(
        self, account_hash: str, order_id: int, order_spec: Dict[str, Any]
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
        from_entered_datetime: Optional[str] = None,
        to_entered_datetime: Optional[str] = None,
        status: Optional[str] = None,
        max_results: Optional[int] = None,
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


# Backward compatibility alias
Client = MockClient
