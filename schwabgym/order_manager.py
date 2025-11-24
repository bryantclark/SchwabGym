"""
SchwabGym Order Manager
=======================

Handles order validation, processing, execution, and history.
"""

import datetime
import logging
from typing import Any, Dict, List, Optional

from schwabgym.account import Account
from schwabgym.prices import PriceEngine
from schwabgym.physics import ExecutionEngine
from schwabgym.orders import MockResponse

logger = logging.getLogger(__name__)


class OrderManager:
    """
    Manages the order book and execution logic.

    Attributes:
        account (Account): Reference to the account.
        price_engine (PriceEngine): Reference to market data.
        execution_engine (ExecutionEngine): Reference to physics engine.
        orders (Dict): Order history {orderId: order_dict}.
        working_orders (List): Active limit/stop orders.
    """

    def __init__(self, account: Account, price_engine: PriceEngine, execution_engine: ExecutionEngine):
        self.account = account
        self.price_engine = price_engine
        self.execution_engine = execution_engine

        self.orders: Dict[int, Dict] = {}
        self.working_orders: List[Dict] = []
        self.next_order_id = 1000

    def reset(self) -> None:
        """Reset order history."""
        self.orders.clear()
        self.working_orders.clear()
        self.next_order_id = 1000

    def place_order(self, order: Dict[str, Any]) -> MockResponse:
        """
        Validate and place an order.

        Returns:
            MockResponse
        """
        # Assign ID
        order_id = self.next_order_id
        self.next_order_id += 1

        current_time = self.price_engine.get_current_time()
        order["orderId"] = order_id
        order["status"] = "WORKING"
        order["enteredTime"] = current_time.isoformat()
        order["closeTime"] = None
        order["accountId"] = int(self.account.account_number)

        self.orders[order_id] = order

        order_type = order.get("orderType", "MARKET")

        if order_type == "MARKET":
            return self._execute_market_order(order)
        elif order_type == "LIMIT":
            self.working_orders.append(order)
            logger.info(f"Queued LIMIT order: {order_id}")
            return self._success_response(order_id)
        else:
            return MockResponse({"error": "Unsupported order type"}, 400)

    def cancel_order(self, order_id: int) -> MockResponse:
        """Cancel an order."""
        # Check working orders
        for i, order in enumerate(self.working_orders):
            if order.get("orderId") == order_id:
                cancelled_order = self.working_orders.pop(i)
                cancelled_order["status"] = "CANCELED"
                cancelled_order["cancelTime"] = self.price_engine.get_current_time().isoformat()
                logger.info(f"Order {order_id} canceled")
                return MockResponse({"orderId": order_id}, 200)

        # Check history
        if order_id in self.orders:
            order = self.orders[order_id]
            if order["status"] in ["FILLED", "CANCELED", "REJECTED", "EXPIRED"]:
                return MockResponse(
                    {"error": f"Order {order_id} is already {order['status']}, cannot cancel."},
                    400
                )

        return MockResponse({"error": "Order not found"}, 404)

    def replace_order(self, order_id: int, order_spec: Dict[str, Any]) -> MockResponse:
        """Replace an order."""
        cancel_resp = self.cancel_order(order_id)
        if cancel_resp.status_code != 200:
            return cancel_resp
        return self.place_order(order_spec)

    def process_working_orders(self) -> None:
        """Check working orders against current market data."""
        remaining_orders = []
        market_data = self.price_engine.get_current_ohlcv()

        for order in self.working_orders:
            leg = order["orderLegCollection"][0]
            instruction = leg["instruction"]
            order_type = order.get("orderType", "MARKET")
            limit_price = float(order.get("price", 0))

            should_fill = False

            if order_type == "LIMIT":
                if instruction in ["BUY", "BUY_TO_COVER"]:
                    # Fill if Low <= Limit
                    if market_data["Low"] <= limit_price:
                        should_fill = True
                else:
                    # Fill if High >= Limit
                    if market_data["High"] >= limit_price:
                        should_fill = True

            if should_fill:
                try:
                    self._execute_trade_leg(leg, limit_price)
                    order["status"] = "FILLED"
                    order["closeTime"] = self.price_engine.get_current_time().isoformat()
                    logger.info(f"Filled LIMIT order {order['orderId']}")
                except Exception as e:
                    logger.error(f"Failed to execute LIMIT order: {e}")
                    order["status"] = "REJECTED"
                    order["cancelTime"] = self.price_engine.get_current_time().isoformat()
            else:
                remaining_orders.append(order)

        self.working_orders = remaining_orders

    def _execute_market_order(self, order: Dict[str, Any]) -> MockResponse:
        """Execute immediate market order."""
        for leg in order["orderLegCollection"]:
            current_price = self.price_engine.get_current_price(leg["instrument"]["symbol"])
            market_data = self.price_engine.get_current_ohlcv()

            exec_price = self.execution_engine.calculate_execution_price(
                base_price=current_price,
                quantity=leg["quantity"],
                instruction=leg["instruction"],
                market_data=market_data
            )

            try:
                self._execute_trade_leg(leg, exec_price)
                logger.info(f"Filled MARKET order {order['orderId']} @ {exec_price}")
                order["status"] = "FILLED"
                order["closeTime"] = self.price_engine.get_current_time().isoformat()
            except ValueError as e:
                order["status"] = "REJECTED"
                order["cancelTime"] = self.price_engine.get_current_time().isoformat()
                return MockResponse({"error": str(e)}, 400)

        return self._success_response(order["orderId"])

    def _execute_trade_leg(self, leg: Dict, exec_price: float):
        """Execute single leg against account."""
        symbol = leg["instrument"]["symbol"]
        qty = leg["quantity"]
        instruction = leg["instruction"]
        asset_type = leg["instrument"].get("assetType", "EQUITY")

        # Current equity/bp needed for checks
        current_equity = self.account.calculate_equity(self.price_engine.get_current_price)
        buying_power = self.account.calculate_buying_power(current_equity)
        current_date = self.price_engine.get_current_time().date()

        # PDT Check
        curr_pos = self.account.positions.get(symbol, {"quantity": 0})["quantity"]
        self.account.check_pdt_rule(symbol, instruction, curr_pos, current_equity, current_date)

        # Execute
        self.account.execute_trade(
            symbol=symbol,
            quantity=qty,
            price=exec_price,
            instruction=instruction,
            asset_type=asset_type,
            trade_date=current_date,
            buying_power_check=buying_power
        )

    def _success_response(self, order_id: int) -> MockResponse:
        return MockResponse(
            {},
            201,
            headers={"Location": f"https://api.schwab.com/v1/accounts/{self.account.account_number}/orders/{order_id}"}
        )
