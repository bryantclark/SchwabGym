"""
SchwabGym Order Manager
=======================

Handles order validation, processing, execution, and history.
"""

import logging
from typing import Any

from schwabgym.account import Account
from schwabgym.orders import MockResponse
from schwabgym.physics import ExecutionEngine
from schwabgym.prices import PriceEngine

logger = logging.getLogger(__name__)


_SUPPORTED_ORDER_TYPES = frozenset({"MARKET", "LIMIT", "STOP", "STOP_LIMIT"})
_QUEUED_ORDER_TYPES = frozenset({"LIMIT", "STOP", "STOP_LIMIT"})
_TERMINAL_STATUSES = frozenset({"FILLED", "CANCELED", "REJECTED", "EXPIRED"})


class OrderManager:
    """
    Manages the order book and execution logic.

    Attributes:
        account: Reference to the account.
        price_engine: Reference to market data.
        execution_engine: Reference to physics engine.
        orders: Order history {orderId: order_dict}.
        working_orders: Active limit/stop orders awaiting fill.
        pending_orders: Orders delayed by simulated latency.
    """

    def __init__(
        self,
        account: Account,
        price_engine: PriceEngine,
        execution_engine: ExecutionEngine,
        latency_mode: bool = True,
        account_hash: str = "",
    ):
        self.account = account
        self.price_engine = price_engine
        self.execution_engine = execution_engine
        self.account_hash = account_hash

        self.orders: dict[int, dict] = {}
        self.working_orders: list[dict] = []
        self.pending_orders: list[dict] = []
        self.next_order_id = 1000

        # Configuration
        self.latency_mode = latency_mode
        self.latency_steps = 1 if latency_mode else 0
        self.strict_limit_orders = False

    def reset(self) -> None:
        """Reset order history."""
        self.orders.clear()
        self.working_orders.clear()
        self.pending_orders.clear()
        self.next_order_id = 1000

    def place_order(self, order: dict[str, Any] | Any) -> MockResponse:
        """
        Validate and place an order.

        Args:
            order: Order specification — accepts a dict (from MockEquities/MockOptions)
                or a schwab-py ``OrderBuilder`` object (calls ``.build()`` automatically).

        Returns:
            MockResponse with 201 on success, 400 on rejection.
        """
        # schwab-py order helpers return OrderBuilder objects with a .build() method.
        # Convert to dict so the rest of the pipeline works identically.
        if hasattr(order, "build") and callable(order.build):
            order = order.build()

        order_type = order.get("orderType", "MARKET")
        if order_type not in _SUPPORTED_ORDER_TYPES:
            return MockResponse({"error": f"Unsupported order type: {order_type}"}, 400)

        # Validate required price fields
        if order_type in ("LIMIT", "STOP_LIMIT") and "price" not in order:
            return MockResponse(
                {"error": "LIMIT/STOP_LIMIT orders require 'price'"}, 400
            )
        if order_type in ("STOP", "STOP_LIMIT") and "stopPrice" not in order:
            return MockResponse(
                {"error": "STOP/STOP_LIMIT orders require 'stopPrice'"}, 400
            )

        # Assign ID and metadata
        order_id = self.next_order_id
        self.next_order_id += 1

        current_time = self.price_engine.get_current_time()
        order["orderId"] = order_id
        order["status"] = "PENDING_ACTIVATION"
        order["enteredTime"] = current_time.isoformat()
        order["closeTime"] = None
        order["accountId"] = int(self.account.account_number)

        self.orders[order_id] = order

        if not self.latency_mode:
            if order_type == "MARKET":
                error_msg = self._execute_market_order(order)

                if order["status"] == "FILLED":
                    return self._success_response(order_id)
                elif order["status"] == "REJECTED":
                    return MockResponse({"error": error_msg or "Order rejected"}, 400)

            elif order_type in _QUEUED_ORDER_TYPES:
                self.working_orders.append(order)
                order["status"] = "WORKING"
                logger.info(f"Queued {order_type} order: {order_id}")
                return self._success_response(order_id)

        # Simulate Latency: Add to pending queue
        release_step = self.price_engine.current_step + self.latency_steps
        self.pending_orders.append({"release_step": release_step, "order": order})

        logger.info(
            f"Order {order_id} placed, pending activation until step {release_step}"
        )

        return self._success_response(order_id)

    def cancel_order(self, order_id: int) -> MockResponse:
        """Cancel an order."""
        current_time_str = self.price_engine.get_current_time().isoformat()

        # Check pending orders first
        for i, item in enumerate(self.pending_orders):
            if item["order"]["orderId"] == order_id:
                cancelled_item = self.pending_orders.pop(i)
                order = cancelled_item["order"]
                order["status"] = "CANCELED"
                order["cancelTime"] = current_time_str
                logger.info(f"Pending Order {order_id} canceled")
                return MockResponse({"orderId": order_id}, 200)

        # Check working orders
        for i, order in enumerate(self.working_orders):
            if order.get("orderId") == order_id:
                cancelled_order = self.working_orders.pop(i)
                cancelled_order["status"] = "CANCELED"
                cancelled_order["cancelTime"] = current_time_str
                logger.info(f"Working Order {order_id} canceled")
                return MockResponse({"orderId": order_id}, 200)

        # Check history
        if order_id in self.orders:
            order = self.orders[order_id]
            if order["status"] in _TERMINAL_STATUSES:
                return MockResponse(
                    {
                        "error": f"Order {order_id} is already {order['status']}, cannot cancel."
                    },
                    400,
                )

        return MockResponse({"error": "Order not found"}, 404)

    def replace_order(
        self, order_id: int, order_spec: dict[str, Any] | Any
    ) -> MockResponse:
        """Replace an order (accepts dict or OrderBuilder)."""
        cancel_resp = self.cancel_order(order_id)
        if cancel_resp.status_code != 200:
            return cancel_resp
        return self.place_order(order_spec)

    def process_working_orders(self) -> None:
        """
        Process order lifecycle:
        1. Activate pending orders if latency period passed.
        2. Check working orders against current market data.
        """
        current_step = self.price_engine.current_step

        # 1. Process Pending Orders
        active_pending = []
        remaining_pending = []
        for item in self.pending_orders:
            if current_step >= item["release_step"]:
                active_pending.append(item["order"])
            else:
                remaining_pending.append(item)
        self.pending_orders = remaining_pending

        # Move activated orders to working or execute immediately if MARKET
        for order in active_pending:
            order["status"] = "WORKING"
            order_type = order.get("orderType", "MARKET")

            if order_type == "MARKET":
                self._execute_market_order(order)
            elif order_type in _QUEUED_ORDER_TYPES:
                self.working_orders.append(order)
                logger.info(f"Activated {order_type} order: {order['orderId']}")
            else:
                order["status"] = "REJECTED"
                order["cancelTime"] = self.price_engine.get_current_time().isoformat()

        # 2. Process Working Orders
        remaining_orders = []

        # Process working orders
        # Note: market_data depends on symbol, so we fetch it per order.

        for order in self.working_orders:
            leg = order["orderLegCollection"][0]
            symbol = leg["instrument"]["symbol"]
            instruction = leg["instruction"]
            order_type = order.get("orderType", "MARKET")
            limit_price = float(order.get("price", 0))

            # Get data for specific symbol
            market_data = self.price_engine.get_current_ohlcv(symbol)

            should_fill = False
            fill_price = limit_price
            high_p = market_data["High"]
            low_p = market_data["Low"]
            volume = int(market_data["Volume"])

            if order_type == "STOP":
                stop_price = float(order["stopPrice"])
                if self._is_stop_triggered(instruction, stop_price, high_p, low_p):
                    current_price = self.price_engine.get_current_price(symbol)
                    fill_price = self.execution_engine.calculate_execution_price(
                        base_price=current_price,
                        quantity=leg["quantity"],
                        instruction=instruction,
                        market_data=market_data,
                    )
                    should_fill = True

            elif order_type == "STOP_LIMIT":
                stop_price = float(order["stopPrice"])
                if self._is_stop_triggered(instruction, stop_price, high_p, low_p):
                    # Once triggered, behave like a limit order
                    should_fill = self.execution_engine.should_limit_fill(
                        limit_price=limit_price,
                        market_high=high_p,
                        market_low=low_p,
                        volume=volume,
                        quantity=leg["quantity"],
                    )

            elif order_type == "LIMIT":
                # Strict mode: price must cross through limit, not just touch
                if self.strict_limit_orders:
                    open_p = market_data["Open"]
                    crossed = False
                    if instruction in ["BUY", "BUY_TO_COVER"]:
                        crossed = open_p < limit_price or low_p < limit_price
                    elif instruction in ["SELL", "SELL_SHORT"]:
                        crossed = open_p > limit_price or high_p > limit_price
                    if not crossed:
                        remaining_orders.append(order)
                        continue

                # Delegate fill decision to the physics engine
                should_fill = self.execution_engine.should_limit_fill(
                    limit_price=limit_price,
                    market_high=high_p,
                    market_low=low_p,
                    volume=volume,
                    quantity=leg["quantity"],
                )

            if should_fill:
                try:
                    self._execute_trade_leg(leg, fill_price)
                    order["status"] = "FILLED"
                    order["closeTime"] = (
                        self.price_engine.get_current_time().isoformat()
                    )
                    logger.info(
                        f"Filled {order_type} order {order['orderId']} @ {fill_price:.4f}"
                    )
                except Exception as e:
                    logger.error(f"Failed to execute {order_type} order: {e}")
                    order["status"] = "REJECTED"
                    order["cancelTime"] = (
                        self.price_engine.get_current_time().isoformat()
                    )
            else:
                remaining_orders.append(order)

        self.working_orders = remaining_orders

    @staticmethod
    def _is_stop_triggered(
        instruction: str, stop_price: float, high: float, low: float
    ) -> bool:
        """Check if a stop order's trigger price has been hit."""
        if instruction in ("BUY", "BUY_TO_COVER"):
            return high >= stop_price
        if instruction in ("SELL", "SELL_SHORT"):
            return low <= stop_price
        return False

    def _execute_market_order(self, order: dict[str, Any]) -> str | None:
        """Execute immediate market order. Returns error string if rejected, None otherwise."""
        for leg in order["orderLegCollection"]:
            symbol = leg["instrument"]["symbol"]
            current_price = self.price_engine.get_current_price(symbol)
            market_data = self.price_engine.get_current_ohlcv(symbol)

            exec_price = self.execution_engine.calculate_execution_price(
                base_price=current_price,
                quantity=leg["quantity"],
                instruction=leg["instruction"],
                market_data=market_data,
            )

            try:
                self._execute_trade_leg(leg, exec_price)
                logger.info(f"Filled MARKET order {order['orderId']} @ {exec_price}")
                order["status"] = "FILLED"
                order["closeTime"] = self.price_engine.get_current_time().isoformat()
            except ValueError as e:
                order["status"] = "REJECTED"
                order["cancelTime"] = self.price_engine.get_current_time().isoformat()
                logger.error(f"Market order rejected: {e}")
                return str(e)  # Return error message

        return None

    def _execute_trade_leg(self, leg: dict, exec_price: float):
        """Execute single leg against account."""
        symbol = leg["instrument"]["symbol"]
        qty = leg["quantity"]
        instruction = leg["instruction"]
        asset_type = leg["instrument"].get("assetType", "EQUITY")

        # Current equity/bp needed for checks
        current_equity = self.account.calculate_equity(
            self.price_engine.get_current_price
        )
        buying_power = self.account.calculate_buying_power(current_equity)
        current_date = self.price_engine.get_current_time().date()

        # PDT Check
        curr_pos = self.account.positions.get(symbol, {"quantity": 0})["quantity"]
        self.account.check_pdt_rule(
            symbol, instruction, curr_pos, current_equity, current_date
        )

        # Execute
        self.account.execute_trade(
            symbol=symbol,
            quantity=qty,
            price=exec_price,
            instruction=instruction,
            asset_type=asset_type,
            trade_date=current_date,
            buying_power_check=buying_power,
        )

    def _success_response(self, order_id: int) -> MockResponse:
        return MockResponse(
            {},
            201,
            headers={
                "Location": f"https://api.schwabapi.com/trader/v1/accounts/{self.account_hash}/orders/{order_id}"
            },
        )
