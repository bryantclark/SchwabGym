"""
SchwabGym Order Manager
=======================

Handles order validation, processing, execution, and history.
"""

import datetime
import logging
from typing import Any, Dict, List, Optional, Tuple

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
        pending_orders (List): Orders delayed by simulated latency.
    """

    def __init__(self, account: Account, price_engine: PriceEngine, execution_engine: ExecutionEngine, latency_mode: bool = True):
        self.account = account
        self.price_engine = price_engine
        self.execution_engine = execution_engine

        self.orders: Dict[int, Dict] = {}
        self.working_orders: List[Dict] = []
        self.pending_orders: List[Dict] = [] # Tuple of (release_time_step, order)
        self.next_order_id = 1000

        # Configuration
        self.latency_mode = latency_mode
        self.latency_steps = 1 if latency_mode else 0
        self.strict_limit_orders = False # If True, requires price to cross limit, not just touch

    def reset(self) -> None:
        """Reset order history."""
        self.orders.clear()
        self.working_orders.clear()
        self.pending_orders.clear()
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
        order["status"] = "PENDING_ACTIVATION"
        order["enteredTime"] = current_time.isoformat()
        order["closeTime"] = None
        order["accountId"] = int(self.account.account_number)

        self.orders[order_id] = order

        if not self.latency_mode:
            # Execute/Process immediately (Legacy Mode)
            order_type = order.get("orderType", "MARKET")
            if order_type == "MARKET":
                # Execute immediately
                error_msg = self._execute_market_order(order)

                if order["status"] == "FILLED":
                    return self._success_response(order_id)
                elif order["status"] == "REJECTED":
                    # We should return 400 if it was rejected immediately
                    return MockResponse({"error": error_msg or "Order rejected"}, 400)

            elif order_type == "LIMIT":
                self.working_orders.append(order)
                order["status"] = "WORKING" # Set to WORKING immediately for non-latency
                logger.info(f"Queued LIMIT order: {order_id}")
                return self._success_response(order_id)
            else:
                 return MockResponse({"error": "Unsupported order type"}, 400)

        # Simulate Latency: Add to pending queue
        # Release time is current step + latency
        release_step = self.price_engine.current_step + self.latency_steps
        self.pending_orders.append({
            "release_step": release_step,
            "order": order
        })

        logger.info(f"Order {order_id} placed, pending activation until step {release_step}")

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
            elif order_type == "LIMIT":
                self.working_orders.append(order)
                logger.info(f"Activated LIMIT order: {order['orderId']}")
            else:
                # Unsupported types rejected immediately upon activation
                order["status"] = "REJECTED"
                order["cancelTime"] = self.price_engine.get_current_time().isoformat()

        # 2. Process Working Orders
        remaining_orders = []

        # We need to loop carefully because market_data depends on symbol
        # But working_orders might have different symbols.
        # So inside loop, get data for that order's symbol.

        for order in self.working_orders:
            leg = order["orderLegCollection"][0]
            symbol = leg["instrument"]["symbol"]
            instruction = leg["instruction"]
            order_type = order.get("orderType", "MARKET")
            limit_price = float(order.get("price", 0))

            # Get data for specific symbol
            market_data = self.price_engine.get_current_ohlcv(symbol)

            should_fill = False

            # Enhanced Microstructure Logic

            if order_type == "LIMIT":
                # Strict Mode: Price must cross through limit
                # Touch Mode: Low/High touching limit is enough (traditional backtest)

                open_p = market_data["Open"]
                high_p = market_data["High"]
                low_p = market_data["Low"]
                close_p = market_data["Close"]

                if instruction in ["BUY", "BUY_TO_COVER"]:
                    if self.strict_limit_orders:
                        # Price crossed down through limit?
                        # Case 1: Gap down (Open < Limit) -> Fill at Open (better price)
                        # Case 2: Intraday cross (Open > Limit, Low < Limit)
                        if open_p < limit_price:
                            should_fill = True # Gapped below
                            # TODO: Price improvement logic (fill at Open)
                        elif low_p < limit_price:
                            # It traded below limit.
                            # If Close > Limit, it bounced.
                            should_fill = True
                    else:
                        # Touch mode (Default)
                        if low_p <= limit_price:
                            should_fill = True

                elif instruction in ["SELL", "SELL_SHORT"]:
                    if self.strict_limit_orders:
                        if open_p > limit_price:
                            should_fill = True
                        elif high_p > limit_price:
                            should_fill = True
                    else:
                         if high_p >= limit_price:
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

    def _execute_market_order(self, order: Dict[str, Any]) -> Optional[str]:
        """Execute immediate market order. Returns error string if rejected, None otherwise."""
        for leg in order["orderLegCollection"]:
            symbol = leg["instrument"]["symbol"]
            current_price = self.price_engine.get_current_price(symbol)
            market_data = self.price_engine.get_current_ohlcv(symbol)

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
                logger.error(f"Market order rejected: {e}")
                return str(e) # Return error message

        return None

    def _execute_trade_leg(self, leg: Dict, exec_price: float):
        """Execute single leg against account."""
        symbol = leg["instrument"]["symbol"]
        qty = leg["quantity"]
        instruction = leg["instruction"]
        asset_type = leg["instrument"].get("assetType", "EQUITY")

        # Current equity/bp needed for checks
        # Note: account equity calc uses current prices of all positions
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
