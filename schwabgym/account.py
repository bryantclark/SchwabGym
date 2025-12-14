"""
SchwabGym Account Logic
=======================

Manages account state, balances, positions, and regulatory rules (PDT, Margin).
"""

import datetime
import logging
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

from schwabgym.fees import FeeCalculator

logger = logging.getLogger(__name__)


class Account:
    """
    Simulates a brokerage account.

    Attributes:
        cash (float): Current cash balance.
        positions (Dict): Open positions.
        day_trades (deque): Rolling window of day trades.
    """

    # Regulatory Constants
    PDT_MIN_EQUITY = 25000.0
    PDT_DAY_TRADE_LIMIT = 4
    PDT_LOOKBACK_DAYS = 5
    INITIAL_MARGIN_RATIO = 0.50
    MAINTENANCE_MARGIN_RATIO = 0.30

    def __init__(self, initial_cash: float = 25000.0, account_number: str = "12345678"):
        """
        Initialize account.

        Args:
            initial_cash: Starting balance.
            account_number: Account ID.
        """
        self.initial_cash = float(initial_cash)
        self.cash = self.initial_cash
        self.account_number = account_number

        # State
        self.positions: Dict[str, Dict[str, Any]] = (
            {}
        )  # {symbol: {quantity, avgPrice, assetType}}
        self.day_trades: deque = deque()
        self.opened_positions_today: set = set()
        self._is_pdt_flagged = False

        self.fee_calculator = FeeCalculator()

    def reset(self, initial_cash: Optional[float] = None) -> None:
        """Reset account to initial state."""
        if initial_cash is not None:
            self.cash = float(initial_cash)
        else:
            self.cash = self.initial_cash

        self.positions.clear()
        self.day_trades.clear()
        self.opened_positions_today.clear()
        self._is_pdt_flagged = False

    def on_new_day(self) -> None:
        """Called when simulation date changes."""
        self.opened_positions_today.clear()

    def calculate_market_value(self, price_lookup_func) -> Tuple[float, float]:
        """
        Calculate long and short market values.

        Args:
            price_lookup_func: Function(symbol) -> float returning current price.

        Returns:
            (long_mv, short_mv)
        """
        long_mv = 0.0
        short_mv = 0.0

        for sym, pos in self.positions.items():
            current_price = price_lookup_func(sym)
            qty = pos["quantity"]
            mv = qty * current_price

            if qty > 0:
                long_mv += mv
            else:
                short_mv += abs(mv)

        return long_mv, short_mv

    def calculate_equity(self, price_lookup_func) -> float:
        """Calculate total account equity."""
        long_mv, short_mv = self.calculate_market_value(price_lookup_func)
        return self.cash + long_mv - short_mv

    def calculate_buying_power(self, equity: float) -> float:
        """Calculate buying power (2:1 margin)."""
        if equity < 2000:
            return self.cash
        return equity * 2.0

    def check_pdt_rule(
        self,
        symbol: str,
        instruction: str,
        curr_qty: int,
        current_equity: float,
        current_date: datetime.date,
    ) -> None:
        """
        Check and enforce Pattern Day Trading rules.

        Raises:
            ValueError if trade is blocked.
        """
        # Update day trades window
        cutoff_date = current_date - datetime.timedelta(days=self.PDT_LOOKBACK_DAYS)
        while self.day_trades and self.day_trades[0] < cutoff_date:
            self.day_trades.popleft()

        is_closing = (instruction in ["SELL", "SELL_TO_CLOSE"] and curr_qty > 0) or (
            instruction in ["BUY_TO_COVER", "BUY_TO_CLOSE"] and curr_qty < 0
        )

        if is_closing and symbol in self.opened_positions_today:
            day_trade_count = len(self.day_trades) + 1
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

    def execute_trade(
        self,
        symbol: str,
        quantity: float,
        price: float,
        instruction: str,
        asset_type: str,
        trade_date: datetime.date,
        buying_power_check: float,
    ) -> None:
        """
        Update account state for a trade execution.

        Args:
            buying_power_check: Available buying power to validate against.
        """
        total_cost = price * quantity

        # Calculate fees
        reg_fees = 0.0
        if instruction in ["SELL", "SELL_SHORT", "SELL_TO_CLOSE"]:
            reg_fees = self.fee_calculator.calculate_total_regulatory_fees(
                trade_date=trade_date,
                quantity=quantity,
                price=price,
                asset_type=asset_type,
            )

        # Init position
        if symbol not in self.positions:
            self.positions[symbol] = {
                "quantity": 0,
                "avgPrice": 0.0,
                "assetType": asset_type,
            }
            self.opened_positions_today.add(symbol)

        curr_pos = self.positions[symbol]
        curr_qty = curr_pos["quantity"]

        # Validate & Update
        if instruction in ["BUY", "BUY_TO_COVER", "BUY_TO_OPEN"]:
            if total_cost > buying_power_check:
                raise ValueError(
                    f"Insufficient Buying Power: Required {total_cost}, Available {buying_power_check}"
                )

            self.cash -= total_cost

            if curr_qty >= 0:
                new_qty = curr_qty + quantity
                new_avg = ((curr_qty * curr_pos["avgPrice"]) + total_cost) / new_qty
                curr_pos["quantity"] = new_qty
                curr_pos["avgPrice"] = new_avg
            else:
                # Covering short
                curr_pos["quantity"] += quantity

        elif instruction in ["SELL", "SELL_SHORT", "SELL_TO_CLOSE", "SELL_TO_OPEN"]:
            if instruction in ["SELL", "SELL_TO_CLOSE"] and curr_qty < quantity:
                raise ValueError(
                    f"Position not available: Required {quantity}, Available {curr_qty}"
                )

            # Note: PDT check should be called BEFORE this method, but if called here
            # we need equity passed in. We'll assume check is done by caller (OrderManager).

            self.cash += total_cost - reg_fees

            if curr_qty <= 0:
                # Adding to short
                new_qty = curr_qty - quantity
                existing_val = abs(curr_qty) * curr_pos["avgPrice"]
                new_avg = (existing_val + total_cost) / abs(new_qty)
                curr_pos["quantity"] = new_qty
                curr_pos["avgPrice"] = new_avg
            else:
                curr_pos["quantity"] -= quantity

            # Record day trade
            if symbol in self.opened_positions_today:
                self.day_trades.append(trade_date)
                logger.warning(f"Day trade recorded. Total: {len(self.day_trades)}")

        # Cleanup
        if abs(self.positions[symbol]["quantity"]) < 1e-6:
            del self.positions[symbol]
