"""
SchwabGym Regulatory Fee Calculator
====================================

Calculates SEC Section 31 and FINRA TAF fees for US equities.

Fee schedules are data-driven and can be updated as rates change.
Last verified: March 2026.

Author: Bryant Clark
Repository: https://github.com/bryantclark/SchwabGym
License: MIT
"""

import datetime
from typing import Literal

# SEC Section 31 fee schedule: list of (effective_date, rate_per_dollar) tuples,
# sorted ascending by date.  The rate in effect is the last entry whose date
# is <= the trade date.
#
# Sources:
#   https://www.sec.gov/rules-regulations/fee-rate-advisories
#   https://www.finra.org/rules-guidance/notices/information-notice-20250424
_SEC_SCHEDULE: list[tuple[datetime.date, float]] = [
    (datetime.date(2020, 1, 1), 27.80 / 1_000_000),  # $27.80/million (baseline)
    (datetime.date(2025, 5, 14), 0.0),  # $0.00 — eliminated
    (datetime.date(2026, 4, 4), 20.60 / 1_000_000),  # $20.60/million (FY2026)
]

# FINRA TAF schedule: list of (effective_date, equity_rate, option_rate, cap).
# Source: https://www.finra.org/rules-guidance/guidance/trading-activity-fee
_TAF_SCHEDULE: list[tuple[datetime.date, float, float, float]] = [
    (datetime.date(2020, 1, 1), 0.000166, 0.00279, 8.30),  # Pre-2026
    (datetime.date(2026, 1, 1), 0.000195, 0.00329, 9.79),  # 2026 rates
]


def _lookup_sec_rate(trade_date: datetime.date) -> float:
    """Return the SEC Section 31 rate in effect on *trade_date*."""
    rate = 0.0
    for effective, r in _SEC_SCHEDULE:
        if trade_date >= effective:
            rate = r
        else:
            break
    return rate


def _lookup_taf(trade_date: datetime.date) -> tuple[float, float, float]:
    """Return (equity_rate, option_rate, cap) in effect on *trade_date*."""
    equity_rate, option_rate, cap = _TAF_SCHEDULE[0][1:]
    for effective, er, opr, c in _TAF_SCHEDULE:
        if trade_date >= effective:
            equity_rate, option_rate, cap = er, opr, c
        else:
            break
    return equity_rate, option_rate, cap


class FeeCalculator:
    """
    Calculate regulatory fees for equity and option transactions.

    Fee schedules are data-driven; update the module-level ``_SEC_SCHEDULE``
    and ``_TAF_SCHEDULE`` lists when rates change.

    All fees apply to sell-side transactions only.

    References:
        - SEC Section 31: https://www.sec.gov/rules-regulations/fee-rate-advisories
        - FINRA TAF: https://www.finra.org/rules-guidance/guidance/trading-activity-fee
    """

    def __init__(self):
        """Initialize fee calculator."""
        pass

    def calculate_sec_fee(
        self, trade_date: datetime.date, notional_value: float
    ) -> float:
        """
        Calculate SEC Section 31 transaction fee.

        Args:
            trade_date: Date of transaction.
            notional_value: Dollar value of trade (price x quantity).

        Returns:
            SEC fee in dollars.
        """
        rate = _lookup_sec_rate(trade_date)
        return notional_value * rate

    def calculate_taf(
        self,
        quantity: int,
        asset_type: Literal["EQUITY", "OPTION"] = "EQUITY",
        trade_date: datetime.date | None = None,
    ) -> float:
        """
        Calculate FINRA Trading Activity Fee.

        Args:
            quantity: Number of shares or contracts.
            asset_type: 'EQUITY' or 'OPTION'.
            trade_date: Date of transaction (defaults to today).

        Returns:
            TAF in dollars (capped per trade).
        """
        if trade_date is None:
            trade_date = datetime.date.today()

        equity_rate, option_rate, cap = _lookup_taf(trade_date)

        if asset_type == "EQUITY":
            rate = equity_rate
        elif asset_type == "OPTION":
            rate = option_rate
        else:
            raise ValueError(f"Unknown asset_type: {asset_type}")

        uncapped_fee = quantity * rate
        return min(uncapped_fee, cap)

    def calculate_total_regulatory_fees(
        self,
        trade_date: datetime.date,
        quantity: int,
        price: float,
        asset_type: Literal["EQUITY", "OPTION"] = "EQUITY",
    ) -> float:
        """
        Calculate total regulatory fees (SEC + TAF).

        Fees only apply to sell-side transactions.

        Args:
            trade_date: Transaction date.
            quantity: Shares or contracts (must be positive).
            price: Execution price per share/contract (must be non-negative).
            asset_type: 'EQUITY' or 'OPTION'.

        Returns:
            Total fees in dollars.
        """
        if quantity <= 0:
            raise ValueError(f"Quantity must be positive, got {quantity}")
        if price < 0:
            raise ValueError(f"Price must be non-negative, got {price}")

        notional_value = quantity * price
        sec_fee = self.calculate_sec_fee(trade_date, notional_value)
        taf_fee = self.calculate_taf(quantity, asset_type, trade_date=trade_date)

        total = sec_fee + taf_fee
        return round(total, 2)

    def estimate_breakeven_profit(
        self,
        trade_date: datetime.date,
        quantity: int,
        entry_price: float,
        asset_type: Literal["EQUITY", "OPTION"] = "EQUITY",
    ) -> float:
        """
        Calculate minimum profit needed to overcome fees.

        Args:
            trade_date: Trade date.
            quantity: Position size.
            entry_price: Entry price (after buy slippage).
            asset_type: 'EQUITY' or 'OPTION'.

        Returns:
            Minimum exit price to break even after fees.
        """
        fees = self.calculate_total_regulatory_fees(
            trade_date, quantity, entry_price, asset_type
        )
        fee_per_share = fees / quantity
        return entry_price + fee_per_share

    def get_fee_schedule_info(self, trade_date: datetime.date) -> dict:
        """
        Get fee schedule parameters for a given date.

        Args:
            trade_date: Date to query.

        Returns:
            Dict of fee schedule parameters.
        """
        sec_rate = _lookup_sec_rate(trade_date)
        equity_rate, option_rate, cap = _lookup_taf(trade_date)

        return {
            "date": trade_date,
            "sec_rate_per_million": sec_rate * 1_000_000,
            "taf_rate_equity": equity_rate,
            "taf_rate_option": option_rate,
            "taf_cap": cap,
        }
