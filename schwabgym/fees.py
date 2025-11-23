"""
SchwabGym Regulatory Fee Calculator
====================================

Calculates SEC Section 31 and FINRA TAF fees for US equities.

Author: Bryant Clark
Repository: https://github.com/bryantclark/SchwabGym
License: MIT
"""

import datetime
from typing import Literal


class FeeCalculator:
    """
    Calculate regulatory fees for equity and option transactions.

    This class implements fee schedules mandated by the SEC and FINRA,
    including the rate change scheduled for May 14, 2025.

    Fee Schedule:
        - SEC Section 31: $27.80/million (pre-May 14, 2025), $0.00 (after)
        - FINRA TAF: $0.000166/share (equities), $0.00279/contract (options)
        - TAF Cap: $8.30 per trade

    All fees apply to sell-side transactions only.

    References:
        - SEC Section 31: https://www.sec.gov/rules-regulations
        - FINRA TAF: https://www.finra.org/rules-guidance
    """

    # SEC Section 31 Fee Schedule
    SEC_FEE_CUTOFF = datetime.date(2025, 5, 14)  # Rate change date
    SEC_RATE_PRE_2025 = 27.80 / 1_000_000  # $27.80 per $1 million
    SEC_RATE_POST_2025 = 0.0  # Eliminated

    # FINRA Trading Activity Fee (TAF)
    TAF_RATE_EQUITY = 0.000166  # $0.000166 per share
    TAF_RATE_OPTION = 0.00279  # $0.00279 per contract
    TAF_CAP = 8.30  # Maximum per trade

    def __init__(self):
        """Initialize fee calculator."""
        pass

    def calculate_sec_fee(
        self, trade_date: datetime.date, notional_value: float
    ) -> float:
        """
        Calculate SEC Section 31 transaction fee.

        The SEC fee is assessed on the principal value of equity sales to
        fund the agency's operations. The rate is set by Congress and can
        change periodically.

        Note: The fee rate changes from $27.80/million to $0.00 on
        May 14, 2025.

        Args:
            trade_date (datetime.date): Date of transaction
            notional_value (float): Dollar value of trade (price × quantity)

        Returns:
            float: SEC fee in dollars
        """
        if trade_date < self.SEC_FEE_CUTOFF:
            return notional_value * self.SEC_RATE_PRE_2025
        else:
            return notional_value * self.SEC_RATE_POST_2025

    def calculate_taf(
        self, quantity: int, asset_type: Literal["EQUITY", "OPTION"] = "EQUITY"
    ) -> float:
        """
        Calculate FINRA Trading Activity Fee.

        The TAF is charged by FINRA to fund market regulation and
        surveillance. It applies per share (equities) or per contract
        (options) and is capped at a maximum per trade.

        Args:
            quantity (int): Number of shares or contracts
            asset_type (str): 'EQUITY' or 'OPTION'

        Returns:
            float: TAF in dollars (capped at $8.30)
        """
        if asset_type == "EQUITY":
            rate = self.TAF_RATE_EQUITY
        elif asset_type == "OPTION":
            rate = self.TAF_RATE_OPTION
        else:
            raise ValueError(f"Unknown asset_type: {asset_type}")

        uncapped_fee = quantity * rate
        return min(uncapped_fee, self.TAF_CAP)

    def calculate_total_regulatory_fees(
        self,
        trade_date: datetime.date,
        quantity: int,
        price: float,
        asset_type: Literal["EQUITY", "OPTION"] = "EQUITY",
    ) -> float:
        """
        Calculate total regulatory fees (SEC + TAF).

        Fees only apply to sell-side transactions. Buy orders have
        zero regulatory fees.

        Args:
            trade_date (datetime.date): Transaction date
            quantity (int): Shares or contracts
            price (float): Execution price per share/contract
            asset_type (str): 'EQUITY' or 'OPTION'

        Returns:
            float: Total fees in dollars
        """
        notional_value = quantity * price

        # SEC Section 31 fee
        sec_fee = self.calculate_sec_fee(trade_date, notional_value)

        # FINRA TAF
        taf_fee = self.calculate_taf(quantity, asset_type)

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
            trade_date (datetime.date): Trade date
            quantity (int): Position size
            entry_price (float): Entry price (after buy slippage)
            asset_type (str): 'EQUITY' or 'OPTION'

        Returns:
            float: Minimum exit price to break even after fees
        """
        # Calculate fees that will be charged on exit
        exit_price = entry_price  # Assume flat exit for calculation
        fees = self.calculate_total_regulatory_fees(
            trade_date, quantity, exit_price, asset_type
        )

        # Fees per share
        fee_per_share = fees / quantity

        # Breakeven is entry + fees
        return entry_price + fee_per_share

    def get_fee_schedule_info(self, trade_date: datetime.date) -> dict:
        """
        Get current fee schedule parameters for a given date.

        Args:
            trade_date (datetime.date): Date to query

        Returns:
            dict: Fee schedule parameters
        """
        is_pre_2025 = trade_date < self.SEC_FEE_CUTOFF

        return {
            "date": trade_date,
            "sec_rate_per_million": (
                self.SEC_RATE_PRE_2025 * 1_000_000 if is_pre_2025 else 0.0
            ),
            "taf_rate_equity": self.TAF_RATE_EQUITY,
            "taf_rate_option": self.TAF_RATE_OPTION,
            "taf_cap": self.TAF_CAP,
            "sec_fee_era": "pre_2025" if is_pre_2025 else "post_2025",
        }
