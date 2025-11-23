"""
SchwabGym Regulatory Fee Calculator
====================================

Accurate calculation of SEC Section 31 and FINRA TAF fees for US equities,
including the May 2025 rate change.

Author: Bryant Clark
Repository: https://github.com/bryantclark/SchwabGym
License: MIT
"""

import datetime
from typing import Literal


class FeeCalculator:
    """
    Calculate regulatory fees for equity and option transactions.
    
    This class implements the exact fee schedules mandated by the SEC
    and FINRA, including the important rate change scheduled for May 14, 2025.
    
    Accurate fee modeling is critical for training agents on high-frequency
    or scalping strategies where fees can erode alpha.
    
    Fee Schedule:
        - SEC Section 31: $27.80/million (pre-May 14, 2025), $0.00 (after)
        - FINRA TAF: $0.000166/share (equities), $0.00279/contract (options)
        - TAF Cap: $8.30 per trade
        
    All fees apply to sell-side transactions only.
    
    Example:
        >>> calc = FeeCalculator()
        >>> 
        >>> # Calculate fees for a sell order
        >>> fees = calc.calculate_total_regulatory_fees(
        ...     trade_date=datetime.date(2024, 12, 1),
        ...     quantity=1000,
        ...     price=100.0,
        ...     asset_type='EQUITY'
        ... )
        >>> print(f"Total fees: ${fees:.2f}")
        Total fees: $2.95
        >>> 
        >>> # After May 2025 (SEC fee eliminated)
        >>> fees = calc.calculate_total_regulatory_fees(
        ...     trade_date=datetime.date(2025, 6, 1),
        ...     quantity=1000,
        ...     price=100.0,
        ...     asset_type='EQUITY'
        ... )
        >>> print(f"Total fees: ${fees:.2f}")
        Total fees: $0.17  # Only FINRA TAF
        
    References:
        - SEC Section 31: https://www.sec.gov/rules-regulations
        - FINRA TAF: https://www.finra.org/rules-guidance
    """
    
    # SEC Section 31 Fee Schedule
    SEC_FEE_CUTOFF = datetime.date(2025, 5, 14)  # Rate change date
    SEC_RATE_PRE_2025 = 27.80 / 1_000_000        # $27.80 per $1 million
    SEC_RATE_POST_2025 = 0.0                      # Eliminated
    
    # FINRA Trading Activity Fee (TAF)
    TAF_RATE_EQUITY = 0.000166   # $0.000166 per share
    TAF_RATE_OPTION = 0.00279    # $0.00279 per contract
    TAF_CAP = 8.30               # Maximum per trade
    
    def __init__(self):
        """Initialize fee calculator."""
        pass
    
    def calculate_sec_fee(
        self,
        trade_date: datetime.date,
        notional_value: float
    ) -> float:
        """
        Calculate SEC Section 31 transaction fee.
        
        The SEC fee is assessed on the principal value of equity sales to
        fund the agency's operations. The rate is set by Congress and can
        change periodically.
        
        CRITICAL: The fee rate changes from $27.80/million to $0.00 on
        May 14, 2025. Agents trained on pre-2025 data must account for
        this when deployed post-2025.
        
        Args:
            trade_date (datetime.date): Date of transaction
            notional_value (float): Dollar value of trade (price × quantity)
            
        Returns:
            float: SEC fee in dollars
            
        Example:
            >>> calc = FeeCalculator()
            >>> 
            >>> # Pre-2025: Fee applies
            >>> fee = calc.calculate_sec_fee(
            ...     trade_date=datetime.date(2024, 12, 1),
            ...     notional_value=100000.0
            ... )
            >>> print(f"${fee:.4f}")  # $2.7800
            >>> 
            >>> # Post-2025: Fee eliminated
            >>> fee = calc.calculate_sec_fee(
            ...     trade_date=datetime.date(2025, 6, 1),
            ...     notional_value=100000.0
            ... )
            >>> print(f"${fee:.4f}")  # $0.0000
        """
        if trade_date < self.SEC_FEE_CUTOFF:
            return notional_value * self.SEC_RATE_PRE_2025
        else:
            return notional_value * self.SEC_RATE_POST_2025
    
    def calculate_taf(
        self,
        quantity: int,
        asset_type: Literal['EQUITY', 'OPTION'] = 'EQUITY'
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
            
        Example:
            >>> calc = FeeCalculator()
            >>> 
            >>> # Small equity order
            >>> fee = calc.calculate_taf(quantity=100, asset_type='EQUITY')
            >>> print(f"${fee:.4f}")  # $0.0166
            >>> 
            >>> # Large equity order (hits cap)
            >>> fee = calc.calculate_taf(quantity=100000, asset_type='EQUITY')
            >>> print(f"${fee:.2f}")  # $8.30 (capped)
            >>> 
            >>> # Options
            >>> fee = calc.calculate_taf(quantity=10, asset_type='OPTION')
            >>> print(f"${fee:.4f}")  # $0.0279
        """
        if asset_type == 'EQUITY':
            rate = self.TAF_RATE_EQUITY
        elif asset_type == 'OPTION':
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
        asset_type: Literal['EQUITY', 'OPTION'] = 'EQUITY'
    ) -> float:
        """
        Calculate total regulatory fees (SEC + TAF).
        
        This is the primary method used by the simulator. It computes the
        complete friction cost that will be deducted from proceeds on sell
        orders.
        
        NOTE: Fees only apply to sell-side transactions. Buy orders have
        zero regulatory fees.
        
        Args:
            trade_date (datetime.date): Transaction date
            quantity (int): Shares or contracts
            price (float): Execution price per share/contract
            asset_type (str): 'EQUITY' or 'OPTION'
            
        Returns:
            float: Total fees in dollars
            
        Example:
            >>> calc = FeeCalculator()
            >>> 
            >>> # Typical equity sell (1000 shares @ $100)
            >>> fees = calc.calculate_total_regulatory_fees(
            ...     trade_date=datetime.date(2024, 12, 1),
            ...     quantity=1000,
            ...     price=100.0,
            ...     asset_type='EQUITY'
            ... )
            >>> print(f"Proceeds: ${1000 * 100:.2f}")
            >>> print(f"Fees: ${fees:.2f}")
            >>> print(f"Net: ${1000 * 100 - fees:.2f}")
            Proceeds: $100000.00
            Fees: $2.95
            Net: $99997.05
            
            >>> # Same trade after May 2025 (lower fees)
            >>> fees = calc.calculate_total_regulatory_fees(
            ...     trade_date=datetime.date(2025, 6, 1),
            ...     quantity=1000,
            ...     price=100.0,
            ...     asset_type='EQUITY'
            ... )
            >>> print(f"Fees: ${fees:.2f}")
            Fees: $0.17  # Only TAF, no SEC fee
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
        asset_type: Literal['EQUITY', 'OPTION'] = 'EQUITY'
    ) -> float:
        """
        Calculate minimum profit needed to overcome fees.
        
        Useful for agents learning to filter out low-alpha trades where
        fees would eat the entire profit.
        
        Args:
            trade_date (datetime.date): Trade date
            quantity (int): Position size
            entry_price (float): Entry price (after buy slippage)
            asset_type (str): 'EQUITY' or 'OPTION'
            
        Returns:
            float: Minimum exit price to break even after fees
            
        Example:
            >>> calc = FeeCalculator()
            >>> 
            >>> # You bought 1000 shares @ $100
            >>> breakeven = calc.estimate_breakeven_profit(
            ...     trade_date=datetime.date(2024, 12, 1),
            ...     quantity=1000,
            ...     entry_price=100.0
            ... )
            >>> print(f"Need to exit above ${breakeven:.4f} to profit")
            Need to exit above $100.0030 to profit
            >>> 
            >>> # This accounts for the $2.95 in sell fees
            >>> # spread across 1000 shares = $0.00295 per share
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
        
        Useful for logging and debugging fee calculations.
        
        Args:
            trade_date (datetime.date): Date to query
            
        Returns:
            dict: Fee schedule parameters
            
        Example:
            >>> calc = FeeCalculator()
            >>> info = calc.get_fee_schedule_info(datetime.date(2024, 12, 1))
            >>> print(info)
            {
                'date': datetime.date(2024, 12, 1),
                'sec_rate_per_million': 27.8,
                'taf_rate_equity': 0.000166,
                'taf_rate_option': 0.00279,
                'taf_cap': 8.3,
                'sec_fee_era': 'pre_2025'
            }
        """
        is_pre_2025 = trade_date < self.SEC_FEE_CUTOFF
        
        return {
            'date': trade_date,
            'sec_rate_per_million': self.SEC_RATE_PRE_2025 * 1_000_000 if is_pre_2025 else 0.0,
            'taf_rate_equity': self.TAF_RATE_EQUITY,
            'taf_rate_option': self.TAF_RATE_OPTION,
            'taf_cap': self.TAF_CAP,
            'sec_fee_era': 'pre_2025' if is_pre_2025 else 'post_2025'
        }