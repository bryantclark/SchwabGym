"""
Fee Calculation Tests
=====================

Test regulatory fee calculations including 2025 changes.

Author: Bryant Clark
"""

import pytest
import datetime
from schwabgym.fees import FeeCalculator


class TestSECFees:
    """Test SEC Section 31 fee calculations."""
    
    def test_sec_fee_pre_2025(self, fee_calculator):
        """Test SEC fee calculation before May 2025."""
        trade_date = datetime.date(2024, 12, 1)
        notional_value = 100000.0  # $100k trade
        
        fee = fee_calculator.calculate_sec_fee(trade_date, notional_value)
        
        expected = 100000 * (27.80 / 1_000_000)
        assert abs(fee - expected) < 0.01
    
    def test_sec_fee_post_2025(self, fee_calculator):
        """Test SEC fee eliminated after May 14, 2025."""
        trade_date = datetime.date(2025, 6, 1)
        notional_value = 100000.0
        
        fee = fee_calculator.calculate_sec_fee(trade_date, notional_value)
        
        assert fee == 0.0
    
    def test_sec_fee_exactly_on_cutoff(self, fee_calculator):
        """Test SEC fee on the exact cutoff date."""
        trade_date = datetime.date(2025, 5, 14)
        notional_value = 100000.0
        
        fee = fee_calculator.calculate_sec_fee(trade_date, notional_value)
        
        # On cutoff date, should use new rate (0.0)
        assert fee == 0.0
    
    def test_sec_fee_day_before_cutoff(self, fee_calculator):
        """Test SEC fee day before cutoff."""
        trade_date = datetime.date(2025, 5, 13)
        notional_value = 100000.0
        
        fee = fee_calculator.calculate_sec_fee(trade_date, notional_value)
        
        # Day before cutoff, should use old rate
        expected = 100000 * (27.80 / 1_000_000)
        assert abs(fee - expected) < 0.01


class TestTAFFees:
    """Test FINRA TAF calculations."""
    
    def test_taf_equity_small_order(self, fee_calculator):
        """Test TAF for small equity order."""
        fee = fee_calculator.calculate_taf(quantity=100, asset_type='EQUITY')
        
        expected = 100 * 0.000166
        assert abs(fee - expected) < 0.0001
    
    def test_taf_equity_capped(self, fee_calculator):
        """Test that TAF is capped at $8.30."""
        fee = fee_calculator.calculate_taf(quantity=100000, asset_type='EQUITY')
        
        # Should be capped
        assert fee == 8.30
    
    def test_taf_option(self, fee_calculator):
        """Test TAF for options."""
        fee = fee_calculator.calculate_taf(quantity=10, asset_type='OPTION')
        
        expected = 10 * 0.00279
        assert abs(fee - expected) < 0.0001
    
    def test_taf_option_not_capped_same(self, fee_calculator):
        """Test that option TAF uses different rate."""
        equity_fee = fee_calculator.calculate_taf(quantity=100, asset_type='EQUITY')
        option_fee = fee_calculator.calculate_taf(quantity=100, asset_type='OPTION')
        
        assert equity_fee != option_fee


class TestTotalRegulatoryFees:
    """Test combined fee calculations."""
    
    def test_total_fees_pre_2025(self, fee_calculator):
        """Test total fees before 2025 change."""
        trade_date = datetime.date(2024, 12, 1)
        quantity = 1000
        price = 100.0
        
        total = fee_calculator.calculate_total_regulatory_fees(
            trade_date, quantity, price, 'EQUITY'
        )
        
        # Should be SEC + TAF
        sec = 100000 * (27.80 / 1_000_000)
        taf = min(1000 * 0.000166, 8.30)
        expected = sec + taf
        
        assert abs(total - expected) < 0.01
    
    def test_total_fees_post_2025(self, fee_calculator):
        """Test total fees after 2025 change."""
        trade_date = datetime.date(2025, 6, 1)
        quantity = 1000
        price = 100.0
        
        total = fee_calculator.calculate_total_regulatory_fees(
            trade_date, quantity, price, 'EQUITY'
        )
        
        # Should be only TAF (no SEC)
        taf = min(1000 * 0.000166, 8.30)
        
        assert abs(total - taf) < 0.01
    
    def test_fees_scale_with_quantity(self, fee_calculator):
        """Test that fees increase with quantity."""
        trade_date = datetime.date(2024, 12, 1)
        price = 100.0
        
        fee_100 = fee_calculator.calculate_total_regulatory_fees(
            trade_date, 100, price, 'EQUITY'
        )
        
        fee_1000 = fee_calculator.calculate_total_regulatory_fees(
            trade_date, 1000, price, 'EQUITY'
        )
        
        assert fee_1000 > fee_100
    
    def test_fees_scale_with_price(self, fee_calculator):
        """Test that fees increase with price (SEC component)."""
        trade_date = datetime.date(2024, 12, 1)
        quantity = 1000
        
        fee_100 = fee_calculator.calculate_total_regulatory_fees(
            trade_date, quantity, 100.0, 'EQUITY'
        )
        
        fee_200 = fee_calculator.calculate_total_regulatory_fees(
            trade_date, quantity, 200.0, 'EQUITY'
        )
        
        assert fee_200 > fee_100


class TestBreakevenCalculations:
    """Test breakeven profit calculations."""
    
    def test_breakeven_small_position(self, fee_calculator):
        """Test breakeven for small position."""
        trade_date = datetime.date(2024, 12, 1)
        quantity = 100
        entry_price = 100.0
        
        breakeven = fee_calculator.estimate_breakeven_profit(
            trade_date, quantity, entry_price, 'EQUITY'
        )
        
        # Should be slightly above entry
        assert breakeven > entry_price
        assert breakeven < entry_price + 0.10  # Should be small
    
    def test_breakeven_large_position(self, fee_calculator):
        """Test breakeven for large position."""
        trade_date = datetime.date(2024, 12, 1)
        quantity = 10000
        entry_price = 100.0
        
        breakeven = fee_calculator.estimate_breakeven_profit(
            trade_date, quantity, entry_price, 'EQUITY'
        )
        
        # Should still be slightly above entry
        assert breakeven > entry_price
        
        # Fee per share should be small even for large position (due to TAF cap)
        fee_per_share = breakeven - entry_price
        assert fee_per_share < 0.01  # Less than 1 cent per share
    
    def test_breakeven_lower_post_2025(self, fee_calculator):
        """Test that breakeven is lower post-2025."""
        quantity = 1000
        entry_price = 100.0
        
        breakeven_pre = fee_calculator.estimate_breakeven_profit(
            datetime.date(2024, 12, 1), quantity, entry_price, 'EQUITY'
        )
        
        breakeven_post = fee_calculator.estimate_breakeven_profit(
            datetime.date(2025, 6, 1), quantity, entry_price, 'EQUITY'
        )
        
        # Post-2025 should be lower (no SEC fee)
        assert breakeven_post < breakeven_pre


class TestFeeScheduleInfo:
    """Test fee schedule information retrieval."""
    
    def test_get_info_pre_2025(self, fee_calculator):
        """Test getting fee schedule info before 2025."""
        info = fee_calculator.get_fee_schedule_info(datetime.date(2024, 12, 1))
        
        assert info['sec_fee_era'] == 'pre_2025'
        assert info['sec_rate_per_million'] == 27.8
    
    def test_get_info_post_2025(self, fee_calculator):
        """Test getting fee schedule info after 2025."""
        info = fee_calculator.get_fee_schedule_info(datetime.date(2025, 6, 1))
        
        assert info['sec_fee_era'] == 'post_2025'
        assert info['sec_rate_per_million'] == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])